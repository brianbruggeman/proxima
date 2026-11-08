//! One prefill matmul reduce on the CUDA backend, per shape, with the real
//! BlockLayout codec and synthetic weight bytes. Run with
//! PROXIMA_CUDA_EVENT_TIMING=1: the driver prints `cuda_event_timing_kernel`
//! rows (GPU event clock) for each launch; this binary prints the chosen kernel
//! identity and the flop count so rows convert to TFLOP/s.
//!
//! args: <features> <depth> <tokens> <runs> [codec]

use std::io;
use std::num::ParseIntError;

use omega::Codec;
use omega::backend::{
    BackendError, Engine, GpuDriver, cuda_kernel_identities, cuda_kernel_launches,
    cuda_replace_kernel_source, execute_plan_named, plan_named,
};
use proxima_numeric::block_layout::BlockLayout;
use proxima_tensor::test_support::Lcg;
use proxima_tensor::{Extent, IndexMap, Keep, NodeId, Op, QuantizedBlock, Reduce, ReduceInit, ScalarOp, append, projection};
use proxima_numeric::{DType, NumericPolicy};

#[derive(Debug, thiserror::Error)]
enum ProbeError {
    #[error("missing argument {0}; args: <features> <depth> <tokens> <runs> [codec] [head_dim]")]
    MissingArgument(&'static str),
    #[error("argument {name} is not an integer: {source}")]
    BadArgument {
        name: &'static str,
        source: ParseIntError,
    },
    #[error("codec has no block layout")]
    NoBlockLayout,
    #[error("plan does not expose cuda kernels")]
    NotCudaPlan,
    #[error("plan has no first kernel to replace")]
    NoFirstKernel,
    #[error("backend failed: {0}")]
    Backend(#[from] BackendError),
    #[error("io failed on {path}: {source}")]
    Io { path: String, source: io::Error },
}

fn parse_argument(args: &[String], at: usize, name: &'static str) -> Result<u32, ProbeError> {
    args.get(at)
        .ok_or(ProbeError::MissingArgument(name))?
        .parse()
        .map_err(|source| ProbeError::BadArgument { name, source })
}

fn matmul_program(features: u32, depth: u32, tokens: u32) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let weight = append(
        &mut program,
        Op::Input {
            dtype: DType::UINT8,
            shape: vec![Extent::Static(features), Extent::Static(depth)],
            name: Some("weight".into()),
        },
    );
    let activation = append(
        &mut program,
        Op::Input {
            dtype: DType::FLOAT32,
            shape: vec![Extent::Static(tokens), Extent::Static(depth)],
            name: Some("activation".into()),
        },
    );
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::FLOAT32,
            body: ScalarOp::Multiply,
            operands: vec![
                (weight, IndexMap::Affine(projection(3, &[0, 2]))),
                (activation, IndexMap::Affine(projection(3, &[1, 2]))),
            ],
            name: None,
        },
    );
    let sum = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::FLOAT32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(projection(3, &[0, 1])),
            keep: Keep::Reduce,
            name: Some("prefill_gemm".into()),
        }),
    );
    (program, sum)
}

fn split_program(heads: u32, head_dim: u32, depth: u32, tokens: u32) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let weight = append(
        &mut program,
        Op::Input {
            dtype: DType::UINT8,
            shape: vec![
                Extent::Static(heads),
                Extent::Static(head_dim),
                Extent::Static(depth),
            ],
            name: Some("weight".into()),
        },
    );
    let activation = append(
        &mut program,
        Op::Input {
            dtype: DType::FLOAT32,
            shape: vec![Extent::Static(tokens), Extent::Static(depth)],
            name: Some("activation".into()),
        },
    );
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::FLOAT32,
            body: ScalarOp::Multiply,
            operands: vec![
                (weight, IndexMap::Affine(projection(4, &[1, 2, 3]))),
                (activation, IndexMap::Affine(projection(4, &[0, 3]))),
            ],
            name: None,
        },
    );
    let sum = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::FLOAT32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(projection(4, &[0, 1, 2, 3])),
            out_map: IndexMap::Affine(projection(4, &[0, 1, 2])),
            keep: Keep::Reduce,
            name: Some("prefill_gemm_split".into()),
        }),
    );
    (program, sum)
}

fn arbitrary_blocks(layout: &BlockLayout, blocks: usize, seed: u64) -> Vec<u8> {
    let mut lcg = Lcg(seed);
    let block_bytes = usize::from(layout.block_bytes);
    let scale_at = usize::from(layout.scale.byte_offset);
    let mut bytes: Vec<u8> = (0..blocks * block_bytes)
        .map(|_| (lcg.next_unit() * 256.0) as u8)
        .collect();
    for block in bytes.chunks_exact_mut(block_bytes) {
        let exponent = 9 + (lcg.next_unit() * 4.0) as u16;
        block[scale_at..scale_at + 2].copy_from_slice(&(exponent << 10).to_le_bytes());
    }
    bytes
}

fn main() -> Result<(), ProbeError> {
    let mut args: Vec<String> = std::env::args().collect();
    let mut load_paths = Vec::new();
    while let Some(at) = args.iter().position(|arg| arg == "--load-source") {
        args.remove(at);
        load_paths.push(args.remove(at));
    }
    let dump_path = args.iter().position(|arg| arg == "--dump-source").map(|at| {
        args.remove(at);
        args.remove(at)
    });
    let features = parse_argument(&args, 1, "features")?;
    let depth = parse_argument(&args, 2, "depth")?;
    let tokens = parse_argument(&args, 3, "tokens")?;
    let runs = parse_argument(&args, 4, "runs")?;
    let codec = match args.get(5).map(String::as_str) {
        Some("q8_0") => Codec::Q8_0,
        Some("q4_0") => Codec::Q4_0,
        _ => Codec::Pq2_0,
    };
    let layout = codec.block_layout().ok_or(ProbeError::NoBlockLayout)?;
    let weight = arbitrary_blocks(
        &layout,
        features as usize * depth as usize / usize::from(layout.elements),
        41,
    );
    let mut lcg = Lcg(59);
    let activation: Vec<f32> = (0..depth as usize * tokens as usize)
        .map(|_| lcg.next_unit() * 4.0 - 2.0)
        .collect();
    let head_dim = match args.get(6) {
        Some(_) => parse_argument(&args, 6, "head_dim")?,
        None => 0,
    };
    let (program, root) = match features.checked_div(head_dim) {
        None => matmul_program(features, depth, tokens),
        Some(heads) => split_program(heads, head_dim, depth, tokens),
    };
    let named: Vec<(&str, QuantizedBlock<'_>)> = vec![
        ("weight", QuantizedBlock::Packed { codec, bytes: &weight }),
        ("activation", QuantizedBlock::Float32(&activation)),
    ];
    let mut plan = plan_named(
        Engine::Gpu,
        Some(GpuDriver::Cuda),
        &program,
        &[],
        &named,
        &[root],
        NumericPolicy::bit_exact()
            .with_reassociation(true)
            .with_activation_quantization(true),
    )?;
    let identities = cuda_kernel_identities(&plan).ok_or(ProbeError::NotCudaPlan)?;
    println!(
        "shape features={features} depth={depth} tokens={tokens} flop={} kernels={identities:?}",
        2u64 * u64::from(features) * u64::from(depth) * u64::from(tokens),
    );
    if let Some(path) = dump_path {
        let launches = cuda_kernel_launches(&plan).ok_or(ProbeError::NotCudaPlan)?;
        let dump: String = launches
            .iter()
            .zip(&identities)
            .map(|(launch, identity)| {
                let source = identity.split_once('\n').map_or("", |(_, source)| source);
                format!("// {launch}\n// block threads from block_width; grid blocks = grid_threads / block_width\n{source}\n")
            })
            .collect();
        std::fs::write(&path, dump).map_err(|source| ProbeError::Io { path, source })?;
    }
    for spec in load_paths {
        let (position, path) = match spec.split_once(':') {
            Some((index, path)) => (index.parse().unwrap_or(0), path.to_owned()),
            None => (0, spec),
        };
        let source = std::fs::read_to_string(&path)
            .map_err(|source| ProbeError::Io { path, source })?;
        if !cuda_replace_kernel_source(&mut plan, position, source) {
            return Err(ProbeError::NoFirstKernel);
        }
    }
    for _ in 0..runs {
        execute_plan_named(&mut plan, &named)?;
    }
    Ok(())
}
