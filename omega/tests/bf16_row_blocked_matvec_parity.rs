//! The BF16 packed matvec reads each lane's run as native `bfloat` (`msl/matmul_form.rs`,
//! `CodeRead::Scalar`). Its output on the real flint router tensor `blk.2.ffn_gate_inp.weight`
//! (BF16, 256 rows of 2048), at the single-token shape decode runs, must equal an independent f64
//! reduction over the gguf dequantizer's (`proxima_gguf::quant::bf16`) values. A control compares the
//! same output against the neighbouring row's reduction and must fail.
#![cfg(all(feature = "metal", target_os = "macos"))]
// workspace lints deny unwrap/expect (Cargo.toml:93-94) and an integration test binary is not cfg(test), so a failed fixture read must be an allowed panic.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

use proxima_gguf::parser::{GgufEvent, GgufParser};
use proxima_gguf::pipe::ParsedGguf;
use proxima_gguf::types::GgmlType;
use proxima_numeric::codec::Codec;
use proxima_numeric::{DType, NumericPolicy};
use proxima_tensor::test_support::Lcg;
use proxima_tensor::{Extent, IndexMap, Keep, NodeId, Op, QuantizedBlock, Reduce, ReduceInit, ScalarOp, append, map};

const FLINT_PATH: &str = "/Users/brianbruggeman/models/llm/flint-gguf/flint-UD-Q4_K_XL.gguf";
const ROUTER: &str = "blk.2.ffn_gate_inp.weight";
const ACTIVATION_SEED: u64 = 2026;
const RELATIVE_TOLERANCE: f64 = 1.0e-6;

fn header(file: &mut File, file_len: u64) -> ParsedGguf {
    let mut prefix_len = 1usize << 22;
    loop {
        let mut buffer = vec![0u8; prefix_len];
        file.seek(SeekFrom::Start(0)).unwrap();
        let read = file.read(&mut buffer).unwrap();
        buffer.truncate(read);
        if let Ok((parser, events)) = GgufParser::new().push(&buffer) {
            let (mut version, mut metadata, mut tensors, mut completion) = (None, Vec::new(), Vec::new(), None);
            for event in events {
                match event {
                    GgufEvent::Header { version: found, .. } => version = Some(found),
                    GgufEvent::Metadata { key, value } => metadata.push((key, value)),
                    GgufEvent::Tensor(tensor) => tensors.push(tensor),
                    GgufEvent::Complete { data_offset, alignment } => completion = Some((data_offset, alignment)),
                }
            }
            if let (Some(version), Some((data_offset, alignment))) = (version, completion) {
                parser.finish().unwrap();
                return ParsedGguf { version, tensor_count: tensors.len() as u64, kv_count: metadata.len() as u64, metadata, tensors, data_offset, alignment };
            }
        }
        assert!((prefix_len as u64) < file_len, "no gguf header in the first {prefix_len} bytes");
        prefix_len *= 2;
    }
}

struct Router {
    bytes: Vec<u8>,
    reduction: usize,
    rows: usize,
}

fn read_router() -> Router {
    let path = proxima_gguf::checkpoint_resolver::require_checkpoint("PROXIMA_LEVEL_CODEC_PARITY_GGUF", FLINT_PATH, 0);
    let mut file = File::open(&path).expect("the checkpoint opens");
    let file_len = file.metadata().unwrap().len();
    let parsed = header(&mut file, file_len);
    let tensor = parsed.tensors.iter().find(|tensor| tensor.name == ROUTER).expect("the router tensor is in the directory");
    assert_eq!(tensor.ggml_type, GgmlType::Bf16, "{ROUTER} is a BF16 tensor");
    let reduction = tensor.dims[0] as usize;
    let rows = tensor.dims[1] as usize;
    let range = parsed.tensor_data_range(tensor, file_len).unwrap();
    let mut bytes = vec![0u8; rows * reduction * proxima_gguf::quant::bf16::BLOCK_BYTES];
    file.seek(SeekFrom::Start(range.start)).unwrap();
    file.read_exact(&mut bytes).unwrap();
    Router { bytes, reduction, rows }
}

fn activation(reduction: usize) -> Vec<f32> {
    let mut lcg = Lcg(ACTIVATION_SEED);
    (0..reduction).map(|_| lcg.next_unit() * 4.0 - 2.0).collect()
}

fn program(router: &Router) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let weight = append(&mut program, Op::Input { dtype: DType::UINT8, shape: vec![Extent::Static(router.rows as u32), Extent::Static(router.reduction as u32)], name: None });
    let activation = append(&mut program, Op::Input { dtype: DType::FLOAT32, shape: vec![Extent::Static(1), Extent::Static(router.reduction as u32)], name: None });
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::FLOAT32,
            body: ScalarOp::Multiply,
            operands: vec![(weight, IndexMap::Affine(map::projection(3, &[1, 2]))), (activation, IndexMap::Affine(map::projection(3, &[0, 2])))],
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
            in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
            keep: Keep::Reduce,
            name: Some("router_logits".into()),
        }),
    );
    (program, sum)
}

/// `(reduction of the host-decoded row against the activation, sum of the absolute terms)` per row.
fn reference(router: &Router, activation: &[f32]) -> Vec<(f64, f64)> {
    let row_bytes = router.reduction * proxima_gguf::quant::bf16::BLOCK_BYTES;
    let mut values = vec![0.0f32; router.reduction];
    (0..router.rows)
        .map(|row| {
            proxima_gguf::quant::bf16::dequantize(&router.bytes[row * row_bytes..(row + 1) * row_bytes], &mut values).expect("the gguf dequantizer reads the row");
            values.iter().zip(activation).map(|(weight, act)| f64::from(*weight) * f64::from(*act)).fold((0.0f64, 0.0f64), |(sum, magnitude), term| (sum + term, magnitude + term.abs()))
        })
        .collect()
}

fn out_of_tolerance(got: &[f32], want: &[(f64, f64)], shift: usize) -> usize {
    got.iter()
        .enumerate()
        .filter(|(at, value)| {
            let (reduced, magnitude) = want[(at + shift) % want.len()];
            (f64::from(**value) - reduced).abs() > RELATIVE_TOLERANCE * magnitude + 1.0e-6
        })
        .count()
}

#[test]
fn bf16_router_row_blocked_matvec_matches_the_host_decode() {
    let router = read_router();
    let activation = activation(router.reduction);
    let (program, output) = program(&router);
    let blocks = [QuantizedBlock::Packed { codec: Codec::BFloat16, bytes: &router.bytes }, QuantizedBlock::Float32(&activation)];
    let outputs = [output];
    let plan = omega::plan(&program, &[], &blocks, &outputs, NumericPolicy::default()).expect("metal plans the router matvec");
    let kernel_keys = plan.kernel_keys().expect("the plan exposes its kernel keys");
    let evaluated = omega::execute_plan(&plan, &blocks).expect("metal runs the router matvec");
    let got = evaluated.root().to_vec();
    let want = reference(&router, &activation);

    assert!(kernel_keys.iter().any(|key| key.contains("_cbfB1")), "the row-blocked kernel was never selected (`B` is the packed_row_block_shape token after the codec tokens `bf`): {kernel_keys:?}");
    assert_eq!(got.len(), want.len(), "one f32 per router row");
    assert!(got.iter().all(|value| value.is_finite()), "a non-finite output would pass the tolerance comparison vacuously");
    let wrong = out_of_tolerance(&got, &want, 0);
    assert_eq!(wrong, 0, "{wrong} of {} router logits differ from the host-decoded f64 reduction", want.len());
    let neighbour = out_of_tolerance(&got, &want, 1);
    assert!(neighbour > want.len() * 9 / 10, "control: only {neighbour} outputs left tolerance against the neighbouring row, the check does not discriminate");
}
