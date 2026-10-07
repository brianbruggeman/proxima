//! Isolated GPU time of one dense `Q4_0` weight matmul at prefill width on the
//! tiled path (`metal-tiled-gemm`), at the gemma4 E2B shapes: K = 1536 against
//! 12288, 6144, 4096 and 2048 output rows (the ffn gate and up, ffn down and
//! attention projections of the 275 tiled dispatches of a 971-token prefill).
//!
//! Timing is `GPUStartTime`/`GPUEndTime` per op ([`omega::metal::execute_plan_named_op_timed`],
//! `instrument` feature): one op in a loop with nothing else on the device, so
//! the number is the production kernel on the production dispatch geometry
//! under whatever `PROXIMA_TILED_GEMM_*` environment the process was started
//! with. Weight bytes are synthetic (random `Q4_0` blocks); correctness is the
//! parity tests' job, and each row prints an FNV-1a hash of the output bits so
//! two builds or two environments can be compared byte for byte.
//!
//! ```sh
//! cargo run --release -p omega --example tiled_gemm_q4_0_speed_probe \
//!   --features metal,instrument
//! ```

#[cfg(all(feature = "metal", feature = "instrument", target_os = "macos"))]
use anyhow::Context;
#[cfg(all(feature = "metal", feature = "instrument", target_os = "macos"))]
use proxima_gguf::quant::q4_0::{BLOCK_BYTES, QK4_0, quantize};
#[cfg(all(feature = "metal", feature = "instrument", target_os = "macos"))]
use proxima_primitives::Codec;
#[cfg(all(feature = "metal", feature = "instrument", target_os = "macos"))]
use proxima_tensor::test_support::Lcg;
#[cfg(all(feature = "metal", feature = "instrument", target_os = "macos"))]
use proxima_tensor::{
    DType, Extent, IndexMap, Keep, NodeId, NumericPolicy, Op, QuantizedBlock, Reduce, ReduceInit,
    ScalarOp, append, projection,
};

fn main() -> anyhow::Result<()> {
    #[cfg(all(feature = "metal", feature = "instrument", target_os = "macos"))]
    return run();
    #[cfg(not(all(feature = "metal", feature = "instrument", target_os = "macos")))]
    {
        println!("tiled_gemm_q4_0_speed_probe requires --features metal,instrument on macOS");
        Ok(())
    }
}

#[cfg(all(feature = "metal", feature = "instrument", target_os = "macos"))]
const RUNS: usize = 21;
#[cfg(all(feature = "metal", feature = "instrument", target_os = "macos"))]
const WARMUP_RUNS: usize = 60;
#[cfg(all(feature = "metal", feature = "instrument", target_os = "macos"))]
const TOKEN_COUNTS: [usize; 3] = [160, 512, 971];
#[cfg(all(feature = "metal", feature = "instrument", target_os = "macos"))]
const SHAPES: [(&str, usize, usize); 4] = [
    ("ffn_gate_up", 12288, 1536),
    ("ffn_down", 6144, 1536),
    ("attn_q", 4096, 1536),
    ("attn_o", 2048, 1536),
];

#[cfg(all(feature = "metal", feature = "instrument", target_os = "macos"))]
fn unit_values(seed: u64, count: usize) -> Vec<f32> {
    let mut lcg = Lcg(seed);
    (0..count).map(|_| lcg.next_unit() * 2.0 - 1.0).collect()
}

#[cfg(all(feature = "metal", feature = "instrument", target_os = "macos"))]
fn weight_blocks(rows: usize, k: usize) -> anyhow::Result<Vec<u8>> {
    let row_bytes = k / QK4_0 * BLOCK_BYTES;
    let mut blocks = vec![0u8; rows * row_bytes];
    let values = unit_values(11, rows * k);
    for (row, row_out) in values.chunks_exact(k).zip(blocks.chunks_exact_mut(row_bytes)) {
        quantize(row, row_out).context("k is a whole number of q4_0 blocks")?;
    }
    Ok(blocks)
}

#[cfg(all(feature = "metal", feature = "instrument", target_os = "macos"))]
fn program(tokens: usize, rows: usize, k: usize) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let weight = append(
        &mut program,
        Op::Input {
            dtype: DType::UInt8,
            shape: vec![Extent::Static(k as u32), Extent::Static(rows as u32)],
            name: Some("weight".into()),
        },
    );
    let activation = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(tokens as u32), Extent::Static(k as u32)],
            name: Some("activation".into()),
        },
    );
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (weight, IndexMap::Affine(projection(3, &[1, 2]))),
                (activation, IndexMap::Affine(projection(3, &[0, 1]))),
            ],
            name: None,
        },
    );
    let sum = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(projection(3, &[0, 2])),
            keep: Keep::Reduce,
            name: None,
        }),
    );
    (program, sum)
}

#[cfg(all(feature = "metal", feature = "instrument", target_os = "macos"))]
fn median(mut samples: Vec<u64>) -> u64 {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

#[cfg(all(feature = "metal", feature = "instrument", target_os = "macos"))]
fn percentile(mut samples: Vec<u64>, percent: usize) -> u64 {
    samples.sort_unstable();
    samples[(samples.len() - 1) * percent / 100]
}

#[cfg(all(feature = "metal", feature = "instrument", target_os = "macos"))]
fn coefficient_of_variation_percent(samples: &[u64]) -> f64 {
    let mean = samples.iter().sum::<u64>() as f64 / samples.len() as f64;
    let variance = samples
        .iter()
        .map(|&sample| (sample as f64 - mean).powi(2))
        .sum::<f64>()
        / samples.len() as f64;
    variance.sqrt() / mean * 100.0
}

#[cfg(all(feature = "metal", feature = "instrument", target_os = "macos"))]
fn fnv1a(bits: impl Iterator<Item = u32>) -> u64 {
    bits.fold(0xcbf2_9ce4_8422_2325u64, |hash, word| {
        (hash ^ u64::from(word)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

#[cfg(all(feature = "metal", feature = "instrument", target_os = "macos"))]
fn measure(name: &str, tokens: usize, rows: usize, k: usize) -> anyhow::Result<()> {
    let weights = weight_blocks(rows, k)?;
    let activation = unit_values(23, tokens * k);
    let (program, root) = program(tokens, rows, k);
    let named = [
        ("weight", QuantizedBlock::Packed { codec: Codec::Q4_0, bytes: &weights }),
        ("activation", QuantizedBlock::Float32(&activation)),
    ];
    let plan = omega::plan_named(&program, &[], &named, &[root], NumericPolicy::default())
        .context("plan compiles")?;
    for _ in 0..WARMUP_RUNS {
        omega::metal::execute_plan_named_op_timed(&plan, &named, None)
            .context("metal warm-up executes on a real device")?;
    }
    let mut samples = Vec::with_capacity(RUNS);
    let mut output_hash = 0;
    for _ in 0..RUNS {
        let (evaluated, timings) = omega::metal::execute_plan_named_op_timed(&plan, &named, None)
            .context("metal executes on a real device")?;
        samples.push(timings.iter().map(|timing| timing.gpu_ns).sum::<u64>());
        output_hash = fnv1a(evaluated.root().iter().map(|value| value.to_bits()));
    }
    let gflop = 2.0 * tokens as f64 * rows as f64 * k as f64 / 1.0e9;
    let microseconds = median(samples.clone()) as f64 / 1000.0;
    let minimum = samples.iter().copied().min().unwrap_or(0) as f64 / 1000.0;
    let lower_quartile = percentile(samples.clone(), 25) as f64 / 1000.0;
    let maximum = samples.iter().copied().max().unwrap_or(0) as f64 / 1000.0;
    println!(
        "shape={name} tokens={tokens} rows={rows} k={k} median_us={microseconds:.1} p25_us={lower_quartile:.1} min_us={minimum:.1} max_us={maximum:.1} cov_pct={:.2} tflops={:.3} output_fnv={output_hash:016x} samples_us={:?}",
        coefficient_of_variation_percent(&samples),
        gflop / microseconds * 1.0e3,
        samples.iter().map(|sample| sample / 1000).collect::<Vec<_>>()
    );
    Ok(())
}

#[cfg(all(feature = "metal", feature = "instrument", target_os = "macos"))]
fn run() -> anyhow::Result<()> {
    println!("probe=tiled_gemm_q4_0 runs={RUNS} unit=us gpu_clock=GPUStartTime..GPUEndTime");
    for (name, rows, k) in SHAPES {
        for tokens in TOKEN_COUNTS {
            measure(name, tokens, rows, k)?;
        }
    }
    Ok(())
}
