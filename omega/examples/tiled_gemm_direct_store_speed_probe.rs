//! Isolated GPU speed of `PROXIMA_TILED_GEMM_DIRECT_STORE` (see
//! `docs/model-interop/discipline.md` ROW C4.11) against today's
//! restage-only path, at the Q4_0
//! weight-matmul shape (`[K=1536,M=12288] x [N=510,K=1536]`, the same shape
//! `q4_0_tiled_gemm_run8_speed_probe.rs`/`staging_switches_isolated_probe.rs`
//! use) and the dense-batched score shape
//! (`token=510, feature=256, batch=8, K=512`, node 162's own admitted
//! shape). Same real `GPUStartTime`/`GPUEndTime`
//! [`omega::metal::execute_plan_op_timed`] isolated-timing technique.
//!
//! Arms are INTERLEAVED per iteration (baseline, then direct-store, then
//! back to baseline, ... for every one of `RUNS` iterations) rather than
//! run back-to-back in blocks -- a back-to-back block conflates a real
//! arm delta with a clock/thermal drift artifact across the run (per this
//! workspace's own `feedback_interleave_gpu_arms_per_iteration` finding: a
//! non-interleaved sweep measured a fake 2.7x that collapsed to a real 1.1x
//! once interleaved).
//!
//! ISOLATED, not end-to-end: one synthetic op run in a loop, nothing else
//! touching the GPU. Weight/activation bytes are synthetic.

fn main() -> anyhow::Result<()> {
    #[cfg(all(feature = "metal", feature = "metal-tiled-gemm", feature = "instrument", target_os = "macos"))]
    return run();
    #[cfg(not(all(feature = "metal", feature = "metal-tiled-gemm", feature = "instrument", target_os = "macos")))]
    {
        println!(
            "tiled_gemm_direct_store_speed_probe requires --features metal,metal-tiled-gemm,instrument on macOS"
        );
        Ok(())
    }
}

#[cfg(all(feature = "metal", feature = "metal-tiled-gemm", feature = "instrument", target_os = "macos"))]
fn run() -> anyhow::Result<()> {
    use anyhow::Context;
    use proxima_gguf::quant::q4_0::{BLOCK_BYTES, QK4_0, quantize};
    use proxima_primitives::Codec;
    use proxima_tensor::test_support::Lcg;
    use proxima_tensor::{
        DType, Extent, IndexMap, Keep, NodeId, NumericPolicy, Op, QuantizedBlock, Reduce,
        ReduceInit, ScalarOp, append, projection,
    };

    const IN_DIM: usize = 1536;
    const OUT_DIM: usize = 12288;
    const TOKENS: usize = 510;
    const RUNS: usize = 12;

    fn random_vec(seed: u64, count: usize) -> Vec<f32> {
        let mut lcg = Lcg(seed);
        (0..count).map(|_| lcg.next_unit()).collect()
    }

    fn pack_rows(rows: &[Vec<f32>], in_dim: usize) -> anyhow::Result<Vec<u8>> {
        let blocks_per_row = in_dim / QK4_0;
        let mut packed = vec![0u8; rows.len() * blocks_per_row * BLOCK_BYTES];
        for (row, row_packed) in rows.iter().zip(packed.chunks_exact_mut(blocks_per_row * BLOCK_BYTES)) {
            quantize(row, row_packed).context("in_dim is a whole multiple of QK4_0")?;
        }
        Ok(packed)
    }

    fn q4_0_matmul_program(tokens: u32, in_dim: u32, out_dim: u32) -> (Vec<Op>, NodeId) {
        let mut program = Vec::new();
        let weight = append(
            &mut program,
            Op::Input {
                dtype: DType::UInt8,
                shape: vec![Extent::Static(in_dim), Extent::Static(out_dim)],
                name: Some("weight".into()),
            },
        );
        let activation = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(tokens), Extent::Static(in_dim)],
                name: None,
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

    fn dense_batched_value_program() -> (Vec<Op>, NodeId, usize, usize, usize, usize) {
        let (token, feature, batch, reduce_len) = (510u32, 256u32, 8u32, 512u32);
        let mut program = Vec::new();
        let weight = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(feature), Extent::Static(batch), Extent::Static(reduce_len)],
                name: None,
            },
        );
        let other = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(token), Extent::Static(batch), Extent::Static(reduce_len)],
                name: None,
            },
        );
        let product = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: vec![
                    (weight, IndexMap::Affine(projection(4, &[1, 2, 3]))),
                    (other, IndexMap::Affine(projection(4, &[0, 2, 3]))),
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
                in_map: IndexMap::Affine(projection(4, &[0, 1, 2, 3])),
                out_map: IndexMap::Affine(projection(4, &[0, 1, 2])),
                keep: Keep::Reduce,
                name: None,
            }),
        );
        (program, sum, token as usize, feature as usize, batch as usize, reduce_len as usize)
    }

    fn gpu_ns_once(program: &[Op], root: NodeId, blocks: &[QuantizedBlock<'_>]) -> anyhow::Result<u64> {
        let plan = omega::plan(program, &[], blocks, &[root], NumericPolicy::default()).context("plan compiles")?;
        let (_, timings) = omega::metal::execute_plan_op_timed(&plan, blocks, None)
            .context("metal executes on a real device")?;
        Ok(timings.iter().map(|timing| timing.gpu_ns).sum())
    }

    fn mean_and_cov(samples: &[u64]) -> (f64, f64) {
        let mean = samples.iter().sum::<u64>() as f64 / samples.len() as f64;
        let variance = samples
            .iter()
            .map(|&value| {
                let diff = value as f64 - mean;
                diff * diff
            })
            .sum::<f64>()
            / samples.len() as f64;
        let stddev = variance.sqrt();
        (mean, if mean > 0.0 { stddev / mean } else { 0.0 })
    }

    if let Ok(output) = std::process::Command::new("pgrep")
        .args(["-fl", "ollama|llama-server|Ollama|cargo|rustc"])
        .output()
        && !output.stdout.is_empty()
    {
        eprintln!(
            "WARNING: another build/GPU process is running elsewhere on this box:\n{}",
            String::from_utf8_lossy(&output.stdout)
        );
    }

    // === Q4_0 weight matmul ===
    let rows: Vec<Vec<f32>> = (0..OUT_DIM).map(|row| random_vec(11 + row as u64, IN_DIM)).collect();
    let packed = pack_rows(&rows, IN_DIM)?;
    let activation = random_vec(97, TOKENS * IN_DIM);
    let (q4_0_program, q4_0_sum) = q4_0_matmul_program(TOKENS as u32, IN_DIM as u32, OUT_DIM as u32);
    let q4_0_blocks = [
        QuantizedBlock::Packed { codec: Codec::Q4_0, bytes: &packed },
        QuantizedBlock::Float32(&activation),
    ];

    println!("=== Q4_0 weight matmul [K={IN_DIM}, M={OUT_DIM}] x [N={TOKENS}, K={IN_DIM}] ({RUNS} interleaved iterations) ===");
    let mut baseline_samples = Vec::with_capacity(RUNS);
    let mut direct_store_samples = Vec::with_capacity(RUNS);
    for iteration in 0..RUNS {
        let baseline_ns = temp_env::with_var("PROXIMA_TILED_GEMM_DIRECT_STORE", None::<&str>, || {
            gpu_ns_once(&q4_0_program, q4_0_sum, &q4_0_blocks)
        })?;
        let direct_store_ns = temp_env::with_var("PROXIMA_TILED_GEMM_DIRECT_STORE", Some("1"), || {
            gpu_ns_once(&q4_0_program, q4_0_sum, &q4_0_blocks)
        })?;
        baseline_samples.push(baseline_ns);
        direct_store_samples.push(direct_store_ns);
        println!(
            "  iter={iteration} baseline={baseline_ns}ns direct_store={direct_store_ns}ns ratio={:.4}x",
            baseline_ns as f64 / direct_store_ns as f64
        );
    }
    let (baseline_mean, baseline_cov) = mean_and_cov(&baseline_samples);
    let (direct_store_mean, direct_store_cov) = mean_and_cov(&direct_store_samples);
    println!(
        "  Q4_0 SUMMARY: baseline mean={:.4}ms CoV={:.2}%  direct_store mean={:.4}ms CoV={:.2}%  mean-ratio={:.4}x",
        baseline_mean / 1e6,
        baseline_cov * 100.0,
        direct_store_mean / 1e6,
        direct_store_cov * 100.0,
        baseline_mean / direct_store_mean
    );

    // === dense-batched score shape ===
    let (dense_program, dense_sum, token, feature, batch, reduce_len) = dense_batched_value_program();
    let weight_data = random_vec(2026, feature * batch * reduce_len);
    let other_data = random_vec(2027, token * batch * reduce_len);
    let dense_blocks = [QuantizedBlock::Float32(&weight_data), QuantizedBlock::Float32(&other_data)];

    println!(
        "=== dense batched [token={token} feature={feature} batch={batch} K={reduce_len}] ({RUNS} interleaved iterations) ==="
    );
    let mut dense_baseline_samples = Vec::with_capacity(RUNS);
    let mut dense_direct_store_samples = Vec::with_capacity(RUNS);
    for iteration in 0..RUNS {
        let baseline_ns = temp_env::with_vars(
            [
                ("PROXIMA_TILED_GEMM_DIRECT_STORE", None::<&str>),
                ("PROXIMA_TILED_GEMM_DENSE", None::<&str>),
            ],
            || gpu_ns_once(&dense_program, dense_sum, &dense_blocks),
        )?;
        let direct_store_ns = temp_env::with_vars(
            [
                ("PROXIMA_TILED_GEMM_DIRECT_STORE", Some("1")),
                ("PROXIMA_TILED_GEMM_DENSE", None::<&str>),
            ],
            || gpu_ns_once(&dense_program, dense_sum, &dense_blocks),
        )?;
        dense_baseline_samples.push(baseline_ns);
        dense_direct_store_samples.push(direct_store_ns);
        println!(
            "  iter={iteration} baseline={baseline_ns}ns direct_store={direct_store_ns}ns ratio={:.4}x",
            baseline_ns as f64 / direct_store_ns as f64
        );
    }
    let (dense_baseline_mean, dense_baseline_cov) = mean_and_cov(&dense_baseline_samples);
    let (dense_direct_store_mean, dense_direct_store_cov) = mean_and_cov(&dense_direct_store_samples);
    println!(
        "  DENSE SUMMARY: baseline mean={:.4}ms CoV={:.2}%  direct_store mean={:.4}ms CoV={:.2}%  mean-ratio={:.4}x",
        dense_baseline_mean / 1e6,
        dense_baseline_cov * 100.0,
        dense_direct_store_mean / 1e6,
        dense_direct_store_cov * 100.0,
        dense_baseline_mean / dense_direct_store_mean
    );
    Ok(())
}
