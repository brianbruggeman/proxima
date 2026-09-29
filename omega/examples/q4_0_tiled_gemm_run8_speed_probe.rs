//! Isolated GPU speed of the new batched `q4_0_run8` tiled-GEMM weight-tile
//! decode (`PROXIMA_TILED_GEMM_Q4_0` at its default, unset) against today's
//! row-blocked path (switch explicitly `"0"`), at `T = 64` and `T = 510`.
//! Uses real `GPUStartTime`/
//! `GPUEndTime` per-op timing ([`omega::execute_plan_op_timed`], `instrument`
//! feature) -- MTLCommandBuffer's own documented GPU occupancy, not a
//! CPU-side wall-clock difference the way `q4k_matvec_probe.rs`'s own
//! two-size-difference technique has to work around driver overhead.
//!
//! ISOLATED, not end-to-end: one synthetic matmul op run in a loop, nothing
//! else touching the GPU. Weight bytes are synthetic (random `Q4_0` blocks,
//! `proxima_gguf::quant::q4_0::quantize`) -- correctness against real
//! checkpoint bytes is `q4_0_tiled_gemm_batched_run8_parity.rs`'s own job,
//! not this probe's.

#![allow(clippy::unwrap_used, clippy::expect_used)]

fn main() {
    #[cfg(all(feature = "metal", feature = "metal-tiled-gemm", feature = "instrument", target_os = "macos"))]
    run();
    #[cfg(not(all(feature = "metal", feature = "metal-tiled-gemm", feature = "instrument", target_os = "macos")))]
    println!(
        "q4_0_tiled_gemm_run8_speed_probe requires --features metal,metal-tiled-gemm,instrument on macOS"
    );
}

#[cfg(all(feature = "metal", feature = "metal-tiled-gemm", feature = "instrument", target_os = "macos"))]
fn run() {
    use proxima_gguf::quant::q4_0::{BLOCK_BYTES, QK4_0, quantize};
    use proxima_primitives::Codec;
    use proxima_tensor::test_support::Lcg;
    use proxima_tensor::{
        DType, Extent, IndexMap, Keep, NodeId, NumericPolicy, Op, QuantizedBlock, Reduce,
        ReduceInit, ScalarOp, append, projection,
    };

    const IN_DIM: usize = 1536;
    const OUT_DIM: usize = 8192;
    const RUNS: usize = 10;

    fn random_vec(seed: u64, count: usize) -> Vec<f32> {
        let mut lcg = Lcg(seed);
        (0..count).map(|_| lcg.next_unit()).collect()
    }

    fn pack_rows(rows: &[Vec<f32>], in_dim: usize) -> Vec<u8> {
        let blocks_per_row = in_dim / QK4_0;
        let mut packed = vec![0u8; rows.len() * blocks_per_row * BLOCK_BYTES];
        for (row, row_packed) in rows
            .iter()
            .zip(packed.chunks_exact_mut(blocks_per_row * BLOCK_BYTES))
        {
            quantize(row, row_packed).expect("in_dim is a whole multiple of QK4_0");
        }
        packed
    }

    fn matmul_program(tokens: u32, in_dim: u32, out_dim: u32) -> (Vec<Op>, NodeId) {
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

    fn median_gpu_ns(
        program: &[Op],
        root: NodeId,
        blocks: &[QuantizedBlock<'_>],
        runs: usize,
    ) -> u64 {
        let plan = omega::plan(program, &[], blocks, &[root], NumericPolicy::default())
            .expect("plan compiles");
        let mut samples = Vec::with_capacity(runs);
        for _ in 0..runs {
            let (_, timings) = omega::metal::execute_plan_op_timed(&plan, blocks, None)
                .expect("metal executes on a real device");
            let total: u64 = timings.iter().map(|timing| timing.gpu_ns).sum();
            samples.push(total);
        }
        samples.sort_unstable();
        samples[samples.len() / 2]
    }

    if let Ok(output) = std::process::Command::new("pgrep").args(["-fl", "cargo|rustc"]).output()
        && !output.stdout.is_empty()
    {
        eprintln!(
            "WARNING: a cargo/rustc process is running elsewhere on this box:\n{}",
            String::from_utf8_lossy(&output.stdout)
        );
    }

    let rows: Vec<Vec<f32>> = (0..OUT_DIM).map(|row| random_vec(11 + row as u64, IN_DIM)).collect();
    let packed = pack_rows(&rows, IN_DIM);
    let weight_bytes = packed.len() as u64;

    for tokens in [64usize, 510] {
        let activation = random_vec(97, tokens * IN_DIM);
        let (program, sum) = matmul_program(tokens as u32, IN_DIM as u32, OUT_DIM as u32);
        let blocks = [
            QuantizedBlock::Packed { codec: Codec::Q4_0, bytes: &packed },
            QuantizedBlock::Float32(&activation),
        ];

        let current_ns = temp_env::with_var("PROXIMA_TILED_GEMM_Q4_0", Some("0"), || {
            median_gpu_ns(&program, sum, &blocks, RUNS)
        });
        let new_ns = median_gpu_ns(&program, sum, &blocks, RUNS);

        let flops = 2.0 * tokens as f64 * IN_DIM as f64 * OUT_DIM as f64;
        let current_gbs = weight_bytes as f64 / (current_ns as f64 / 1e9) / 1e9;
        let new_gbs = weight_bytes as f64 / (new_ns as f64 / 1e9) / 1e9;
        let current_gflops = flops / (current_ns as f64 / 1e9) / 1e9;
        let new_gflops = flops / (new_ns as f64 / 1e9) / 1e9;

        println!(
            "ISOLATED T={tokens} in_dim={IN_DIM} out_dim={OUT_DIM} runs={RUNS} (median of {RUNS}):\n\
             \x20 current (row-blocked)     = {current_ns} ns = {:.4} ms  {current_gbs:.2} GB/s(weights)  {current_gflops:.2} GFLOP/s\n\
             \x20 new (tiled q4_0_run8)      = {new_ns} ns = {:.4} ms  {new_gbs:.2} GB/s(weights)  {new_gflops:.2} GFLOP/s\n\
             \x20 speedup (current/new)      = {:.3}x",
            current_ns as f64 / 1e6,
            new_ns as f64 / 1e6,
            current_ns as f64 / new_ns as f64,
        );
    }
}
