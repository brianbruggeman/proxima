//! What a packed `Q4_K` matvec actually costs on this GPU, and how much of
//! that is the kernel versus the driver around it.
//!
//! Decode is a weight sweep: 0.012 bytes/mac, so the only number that
//! matters is achieved read bandwidth over the packed weight buffer. The
//! bar, measured on this box with llama.cpp `-ngl 99` on a 7B `Q4_K_S`
//! checkpoint: 17.62 ms/token over a 3.784 GB sweep = 214.7 GB/s.
//!
//! HONEST SCOPE, because the driver shape contaminates this: `omega::execute`
//! compiles the MSL and uploads every block on EVERY call, and reads the
//! output back before returning. So a single-call number is compile plus
//! upload plus dispatch plus readback, not kernel bandwidth. This probe
//! therefore reports both a per-call wall time AND a two-size difference,
//! which cancels the per-call fixed cost and leaves the marginal cost of the
//! extra bytes. That difference is the closest thing to a kernel-bandwidth
//! figure this driver can produce today, and it is still an upper bound on
//! the cost rather than a clean kernel measurement.


// a probe, not library code: a failure here should abort loudly with the
// message rather than thread a Result out to `main`, the same way this
// crate's own benches and parity tests do.
#![allow(clippy::unwrap_used, clippy::expect_used)]
fn main() {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    run();
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    println!("q4k_matvec_probe requires --features metal on macOS");
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn run() {
    use std::time::Instant;

    use proxima_gguf::quant::q4_k::{BLOCK_BYTES, QK_K, quantize};
    use proxima_tensor::test_support::Lcg;
    use proxima_tensor::{
        DType, Extent, IndexMap, Keep, NodeId, Op, QuantizedBlock, Reduce, ReduceInit, ScalarOp,
        append, projection,
    };

    fn random_vec(seed: u64, count: usize) -> Vec<f32> {
        let mut lcg = Lcg(seed);
        (0..count).map(|_| lcg.next_unit()).collect()
    }

    fn matvec_program(rows: u32, k: u32) -> (Vec<Op>, NodeId) {
        let mut program = Vec::new();
        let weight = append(
            &mut program,
            Op::Input {
                dtype: DType::UInt8,
                shape: vec![Extent::Static(rows), Extent::Static(k)],
                name: None,
            },
        );
        let activation = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(k), Extent::Static(1)],
                name: None,
            },
        );
        let product = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: vec![
                    (weight, IndexMap::Affine(projection(3, &[0, 2]))),
                    (activation, IndexMap::Affine(projection(3, &[2, 1]))),
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
                out_map: IndexMap::Affine(projection(3, &[0, 1])),
                keep: Keep::Reduce,
                name: None,
            }),
        );
        (program, sum)
    }

    fn measure(rows: u32, k: u32, runs: usize) -> (f64, f64) {
        let blocks_per_row = k as usize / QK_K;
        let weight_f32 = random_vec(17, rows as usize * k as usize);
        let mut packed = vec![0u8; rows as usize * blocks_per_row * BLOCK_BYTES];
        for (row, row_packed) in weight_f32
            .chunks_exact(k as usize)
            .zip(packed.chunks_exact_mut(blocks_per_row * BLOCK_BYTES))
        {
            quantize(row, row_packed).expect("k is a whole multiple of QK_K");
        }
        let activation = random_vec(13, k as usize);
        let (program, sum) = matvec_program(rows, k);
        let blocks = [
            QuantizedBlock::Q4K(&packed),
            QuantizedBlock::Float32(&activation),
        ];

        omega::execute(&program, &[], &blocks, &[sum]).expect("warmup executes");
        let mut samples = Vec::with_capacity(runs);
        for _ in 0..runs {
            let started = Instant::now();
            let out = omega::execute(&program, &[], &blocks, &[sum]).expect("probe executes");
            samples.push(started.elapsed().as_secs_f64() * 1000.0);
            assert_eq!(out.root().len(), rows as usize, "degenerate probe: no output");
        }
        samples.sort_by(f64::total_cmp);
        (samples[samples.len() / 2], packed.len() as f64)
    }

    const RUNS: usize = 21;
    const K: u32 = 4096;
    let (small_ms, small_bytes) = measure(1024, K, RUNS);
    let (large_ms, large_bytes) = measure(4096, K, RUNS);

    let delta_ms = large_ms - small_ms;
    let delta_bytes = large_bytes - small_bytes;
    let marginal_gbs = (delta_bytes / 1e9) / (delta_ms / 1000.0);

    println!("q4k_matvec_probe runs={RUNS} k={K}");
    println!(
        "  rows=1024  packed={:.2} MB  median={small_ms:.3} ms  end_to_end={:.1} GB/s",
        small_bytes / 1e6,
        (small_bytes / 1e9) / (small_ms / 1000.0)
    );
    println!(
        "  rows=4096  packed={:.2} MB  median={large_ms:.3} ms  end_to_end={:.1} GB/s",
        large_bytes / 1e6,
        (large_bytes / 1e9) / (large_ms / 1000.0)
    );
    println!(
        "  marginal (large-small, cancels per-call compile+upload fixed cost): \
         {:.2} MB in {delta_ms:.3} ms = {marginal_gbs:.1} GB/s",
        delta_bytes / 1e6
    );
    println!("  bar: llama.cpp Metal on 7B Q4_K_S = 214.7 GB/s (3.784 GB in 17.62 ms/token)");
}
