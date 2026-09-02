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

        // plan ONCE, execute per iteration -- the serving-loop shape. This
        // is what takes `infer`/`bind`/codec-resolution out of the timed
        // region, where they never belonged.
        let resolved = omega::plan(&program, &[], &blocks, &[sum]).expect("probe plans");
        omega::execute_plan(&resolved, &blocks).expect("warmup executes");
        let mut samples = Vec::with_capacity(runs);
        for _ in 0..runs {
            let started = Instant::now();
            let out = omega::execute_plan(&resolved, &blocks).expect("probe executes");
            samples.push(started.elapsed().as_secs_f64() * 1000.0);
            assert_eq!(
                out.root().len(),
                rows as usize,
                "degenerate probe: no output"
            );
        }
        samples.sort_by(f64::total_cmp);
        // min, not median: a sibling process on this box interferes, and
        // the minimum is the least-interfered sample. Under contention the
        // median tracks the interferer, not the kernel.
        (samples[0], packed.len() as f64)
    }

    // is the `Reduce(Elementwise)` fusion firing? If it is not, omega
    // materializes the full [rows, 1, k] product before reducing it --
    // rows*k floats of intermediate for a rows*k/2-byte weight, which would
    // dominate everything else here.
    {
        use proxima_tensor::{bind, infer};
        let (program, sum) = matvec_program(4096, 4096);
        let shapes = infer(&program, &[]).expect("probe program infers");
        let nests = bind(&program, &shapes, &[sum]).expect("probe program binds");
        println!(
            "q4k_matvec_probe bound_ops={} (1 == fused, 2 == materializing)",
            nests.len()
        );
        let kernel =
            omega::emit(&nests[0], &std::collections::BTreeMap::new()).expect("probe kernel emits");
        println!(
            "--- emitted kernel entry={} grid={:?}",
            kernel.entry, kernel.grid
        );
        println!("{}", kernel.source);
        println!("--- end kernel");
        for (index, bound) in nests.iter().enumerate() {
            println!(
                "  op{index} kind={:?} output_elements={}",
                core::mem::discriminant(&bound.kind),
                bound
                    .extents
                    .iter()
                    .map(|extent| *extent as usize)
                    .product::<usize>()
            );
        }
    }

    // a marginal figure is a DIFFERENCE of two minima, so sampling error in
    // either one is amplified. 21 samples produced a 5.7x swing between
    // consecutive runs of this probe; this is the count that makes the
    // difference readable rather than the noise.
    const RUNS: usize = 51;
    const K: u32 = 4096;
    // sizes chosen so the KERNEL dominates the ~0.2-0.3 ms per-call
    // intercept. At 1024/4096 rows both arms ran in ~0.3 ms, i.e. mostly
    // intercept, and the marginal figure swung 3x between runs.
    let (small_ms, small_bytes) = measure(4096, K, RUNS);
    let (large_ms, large_bytes) = measure(16384, K, RUNS);

    let delta_ms = large_ms - small_ms;
    let delta_bytes = large_bytes - small_bytes;
    let marginal_gbs = (delta_bytes / 1e9) / (delta_ms / 1000.0);

    println!("q4k_matvec_probe runs={RUNS} k={K}");
    println!(
        "  small  packed={:.2} MB  median={small_ms:.3} ms  end_to_end={:.1} GB/s",
        small_bytes / 1e6,
        (small_bytes / 1e9) / (small_ms / 1000.0)
    );
    println!(
        "  large  packed={:.2} MB  median={large_ms:.3} ms  end_to_end={:.1} GB/s",
        large_bytes / 1e6,
        (large_bytes / 1e9) / (large_ms / 1000.0)
    );
    println!(
        "  marginal (large-small, cancels per-call compile+upload fixed cost): \
         {:.2} MB in {delta_ms:.3} ms = {marginal_gbs:.1} GB/s",
        delta_bytes / 1e6
    );

    // control: the IDENTICAL kernel shape with an f32 weight. Same iteration
    // space, same reduce, same everything except the operand read — so the
    // difference isolates what `q4k_element` costs per element against a
    // single flat load. 4x the bytes, so if packed is not FASTER per byte
    // the unpack is eating more than the traffic it saves.
    fn measure_f32(rows: u32, k: u32, runs: usize) -> f64 {
        let weight = random_vec(17, rows as usize * k as usize);
        let activation = random_vec(13, k as usize);
        let mut program = Vec::new();
        let weight_node = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(rows), Extent::Static(k)],
                name: None,
            },
        );
        let activation_node = append(
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
                    (weight_node, IndexMap::Affine(projection(3, &[0, 2]))),
                    (activation_node, IndexMap::Affine(projection(3, &[2, 1]))),
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
        let blocks = [
            QuantizedBlock::Float32(&weight),
            QuantizedBlock::Float32(&activation),
        ];
        let resolved = omega::plan(&program, &[], &blocks, &[sum]).expect("f32 control plans");
        omega::execute_plan(&resolved, &blocks).expect("f32 control warms up");
        let mut samples = Vec::with_capacity(runs);
        for _ in 0..runs {
            let started = Instant::now();
            omega::execute_plan(&resolved, &blocks).expect("f32 control executes");
            samples.push(started.elapsed().as_secs_f64() * 1000.0);
        }
        samples.sort_by(f64::total_cmp);
        samples[0]
    }

    // DISSECTION. At these sizes the per-call fixed cost (prepare/infer/bind,
    // output + uniforms buffer allocation, command buffer, submit,
    // waitUntilCompleted, readback) is comparable to the kernel itself, so a
    // single-size number cannot tell them apart. Two sizes per arm cancel it:
    // the DIFFERENCE is marginal cost, which is the kernel.
    let f32_small_ms = measure_f32(4096, K, RUNS);
    let f32_large_ms = measure_f32(16384, K, RUNS);
    let f32_small_bytes = 4096.0 * f64::from(K) * 4.0;
    let f32_large_bytes = 16384.0 * f64::from(K) * 4.0;
    let f32_marginal_gbs =
        ((f32_large_bytes - f32_small_bytes) / 1e9) / ((f32_large_ms - f32_small_ms) / 1000.0);
    // fixed cost, extrapolated back to zero bytes from the two points
    let f32_slope_ms_per_byte = (f32_large_ms - f32_small_ms) / (f32_large_bytes - f32_small_bytes);
    let f32_fixed_ms = f32_small_ms - f32_slope_ms_per_byte * f32_small_bytes;
    let packed_slope = (large_ms - small_ms) / (large_bytes - small_bytes);
    let packed_fixed_ms = small_ms - packed_slope * small_bytes;

    println!("  --- dissection: two sizes per arm, fixed cost cancelled ---");
    println!(
        "  f32    small={f32_small_ms:.3} ms ({:.1} MB)  large={f32_large_ms:.3} ms ({:.1} MB)",
        f32_small_bytes / 1e6,
        f32_large_bytes / 1e6
    );
    println!(
        "  f32    MARGINAL = {f32_marginal_gbs:.1} GB/s   fixed-cost intercept = {f32_fixed_ms:.3} ms"
    );
    println!(
        "  packed MARGINAL = {marginal_gbs:.1} GB/s   fixed-cost intercept = {packed_fixed_ms:.3} ms"
    );
    println!(
        "  bar 214.7 GB/s => f32 kernel is {:.2}x off on MARGINAL bandwidth",
        214.7 / f32_marginal_gbs
    );
    println!(
        "  uploads: nocopy_attempts={} of which REUSED={} (so {} real wires), copying={}",
        omega::metal::NOCOPY_BUFFER_UPLOADS.get(),
        omega::metal::NOCOPY_BUFFER_REUSES.get(),
        omega::metal::NOCOPY_BUFFER_UPLOADS.get() - omega::metal::NOCOPY_BUFFER_REUSES.get(),
        omega::metal::COPYING_BUFFER_UPLOADS.get()
    );
    println!("  bar: llama.cpp Metal on 7B Q4_K_S = 214.7 GB/s (3.784 GB in 17.62 ms/token)");
}
