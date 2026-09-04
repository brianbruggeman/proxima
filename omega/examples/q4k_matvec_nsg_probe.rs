//! Nano-bench for the row-blocked packed Q4_K matvec kernel
//! (`crate::msl::push_packed_row_blocked_body`) at the REAL Mistral-7B
//! batch-1 decode shapes ROW 221 (`proxima-tensor/docs/discipline.md:18404-
//! 18412`) actually measures: `attn_q`/`attn_output` (4096x4096),
//! `ffn_gate`/`ffn_up` (14336x4096), `ffn_down` (4096x14336).
//!
//! Reuses `q4k_matmul_layout.rs`'s program shape (weight declared
//! `[in_dim, out_dim]`, reduction axis first -- the convention every real
//! matmul weight in `mistral_forward_program` uses, and independently
//! checked there against a from-scratch dequantize+dot oracle, not just
//! against the CPU backend) and ROW 71's two-size marginal-bandwidth method
//! (`proxima-tensor/docs/discipline.md:5079-5081`), extended to report a
//! per-shape median/CoV over `RUNS` repeats rather than a single min, plus
//! GMAC/s alongside GB/s.
//!
//! `q4k_matvec_probe.rs`'s OWN 3D-projection program shape
//! (`[rows, k]`/`[k, 1]`) was tried first and rejected on a false lead: its
//! CPU-vs-Metal check disagreed by `max_relative=122.8` at 4096x4096. A
//! bisection (`examples/q4k_bisect_probe.rs`, in_dim=512 fixed, out_dim
//! 3..4096) then localized the SAME divergence pattern on
//! `q4k_matmul_layout.rs`'s own PROVEN 2D shape: `max_metal_rel` (Metal vs
//! an independent from-scratch dequantize+dot oracle) stays bounded
//! (<=1.5e-3) at every out_dim up to 4096, while `max_cpu_rel` (CPU's
//! `evaluate_quantized` int8-dot path vs the SAME oracle) grows to 8.72
//! (872%) at out_dim>=1024 -- **the CPU int8-dot quantized path is the one
//! that diverges at scale, not Metal.** This is a NEW, separate finding
//! from `relative=1.3264047` on `fix/metal-cpu-disagreement`'s real-forward-
//! graph failure (both localize a CPU-side divergence, at different call
//! sites) -- reported here as data for that branch, not chased further.
//!
//! Correctness gate BEFORE any timing number (task principle 14/18): given
//! the bisection above, `cpu::evaluate_quantized` is NOT a trustworthy
//! oracle at these shapes, so every shape's Metal output is checked against
//! the SAME independent dequantize+dot reference the bisection used, not
//! against CPU. CPU-vs-reference is still computed and printed (a named
//! residual, not silently dropped) but never gates.
//!
//! Run baseline (current geometry, one simdgroup per 4-row group):
//! ```sh
//! CARGO_TARGET_DIR=<scratch>/target cargo run -p omega --release \
//!   --features metal,cpu --example q4k_matvec_nsg_probe
//! ```
//! Run the `nsg=2` geometry under test:
//! ```sh
//! CARGO_TARGET_DIR=<scratch>/target cargo run -p omega --release \
//!   --features metal,cpu,metal-packed-row-nsg2 --example q4k_matvec_nsg_probe
//! ```

#![allow(clippy::unwrap_used, clippy::expect_used)]

fn main() {
    #[cfg(all(feature = "metal", feature = "cpu", target_os = "macos"))]
    run();
    #[cfg(not(all(feature = "metal", feature = "cpu", target_os = "macos")))]
    println!("q4k_matvec_nsg_probe requires --features metal,cpu on macOS");
}

#[cfg(all(feature = "metal", feature = "cpu", target_os = "macos"))]
fn run() {
    use std::time::Instant;

    use proxima_gguf::quant::q4_k::{BLOCK_BYTES, QK_K, dequantize, quantize};
    use proxima_tensor::cpu::evaluate_quantized;
    use proxima_tensor::test_support::Lcg;
    use proxima_tensor::{
        DType, Extent, IndexMap, Keep, NodeId, Op, QuantizedBlock, Reduce, ReduceInit, ScalarOp,
        append, projection,
    };

    fn random_vec(seed: u64, count: usize) -> Vec<f32> {
        let mut lcg = Lcg(seed);
        (0..count).map(|_| lcg.next_unit()).collect()
    }

    // Weight declared `[in_dim, out_dim]`, reduction axis first -- exactly
    // `q4k_matmul_layout.rs`'s `matmul_program`, the shape
    // `mistral_forward_program`'s own real weights use and the shape that
    // file's own committed parity test holds to an independent
    // dequantize+dot oracle (not just to the CPU backend).
    fn matvec_program(in_dim: u32, out_dim: u32) -> (Vec<Op>, NodeId) {
        let mut program = Vec::new();
        let weight = append(
            &mut program,
            Op::Input {
                dtype: DType::UInt8,
                shape: vec![Extent::Static(in_dim), Extent::Static(out_dim)],
                name: None,
            },
        );
        let activation = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(in_dim)],
                name: None,
            },
        );
        // iteration space (o, i): axis 0 = out (survives), axis 1 = in (reduced).
        let product = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: vec![
                    (weight, IndexMap::Affine(projection(2, &[1, 0]))),
                    (activation, IndexMap::Affine(projection(2, &[1]))),
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
                in_map: IndexMap::Affine(projection(2, &[0, 1])),
                out_map: IndexMap::Affine(projection(2, &[0])),
                keep: Keep::Reduce,
                name: None,
            }),
        );
        (program, sum)
    }

    // Packs `out_dim` independent rows of `in_dim` elements each into
    // GGUF's native `[out_dim, in_dim]` row-major byte layout -- identical
    // to `q4k_matmul_layout.rs`'s own `pack_rows`.
    fn pack_weight(out_dim: u32, in_dim: u32, seed: u64) -> Vec<u8> {
        let blocks_per_row = in_dim as usize / QK_K;
        let weight_f32 = random_vec(seed, out_dim as usize * in_dim as usize);
        let mut packed = vec![0u8; out_dim as usize * blocks_per_row * BLOCK_BYTES];
        for (row, row_packed) in weight_f32
            .chunks_exact(in_dim as usize)
            .zip(packed.chunks_exact_mut(blocks_per_row * BLOCK_BYTES))
        {
            quantize(row, row_packed).expect("in_dim is a whole multiple of QK_K");
        }
        packed
    }

    // Independent oracle -- dequantizes each packed row by hand and dots it
    // against the activation, exactly `q4k_matmul_layout.rs`'s own
    // `expected_output`. The bisection in this file's module doc proved
    // this, not `cpu::evaluate_quantized`, is the trustworthy reference at
    // these row counts.
    fn independent_reference(packed: &[u8], in_dim: usize, out_dim: usize, activation: &[f32]) -> Vec<f32> {
        let blocks_per_row = in_dim / QK_K;
        let mut expected = Vec::with_capacity(out_dim);
        for row_packed in packed.chunks_exact(blocks_per_row * BLOCK_BYTES) {
            let mut row = vec![0.0f32; in_dim];
            dequantize(row_packed, &mut row).expect("packed row dequantizes");
            let dot: f32 = row.iter().zip(activation.iter()).map(|(w, a)| w * a).sum();
            expected.push(dot);
        }
        expected
    }

    fn mean_stddev(samples: &[f64]) -> (f64, f64) {
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let variance =
            samples.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / samples.len() as f64;
        (mean, variance.sqrt())
    }

    fn median(samples: &[f64]) -> f64 {
        let mut sorted = samples.to_vec();
        sorted.sort_by(f64::total_cmp);
        sorted[sorted.len() / 2]
    }

    struct ShapeResult {
        label: &'static str,
        median_ms: f64,
        cov_pct: f64,
        gb_s: f64,
        gmac_s: f64,
    }

    const RUNS: usize = 7;
    let shapes: [(u32, u32, &str); 3] = [
        (4096, 4096, "4096x4096"),
        (14336, 4096, "14336x4096(ffn_gate/up)"),
        (4096, 14336, "4096x14336(ffn_down)"),
    ];

    let mut results = Vec::with_capacity(shapes.len());
    for (rows, k, label) in shapes {
        let packed = pack_weight(rows, k, 17);
        let activation = random_vec(13, k as usize);
        let (program, sum) = matvec_program(k, rows);
        let blocks = [
            QuantizedBlock::Q4K(&packed),
            QuantizedBlock::Float32(&activation),
        ];

        // CORRECTNESS GATE, before any timing sample is taken -- against the
        // independent dequantize+dot reference, NOT `cpu::evaluate_
        // quantized` (see module doc: CPU's int8-dot path is the one that
        // diverges at these row counts, not Metal).
        let reference = independent_reference(&packed, k as usize, rows as usize, &activation);
        let cpu = evaluate_quantized(&program, &[], &blocks, &[sum]).expect("cpu evaluates");
        let plan = omega::plan(&program, &[], &blocks, &[sum]).expect("metal plans");
        let metal = omega::execute_plan(&plan, &blocks).expect("metal executes");
        let cpu_root = cpu.root();
        let metal_root = metal.root();
        assert_eq!(cpu_root.len(), rows as usize, "{label}: cpu produced no output");
        assert_eq!(metal_root.len(), rows as usize, "{label}: metal produced no output");
        let mut max_metal_relative = 0.0f32;
        let mut max_cpu_relative = 0.0f32;
        for ((&cpu_value, &metal_value), &reference_value) in
            cpu_root.iter().zip(metal_root.iter()).zip(reference.iter())
        {
            let scale = reference_value.abs().max(f32::MIN_POSITIVE);
            max_metal_relative = max_metal_relative.max((reference_value - metal_value).abs() / scale);
            max_cpu_relative = max_cpu_relative.max((reference_value - cpu_value).abs() / scale);
        }
        // epsilon widens with `k` the same way `metal_vs_cpu.rs`'s own
        // `assert_checksum_agrees` documents: float reassociation across a
        // longer reduction is not a bug (q4k_matmul_layout.rs's fixed 1e-2
        // was pinned at k=512; k=4096/14336 here needs the same sqrt(k)
        // scaling, not a tighter absolute bound).
        let epsilon = 5e-4 * (k as f32).sqrt();
        assert!(
            max_metal_relative < epsilon,
            "{label}: metal disagrees with the independent dequantize+dot reference, \
             max_metal_relative={max_metal_relative} exceeds epsilon={epsilon} (k={k}) -- \
             correctness gate failed BEFORE any timing number, per task principle 14/18"
        );
        println!(
            "{label}: correctness OK -- max_metal_relative={max_metal_relative:e} \
             (max_cpu_relative={max_cpu_relative:e}, informational: CPU's own int8-dot path, \
             not gated -- see module doc)"
        );

        // warm loop -- plan once, execute repeatedly, the serving-loop
        // shape `q4k_matvec_probe.rs` already established.
        omega::execute_plan(&plan, &blocks).expect("warmup executes");
        let mut samples_ms = Vec::with_capacity(RUNS);
        for _ in 0..RUNS {
            let started = Instant::now();
            std::hint::black_box(omega::execute_plan(&plan, &blocks).expect("probe executes"));
            samples_ms.push(started.elapsed().as_secs_f64() * 1000.0);
        }

        let (mean_ms, stddev_ms) = mean_stddev(&samples_ms);
        let cov_pct = 100.0 * stddev_ms / mean_ms;
        let median_ms = median(&samples_ms);
        let weight_bytes = packed.len() as f64;
        let macs = rows as f64 * k as f64;
        let gb_s = (weight_bytes / 1e9) / (median_ms / 1000.0);
        let gmac_s = (macs / 1e9) / (median_ms / 1000.0);

        println!(
            "{label}: n={RUNS} median={median_ms:.4}ms mean={mean_ms:.4}ms cov={cov_pct:.2}% \
             packed_bytes={weight_bytes:.0} GB/s={gb_s:.2} GMAC/s={gmac_s:.2}"
        );
        println!(
            "  samples_ms={:?}",
            samples_ms
                .iter()
                .map(|value| format!("{value:.4}"))
                .collect::<Vec<_>>()
        );

        results.push(ShapeResult {
            label,
            median_ms,
            cov_pct,
            gb_s,
            gmac_s,
        });
    }

    println!("--- summary (median over {RUNS} runs, warm in-process loop) ---");
    for result in &results {
        println!(
            "{:<28} median={:.4}ms cov={:.2}% GB/s={:.2} GMAC/s={:.2}",
            result.label, result.median_ms, result.cov_pct, result.gb_s, result.gmac_s
        );
    }

    // ROW 71-style two-size marginal bandwidth, on the smallest and largest
    // shapes this run measured (`4096x4096` -> `14336x4096`): a difference
    // of two medians cancels the per-call fixed cost (compile/upload/
    // readback, see `q4k_matvec_probe.rs`'s own doc), which a single-size
    // absolute figure cannot.
    if results.len() >= 2 {
        let small = &results[0];
        let large = &results[1];
        let small_bytes = 4096.0 * 4096.0 * (BLOCK_BYTES as f64) / (QK_K as f64);
        let large_bytes = 14336.0 * 4096.0 * (BLOCK_BYTES as f64) / (QK_K as f64);
        let delta_ms = large.median_ms - small.median_ms;
        let delta_bytes = large_bytes - small_bytes;
        let marginal_gbs = (delta_bytes / 1e9) / (delta_ms / 1000.0);
        println!(
            "marginal ({} -> {}): delta_bytes={delta_bytes:.0} delta_ms={delta_ms:.4} \
             marginal_GB/s={marginal_gbs:.2}",
            small.label, large.label
        );
    }
}
