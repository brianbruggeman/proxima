//! Full-output correctness check for the NEON 6x4 tile kernel
//! (`proxima_tensor::cpu::gemm_tile_neon`, exercised through `run_reduce`'s
//! transposed-RHS fast path). One element (`root[0]`) and an m=8,n=8 test
//! were the only evidence before this file — neither forces the row/column
//! remainder paths a non-multiple-of-`TILE_ROWS`/`TILE_COLS` size exercises.
//! 260 is not a multiple of `TILE_ROWS` (6), so it already forces the row
//! remainder path even though it is a multiple of `TILE_COLS` (4).
//!
//! this file does NOT assert a relative-tolerance match between the tile
//! kernel and a naive f32 triple-loop: the naive loop is a worse oracle
//! than the kernel under test. the tile does pairwise summation across its
//! 4-wide FMA lanes, which has strictly lower error growth than a naive
//! sequential sum over k terms, and was measured (against an f64-accumulated
//! ground truth) at 0.25-0.37x the naive loop's own RMS error. asserting
//! f32-vs-f32 equality-within-tolerance against that oracle just compares
//! two differently-rounded answers and fails near zero crossings — it was
//! never testing whether the kernel is correct. the invariants below check
//! against an f64 ground truth instead.

use proxima_tensor::{Extent, IndexMap, NodeId, Op, ReduceInit, ScalarOp, append, evaluate, map};

/// Verbatim copy of `examples/profile_hot.rs::matmul_program_rhs_transposed`
/// — same transposed-RHS GEMM program the tile kernel's fast path targets.
fn matmul_program_rhs_transposed(m: u32, k: u32, n: u32) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let lhs = append(
        &mut program,
        Op::Input {
            dtype: proxima_tensor::DType::Float32,
            shape: vec![Extent::Static(m), Extent::Static(k)],
            name: None,
        },
    );
    let rhs = append(
        &mut program,
        Op::Input {
            dtype: proxima_tensor::DType::Float32,
            shape: vec![Extent::Static(n), Extent::Static(k)],
            name: None,
        },
    );
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: proxima_tensor::DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (lhs, IndexMap::Affine(map::projection(3, &[0, 2]))),
                (rhs, IndexMap::Affine(map::projection(3, &[1, 2]))),
            ],
            name: None,
        },
    );
    let sum = append(
        &mut program,
        Op::Reduce(proxima_tensor::Reduce {
            dtype: proxima_tensor::DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
            keep: proxima_tensor::Keep::Reduce,
            name: Some("matmul_rhs_transposed".into()),
        }),
    );
    (program, sum)
}

/// Plain (non-transposed) RHS layout, `[k, n]` contiguous in `n` — this is
/// the layout `run_reduce`'s width-dim fast path (`width_tile_plan`,
/// `gemm_width_tile_neon`) targets, as opposed to
/// `matmul_program_rhs_transposed`'s `[n, k]` layout above, which the dot
/// tile targets instead (`cpu.rs`'s own doc on
/// `evaluate_typed_float32_matmul_shaped_reduce_fires_the_neon_tile`).
/// Verbatim copy of `cpu.rs`'s private `#[cfg(test)] matmul_program`.
fn matmul_program(m: u32, k: u32, n: u32) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let lhs = append(
        &mut program,
        Op::Input {
            dtype: proxima_tensor::DType::Float32,
            shape: vec![Extent::Static(m), Extent::Static(k)],
            name: None,
        },
    );
    let rhs = append(
        &mut program,
        Op::Input {
            dtype: proxima_tensor::DType::Float32,
            shape: vec![Extent::Static(k), Extent::Static(n)],
            name: None,
        },
    );
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: proxima_tensor::DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (lhs, IndexMap::Affine(map::projection(3, &[0, 2]))),
                (rhs, IndexMap::Affine(map::projection(3, &[2, 1]))),
            ],
            name: None,
        },
    );
    let sum = append(
        &mut program,
        Op::Reduce(proxima_tensor::Reduce {
            dtype: proxima_tensor::DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
            keep: proxima_tensor::Keep::Reduce,
            name: Some("matmul".into()),
        }),
    );
    (program, sum)
}

fn naive_reference(a: &[f32], b_transposed: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for step in 0..k {
                acc += a[row * k + step] * b_transposed[col * k + step];
            }
            out[row * n + col] = acc;
        }
    }
    out
}

/// f64-accumulated ground truth — used to tell apart "the tile kernel is
/// wrong" from "the tile kernel and the f32 naive loop just summed `k`
/// terms in a different order, and both are within normal f32 rounding of
/// the true answer." A relative-error comparison between two f32 results
/// alone cannot distinguish those two cases near a zero crossing.
fn high_precision_reference(
    a: &[f32],
    b_transposed: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<f64> {
    let mut out = vec![0.0f64; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f64;
            for step in 0..k {
                acc += a[row * k + step] as f64 * b_transposed[col * k + step] as f64;
            }
            out[row * n + col] = acc;
        }
    }
    out
}

/// `naive_reference`'s sibling for the width-tile's plain `[k, n]` RHS
/// layout: `b[step * n + col]` instead of `b_transposed[col * k + step]`.
fn naive_reference_plain_rhs(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for step in 0..k {
                acc += a[row * k + step] * b[step * n + col];
            }
            out[row * n + col] = acc;
        }
    }
    out
}

/// `high_precision_reference`'s sibling for the width-tile's plain `[k, n]`
/// RHS layout — same f64-accumulated ground truth role documented above.
fn high_precision_reference_plain_rhs(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<f64> {
    let mut out = vec![0.0f64; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f64;
            for step in 0..k {
                acc += a[row * k + step] as f64 * b[step * n + col] as f64;
            }
            out[row * n + col] = acc;
        }
    }
    out
}

/// Width-tile counterpart to `check_size` above: same f64-ground-truth
/// correctness check plus the width-tile's own coverage identity
/// (`invocations * (WIDTH_TILE_ROWS * WIDTH_TILE_VECS * 4) + fallback_elements
/// == m*n`, `cpu.rs`'s doc on `WIDTH_TILE_GATE_PASSES`), driven by
/// `matmul_program`'s plain `[k, n]` RHS layout, which is what actually
/// selects `width_tile_plan` (`fast_path` in `run_reduce`, not
/// `reduction_fast_path`) rather than the dot tile `check_size` exercises.
/// `invocations_delta > 0` is asserted explicitly, not just the coverage
/// identity — a silent fallthrough to the generic loop would still satisfy
/// `covered == m*n` (`fallback_delta` alone would cover it), and that is
/// exactly the failure mode this test exists to catch.
fn check_width_size(size: usize) {
    let (m, k, n) = (size, size, size);
    let a: Vec<f32> = (0..m * k)
        .map(|index| (index as f32 * 0.0137).sin())
        .collect();
    let b: Vec<f32> = (0..k * n)
        .map(|index| (index as f32 * 0.0271).cos())
        .collect();

    #[cfg(all(target_arch = "aarch64", feature = "instrument"))]
    let (gate_before, invocations_before, fallback_before) =
        proxima_tensor::cpu::width_tile_counters();
    #[cfg(all(target_arch = "aarch64", feature = "instrument"))]
    let row_remainder_elements_before = proxima_tensor::cpu::width_tile_row_remainder_elements();

    let (program, _sum) = matmul_program(m as u32, k as u32, n as u32);
    let evaluated = match evaluate(&program, &[], &[&a, &b], &[]) {
        Ok(evaluated) => evaluated,
        Err(error) => panic!("width_tile size={size}: plain-rhs gemm evaluates: {error}"),
    };
    let actual = evaluated.root();

    let expected = naive_reference_plain_rhs(&a, &b, m, k, n);
    let ground_truth = high_precision_reference_plain_rhs(&a, &b, m, k, n);

    assert_eq!(
        actual.len(),
        expected.len(),
        "width_tile size={size}: output length mismatch"
    );

    let mut tile_error_vs_truth = 0.0f64;
    let mut naive_error_vs_truth = 0.0f64;
    let mut max_absolute_error_vs_truth = 0.0f64;
    let mut worst_absolute_index = 0usize;
    for index in 0..actual.len() {
        let truth = ground_truth[index];
        let tile_absolute_error = ((actual[index] as f64) - truth).abs();
        tile_error_vs_truth += tile_absolute_error.powi(2);
        naive_error_vs_truth += ((expected[index] as f64) - truth).powi(2);
        if tile_absolute_error > max_absolute_error_vs_truth {
            max_absolute_error_vs_truth = tile_absolute_error;
            worst_absolute_index = index;
        }
    }
    let tile_rms_vs_truth = (tile_error_vs_truth / actual.len() as f64).sqrt();
    let naive_rms_vs_truth = (naive_error_vs_truth / actual.len() as f64).sqrt();
    println!(
        "width_tile size={size}: tile_rms_vs_f64_truth={tile_rms_vs_truth:e} \
         naive_f32_rms_vs_f64_truth={naive_rms_vs_truth:e} ratio={:.3} \
         max_absolute_error_vs_f64_truth={max_absolute_error_vs_truth:e} \
         worst_absolute_index={worst_absolute_index}",
        tile_rms_vs_truth / naive_rms_vs_truth.max(1e-30)
    );

    assert!(
        tile_rms_vs_truth <= naive_rms_vs_truth,
        "width_tile size={size}: tile kernel RMS error vs f64 truth ({tile_rms_vs_truth:e}) exceeded the naive \
         f32 loop's own RMS error vs f64 truth ({naive_rms_vs_truth:e}) — the kernel must be no worse than the \
         obvious implementation"
    );

    let max_absolute_error_bound = 1e-6 * (k as f64).max(1.0);
    assert!(
        max_absolute_error_vs_truth <= max_absolute_error_bound,
        "width_tile size={size}: max absolute error vs f64 truth ({max_absolute_error_vs_truth:e}) exceeded bound \
         ({max_absolute_error_bound:e}) at worst_absolute_index={worst_absolute_index}"
    );

    #[cfg(all(target_arch = "aarch64", feature = "instrument"))]
    {
        let (gate_after, invocations_after, fallback_after) =
            proxima_tensor::cpu::width_tile_counters();
        let row_remainder_elements_after = proxima_tensor::cpu::width_tile_row_remainder_elements();
        let gate_delta = gate_after - gate_before;
        let invocations_delta = invocations_after - invocations_before;
        let row_remainder_elements_delta =
            row_remainder_elements_after - row_remainder_elements_before;
        let fallback_delta = fallback_after - fallback_before;
        // WIDTH_TILE_ROWS(4) * WIDTH_TILE_VECS(4) * 4 lanes/vec = 64
        // outputs/call (`cpu.rs::gemm_width_tile_neon` doc). Leading-row
        // remainders (`leading_total % WIDTH_TILE_ROWS`) no longer fall
        // through to the scalar path (`row_remainder_elements`, the 2-row/
        // 1-row NEON variants) -- only a column tail still reaches
        // `fallback_elements` (ROW 200's fix, `cpu.rs::run_width_tile_neon`).
        let covered = invocations_delta * 64 + row_remainder_elements_delta + fallback_delta;
        println!(
            "width_tile size={size}: gate_passes={gate_delta} invocations={invocations_delta} \
             row_remainder_elements={row_remainder_elements_delta} fallback_elements={fallback_delta} \
             covered={covered} m*n={expected_total}",
            expected_total = m * n
        );
        assert!(
            invocations_delta > 0,
            "width_tile size={size}: zero tile invocations — the width tile never fired, so the coverage \
             identity below would pass on the generic fallback alone and prove nothing"
        );
        assert_eq!(
            covered,
            (m * n) as u64,
            "width_tile size={size}: invocations*64 + row_remainder_elements + fallback_elements ({covered}) != \
             m*n ({expected})",
            expected = m * n
        );
        assert_eq!(
            gate_delta, 1,
            "width_tile size={size}: expected exactly one gate pass for one bound op"
        );
    }
}

fn check_size(size: usize) {
    let (m, k, n) = (size, size, size);
    let a: Vec<f32> = (0..m * k)
        .map(|index| (index as f32 * 0.0137).sin())
        .collect();
    let b_transposed: Vec<f32> = (0..n * k)
        .map(|index| (index as f32 * 0.0271).cos())
        .collect();

    #[cfg(all(target_arch = "aarch64", feature = "instrument"))]
    let (gate_before, invocations_before, fallback_before) =
        proxima_tensor::cpu::neon_tile_counters();
    #[cfg(all(target_arch = "aarch64", feature = "instrument"))]
    let row_remainder_elements_before = proxima_tensor::cpu::neon_tile_row_remainder_elements();

    let (program, _sum) = matmul_program_rhs_transposed(m as u32, k as u32, n as u32);
    let evaluated = match evaluate(&program, &[], &[&a, &b_transposed], &[]) {
        Ok(evaluated) => evaluated,
        Err(error) => panic!("size={size}: transposed-rhs gemm evaluates: {error}"),
    };
    let actual = evaluated.root();

    let expected = naive_reference(&a, &b_transposed, m, k, n);
    let ground_truth = high_precision_reference(&a, &b_transposed, m, k, n);

    assert_eq!(
        actual.len(),
        expected.len(),
        "size={size}: output length mismatch"
    );

    // Accuracy vs f64 ground truth: the tile kernel must be no worse than
    // the naive f32 loop, since that is the property that actually matters
    // (correctness), unlike f32-vs-f32 equality against a worse oracle.
    let mut tile_error_vs_truth = 0.0f64;
    let mut naive_error_vs_truth = 0.0f64;
    let mut max_absolute_error_vs_truth = 0.0f64;
    let mut worst_absolute_index = 0usize;
    for index in 0..actual.len() {
        let truth = ground_truth[index];
        let tile_absolute_error = ((actual[index] as f64) - truth).abs();
        tile_error_vs_truth += tile_absolute_error.powi(2);
        naive_error_vs_truth += ((expected[index] as f64) - truth).powi(2);
        if tile_absolute_error > max_absolute_error_vs_truth {
            max_absolute_error_vs_truth = tile_absolute_error;
            worst_absolute_index = index;
        }
    }
    let tile_rms_vs_truth = (tile_error_vs_truth / actual.len() as f64).sqrt();
    let naive_rms_vs_truth = (naive_error_vs_truth / actual.len() as f64).sqrt();
    println!(
        "size={size}: tile_rms_vs_f64_truth={tile_rms_vs_truth:e} naive_f32_rms_vs_f64_truth={naive_rms_vs_truth:e} \
         ratio={:.3} max_absolute_error_vs_f64_truth={max_absolute_error_vs_truth:e} \
         worst_absolute_index={worst_absolute_index}",
        tile_rms_vs_truth / naive_rms_vs_truth.max(1e-30)
    );

    assert!(
        tile_rms_vs_truth <= naive_rms_vs_truth,
        "size={size}: tile kernel RMS error vs f64 truth ({tile_rms_vs_truth:e}) exceeded the naive f32 loop's own \
         RMS error vs f64 truth ({naive_rms_vs_truth:e}) — the kernel must be no worse than the obvious \
         implementation"
    );

    // absolute bound: measured max abs error vs f64 truth ranged from
    // ~8e-6 at k=64 to ~4.9e-5 at k=1024 (f32 accumulation over k terms).
    // scale headroom with k, at ~13-21x the measured value per size, since
    // f32 accumulation error grows with the number of summed terms.
    let max_absolute_error_bound = 1e-6 * (k as f64).max(1.0);
    assert!(
        max_absolute_error_vs_truth <= max_absolute_error_bound,
        "size={size}: max absolute error vs f64 truth ({max_absolute_error_vs_truth:e}) exceeded bound \
         ({max_absolute_error_bound:e}) at worst_absolute_index={worst_absolute_index}"
    );

    #[cfg(all(target_arch = "aarch64", feature = "instrument"))]
    {
        let (gate_after, invocations_after, fallback_after) =
            proxima_tensor::cpu::neon_tile_counters();
        let row_remainder_elements_after = proxima_tensor::cpu::neon_tile_row_remainder_elements();
        let gate_delta = gate_after - gate_before;
        let invocations_delta = invocations_after - invocations_before;
        let row_remainder_elements_delta =
            row_remainder_elements_after - row_remainder_elements_before;
        let fallback_delta = fallback_after - fallback_before;
        // main 6x4 tile: 24 outputs/call. row-remainder tiles (widths 1..=5):
        // `rows * TILE_COLS` outputs/call, already summed into
        // `row_remainder_elements_delta` regardless of which width(s) fired.
        // everything else fell through to the scalar remainder path and is
        // already counted in `fallback_delta`.
        let covered = invocations_delta * 24 + row_remainder_elements_delta + fallback_delta;
        println!(
            "size={size}: gate_passes={gate_delta} invocations={invocations_delta} \
             row_remainder_elements={row_remainder_elements_delta} fallback_elements={fallback_delta} \
             covered={covered} m*n={expected_total}",
            expected_total = m * n
        );
        assert_eq!(
            covered,
            (m * n) as u64,
            "size={size}: invocations*24 + row_remainder_elements + fallback_elements ({covered}) != m*n \
             ({expected})",
            expected = m * n
        );
        assert_eq!(
            gate_delta, 1,
            "size={size}: expected exactly one gate pass for one bound op"
        );
    }
}

#[test]
fn neon_tile_full_output_64() {
    check_size(64);
}

#[test]
fn neon_tile_full_output_260() {
    check_size(260);
}

#[test]
fn neon_tile_full_output_1024() {
    check_size(1024);
}

/// 257 is not a multiple of either `TILE_ROWS` (6) or `TILE_COLS` (4), so
/// it forces both the row and column remainder paths.
#[test]
fn neon_tile_full_output_257_remainder_path() {
    check_size(257);
}

/// 1023 mod 6 == 3: row-remainder tile fires at ROWS=3.
#[test]
fn neon_tile_full_output_1023_row_remainder_boundary() {
    check_size(1023);
}

/// 1025 mod 6 == 5: row-remainder tile fires at ROWS=5, the tightest
/// register-budget case.
#[test]
fn neon_tile_full_output_1025_row_remainder_boundary() {
    check_size(1025);
}

/// 1021 mod 6 == 1, 1022 mod 6 == 2, 1026 mod 6 == 0 — the remaining arms of
/// the row-remainder match not covered by the other boundary tests above.
#[test]
fn neon_tile_full_output_1021_row_remainder_boundary() {
    check_size(1021);
}

#[test]
fn neon_tile_full_output_1022_row_remainder_boundary() {
    check_size(1022);
}

#[test]
fn neon_tile_full_output_1026_row_remainder_boundary() {
    check_size(1026);
}

/// 1023 mod WIDTH_TILE_ROWS(4) == 3, 1023 mod (WIDTH_TILE_VECS(4)*4=16) == 15:
/// forces both a row tail and a column tail in the width tile's own
/// remainder loops (`run_width_tile_neon`).
#[test]
fn width_tile_full_output_1023_row_and_column_remainder() {
    check_width_size(1023);
}

/// 1025 mod WIDTH_TILE_ROWS(4) == 1, 1025 mod 16 == 1: the other tail arm
/// not covered by 1023 above.
#[test]
fn width_tile_full_output_1025_row_and_column_remainder() {
    check_width_size(1025);
}

/// ROW 200's own correctness check: real BGE-shaped `M` values (7, 8, 9 are
/// BGE's own real per-sentence token counts, `docs/discipline.md` ROW 199;
/// 1, 2, 3, 5, 6 fill in every arm the greedy 2-then-1 row-remainder
/// dispatch (`cpu.rs::run_width_tile_neon`) can take: `m % WIDTH_TILE_ROWS
/// (4)` of 0 (m=8, main tile only), 1 (m=1,5,9 -- a single `ROWS=1` tile),
/// 2 (m=2,6 -- a single `ROWS=2` tile), 3 (m=3,7 -- `ROWS=2` THEN `ROWS=1`,
/// the only two-variant-in-one-call case)). `k=64, n=384` mirrors BGE's own
/// hidden size (`bge_epilogue_profile.rs`, ROW 195) and is a clean multiple
/// of `WIDTH_TILE_VECS*4=16`, so `fallback_delta` (the genuinely-scalar
/// column tail) is asserted `== 0` here — isolating this test to the row
/// remainder alone, since the column tail is already covered by
/// `width_tile_full_output_1023/1025_row_and_column_remainder` above.
///
/// `docs/discipline.md` ROW 201 lands the fix ROW 200 named but left
/// upstream (`resolve_reduce_axis_shape`, `cpu.rs`): a lone size-1 leading
/// axis (M=1, no other leading axis to survive the elide) now restores one
/// axis instead of squeezing to an empty list, so `width_tile_plan`'s gate
/// sees `leading_output_axes.len() == 1` again. `m=1` no longer needs its
/// own special case below — `m % WIDTH_TILE_ROWS == 1` for `m=1` exactly as
/// it does for `m=5,9`, so `m=1` now falls straight through the SAME
/// general assertions those arms already exercise (main tile does not fire,
/// a single `ROWS=1` NEON row-remainder tile does, column tail does not).
fn check_width_tile_small_m(m: usize) {
    let (k, n) = (64usize, 384usize);
    let a: Vec<f32> = (0..m * k)
        .map(|index| (index as f32 * 0.0137).sin())
        .collect();
    let b: Vec<f32> = (0..k * n)
        .map(|index| (index as f32 * 0.0271).cos())
        .collect();

    #[cfg(all(target_arch = "aarch64", feature = "instrument"))]
    let (gate_before, invocations_before, fallback_before) =
        proxima_tensor::cpu::width_tile_counters();
    #[cfg(all(target_arch = "aarch64", feature = "instrument"))]
    let row_remainder_invocations_before =
        proxima_tensor::cpu::width_tile_row_remainder_invocations();
    #[cfg(all(target_arch = "aarch64", feature = "instrument"))]
    let row_remainder_elements_before = proxima_tensor::cpu::width_tile_row_remainder_elements();

    let (program, _sum) = matmul_program(m as u32, k as u32, n as u32);
    let evaluated = match evaluate(&program, &[], &[&a, &b], &[]) {
        Ok(evaluated) => evaluated,
        Err(error) => panic!("width_tile small_m={m}: plain-rhs gemm evaluates: {error}"),
    };
    let actual = evaluated.root();

    let expected = naive_reference_plain_rhs(&a, &b, m, k, n);
    let ground_truth = high_precision_reference_plain_rhs(&a, &b, m, k, n);

    assert_eq!(
        actual.len(),
        expected.len(),
        "width_tile small_m={m}: output length mismatch"
    );

    let mut tile_error_vs_truth = 0.0f64;
    let mut naive_error_vs_truth = 0.0f64;
    let mut max_absolute_error_vs_truth = 0.0f64;
    let mut worst_absolute_index = 0usize;
    for index in 0..actual.len() {
        let truth = ground_truth[index];
        let tile_absolute_error = ((actual[index] as f64) - truth).abs();
        tile_error_vs_truth += tile_absolute_error.powi(2);
        naive_error_vs_truth += ((expected[index] as f64) - truth).powi(2);
        if tile_absolute_error > max_absolute_error_vs_truth {
            max_absolute_error_vs_truth = tile_absolute_error;
            worst_absolute_index = index;
        }
    }
    let tile_rms_vs_truth = (tile_error_vs_truth / actual.len() as f64).sqrt();
    let naive_rms_vs_truth = (naive_error_vs_truth / actual.len() as f64).sqrt();
    println!(
        "width_tile small_m={m}: tile_rms_vs_f64_truth={tile_rms_vs_truth:e} \
         naive_f32_rms_vs_f64_truth={naive_rms_vs_truth:e} ratio={:.3} \
         max_absolute_error_vs_f64_truth={max_absolute_error_vs_truth:e} worst_absolute_index={worst_absolute_index}",
        tile_rms_vs_truth / naive_rms_vs_truth.max(1e-30)
    );

    assert!(
        tile_rms_vs_truth <= naive_rms_vs_truth,
        "width_tile small_m={m}: tile kernel RMS error vs f64 truth ({tile_rms_vs_truth:e}) exceeded the naive f32 \
         loop's own RMS error vs f64 truth ({naive_rms_vs_truth:e})"
    );

    let max_absolute_error_bound = 1e-6 * (k as f64).max(1.0);
    assert!(
        max_absolute_error_vs_truth <= max_absolute_error_bound,
        "width_tile small_m={m}: max absolute error vs f64 truth ({max_absolute_error_vs_truth:e}) exceeded bound \
         ({max_absolute_error_bound:e}) at worst_absolute_index={worst_absolute_index}"
    );

    #[cfg(all(target_arch = "aarch64", feature = "instrument"))]
    {
        let (gate_after, invocations_after, fallback_after) =
            proxima_tensor::cpu::width_tile_counters();
        let row_remainder_invocations_after =
            proxima_tensor::cpu::width_tile_row_remainder_invocations();
        let row_remainder_elements_after = proxima_tensor::cpu::width_tile_row_remainder_elements();
        let gate_delta = gate_after - gate_before;
        let invocations_delta = invocations_after - invocations_before;
        let row_remainder_invocations_delta =
            row_remainder_invocations_after - row_remainder_invocations_before;
        let row_remainder_elements_delta =
            row_remainder_elements_after - row_remainder_elements_before;
        let fallback_delta = fallback_after - fallback_before;
        let covered = invocations_delta * 64 + row_remainder_elements_delta + fallback_delta;
        println!(
            "width_tile small_m={m}: gate_passes={gate_delta} invocations={invocations_delta} \
             row_remainder_invocations={row_remainder_invocations_delta} \
             row_remainder_elements={row_remainder_elements_delta} fallback_elements={fallback_delta} \
             covered={covered} m*n={expected_total}",
            expected_total = m * n
        );

        let main_tiles_expected = m / proxima_tensor::sized::WIDTH_TILE_ROWS;
        if main_tiles_expected > 0 {
            assert!(
                invocations_delta > 0,
                "width_tile small_m={m}: expected the main {rows}-row tile to fire ({main_tiles_expected} whole \
                 tile(s) of rows), got zero invocations",
                rows = proxima_tensor::sized::WIDTH_TILE_ROWS
            );
        } else {
            assert_eq!(
                invocations_delta, 0,
                "width_tile small_m={m}: m < WIDTH_TILE_ROWS, the main tile must never fire"
            );
        }

        let remainder_rows = m % proxima_tensor::sized::WIDTH_TILE_ROWS;
        if remainder_rows == 0 {
            assert_eq!(
                row_remainder_invocations_delta, 0,
                "width_tile small_m={m}: m is a clean multiple of WIDTH_TILE_ROWS, no remainder variant should fire"
            );
        } else {
            assert!(
                row_remainder_invocations_delta > 0,
                "width_tile small_m={m}: remainder_rows={remainder_rows} > 0 but zero row-remainder tile \
                 invocations — N==0 would mean the 2-row/1-row NEON variants never fired and the scalar path \
                 silently absorbed the remainder instead"
            );
        }

        assert_eq!(
            fallback_delta, 0,
            "width_tile small_m={m}: n=384 is a clean multiple of WIDTH_TILE_VECS*4=16, so the column tail (the \
             only path still allowed to reach width_tile_scalar_cell) must never fire — a nonzero fallback here \
             means some leading-row remainder silently fell through to the scalar row-tail again"
        );

        assert_eq!(
            covered,
            (m * n) as u64,
            "width_tile small_m={m}: invocations*64 + row_remainder_elements + fallback_elements ({covered}) != \
             m*n ({expected})",
            expected = m * n
        );
        assert_eq!(
            gate_delta, 1,
            "width_tile small_m={m}: expected exactly one gate pass for one bound op"
        );
    }
}

/// ROW 200 found M=1 never even reached `width_tile_plan`'s gate -- its own
/// leading axis (extent 1) squeezed to an empty `leading_output_axes` list
/// upstream, in `resolve_reduce_axis_shape`, indistinguishable there from a
/// genuine batch axis. ROW 201 fixes the upstream squeeze (`cpu.rs`:
/// restore one axis when filtering would otherwise empty the list), so M=1
/// now reaches the SAME general row-remainder assertions `m=5,9` already
/// exercise (`m % WIDTH_TILE_ROWS == 1` for all three) -- a single `ROWS=1`
/// NEON tile, not the scalar `width_tile_scalar_cell` path.
#[test]
fn width_tile_small_m_1_fires_the_one_row_neon_remainder() {
    check_width_tile_small_m(1);
}

#[test]
fn width_tile_small_m_2_pure_two_row_remainder() {
    check_width_tile_small_m(2);
}

#[test]
fn width_tile_small_m_3_two_then_one_row_remainder() {
    check_width_tile_small_m(3);
}

#[test]
fn width_tile_small_m_5_main_tile_plus_one_row_remainder() {
    check_width_tile_small_m(5);
}

#[test]
fn width_tile_small_m_6_main_tile_plus_two_row_remainder() {
    check_width_tile_small_m(6);
}

/// BGE's own real "quantum physics..." sentence, `M=7` (ROW 199).
#[test]
fn width_tile_small_m_7_bge_real_sentence_two_then_one_remainder() {
    check_width_tile_small_m(7);
}

/// BGE's own real "a cat is sitting..." sentence, `M=9` (ROW 199).
#[test]
fn width_tile_small_m_9_bge_real_sentence_one_row_remainder() {
    check_width_tile_small_m(9);
}
