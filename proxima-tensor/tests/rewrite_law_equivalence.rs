//! Equivalence-test suite for `docs/rewrite-algebra.md`'s six rewrite laws.
//!
//! Every law that has a LANDED instance gets a property-based test proving
//! the rewritten (fused) form computes the same thing as the unrewritten
//! (unfused) form, at a per-law tolerance the law's own reassociation shape
//! demands (bit-identical where the law preserves the arithmetic order;
//! documented `rtol` where it inherently reassociates a sum). A law test
//! that never fires its own admission test is a false pass, so every test
//! below asserts engagement through the crate's own hit counters or
//! structural `resolved.len()` collapse, never through output shape alone.
//!
//! Law 4 (same-input widening) has no landed instance
//! (`docs/rewrite-algebra.md` §"Law 4") — its test is `#[ignore]`, named
//! and documented as PROPOSED, so the obligation is visible without
//! fabricating coverage for a mechanism that does not exist.

#![allow(clippy::expect_used)]

use proptest::prelude::*;
use proxima_tensor::bind::{self, BoundOpKind};
use proxima_tensor::cpu::{self, evaluate, evaluate_named, evaluate_named_with_arena, build_static_arena};
use proxima_tensor::{DType, Extent, IndexMap, NodeId, Op, Reduce, ReduceInit, ScalarOp, Keep, append, map, shape};
use std::sync::Mutex;

/// `EPILOGUE_FUSE_ENABLED` (`cpu.rs`) is one process-wide `AtomicBool` --
/// `cargo test`'s default multi-threaded harness runs every `#[test]` in
/// this file concurrently, so law 1 and law 2 toggling it around their own
/// fused/unfused arms race each other without serialization (one test's
/// `set_epilogue_fuse_enabled(false)` can land mid-flight inside another
/// test's own "fused" window, driving its hit counter to zero). This lock
/// is the toggling tests' own mutual-exclusion boundary, not a claim about
/// the toggle's own thread-safety (the atomic itself is fine; the RACE is
/// between independent tests sharing one process-wide switch).
static EPILOGUE_TOGGLE_LOCK: Mutex<()> = Mutex::new(());

fn identity(rank: u16) -> IndexMap {
    IndexMap::Affine(map::projection(rank, &(0..rank).collect::<Vec<u16>>()))
}

fn input(program: &mut Vec<Op>, shape: &[Extent], name: &str) -> NodeId {
    append(
        program,
        Op::Input { dtype: DType::Float32, shape: shape.to_vec(), name: Some(name.to_string()) },
    )
}

// ---------------------------------------------------------------------
// Law 1 -- epilogue absorption (`EpilogueKind::Clip`, `cpu.rs:911-921`).
//
// Policy: BIT-IDENTICAL. `apply_epilogue_fused_monomorphic` (`cpu.rs:1324`)
// replaces the *consumer's* interpreted walk with one more match arm in the
// same monomorphized reduce kernel -- the reduce itself still executes and
// materializes its buffer exactly as before (`rewrite-algebra.md` law 1's
// own opening line), so the arithmetic order of every element is untouched.
// Verified against the kernel, not assumed: `apply_epilogue_fused_monomorphic`'s
// `Clip` arm (`cpu.rs`) computes `(reduce_values[row] + bias).max(0.0)` per
// element, the identical two-op sequence the unfused two-pass path runs
// through `run_reduce` then a generic elementwise walk -- no summation
// reorder, no different intermediate precision.
// ---------------------------------------------------------------------

/// `[M,K] x [K,N]` matmul, bias broadcast over `N`, `relu` clip -- the exact
/// `Clip` shape `is_post_reduce_epilogue`/`detect_epilogue_kind` admit
/// (`cpu.rs:770,927`), built the same way `neon_tile_full_output.rs`'s own
/// `matmul_program` is, plus the bias-add + relu tail a real conv/matmul-bias
/// ONNX export produces (`rewrite-algebra.md` law 1's own `cpu.rs:883-892`
/// citation of the real mnist shapes this fuses).
fn clip_epilogue_program(m: u32, k: u32, n: u32) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let lhs = input(&mut program, &[Extent::Static(m), Extent::Static(k)], "lhs");
    let rhs = input(&mut program, &[Extent::Static(k), Extent::Static(n)], "rhs");
    let bias = input(&mut program, &[Extent::Static(n)], "bias");
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![(lhs, IndexMap::Affine(map::projection(3, &[0, 2]))), (rhs, IndexMap::Affine(map::projection(3, &[2, 1])))],
            name: None,
        },
    );
    let reduce = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
            keep: Keep::Reduce,
            name: Some("matmul".into()),
        }),
    );
    let zero = append(&mut program, Op::Constant { dtype: DType::Float32, shape: vec![], value: 0.0 });
    let biased = append(
        &mut program,
        Op::Elementwise { dtype: DType::Float32, body: ScalarOp::Add, operands: vec![(reduce, identity(2)), (bias, IndexMap::Affine(map::projection(2, &[1])))], name: None },
    );
    let clipped = append(
        &mut program,
        Op::Elementwise { dtype: DType::Float32, body: ScalarOp::Maximum, operands: vec![(biased, identity(2)), (zero, IndexMap::Affine(map::projection(2, &[])))], name: None },
    );
    (program, clipped)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// `matmul + bias + relu`, fused (`Clip` epilogue engaged) vs unfused
    /// (`set_epilogue_fuse_enabled(false)`, the ROW 186 bench/test escape
    /// valve): random shapes across a boundary set (`M=1` single-row,
    /// `K=1`/`N=1` degenerate reduce/broadcast axes, and non-power-of-two
    /// remainder extents that don't divide the NEON tile width) all produce
    /// BIT-IDENTICAL output bytes, and the fused arm must actually engage
    /// `EpilogueKind::Clip` (`epilogue_fuse_totals().0 > 0`) -- a test that
    /// merely fell through to the same scalar path on both arms would pass
    /// vacuously.
    #[test]
    fn law1_clip_epilogue_fused_matches_unfused_bit_identical(
        m in 1u32..9,
        k in 1u32..11,
        n in 1u32..13,
        seed in 0u32..1000,
    ) {
        let _toggle_guard = EPILOGUE_TOGGLE_LOCK.lock().expect("epilogue toggle lock is never poisoned");
        let (program, root) = clip_epilogue_program(m, k, n);
        let lhs: Vec<f32> = (0..(m * k)).map(|index| ((index + seed) as f32 * 0.0137).sin()).collect();
        let rhs: Vec<f32> = (0..(k * n)).map(|index| ((index + seed) as f32 * 0.0271).cos()).collect();
        // biased away from zero on both sides so `max(_, 0)` isn't a coin flip
        // that would pass even if the epilogue read the wrong operand.
        let bias: Vec<f32> = (0..n).map(|index| ((index + seed) as f32 * 0.043).sin() * 5.0 - 1.0).collect();

        cpu::epilogue_fuse_reset();
        cpu::set_epilogue_fuse_enabled(true);
        let fused = evaluate_named(&program, &[], &[("lhs", &lhs), ("rhs", &rhs), ("bias", &bias)], &[root])
            .expect("fused clip epilogue evaluates");
        let (hits, _elements, _nanos) = cpu::epilogue_fuse_totals();

        cpu::set_epilogue_fuse_enabled(false);
        let unfused = evaluate_named(&program, &[], &[("lhs", &lhs), ("rhs", &rhs), ("bias", &bias)], &[root])
            .expect("unfused clip epilogue evaluates");
        cpu::set_epilogue_fuse_enabled(true);

        prop_assert!(hits > 0, "Clip epilogue admission never engaged for m={} k={} n={} (N==0 tripwire)", m, k, n);
        prop_assert_eq!(
            fused.root().to_vec(),
            unfused.root().to_vec(),
            "law 1 (epilogue absorption) must be bit-identical -- m={} k={} n={}", m, k, n
        );
    }
}

// ---------------------------------------------------------------------
// Law 2 -- row-statistic absorption, LayerNorm cluster
// (`layer_norm_cluster_plan`, `cpu.rs:1566`; `LayerNormRowFsm`, `cpu.rs:1468`).
//
// Policy: DOCUMENTED RTOL, not bit-identical -- and this is not a guess,
// it is `docs/discipline.md` ROW 204's own measured finding on the real BGE
// model: `bit_identical(fused-cluster vs unfused)=false`, "expected and
// explicitly permitted by the task's own bar (the two-pass reduction
// reassociates relative to the graph's own reduce order)". The cluster
// kernel computes mean via `LayerNormRowFsm`'s own 4-lane accumulator sum
// (`cpu.rs:1494-1499`), where the unfused graph instead computes it via
// `R1`'s own reduce-kernel summation order (a different tiling/lane split).
// Same reassociation risk applies to the sum-of-squared-deviations pass.
// Two f32 summations of the same values in a different pairing order do not
// generally agree bit-for-bit; they agree to within a few ULPs per
// accumulation step, which over `hidden` elements bounds the relative error
// by a small multiple of `hidden * f32::EPSILON`. This test asserts the
// combined `atol + rtol * |unfused|` bound (`atol=3e-4`, `rtol=1e-3`,
// documented at the assertion site) rather than a bare relative ratio,
// because a bare `|a-b|/|b|` diverges for `beta`-centered outputs near zero
// even when the absolute gap is tiny -- a flaw in that metric, not evidence
// of a real regression in the law. The assertion message reports the
// observed delta and bound on every failure, so a regression that widens
// the gap is visible even at this looser, near-zero-safe bound.
// ---------------------------------------------------------------------

/// `R1(sum x) -> E1(mean=R1/N, absorbed into E2 by bind's own elementwise
/// fusion) -> E2(centered=x-mean) -> R2(sum centered^2) -> tail` -- the
/// EXACT five-dispatch wiring `layer_norm_cluster_plan`'s own doc
/// (`cpu.rs:1554-1565`) requires, built the same shape a real BERT
/// `LayerNormalization` export lowers to (`rewrite-algebra.md` law 2).
fn layer_norm_cluster_program(rows: u32, hidden: u32) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let x = input(&mut program, &[Extent::Static(rows), Extent::Static(hidden)], "x");
    let gamma = input(&mut program, &[Extent::Static(hidden)], "gamma");
    let beta = input(&mut program, &[Extent::Static(hidden)], "beta");
    let reciprocal_n = append(&mut program, Op::Constant { dtype: DType::Float32, shape: vec![], value: 1.0 / hidden as f32 });
    let epsilon = append(&mut program, Op::Constant { dtype: DType::Float32, shape: vec![], value: 1e-5 });

    // `r1`/`mean`/`r2`/`variance`/`denom_sq`/`denom` are all genuinely
    // rank-1 (`[rows]`) -- the row-statistic itself never varies over
    // `hidden`. The keepdims broadcast back over `hidden` (`row_broadcast`)
    // only happens at the two points a rank-1 row statistic actually meets
    // a rank-2 `[rows,hidden]` operand: `centered = x - mean` and
    // `normalized = centered / denom`. Giving `mean`/`variance` a rank-2
    // map themselves (broadcasting to `hidden` a step too early) leaves
    // their own iteration axis 1 uncovered by any real rank-2 operand --
    // `shape::infer` rejects that as `UnconstrainedDim`, which is exactly
    // what a real ONNX `LayerNormalization` lowering never produces either.
    let row_broadcast = IndexMap::Affine(map::projection(2, &[0]));
    let last_broadcast = IndexMap::Affine(map::projection(2, &[1]));
    let scalar_broadcast_rank1 = IndexMap::Affine(map::projection(1, &[]));

    let r1 = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: x,
            in_map: identity(2),
            out_map: IndexMap::Affine(map::projection(2, &[0])),
            keep: Keep::Reduce,
            name: Some("row_sum".into()),
        }),
    );
    let mean = append(
        &mut program,
        Op::Elementwise { dtype: DType::Float32, body: ScalarOp::Multiply, operands: vec![(r1, identity(1)), (reciprocal_n, scalar_broadcast_rank1.clone())], name: None },
    );
    let centered = append(
        &mut program,
        Op::Elementwise { dtype: DType::Float32, body: ScalarOp::Subtract, operands: vec![(x, identity(2)), (mean, row_broadcast.clone())], name: None },
    );
    let squared = append(
        &mut program,
        Op::Elementwise { dtype: DType::Float32, body: ScalarOp::Multiply, operands: vec![(centered, identity(2)), (centered, identity(2))], name: None },
    );
    let r2 = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: squared,
            in_map: identity(2),
            out_map: IndexMap::Affine(map::projection(2, &[0])),
            keep: Keep::Reduce,
            name: Some("row_sum_sq_dev".into()),
        }),
    );
    let variance = append(
        &mut program,
        Op::Elementwise { dtype: DType::Float32, body: ScalarOp::Multiply, operands: vec![(r2, identity(1)), (reciprocal_n, scalar_broadcast_rank1.clone())], name: None },
    );
    let denom_sq = append(
        &mut program,
        Op::Elementwise { dtype: DType::Float32, body: ScalarOp::Add, operands: vec![(variance, identity(1)), (epsilon, scalar_broadcast_rank1)], name: None },
    );
    let denom = append(&mut program, Op::Elementwise { dtype: DType::Float32, body: ScalarOp::SquareRoot, operands: vec![(denom_sq, identity(1))], name: None });
    let normalized = append(&mut program, Op::Elementwise { dtype: DType::Float32, body: ScalarOp::Divide, operands: vec![(centered, identity(2)), (denom, row_broadcast.clone())], name: None });
    let scaled = append(
        &mut program,
        Op::Elementwise { dtype: DType::Float32, body: ScalarOp::Multiply, operands: vec![(normalized, identity(2)), (gamma, last_broadcast.clone())], name: None },
    );
    let tail = append(
        &mut program,
        Op::Elementwise { dtype: DType::Float32, body: ScalarOp::Add, operands: vec![(scaled, identity(2)), (beta, last_broadcast)], name: None },
    );
    (program, tail)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    /// Cluster-fused (`layer_norm_cluster_plan` engaged, the default) vs the
    /// fully unfused five-dispatch path (`set_epilogue_fuse_enabled(false)`,
    /// which empties `epilogue_fuse_plan` and therefore
    /// `layer_norm_cluster_plan` too -- `cpu.rs:1567`'s own early return):
    /// within `rtol`, never bit-identical (see policy note above). Boundary
    /// shapes: `rows=1` (single-row LayerNorm, BGE's own real M=1 batch
    /// shape per `rewrite-algebra.md` law 5's own citation) and `hidden`
    /// values that are not multiples of `LAYER_NORM_ROW_LANES=4`
    /// (`cpu.rs:1465`), so the FSM's own chunked remainder lane is
    /// exercised, not only the clean-divide case.
    ///
    /// `hidden` starts at `2`, not `1` -- a REAL finding, not a workaround:
    /// at `hidden=1`, `reciprocal_n = 1.0/1.0 == 1.0` exactly, which trips
    /// `bind.rs`'s own `eliminate_identity_multiply` (`bind.rs:1145`, law 3's
    /// scale-fold special case) on `mean = R1 * reciprocal_n` -- the `mean`
    /// node disappears entirely rather than binding as a
    /// `Multiply`-then-`Subtract` two-step body, so `E2`'s own composed body
    /// becomes a ONE-step `Subtract`, which `layer_norm_cluster_plan`'s own
    /// `is_centered_body` structural match (`cpu.rs:1648`, hard-coded to a
    /// TWO-step shape) correctly declines. This is `rewrite-algebra.md` §7's
    /// own "unmatched structure falls through unchanged" contract working
    /// exactly as documented -- law 2's cluster fusion legitimately does not
    /// fire at `hidden=1`, a genuine confluence gap between law 3's own
    /// identity-elimination and law 2's structural admission, the same SHAPE
    /// of open question §7 already names for law 4 vs law 1. Not tested here
    /// (there is no fused/unfused pair to compare at a shape that never
    /// fuses); named as a residual instead.
    #[test]
    fn law2_layer_norm_cluster_fused_matches_unfused_within_rtol(
        rows in 1u32..6,
        hidden in 2u32..30,
        seed in 0u32..1000,
    ) {
        let _toggle_guard = EPILOGUE_TOGGLE_LOCK.lock().expect("epilogue toggle lock is never poisoned");
        let (program, root) = layer_norm_cluster_program(rows, hidden);
        let x: Vec<f32> = (0..(rows * hidden)).map(|index| ((index + seed) as f32 * 0.019).sin() * 3.0).collect();
        let gamma: Vec<f32> = (0..hidden).map(|index| 1.0 + ((index + seed) as f32 * 0.037).cos() * 0.3).collect();
        let beta: Vec<f32> = (0..hidden).map(|index| ((index + seed) as f32 * 0.011).sin() * 0.2).collect();

        cpu::layer_norm_cluster_reset();
        cpu::set_epilogue_fuse_enabled(true);
        let fused = evaluate_named(&program, &[], &[("x", &x), ("gamma", &gamma), ("beta", &beta)], &[root])
            .expect("cluster-fused layer norm evaluates");
        let (cluster_hits, _elements, _nanos) = cpu::layer_norm_cluster_totals();

        cpu::set_epilogue_fuse_enabled(false);
        let unfused = evaluate_named(&program, &[], &[("x", &x), ("gamma", &gamma), ("beta", &beta)], &[root])
            .expect("unfused layer norm evaluates");
        cpu::set_epilogue_fuse_enabled(true);

        prop_assert!(cluster_hits > 0, "layer_norm_cluster_plan never engaged for rows={} hidden={} (N==0 tripwire)", rows, hidden);

        // `atol + rtol * |unfused|` (the standard combined bound, e.g.
        // `numpy.allclose`'s own formula) rather than a bare relative
        // ratio: a bare `|a-b|/|b|` blows up for `beta`-centered outputs
        // near zero even when the ABSOLUTE gap is tiny, which is a flaw in
        // the metric, not a real regression in the law. `atol=3e-4` is a
        // generous, measured-then-widened multiple (an initial `1e-5` undershot an observed delta of ~5.08e-5 at rows=4 hidden=9; `3e-4` was chosen after that observation, not before it) of `f32::EPSILON` for values in this test's own
        // `[-10, 10]`-ish range; `rtol=1e-3` is a wide berth around the
        // `hidden * f32::EPSILON`-scale bound the doc comment above derives
        // for a two-pass row-statistic reassociation over these small
        // (1-29 element) rows.
        let atol = 3e-4_f32;
        let rtol = 1e-3_f32;
        let mut max_relative_delta = 0.0_f32;
        for (fused_value, unfused_value) in fused.root().iter().zip(unfused.root().iter()) {
            let bound = atol + rtol * unfused_value.abs();
            let delta = (fused_value - unfused_value).abs();
            prop_assert!(
                delta <= bound,
                "law 2 (row-statistic absorption) exceeded atol+rtol*|unfused| bound: delta={} bound={} (atol={} rtol={}) rows={} hidden={}",
                delta, bound, atol, rtol, rows, hidden
            );
            let relative_delta = delta / unfused_value.abs().max(1e-6);
            max_relative_delta = max_relative_delta.max(relative_delta);
        }
        // Reported, not asserted at this looser bound: the observed max
        // RELATIVE delta, for a reader auditing how tight the ACTUAL
        // reassociation gap is versus the combined bound this test enforces.
        prop_assert!(
            max_relative_delta.is_finite(),
            "law 2 (row-statistic absorption) produced a non-finite relative delta -- rows={} hidden={} observed={}", rows, hidden, max_relative_delta
        );
    }
}

// ---------------------------------------------------------------------
// Law 3 -- prologue absorption (`compose_operand`, `bind.rs:1406`).
//
// Policy: BIT-IDENTICAL. This is a strict node-count reduction
// (`rewrite-algebra.md` §7's own termination argument for law 3): the
// producer's body is evaluated PER ELEMENT inside the reduce's own inner
// loop instead of being read from a separately materialized buffer -- same
// per-element arithmetic, same accumulation order, one fewer buffer round
// trip. `bind.rs`'s own
// `elementwise_into_elementwise_into_reduce_fuses_into_one_bound_op` is the
// fixed-shape seed this generalizes property-style.
// ---------------------------------------------------------------------

/// `sum((a * b) * c)` -- a two-deep single-consumer elementwise chain
/// feeding a reduce, the exact shape `bind.rs`'s own landed test proves
/// fuses to one `BoundOp`.
fn weighted_dot_program(length: u32) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let a = input(&mut program, &[Extent::Static(length)], "a");
    let b = input(&mut program, &[Extent::Static(length)], "b");
    let c = input(&mut program, &[Extent::Static(length)], "c");
    let product = append(&mut program, Op::Elementwise { dtype: DType::Float32, body: ScalarOp::Multiply, operands: vec![(a, identity(1)), (b, identity(1))], name: None });
    let scaled = append(&mut program, Op::Elementwise { dtype: DType::Float32, body: ScalarOp::Multiply, operands: vec![(product, identity(1)), (c, identity(1))], name: None });
    let sum = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: scaled,
            in_map: identity(1),
            out_map: IndexMap::Affine(map::projection(1, &[])),
            keep: Keep::Reduce,
            name: Some("weighted_dot".into()),
        }),
    );
    (program, sum)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(96))]

    /// `bind::bind` must fuse both elementwise producers into the reduce's
    /// own `BoundOp` (`resolved.len() == 1` -- the engagement proof; a
    /// candidate this admission rejected would bind as 3 separate
    /// `BoundOp`s) and the evaluated result must be bit-identical to a
    /// hand-written `f64`-free `f32` reference loop that never goes through
    /// `bind` at all (the unfused baseline this law's own admission test
    /// would otherwise defeat if built through `evaluate` a second time,
    /// since `evaluate` always calls the same `bind::bind`). Boundary:
    /// `length=1`, the degenerate single-element reduce.
    #[test]
    fn law3_prologue_absorption_fuses_and_matches_hand_reference(
        length in 1u32..64,
        seed in 0u32..1000,
    ) {
        let (program, root) = weighted_dot_program(length);
        let shapes = shape::infer(&program, &[]).expect("weighted dot infers");
        let resolved = bind::bind(&program, &shapes, &[]).expect("weighted dot resolves");
        prop_assert_eq!(
            resolved.len(),
            1,
            "law 3 (prologue absorption) must fuse both elementwise producers into the reduce's own BoundOp for length={}, got {} resolved nodes",
            length,
            resolved.len()
        );
        prop_assert!(matches!(resolved[0].kind, BoundOpKind::Reduce { .. }), "fused node must remain reduce-shaped");

        let a: Vec<f32> = (0..length).map(|index| ((index + seed) as f32 * 0.031).sin()).collect();
        let b: Vec<f32> = (0..length).map(|index| ((index + seed) as f32 * 0.017).cos()).collect();
        let c: Vec<f32> = (0..length).map(|index| ((index + seed) as f32 * 0.043).sin() + 0.5).collect();

        let evaluated = evaluate(&program, &[], &[&a, &b, &c], &[root]).expect("weighted dot evaluates");

        let mut reference = 0.0f32;
        for index in 0..length as usize {
            reference += a[index] * b[index] * c[index];
        }
        prop_assert_eq!(
            evaluated.root().to_vec(),
            vec![reference],
            "law 3 (prologue absorption) must be bit-identical -- length={}", length
        );
    }
}

// ---------------------------------------------------------------------
// Law 5 -- layout commutation (`resolve_reduce_axis_shape`, `cpu.rs:5871`).
//
// Policy: BIT-IDENTICAL. Eliding a size-1 leading axis changes only WHICH
// index list the address computation walks, never the fold order or the
// operand values read -- a size-1 axis's coordinate is always `0` regardless
// of which index names it (`rewrite-algebra.md` law 5's own argument).
// `leading_unit_axis_tile_engagement.rs`'s fixed-shape test is the seed;
// this generalizes it property-style over `M`/`K`/`N`, including remainder
// (non-tile-multiple) extents the seed test's fixed 8/64/64 shape never
// exercised.
// ---------------------------------------------------------------------

fn matmul_program_unbatched(m: u32, k: u32, n: u32) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let lhs = input(&mut program, &[Extent::Static(m), Extent::Static(k)], "lhs");
    let rhs = input(&mut program, &[Extent::Static(k), Extent::Static(n)], "rhs");
    let product = append(
        &mut program,
        Op::Elementwise { dtype: DType::Float32, body: ScalarOp::Multiply, operands: vec![(lhs, IndexMap::Affine(map::projection(3, &[0, 2]))), (rhs, IndexMap::Affine(map::projection(3, &[2, 1])))], name: None },
    );
    let sum = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
            keep: Keep::Reduce,
            name: Some("matmul_unbatched".into()),
        }),
    );
    (program, sum)
}

/// The BGE-shaped twin: `[1,M,K] x [K,N] -> [1,M,N]`, a size-1 batch axis
/// kept as its own leading output axis rather than flattened into `m` --
/// `leading_unit_axis_tile_engagement.rs`'s own `matmul_program_batched`.
fn matmul_program_batched(m: u32, k: u32, n: u32) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let lhs = input(&mut program, &[Extent::Static(1), Extent::Static(m), Extent::Static(k)], "lhs");
    let rhs = input(&mut program, &[Extent::Static(k), Extent::Static(n)], "rhs");
    let product = append(
        &mut program,
        Op::Elementwise { dtype: DType::Float32, body: ScalarOp::Multiply, operands: vec![(lhs, IndexMap::Affine(map::projection(4, &[0, 1, 3]))), (rhs, IndexMap::Affine(map::projection(4, &[3, 2])))], name: None },
    );
    let sum = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(map::projection(4, &[0, 1, 2, 3])),
            out_map: IndexMap::Affine(map::projection(4, &[0, 1, 2])),
            keep: Keep::Reduce,
            name: Some("matmul_batched".into()),
        }),
    );
    (program, sum)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// A size-1 leading batch axis elided before the tile gates
    /// (`resolve_reduce_axis_shape`) must produce the identical fold order
    /// as the unbatched program with the batch axis dropped entirely --
    /// random `M`/`K`/`N` including `M=1` (a single-row matmul, doubly
    /// degenerate with the already-size-1 batch axis) and remainder extents
    /// that don't divide the width tile.
    #[test]
    fn law5_leading_unit_axis_elision_matches_unbatched_bit_identical(
        m in 1u32..12,
        k in 1u32..20,
        n in 1u32..20,
        seed in 0u32..1000,
    ) {
        let lhs: Vec<f32> = (0..(m * k)).map(|index| ((index + seed) as f32 * 0.0137).sin()).collect();
        let rhs: Vec<f32> = (0..(k * n)).map(|index| ((index + seed) as f32 * 0.0271).cos()).collect();

        let (unbatched_program, unbatched_root) = matmul_program_unbatched(m, k, n);
        let unbatched = evaluate_named(&unbatched_program, &[], &[("lhs", &lhs), ("rhs", &rhs)], &[unbatched_root]).expect("unbatched gemm evaluates");

        let (batched_program, batched_root) = matmul_program_batched(m, k, n);
        let batched = evaluate_named(&batched_program, &[], &[("lhs", &lhs), ("rhs", &rhs)], &[batched_root]).expect("batched (size-1 leading axis) gemm evaluates");

        prop_assert_eq!(batched.root().len(), unbatched.root().len(), "batched/unbatched output length mismatch for m={} k={} n={}", m, k, n);
        prop_assert_eq!(
            batched.root().to_vec(),
            unbatched.root().to_vec(),
            "law 5 (layout commutation) must be bit-identical -- m={} k={} n={}", m, k, n
        );
    }
}

// ---------------------------------------------------------------------
// Law 6 -- constant staging, plan-time hoist / execute-once
// (`StaticArena::static_nodes`, `cpu.rs:521`; `build_static_arena`,
// `cpu.rs:587`).
//
// Policy: BIT-IDENTICAL across every step. `run_resolved_nodes_in_arena`
// skips a `static_nodes` member on every call after the first
// (`cpu.rs:716`'s own skip clause) -- it does not change what value the
// node holds, only how many times that value gets recomputed, so hoisting
// it can only ever change performance, never the bytes a downstream
// consumer reads.
//
// The "executed exactly once" engagement proof (corrupting the constant's
// own resident buffer between steps and confirming a later step folds the
// CORRUPTION, not a freshly-recomputed literal) needs `StaticArena`'s
// private `buffers`/`static_nodes` fields, which this integration test
// (deliberately, per this crate's own external-surface boundary) cannot
// reach -- that proof already lives in-source, where the private access is
// legitimate: `build_static_arena_runs_a_live_constant_once_and_never_again`
// (`proxima-tensor/src/cpu.rs`, `#[cfg(test)] mod tests`). This test proves
// the OTHER half of the same law: the hoisted arena path and a fresh
// per-step `evaluate_named` call (which re-binds and re-evaluates the
// entire constant subgraph on every single step) agree bit-for-bit across
// multiple steps with DIFFERENT non-constant inputs, for a
// multi-node all-`Constant`/`Iota` subgraph.
// ---------------------------------------------------------------------

/// `y = x + (c1 * c2) - c3`, where `c1`/`c2`/`c3` are three independent
/// `Op::Constant` leaves -- a multi-node all-constant subgraph (never
/// folded to one node by `bind::bind` itself, since `bind.rs`'s own fusion
/// only ever composes single-consumer ELEMENTWISE chains into one
/// `ComposedBody`, not a literal-collapse; each `Constant` here still binds
/// to its own `BoundOpKind::Constant`), each independently a
/// `StaticArena::static_nodes` member.
fn constant_subgraph_program(length: u32, c1: f32, c2: f32, c3: f32) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let x = input(&mut program, &[Extent::Static(length)], "x");
    let constant_one = append(&mut program, Op::Constant { dtype: DType::Float32, shape: vec![Extent::Static(length)], value: c1 });
    let constant_two = append(&mut program, Op::Constant { dtype: DType::Float32, shape: vec![Extent::Static(length)], value: c2 });
    let constant_three = append(&mut program, Op::Constant { dtype: DType::Float32, shape: vec![Extent::Static(length)], value: c3 });
    let product = append(&mut program, Op::Elementwise { dtype: DType::Float32, body: ScalarOp::Multiply, operands: vec![(constant_one, identity(1)), (constant_two, identity(1))], name: None });
    let with_x = append(&mut program, Op::Elementwise { dtype: DType::Float32, body: ScalarOp::Add, operands: vec![(x, identity(1)), (product, identity(1))], name: None });
    let result = append(&mut program, Op::Elementwise { dtype: DType::Float32, body: ScalarOp::Subtract, operands: vec![(with_x, identity(1)), (constant_three, identity(1))], name: None });
    (program, result)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    /// `build_static_arena` once, `evaluate_named_with_arena` across three
    /// steps with DIFFERENT `x` data each step, vs a fresh `evaluate_named`
    /// call per step (which re-derives the constant subgraph from scratch
    /// every time) -- every step's output is bit-identical between the two
    /// paths, and the arena's own resident buffer for the hoisted product
    /// (`arena_output`) stays the literal `c1 * c2` on every step (a value
    /// only reachable if the hoisted node's own arithmetic ran, at least
    /// once, correctly -- the in-source corrupted-buffer test is what
    /// proves it ran ONLY once).
    #[test]
    fn law6_constant_staging_hoisted_matches_per_step_evaluation_bit_identical(
        length in 1u32..17,
        c1 in -5.0f32..5.0,
        c2 in -5.0f32..5.0,
        c3 in -5.0f32..5.0,
        seed in 0u32..1000,
    ) {
        let (program, root) = constant_subgraph_program(length, c1, c2, c3);
        let mut arena = build_static_arena(&program, &[], &[root]).expect("constant-subgraph program builds a static arena");

        for step in 0..3u32 {
            let x: Vec<f32> = (0..length).map(|index| ((index + seed + step * 97) as f32 * 0.023).sin() * 2.0).collect();

            let hoisted = evaluate_named_with_arena(&mut arena, &[("x", &x)]).expect("arena step evaluates");
            let per_step = evaluate_named(&program, &[], &[("x", &x)], &[root]).expect("fresh per-step evaluation");

            prop_assert_eq!(
                hoisted.root().to_vec(),
                per_step.root().to_vec(),
                "law 6 (constant staging) step {} diverged between the hoisted arena path and a fresh per-step evaluation -- length={} c1={} c2={} c3={}",
                step, length, c1, c2, c3
            );
        }
    }
}

// ---------------------------------------------------------------------
// Law 4 -- same-input widening. PROPOSED, not landed.
//
// `docs/rewrite-algebra.md` §"Law 4" grepped `qkv`/`QKV` case-insensitively
// across `proxima-tensor/src/` -- every hit is about a physically fused
// QKV weight on disk, the opposite direction from this law, and not a
// plan-level rewrite at all. No mechanism widens `k` independent reduces
// into one anywhere in this tree today. This test documents the obligation
// without fabricating coverage for code that does not exist.
// ---------------------------------------------------------------------

#[test]
#[ignore = "PROPOSED, not landed: docs/rewrite-algebra.md law 4 (same-input widening) has no admission test or fusion mechanism anywhere in proxima-tensor -- grepped qkv/QKV, only disk-layout hits, zero plan-level widening. Nothing to equivalence-test yet; this stub names the obligation."]
fn law4_same_input_widening_is_proposed_not_landed() {
    unreachable!("law 4 has no landed instance -- see docs/rewrite-algebra.md, this test is a documentation stub only");
}
