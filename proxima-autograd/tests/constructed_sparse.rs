//! Does the autograd core support CONSTRUCTING a sparse graph directly -- fixed
//! topology known at build time, never a dense graph pruned down after the
//! fact -- and does the resulting program stay differentiable and correct?
//!
//! The construction: a "block layer" partitions a `dim`-wide vector into
//! `block_count` disjoint groups of `block_size` and connects each group to
//! its own private `block_size x block_size` weight, never to any other
//! group -- literally the `a_static_block_sparse_matmul_needs_no_data_dependent_map`
//! precedent (`proxima-tensor/src/cpu.rs:16236-16243`), generalized from its
//! own two hand-written blocks to an arbitrary `block_count` by making
//! "which block" an ordinary iteration axis instead of a second `NodeId`.
//! Because every operand map stays a plain, single-term
//! [`proxima_tensor::map::projection`] per axis (never two terms combined
//! into one flat index -- that would break
//! `proxima_autograd::adjoint`'s `is_pure_projection` requirement for
//! routing a gradient back through an operand map), the WHOLE block-diagonal
//! layer is ONE `Op::Elementwise` plus ONE `Op::Reduce`, regardless of
//! `block_count`. That is the finding this file exists to nail down: the
//! GRAPH does not grow with the topology's sparsity, only the multiply-add
//! COUNT implied by the iteration space does, because `shape.rs:166`'s
//! scatter gate is never in the picture -- nothing here is data-dependent.
#![allow(clippy::unwrap_used, clippy::expect_used)]

extern crate alloc;

use proxima_autograd::adjoint::differentiate;
use proxima_tensor::cpu::evaluate_named;
use proxima_tensor::dtype::DType;
use proxima_tensor::error::TensorError;
use proxima_tensor::map::{self, IndexMap};
use proxima_tensor::op::{self, Extent, NodeId, Op, ReduceInit, ScalarOp};
use proxima_tensor::shape;

fn leaf(program: &mut Vec<Op>, name: &str, shape: alloc::vec::Vec<Extent>) -> NodeId {
    op::append(program, Op::Input { dtype: DType::Float32, shape, name: Some(name.into()) })
}

fn elementwise(program: &mut Vec<Op>, body: ScalarOp, operands: alloc::vec::Vec<(NodeId, IndexMap)>) -> NodeId {
    op::append(program, Op::Elementwise { dtype: DType::Float32, body, operands, name: None })
}

fn reduce_add(program: &mut Vec<Op>, operand: NodeId, in_map: IndexMap, out_map: IndexMap) -> NodeId {
    op::append(
        program,
        Op::Reduce(proxima_tensor::op::Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand,
            in_map,
            out_map,
            keep: proxima_tensor::op::Keep::Reduce,
            name: None,
        }),
    )
}

fn proj(iter_rank: u16, axes: &[u16]) -> IndexMap {
    IndexMap::Affine(map::projection(iter_rank, axes))
}

fn identity(rank: u16) -> IndexMap {
    proj(rank, &(0..rank).collect::<alloc::vec::Vec<u16>>())
}

fn empty(rank: u16) -> IndexMap {
    proj(rank, &[])
}

/// Sums the iteration-space size of every `Reduce(Add)` node in `program`,
/// starting the scan at `start` (so a caller can measure only the ops a
/// transform APPENDED, excluding the shared prefix it left untouched) --
/// the multiply-add count the preceding `Elementwise(Multiply)` plus this
/// `Reduce` jointly spend. Every `Reduce` this file builds (forward AND
/// adjoint -- `proxima_autograd::adjoint`'s own `route_contribution` always
/// passes `expr::identity(iter_rank)` as `in_map`, exactly this file's own
/// `reduce_add` convention) uses an identity `in_map`, so the reduce
/// operand's own inferred shape already IS the iteration space in axis
/// order -- no separate iteration-space reconstruction needed, just read
/// `shapes.of(reduce.operand)` and multiply its extents. `shape::infer`
/// always runs over the FULL `program` (never a slice) because `Op`
/// operands are absolute `NodeId` indices into it; `start` only filters
/// which ops get summed afterward.
fn total_macs_from(program: &[Op], start: usize) -> Result<u64, TensorError> {
    let shapes = shape::infer(program, &[])?;
    let mut macs = 0u64;
    for op in program.iter().skip(start) {
        if let Op::Reduce(reduce) = op
            && matches!(reduce.body, ScalarOp::Add)
        {
            macs += shapes.of(reduce.operand).iter().product::<u64>();
        }
    }
    Ok(macs)
}

fn total_macs(program: &[Op]) -> Result<u64, TensorError> {
    total_macs_from(program, 0)
}

struct BlockLayer {
    program: alloc::vec::Vec<Op>,
    output: NodeId,
}

/// `output[b, block, o] = sum_i x[b, block, i] * w[block, o, i]` -- iteration
/// space `(b, block, o, i)`, `i` reduced. `block` is an ordinary iteration
/// axis shared by every operand's own map, exactly the way `s` (sequence
/// position) already is in `proxima-autograd/tests/language_model.rs`'s
/// attention block; nothing about the transform needs to know "block" means
/// "an independent connected component" rather than "a batch position" --
/// that is the whole point. `x`'s natural shape is `[batch, block_count,
/// block_size]` (the `dim = block_count * block_size` vector, reshaped, not
/// flattened via a two-term affine axis -- a two-term axis would fail
/// `proxima_autograd::adjoint::expr::is_pure_projection` and this program
/// would not differentiate). `w`'s shape is `[block_count, block_size,
/// block_size]`: one private dense matrix per block, batched in axis 0.
fn build_block_layer(batch: usize, block_count: usize, block_size: usize) -> BlockLayer {
    let mut program = alloc::vec::Vec::new();
    let x = leaf(
        &mut program,
        "x",
        alloc::vec![Extent::Static(batch as u32), Extent::Static(block_count as u32), Extent::Static(block_size as u32)],
    );
    let w = leaf(
        &mut program,
        "w",
        alloc::vec![
            Extent::Static(block_count as u32),
            Extent::Static(block_size as u32),
            Extent::Static(block_size as u32)
        ],
    );

    // iter (b, block, o, i): x reads (b, block, i) -> axes (0,1,3);
    // w reads (block, o, i), broadcasting over b -> axes (1,2,3).
    let product = elementwise(&mut program, ScalarOp::Multiply, alloc::vec![(x, proj(4, &[0, 1, 3])), (w, proj(4, &[1, 2, 3]))]);
    let output = reduce_add(&mut program, product, identity(4), proj(4, &[0, 1, 2]));

    BlockLayer { program, output }
}

fn counter_pattern(seed: usize, count: usize) -> alloc::vec::Vec<f32> {
    (0..count).map(|index| (((seed + index) * 7 % 13) as f32 - 6.0) / 24.0).collect()
}

/// Node count for [`build_block_layer`] never depends on `block_count`: two
/// leaves, one `Elementwise`, one `Reduce` -- four ops whether `block_count`
/// is 1 (one big dense block, `dim = block_size`) or 100 (`block_size = 1`,
/// a degenerate per-element scale). The multiply-add count DOES depend on
/// it, and does so linearly in `block_count` for fixed `block_size` --
/// `dim * block_size`, the exact parameter count of `w` -- while a dense
/// `dim x dim` equivalent costs `dim * dim`. This is the whole claim: cost
/// tracks the nonzero count the topology chooses, not the dense shape it
/// happens to be embeddable in.
#[proxima::test]
async fn program_size_is_constant_while_macs_track_nonzeros_not_dense_shape() {
    const BATCH: usize = 4;
    const DIM: usize = 100;
    let sweep: alloc::vec::Vec<(usize, usize)> = alloc::vec![(1, 100), (2, 50), (10, 10), (100, 1)];

    let mut node_counts: alloc::vec::Vec<usize> = alloc::vec::Vec::new();
    for &(block_count, block_size) in &sweep {
        assert_eq!(block_count * block_size, DIM, "sweep case must actually partition the same dim");
        let layer = build_block_layer(BATCH, block_count, block_size);
        node_counts.push(layer.program.len());

        let macs = total_macs(&layer.program).expect("static-shaped block layer infers");
        let expected_macs = (BATCH * block_count * block_size * block_size) as u64;
        let dense_macs = (BATCH * DIM * DIM) as u64;
        std::eprintln!(
            "block_count={block_count} block_size={block_size}: program.len()={} macs={macs} \
             (expected {expected_macs}), dense-equivalent macs={dense_macs}, ratio={:.4}",
            layer.program.len(),
            macs as f64 / dense_macs as f64
        );
        assert_eq!(macs, expected_macs, "mac count must equal batch * block_count * block_size^2");
    }

    assert_eq!(
        node_counts.iter().collect::<alloc::collections::BTreeSet<_>>().len(),
        1,
        "program.len() must be the SAME constant across every block_count in the sweep: {node_counts:?}"
    );
    std::eprintln!(
        "program.len() is constant at {} across block_count in {:?}",
        node_counts[0],
        sweep.iter().map(|(c, _)| *c).collect::<alloc::vec::Vec<_>>()
    );
}

/// Proves the claim above can actually fail: force a wrong expected-mac
/// count and confirm it does NOT match the measured count, so the sweep
/// above is discriminating rather than vacuously true regardless of what
/// `build_block_layer` computes.
#[proxima::test]
async fn program_size_claim_is_falsifiable_a_wrong_mac_count_is_caught() {
    let layer = build_block_layer(4, 10, 10);
    let macs = total_macs(&layer.program).expect("static-shaped block layer infers");
    let genuinely_wrong_expectation = (4 * 10 * 10 * 10 * 2) as u64; // double the real cost
    assert_ne!(
        macs, genuinely_wrong_expectation,
        "a deliberately wrong expected-mac-count must NOT match the measured count \
         (proves total_macs is actually discriminating, not vacuously true)"
    );
}

/// The constructed sparse layer against a zero-padded DENSE `[dim, dim]`
/// reference carrying the exact same per-block values off-diagonal-zeroed --
/// same operation count concern aside, this is the faithfulness check: a
/// sparse CONSTRUCTION is only meaningful if it computes the same answer a
/// dense equivalent would, not merely a cheaper one.
#[proxima::test]
async fn constructed_sparse_output_matches_a_zero_padded_dense_reference() {
    const BATCH: usize = 3;
    const BLOCK_COUNT: usize = 10;
    const BLOCK_SIZE: usize = 10;
    const DIM: usize = BLOCK_COUNT * BLOCK_SIZE;

    let layer = build_block_layer(BATCH, BLOCK_COUNT, BLOCK_SIZE);
    let x_values = counter_pattern(3, BATCH * BLOCK_COUNT * BLOCK_SIZE);
    let w_values = counter_pattern(11, BLOCK_COUNT * BLOCK_SIZE * BLOCK_SIZE);

    let sparse_evaluated = evaluate_named(&layer.program, &[], &[("x", &x_values), ("w", &w_values)], &[layer.output])
        .expect("block layer lowers and evaluates");
    let sparse_output = sparse_evaluated.get(layer.output).expect("output requested").0.to_vec();

    let mut dense_program = alloc::vec::Vec::new();
    let dense_x = leaf(&mut dense_program, "x_flat", alloc::vec![Extent::Static(BATCH as u32), Extent::Static(DIM as u32)]);
    let dense_w = leaf(&mut dense_program, "w_dense", alloc::vec![Extent::Static(DIM as u32), Extent::Static(DIM as u32)]);
    let dense_product = elementwise(
        &mut dense_program,
        ScalarOp::Multiply,
        alloc::vec![(dense_x, proj(3, &[0, 2])), (dense_w, proj(3, &[1, 2]))],
    );
    let dense_output = reduce_add(&mut dense_program, dense_product, identity(3), proj(3, &[0, 1]));

    let mut w_dense_values = alloc::vec![0.0f32; DIM * DIM];
    for block in 0..BLOCK_COUNT {
        for out_local in 0..BLOCK_SIZE {
            for in_local in 0..BLOCK_SIZE {
                let block_value = w_values[block * BLOCK_SIZE * BLOCK_SIZE + out_local * BLOCK_SIZE + in_local];
                let row = block * BLOCK_SIZE + out_local;
                let column = block * BLOCK_SIZE + in_local;
                w_dense_values[row * DIM + column] = block_value;
            }
        }
    }

    let dense_evaluated = evaluate_named(&dense_program, &[], &[("x_flat", &x_values), ("w_dense", &w_dense_values)], &[dense_output])
        .expect("dense reference lowers and evaluates");
    let dense_output_values = dense_evaluated.get(dense_output).expect("output requested").0;

    assert_eq!(sparse_output.len(), dense_output_values.len());
    let max_diff = sparse_output
        .iter()
        .zip(dense_output_values.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    std::eprintln!("constructed-sparse vs zero-padded-dense max abs diff = {max_diff}");
    assert!(max_diff < 1e-4, "constructed sparse layer must match its dense zero-padded equivalent, max diff {max_diff}");
}

/// Same comparison, deliberately broken: place every block TRANSPOSED
/// (row/col swapped) in the dense reference so the two programs compute
/// genuinely different answers -- proves the equivalence check above is
/// not vacuously passing regardless of the dense reference's construction.
#[proxima::test]
async fn dense_reference_mismatch_is_actually_caught() {
    const BATCH: usize = 2;
    const BLOCK_COUNT: usize = 4;
    const BLOCK_SIZE: usize = 5;
    const DIM: usize = BLOCK_COUNT * BLOCK_SIZE;

    let layer = build_block_layer(BATCH, BLOCK_COUNT, BLOCK_SIZE);
    let x_values = counter_pattern(5, BATCH * BLOCK_COUNT * BLOCK_SIZE);
    let w_values = counter_pattern(17, BLOCK_COUNT * BLOCK_SIZE * BLOCK_SIZE);

    let sparse_evaluated = evaluate_named(&layer.program, &[], &[("x", &x_values), ("w", &w_values)], &[layer.output])
        .expect("block layer lowers and evaluates");
    let sparse_output = sparse_evaluated.get(layer.output).expect("output requested").0.to_vec();

    let mut dense_program = alloc::vec::Vec::new();
    let dense_x = leaf(&mut dense_program, "x_flat", alloc::vec![Extent::Static(BATCH as u32), Extent::Static(DIM as u32)]);
    let dense_w = leaf(&mut dense_program, "w_dense", alloc::vec![Extent::Static(DIM as u32), Extent::Static(DIM as u32)]);
    let dense_product = elementwise(
        &mut dense_program,
        ScalarOp::Multiply,
        alloc::vec![(dense_x, proj(3, &[0, 2])), (dense_w, proj(3, &[1, 2]))],
    );
    let dense_output = reduce_add(&mut dense_program, dense_product, identity(3), proj(3, &[0, 1]));

    // deliberately WRONG: place every block transposed (row/col swapped).
    let mut w_dense_values = alloc::vec![0.0f32; DIM * DIM];
    for block in 0..BLOCK_COUNT {
        for out_local in 0..BLOCK_SIZE {
            for in_local in 0..BLOCK_SIZE {
                let block_value = w_values[block * BLOCK_SIZE * BLOCK_SIZE + out_local * BLOCK_SIZE + in_local];
                let row = block * BLOCK_SIZE + in_local; // swapped on purpose
                let column = block * BLOCK_SIZE + out_local; // swapped on purpose
                w_dense_values[row * DIM + column] = block_value;
            }
        }
    }

    let dense_evaluated = evaluate_named(&dense_program, &[], &[("x_flat", &x_values), ("w_dense", &w_dense_values)], &[dense_output])
        .expect("dense reference lowers and evaluates");
    let dense_output_values = dense_evaluated.get(dense_output).expect("output requested").0;

    let max_diff = sparse_output
        .iter()
        .zip(dense_output_values.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    std::eprintln!("deliberately-transposed dense reference max abs diff = {max_diff}");
    assert!(max_diff > 1e-3, "a transposed dense reference must NOT match (asymmetric block weights make this observable): diff {max_diff}");
}

const GRADIENT_CHECK_ATOL: f32 = 1e-2;
const GRADIENT_CHECK_RTOL: f32 = 1e-2;

fn within_tolerance(analytic: f32, numeric: f32) -> bool {
    (analytic - numeric).abs() <= GRADIENT_CHECK_ATOL + GRADIENT_CHECK_RTOL * numeric.abs()
}

fn checked_indices(len: usize) -> impl Iterator<Item = usize> {
    let stride = (len / 10).max(1);
    (0..len).step_by(stride)
}

/// Differentiates [`build_block_layer`]'s output (summed to a scalar loss)
/// and checks two things: the adjoint appends a BOUNDED number of extra
/// multiply-adds, never one that scales toward the DENSE `dim^2` a bug
/// reintroducing a dense-shaped backward pass would produce. Reading the
/// measured ratio (exactly 2.0x forward here) traces to two full-cost
/// routing reduces, not three: `differentiate` walks every node from the
/// loss down, and the forward layer's own `Reduce` (routing gradient from
/// `output` back to the `Multiply` node) has an identity `in_map` -- ROW
/// 163 (`proxima-autograd/src/adjoint.rs`'s `differentiate_reduce`) skips
/// materializing an un-reduce wrapper whenever `reduce.in_map == full`,
/// since that case can only ever be a one-to-one copy of `contribution`,
/// never an accumulation, so the routed gradient becomes the `Elementwise`
/// contribution directly with zero extra reduce cost. What remains
/// full-cost is the `Multiply`'s own two operand routings (`x` and `w`)
/// plus one tiny pass-through for the scalar loss's own reduce (`batch *
/// block_count * block_size`, a few percent of the total, still counted
/// in `adjoint_appended_macs` but negligible against the 2x floor).
/// Central difference over the real scalar loss then confirms the adjoint
/// is not just bounded but CORRECT, under PyTorch's combined `atol + rtol
/// * |numeric|` criterion (`language_model.rs`'s own documented reason a
/// bare relative error is not enough for near-zero gradients) --
/// re-verified directly against this eliminated-reduce shape by
/// temporarily running the gradient check ahead of the mac-floor
/// assertion (bea561b's adjudication): gradients agreed with central
/// difference at both `(block_count, block_size)` cases, so the floor
/// below tracks the new structural truth, not a regression.
#[proxima::test]
async fn adjoint_of_a_constructed_sparse_layer_stays_small_and_gradient_checks() {
    for &(block_count, block_size) in &[(2usize, 25usize), (10usize, 10usize)] {
        let batch = 3usize;
        let layer = build_block_layer(batch, block_count, block_size);
        let forward_macs = total_macs(&layer.program).expect("forward infers");

        let mut program = layer.program.clone();
        let loss = reduce_add(&mut program, layer.output, identity(3), empty(3));

        let differentiated = differentiate(&program, loss).expect("scalar loss differentiates");
        let grad_x = differentiated.gradient_of_named("x").expect("x feeds the loss");
        let grad_w = differentiated.gradient_of_named("w").expect("w feeds the loss");

        let adjoint_appended_macs =
            total_macs_from(&differentiated.program, program.len()).expect("adjoint-appended slice infers");
        std::eprintln!(
            "block_count={block_count} block_size={block_size}: forward program.len()={} \
             adjoint program.len()={} forward_macs={forward_macs} adjoint_appended_macs={adjoint_appended_macs} \
             (appended/forward ratio {:.3})",
            layer.program.len(),
            differentiated.program.len(),
            adjoint_appended_macs as f64 / forward_macs as f64
        );
        assert!(
            adjoint_appended_macs >= forward_macs * 2,
            "adjoint must route two full-cost reduces (product->x, product->w); output->product is a pure identity-mapped \
             copy that ROW 163 routes without a materializing reduce: {adjoint_appended_macs} < {}",
            forward_macs * 2
        );
        assert!(
            adjoint_appended_macs < forward_macs * 3,
            "adjoint cost must stay a small bounded multiple of the forward cost, not scale toward the dense shape: \
             {adjoint_appended_macs} >= {}",
            forward_macs * 3
        );

        let x_values = counter_pattern(19, batch * block_count * block_size);
        let w_values = counter_pattern(23, block_count * block_size * block_size);

        let evaluated = evaluate_named(&differentiated.program, &[], &[("x", &x_values), ("w", &w_values)], &[grad_x, grad_w])
            .expect("adjoint program lowers and evaluates");
        let analytic_x = evaluated.get(grad_x).expect("requested").0.to_vec();
        let analytic_w = evaluated.get(grad_w).expect("requested").0.to_vec();

        let step = 1e-3f32;
        let loss_at = |x: &[f32], w: &[f32]| -> f32 {
            evaluate_named(&program, &[], &[("x", x), ("w", w)], &[loss])
                .expect("forward program lowers and evaluates")
                .get(loss)
                .expect("loss requested")
                .0[0]
        };

        let mut worst_violation: Option<(&str, usize, f32, f32)> = None;
        for (label, values, analytic, other_fixed) in [
            ("x", x_values.clone(), &analytic_x, w_values.clone()),
            ("w", w_values.clone(), &analytic_w, x_values.clone()),
        ] {
            let mut perturbed = values;
            for index in checked_indices(perturbed.len()) {
                let original = perturbed[index];
                perturbed[index] = original + step;
                let plus = if label == "x" { loss_at(&perturbed, &other_fixed) } else { loss_at(&other_fixed, &perturbed) };
                perturbed[index] = original - step;
                let minus = if label == "x" { loss_at(&perturbed, &other_fixed) } else { loss_at(&other_fixed, &perturbed) };
                perturbed[index] = original;

                let numeric = (plus - minus) / (2.0 * step);
                if !within_tolerance(analytic[index], numeric) {
                    worst_violation = Some((label, index, analytic[index], numeric));
                }
            }
        }
        assert!(
            worst_violation.is_none(),
            "block_count={block_count}: analytic gradient disagreed with central difference beyond tolerance: {worst_violation:?}"
        );
    }
}

/// Proves the gradient check above can actually fail: swap in an
/// intentionally WRONG "analytic" gradient (the reversed array) and confirm
/// the combined-tolerance comparison rejects it, rather than the check
/// passing regardless of what the adjoint produced.
#[proxima::test]
async fn gradient_check_tolerance_rejects_a_deliberately_wrong_gradient() {
    let batch = 3usize;
    let (block_count, block_size) = (2usize, 5usize);
    let layer = build_block_layer(batch, block_count, block_size);
    let mut program = layer.program.clone();
    let loss = reduce_add(&mut program, layer.output, identity(3), empty(3));
    let differentiated = differentiate(&program, loss).expect("scalar loss differentiates");
    let grad_x = differentiated.gradient_of_named("x").expect("x feeds the loss");

    let x_values = counter_pattern(7, batch * block_count * block_size);
    let w_values = counter_pattern(13, block_count * block_size * block_size);
    let evaluated = evaluate_named(&differentiated.program, &[], &[("x", &x_values), ("w", &w_values)], &[grad_x])
        .expect("adjoint program lowers and evaluates");
    let analytic_x = evaluated.get(grad_x).expect("requested").0.to_vec();

    let step = 1e-3f32;
    let loss_at = |x: &[f32]| -> f32 {
        evaluate_named(&program, &[], &[("x", x), ("w", &w_values)], &[loss])
            .expect("forward program lowers and evaluates")
            .get(loss)
            .expect("loss requested")
            .0[0]
    };
    let mut perturbed = x_values.clone();
    let index = 0usize;
    let original = perturbed[index];
    perturbed[index] = original + step;
    let plus = loss_at(&perturbed);
    perturbed[index] = original - step;
    let minus = loss_at(&perturbed);
    let numeric = (plus - minus) / (2.0 * step);

    let deliberately_wrong_analytic = analytic_x[index] + 10.0; // no legitimate gradient is off by 10.0 here
    assert!(
        !within_tolerance(deliberately_wrong_analytic, numeric),
        "a gradient wrong by 10.0 must fail the combined atol+rtol check (numeric={numeric}, wrong analytic={deliberately_wrong_analytic})"
    );
    assert!(
        within_tolerance(analytic_x[index], numeric),
        "the REAL analytic gradient must still pass, proving the check discriminates rather than always rejecting"
    );
}

/// Where the free lunch stops: a topology built from TWO differently-shaped
/// block groups (8 blocks of 10, then 4 blocks of 5 -- still fixed at build
/// time, still zero data-dependence) needs one `Elementwise` + `Reduce`
/// PAIR per distinct shape, not one total. `program.len()` grows with the
/// number of distinct block shapes, not with `block_count` inside a
/// uniform group -- the uniform sweep above stays flat because every block
/// in it shares one shape and one iteration axis; the moment shapes differ,
/// that axis can no longer be shared, and this is the mechanical reason
/// (not an assertion) that irregular, magnitude-derived topology (this
/// session's FFN band-pruning arm) cannot get the same O(1) node count this
/// file's uniform construction gets for free.
#[proxima::test]
async fn mixed_block_shapes_cost_one_op_pair_per_distinct_shape() {
    let uniform = build_block_layer(2, 12, 5); // one shape group: program.len() == 4
    let group_a = build_block_layer(2, 8, 10);
    let group_b = build_block_layer(2, 4, 5);

    let uniform_macs = total_macs(&uniform.program).expect("uniform infers");
    let group_a_macs = total_macs(&group_a.program).expect("group a infers");
    let group_b_macs = total_macs(&group_b.program).expect("group b infers");

    std::eprintln!("uniform (1 shape group, block_count=12): program.len()={} macs={uniform_macs}", uniform.program.len());
    std::eprintln!(
        "mixed (2 shape groups, 8x10 + 4x5): combined program.len()={} combined macs={}",
        group_a.program.len() + group_b.program.len(),
        group_a_macs + group_b_macs
    );

    assert_eq!(uniform.program.len(), 4, "one shape group is exactly Input(x) + Input(w) + Elementwise + Reduce");
    assert_eq!(
        group_a.program.len() + group_b.program.len(),
        8,
        "two distinct block shapes cost exactly two independent 4-op programs -- node count does NOT stay O(1) here"
    );
    assert_eq!(
        group_a_macs + group_b_macs,
        (2 * 8 * 10 * 10 + 2 * 4 * 5 * 5) as u64,
        "macs still track nonzeros, just split across two ops now"
    );
}
