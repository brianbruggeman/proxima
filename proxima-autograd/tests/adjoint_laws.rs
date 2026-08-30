//! [`differentiate`] specified as a homomorphism, not just an output-checker.
//!
//! Conal Elliott's *The Simple Essence of Automatic Differentiation*
//! (arXiv:1804.00746) frames a correct AD implementation as one that
//! commutes with the vocabulary's own operations: differentiating a
//! composite must equal composing the differentiated pieces, and the
//! transform must be linear in the seed it is handed. Central-difference
//! gradient checks (`training_loop.rs`'s
//! `central_difference_matches_the_analytic_gradient_on_every_parameter`)
//! only ever probe ONE point of ONE whole program; they cannot discriminate
//! a rule that is wrong in a way the check's own inputs happen not to
//! exercise. Two of this session's three real defects are exactly that
//! shape:
//!
//! - `Reduce(Maximum)`'s adjoint dropped its argmax mask. The whole-model
//!   central-difference check STAYED GREEN, because softmax is exactly
//!   shift-invariant: the true gradient through a max-subtraction feeding a
//!   softmax is mathematically zero, so a check built on a softmax-shaped
//!   program cannot see this rule is broken no matter how many points it
//!   samples. `reduce_max_and_min_route_only_to_the_argmax_or_argmin_mask`
//!   below is a closed-form check with no softmax anywhere near it.
//! - `differentiate` indexed a `grad_of` sized `loss_index + 1` against
//!   every `Op::Input` in the FULL input program, panicking the moment a
//!   caller's program grew past the loss node. No single-program gradient
//!   check can even construct this bug's trigger, because the bug is about
//!   what `differentiate` does to TWO overlapping sub-programs of one
//!   larger slice, not about any one program's numbers.
//!   `differentiating_a_program_at_an_earlier_node_composes_with_the_rest_of_the_graph`
//!   below is exactly that shape.
//!
//! Four laws, each derived below before it is coded, each demonstrated to
//! discriminate its own rule by literally breaking that rule in
//! `src/adjoint.rs`, running the suite, and reverting (see this crate's own
//! report for both sets of pasted `cargo nextest` output).
//!
//! ## Law 1 — composition (`differentiating_a_program_at_an_earlier_node_composes_with_the_rest_of_the_graph`)
//!
//! For a forward computation `x -> u -> loss` split at `u` into `f: x -> u`
//! and `g: u -> loss`, reverse-mode AD's defining property is
//! `adjoint(g ; f) = adjoint(f) ; adjoint(g)` (adjoints compose in reverse
//! order) — so `dL/dx = (dL/du) * (du/dx)`, where `dL/du` is `g`'s OWN
//! adjoint (computed by differentiating `g` in isolation, treating `u` as
//! `g`'s leaf) and `du/dx` is `f`'s OWN adjoint (computed by differentiating
//! `f` in isolation, treating `u` as `f`'s loss). Both `f` and `g` here are
//! literal, independent calls to [`differentiate`] — "composing the pieces'
//! gradients by hand" means the TEST does the multiplication, not the
//! system; the system only ever runs each piece once.
//!
//! A finding worth recording plainly: a per-`ScalarOp` math bug (e.g.
//! "forgot to multiply by the incoming gradient") is INVISIBLE to this law
//! whenever the same buggy rule fires identically inside both the isolated
//! piece and the composite (see `law_2` below for why that is NOT true of
//! every bug). What this law uniquely catches is a bug in how
//! [`differentiate`] treats a program that keeps growing past the node
//! being differentiated — precisely defect 3's shape, reproduced verbatim
//! below as the deliberate break.
//!
//! ## Law 2 — linearity in the seed (`the_adjoint_is_linear_in_its_seed`)
//!
//! `differentiate` always seeds the loss's own cotangent at exactly `1.0`
//! (`adjoint.rs:201`), so seeding IS the value bound to whatever `Input`
//! feeds the loss multiplicatively. For a fixed program computing
//! `loss = sum(f(x) * weight)`, `dL/dx` is `weight * f'(x)` pointwise — LINEAR
//! in `weight`. So evaluating the SAME differentiated program three times,
//! swapping only the bound `weight` array between `a`, `b`, and `a + b`,
//! must satisfy `grad_x(a + b) = grad_x(a) + grad_x(b)` exactly (up to f32
//! rounding). Unlike law 1, this DOES catch "forgot to multiply by the
//! incoming gradient": dropping that factor disconnects `weight` from
//! `grad_x`'s expression graph entirely, making `grad_x` numerically
//! constant across all three evaluations — `2c != c` for any `c != 0`.
//!
//! ## Law 3 — per-`ScalarOp` local derivative
//!
//! Every unary/binary `ScalarOp` `differentiate_elementwise` handles emits
//! `weight * op'(x)` when wrapped as `loss = sum(op(x) * weight)` (chain
//! rule again, this time recovering the LOCAL partial by dividing the
//! system's `grad_x` by the independently-bound `weight`). Enumerated from
//! `proxima-tensor/src/op.rs:60-78`'s 17-variant `ScalarOp` enum against
//! `adjoint.rs:293-380`'s match: every variant has an arm — `Identity`,
//! `Negate`, `Add`, `Subtract`, `Multiply`, `Divide`, `Maximum`, `Minimum`,
//! `Reciprocal`, `Exponential`, `Logarithm`, `SquareRoot`, `Tanh`, `Erf`,
//! `Select` each get a real local-partial rule; `Greater`/`Equal` get the
//! (correct) "no gradient flows" rule, asserted explicitly below rather
//! than assumed. No variant is silently unhandled — the match has no
//! wildcard arm, so the compiler itself enforces this; this file's cases
//! are the runtime witness of that exhaustiveness.
//!
//! ## Law 4 — `Reduce` adjoints per body
//!
//! `Reduce(Add)`'s adjoint broadcasts the incoming gradient identically to
//! every element that fed the sum (`adjoint.rs:507-513`).
//! `Reduce(Maximum)`/`Reduce(Minimum)`'s adjoint instead routes the FULL
//! incoming gradient only to elements equal to the reduce's own output —
//! the argmax/argmin mask (`adjoint.rs:514-537`). Ties: the mask is built
//! from `ScalarOp::Equal` against the broadcast output, so EVERY element
//! tied for the max/min gets the mask value `1.0`, not a 1/N split — this
//! crate's own module doc (`adjoint.rs:31-34`) names this convention
//! explicitly (it matches TensorFlow's `reduce_max` gradient; PyTorch's
//! kernel instead routes to one argmax only). The tie case below asserts
//! this precisely rather than assuming it.
#![allow(clippy::unwrap_used, clippy::expect_used)]

extern crate alloc;

use alloc::format;
use alloc::vec;
use alloc::vec::Vec;

use proxima_autograd::adjoint::differentiate;
use proxima_tensor::cpu::evaluate_named;
use proxima_tensor::dtype::DType;
use proxima_tensor::map::{self, IndexMap};
use proxima_tensor::op::{self, Extent, NodeId, Op, Reduce, ReduceInit, ScalarOp};

fn leaf(program: &mut Vec<Op>, name: &str, extent: usize) -> NodeId {
    op::append(
        program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(extent as u32)],
            name: Some(name.into()),
        },
    )
}

fn leaf_shaped(program: &mut Vec<Op>, name: &str, shape: Vec<Extent>) -> NodeId {
    op::append(program, Op::Input { dtype: DType::Float32, shape, name: Some(name.into()) })
}

fn elementwise(program: &mut Vec<Op>, body: ScalarOp, operands: Vec<(NodeId, IndexMap)>) -> NodeId {
    op::append(program, Op::Elementwise { dtype: DType::Float32, body, operands, name: None })
}

fn ident_map(rank: u16) -> IndexMap {
    IndexMap::Affine(map::projection(rank, &(0..rank).collect::<Vec<u16>>()))
}

fn broadcast(rank: u16) -> IndexMap {
    IndexMap::Affine(map::projection(rank, &[]))
}

#[allow(clippy::too_many_arguments)]
fn reduce_node(
    program: &mut Vec<Op>,
    body: ScalarOp,
    init: ReduceInit,
    operand: NodeId,
    in_map: IndexMap,
    out_map: IndexMap,
) -> NodeId {
    op::append(
        program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body,
            init,
            operand,
            in_map,
            out_map,
            keep: proxima_tensor::op::Keep::Reduce,
            name: None,
        }),
    )
}

/// `sum(operand)` over a rank-`rank` operand -- the scalar-loss adapter
/// every law below uses to turn an elementwise or partial-reduce shape into
/// something [`differentiate`] accepts (`adjoint.rs:194-196` rejects a
/// non-scalar loss).
fn reduce_scalar_add(program: &mut Vec<Op>, operand: NodeId, rank: u16) -> NodeId {
    reduce_node(program, ScalarOp::Add, ReduceInit::Zero, operand, ident_map(rank), broadcast(rank))
}

fn get(evaluated: &proxima_tensor::cpu::Evaluated, node: NodeId) -> Vec<f32> {
    evaluated.get(node).expect("requested node evaluated").0.to_vec()
}

/// The tolerance this file uses whenever the "known" side of a comparison
/// is itself a numeric evaluation (an independently computed closed-form
/// derivative, or a finite difference) rather than a second symbolic
/// expression -- the combined criterion this crate's own conventions call
/// for. Algebraic identities (composition, linearity) use a much tighter
/// fixed tolerance instead, asserted inline at their call sites.
fn combined_tolerance(numeric: f32) -> f32 {
    1e-2 + 1e-2 * numeric.abs()
}

fn assert_close(analytic: f32, numeric: f32, tolerance: f32, context: alloc::string::String) {
    let diff = (analytic - numeric).abs();
    assert!(diff <= tolerance, "{context}: analytic={analytic} numeric={numeric} diff={diff} tolerance={tolerance}");
}

// ---------------------------------------------------------------------
// Law 1: composition
// ---------------------------------------------------------------------

/// Builds `f: x -> u = x * x` as its OWN standalone program with `u`'s sum
/// as the loss, so differentiating it yields `du/dx = 2x` (diagonal: `u`
/// is elementwise in `x`, so `d(sum u)/dx_j = du_j/dx_j`).
fn build_f_program(x_values: &[f32]) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let x = leaf(&mut program, "x", x_values.len());
    let u = elementwise(&mut program, ScalarOp::Multiply, vec![(x, ident_map(1)), (x, ident_map(1))]);
    let loss = reduce_scalar_add(&mut program, u, 1);
    (program, loss)
}

/// Builds `g: u -> loss = sum(exp(u) * weight)` as its OWN standalone
/// program with `u` as a leaf, so differentiating it yields
/// `dL/du = weight * exp(u)` -- a genuinely non-constant seed (not the
/// trivial all-ones case a direct loss would give `exp` alone), which is
/// exactly what makes law 2 (not this one) sensitive to a "dropped
/// gradient factor" bug in `Exponential`'s rule.
fn build_g_program(weight_values: &[f32]) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let u = leaf(&mut program, "u", weight_values.len());
    let weight = leaf(&mut program, "weight", weight_values.len());
    let v = elementwise(&mut program, ScalarOp::Exponential, vec![(u, ident_map(1))]);
    let w = elementwise(&mut program, ScalarOp::Multiply, vec![(v, ident_map(1)), (weight, ident_map(1))]);
    let loss = reduce_scalar_add(&mut program, w, 1);
    (program, loss)
}

/// `full_program`'s OWN loss1 sits at index 2, with an UNRELATED second
/// `Input` (`y`) appended after it (index 3) before the rest of the
/// composite computation continues to loss2 -- exactly the "a program that
/// keeps growing past the node being differentiated" shape defect 3 needed
/// to trigger (`proxima-autograd/tests/actor_critic.rs`'s two-loss-node
/// policy/value split is the real-world instance; this is the minimal
/// one).
fn build_full_program(x_values: &[f32], y_values: &[f32], weight_values: &[f32]) -> (Vec<Op>, NodeId, NodeId) {
    let mut program = Vec::new();
    let x = leaf(&mut program, "x", x_values.len()); // 0
    let u = elementwise(&mut program, ScalarOp::Multiply, vec![(x, ident_map(1)), (x, ident_map(1))]); // 1
    let loss1 = reduce_scalar_add(&mut program, u, 1); // 2
    let _y = leaf(&mut program, "y", y_values.len()); // 3 -- past loss1's index
    let weight = leaf(&mut program, "weight", weight_values.len()); // 4
    let v = elementwise(&mut program, ScalarOp::Exponential, vec![(u, ident_map(1))]); // 5
    let w = elementwise(&mut program, ScalarOp::Multiply, vec![(v, ident_map(1)), (weight, ident_map(1))]); // 6
    let loss2 = reduce_scalar_add(&mut program, w, 1); // 7
    (program, loss1, loss2)
}

/// **Law 1.** `dL/dx` computed end to end (`loss2` on `full_program`) must
/// equal `dL/du` (from `g` in isolation) times `du/dx` (from `f`, computed
/// AT `loss1` embedded inside the very same `full_program` that keeps
/// growing past it) -- the literal chain rule, with both factors coming
/// from real, independent [`differentiate`] calls, composed by this test,
/// not by the system.
///
/// Broken by reverting `differentiate`'s gradient scan to the full
/// `program` slice instead of `program[..=loss_index]` (defect 3's exact
/// shape, `adjoint.rs:241`): `differentiate(&full_program, loss1)` then
/// scans the FULL 8-node program for `Op::Input`, finds `y` at index 3 and
/// `weight` at index 4, and indexes a `grad_of` sized `loss_index + 1 == 3`
/// at those positions -- an out-of-bounds panic, not a wrong number.
#[proxima::test]
async fn differentiating_a_program_at_an_earlier_node_composes_with_the_rest_of_the_graph() {
    let x_values = [0.6_f32, -1.4, 2.1, 0.3];
    let y_values = [9.0_f32, -4.0];
    let weight_values = [0.5_f32, -2.0, 1.25, 3.0];
    let u_values: Vec<f32> = x_values.iter().map(|value| value * value).collect();

    let (f_program, f_loss) = build_f_program(&x_values);
    let f_differentiated = differentiate(&f_program, f_loss).expect("f differentiates in isolation");
    let grad_x_f_node = f_differentiated.gradient_of_named("x").expect("x feeds f's loss");
    let grad_x_f = get(
        &evaluate_named(&f_differentiated.program, &[], &[("x", &x_values)], &[grad_x_f_node])
            .expect("f's adjoint program evaluates"),
        grad_x_f_node,
    );

    let (g_program, g_loss) = build_g_program(&weight_values);
    let g_differentiated = differentiate(&g_program, g_loss).expect("g differentiates in isolation");
    let grad_u_g_node = g_differentiated.gradient_of_named("u").expect("u feeds g's loss");
    let grad_u_g = get(
        &evaluate_named(&g_differentiated.program, &[], &[("u", &u_values), ("weight", &weight_values)], &[grad_u_g_node])
            .expect("g's adjoint program evaluates"),
        grad_u_g_node,
    );

    let (full_program, _loss1, loss2) = build_full_program(&x_values, &y_values, &weight_values);
    let composite_differentiated = differentiate(&full_program, loss2).expect(
        "full_program differentiates at its final loss even though it also contains an \
         earlier loss node and a trailing, unrelated Input -- this is the composition \
         defect 3's fix protects",
    );
    let grad_x_composite_node = composite_differentiated.gradient_of_named("x").expect("x feeds the composite loss");
    let grad_x_composite = get(
        &evaluate_named(
            &composite_differentiated.program,
            &[],
            &[("x", &x_values), ("y", &y_values), ("weight", &weight_values)],
            &[grad_x_composite_node],
        )
        .expect("composite adjoint program evaluates"),
        grad_x_composite_node,
    );

    // Sanity check the earlier-node call ALSO still works standing on its
    // own two feet: differentiating full_program at loss1 (embedded, with
    // y and weight appended afterward) must reproduce f's own isolated
    // du/dx -- this is precisely where the reverted defect 3 would panic.
    let f_within_full = differentiate(&full_program, _loss1).expect(
        "differentiating full_program at its EARLIER loss node, with an unrelated Input \
         appended afterward, must not panic -- this is defect 3's exact trigger",
    );
    let grad_x_embedded_node = f_within_full.gradient_of_named("x").expect("x feeds loss1");
    let grad_x_embedded = get(
        &evaluate_named(&f_within_full.program, &[], &[("x", &x_values)], &[grad_x_embedded_node])
            .expect("embedded f adjoint program evaluates"),
        grad_x_embedded_node,
    );

    for index in 0..x_values.len() {
        assert_close(
            grad_x_embedded[index],
            grad_x_f[index],
            1e-4,
            format!("embedded du/dx must match du/dx computed in isolation at index {index}"),
        );
        let composed = grad_u_g[index] * grad_x_f[index];
        assert_close(
            grad_x_composite[index],
            composed,
            1e-4 + 1e-4 * composed.abs(),
            format!(
                "chain rule: dL/dx must equal (dL/du) * (du/dx) at index {index}, \
                 dL/du={} du/dx={}",
                grad_u_g[index], grad_x_f[index]
            ),
        );
    }
}

// ---------------------------------------------------------------------
// Law 2: linearity in the seed
// ---------------------------------------------------------------------

/// **Law 2.** One program, differentiated ONCE, evaluated three times with
/// only the bound `weight` array changed between `a`, `b`, and `a + b`.
/// `dL/dx = weight * exp(x*x) * 2x` is linear in `weight`, so
/// `grad_x(a + b)` must equal `grad_x(a) + grad_x(b)` to f32 rounding.
///
/// Broken by dropping the `gradient` factor from `Exponential`'s adjoint
/// (`adjoint.rs:330`, `vec![Some(node)]` instead of
/// `vec![Some(Multiply(gradient, node)))]`): this disconnects `weight`
/// from `grad_x`'s expression graph entirely (the graph built for `grad_x`
/// no longer references the `weight` node transitively at all), so
/// `grad_x` evaluates to the SAME constant regardless of which array is
/// bound to `weight` -- `grad_x(a) + grad_x(b) = 2c` while
/// `grad_x(a + b) = c`, which fails for any `c != 0`. This is exactly the
/// "forgot to multiply by the incoming gradient" bug law 1 could NOT see
/// (there, the same drop fires identically inside both the isolated piece
/// and the composite, so their product still matched).
#[proxima::test]
async fn the_adjoint_is_linear_in_its_seed() {
    let x_values = [0.6_f32, -1.4, 2.1, 0.3];
    let a_values = [0.5_f32, -2.0, 1.25, 3.0];
    let b_values = [-1.0_f32, 0.75, 2.5, -0.5];
    let sum_values: Vec<f32> = a_values.iter().zip(b_values).map(|(left, right)| left + right).collect();

    let mut program = Vec::new();
    let x = leaf(&mut program, "x", x_values.len());
    let weight = leaf(&mut program, "weight", x_values.len());
    let u = elementwise(&mut program, ScalarOp::Multiply, vec![(x, ident_map(1)), (x, ident_map(1))]);
    let v = elementwise(&mut program, ScalarOp::Exponential, vec![(u, ident_map(1))]);
    let w = elementwise(&mut program, ScalarOp::Multiply, vec![(v, ident_map(1)), (weight, ident_map(1))]);
    let loss = reduce_scalar_add(&mut program, w, 1);

    let differentiated = differentiate(&program, loss).expect("network differentiates");
    let grad_x_node = differentiated.gradient_of_named("x").expect("x feeds the loss");

    let evaluate_with = |weight_bound: &[f32]| {
        get(
            &evaluate_named(&differentiated.program, &[], &[("x", &x_values), ("weight", weight_bound)], &[grad_x_node])
                .expect("adjoint program evaluates"),
            grad_x_node,
        )
    };

    let grad_x_a = evaluate_with(&a_values);
    let grad_x_b = evaluate_with(&b_values);
    let grad_x_sum = evaluate_with(&sum_values);

    for index in 0..x_values.len() {
        let additive = grad_x_a[index] + grad_x_b[index];
        assert_close(
            grad_x_sum[index],
            additive,
            1e-3 + 1e-3 * additive.abs(),
            format!(
                "linearity: grad_x(a+b) must equal grad_x(a)+grad_x(b) at index {index}, \
                 grad_x(a)={} grad_x(b)={} grad_x(a+b)={}",
                grad_x_a[index], grad_x_b[index], grad_x_sum[index]
            ),
        );
    }
}

// ---------------------------------------------------------------------
// Law 3: per-ScalarOp local derivative
// ---------------------------------------------------------------------

type UnaryDerivative = fn(f32) -> f32;
type BinaryDerivative = fn(f32, f32) -> f32;

fn d_identity(_x: f32) -> f32 {
    1.0
}
fn d_negate(_x: f32) -> f32 {
    -1.0
}
fn d_reciprocal(x: f32) -> f32 {
    -1.0 / (x * x)
}
fn d_exponential(x: f32) -> f32 {
    x.exp()
}
fn d_logarithm(x: f32) -> f32 {
    1.0 / x
}
fn d_sqrt(x: f32) -> f32 {
    1.0 / (2.0 * x.sqrt())
}
fn d_tanh(x: f32) -> f32 {
    let t = x.tanh();
    1.0 - t * t
}
fn d_erf(x: f32) -> f32 {
    (2.0 / core::f32::consts::PI.sqrt()) * (-x * x).exp()
}

/// Wraps `body(x)` as `loss = sum(body(x) * weight)`, differentiates once,
/// and recovers `body'(x)` by dividing the system's own `grad_x` by the
/// independently bound `weight` -- the seed must not be trivially `1.0` at
/// this op (see law 1/2's doc for why that would hide a "dropped gradient"
/// bug), which is exactly what wrapping in `Multiply(_, weight)` avoids.
#[proxima::test]
#[case::unary_identity(ScalarOp::Identity, &[0.6_f32, -1.4, 2.1, 0.3], d_identity as UnaryDerivative)]
#[case::negate(ScalarOp::Negate, &[0.6_f32, -1.4, 2.1, 0.3], d_negate as UnaryDerivative)]
#[case::reciprocal(ScalarOp::Reciprocal, &[0.6_f32, -1.4, 2.1, 0.3], d_reciprocal as UnaryDerivative)]
#[case::exponential(ScalarOp::Exponential, &[0.6_f32, -1.4, 2.1, 0.3], d_exponential as UnaryDerivative)]
#[case::logarithm(ScalarOp::Logarithm, &[0.6_f32, 1.4, 2.1, 3.3], d_logarithm as UnaryDerivative)]
#[case::square_root(ScalarOp::SquareRoot, &[0.6_f32, 1.4, 2.1, 3.3], d_sqrt as UnaryDerivative)]
#[case::tanh(ScalarOp::Tanh, &[0.6_f32, -1.4, 2.1, 0.3], d_tanh as UnaryDerivative)]
#[case::erf(ScalarOp::Erf, &[0.6_f32, -1.4, 2.1, 0.3], d_erf as UnaryDerivative)]
async fn unary_scalar_op_local_derivative_matches_the_closed_form(
    #[case] body: ScalarOp,
    #[case] x_values: &[f32],
    #[case] closed_form: UnaryDerivative,
) {
    let weight_values = [0.7_f32, -1.3, 2.0, 0.4];
    let mut program = Vec::new();
    let x = leaf(&mut program, "x", x_values.len());
    let weight = leaf(&mut program, "weight", weight_values.len());
    let u = elementwise(&mut program, body, vec![(x, ident_map(1))]);
    let w = elementwise(&mut program, ScalarOp::Multiply, vec![(u, ident_map(1)), (weight, ident_map(1))]);
    let loss = reduce_scalar_add(&mut program, w, 1);

    let differentiated = differentiate(&program, loss).expect("elementwise unary op differentiates");
    let grad_x_node = differentiated.gradient_of_named("x").expect("x feeds the loss");
    let grad_x = get(
        &evaluate_named(&differentiated.program, &[], &[("x", x_values), ("weight", &weight_values)], &[grad_x_node])
            .expect("adjoint program evaluates"),
        grad_x_node,
    );

    for index in 0..x_values.len() {
        let recovered = grad_x[index] / weight_values[index];
        let expected = closed_form(x_values[index]);
        assert_close(
            recovered,
            expected,
            combined_tolerance(expected),
            format!("{body:?}'({}) recovered={recovered} expected={expected}", x_values[index]),
        );
    }
}

fn d_add_da(_a: f32, _b: f32) -> f32 {
    1.0
}
fn d_add_db(_a: f32, _b: f32) -> f32 {
    1.0
}
fn d_sub_da(_a: f32, _b: f32) -> f32 {
    1.0
}
fn d_sub_db(_a: f32, _b: f32) -> f32 {
    -1.0
}
fn d_mul_da(_a: f32, b: f32) -> f32 {
    b
}
fn d_mul_db(a: f32, _b: f32) -> f32 {
    a
}
fn d_div_da(_a: f32, b: f32) -> f32 {
    1.0 / b
}
fn d_div_db(a: f32, b: f32) -> f32 {
    -a / (b * b)
}
/// Ties favor the FIRST operand (`a`) for both `Maximum` and `Minimum` --
/// `maximum_minimum_grads`'s own doc (`adjoint.rs:390-392`): at `a == b`,
/// `Greater` returns `0`, so `first_operand_wins = 1 - 0 = 1` regardless of
/// `is_maximum`. This is a DIFFERENT convention from `Reduce(Maximum)`'s
/// (law 4 below), which routes to EVERY tied position, not just the first.
fn d_max_da(a: f32, b: f32) -> f32 {
    if a >= b { 1.0 } else { 0.0 }
}
fn d_max_db(a: f32, b: f32) -> f32 {
    1.0 - d_max_da(a, b)
}
fn d_min_da(a: f32, b: f32) -> f32 {
    if a <= b { 1.0 } else { 0.0 }
}
fn d_min_db(a: f32, b: f32) -> f32 {
    1.0 - d_min_da(a, b)
}

/// `a_values`/`b_values` include TWO tie positions (index 0: `2.0 == 2.0`,
/// index 2: `3.0 == 3.0`) alongside two non-tie, mixed-sign positions --
/// asymmetric and non-degenerate, and enough to witness the elementwise
/// `Maximum`/`Minimum` tie convention directly, not just assume it.
#[proxima::test]
#[case::add(ScalarOp::Add, &[3.0_f32, -2.0, 5.0, 0.8], &[1.5_f32, 4.0, -2.0, 0.25], d_add_da as BinaryDerivative, d_add_db as BinaryDerivative)]
#[case::subtract(ScalarOp::Subtract, &[3.0_f32, -2.0, 5.0, 0.8], &[1.5_f32, 4.0, -2.0, 0.25], d_sub_da as BinaryDerivative, d_sub_db as BinaryDerivative)]
#[case::multiply(ScalarOp::Multiply, &[3.0_f32, -2.0, 5.0, 0.8], &[1.5_f32, 4.0, -2.0, 0.25], d_mul_da as BinaryDerivative, d_mul_db as BinaryDerivative)]
#[case::divide(ScalarOp::Divide, &[3.0_f32, -2.0, 5.0, 0.8], &[1.5_f32, 4.0, -2.0, 0.25], d_div_da as BinaryDerivative, d_div_db as BinaryDerivative)]
#[case::maximum_with_ties(ScalarOp::Maximum, &[2.0_f32, -1.0, 3.0, 0.5], &[2.0_f32, 1.0, 3.0, -0.5], d_max_da as BinaryDerivative, d_max_db as BinaryDerivative)]
#[case::minimum_with_ties(ScalarOp::Minimum, &[2.0_f32, -1.0, 3.0, 0.5], &[2.0_f32, 1.0, 3.0, -0.5], d_min_da as BinaryDerivative, d_min_db as BinaryDerivative)]
async fn binary_scalar_op_local_derivative_matches_the_closed_form(
    #[case] body: ScalarOp,
    #[case] a_values: &[f32],
    #[case] b_values: &[f32],
    #[case] d_da: BinaryDerivative,
    #[case] d_db: BinaryDerivative,
) {
    let weight_values = [0.7_f32, -1.3, 2.0, 0.4];
    let mut program = Vec::new();
    let a = leaf(&mut program, "a", a_values.len());
    let b = leaf(&mut program, "b", b_values.len());
    let weight = leaf(&mut program, "weight", weight_values.len());
    let u = elementwise(&mut program, body, vec![(a, ident_map(1)), (b, ident_map(1))]);
    let w = elementwise(&mut program, ScalarOp::Multiply, vec![(u, ident_map(1)), (weight, ident_map(1))]);
    let loss = reduce_scalar_add(&mut program, w, 1);

    let differentiated = differentiate(&program, loss).expect("elementwise binary op differentiates");
    let grad_a_node = differentiated.gradient_of_named("a").expect("a feeds the loss");
    let grad_b_node = differentiated.gradient_of_named("b").expect("b feeds the loss");
    let evaluated = evaluate_named(
        &differentiated.program,
        &[],
        &[("a", a_values), ("b", b_values), ("weight", &weight_values)],
        &[grad_a_node, grad_b_node],
    )
    .expect("adjoint program evaluates");
    let grad_a = get(&evaluated, grad_a_node);
    let grad_b = get(&evaluated, grad_b_node);

    for index in 0..a_values.len() {
        let recovered_da = grad_a[index] / weight_values[index];
        let expected_da = d_da(a_values[index], b_values[index]);
        assert_close(
            recovered_da,
            expected_da,
            combined_tolerance(expected_da),
            format!("{body:?} d/da at (a={}, b={})", a_values[index], b_values[index]),
        );

        let recovered_db = grad_b[index] / weight_values[index];
        let expected_db = d_db(a_values[index], b_values[index]);
        assert_close(
            recovered_db,
            expected_db,
            combined_tolerance(expected_db),
            format!("{body:?} d/db at (a={}, b={})", a_values[index], b_values[index]),
        );
    }
}

/// **Law 3, comparators.** `Greater`/`Equal` have arity 2 and NO adjoint
/// rule by design (`adjoint.rs:358`: `operands.iter().map(|_| None).collect()`)
/// -- neither operand may pick up a dense gradient purely through a
/// comparison, confirmed here rather than assumed.
#[proxima::test]
#[case::greater(ScalarOp::Greater)]
#[case::equal(ScalarOp::Equal)]
async fn comparator_ops_never_produce_a_gradient(#[case] body: ScalarOp) {
    let a_values = [3.0_f32, -1.0, 0.5, 2.0];
    let b_values = [1.0_f32, 2.0, 0.5, 0.5];
    let mut program = Vec::new();
    let a = leaf(&mut program, "a", a_values.len());
    let b = leaf(&mut program, "b", b_values.len());
    let mask = elementwise(&mut program, body, vec![(a, ident_map(1)), (b, ident_map(1))]);
    let squared = elementwise(&mut program, ScalarOp::Multiply, vec![(mask, ident_map(1)), (mask, ident_map(1))]);
    let loss = reduce_scalar_add(&mut program, squared, 1);

    let differentiated = differentiate(&program, loss).expect("comparator-only program differentiates");
    assert!(
        differentiated.gradient_of_named("a").is_none(),
        "{body:?} must not route a gradient to its left operand"
    );
    assert!(
        differentiated.gradient_of_named("b").is_none(),
        "{body:?} must not route a gradient to its right operand"
    );
}

/// **Law 3, `Select`.** `condition = Greater(a, b)` is a real 0/1 mask (the
/// causal-mask idiom `select_broadcast_condition_tests` also builds), with
/// BOTH a true and a false position AND one tie (`a[2] == b[2] == 0.5`,
/// where `Greater` is false, so `Select` routes to `false_branch`).
/// `Select`'s adjoint (`adjoint.rs:359-380`) must route
/// `weight * condition` to `true_branch` and `weight * (1 - condition)` to
/// `false_branch`.
#[proxima::test]
async fn select_routes_the_seed_by_its_own_condition_mask() {
    let a_values = [3.0_f32, -1.0, 0.5, 2.0];
    let b_values = [1.0_f32, 2.0, 0.5, 0.5];
    let true_values = [10.0_f32, 20.0, 30.0, 40.0];
    let false_values = [-5.0_f32, -6.0, -7.0, -8.0];
    let weight_values = [0.5_f32, -2.0, 1.25, 3.0];
    let condition_values = [1.0_f32, 0.0, 0.0, 1.0]; // Greater(a, b) at these inputs

    let mut program = Vec::new();
    let a = leaf(&mut program, "a", a_values.len());
    let b = leaf(&mut program, "b", b_values.len());
    let condition = elementwise(&mut program, ScalarOp::Greater, vec![(a, ident_map(1)), (b, ident_map(1))]);
    let true_branch = leaf(&mut program, "true_branch", true_values.len());
    let false_branch = leaf(&mut program, "false_branch", false_values.len());
    let weight = leaf(&mut program, "weight", weight_values.len());
    let selected = elementwise(
        &mut program,
        ScalarOp::Select,
        vec![(condition, ident_map(1)), (true_branch, ident_map(1)), (false_branch, ident_map(1))],
    );
    let w = elementwise(&mut program, ScalarOp::Multiply, vec![(selected, ident_map(1)), (weight, ident_map(1))]);
    let loss = reduce_scalar_add(&mut program, w, 1);

    let differentiated = differentiate(&program, loss).expect("select program differentiates");
    let grad_true_node = differentiated.gradient_of_named("true_branch").expect("true_branch feeds the loss");
    let grad_false_node = differentiated.gradient_of_named("false_branch").expect("false_branch feeds the loss");
    let evaluated = evaluate_named(
        &differentiated.program,
        &[],
        &[
            ("a", &a_values),
            ("b", &b_values),
            ("true_branch", &true_values),
            ("false_branch", &false_values),
            ("weight", &weight_values),
        ],
        &[grad_true_node, grad_false_node],
    )
    .expect("adjoint program evaluates");
    let grad_true = get(&evaluated, grad_true_node);
    let grad_false = get(&evaluated, grad_false_node);

    for index in 0..a_values.len() {
        let expected_true = weight_values[index] * condition_values[index];
        let expected_false = weight_values[index] * (1.0 - condition_values[index]);
        assert_close(grad_true[index], expected_true, 1e-4, format!("select true_branch mask at index {index}"));
        assert_close(grad_false[index], expected_false, 1e-4, format!("select false_branch mask at index {index}"));
    }
}

// ---------------------------------------------------------------------
// Law 4: Reduce adjoints per body
// ---------------------------------------------------------------------

/// **Law 4, `Add`.** `x` is `[3, 2]` (3 asymmetric rows, 2 columns);
/// reducing over the column axis and weighting each row's scalar output
/// broadcasts that SAME row weight back to BOTH columns of that row --
/// `Reduce(Add)`'s adjoint is a pure broadcast, not a mask.
///
/// Broken by deleting the `ScalarOp::Add` arm from `differentiate_reduce`'s
/// match (`adjoint.rs:507-513`), which falls through to
/// `other => Err(UnsupportedReduceBody)`: `differentiate` then returns an
/// `Err` instead of `Ok`, which is as unambiguous a failure as this suite
/// produces.
#[proxima::test]
async fn reduce_add_broadcasts_the_seed_to_every_contributing_element() {
    let x_values = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3 rows x 2 cols, row-major
    let weight_values = [0.5_f32, -2.0, 1.5]; // one weight per row

    let mut program = Vec::new();
    let x = leaf_shaped(&mut program, "x", vec![Extent::Static(3), Extent::Static(2)]);
    let weight = leaf(&mut program, "weight", weight_values.len());
    let row_sum = reduce_node(&mut program, ScalarOp::Add, ReduceInit::Zero, x, ident_map(2), IndexMap::Affine(map::projection(2, &[0])));
    let w = elementwise(&mut program, ScalarOp::Multiply, vec![(row_sum, ident_map(1)), (weight, ident_map(1))]);
    let loss = reduce_scalar_add(&mut program, w, 1);

    let differentiated = differentiate(&program, loss).expect("Reduce(Add) differentiates");
    let grad_x_node = differentiated.gradient_of_named("x").expect("x feeds the loss");
    let grad_x = get(
        &evaluate_named(&differentiated.program, &[], &[("x", &x_values), ("weight", &weight_values)], &[grad_x_node])
            .expect("adjoint program evaluates"),
        grad_x_node,
    );

    for (row, row_weight) in weight_values.iter().enumerate() {
        for col in 0..2 {
            let index = row * 2 + col;
            assert_close(
                grad_x[index],
                *row_weight,
                1e-4,
                format!("Reduce(Add) must broadcast row {row}'s weight to column {col}"),
            );
        }
    }
}

/// **Law 4, `Maximum`/`Minimum` with a tie.** `x` has two elements tied for
/// the extreme value (`Maximum`: indices 1 and 2 both `5.0`; `Minimum`:
/// indices 1 and 2 both `-5.0`), surrounded by asymmetric, distinct,
/// non-degenerate values. Every position tied for the extreme gets the
/// FULL seed (`1.0` each here, not `0.5` split) -- this crate's documented
/// convention (`adjoint.rs:31-34`), matching TensorFlow's `reduce_max`
/// gradient rather than PyTorch's single-argmax-kernel convention.
///
/// Broken by dropping the `mask` factor from the `Maximum | Minimum` arm
/// (`adjoint.rs:514-537`, using `gradient_broadcast` alone instead of
/// `Multiply(mask, gradient_broadcast)`) -- this is defect 2 verbatim: the
/// adjoint degenerates into a plain broadcast (as if it were `Add`), so
/// EVERY position gets the seed, not just the tied extremes: `[1,1,1,1,1]`
/// instead of `[0,1,1,0,0]`.
#[proxima::test]
#[case::maximum_ties_route_full_gradient_to_both(ScalarOp::Maximum, ReduceInit::NegativeInfinity, &[3.0_f32, 5.0, 5.0, 1.0, -2.0], &[0.0_f32, 1.0, 1.0, 0.0, 0.0])]
#[case::minimum_ties_route_full_gradient_to_both(ScalarOp::Minimum, ReduceInit::PositiveInfinity, &[3.0_f32, -5.0, -5.0, 1.0, 2.0], &[0.0_f32, 1.0, 1.0, 0.0, 0.0])]
async fn reduce_max_and_min_route_only_to_the_argmax_or_argmin_mask(
    #[case] body: ScalarOp,
    #[case] init: ReduceInit,
    #[case] x_values: &[f32],
    #[case] expected: &[f32],
) {
    let mut program = Vec::new();
    let x = leaf(&mut program, "x", x_values.len());
    let loss = reduce_node(&mut program, body, init, x, ident_map(1), broadcast(1));

    let differentiated = differentiate(&program, loss).expect("Reduce(Maximum/Minimum) differentiates");
    let grad_x_node = differentiated.gradient_of_named("x").expect("x feeds the loss");
    let grad_x = get(
        &evaluate_named(&differentiated.program, &[], &[("x", x_values)], &[grad_x_node]).expect("adjoint program evaluates"),
        grad_x_node,
    );

    assert_eq!(
        grad_x, expected,
        "{body:?} must route the full seed to every tied extreme position and zero elsewhere, got {grad_x:?}"
    );
}

/// **Law 4, `Multiply`.** `x` is `[3, 5]` (no zero, so the divide-form rule
/// is well-defined everywhere), weighted by a real non-trivial seed so this
/// law -- like law 2 -- would catch a dropped `gradient` factor, not just a
/// wrong shape. `prod(x) = 15`, so `d(prod x)/dx = [prod(x)/x_0,
/// prod(x)/x_1] = [5, 3]`, weighted by `weight = 2.0`: `grad_x = [10, 6]`
/// exactly.
///
/// Before this crate implemented `Reduce(Multiply)`'s adjoint,
/// `differentiate` returned `Err(UnsupportedReduceBody)` for this program --
/// `ScalarOp::Multiply.is_associative()` is `true`
/// (`proxima-tensor/src/op.rs:112-117`), so this reduce body was always a
/// legal REDUCE to build, just one this crate's adjoint transform could not
/// yet differentiate.
#[proxima::test]
async fn reduce_multiply_divides_the_seed_by_each_input_exactly() {
    let x_values = [3.0_f32, 5.0];
    let weight_value = 2.0_f32;

    let mut program = Vec::new();
    let x = leaf(&mut program, "x", x_values.len());
    let weight = leaf_shaped(&mut program, "weight", vec![]);
    let product = reduce_node(&mut program, ScalarOp::Multiply, ReduceInit::One, x, ident_map(1), broadcast(1));
    let loss = elementwise(&mut program, ScalarOp::Multiply, vec![(product, ident_map(0)), (weight, ident_map(0))]);

    let differentiated = differentiate(&program, loss).expect("Reduce(Multiply) differentiates");
    let grad_x_node = differentiated.gradient_of_named("x").expect("x feeds the loss");
    let grad_x = get(
        &evaluate_named(&differentiated.program, &[], &[("x", &x_values), ("weight", &[weight_value])], &[grad_x_node])
            .expect("adjoint program evaluates"),
        grad_x_node,
    );

    assert_close(grad_x[0], 10.0, 1e-4, format!("Reduce(Multiply) d/dx_0, got {grad_x:?}"));
    assert_close(grad_x[1], 6.0, 1e-4, format!("Reduce(Multiply) d/dx_1, got {grad_x:?}"));
}
