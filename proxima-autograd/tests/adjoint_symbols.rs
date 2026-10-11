//! `differentiate_bound` reads shapes under a symbol binding, so a program
//! with `Extent::Symbolic` extents is differentiated as built. The program
//! is the rmsnorm-shaped core granite uses: `loss = sum((x * w + eps)^2 + eps)`
//! where `eps` is a rank-1 `[1]` input read through a 0-axis map at two
//! consumers; summing the two contributions read the rank-0 gradient through
//! a rank-1 identity and `shape::infer` rejected the adjoint with
//! `UnconstrainedDim`.
//!
//! `d loss / d x[i] = 2 * (x[i] * w[i] + eps) * w[i]`,
//! `d loss / d w[i] = 2 * (x[i] * w[i] + eps) * x[i]` and
//! `d loss / d eps = sum_i (2 * (x[i] * w[i] + eps) + 1)`. The f64 loop below
//! evaluates those formulas with no call into the adjoint.
#![allow(clippy::unwrap_used, clippy::expect_used)]

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use proxima_autograd::AutogradError;
use proxima_autograd::adjoint::{Differentiated, differentiate, differentiate_bound};
use proxima_tensor::TensorError;
use proxima_tensor::cpu::evaluate_named;
use proxima_tensor::dtype::DType;
use proxima_tensor::map::{self, IndexMap};
use proxima_tensor::op::{self, Extent, Keep, NodeId, Op, Reduce, ReduceInit, ScalarOp};
use proxima_tensor::shape;

const HIDDEN_ROW: [f32; 8] = [0.18, -0.42, 0.07, 0.91, -0.33, 0.26, -0.58, 0.64];
const GAIN_ROW: [f32; 8] = [1.02, 0.97, 1.11, 0.88, 1.05, 0.93, 1.00, 1.08];
const EPSILON: [f32; 1] = [1.0e-5];

fn input(program: &mut Vec<Op>, name: &str, shape: Vec<Extent>) -> NodeId {
    op::append(
        program,
        Op::Input {
            dtype: DType::Float32,
            shape,
            name: Some(name.into()),
        },
    )
}

fn elementwise(program: &mut Vec<Op>, body: ScalarOp, operands: Vec<(NodeId, IndexMap)>) -> NodeId {
    op::append(
        program,
        Op::Elementwise {
            dtype: DType::Float32,
            body,
            operands,
            name: None,
        },
    )
}

fn identity() -> IndexMap {
    IndexMap::Affine(map::projection(1, &[0]))
}

fn zero_axis_read() -> IndexMap {
    IndexMap::Affine(map::projection(1, &[]))
}

fn norm_loss(extent: Extent) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let hidden = input(&mut program, "x", vec![extent]);
    let gain = input(&mut program, "w", vec![extent]);
    let epsilon = input(&mut program, "eps", vec![Extent::Static(1)]);
    let scaled = elementwise(
        &mut program,
        ScalarOp::Multiply,
        vec![(hidden, identity()), (gain, identity())],
    );
    let shifted = elementwise(
        &mut program,
        ScalarOp::Add,
        vec![(scaled, identity()), (epsilon, zero_axis_read())],
    );
    let squared = elementwise(
        &mut program,
        ScalarOp::Multiply,
        vec![(shifted, identity()), (shifted, identity())],
    );
    let offset = elementwise(
        &mut program,
        ScalarOp::Add,
        vec![(squared, identity()), (epsilon, zero_axis_read())],
    );
    let loss = op::append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: offset,
            in_map: identity(),
            out_map: IndexMap::Affine(map::projection(1, &[])),
            keep: Keep::Reduce,
            name: None,
        }),
    );
    (program, loss)
}

fn f64_loop_gradients(hidden: &[f32], gain: &[f32], epsilon: f32) -> [Vec<f64>; 3] {
    let residual =
        |index: usize| f64::from(hidden[index]) * f64::from(gain[index]) + f64::from(epsilon);
    let hidden_gradient = (0..hidden.len())
        .map(|index| 2.0 * residual(index) * f64::from(gain[index]))
        .collect();
    let gain_gradient = (0..hidden.len())
        .map(|index| 2.0 * residual(index) * f64::from(hidden[index]))
        .collect();
    let epsilon_gradient = (0..hidden.len())
        .map(|index| 2.0 * residual(index) + 1.0)
        .sum();
    [hidden_gradient, gain_gradient, vec![epsilon_gradient]]
}

fn gradient_nodes(differentiated: &Differentiated) -> [NodeId; 3] {
    ["x", "w", "eps"].map(|name| {
        differentiated
            .gradient_of_named(name)
            .expect("every input feeds the loss")
    })
}

fn evaluated_gradients(differentiated: &Differentiated, symbols: &[u64]) -> [Vec<f32>; 3] {
    let nodes = gradient_nodes(differentiated);
    let evaluated = evaluate_named(
        &differentiated.program,
        symbols,
        &[
            ("x", &HIDDEN_ROW[..]),
            ("w", &GAIN_ROW[..]),
            ("eps", &EPSILON[..]),
        ],
        &nodes,
    )
    .expect("adjoint program evaluates under the binding");
    nodes.map(|node| evaluated.get(node).expect("gradient evaluated").0.to_vec())
}

fn assert_matches_loop(got: &[Vec<f32>; 3], expected: &[Vec<f64>; 3]) {
    for (got_gradient, want_gradient) in got.iter().zip(expected) {
        assert_eq!(got_gradient.len(), want_gradient.len());
        for (got_value, want_value) in got_gradient.iter().zip(want_gradient) {
            assert!(
                (f64::from(*got_value) - want_value).abs() <= 1e-5 * (1.0 + want_value.abs()),
                "adjoint {got_value} vs f64 loop {want_value}"
            );
        }
    }
}

#[test]
fn adjoint_symbols_bound_equals_constant_rewrite() {
    let (symbolic, symbolic_loss) = norm_loss(Extent::Symbolic(0));
    let (constant, constant_loss) = norm_loss(Extent::Static(8));

    let bound =
        differentiate_bound(&symbolic, symbolic_loss, &[8]).expect("bound program differentiates");
    let rewritten =
        differentiate(&constant, constant_loss).expect("constant program differentiates");

    assert_eq!(bound.program.len(), rewritten.program.len());
    let from_bound = evaluated_gradients(&bound, &[8]);
    let from_constant = evaluated_gradients(&rewritten, &[]);
    assert_eq!(from_bound, from_constant);
    assert_matches_loop(
        &from_bound,
        &f64_loop_gradients(&HIDDEN_ROW, &GAIN_ROW, EPSILON[0]),
    );
}

#[test]
fn adjoint_symbols_rank1_input_read_through_zero_axis_map_differentiates() {
    let (program, loss) = norm_loss(Extent::Static(8));

    let differentiated = differentiate(&program, loss).expect("eps gradient is built");

    let epsilon_gradient = differentiated
        .gradient_of_named("eps")
        .expect("eps feeds the loss");
    let shapes = shape::infer(&differentiated.program, &[]).expect("the adjoint program infers");
    assert_eq!(shapes.of(epsilon_gradient), &[] as &[u64]);
    assert_matches_loop(
        &evaluated_gradients(&differentiated, &[]),
        &f64_loop_gradients(&HIDDEN_ROW, &GAIN_ROW, EPSILON[0]),
    );
}

#[test]
fn adjoint_symbols_unbound_still_errors_unbound_symbol() {
    let (program, loss) = norm_loss(Extent::Symbolic(0));

    let unbound = differentiate_bound(&program, loss, &[])
        .err()
        .expect("symbol 0 has no binding");
    let plain = differentiate(&program, loss)
        .err()
        .expect("plain differentiate binds nothing");

    let expected = AutogradError::ShapeInference(TensorError::UnboundSymbol { symbol: 0 });
    assert_eq!(unbound, expected);
    assert_eq!(plain, expected);
}

#[test]
fn adjoint_symbols_control_wrong_binding_changes_the_gradient_shape() {
    let (program, loss) = norm_loss(Extent::Symbolic(0));

    let narrow = differentiate_bound(&program, loss, &[3]).expect("binding 3 differentiates");
    let wide = differentiate_bound(&program, loss, &[8]).expect("binding 8 differentiates");

    let narrow_shapes = shape::infer(&narrow.program, &[3]).expect("narrow infers");
    let wide_shapes = shape::infer(&wide.program, &[8]).expect("wide infers");
    let narrow_gradient = narrow.gradient_of_named("x").expect("x feeds the loss");
    let wide_gradient = wide.gradient_of_named("x").expect("x feeds the loss");
    assert_eq!(narrow_shapes.of(narrow_gradient), &[3]);
    assert_eq!(wide_shapes.of(wide_gradient), &[8]);
}
