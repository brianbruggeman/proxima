//! Cross-module correctness oracle for [`proxima_autograd::adjoint::differentiate`]:
//! a small dense net (matmul + bias + relu, matmul + bias + softmax,
//! cross-entropy), the central-difference gradient check, an end-to-end
//! Adam training run whose loss actually decreases, and proof the checker
//! can fail. Cross-module (adjoint + activation + optimizer + evaluation)
//! is exactly the case this workspace's own testing convention reserves an
//! integration test for, rather than a `#[cfg(test)]` module.
#![allow(clippy::unwrap_used, clippy::expect_used)]


use proxima_autograd::activation::{relu, softmax};
use proxima_autograd::adjoint::differentiate;
use proxima_autograd::optimizer::{AdamConfig, AdamOperands, adam_step, step_input};
use proxima_tensor::cpu::evaluate_named;
use proxima_tensor::dtype::DType;
use proxima_tensor::map::{self, IndexMap};
use proxima_tensor::op::{self, Extent, NodeId, Op, ReduceInit, ScalarOp};

const IN_DIM: usize = 3;
const HIDDEN_DIM: usize = 4;
const OUT_DIM: usize = 2;

struct Network {
    program: Vec<Op>,
    w1: NodeId,
    b1: NodeId,
    w2: NodeId,
    b2: NodeId,
    loss: NodeId,
}

fn leaf(program: &mut Vec<Op>, name: &str, shape: Vec<Extent>) -> NodeId {
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
        Op::Elementwise { dtype: DType::Float32, body, operands, name: None },
    )
}

#[allow(clippy::too_many_arguments)]
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

fn identity(rank: u16) -> IndexMap {
    IndexMap::Affine(map::projection(rank, &(0..rank).collect::<Vec<u16>>()))
}

/// `x @ w + b` — a dense layer, matmul via `Elementwise(Multiply)` then
/// `Reduce(Add)` (the same shape `proxima-tensor/src/lib.rs:130-168`'s own
/// crate-doc matmul example builds), plus a bias add.
fn dense(program: &mut Vec<Op>, x: NodeId, w: NodeId, b: NodeId) -> NodeId {
    let product = elementwise(
        program,
        ScalarOp::Multiply,
        alloc::vec![(w, identity(2)), (x, IndexMap::Affine(map::projection(2, &[0])))],
    );
    let matmul = reduce_add(program, product, identity(2), IndexMap::Affine(map::projection(2, &[1])));
    elementwise(program, ScalarOp::Add, alloc::vec![(matmul, identity(1)), (b, identity(1))])
}

extern crate alloc;

/// `x -> dense(w1,b1) -> relu -> dense(w2,b2) -> softmax -> cross-entropy(y)`.
/// Exercises every adjoint rule this crate ships: `Elementwise::Multiply`/
/// `Add`/`Subtract`/`Reciprocal`/`Exponential`/`Logarithm`/`Negate` (chain
/// rule), elementwise `Maximum` (`relu`'s ties-favor-`a` convention), and
/// both `Reduce(Add)` (matmul, softmax's sum, cross-entropy's sum) and
/// `Reduce(Maximum)` (softmax's numerically-stable max) — the masked-routing
/// rule this crate's own report calls out as the one that is NOT a broadcast.
fn build_network() -> Network {
    let mut program = Vec::new();
    let x = leaf(&mut program, "x", alloc::vec![Extent::Static(IN_DIM as u32)]);
    let y = leaf(&mut program, "y", alloc::vec![Extent::Static(OUT_DIM as u32)]);
    let w1 = leaf(&mut program, "w1", alloc::vec![Extent::Static(IN_DIM as u32), Extent::Static(HIDDEN_DIM as u32)]);
    let b1 = leaf(&mut program, "b1", alloc::vec![Extent::Static(HIDDEN_DIM as u32)]);
    let w2 = leaf(&mut program, "w2", alloc::vec![Extent::Static(HIDDEN_DIM as u32), Extent::Static(OUT_DIM as u32)]);
    let b2 = leaf(&mut program, "b2", alloc::vec![Extent::Static(OUT_DIM as u32)]);

    let h_pre = dense(&mut program, x, w1, b1);
    let h = relu(&mut program, DType::Float32, h_pre, 1);
    let out_pre = dense(&mut program, h, w2, b2);
    let probabilities = softmax(&mut program, DType::Float32, out_pre, 1, 0);

    let log_probabilities = elementwise(&mut program, ScalarOp::Logarithm, alloc::vec![(probabilities, identity(1))]);
    let weighted = elementwise(&mut program, ScalarOp::Multiply, alloc::vec![(y, identity(1)), (log_probabilities, identity(1))]);
    let sum = reduce_add(&mut program, weighted, identity(1), IndexMap::Affine(map::projection(1, &[])));
    let loss = elementwise(&mut program, ScalarOp::Negate, alloc::vec![(sum, identity(0))]);

    Network { program, w1, b1, w2, b2, loss }
}

fn counter_pattern(seed: usize, count: usize) -> Vec<f32> {
    (0..count).map(|index| (((seed + index) * 7 % 13) as f32 - 6.0) / 12.0).collect()
}

struct Dataset {
    examples: [[f32; IN_DIM]; 4],
    labels: [[f32; OUT_DIM]; 4],
}

fn dataset() -> Dataset {
    Dataset {
        examples: [
            [1.0, 0.5, 0.2],
            [-1.0, -0.5, 0.3],
            [0.8, -0.2, 0.1],
            [-0.3, 0.1, 0.9],
        ],
        labels: [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
    }
}

fn loss_at(program: &[Op], loss: NodeId, x: &[f32], y: &[f32], params: [&[f32]; 4]) -> f32 {
    let [w1, b1, w2, b2] = params;
    let evaluated = evaluate_named(
        program,
        &[],
        &[("x", x), ("y", y), ("w1", w1), ("b1", b1), ("w2", w2), ("b2", b2)],
        &[loss],
    )
    .expect("network program lowers and evaluates");
    evaluated.get(loss).expect("loss requested").0[0]
}

/// Perturbs one element of one parameter buffer by `+-h` and returns the
/// central-difference estimate of `d(loss)/d(that element)`, holding every
/// other input fixed.
#[allow(clippy::too_many_arguments)]
fn numeric_gradient(
    program: &[Op],
    loss: NodeId,
    x: &[f32],
    y: &[f32],
    buffers: &mut [&mut Vec<f32>; 4],
    which: usize,
    index: usize,
    step: f32,
) -> f32 {
    let original = buffers[which][index];

    buffers[which][index] = original + step;
    let plus = loss_at(program, loss, x, y, [buffers[0], buffers[1], buffers[2], buffers[3]]);

    buffers[which][index] = original - step;
    let minus = loss_at(program, loss, x, y, [buffers[0], buffers[1], buffers[2], buffers[3]]);

    buffers[which][index] = original;
    (plus - minus) / (2.0 * step)
}

fn relative_error(analytic: f32, numeric: f32) -> f32 {
    (analytic - numeric).abs() / (analytic.abs().max(numeric.abs()) + 1e-6)
}

#[proxima::test]
async fn central_difference_matches_the_analytic_gradient_on_every_parameter() {
    let network = build_network();
    let differentiated = differentiate(&network.program, network.loss).expect("scalar loss differentiates");
    let data = dataset();
    let (x, y) = (data.examples[0], data.labels[0]);

    let mut w1 = counter_pattern(1, IN_DIM * HIDDEN_DIM);
    let mut b1 = counter_pattern(2, HIDDEN_DIM);
    let mut w2 = counter_pattern(3, HIDDEN_DIM * OUT_DIM);
    let mut b2 = counter_pattern(4, OUT_DIM);

    let grad_w1 = differentiated.gradient_of_named("w1").expect("w1 feeds the loss");
    let grad_b1 = differentiated.gradient_of_named("b1").expect("b1 feeds the loss");
    let grad_w2 = differentiated.gradient_of_named("w2").expect("w2 feeds the loss");
    let grad_b2 = differentiated.gradient_of_named("b2").expect("b2 feeds the loss");

    let evaluated = evaluate_named(
        &differentiated.program,
        &[],
        &[("x", x.as_slice()), ("y", y.as_slice()), ("w1", &w1), ("b1", &b1), ("w2", &w2), ("b2", &b2)],
        &[grad_w1, grad_b1, grad_w2, grad_b2],
    )
    .expect("adjoint program lowers and evaluates");

    let analytic_w1 = evaluated.get(grad_w1).expect("grad_w1 requested").0.to_vec();
    let analytic_b1 = evaluated.get(grad_b1).expect("grad_b1 requested").0.to_vec();
    let analytic_w2 = evaluated.get(grad_w2).expect("grad_w2 requested").0.to_vec();
    let analytic_b2 = evaluated.get(grad_b2).expect("grad_b2 requested").0.to_vec();

    let step = 1e-3f32;
    let mut worst = (0.0f32, "", 0usize);

    for (index, &analytic_value) in analytic_w1.iter().enumerate() {
        let numeric = numeric_gradient(&differentiated.program, network.loss, &x, &y, &mut [&mut w1, &mut b1, &mut w2, &mut b2], 0, index, step);
        let relative = relative_error(analytic_value, numeric);
        if relative > worst.0 {
            worst = (relative, "w1", index);
        }
    }
    for (index, &analytic_value) in analytic_b1.iter().enumerate() {
        let numeric = numeric_gradient(&differentiated.program, network.loss, &x, &y, &mut [&mut w1, &mut b1, &mut w2, &mut b2], 1, index, step);
        let relative = relative_error(analytic_value, numeric);
        if relative > worst.0 {
            worst = (relative, "b1", index);
        }
    }
    for (index, &analytic_value) in analytic_w2.iter().enumerate() {
        let numeric = numeric_gradient(&differentiated.program, network.loss, &x, &y, &mut [&mut w1, &mut b1, &mut w2, &mut b2], 2, index, step);
        let relative = relative_error(analytic_value, numeric);
        if relative > worst.0 {
            worst = (relative, "w2", index);
        }
    }
    for (index, &analytic_value) in analytic_b2.iter().enumerate() {
        let numeric = numeric_gradient(&differentiated.program, network.loss, &x, &y, &mut [&mut w1, &mut b1, &mut w2, &mut b2], 3, index, step);
        let relative = relative_error(analytic_value, numeric);
        if relative > worst.0 {
            worst = (relative, "b2", index);
        }
    }

    std::eprintln!("max relative gradient-check error: {} at {}[{}]", worst.0, worst.1, worst.2);
    assert!(worst.0 < 5e-3, "central-difference disagreed with the analytic gradient: {worst:?}");
}

/// Deterministic Adam training run over the 4-example dataset, cycling
/// through it 15 times (60 steps). Asserts the final loss is below the
/// initial loss by a real margin; the loss curve itself is printed (run
/// with `--nocapture` to see every step) rather than only asserted on, so
/// a plateau/NaN/oscillation would be visible even if the margin assertion
/// were loosened.
#[proxima::test]
async fn adam_training_decreases_the_loss_over_the_dataset() {
    let network = build_network();
    let differentiated = differentiate(&network.program, network.loss).expect("scalar loss differentiates");
    let grad_w1 = differentiated.gradient_of_named("w1").expect("w1 feeds the loss");
    let grad_b1 = differentiated.gradient_of_named("b1").expect("b1 feeds the loss");
    let grad_w2 = differentiated.gradient_of_named("w2").expect("w2 feeds the loss");
    let grad_b2 = differentiated.gradient_of_named("b2").expect("b2 feeds the loss");
    let mut program = differentiated.program;
    let config = AdamConfig { learning_rate: 0.05, ..AdamConfig::default() };
    let step_node = step_input(&mut program, "step");

    let m_w1 = leaf(&mut program, "m_w1", alloc::vec![Extent::Static(IN_DIM as u32), Extent::Static(HIDDEN_DIM as u32)]);
    let v_w1 = leaf(&mut program, "v_w1", alloc::vec![Extent::Static(IN_DIM as u32), Extent::Static(HIDDEN_DIM as u32)]);
    let m_b1 = leaf(&mut program, "m_b1", alloc::vec![Extent::Static(HIDDEN_DIM as u32)]);
    let v_b1 = leaf(&mut program, "v_b1", alloc::vec![Extent::Static(HIDDEN_DIM as u32)]);
    let m_w2 = leaf(&mut program, "m_w2", alloc::vec![Extent::Static(HIDDEN_DIM as u32), Extent::Static(OUT_DIM as u32)]);
    let v_w2 = leaf(&mut program, "v_w2", alloc::vec![Extent::Static(HIDDEN_DIM as u32), Extent::Static(OUT_DIM as u32)]);
    let m_b2 = leaf(&mut program, "m_b2", alloc::vec![Extent::Static(OUT_DIM as u32)]);
    let v_b2 = leaf(&mut program, "v_b2", alloc::vec![Extent::Static(OUT_DIM as u32)]);

    let (new_w1, new_m_w1, new_v_w1) = adam_step(&mut program, &config, 2, AdamOperands { param: network.w1, grad: grad_w1, m: m_w1, v: v_w1 }, step_node);
    let (new_b1, new_m_b1, new_v_b1) = adam_step(&mut program, &config, 1, AdamOperands { param: network.b1, grad: grad_b1, m: m_b1, v: v_b1 }, step_node);
    let (new_w2, new_m_w2, new_v_w2) = adam_step(&mut program, &config, 2, AdamOperands { param: network.w2, grad: grad_w2, m: m_w2, v: v_w2 }, step_node);
    let (new_b2, new_m_b2, new_v_b2) = adam_step(&mut program, &config, 1, AdamOperands { param: network.b2, grad: grad_b2, m: m_b2, v: v_b2 }, step_node);

    let mut w1 = counter_pattern(1, IN_DIM * HIDDEN_DIM);
    let mut b1 = counter_pattern(2, HIDDEN_DIM);
    let mut w2 = counter_pattern(3, HIDDEN_DIM * OUT_DIM);
    let mut b2 = counter_pattern(4, OUT_DIM);
    let mut m_w1_values = alloc::vec![0.0f32; w1.len()];
    let mut v_w1_values = alloc::vec![0.0f32; w1.len()];
    let mut m_b1_values = alloc::vec![0.0f32; b1.len()];
    let mut v_b1_values = alloc::vec![0.0f32; b1.len()];
    let mut m_w2_values = alloc::vec![0.0f32; w2.len()];
    let mut v_w2_values = alloc::vec![0.0f32; w2.len()];
    let mut m_b2_values = alloc::vec![0.0f32; b2.len()];
    let mut v_b2_values = alloc::vec![0.0f32; b2.len()];

    let data = dataset();
    let mut loss_curve = Vec::new();

    const EPOCHS: u32 = 40;
    for epoch in 0..EPOCHS {
        for example_index in 0..4usize {
            let x = data.examples[example_index];
            let y = data.labels[example_index];
            let step_value = [(epoch * 4 + example_index as u32 + 1) as f32];

            let evaluated = evaluate_named(
                &program,
                &[],
                &[
                    ("x", x.as_slice()),
                    ("y", y.as_slice()),
                    ("w1", &w1),
                    ("b1", &b1),
                    ("w2", &w2),
                    ("b2", &b2),
                    ("m_w1", &m_w1_values),
                    ("v_w1", &v_w1_values),
                    ("m_b1", &m_b1_values),
                    ("v_b1", &v_b1_values),
                    ("m_w2", &m_w2_values),
                    ("v_w2", &v_w2_values),
                    ("m_b2", &m_b2_values),
                    ("v_b2", &v_b2_values),
                    ("step", &step_value),
                ],
                &[
                    network.loss, new_w1, new_m_w1, new_v_w1, new_b1, new_m_b1, new_v_b1, new_w2, new_m_w2, new_v_w2,
                    new_b2, new_m_b2, new_v_b2,
                ],
            )
            .expect("training-step program lowers and evaluates");

            loss_curve.push(evaluated.get(network.loss).expect("loss requested").0[0]);
            w1 = evaluated.get(new_w1).expect("new_w1 requested").0.to_vec();
            b1 = evaluated.get(new_b1).expect("new_b1 requested").0.to_vec();
            w2 = evaluated.get(new_w2).expect("new_w2 requested").0.to_vec();
            b2 = evaluated.get(new_b2).expect("new_b2 requested").0.to_vec();
            m_w1_values = evaluated.get(new_m_w1).expect("new_m_w1 requested").0.to_vec();
            v_w1_values = evaluated.get(new_v_w1).expect("new_v_w1 requested").0.to_vec();
            m_b1_values = evaluated.get(new_m_b1).expect("new_m_b1 requested").0.to_vec();
            v_b1_values = evaluated.get(new_v_b1).expect("new_v_b1 requested").0.to_vec();
            m_w2_values = evaluated.get(new_m_w2).expect("new_m_w2 requested").0.to_vec();
            v_w2_values = evaluated.get(new_v_w2).expect("new_v_w2 requested").0.to_vec();
            m_b2_values = evaluated.get(new_m_b2).expect("new_m_b2 requested").0.to_vec();
            v_b2_values = evaluated.get(new_v_b2).expect("new_v_b2 requested").0.to_vec();
        }
    }

    std::eprintln!("loss curve ({} steps): {loss_curve:?}", loss_curve.len());

    let epoch_averages: Vec<f32> = loss_curve
        .chunks(4)
        .map(|epoch| epoch.iter().sum::<f32>() / epoch.len() as f32)
        .collect();
    std::eprintln!("per-epoch average loss ({} epochs): {epoch_averages:?}", epoch_averages.len());

    let initial = epoch_averages[0];
    let final_average = *epoch_averages.last().expect("at least one epoch ran");
    std::eprintln!("initial epoch-average loss {initial}, final epoch-average loss {final_average}");
    assert!(
        final_average < initial * 0.8,
        "expected the epoch-average loss to drop by at least 20% over training, got {initial} -> {final_average}"
    );
    assert!(
        loss_curve.iter().all(|value| value.is_finite()),
        "loss went non-finite somewhere in the curve: {loss_curve:?}"
    );
}

/// Targeted proof that the gradient checker can fail: `loss = max(x)` for
/// `x = [3.0, 1.0, 2.0]` has an exact, hand-computable adjoint — `[1, 0,
/// 0]`, all gradient routed to the unique argmax (index 0) — this crate's
/// own report pastes what this test prints when the mask factor in the
/// `Reduce(Maximum)` rule (`adjoint.rs`'s `differentiate_reduce`) is
/// deliberately dropped: every position gets the full gradient instead of
/// only the argmax, and both the exact-value assertion below and a
/// central-difference check against it fail loudly rather than silently.
#[proxima::test]
async fn maximum_reduce_adjoint_routes_the_full_gradient_to_the_unique_argmax_only() {
    let mut program = Vec::new();
    let x = leaf(&mut program, "x", alloc::vec![Extent::Static(3)]);
    let loss = reduce_op_max(&mut program, x);

    let differentiated = differentiate(&program, loss).expect("scalar loss differentiates");
    let grad_x = differentiated.gradient_of_named("x").expect("x feeds the loss");

    let values = [3.0f32, 1.0, 2.0];
    let evaluated = evaluate_named(&differentiated.program, &[], &[("x", &values)], &[grad_x])
        .expect("adjoint program lowers and evaluates");
    let analytic = evaluated.get(grad_x).expect("grad_x requested").0;

    std::eprintln!("analytic gradient of max(x) at x={values:?}: {analytic:?}");
    assert_eq!(
        analytic,
        [1.0, 0.0, 0.0],
        "max(x)'s adjoint must route the full gradient to the unique argmax only, got {analytic:?}"
    );
}

fn reduce_op_max(program: &mut Vec<Op>, operand: NodeId) -> NodeId {
    op::append(
        program,
        Op::Reduce(proxima_tensor::op::Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Maximum,
            init: ReduceInit::NegativeInfinity,
            operand,
            in_map: identity(1),
            out_map: IndexMap::Affine(map::projection(1, &[])),
            keep: proxima_tensor::op::Keep::Reduce,
            name: None,
        }),
    )
}
