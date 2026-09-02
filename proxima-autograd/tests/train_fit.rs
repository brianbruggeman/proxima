//! Integration proof for [`proxima_autograd::train::fit`]: the same dense
//! net `training_loop.rs`'s `adam_training_decreases_the_loss_over_the_dataset`
//! trains by hand (matmul + bias + relu, matmul + bias + softmax,
//! cross-entropy), now built with [`proxima_autograd::loss::softmax_cross_entropy`]
//! instead of the inline shape, trained through `fit`'s two free functions
//! instead of the hand-written nested loop -- proof `train`/`loss` compose
//! with `activation`/`adjoint`/`optimizer` rather than duplicating them.
#![allow(clippy::unwrap_used, clippy::expect_used)]

extern crate alloc;

use proxima_autograd::activation::relu;
use proxima_autograd::adjoint::differentiate;
use proxima_autograd::loss::softmax_cross_entropy;
use proxima_autograd::optimizer::{AdamConfig, AdamOperands, adam_step, step_input};
use proxima_autograd::train::fit;
use proxima_tensor::dtype::DType;
use proxima_tensor::map::{self, IndexMap};
use proxima_tensor::op::{self, Extent, NodeId, Op, ReduceInit, ScalarOp};

const IN_DIM: usize = 3;
const HIDDEN_DIM: usize = 4;
const OUT_DIM: usize = 2;

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
        Op::Elementwise {
            dtype: DType::Float32,
            body,
            operands,
            name: None,
        },
    )
}

fn reduce_add(
    program: &mut Vec<Op>,
    operand: NodeId,
    in_map: IndexMap,
    out_map: IndexMap,
) -> NodeId {
    op::append(
        program,
        Op::Reduce(op::Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand,
            in_map,
            out_map,
            keep: op::Keep::Reduce,
            name: None,
        }),
    )
}

fn identity(rank: u16) -> IndexMap {
    IndexMap::Affine(map::projection(rank, &(0..rank).collect::<Vec<u16>>()))
}

fn dense(program: &mut Vec<Op>, x: NodeId, w: NodeId, b: NodeId) -> NodeId {
    let product = elementwise(
        program,
        ScalarOp::Multiply,
        vec![
            (w, identity(2)),
            (x, IndexMap::Affine(map::projection(2, &[0]))),
        ],
    );
    let matmul = reduce_add(
        program,
        product,
        identity(2),
        IndexMap::Affine(map::projection(2, &[1])),
    );
    elementwise(
        program,
        ScalarOp::Add,
        vec![(matmul, identity(1)), (b, identity(1))],
    )
}

struct Network {
    program: Vec<Op>,
    w1: NodeId,
    b1: NodeId,
    w2: NodeId,
    b2: NodeId,
    loss: NodeId,
}

fn build_network() -> Network {
    let mut program = Vec::new();
    let x = leaf(&mut program, "x", vec![Extent::Static(IN_DIM as u32)]);
    let y = leaf(&mut program, "y", vec![Extent::Static(OUT_DIM as u32)]);
    let w1 = leaf(
        &mut program,
        "w1",
        vec![
            Extent::Static(IN_DIM as u32),
            Extent::Static(HIDDEN_DIM as u32),
        ],
    );
    let b1 = leaf(&mut program, "b1", vec![Extent::Static(HIDDEN_DIM as u32)]);
    let w2 = leaf(
        &mut program,
        "w2",
        vec![
            Extent::Static(HIDDEN_DIM as u32),
            Extent::Static(OUT_DIM as u32),
        ],
    );
    let b2 = leaf(&mut program, "b2", vec![Extent::Static(OUT_DIM as u32)]);

    let h_pre = dense(&mut program, x, w1, b1);
    let h = relu(&mut program, DType::Float32, h_pre, 1);
    let out_pre = dense(&mut program, h, w2, b2);
    let loss = softmax_cross_entropy(&mut program, DType::Float32, out_pre, y, 1, 0);

    Network {
        program,
        w1,
        b1,
        w2,
        b2,
        loss,
    }
}

fn counter_pattern(seed: usize, count: usize) -> Vec<f32> {
    (0..count)
        .map(|index| (((seed + index) * 7 % 13) as f32 - 6.0) / 12.0)
        .collect()
}

/// Trains [`build_network`] through [`fit`] over the same 4-example dataset
/// `training_loop.rs` uses, and asserts the loss decreases -- the same
/// margin assertion, reached through the new free-function API instead of
/// the hand-written nested loop.
#[proxima::test]
async fn fit_trains_the_same_mlp_training_loop_rs_trains_by_hand() {
    let network = build_network();
    let differentiated =
        differentiate(&network.program, network.loss).expect("scalar loss differentiates");
    let grad_w1 = differentiated
        .gradient_of_named("w1")
        .expect("w1 feeds the loss");
    let grad_b1 = differentiated
        .gradient_of_named("b1")
        .expect("b1 feeds the loss");
    let grad_w2 = differentiated
        .gradient_of_named("w2")
        .expect("w2 feeds the loss");
    let grad_b2 = differentiated
        .gradient_of_named("b2")
        .expect("b2 feeds the loss");
    let mut program = differentiated.program;
    let config = AdamConfig {
        learning_rate: 0.05,
        ..AdamConfig::default()
    };
    let step_node = step_input(&mut program, "step");

    let m_w1 = leaf(
        &mut program,
        "m_w1",
        vec![
            Extent::Static(IN_DIM as u32),
            Extent::Static(HIDDEN_DIM as u32),
        ],
    );
    let v_w1 = leaf(
        &mut program,
        "v_w1",
        vec![
            Extent::Static(IN_DIM as u32),
            Extent::Static(HIDDEN_DIM as u32),
        ],
    );
    let m_b1 = leaf(
        &mut program,
        "m_b1",
        vec![Extent::Static(HIDDEN_DIM as u32)],
    );
    let v_b1 = leaf(
        &mut program,
        "v_b1",
        vec![Extent::Static(HIDDEN_DIM as u32)],
    );
    let m_w2 = leaf(
        &mut program,
        "m_w2",
        vec![
            Extent::Static(HIDDEN_DIM as u32),
            Extent::Static(OUT_DIM as u32),
        ],
    );
    let v_w2 = leaf(
        &mut program,
        "v_w2",
        vec![
            Extent::Static(HIDDEN_DIM as u32),
            Extent::Static(OUT_DIM as u32),
        ],
    );
    let m_b2 = leaf(&mut program, "m_b2", vec![Extent::Static(OUT_DIM as u32)]);
    let v_b2 = leaf(&mut program, "v_b2", vec![Extent::Static(OUT_DIM as u32)]);

    let (new_w1, new_m_w1, new_v_w1) = adam_step(
        &mut program,
        &config,
        2,
        AdamOperands {
            param: network.w1,
            grad: grad_w1,
            m: m_w1,
            v: v_w1,
        },
        step_node,
    );
    let (new_b1, new_m_b1, new_v_b1) = adam_step(
        &mut program,
        &config,
        1,
        AdamOperands {
            param: network.b1,
            grad: grad_b1,
            m: m_b1,
            v: v_b1,
        },
        step_node,
    );
    let (new_w2, new_m_w2, new_v_w2) = adam_step(
        &mut program,
        &config,
        2,
        AdamOperands {
            param: network.w2,
            grad: grad_w2,
            m: m_w2,
            v: v_w2,
        },
        step_node,
    );
    let (new_b2, new_m_b2, new_v_b2) = adam_step(
        &mut program,
        &config,
        1,
        AdamOperands {
            param: network.b2,
            grad: grad_b2,
            m: m_b2,
            v: v_b2,
        },
        step_node,
    );

    let rebind: Vec<(NodeId, &str)> = vec![
        (new_w1, "w1"),
        (new_m_w1, "m_w1"),
        (new_v_w1, "v_w1"),
        (new_b1, "b1"),
        (new_m_b1, "m_b1"),
        (new_v_b1, "v_b1"),
        (new_w2, "w2"),
        (new_m_w2, "m_w2"),
        (new_v_w2, "v_w2"),
        (new_b2, "b2"),
        (new_m_b2, "m_b2"),
        (new_v_b2, "v_b2"),
    ];

    let w1 = counter_pattern(1, IN_DIM * HIDDEN_DIM);
    let b1 = counter_pattern(2, HIDDEN_DIM);
    let w2 = counter_pattern(3, HIDDEN_DIM * OUT_DIM);
    let b2 = counter_pattern(4, OUT_DIM);
    let zeros = |count: usize| alloc::vec![0.0f32; count];

    let initial_state: Vec<(alloc::string::String, Vec<f32>)> = alloc::vec![
        ("w1".into(), w1),
        ("m_w1".into(), zeros(IN_DIM * HIDDEN_DIM)),
        ("v_w1".into(), zeros(IN_DIM * HIDDEN_DIM)),
        ("b1".into(), b1),
        ("m_b1".into(), zeros(HIDDEN_DIM)),
        ("v_b1".into(), zeros(HIDDEN_DIM)),
        ("w2".into(), w2),
        ("m_w2".into(), zeros(HIDDEN_DIM * OUT_DIM)),
        ("v_w2".into(), zeros(HIDDEN_DIM * OUT_DIM)),
        ("b2".into(), b2),
        ("m_b2".into(), zeros(OUT_DIM)),
        ("v_b2".into(), zeros(OUT_DIM)),
    ];

    let examples: [[f32; IN_DIM]; 4] = [
        [1.0, 0.5, 0.2],
        [-1.0, -0.5, 0.3],
        [0.8, -0.2, 0.1],
        [-0.3, 0.1, 0.9],
    ];
    let labels: [[f32; OUT_DIM]; 4] = [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]];
    let steps: Vec<[f32; 1]> = (1..=examples.len() as u32)
        .map(|value| [value as f32])
        .collect();
    let batches: Vec<Vec<(&str, &[f32])>> = (0..examples.len())
        .map(|index| {
            vec![
                ("x", examples[index].as_slice()),
                ("y", labels[index].as_slice()),
                ("step", steps[index].as_slice()),
            ]
        })
        .collect();

    const EPOCHS: u32 = 40;
    let (_final_state, loss_curve) = fit(
        &program,
        network.loss,
        &rebind,
        initial_state,
        EPOCHS,
        &batches,
    )
    .expect("fit runs to completion");

    std::eprintln!(
        "fit loss curve ({} steps): {loss_curve:?}",
        loss_curve.len()
    );
    assert_eq!(loss_curve.len(), EPOCHS as usize * batches.len());

    let epoch_averages: Vec<f32> = loss_curve
        .chunks(batches.len())
        .map(|epoch| epoch.iter().sum::<f32>() / epoch.len() as f32)
        .collect();
    let initial = epoch_averages[0];
    let final_average = *epoch_averages.last().expect("at least one epoch ran");
    std::eprintln!(
        "initial epoch-average loss {initial}, final epoch-average loss {final_average}"
    );

    assert!(
        final_average < initial * 0.8,
        "expected the epoch-average loss to drop by at least 20% over training via fit, got {initial} -> {final_average}"
    );
    assert!(
        loss_curve.iter().all(|value| value.is_finite()),
        "loss went non-finite: {loss_curve:?}"
    );
}
