use proxima_autograd::adjoint::differentiate;
use proxima_autograd::sparse::dedupe_and_sum_rows;
use proxima_tensor::cpu::evaluate_named;
use proxima_tensor::dtype::DType;
use proxima_tensor::map::{self, AxisIndex, AxisTerm, IndexMap, IndexPattern};
use proxima_tensor::op::{self, Extent, Keep, Op, Reduce, ReduceInit, ScalarOp};

fn one_token_sparse_moe() -> (Vec<Op>, proxima_tensor::op::NodeId) {
    let mut program = Vec::new();
    let expert_weights = op::append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(2), Extent::Static(1)],
            name: Some("experts".into()),
        },
    );
    let route_ids = op::append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(1)],
            name: Some("expert_ids".into()),
        },
    );
    let inputs = op::append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(1), Extent::Static(1)],
            name: Some("x".into()),
        },
    );
    let targets = op::append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(1), Extent::Static(1)],
            name: Some("target".into()),
        },
    );
    let gather_map = IndexMap::Computed {
        indices: route_ids,
        index_map: map::projection(2, &[0]),
        base: IndexPattern {
            iter_rank: 2,
            axes: vec![
                AxisIndex::default(),
                AxisIndex {
                    terms: core::iter::once(AxisTerm::projection(1)).collect(),
                    offset: 0,
                    len: None,
                },
            ],
        },
        gathered_dim: 0,
    };
    let selected_weights = op::append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Identity,
            operands: vec![(expert_weights, gather_map)],
            name: None,
        },
    );
    let identity_map = IndexMap::Affine(map::projection(2, &[0, 1]));
    let predictions = op::append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (inputs, identity_map.clone()),
                (selected_weights, identity_map.clone()),
            ],
            name: None,
        },
    );
    let differences = op::append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Subtract,
            operands: vec![
                (predictions, identity_map.clone()),
                (targets, identity_map.clone()),
            ],
            name: None,
        },
    );
    let squared = op::append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (differences, identity_map.clone()),
                (differences, identity_map.clone()),
            ],
            name: None,
        },
    );
    let half = op::append(
        &mut program,
        Op::Constant {
            dtype: DType::Float32,
            shape: Vec::new(),
            value: 0.5,
        },
    );
    let scaled_loss = op::append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (squared, identity_map),
                (half, IndexMap::Affine(map::projection(2, &[]))),
            ],
            name: None,
        },
    );
    let loss = op::append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: scaled_loss,
            in_map: IndexMap::Affine(map::projection(2, &[0, 1])),
            out_map: IndexMap::Affine(map::projection(2, &[])),
            keep: Keep::Reduce,
            name: None,
        }),
    );
    (program, loss)
}

#[test]
fn sparse_moe_training_step_updates_only_routed_experts() {
    let (program, loss) = one_token_sparse_moe();
    let differentiated = differentiate(&program, loss).expect("sparse MoE loss differentiates");
    let expert_table = program
        .iter()
        .position(|operation| matches!(operation, Op::Input { name: Some(name), .. } if name == "experts"))
        .map(|index| proxima_tensor::op::NodeId(index as u32))
        .expect("expert table input");
    let gathered = differentiated
        .gathered_gradients_of(expert_table)
        .next()
        .expect("gathered expert gradient contribution");
    let evaluated = evaluate_named(
        &differentiated.program,
        &[],
        &[
            ("experts", &[0.5, -0.5]),
            ("expert_ids", &[0.0]),
            ("x", &[1.0]),
            ("target", &[1.0]),
        ],
        &[loss, gathered.values],
    )
    .expect("fixed route evaluates through computed expert gather");
    assert_eq!(evaluated.get(loss).expect("loss output").0, [0.125]);

    let compact_gradient = evaluated
        .get(gathered.values)
        .expect("compact gathered gradient")
        .0;
    let route_payload = [0.0_f32];
    let (unique_experts, summed_gradients) =
        dedupe_and_sum_rows(&route_payload, compact_gradient, 1)
            .expect("one scalar contribution per routed expert");
    assert_eq!(unique_experts, [0]);
    assert_eq!(summed_gradients, [-0.5]);

    let mut dense_gradient = [0.0_f32; 2];
    for (expert_id, gradient) in unique_experts.into_iter().zip(summed_gradients) {
        dense_gradient[expert_id as usize] = gradient;
    }
    assert_eq!(dense_gradient, [-0.5, 0.0]);
}
