use proxima_autograd::adjoint::differentiate;
use proxima_autograd::low_precision::{WeightFormat, weight_view};
use proxima_autograd::optimizer::{AdamConfig, AdamOperands, adam_step, step_input};
use proxima_tensor::cpu::evaluate_named;
use proxima_tensor::dtype::DType;
use proxima_tensor::map::{self, IndexMap};
use proxima_tensor::op::{self, Extent, Op, Reduce, ReduceInit, ScalarOp};

fn single_weight_squared_error() -> (Vec<Op>, proxima_tensor::op::NodeId) {
    let mut program = Vec::new();
    let input_shape = vec![Extent::Static(1)];
    let input = op::append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: input_shape.clone(),
            name: Some("x".into()),
        },
    );
    let weight = op::append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: input_shape.clone(),
            name: Some("w".into()),
        },
    );
    let target = op::append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: input_shape.clone(),
            name: Some("target".into()),
        },
    );
    let vector_map = IndexMap::Affine(map::projection(1, &[0]));
    let product = op::append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![(input, vector_map.clone()), (weight, vector_map.clone())],
            name: None,
        },
    );
    let difference = op::append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Subtract,
            operands: vec![(product, vector_map.clone()), (target, vector_map.clone())],
            name: None,
        },
    );
    let squared = op::append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (difference, vector_map.clone()),
                (difference, vector_map.clone()),
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
    let half_map = IndexMap::Affine(map::projection(1, &[]));
    let scaled = op::append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![(squared, vector_map.clone()), (half, half_map)],
            name: None,
        },
    );
    let loss = op::append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: scaled,
            in_map: vector_map,
            out_map: IndexMap::Affine(map::projection(1, &[])),
            keep: op::Keep::Reduce,
            name: None,
        }),
    );
    (program, loss)
}

#[test]
fn low_precision_weight_gradient_matches_f32_reference() {
    let (program, loss) = single_weight_squared_error();
    let mut differentiated = differentiate(&program, loss).expect("squared error differentiates");
    let weight_gradient = differentiated
        .gradient_of_named("w")
        .expect("loss depends on the weight input");
    let master_weights = [0.5_f32];
    let master = op::append(
        &mut differentiated.program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(1)],
            name: Some("master".into()),
        },
    );
    let moment_one = op::append(
        &mut differentiated.program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(1)],
            name: Some("moment_one".into()),
        },
    );
    let moment_two = op::append(
        &mut differentiated.program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(1)],
            name: Some("moment_two".into()),
        },
    );
    let step = step_input(&mut differentiated.program, "step");
    let (updated_master, updated_moment_one, updated_moment_two) = adam_step(
        &mut differentiated.program,
        &AdamConfig::default(),
        1,
        AdamOperands {
            param: master,
            grad: weight_gradient,
            m: moment_one,
            v: moment_two,
        },
        step,
    );
    for operation in &differentiated.program {
        assert_eq!(
            operation.dtype(),
            DType::Float32,
            "all forward, backward, and Adam nodes use FP32"
        );
    }

    for format in [WeightFormat::Bf8E5M2, WeightFormat::Bf4E2M1] {
        let view = weight_view(&master_weights, format).expect("finite scalar weight");
        assert_eq!(view, [0.5]);
        let evaluated = evaluate_named(
            &differentiated.program,
            &[],
            &[
                ("x", &[1.0]),
                ("w", &view),
                ("target", &[1.0]),
                ("master", &master_weights),
                ("moment_one", &[0.0]),
                ("moment_two", &[0.0]),
                ("step", &[1.0]),
            ],
            &[
                loss,
                weight_gradient,
                updated_master,
                updated_moment_one,
                updated_moment_two,
            ],
        )
        .expect("forward and backward evaluate in f32");
        assert_eq!(evaluated.get(loss).expect("loss output").0, [0.125]);
        assert_eq!(
            evaluated.get(weight_gradient).expect("weight gradient").0,
            [-0.5]
        );
        assert!(
            evaluated.get(updated_master).expect("updated master").0[0] > master_weights[0],
            "the FP32 master parameter consumes the identity-STE gradient"
        );
        let first_moment = evaluated
            .get(updated_moment_one)
            .expect("updated first moment")
            .0[0];
        let second_moment = evaluated
            .get(updated_moment_two)
            .expect("updated second moment")
            .0[0];
        assert!((first_moment - -0.05).abs() < 2.0e-8);
        assert!((second_moment - 0.00025).abs() < 2.0e-8);
    }

    assert_eq!(
        master_weights,
        [0.5],
        "the quantized view does not mutate master weights"
    );
}
