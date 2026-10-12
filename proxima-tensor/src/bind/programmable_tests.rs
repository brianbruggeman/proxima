use super::*;
use crate::config::TensorExecutionConfig;
use crate::cpu::{Evaluated, QuantizedBlock, evaluate_quantized_named_with_config};
use crate::map;
use crate::op::{Extent, append};

fn input(program: &mut Vec<Op>, name: &str, extents: &[u32]) -> NodeId {
    append(
        program,
        Op::Input {
            dtype: DType::Float32,
            shape: extents
                .iter()
                .map(|extent| Extent::Static(*extent))
                .collect(),
            name: Some(name.into()),
        },
    )
}

fn chain() -> (Vec<Op>, NodeId, NodeId) {
    let mut program = Vec::new();
    let source = input(&mut program, "source", &[2]);
    let ones = input(&mut program, "ones", &[2]);
    let fours = input(&mut program, "fours", &[2]);
    let summed = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            operands: vec![
                (source, IndexMap::Affine(map::projection(1, &[0]))),
                (ones, IndexMap::Affine(map::projection(1, &[0]))),
            ],
            name: None,
        },
    );
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (summed, IndexMap::Affine(map::projection(1, &[0]))),
                (fours, IndexMap::Affine(map::projection(1, &[0]))),
            ],
            name: None,
        },
    );
    (program, summed, product)
}

fn evaluate_chain(program: &[Op], outputs: &[NodeId], allow_fusion: bool) -> Evaluated {
    let source = [2.0, 3.0];
    let ones = [1.0, 1.0];
    let fours = [4.0, 4.0];
    let named = [
        ("source", QuantizedBlock::Float32(&source)),
        ("ones", QuantizedBlock::Float32(&ones)),
        ("fours", QuantizedBlock::Float32(&fours)),
    ];
    let config = TensorExecutionConfig::builder()
        .allow_fusion(allow_fusion)
        .exact_activations(true)
        .build();
    evaluate_quantized_named_with_config(
        program,
        &[],
        &named,
        outputs,
        (&mut Vec::new(), &mut None),
        None,
        &config,
    )
    .unwrap_or_else(|error| panic!("the named affine chain must evaluate: {error}"))
}

fn mismatches(expected: &[f32], observed: &[f32]) -> Vec<usize> {
    assert_eq!(
        expected.len(),
        observed.len(),
        "the comparator requires equal coordinate counts"
    );
    expected
        .iter()
        .zip(observed)
        .enumerate()
        .filter_map(|(index, (expected, observed))| {
            (expected.to_bits() != observed.to_bits()).then_some(index)
        })
        .collect()
}

#[test]
fn programmable_binding_unfused_chain() {
    let (program, summed, product) = chain();
    let shapes = shape::infer(&program, &[])
        .unwrap_or_else(|error| panic!("chain shapes must resolve: {error}"));
    let bound = bind_unfused(&program, &shapes, &[product], NumericPolicy::bit_exact())
        .unwrap_or_else(|error| panic!("unfused chain must bind: {error}"));
    assert_eq!(
        bound.iter().map(|bound| bound.node).collect::<Vec<_>>(),
        [summed, product]
    );
    assert_eq!(bound[0].element_body().steps[0].op, ScalarOp::Add);
    assert_eq!(bound[1].element_body().steps[0].op, ScalarOp::Multiply);
    assert_eq!(
        bound[1]
            .operands()
            .iter()
            .map(|operand| operand.0)
            .collect::<Vec<_>>(),
        [NodeId(2), summed]
    );
    assert_eq!(
        mismatches(
            &[12.0, 16.0],
            evaluate_chain(&program, &[product], false).root()
        ),
        Vec::<usize>::new()
    );
}

#[test]
fn programmable_binding_fused_chain() {
    let (program, summed, product) = chain();
    let shapes = shape::infer(&program, &[])
        .unwrap_or_else(|error| panic!("chain shapes must resolve: {error}"));
    let bound = bind_with_fusion(
        &program,
        &shapes,
        &[product],
        true,
        NumericPolicy::bit_exact(),
    )
    .unwrap_or_else(|error| panic!("fused chain must bind: {error}"));
    assert!(bound.iter().all(|bound| bound.node != summed));
    let output = bound
        .iter()
        .find(|bound| bound.node == product)
        .unwrap_or_else(|| panic!("requested product must remain"));
    assert_eq!(
        output
            .element_body()
            .steps
            .iter()
            .map(|step| step.op)
            .collect::<Vec<_>>(),
        [ScalarOp::Add, ScalarOp::Multiply]
    );
    assert!(
        output.element_body().steps[1]
            .args
            .contains(&StepArg::Step(0))
    );
    assert_eq!(
        output
            .operands()
            .iter()
            .map(|operand| operand.0)
            .collect::<Vec<_>>(),
        [NodeId(2), NodeId(0), NodeId(1)]
    );
    assert_eq!(
        output.element_body().steps[0].args.as_slice(),
        &[StepArg::Operand(1), StepArg::Operand(2)]
    );
    assert_eq!(
        output.element_body().steps[1].args.as_slice(),
        &[StepArg::Step(0), StepArg::Operand(0)]
    );
    assert_eq!(
        mismatches(
            &[12.0, 16.0],
            evaluate_chain(&program, &[product], true).root()
        ),
        Vec::<usize>::new()
    );
}

#[test]
fn programmable_binding_unfused_projection() {
    let mut packed_program = Vec::new();
    let activation = input(&mut packed_program, "activation", &[2, 2]);
    let weight = input(&mut packed_program, "weight", &[2, 2]);
    let packed_product = append(
        &mut packed_program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (activation, IndexMap::Affine(map::projection(3, &[0, 1]))),
                (weight, IndexMap::Affine(map::projection(3, &[1, 2]))),
            ],
            name: None,
        },
    );
    let packed_output = append(
        &mut packed_program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: packed_product,
            in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(map::projection(3, &[0, 2])),
            keep: Keep::Reduce,
            name: None,
        }),
    );
    let packed_weights = [0x80, 0x3f, 0x00, 0x40, 0x40, 0x40, 0x80, 0x40];
    let packed_named = [
        ("activation", QuantizedBlock::Float32(&[5.0, 6.0, 7.0, 8.0])),
        (
            "weight",
            QuantizedBlock::Packed {
                codec: proxima_primitives::Codec::BFloat16,
                bytes: &packed_weights,
            },
        ),
    ];
    for allow_fusion in [false, true] {
        let config = TensorExecutionConfig::builder()
            .allow_fusion(allow_fusion)
            .exact_activations(true)
            .build();
        let result = evaluate_quantized_named_with_config(
            &packed_program,
            &[],
            &packed_named,
            &[packed_output],
            (&mut Vec::new(), &mut None),
            None,
            &config,
        )
        .unwrap_or_else(|error| {
            panic!("native BF16 projection must execute in both fusion modes: {error}")
        });
        assert_eq!(
            result.root(),
            &[17.0, 39.0, 23.0, 53.0],
            "fusion permission {allow_fusion}"
        );
    }
    let activation_layout = Layout {
        base: 0,
        strides: smallvec::smallvec![3, 1, 0],
    };
    assert_eq!(
        native_packed_layout(&[2, 3, 4], &[0, 2], &activation_layout)
            .strides
            .as_slice(),
        &[3, 1, 0]
    );
    let weight_layout = Layout {
        base: 0,
        strides: smallvec::smallvec![0, 4, 1],
    };
    assert_eq!(
        native_packed_layout(&[2, 3, 4], &[0, 2], &weight_layout)
            .strides
            .as_slice(),
        &[0, 1, 3]
    );
    let mut program = Vec::new();
    let source = input(&mut program, "source", &[2]);
    let weights = input(&mut program, "weights", &[3, 2]);
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (source, IndexMap::Affine(map::projection(2, &[1]))),
                (weights, IndexMap::Affine(map::projection(2, &[0, 1]))),
            ],
            name: None,
        },
    );
    let output = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(map::projection(2, &[0, 1])),
            out_map: IndexMap::Affine(map::projection(2, &[0])),
            keep: Keep::Reduce,
            name: None,
        }),
    );
    let shapes = shape::infer(&program, &[])
        .unwrap_or_else(|error| panic!("projection shapes must resolve: {error}"));
    let bound = bind_unfused(&program, &shapes, &[output], NumericPolicy::bit_exact())
        .unwrap_or_else(|error| panic!("projection must bind: {error}"));
    assert_eq!(
        bound.iter().map(|bound| bound.node).collect::<Vec<_>>(),
        [product, output]
    );
    assert_eq!(bound[0].element_body().steps[0].op, ScalarOp::Multiply);
    assert_eq!(bound[1].operands()[0].0, product);
    let named = [
        ("source", QuantizedBlock::Float32(&[1.0, 2.0])),
        (
            "weights",
            QuantizedBlock::Float32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        ),
    ];
    let config = TensorExecutionConfig::builder().allow_fusion(false).build();
    let evaluated = evaluate_quantized_named_with_config(
        &program,
        &[],
        &named,
        &[output],
        (&mut Vec::new(), &mut None),
        None,
        &config,
    )
    .unwrap_or_else(|error| panic!("affine projection must execute: {error}"));
    assert_eq!(
        mismatches(&[5.0, 11.0, 17.0], evaluated.root()),
        Vec::<usize>::new()
    );
}

#[test]
fn programmable_binding_requested_intermediate() {
    let (program, summed, product) = chain();
    for allow_fusion in [false, true] {
        let evaluated = evaluate_chain(&program, &[summed, product], allow_fusion);
        assert_eq!(
            evaluated
                .get(summed)
                .unwrap_or_else(|| panic!("requested sum must remain"))
                .0,
            &[3.0, 4.0]
        );
        assert_eq!(
            evaluated
                .get(product)
                .unwrap_or_else(|| panic!("requested product must remain"))
                .0,
            &[12.0, 16.0]
        );
    }
}

#[test]
fn programmable_binding_unreachable_branch() {
    let (mut program, _, product) = chain();
    let unused = append(
        &mut program,
        Op::Constant {
            dtype: DType::Float32,
            shape: vec![Extent::Static(2)],
            value: 99.0,
        },
    );
    let shapes = shape::infer(&program, &[])
        .unwrap_or_else(|error| panic!("the unused branch must be well formed: {error}"));
    let bound = bind_unfused(&program, &shapes, &[product], NumericPolicy::bit_exact())
        .unwrap_or_else(|error| panic!("requested reachable graph must bind: {error}"));
    assert!(bound.iter().all(|bound| bound.node != unused));
    assert!(bound.iter().any(|bound| bound.node == product));
    assert_eq!(
        evaluate_chain(&program, &[product], false)
            .get(product)
            .unwrap_or_else(|| panic!("requested product must be present"))
            .0,
        &[12.0, 16.0]
    );
}

#[test]
fn programmable_binding_signed_zero_permission() {
    let mut program = Vec::new();
    let source = input(&mut program, "source", &[1]);
    let zero = append(
        &mut program,
        Op::Constant {
            dtype: DType::Float32,
            shape: vec![Extent::Static(1)],
            value: 0.0,
        },
    );
    let output = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            operands: vec![
                (source, IndexMap::Affine(map::projection(1, &[0]))),
                (zero, IndexMap::Affine(map::projection(1, &[0]))),
            ],
            name: None,
        },
    );
    let shapes = shape::infer(&program, &[])
        .unwrap_or_else(|error| panic!("signed zero shapes must resolve: {error}"));
    let bound = bind_unfused(&program, &shapes, &[output], NumericPolicy::bit_exact())
        .unwrap_or_else(|error| panic!("signed zero graph must bind: {error}"));
    assert_eq!(
        bound
            .iter()
            .find(|bound| bound.node == output)
            .unwrap_or_else(|| panic!("addition must remain"))
            .element_body()
            .steps[0]
            .op,
        ScalarOp::Add
    );
    let named = [("source", QuantizedBlock::Float32(&[-0.0]))];
    let config = TensorExecutionConfig::builder().allow_fusion(false).build();
    let evaluated = evaluate_quantized_named_with_config(
        &program,
        &[],
        &named,
        &[output],
        (&mut Vec::new(), &mut None),
        None,
        &config,
    )
    .unwrap_or_else(|error| panic!("signed zero addition must execute: {error}"));
    assert_eq!(evaluated.root()[0].to_bits(), 0.0_f32.to_bits());
}

#[test]
fn programmable_binding_oracle_self_control() {
    assert_eq!(
        mismatches(&[12.0, 16.0], &[12.0, 16.0]),
        Vec::<usize>::new()
    );
}

#[test]
fn programmable_binding_oracle_corrupt_control() {
    assert_eq!(mismatches(&[12.0, 16.0], &[13.0, 16.0]), [0]);
}
