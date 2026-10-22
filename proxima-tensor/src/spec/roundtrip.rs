use conflaguration::Validate;

use super::*;
use crate::test_support::Lcg;

fn round_trip_through_toml(program: &[Op]) -> Vec<Op> {
    let spec = ProgramSpec::try_from(program).expect("program built from ops has a spec form");
    let text = toml::to_string(&spec).expect("spec serialises to toml");
    let parsed: ProgramSpec = toml::from_str(&text).expect("serialised spec parses back");
    parsed
        .validate()
        .expect("round-tripped spec is structurally sound");
    Vec::<Op>::try_from(&parsed).expect("parsed spec lowers back to ops")
}

fn leaf(program: &mut Vec<Op>, dtype: DType, shape: Vec<Extent>, name: &str) -> NodeId {
    op::append(
        program,
        Op::Input {
            dtype,
            shape,
            name: Some(name.into()),
        },
    )
}

fn matmul_program() -> Vec<Op> {
    let mut program = Vec::new();
    let lhs = leaf(
        &mut program,
        DType::FLOAT32,
        alloc::vec![Extent::Symbolic(0), Extent::Static(768)],
        "lhs",
    );
    let rhs = leaf(
        &mut program,
        DType::FLOAT32,
        alloc::vec![Extent::Static(768), Extent::Static(3072)],
        "rhs",
    );
    let product = op::append(
        &mut program,
        Op::Elementwise {
            dtype: DType::FLOAT32,
            body: ScalarOp::Multiply,
            operands: alloc::vec![
                (lhs, IndexMap::Affine(map::projection(3, &[0, 2]))),
                (rhs, IndexMap::Affine(map::projection(3, &[2, 1]))),
            ],
            name: None,
        },
    );
    op::append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::FLOAT32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
            keep: Keep::Reduce,
            name: Some("matmul".into()),
        }),
    );
    program
}

fn embedding_gather(program: &mut Vec<Op>) {
    let table = leaf(
        program,
        DType::FLOAT32,
        alloc::vec![Extent::Static(50000), Extent::Static(8)],
        "table",
    );
    let ids = leaf(program, DType::INT32, alloc::vec![Extent::Static(4)], "ids");
    let gathered = IndexMap::Computed {
        indices: ids,
        index_map: map::projection(2, &[0]),
        base: IndexPattern {
            iter_rank: 2,
            axes: alloc::vec![
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
    op::append(
        program,
        Op::Elementwise {
            dtype: DType::FLOAT32,
            body: ScalarOp::Identity,
            operands: alloc::vec![(table, gathered)],
            name: None,
        },
    );
}

#[test]
fn program_spec_round_trips_input_leaves() {
    let program = matmul_program();
    let inputs: Vec<Op> = program.iter().take(2).cloned().collect();
    assert_eq!(inputs.len(), 2, "matmul program starts with two leaves");
    assert_eq!(round_trip_through_toml(&inputs), inputs);
}

#[test]
fn program_spec_round_trips_elementwise_projections_and_gathers() {
    let mut program = Vec::new();
    embedding_gather(&mut program);
    let table_t = leaf(
        &mut program,
        DType::FLOAT32,
        alloc::vec![Extent::Static(2), Extent::Static(3)],
        "grid",
    );
    op::append(
        &mut program,
        Op::Elementwise {
            dtype: DType::FLOAT32,
            body: ScalarOp::Identity,
            operands: alloc::vec![(table_t, IndexMap::Affine(map::projection(2, &[1, 0])))],
            name: Some("transpose".into()),
        },
    );
    assert_eq!(round_trip_through_toml(&program), program);
}

#[test]
fn program_spec_round_trips_reduce_contraction() {
    let program = matmul_program();
    assert!(
        matches!(program.last(), Some(Op::Reduce(_))),
        "matmul program ends in a reduce"
    );
    assert_eq!(round_trip_through_toml(&program), program);
}

#[test]
fn program_spec_round_trips_iota_leaves() {
    let program = alloc::vec![
        Op::Iota {
            dtype: DType::INT32,
            extent: Extent::Symbolic(0),
        },
        Op::Iota {
            dtype: DType::INT32,
            extent: Extent::Static(4),
        },
    ];
    assert_eq!(round_trip_through_toml(&program), program);
}

#[test]
fn program_spec_round_trips_constant_leaves() {
    let program = alloc::vec![
        Op::Constant {
            dtype: DType::FLOAT32,
            shape: Vec::new(),
            value: 1e-5,
        },
        Op::Constant {
            dtype: DType::FLOAT32,
            shape: alloc::vec![Extent::Static(2), Extent::Static(4)],
            value: -0.5,
        },
    ];
    assert_eq!(round_trip_through_toml(&program), program);
}

#[test]
fn program_spec_rejects_data_dependent_reduce_with_named_error() {
    let mut program = Vec::new();
    let values = leaf(
        &mut program,
        DType::FLOAT32,
        alloc::vec![Extent::Static(4)],
        "values",
    );
    let indices = leaf(
        &mut program,
        DType::INT32,
        alloc::vec![Extent::Static(4)],
        "indices",
    );
    let scattered = IndexMap::scatter(indices, map::projection(1, &[0]), 1, &[], 0, 8);
    op::append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::FLOAT32,
            body: ScalarOp::Identity,
            init: ReduceInit::Zero,
            operand: values,
            in_map: IndexMap::Affine(map::projection(1, &[0])),
            out_map: scattered,
            keep: Keep::Reduce,
            name: None,
        }),
    );
    let error = ProgramSpec::try_from(program.as_slice())
        .expect_err("a scatter has no spec form and must be rejected");
    assert!(
        matches!(
            error,
            TensorError::NoConfigFace {
                node: NodeId(2),
                ..
            }
        ),
        "error must name the scatter reduce at node 2, got {error}"
    );
}

const MATMUL_SILU_SOFTMAX_TOML: &str = include_str!("fixtures/matmul_silu_softmax.toml");

fn spec_input(id: &str, name: &str, shape: &[u32]) -> NodeSpec {
    NodeSpec::Input {
        id: id.into(),
        dtype: DType::FLOAT32,
        shape: shape.iter().copied().map(ExtentSpec::Static).collect(),
        name: Some(name.into()),
    }
}

fn spec_elementwise(
    id: &str,
    body: ScalarOp,
    operands: &[(&str, &str)],
    name: Option<&str>,
) -> NodeSpec {
    NodeSpec::Elementwise {
        id: id.into(),
        dtype: DType::FLOAT32,
        body,
        inputs: operands.iter().map(|(node, _)| (*node).into()).collect(),
        maps: operands
            .iter()
            .map(|(_, notation)| MapSpec::Projection((*notation).into()))
            .collect(),
        name: name.map(Into::into),
    }
}

fn spec_reduce(
    id: &str,
    body: ScalarOp,
    init: ReduceInit,
    input: &str,
    out_map: &str,
    name: &str,
) -> NodeSpec {
    NodeSpec::Reduce {
        id: id.into(),
        dtype: DType::FLOAT32,
        body,
        init,
        input: input.into(),
        in_map: "ab->ab".into(),
        out_map: out_map.into(),
        keep: Keep::Reduce,
        name: Some(name.into()),
    }
}

fn matmul_silu_softmax_builder_spec() -> ProgramSpec {
    let matmul_nodes = [
        spec_input("n0", "x", &[2, 4]),
        spec_input("n1", "w", &[4, 3]),
        spec_elementwise(
            "n2",
            ScalarOp::Multiply,
            &[("n0", "ac->abc"), ("n1", "cb->abc")],
            None,
        ),
        NodeSpec::Reduce {
            id: "n3".into(),
            dtype: DType::FLOAT32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            input: "n2".into(),
            in_map: "abc->abc".into(),
            out_map: "ab->abc".into(),
            keep: Keep::Reduce,
            name: Some("matmul".into()),
        },
    ];
    let silu_nodes = [
        spec_elementwise("n4", ScalarOp::Negate, &[("n3", "ab->ab")], None),
        spec_elementwise("n5", ScalarOp::Exponential, &[("n4", "ab->ab")], None),
        NodeSpec::Constant {
            id: "n6".into(),
            dtype: DType::FLOAT32,
            shape: Vec::new(),
            value: 1.0,
        },
        spec_elementwise(
            "n7",
            ScalarOp::Add,
            &[("n5", "ab->ab"), ("n6", "->ab")],
            None,
        ),
        spec_elementwise(
            "n8",
            ScalarOp::Divide,
            &[("n3", "ab->ab"), ("n7", "ab->ab")],
            Some("silu"),
        ),
    ];
    let softmax_nodes = [
        spec_reduce(
            "n9",
            ScalarOp::Maximum,
            ReduceInit::NegativeInfinity,
            "n8",
            "a->ab",
            "row_max",
        ),
        spec_elementwise(
            "n10",
            ScalarOp::Subtract,
            &[("n8", "ab->ab"), ("n9", "a->ab")],
            None,
        ),
        spec_elementwise("n11", ScalarOp::Exponential, &[("n10", "ab->ab")], None),
        spec_reduce(
            "n12",
            ScalarOp::Add,
            ReduceInit::Zero,
            "n11",
            "a->ab",
            "row_sum",
        ),
        spec_elementwise(
            "n13",
            ScalarOp::Divide,
            &[("n11", "ab->ab"), ("n12", "a->ab")],
            Some("softmax"),
        ),
    ];
    ProgramSpec::builder()
        .node(
            matmul_nodes
                .into_iter()
                .chain(silu_nodes)
                .chain(softmax_nodes)
                .collect(),
        )
        .outputs(vec![OutputSpec {
            name: "softmax".into(),
            node: "n13".into(),
        }])
        .binds(vec![
            BindSpec {
                name: "x".into(),
                node: "n0".into(),
            },
            BindSpec {
                name: "w".into(),
                node: "n1".into(),
            },
        ])
        .build()
}

#[test]
fn plan_envelope_toml_fixture_equals_builder_built_spec() {
    let parsed: ProgramSpec =
        toml::from_str(MATMUL_SILU_SOFTMAX_TOML).expect("matmul-silu-softmax fixture parses");
    let built = matmul_silu_softmax_builder_spec();
    assert_eq!(
        parsed.node.len(),
        14,
        "fixture declares every node of the program"
    );
    assert_eq!(parsed, built);
    parsed.validate().expect("fixture is a sound plan");
}

#[test]
fn plan_envelope_rejects_unsupported_version_with_named_error() {
    let text = MATMUL_SILU_SOFTMAX_TOML.replacen("version = 1", "version = 2", 1);
    let parsed: ProgramSpec = toml::from_str(&text).expect("version 2 still parses");
    let error = parsed
        .validate()
        .expect_err("version 2 is not a plan this build reads");
    assert!(
        error.to_string().contains("unsupported plan version 2"),
        "error names the rejected version: {error}"
    );
}

#[test]
fn plan_envelope_rejects_unknown_top_level_field() {
    let text = format!("bogus_field = true\n{MATMUL_SILU_SOFTMAX_TOML}");
    let error = toml::from_str::<ProgramSpec>(&text)
        .expect_err("deny_unknown_fields refuses a field this build does not know");
    assert!(
        error.to_string().contains("unknown field `bogus_field`"),
        "error names the unknown field: {error}"
    );
}

#[test]
fn fluent_and_toml_built_matmul_silu_softmax_run_bit_identical() {
    let fluent = matmul_silu_softmax_builder_spec();
    let text = toml::to_string(&fluent).expect("fluent spec serialises to toml");
    let parsed: ProgramSpec = toml::from_str(&text).expect("serialised spec parses back");
    parsed
        .validate()
        .expect("round-tripped spec is a sound plan");

    let fluent_program = Vec::<Op>::try_from(&fluent).expect("fluent spec lowers");
    let toml_program = Vec::<Op>::try_from(&parsed).expect("toml spec lowers");
    assert_eq!(
        toml_program, fluent_program,
        "the toml-loaded plan must lower to the same op vector as the fluent one"
    );

    let mut lcg = Lcg(0x5eed_0003);
    let x: Vec<f32> = (0..8).map(|_| lcg.next_unit() - 0.5).collect();
    let w: Vec<f32> = (0..12).map(|_| lcg.next_unit() - 0.5).collect();
    let root = NodeId(
        u32::try_from(fluent_program.len() - 1).expect("program length fits a node id"),
    );
    let run = |program: &[Op]| {
        crate::cpu::evaluate_named(program, &[], &[("x", &x), ("w", &w)], &[root])
            .expect("matmul-silu-softmax evaluates on cpu")
    };
    let fluent_run = run(&fluent_program);
    let toml_run = run(&toml_program);
    let (fluent_values, fluent_shape) = fluent_run.get(root).expect("fluent output present");
    let (toml_values, toml_shape) = toml_run.get(root).expect("toml output present");

    assert_eq!(fluent_shape, [2, 3], "output is the 2x3 product shape");
    assert_eq!(toml_shape, fluent_shape, "both runs report the same shape");
    assert_eq!(fluent_values.len(), 6, "six output elements are compared");
    let max_abs_diff = fluent_values
        .iter()
        .zip(toml_values)
        .map(|(fluent_value, toml_value)| (fluent_value - toml_value).abs())
        .fold(0.0_f32, f32::max);
    assert_eq!(
        max_abs_diff, 0.0,
        "fluent and toml plans must run bit-identically: {fluent_values:?} vs {toml_values:?}"
    );
}
