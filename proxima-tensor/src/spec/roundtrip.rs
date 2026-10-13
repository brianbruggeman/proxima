use conflaguration::Validate;

use super::*;

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
        DType::Float32,
        alloc::vec![Extent::Symbolic(0), Extent::Static(768)],
        "lhs",
    );
    let rhs = leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(768), Extent::Static(3072)],
        "rhs",
    );
    let product = op::append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
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
    program
}

fn embedding_gather(program: &mut Vec<Op>) {
    let table = leaf(
        program,
        DType::Float32,
        alloc::vec![Extent::Static(50000), Extent::Static(8)],
        "table",
    );
    let ids = leaf(program, DType::Int32, alloc::vec![Extent::Static(4)], "ids");
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
            dtype: DType::Float32,
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
        DType::Float32,
        alloc::vec![Extent::Static(2), Extent::Static(3)],
        "grid",
    );
    op::append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
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
            dtype: DType::Int32,
            extent: Extent::Symbolic(0),
        },
        Op::Iota {
            dtype: DType::Int32,
            extent: Extent::Static(4),
        },
    ];
    assert_eq!(round_trip_through_toml(&program), program);
}

#[test]
fn program_spec_round_trips_constant_leaves() {
    let program = alloc::vec![
        Op::Constant {
            dtype: DType::Float32,
            shape: Vec::new(),
            value: 1e-5,
        },
        Op::Constant {
            dtype: DType::Float32,
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
        DType::Float32,
        alloc::vec![Extent::Static(4)],
        "values",
    );
    let indices = leaf(
        &mut program,
        DType::Int32,
        alloc::vec![Extent::Static(4)],
        "indices",
    );
    let scattered = IndexMap::scatter(indices, map::projection(1, &[0]), 1, &[], 0, 8);
    op::append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
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
