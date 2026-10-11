//! The MIL census classifies every node of a lowered program exactly once.
//! The scalar table is checked against the compiler-verified probe recorded in
//! `tests/fixtures/coreml-lane/mil_ops_probe.txt`; the forms are checked on
//! programs built from the algebra's own constructors and on the real bge-small
//! and mnist checkpoints.

#![cfg(feature = "mil-writer")]
// fixtures are hand-built to succeed; an expect failure here IS the test failing.
#![allow(clippy::unwrap_used, clippy::expect_used)]

// the shared bge fixture serves several binaries and this one reads only the model bytes.
#[allow(dead_code)]
#[path = "support/bge.rs"]
mod bge;

use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result, ensure};
use omega::door::{Pins, Reader};
use omega::onnx_door::OnnxReader;
use proxima_mil::census::{MilCensus, MilForm, Refusal, SCALAR_OPS, census, scalar_form};
use proxima_tensor::{
    AxisTerm, DType, Extent, IndexMap, Keep, NodeId, Op, Reduce, ReduceInit, ScalarOp, affine,
    append, projection,
};

const BGE_MATMUL_NODES: usize = 96;
const MNIST_MODEL: &str =
    "/Users/brianbruggeman/repos/others/burn/examples/onnx-inference/src/model/mnist.onnx";

fn fixture(name: &str) -> Result<String> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/coreml-lane")
        .join(name);
    fs::read_to_string(&path).with_context(|| format!("read fixture {}", path.display()))
}

fn table() -> Vec<(ScalarOp, MilForm)> {
    SCALAR_OPS
        .iter()
        .map(|body| (*body, scalar_form(*body)))
        .collect()
}

fn verdict_of(probe: &str, operator: &str) -> Vec<String> {
    let wanted = format!("operator={operator} ");
    probe
        .lines()
        .filter(|line| line.starts_with("probe ") && line.contains(&wanted))
        .filter_map(|line| {
            line.split_whitespace()
                .find_map(|field| field.strip_prefix("verdict="))
        })
        .map(str::to_owned)
        .collect()
}

fn compiled(probe: &str, operator: &str) -> bool {
    verdict_of(probe, operator)
        .iter()
        .any(|verdict| verdict == "compiled")
}

fn native_name_of_composition(body: ScalarOp) -> Option<&'static str> {
    match body {
        ScalarOp::Negate => Some("neg"),
        _ => None,
    }
}

fn table_matches_probe(rows: &[(ScalarOp, MilForm)], probe: &str) -> bool {
    rows.len() == SCALAR_OPS.len()
        && rows.iter().all(|(body, form)| {
            form.operators().iter().all(|name| compiled(probe, name))
                && match (form, native_name_of_composition(*body)) {
                    (MilForm::Composition(_), Some(native)) => {
                        verdict_of(probe, native) == ["unknown_operator"]
                    }
                    (MilForm::Operator(_), None) => true,
                    _ => false,
                }
        })
}

fn render_probe(rows: &[(ScalarOp, MilForm)]) -> String {
    rows.iter()
        .flat_map(|(body, form)| {
            let native = native_name_of_composition(*body)
                .map(|name| format!("probe operator={name} verdict=unknown_operator"));
            let used = form
                .operators()
                .iter()
                .map(|name| format!("probe operator={name} verdict=compiled emitted=1 compiled=1"));
            used.chain(native).collect::<Vec<_>>()
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn input(dtype: DType, shape: &[u32]) -> Op {
    Op::Input {
        dtype,
        shape: shape.iter().map(|extent| Extent::Static(*extent)).collect(),
        name: None,
    }
}

fn reduce_program(body: ScalarOp, init: ReduceInit, keep: Keep) -> Vec<Op> {
    let mut program = vec![input(DType::Float32, &[2, 3])];
    append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body,
            init,
            operand: NodeId(0),
            in_map: IndexMap::Affine(projection(2, &[0, 1])),
            out_map: IndexMap::Affine(projection(
                2,
                &match keep {
                    Keep::Reduce => vec![0],
                    Keep::Scan => vec![0, 1],
                },
            )),
            keep,
            name: None,
        }),
    );
    program
}

fn counted(census: &MilCensus, operator: &str) -> usize {
    census.operators.get(operator).copied().unwrap_or(0)
}

#[test]
fn mil_census_scalar_ops_table_matches_the_probe() -> Result<()> {
    let probe = fixture("mil_ops_probe.txt")?;

    ensure!(
        SCALAR_OPS.len() == 17,
        "the table lists {} scalar ops",
        SCALAR_OPS.len()
    );
    ensure!(
        table_matches_probe(&table(), &probe),
        "a scalar table row names an operator the compiler did not verify"
    );
    Ok(())
}

#[test]
fn mil_census_scalar_table_degenerate_altered_erf_row_fails() -> Result<()> {
    let probe = fixture("mil_ops_probe.txt")?;
    let altered: Vec<(ScalarOp, MilForm)> = table()
        .into_iter()
        .map(|(body, form)| match body {
            ScalarOp::Erf => (body, MilForm::Operator("erff")),
            _ => (body, form),
        })
        .collect();

    ensure!(
        !table_matches_probe(&altered, &probe),
        "a table naming the operator `erff` matched the compiler's probe"
    );
    Ok(())
}

#[test]
fn mil_census_scalar_table_identity_fixture_against_itself_passes() -> Result<()> {
    let rows = table();

    ensure!(
        table_matches_probe(&rows, &render_probe(&rows)),
        "the table did not match a probe rendered from itself"
    );
    Ok(())
}

#[test]
fn mil_census_negate_is_composed_from_mul_by_minus_one() -> Result<()> {
    let probe = fixture("mil_ops_probe.txt")?;
    let mut program = vec![input(DType::Float32, &[4])];
    append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Negate,
            operands: vec![(NodeId(0), IndexMap::Affine(projection(1, &[0])))],
            name: None,
        },
    );

    let result = census(&program, &[])?;

    ensure!(
        scalar_form(ScalarOp::Negate) == MilForm::Composition(&["mul"]),
        "negate is not mul"
    );
    ensure!(
        verdict_of(&probe, "neg") == ["unknown_operator"],
        "the probe accepted neg"
    );
    ensure!(
        result.refused.is_empty(),
        "negate was refused: {:?}",
        result.refused
    );
    ensure!(
        counted(&result, "mul") == 1,
        "operators were {:?}",
        result.operators
    );
    Ok(())
}

#[test]
fn mil_census_reduce_bodies_map_to_reduce_sum_prod_max_min() -> Result<()> {
    let cases = [
        (ScalarOp::Add, ReduceInit::Zero, "reduce_sum"),
        (ScalarOp::Multiply, ReduceInit::One, "reduce_prod"),
        (
            ScalarOp::Maximum,
            ReduceInit::NegativeInfinity,
            "reduce_max",
        ),
        (
            ScalarOp::Minimum,
            ReduceInit::PositiveInfinity,
            "reduce_min",
        ),
    ];

    for (body, init, operator) in cases {
        let result = census(&reduce_program(body, init, Keep::Reduce), &[])?;

        ensure!(
            result.refused.is_empty(),
            "{body:?} was refused: {:?}",
            result.refused
        );
        ensure!(
            counted(&result, operator) == 1,
            "{body:?} gave {:?}",
            result.operators
        );
    }
    Ok(())
}

#[test]
fn mil_census_scan_add_maps_to_cumsum() -> Result<()> {
    let result = census(
        &reduce_program(ScalarOp::Add, ReduceInit::Zero, Keep::Scan),
        &[],
    )?;

    ensure!(
        result.refused.is_empty(),
        "cumsum was refused: {:?}",
        result.refused
    );
    ensure!(
        counted(&result, "cumsum") == 1,
        "operators were {:?}",
        result.operators
    );
    Ok(())
}

fn refused_scan(body: ScalarOp, init: ReduceInit) -> Result<Refusal> {
    let result = census(&reduce_program(body, init, Keep::Scan), &[])?;
    let refusal = result.refused.first().context("the scan was not refused")?;
    ensure!(
        refusal.node == NodeId(1),
        "the refusal named {}",
        refusal.node
    );
    ensure!(result.mapped == 1, "mapped was {}", result.mapped);
    Ok(refusal.reason)
}

#[test]
fn mil_census_scan_maximum_refused_naming_cummax() -> Result<()> {
    let reason = refused_scan(ScalarOp::Maximum, ReduceInit::NegativeInfinity)?;

    ensure!(
        reason
            == Refusal::ScanBody {
                body: ScalarOp::Maximum,
                missing: "cummax"
            },
        "reason was {reason:?}"
    );
    Ok(())
}

#[test]
fn mil_census_scan_minimum_refused_naming_cummin() -> Result<()> {
    let reason = refused_scan(ScalarOp::Minimum, ReduceInit::PositiveInfinity)?;

    ensure!(
        reason
            == Refusal::ScanBody {
                body: ScalarOp::Minimum,
                missing: "cummin"
            },
        "reason was {reason:?}"
    );
    Ok(())
}

#[test]
fn mil_census_scan_multiply_refused_naming_cumprod() -> Result<()> {
    let reason = refused_scan(ScalarOp::Multiply, ReduceInit::One)?;

    ensure!(
        reason
            == Refusal::ScanBody {
                body: ScalarOp::Multiply,
                missing: "cumprod"
            },
        "reason was {reason:?}"
    );
    Ok(())
}

#[test]
fn mil_census_data_dependent_out_map_refused_naming_scatter() -> Result<()> {
    let mut program = vec![input(DType::Float32, &[4]), input(DType::Int32, &[4])];
    append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: NodeId(0),
            in_map: IndexMap::Affine(projection(1, &[0])),
            out_map: IndexMap::scatter(NodeId(1), projection(1, &[0]), 1, &[], 0, 8),
            keep: Keep::Reduce,
            name: None,
        }),
    );

    let result = census(&program, &[])?;

    let refusal = result
        .refused
        .first()
        .context("the scatter was not refused")?;
    ensure!(
        refusal.node == NodeId(2),
        "the refusal named {}",
        refusal.node
    );
    ensure!(
        refusal.reason == Refusal::Scatter,
        "reason was {:?}",
        refusal.reason
    );
    Ok(())
}

#[test]
fn mil_census_computed_pattern_operand_maps_to_gather() -> Result<()> {
    let mut program = vec![input(DType::Float32, &[5, 3]), input(DType::Int32, &[4])];
    append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Identity,
            operands: vec![(
                NodeId(0),
                IndexMap::Computed {
                    indices: NodeId(1),
                    index_map: projection(2, &[0]),
                    base: affine(2, &[(&[], 0), (&[AxisTerm::projection(1)], 0)]),
                    gathered_dim: 0,
                },
            )],
            name: None,
        },
    );

    let result = census(&program, &[])?;

    ensure!(
        result.refused.is_empty(),
        "the gather was refused: {:?}",
        result.refused
    );
    ensure!(
        counted(&result, "gather") == 1,
        "operators were {:?}",
        result.operators
    );
    Ok(())
}

#[test]
fn mil_census_contraction_classifies_two_operand_multiply_add_reduce_as_matmul() -> Result<()> {
    let mut program = vec![
        input(DType::Float32, &[2, 3]),
        input(DType::Float32, &[3, 4]),
    ];
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (NodeId(0), IndexMap::Affine(projection(3, &[0, 2]))),
                (NodeId(1), IndexMap::Affine(projection(3, &[2, 1]))),
            ],
            name: None,
        },
    );
    append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(projection(3, &[0, 1])),
            keep: Keep::Reduce,
            name: None,
        }),
    );

    let result = census(&program, &[])?;

    ensure!(
        result.refused.is_empty(),
        "the contraction was refused: {:?}",
        result.refused
    );
    ensure!(
        counted(&result, "matmul") == 1,
        "operators were {:?}",
        result.operators
    );
    ensure!(
        counted(&result, "reduce_sum") == 0,
        "the reduce was also counted as reduce_sum"
    );

    let squares = census(&sum_of_squares_program(), &[])?;
    ensure!(
        counted(&squares, "matmul") == 0,
        "a sum of squares was counted as matmul"
    );
    ensure!(
        counted(&squares, "reduce_sum") == 1,
        "operators were {:?}",
        squares.operators
    );
    Ok(())
}

#[test]
fn mil_census_empty_program_is_nodes_0_mapped_0_refused_0() -> Result<()> {
    let result = census(&[], &[])?;

    ensure!(
        result.to_string() == "mil_census nodes=0 mapped=0 refused=0",
        "the line was {result}"
    );
    Ok(())
}

fn lower_pinned(bytes: &[u8], pins: &Pins) -> Result<Vec<Op>> {
    let reader = OnnxReader::new();
    let parsed = reader.parse(bytes)?;
    let validated = reader.validate(parsed, pins)?;
    Ok(reader.lower(&validated, pins)?.program)
}

#[test]
fn mil_census_bge_small_pinned_batch1_seq16_refused_0() -> Result<()> {
    let pins = Pins::from([
        ("batch_size".to_owned(), 1),
        ("sequence_length".to_owned(), 16),
    ]);
    let program = lower_pinned(&bge::read_model()?, &pins)?;

    let result = census(&program, &[])?;

    println!("{result} matmul={}", counted(&result, "matmul"));
    ensure!(
        counted(&result, "matmul") == BGE_MATMUL_NODES,
        "matmul was {}",
        counted(&result, "matmul")
    );
    ensure!(result.refused.is_empty(), "refused: {:?}", result.refused);
    ensure!(
        result.nodes == program.len(),
        "nodes {} of {}",
        result.nodes,
        program.len()
    );
    ensure!(
        result.mapped + result.refused.len() == result.nodes,
        "{result}"
    );
    Ok(())
}

#[test]
fn mil_census_mnist_mapped_plus_refused_equals_nodes() -> Result<()> {
    let bytes = fs::read(MNIST_MODEL).with_context(|| format!("read {MNIST_MODEL}"))?;
    let program = lower_pinned(&bytes, &Pins::new())?;

    let result = census(&program, &[])?;

    println!("{result}");
    result
        .refused
        .iter()
        .for_each(|refusal| println!("  {refusal}"));
    ensure!(
        result.nodes == program.len(),
        "nodes {} of {}",
        result.nodes,
        program.len()
    );
    ensure!(
        result.mapped + result.refused.len() == result.nodes,
        "{result}"
    );
    Ok(())
}

fn sum_of_squares_program() -> Vec<Op> {
    let mut program = vec![input(DType::Float32, &[2, 3])];
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (NodeId(0), IndexMap::Affine(projection(2, &[0, 1]))),
                (NodeId(0), IndexMap::Affine(projection(2, &[0, 1]))),
            ],
            name: None,
        },
    );
    append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(projection(2, &[0, 1])),
            out_map: IndexMap::Affine(projection(2, &[0])),
            keep: Keep::Reduce,
            name: None,
        }),
    );
    program
}
