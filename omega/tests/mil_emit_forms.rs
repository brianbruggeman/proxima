//! Forms beyond the two-layer MLP that the writer emits: the gather an
//! embedding lookup is, the cast an integer-typed identity is, and the real
//! bge-small program at batch 1 and sequence 16 (96 `matmul`, 39 `gather`, 12
//! head-merging `reshape`), each decoded by `protoc` and counted from the
//! decoded operations.

#![cfg(feature = "mil-writer")]
// fixtures are hand-built to succeed; an expect failure here IS the test failing.
#![allow(clippy::unwrap_used, clippy::expect_used)]

// the shared bge fixture serves several binaries and this one reads only the model bytes.
#[allow(dead_code)]
#[path = "support/bge.rs"]
mod bge;
// the helper module serves several binaries and each uses part of it.
#[allow(dead_code)]
mod mil_support;

use mil_support::{operator_sequence, protoc_decode};
use omega::door::{Pins, Reader};
use omega::onnx_door::OnnxReader;
use proxima_mil::weights::SourceDType;
use proxima_mil::writer::{MilError, MilPackage, MilRequest, MilWriter, WeightView};
use proxima_tensor::{
    AxisTerm, DType, Extent, IndexMap, NodeId, Op, ScalarOp, affine, append, infer, projection,
};

const BGE_MATMUL_NODES: usize = 96;
const BGE_GATHER_NODES: usize = 39;
const BGE_HEAD_MERGES: usize = 12;

fn input(dtype: DType, shape: &[u32], name: &str) -> Op {
    Op::Input {
        dtype,
        shape: shape.iter().map(|extent| Extent::Static(*extent)).collect(),
        name: Some(name.to_owned()),
    }
}

fn write(program: &[Op], weights: &[WeightView<'_>], output: &str) -> MilPackage {
    let shapes = infer(program, &[]).expect("the program has shapes");
    let request = MilRequest {
        program,
        shapes: &shapes,
        weights,
        output,
    };
    MilWriter::new()
        .write(&request)
        .expect("the writer emits the program")
}

fn operators(package: &MilPackage) -> Vec<String> {
    let text = protoc_decode(&package.model).expect("protoc decodes the model");
    operator_sequence(&text)
}

fn counted(sequence: &[String], operator: &str) -> usize {
    sequence.iter().filter(|name| *name == operator).count()
}

#[test]
fn mil_emit_embedding_lookup_is_one_gather_of_int32_rows_from_a_blob_table() {
    let table: Vec<u8> = (0..15_u8)
        .flat_map(|value| f32::from(value).to_le_bytes())
        .collect();
    let weights = [WeightView {
        name: "table",
        dtype: SourceDType::FLOAT32,
        bytes: &table,
    }];
    let mut program = vec![
        input(DType::FLOAT32, &[5, 3], "table"),
        input(DType::INT32, &[4], "rows"),
    ];
    append(
        &mut program,
        Op::Elementwise {
            dtype: DType::FLOAT32,
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

    let sequence = operators(&write(&program, &weights, "picked"));

    assert_eq!(sequence, ["const", "cast", "gather", "identity", "cast"]);
}

#[test]
fn mil_emit_integer_identity_over_a_float_operand_is_a_cast_not_an_identity() {
    let mut program = vec![input(DType::FLOAT32, &[4], "values")];
    append(
        &mut program,
        Op::Elementwise {
            dtype: DType::INT32,
            body: ScalarOp::Identity,
            operands: vec![(NodeId(0), IndexMap::Affine(projection(1, &[0])))],
            name: None,
        },
    );

    let sequence = operators(&write(&program, &[], "truncated"));

    assert_eq!(sequence, ["cast", "cast", "cast"]);
    assert_eq!(counted(&sequence, "identity"), 0);
}

#[test]
fn mil_emit_bge_small_pinned_batch1_seq16_has_96_matmul_39_gather_and_12_head_merges() {
    let pins = Pins::from([
        ("batch_size".to_owned(), 1),
        ("sequence_length".to_owned(), 16),
    ]);
    let reader = OnnxReader::new();
    let bytes = bge::read_model().expect("the bge-small model is cached");
    let parsed = reader.parse(&bytes).expect("the model parses");
    let validated = reader.validate(parsed, &pins).expect("the pins validate");
    let lowered = reader.lower(&validated, &pins).expect("the model lowers");
    let encoded: Vec<(String, Vec<u8>)> = lowered
        .initializers
        .iter()
        .map(|(name, values)| {
            let bytes = values.iter().flat_map(|value| value.to_le_bytes());
            (name.clone(), bytes.collect())
        })
        .collect();
    let weights: Vec<WeightView<'_>> = encoded
        .iter()
        .map(|(name, bytes)| WeightView {
            name,
            dtype: SourceDType::FLOAT32,
            bytes,
        })
        .collect();
    let (output, _) = &lowered.graph_outputs[0];

    let package = write(&lowered.program, &weights, output);
    let sequence = operators(&package);

    assert_eq!(counted(&sequence, "matmul"), BGE_MATMUL_NODES);
    assert_eq!(counted(&sequence, "gather"), BGE_GATHER_NODES);
    assert_eq!(counted(&sequence, "reshape"), BGE_HEAD_MERGES);
    assert_eq!(
        u32::from_le_bytes(package.weights[0..4].try_into().expect("four bytes")) as usize,
        weights.len()
    );
}

fn add_of_broadcast_operand(operand_rows: u32) -> Vec<Op> {
    let mut program = vec![
        input(DType::FLOAT32, &[3, 4], "left"),
        input(DType::FLOAT32, &[operand_rows, 4], "right"),
    ];
    append(
        &mut program,
        Op::Elementwise {
            dtype: DType::FLOAT32,
            body: ScalarOp::Add,
            operands: vec![
                (NodeId(0), IndexMap::Affine(projection(2, &[0, 1]))),
                (
                    NodeId(1),
                    IndexMap::Affine(affine(2, &[(&[], 0), (&[AxisTerm::projection(1)], 0)])),
                ),
            ],
            name: None,
        },
    );
    program
}

#[test]
fn mil_emit_operand_of_extent_one_on_an_unread_axis_broadcasts_without_a_view() {
    let program = add_of_broadcast_operand(1);

    let sequence = operators(&write(&program, &[], "sum"));

    assert_eq!(sequence, ["cast", "cast", "add", "cast"]);
}

#[test]
fn mil_emit_operand_with_a_constant_index_into_a_longer_axis_is_not_emitted() {
    let program = add_of_broadcast_operand(2);
    let shapes = infer(&program, &[]).expect("the program has shapes");
    let request = MilRequest {
        program: &program,
        shapes: &shapes,
        weights: &[],
        output: "sum",
    };

    let verdict = MilWriter::new().write(&request);

    assert!(
        matches!(
            verdict,
            Err(MilError::NotEmitted {
                node: NodeId(2),
                ..
            })
        ),
        "{verdict:?}"
    );
}
