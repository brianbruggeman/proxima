//! `MilSink` is the door's writing half over `MilWriter`: emit refuses what MIL
//! has no form for, package is `MilWriter`'s own bytes, and each refusal names
//! the node or file it is about.

#![cfg(feature = "mil-writer")]
// fixtures are hand-built to succeed; an expect failure here IS the test failing.
#![allow(clippy::unwrap_used, clippy::expect_used)]

// the shared door fixture serves several binaries and this one reads only the MLP bytes.
#[allow(dead_code)]
mod door_support;

use door_support::two_layer_mlp_bytes;
use omega::door::{MilSink, Writer};
use proxima_mil::census::Refusal;
use proxima_mil::weights::SourceDType;
use proxima_mil::writer::{MilError, MilPackage, MilRequest, MilWriter, WeightView};
use proxima_onnx::lower::lower_graph;
use proxima_onnx::pipe::parse_complete;
use proxima_tensor::{
    DType, Extent, IndexMap, Keep, Lowered, NodeId, Op, Reduce, ReduceInit, ScalarOp, infer,
    projection,
};

fn lowered_mlp() -> Lowered {
    let bytes = two_layer_mlp_bytes();
    let model = parse_complete(&bytes).expect("the mlp parses");
    let graph = model.graph.expect("the mlp carries a graph");
    lower_graph(&graph).expect("the mlp lowers")
}

fn input(extent: Extent) -> Op {
    Op::Input {
        dtype: DType::Float32,
        shape: vec![extent, Extent::Static(3)],
        name: Some("x".to_owned()),
    }
}

fn scan(body: ScalarOp, init: ReduceInit) -> Op {
    Op::Reduce(Reduce {
        dtype: DType::Float32,
        body,
        init,
        operand: NodeId(0),
        in_map: IndexMap::Affine(projection(2, &[0, 1])),
        out_map: IndexMap::Affine(projection(2, &[0, 1])),
        keep: Keep::Scan,
        name: None,
    })
}

fn program_of(program: Vec<Op>, outputs: Vec<(String, NodeId)>) -> Lowered {
    Lowered {
        program,
        initializers: Vec::new(),
        graph_inputs: vec!["x".to_owned()],
        graph_outputs: outputs,
        matmul_names: Vec::new(),
    }
}

fn error_of<T>(result: Result<T, MilError>) -> MilError {
    result.err().expect("the sink must refuse this program")
}

#[test]
fn mil_sink_package_equals_the_bytes_of_calling_the_writer_directly() {
    let lowered = lowered_mlp();
    let shapes = infer(&lowered.program, &[]).expect("the mlp has shapes");
    let encoded: Vec<Vec<u8>> = lowered
        .initializers
        .iter()
        .map(|(_, values)| {
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect()
        })
        .collect();
    let weights: Vec<WeightView<'_>> = lowered
        .initializers
        .iter()
        .zip(&encoded)
        .map(|((name, _), bytes)| WeightView {
            name,
            dtype: SourceDType::Float32,
            bytes,
        })
        .collect();
    let request = MilRequest {
        program: &lowered.program,
        shapes: &shapes,
        weights: &weights,
        output: "y",
    };
    let direct = MilWriter::new()
        .write(&request)
        .expect("the writer emits the mlp");

    let emitted = MilSink.emit(&lowered).expect("the sink emits the mlp");
    let through_sink = MilSink
        .package(&lowered, emitted)
        .expect("the sink packages the mlp");

    assert_eq!(through_sink, direct);
}

#[test]
fn mil_sink_emit_without_a_declared_output_is_no_output() {
    let lowered = program_of(vec![input(Extent::Static(2))], Vec::new());

    let error = error_of(MilSink.emit(&lowered));

    assert_eq!(error, MilError::NoOutput);
}

#[test]
fn mil_sink_emit_names_an_output_that_is_not_the_final_node() {
    let program = vec![
        input(Extent::Static(2)),
        scan(ScalarOp::Add, ReduceInit::Zero),
    ];
    let lowered = program_of(program, vec![("y".to_owned(), NodeId(0))]);

    let error = error_of(MilSink.emit(&lowered));

    assert_eq!(error, MilError::OutputNotFinal { node: NodeId(0) });
}

#[test]
fn mil_sink_emit_refuses_a_scan_body_mil_has_no_operator_for() {
    let program = vec![
        input(Extent::Static(2)),
        scan(ScalarOp::Maximum, ReduceInit::NegativeInfinity),
    ];
    let lowered = program_of(program, vec![("y".to_owned(), NodeId(1))]);

    let error = error_of(MilSink.emit(&lowered));

    let MilError::NoMilForm { offenders } = error else {
        panic!("expected a typed refusal, got {error:?}");
    };
    let refused = offenders[0];
    assert_eq!(offenders.len(), 1);
    assert_eq!(refused.node, NodeId(1));
    assert!(
        matches!(refused.reason, Refusal::ScanBody { .. }),
        "{refused:?}"
    );
}

#[test]
fn mil_sink_emit_reports_a_program_whose_shapes_cannot_resolve() {
    let lowered = program_of(
        vec![input(Extent::Symbolic(0))],
        vec![("y".to_owned(), NodeId(0))],
    );

    let error = error_of(MilSink.emit(&lowered));

    assert!(matches!(error, MilError::Shape(_)), "{error:?}");
}

#[test]
fn mil_sink_package_reports_a_form_the_writer_does_not_emit() {
    let program = vec![
        input(Extent::Static(2)),
        scan(ScalarOp::Add, ReduceInit::Zero),
    ];
    let lowered = program_of(program, vec![("y".to_owned(), NodeId(1))]);
    let emitted = MilSink
        .emit(&lowered)
        .expect("the census maps a cumulative sum");

    let error = error_of(MilSink.package(&lowered, emitted));

    assert!(
        matches!(
            error,
            MilError::NotEmitted {
                node: NodeId(1),
                ..
            }
        ),
        "{error:?}"
    );
}

#[test]
fn mil_sink_verify_accepts_a_written_package_and_names_an_empty_file() {
    let lowered = lowered_mlp();
    let emitted = MilSink.emit(&lowered).expect("the sink emits the mlp");
    let package = MilSink
        .package(&lowered, emitted)
        .expect("the sink packages the mlp");
    let emptied = MilPackage {
        weights: Vec::new(),
        ..package.clone()
    };

    let accepted = MilSink.verify(&package);
    let refused = error_of(MilSink.verify(&emptied));

    assert_eq!(accepted, Ok(()));
    assert_eq!(
        refused,
        MilError::EmptyPackageFile {
            path: "Data/com.apple.CoreML/weights/weight.bin"
        }
    );
}
