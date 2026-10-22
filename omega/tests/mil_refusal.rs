//! Typed refusals: a program MIL has no form for is refused by the writer at
//! the door's Emit stage naming every offender, and a package the compiler
//! refuses is refused at the Verify stage naming the operator and, where the
//! compiler names it, the parameter. The compiler's refusals are real
//! `xcrun coremlcompiler` output on packages carrying `neg` (not a MIL
//! operator) and `cumsum` without its `reverse` parameter.

#![cfg(all(feature = "coreml-lane", feature = "onnx-door", target_os = "macos"))]
// fixtures are hand-built to succeed; an expect failure here IS the test failing.
#![allow(clippy::unwrap_used, clippy::expect_used)]

// the shared door fixture serves several binaries and this one reads only the MLP bytes.
#[allow(dead_code)]
mod door_support;
// the helper module serves several binaries and each uses part of it.
#[allow(dead_code)]
mod mil_support;

use std::fs;
use std::path::PathBuf;

use door_support::two_layer_mlp_bytes;
use mil_support::protoc_encode;
use omega::coreml::compile::{CompileError, Compiler};
use omega::coreml::sink::{CoreMlSink, LaneError};
use omega::door::{Door, DoorError, StageKind, Writer};
use omega::onnx_door::{DoorConfig, OnnxReader};
use proxima_mil::census::Refusal;
use proxima_mil::writer::{MilError, MilPackage};
use proxima_onnx::lower::lower_graph;
use proxima_onnx::pipe::parse_complete;
use proxima_tensor::{
    DType, Extent, IndexMap, Keep, Lowered, NodeId, Op, Reduce, ReduceInit, ScalarOp, Shapes,
    projection,
};
use tempfile::TempDir;

type LaneDoor<W> = Door<OnnxReader, W>;

#[derive(Debug)]
struct CarryingModel {
    inner: CoreMlSink,
    model_text: String,
}

impl Writer for CarryingModel {
    type Emitted = (Shapes, Vec<Vec<u8>>);
    type Package = MilPackage;
    type Error = LaneError;

    fn emit(&self, lowered: &Lowered) -> Result<Self::Emitted, LaneError> {
        self.inner.emit(lowered)
    }

    fn package(&self, lowered: &Lowered, emitted: Self::Emitted) -> Result<MilPackage, LaneError> {
        let carrier = self.inner.package(lowered, emitted)?;
        Ok(MilPackage {
            model: protoc_encode(&self.model_text),
            ..carrier
        })
    }

    fn verify(&self, package: &MilPackage) -> Result<(), LaneError> {
        self.inner.verify(package)
    }
}

fn fixture(name: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/coreml-lane")
        .join(name);
    fs::read_to_string(&path).expect("the refusal fixture is vendored")
}

fn lowered_mlp() -> Lowered {
    let bytes = two_layer_mlp_bytes();
    let model = parse_complete(&bytes).expect("the mlp parses");
    let graph = model.graph.expect("the mlp carries a graph");
    lower_graph(&graph).expect("the mlp lowers")
}

fn scan_program(bodies: &[(ScalarOp, ReduceInit)]) -> Lowered {
    let mut program = vec![Op::Input {
        dtype: DType::FLOAT32,
        shape: vec![Extent::Static(2), Extent::Static(3)],
        name: Some("x".to_owned()),
    }];
    program.extend(bodies.iter().map(|(body, init)| scan(*body, *init)));
    let last = NodeId(u32::try_from(program.len() - 1).expect("a small program"));
    Lowered {
        program,
        initializers: Vec::new(),
        graph_inputs: vec!["x".to_owned()],
        graph_outputs: vec![("y".to_owned(), last)],
        matmul_names: Vec::new(),
    }
}

fn scan(body: ScalarOp, init: ReduceInit) -> Op {
    Op::Reduce(Reduce {
        dtype: DType::FLOAT32,
        body,
        init,
        operand: NodeId(0),
        in_map: IndexMap::Affine(projection(2, &[0, 1])),
        out_map: IndexMap::Affine(projection(2, &[0, 1])),
        keep: Keep::Scan,
        name: None,
    })
}

fn scatter_program() -> Lowered {
    let program = vec![
        Op::Input {
            dtype: DType::FLOAT32,
            shape: vec![Extent::Static(4)],
            name: Some("x".to_owned()),
        },
        Op::Input {
            dtype: DType::INT32,
            shape: vec![Extent::Static(4)],
            name: Some("slots".to_owned()),
        },
        Op::Reduce(Reduce {
            dtype: DType::FLOAT32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: NodeId(0),
            in_map: IndexMap::Affine(projection(1, &[0])),
            out_map: IndexMap::scatter(NodeId(1), projection(1, &[0]), 1, &[], 0, 8),
            keep: Keep::Reduce,
            name: None,
        }),
    ];
    Lowered {
        program,
        initializers: Vec::new(),
        graph_inputs: vec!["x".to_owned(), "slots".to_owned()],
        graph_outputs: vec![("y".to_owned(), NodeId(2))],
        matmul_names: Vec::new(),
    }
}

fn compiler_in(cache: &TempDir) -> Compiler {
    Compiler::locate(cache.path()).expect("xcrun finds coremlcompiler")
}

fn door_of<W: Writer>(writer: W) -> LaneDoor<W> {
    let settings = DoorConfig::default()
        .settings()
        .expect("the default config resolves");
    Door::new(OnnxReader::new(), writer, Vec::new(), settings)
}

fn offenders_of(error: MilError) -> Vec<proxima_mil::census::NoMilForm> {
    match error {
        MilError::NoMilForm { offenders } => offenders,
        other => panic!("expected a typed refusal, got {other:?}"),
    }
}

fn compile_error_of(model_text: &str) -> CompileError {
    let carrier = {
        let lowered = lowered_mlp();
        let sink = omega::door::MilSink;
        let emitted = sink.emit(&lowered).expect("the mlp emits");
        sink.package(&lowered, emitted).expect("the mlp packages")
    };
    let package = MilPackage {
        model: protoc_encode(model_text),
        ..carrier
    };
    let cache = TempDir::new().expect("a temporary cache directory");
    compiler_in(&cache)
        .compile(&package)
        .expect_err("the compiler refuses this package")
}

#[test]
fn mil_refusal_writer_refuses_scatter_naming_it() {
    let error = omega::door::MilSink
        .emit(&scatter_program())
        .expect_err("a scatter has no mil form");

    let offenders = offenders_of(error);

    assert_eq!(offenders.len(), 1);
    assert_eq!(offenders[0].node, NodeId(2));
    assert_eq!(offenders[0].reason, Refusal::Scatter);
    assert!(offenders[0].to_string().contains("scatter"));
}

#[test]
fn mil_refusal_writer_refuses_scan_maximum_naming_cummax() {
    let lowered = scan_program(&[(ScalarOp::Maximum, ReduceInit::NegativeInfinity)]);

    let error = omega::door::MilSink
        .emit(&lowered)
        .expect_err("a maximum scan has no mil form");

    assert!(error.to_string().contains("cummax"), "{error}");
}

#[test]
fn mil_refusal_compiler_unknown_operator_neg_is_typed_naming_neg() {
    let error = compile_error_of(&fixture("mil_refusal_neg.pbtxt"));

    assert!(
        matches!(&error, CompileError::UnknownOperator { operator } if operator == "neg"),
        "{error:?}"
    );
}

#[test]
fn mil_refusal_compiler_missing_param_is_typed_naming_cumsum_and_reverse() {
    let error = compile_error_of(&fixture("mil_refusal_cumsum_no_reverse.pbtxt"));

    assert!(
        matches!(
            &error,
            CompileError::MissingParam { operator, param }
                if operator == "cumsum" && param == "reverse"
        ),
        "{error:?}"
    );
}

#[test]
fn mil_refusal_door_reports_stage_emit_for_a_writer_refusal() {
    let cache = TempDir::new().expect("a temporary cache directory");
    let door = door_of(CoreMlSink::new(compiler_in(&cache)));

    let error = door
        .write(scatter_program())
        .expect_err("the writer refuses a scatter");

    assert_eq!(error.stage(), StageKind::Emit);
    assert!(
        matches!(
            error,
            DoorError::Emit(LaneError::Mil(MilError::NoMilForm { .. }))
        ),
        "{error:?}"
    );
}

#[test]
fn mil_refusal_door_reports_stage_verify_for_a_compiler_refusal() {
    let cache = TempDir::new().expect("a temporary cache directory");
    let door = door_of(CarryingModel {
        inner: CoreMlSink::new(compiler_in(&cache)),
        model_text: fixture("mil_refusal_neg.pbtxt"),
    });

    let error = door
        .write(lowered_mlp())
        .expect_err("the compiler refuses a package carrying neg");

    assert_eq!(error.stage(), StageKind::Verify);
    assert!(
        matches!(
            &error,
            DoorError::Verify(LaneError::Compile(CompileError::UnknownOperator { operator }))
                if operator == "neg"
        ),
        "{error:?}"
    );
}

#[test]
fn mil_refusal_error_display_is_lowercase_and_names_the_operator() {
    let unknown = CompileError::UnknownOperator {
        operator: "neg".to_owned(),
    };
    let missing = CompileError::MissingParam {
        operator: "cumsum".to_owned(),
        param: "reverse".to_owned(),
    };

    let texts = [unknown.to_string(), missing.to_string()];

    assert!(texts.iter().all(|text| *text == text.to_lowercase()));
    assert!(texts[0].contains("`neg`"));
    assert!(texts[1].contains("`cumsum`") && texts[1].contains("`reverse`"));
}

#[test]
fn mil_refusal_clean_program_yields_no_refusal() {
    let cache = TempDir::new().expect("a temporary cache directory");
    let door = door_of(CoreMlSink::new(compiler_in(&cache)));

    let package = door
        .write(lowered_mlp())
        .expect("the two-layer mlp passes emit, package and verify");

    assert!(package.files().iter().all(|(_, bytes)| !bytes.is_empty()));
}

#[test]
fn mil_refusal_lists_every_offender() {
    let lowered = scan_program(&[
        (ScalarOp::Maximum, ReduceInit::NegativeInfinity),
        (ScalarOp::Minimum, ReduceInit::PositiveInfinity),
    ]);

    let error = omega::door::MilSink
        .emit(&lowered)
        .expect_err("both scans are refused");

    let text = error.to_string();
    let nodes: Vec<NodeId> = offenders_of(error)
        .iter()
        .map(|offender| offender.node)
        .collect();
    assert_eq!(nodes, vec![NodeId(1), NodeId(2)]);
    assert!(text.contains("cummax") && text.contains("cummin"), "{text}");
}
