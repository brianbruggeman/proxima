//! The core of the shared door: the read and write directions run on the same
//! stage enum, and the door source names no format.

#![cfg(feature = "onnx-door")]
// every expect below runs against a model this file builds itself; a failure
// there IS the test failing, not a case to recover.
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod door_support;
#[path = "door_support/walkthrough.rs"]
mod walkthrough;

use std::cell::RefCell;
use std::collections::BTreeSet;
use std::convert::Infallible;
use std::fs;
use std::path::Path;

use door_support::{mlp_batch, two_layer_mlp_bytes};
use omega::door::{Door, DoorSettings, DoorStage, StageKind, Writer};
use omega::onnx_door::{DoorConfig, OnnxReader};
use proxima_onnx::lower::lower_graph;
use proxima_onnx::pipe::parse_complete;
use proxima_tensor::Lowered;

const DOOR_ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src/door");

const FORMAT_NAMES: [&str; 9] = [
    "onnx", "litert", "tflite", "litertlm", "gecko", "granite", "gemma", "bge", "mnist",
];

#[derive(Debug, Default)]
struct StubWriter {
    calls: RefCell<Vec<&'static str>>,
}

impl Writer for StubWriter {
    type Emitted = usize;
    type Package = Vec<u8>;
    type Error = Infallible;

    fn emit(&self, lowered: &Lowered) -> Result<usize, Infallible> {
        self.calls.borrow_mut().push("emit");
        Ok(lowered.program.len())
    }

    fn package(&self, _lowered: &Lowered, emitted: usize) -> Result<Vec<u8>, Infallible> {
        self.calls.borrow_mut().push("package");
        Ok(emitted.to_le_bytes().to_vec())
    }

    fn verify(&self, _package: &Vec<u8>) -> Result<(), Infallible> {
        self.calls.borrow_mut().push("verify");
        Ok(())
    }
}

fn lowered_mlp() -> Lowered {
    let bytes = two_layer_mlp_bytes();
    let model = parse_complete(&bytes).expect("the mlp parses");
    let graph = model.graph.expect("the mlp carries a graph");
    lower_graph(&graph).expect("the mlp lowers")
}

fn stage_kinds_of_write() -> Vec<StageKind> {
    let door = Door::new(
        OnnxReader::new(),
        StubWriter::default(),
        Vec::new(),
        settings(),
    );
    let batch = mlp_batch();
    let mut stage = DoorStage::emit(door.writer(), lowered_mlp()).expect("the mlp emits");
    let mut kinds = vec![stage.kind()];
    for _ in 0..3 {
        stage = stage.advance(&door, &batch).expect("every stage succeeds");
        kinds.push(stage.kind());
    }
    kinds
}

fn settings() -> DoorSettings {
    DoorConfig::default()
        .settings()
        .expect("the default config resolves")
}

fn count_format_names(source: &str) -> usize {
    let lowered = source.to_lowercase();
    FORMAT_NAMES
        .iter()
        .map(|name| lowered.matches(name).count())
        .sum()
}

#[test]
fn door_shared_core_stage_kinds_cover_read_and_write() {
    let names: BTreeSet<&str> = StageKind::ALL.iter().map(|kind| kind.name()).collect();

    assert_eq!(names.len(), 8);
    assert!(
        ["parse", "validate", "lower", "emit", "package", "verify"]
            .iter()
            .all(|name| names.contains(name))
    );
}

#[test]
fn door_shared_core_write_direction_runs_lower_emit_package_verify_in_order() {
    let lowered = lowered_mlp();
    let nodes = lowered.program.len();
    let door = Door::new(
        OnnxReader::new(),
        StubWriter::default(),
        Vec::new(),
        settings(),
    );

    let package = door.write(lowered).expect("the stub writer never fails");

    assert_eq!(package, nodes.to_le_bytes().to_vec());
    assert_eq!(*door.writer().calls.borrow(), ["emit", "package", "verify"]);
    assert_eq!(
        stage_kinds_of_write(),
        [
            StageKind::Emit,
            StageKind::Package,
            StageKind::Verify,
            StageKind::Lower
        ]
    );
}

#[test]
fn door_shared_core_door_module_is_format_name_free() {
    let mentions: usize = fs::read_dir(Path::new(DOOR_ROOT))
        .expect("the door directory exists")
        .map(|entry| entry.expect("a directory entry reads").path())
        .map(|path| fs::read_to_string(path).expect("a door source reads"))
        .map(|source| count_format_names(&source))
        .sum();

    assert_eq!(mentions, 0);
}

#[test]
fn door_shared_core_format_name_counter_control_counts_one_in_a_string_containing_the_name() {
    assert_eq!(count_format_names("pub struct TfLiteWriter;"), 1);
}

fn lines_containing(lines: &[String], needle: &str) -> usize {
    lines.iter().filter(|line| line.contains(needle)).count()
}

#[test]
fn door_shared_core_onnx_walkthrough_prints_five_advances_and_five_failure_exits() {
    let lines = walkthrough::onnx_lines().expect("the onnx walkthrough runs");

    assert_eq!(lines.len(), 10);
    assert!(lines.iter().all(|line| line.starts_with("transition ")));
    assert_eq!(lines_containing(&lines, "outcome=advanced"), 5);
    assert_eq!(lines_containing(&lines, "outcome=failed"), 5);
}

#[test]
fn door_shared_core_onnx_walkthrough_fails_each_reading_stage_exactly_once() {
    let lines = walkthrough::onnx_lines().expect("the onnx walkthrough runs");

    let stages = [
        StageKind::Parse,
        StageKind::Validate,
        StageKind::Lower,
        StageKind::Plan,
        StageKind::Execute,
    ];
    let per_stage: Vec<usize> = stages
        .iter()
        .map(|stage| lines_containing(&lines, &format!("stage={} outcome=failed", stage.name())))
        .collect();

    assert_eq!(per_stage, vec![1; 5]);
    assert_eq!(lines_containing(&lines, "stage=emit"), 0);
}

#[test]
fn door_shared_core_walkthrough_counter_control_counts_zero_for_a_stage_it_never_fails() {
    let lines = walkthrough::onnx_lines().expect("the onnx walkthrough runs");

    assert_eq!(lines_containing(&lines, "stage=verify outcome=failed"), 0);
}
