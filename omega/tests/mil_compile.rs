//! The compile edge judged by the compiler itself: a package the writer
//! produced is compiled by `xcrun coremlcompiler`, and the operators it kept
//! (read back from the compiled `model.mil`) are counted against the operators
//! `protoc` decodes from the emitted model, an oracle that shares no code with
//! either the writer or the edge.

#![cfg(all(feature = "coreml-lane", feature = "onnx-door", target_os = "macos"))]
// fixtures are hand-built to succeed; an expect failure here IS the test failing.
#![allow(clippy::unwrap_used, clippy::expect_used)]

// the shared door fixture serves several binaries and this one reads only the MLP bytes.
#[allow(dead_code)]
mod door_support;
// the shared bge fixture serves several binaries and this one reads only the model bytes.
#[allow(dead_code)]
#[path = "support/bge.rs"]
mod bge;
// the helper module serves several binaries and each uses part of it.
#[allow(dead_code)]
mod mil_support;

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use door_support::two_layer_mlp_bytes;
use mil_support::{operator_sequence, protoc_decode, protoc_encode};
use omega::coreml::compile::{CompileError, Compiled, Compiler, operator_counts, package_sha256};
use omega::door::{Pins, Reader};
use omega::onnx_door::OnnxReader;
use proxima_mil::weights::SourceDType;
use proxima_mil::writer::{MilPackage, MilRequest, MilWriter, WeightView};
use proxima_onnx::lower::lower_graph;
use proxima_onnx::pipe::parse_complete;
use proxima_tensor::{Op, Shapes, infer};
use tempfile::TempDir;

const BGE_MATMUL_NODES: usize = 96;
const MLP_OUTPUT: &str = "y";
const MISSING_TOOL: &str = "/nonexistent/coremlcompiler";

fn probe_text() -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/coreml-lane/mil_ops_probe.pbtxt");
    fs::read_to_string(&path).expect("the probe fixture is vendored")
}

fn write_package(
    program: &[Op],
    shapes: &Shapes,
    initializers: &[(String, Vec<f32>)],
    output: &str,
) -> MilPackage {
    let encoded: Vec<Vec<u8>> = initializers
        .iter()
        .map(|(_, values)| {
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect()
        })
        .collect();
    let weights: Vec<WeightView<'_>> = initializers
        .iter()
        .zip(&encoded)
        .map(|((name, _), bytes)| WeightView {
            name,
            dtype: SourceDType::Float32,
            bytes,
        })
        .collect();
    let request = MilRequest {
        program,
        shapes,
        weights: &weights,
        output,
    };
    MilWriter::new()
        .write(&request)
        .expect("the writer emits the program")
}

fn mlp_package() -> MilPackage {
    let bytes = two_layer_mlp_bytes();
    let model = parse_complete(&bytes).expect("the mlp bytes parse");
    let graph = model.graph.as_ref().expect("the mlp has a graph");
    let lowered = lower_graph(graph).expect("the mlp lowers");
    let shapes = infer(&lowered.program, &[]).expect("the mlp program has shapes");
    write_package(&lowered.program, &shapes, &lowered.initializers, MLP_OUTPUT)
}

fn bge_package(batch: u64, sequence: u64) -> MilPackage {
    let pins = Pins::from([
        ("batch_size".to_owned(), batch),
        ("sequence_length".to_owned(), sequence),
    ]);
    let reader = OnnxReader::new();
    let bytes = bge::read_model().expect("the bge-small model is cached");
    let parsed = reader.parse(&bytes).expect("the model parses");
    let validated = reader.validate(parsed, &pins).expect("the pins validate");
    let lowered = reader.lower(&validated, &pins).expect("the model lowers");
    let shapes = infer(&lowered.program, &[]).expect("the program has shapes");
    let (output, _) = &lowered.graph_outputs[0];
    write_package(&lowered.program, &shapes, &lowered.initializers, output)
}

fn probe_package() -> MilPackage {
    let carrier = mlp_package();
    MilPackage {
        model: protoc_encode(&probe_text()),
        ..carrier
    }
}

fn compiler_in(cache: &Path) -> Compiler {
    Compiler::locate(cache).expect("xcrun finds coremlcompiler")
}

fn compile_fresh(package: &MilPackage) -> (TempDir, Compiled) {
    let cache = TempDir::new().expect("a temporary cache directory");
    let compiled = compiler_in(cache.path())
        .compile(package)
        .expect("the compiler accepts the package");
    (cache, compiled)
}

fn emitted_operators(package: &MilPackage) -> BTreeMap<String, usize> {
    let text = protoc_decode(&package.model).expect("protoc decodes the model");
    operator_sequence(&text)
        .into_iter()
        .fold(BTreeMap::new(), |mut counts, name| {
            *counts.entry(name).or_insert(0) += 1;
            counts
        })
}

fn compiled_operators(compiled: &Compiled) -> BTreeMap<String, usize> {
    operator_counts(
        &compiled
            .model_mil()
            .expect("the compiled model has model.mil"),
    )
}

#[test]
fn mil_compile_probe_program_compiles_exit_0() {
    let package = probe_package();

    let (_cache, compiled) = compile_fresh(&package);

    assert!(!compiled.cache_hit);
    assert!(compiled.directory.join("model.mil").is_file());
    assert_eq!(compiled.sha256, package_sha256(&package));
}

#[test]
fn mil_compile_probe_compiled_operator_multiset_equals_emitted_multiset() {
    let package = probe_package();

    let (_cache, compiled) = compile_fresh(&package);

    let emitted = emitted_operators(&package);
    let kept = compiled_operators(&compiled);
    assert_eq!(emitted.values().sum::<usize>(), 41);
    assert_eq!(kept, emitted);
}

#[test]
fn mil_compile_mlp_package_compiles_and_compiled_dir_holds_model_mil() {
    let package = mlp_package();

    let (_cache, compiled) = compile_fresh(&package);

    let model_mil = compiled.model_mil().expect("model.mil is readable");
    assert!(model_mil.starts_with("program(1.0)"), "{model_mil}");
    assert_eq!(
        compiled
            .directory
            .file_name()
            .and_then(|name| name.to_str()),
        Some(format!("{}.mlmodelc", compiled.sha256).as_str())
    );
    assert_eq!(compiled_operators(&compiled), emitted_operators(&package));
}

#[test]
fn mil_compile_bge_pin_b1_s16_compiles_exit_0() {
    let package = bge_package(1, 16);

    let (_cache, compiled) = compile_fresh(&package);

    assert!(compiled.directory.join("model.mil").is_file());
}

#[test]
fn mil_compile_bge_pin_b1_s16_compiled_matmul_count_is_96() {
    let package = bge_package(1, 16);

    let (_cache, compiled) = compile_fresh(&package);

    let emitted = emitted_operators(&package);
    let kept = compiled_operators(&compiled);
    println!(
        "mil_compile pin=b1_s16 ops_emitted={} ops_compiled={} matmul_emitted={} matmul_compiled={}",
        emitted.values().sum::<usize>(),
        kept.values().sum::<usize>(),
        emitted.get("matmul").copied().unwrap_or(0),
        kept.get("matmul").copied().unwrap_or(0),
    );
    assert_eq!(kept.get("matmul").copied(), Some(BGE_MATMUL_NODES));
}

#[test]
fn mil_compile_bge_pin_b32_s48_compiles_exit_0() {
    let package = bge_package(32, 48);

    let (_cache, compiled) = compile_fresh(&package);

    assert!(compiled.directory.join("model.mil").is_file());
}

#[test]
fn mil_compile_second_compile_of_the_same_package_spawns_zero_processes() {
    let package = mlp_package();
    let (cache, first) = compile_fresh(&package);
    let silent = Compiler::with_tool(MISSING_TOOL, cache.path());

    let second = silent.compile(&package).expect("the cache answers");

    assert!(!first.cache_hit);
    assert!(second.cache_hit);
    assert_eq!(second.directory, first.directory);
    let changed = MilPackage {
        weights: package.weights.iter().map(|byte| byte ^ 0x01).collect(),
        ..package
    };
    let control = silent.compile(&changed);
    assert!(
        matches!(control, Err(CompileError::Spawn { .. })),
        "a changed package must reach the compiler: {control:?}"
    );
}

#[test]
fn mil_compile_failure_is_typed_with_the_compilers_message() {
    let mut package = mlp_package();
    package.model[0] ^= 0x07;
    let cache = TempDir::new().expect("a temporary cache directory");

    let verdict = compiler_in(cache.path()).compile(&package);

    match verdict {
        Err(CompileError::Rejected { message, .. }) => assert!(
            message.contains("Failed to parse the model specification"),
            "{message}"
        ),
        other => panic!("expected the compiler's rejection, got {other:?}"),
    }
    assert_eq!(
        fs::read_dir(cache.path()).expect("the cache lists").count(),
        0
    );
}
