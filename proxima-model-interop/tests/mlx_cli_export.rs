#![cfg(feature = "std")]
#![allow(clippy::unwrap_used, clippy::expect_used)]
//! `proxima model export-mlx --quantize 4,64` run as a child process on the
//! granite 3.1 1b moe and gemma4 E2B GGUF blobs, with the written directory
//! checked against the source GGUF header and the vendored affine4 references
//! under `tests/fixtures/mlx-reference`.
//!
//! The CLI runs once per model: nextest starts one process per test, so the
//! first test to need a model takes an exclusive file lock, runs the CLI into
//! a directory keyed by the binary's size and mtime, and the rest read it.
//! Blobs and the CLI binary are required, never skipped.

use std::collections::BTreeSet;
use std::fs::{File, create_dir_all, metadata, read, read_dir, write};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::UNIX_EPOCH;

use memmap2::Mmap;
use proxima_gguf::parse_complete;
use serde_json::Value;

const DONE_MARKER: &str = "export.done";
const EXPECTED_FILES: [&str; 4] = [
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
];
const E2B_TOWER_PREFIXES: [&str; 4] =
    ["audio_tower", "embed_audio", "embed_vision", "vision_tower"];
const FUSED_HALF_SUFFIX: &str = "ffn_up_exps.weight";
const EXTRA_LEAF: &str = "rope_freqs.weight";

struct Model {
    name: &'static str,
    gguf: &'static str,
    reference: &'static str,
    tower_prefixes: &'static [&'static str],
}

const GRANITE: Model = Model {
    name: "granite",
    gguf: "/Users/brianbruggeman/.ollama/models/blobs/sha256-cd60b3e8bb445d4c05e0b0b99b1bb41e8bb77211b161e783c71931168131df80",
    reference: "granite-3.1-1b-a400m-moe",
    tower_prefixes: &[],
};

const E2B: Model = Model {
    name: "e2b",
    gguf: "/Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd",
    reference: "gemma4-e2b",
    tower_prefixes: &E2B_TOWER_PREFIXES,
};

fn agrees(actual: usize, expected: usize) -> Result<(), String> {
    match actual == expected {
        true => Ok(()),
        false => Err(format!("counted {actual}, expected {expected}")),
    }
}

fn proxima_binary() -> PathBuf {
    let built = Command::new(env!("CARGO"))
        .args(["build", "-p", "proxima-cli", "--bin", "proxima", "-j2"])
        .status()
        .expect("spawn cargo to build the proxima cli");
    assert!(
        built.success(),
        "cargo build -p proxima-cli failed: {built}"
    );
    let test_binary = std::env::current_exe().expect("test binary path");
    let profile_dir = test_binary
        .parent()
        .and_then(Path::parent)
        .expect("profile directory above deps");
    profile_dir.join("proxima")
}

fn fingerprint(binary: &Path) -> String {
    let info = metadata(binary).expect("proxima binary metadata");
    let modified = info.modified().expect("proxima binary mtime");
    let nanos = modified
        .duration_since(UNIX_EPOCH)
        .expect("mtime after epoch");
    format!("{}-{}", info.len(), nanos.as_nanos())
}

fn run_cli(binary: &Path, model: &Model, directory: &Path) {
    assert!(
        Path::new(model.gguf).exists(),
        "host-local gguf blob missing at {}",
        model.gguf
    );
    let output = Command::new(binary)
        .args(["model", "export-mlx", model.gguf, "--out"])
        .arg(directory)
        .args(["--quantize", "4,64"])
        .output()
        .expect("spawn proxima model export-mlx");
    assert!(
        output.status.success(),
        "export-mlx failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let printed = String::from_utf8_lossy(&output.stdout).lines().count();
    assert_eq!(printed, EXPECTED_FILES.len(), "paths the cli printed");
}

fn exported_directory(model: &Model) -> PathBuf {
    let binary = proxima_binary();
    let root = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("mlx_cli_export")
        .join(fingerprint(&binary));
    create_dir_all(&root).expect("create the export cache directory");
    let guard = File::create(root.join(format!("{}.lock", model.name))).expect("create lock file");
    guard.lock().expect("lock the export cache");
    let directory = root.join(model.name);
    if !directory.join(DONE_MARKER).exists() {
        run_cli(&binary, model, &directory);
        write(directory.join(DONE_MARKER), b"done").expect("write done marker");
    }
    directory
}

fn written_files(directory: &Path) -> BTreeSet<String> {
    read_dir(directory)
        .expect("read export directory")
        .map(|entry| entry.expect("directory entry").file_name())
        .filter_map(|name| name.into_string().ok())
        .filter(|name| name != DONE_MARKER)
        .collect()
}

fn header_names(path: &Path) -> Vec<String> {
    let mut file = File::open(path).expect("open safetensors");
    let mut length = [0u8; 8];
    file.read_exact(&mut length).expect("read header length");
    let mut header = vec![0u8; usize::try_from(u64::from_le_bytes(length)).expect("header fits")];
    file.read_exact(&mut header).expect("read header");
    let parsed: Value = serde_json::from_slice(&header).expect("header is json");
    let object = parsed.as_object().expect("header is an object");
    object
        .keys()
        .filter(|name| *name != "__metadata__")
        .cloned()
        .collect()
}

fn exported_names(model: &Model) -> Vec<String> {
    header_names(&exported_directory(model).join("model.safetensors"))
}

fn quantized_leaf_count(names: &[String]) -> usize {
    let present: BTreeSet<&str> = names.iter().map(String::as_str).collect();
    names
        .iter()
        .filter_map(|name| name.strip_suffix(".scales"))
        .filter(|base| present.contains(format!("{base}.biases").as_str()))
        .count()
}

fn float_plus_quantized_leaves(names: &[String]) -> usize {
    names.len() - 2 * quantized_leaf_count(names)
}

fn gguf_names(model: &Model) -> (u64, Vec<String>) {
    let file = File::open(model.gguf).expect("open gguf blob");
    let mapping = unsafe { Mmap::map(&file) }.expect("map gguf blob");
    let parsed = parse_complete(&mapping).expect("parse gguf header");
    let names = parsed
        .tensors
        .iter()
        .map(|tensor| tensor.name.clone())
        .collect();
    (parsed.tensor_count, names)
}

fn expected_leaf_count(model: &Model) -> usize {
    let (declared, names) = gguf_names(model);
    assert_eq!(declared, names.len() as u64, "tensor_count vs directory");
    let absorbed = names
        .iter()
        .filter(|name| name.ends_with(FUSED_HALF_SUFFIX))
        .count();
    let extra = names.iter().filter(|name| *name == EXTRA_LEAF).count();
    names.len() - absorbed - extra
}

fn reference_names(model: &Model) -> Vec<String> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/mlx-reference")
        .join(model.reference)
        .join("affine4/header.json");
    let header: Value = serde_json::from_slice(&read(path).expect("read affine4 header"))
        .expect("affine4 header is json");
    let object = header.as_object().expect("affine4 header is an object");
    object
        .keys()
        .filter(|name| *name != "__metadata__")
        .cloned()
        .collect()
}

fn without_towers(model: &Model, names: &[String]) -> usize {
    let is_tower = |name: &String| {
        let head = name.split('.').next().unwrap_or_default();
        model.tower_prefixes.contains(&head)
    };
    names.iter().filter(|name| !is_tower(name)).count()
}

fn assert_exact_files(model: &Model) {
    let directory = exported_directory(model);
    let expected: BTreeSet<String> = EXPECTED_FILES
        .iter()
        .map(|name| (*name).to_string())
        .collect();
    assert_eq!(written_files(&directory), expected);
}

fn assert_leaves_equal_gguf_directory(model: &Model) {
    let leaves = float_plus_quantized_leaves(&exported_names(model));
    assert_eq!(agrees(leaves, expected_leaf_count(model)), Ok(()));
}

#[test]
fn mlx_cli_export_granite_writes_exactly_config_weights_and_two_tokenizer_files() {
    assert_exact_files(&GRANITE);
}

#[test]
fn mlx_cli_export_e2b_writes_exactly_config_weights_and_two_tokenizer_files() {
    assert_exact_files(&E2B);
}

#[test]
fn mlx_cli_export_granite_float_tensors_plus_quantized_leaves_equal_gguf_tensors_less_fused_halves()
{
    assert_leaves_equal_gguf_directory(&GRANITE);
}

#[test]
fn mlx_cli_export_e2b_float_tensors_plus_quantized_leaves_equal_gguf_tensors_less_extra_leaves() {
    assert_leaves_equal_gguf_directory(&E2B);
}

#[test]
fn mlx_cli_export_identity_control_gguf_tensor_count_compared_to_itself_agrees() {
    let (declared, _) = gguf_names(&GRANITE);
    assert_eq!(agrees(declared as usize, declared as usize), Ok(()));
}

#[test]
fn mlx_cli_export_degenerate_control_gguf_tensor_count_off_by_one_disagrees() {
    let (declared, _) = gguf_names(&GRANITE);
    assert!(agrees(declared as usize + 1, declared as usize).is_err());
}

#[test]
fn mlx_cli_export_granite_tensor_count_equals_the_affine4_header_count() {
    let reference = reference_names(&GRANITE);
    let exported = exported_names(&GRANITE);
    assert_eq!(agrees(exported.len(), reference.len()), Ok(()));
}

#[test]
fn mlx_cli_export_e2b_tensor_count_equals_the_affine4_header_count_less_vision_and_audio_towers() {
    let reference = reference_names(&E2B);
    let exported = exported_names(&E2B);
    assert_eq!(
        agrees(exported.len(), without_towers(&E2B, &reference)),
        Ok(())
    );
}

#[test]
fn mlx_cli_export_reference_identity_control_header_compared_to_itself_agrees() {
    let reference = reference_names(&GRANITE);
    assert_eq!(agrees(reference.len(), reference.len()), Ok(()));
}

#[test]
fn mlx_cli_export_reference_degenerate_control_header_missing_one_tensor_disagrees() {
    let mut reference = reference_names(&GRANITE);
    reference.pop();
    let exported = exported_names(&GRANITE);
    assert!(agrees(exported.len(), reference.len()).is_err());
}
