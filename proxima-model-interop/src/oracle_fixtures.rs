//! Integrity tests for the committed real-checkpoint oracle fixtures under
//! `tests/fixtures/flint/`.
//!
//! Every fixture here is captured from the real flint GGUF by tools that share
//! no code with `proxima-gguf` (`llama-cli`'s model loader, `llama-gguf`, and
//! `tools/gguf-kv-extract`), so the later slices that read this checkpoint are
//! held to an independent oracle rather than to themselves.
//! `layer_patterns.txt` records why it needs the third tool: llama.cpp truncates
//! every metadata value at 40 characters (`llama-model-loader.cpp:815`,
//! `MAX_VALUE_LEN`), so the full per-layer arrays come from `gguf-kv-extract`
//! and are corroborated here against the loader's `arr[bool,49]` records and
//! their leading elements.
//!
//! [`first_id_divergence`] is the comparator the greedy-parity test uses;
//! `tests/fsm_oracle_control.rs` carries the integration-test copy of the same
//! rule for the FSM fixtures.

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde_json::Value;
use sha2::{Digest, Sha256};

const FIXTURE_DIRECTORY: &str = "tests/fixtures/flint";
const HEADER_FIXTURE: &str = "gguf_kv_loader.txt";
const TENSOR_FIXTURE: &str = "tensor_dir.txt";
const LAYER_PATTERN_FIXTURE: &str = "layer_patterns.txt";
const LLAMA_ID_FIXTURE: &str = "llama_ids.json";
const TOKENIZER_ID_FIXTURE: &str = "tokenizer_ids.json";

/// sha256 of each fixture's exact bytes. Without this the files are the oracle
/// and nothing pins them: a mutated id list would satisfy every comparison in
/// this module, and slice 12's parity verdict would rest on bytes no test had
/// ever fixed.
const FIXTURE_DIGESTS: [(&str, &str); 5] = [
    (
        HEADER_FIXTURE,
        "a940a4c39a389a50c02eeb342bce0b3afcced30995034f66cdc5269ca5b20b2e",
    ),
    (
        TENSOR_FIXTURE,
        "5db66d8f4876ed96ffcf72ed163402b5c5992008f46997aa83d4c1cabd423632",
    ),
    (
        LAYER_PATTERN_FIXTURE,
        "568482a96009bca0f0982cfb7c47daaee6d2e187bdea78e26e995e3a23275427",
    ),
    (
        LLAMA_ID_FIXTURE,
        "1d682a7b77b45bcb1a3da9a78b41e1b0d75026c6620bf0e18cea9c86b3e1d82b",
    ),
    (
        TOKENIZER_ID_FIXTURE,
        "75afdc7a2539d4ba8319c5779946f0b64d19c1f790099b131846523ff520d179",
    ),
];

#[derive(Clone)]
pub(crate) struct HeaderEntry {
    pub(crate) key: String,
    pub(crate) kind: String,
    pub(crate) value: String,
}

impl HeaderEntry {
    /// `arr[bool,49]` -> `Some(49)`; `None` for a scalar entry.
    fn array_len(&self) -> Option<usize> {
        let inner = self.kind.strip_prefix("arr[")?;
        let (_, count) = inner.rsplit_once(',')?;
        count.strip_suffix(']')?.parse().ok()
    }
}

#[derive(Clone)]
struct TensorRow {
    name: String,
    ggml_type: String,
    size_bytes: u64,
    dimensions: Vec<u64>,
}

pub(crate) fn fixture_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(FIXTURE_DIRECTORY)
        .join(name)
}

pub(crate) fn read_fixture(name: &str) -> String {
    let path = fixture_path(name);
    std::fs::read_to_string(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()))
}

fn verify_fixture_digests() {
    for (name, expected) in FIXTURE_DIGESTS {
        let path = fixture_path(name);
        let bytes =
            std::fs::read(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
        let actual: String = Sha256::digest(&bytes)
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        assert_eq!(actual, expected, "{name} changed since its capture");
    }
}

/// The loader writes `- kv   0: <key> <type> = <value>`; comments start with `#`.
pub(crate) fn parse_header(text: &str) -> Vec<HeaderEntry> {
    text.lines()
        .filter_map(|line| {
            let rest = line.strip_prefix("- kv")?;
            let (_, rest) = rest.split_once(':')?;
            let rest = rest.trim_start();
            let (key, rest) = rest.split_once(char::is_whitespace)?;
            let rest = rest.trim_start();
            let (kind, rest) = rest.split_once(char::is_whitespace)?;
            let value = rest.trim_start().strip_prefix('=')?.trim_start();
            Some(HeaderEntry {
                key: key.to_string(),
                kind: kind.to_string(),
                value: value.to_string(),
            })
        })
        .collect()
}

fn header_entry<'a>(entries: &'a [HeaderEntry], key: &str) -> &'a HeaderEntry {
    entries
        .iter()
        .find(|entry| entry.key == key)
        .unwrap_or_else(|| panic!("the captured header has no {key} entry"))
}

pub(crate) fn header_scalar<'a>(entries: &'a [HeaderEntry], key: &str) -> &'a str {
    &header_entry(entries, key).value
}

/// `tensor   0: name = output.weight, type = q8_0, size_bytes = 326713344, dims = (2048, 150144, 1, 1)`
fn parse_tensor_rows(text: &str) -> Vec<TensorRow> {
    text.lines()
        .filter_map(|line| {
            let rest = line.strip_prefix("tensor")?;
            let (_, rest) = rest.split_once(':')?;
            let (name, rest) = rest
                .trim_start()
                .strip_prefix("name = ")?
                .split_once(", type = ")?;
            let (ggml_type, rest) = rest.split_once(", size_bytes = ")?;
            let (size_bytes, rest) = rest.split_once(", dims = (")?;
            let dimensions = rest
                .strip_suffix(')')?
                .split(", ")
                .map(|dimension| {
                    dimension
                        .trim()
                        .parse()
                        .unwrap_or_else(|error| panic!("{name}: dim {dimension}: {error}"))
                })
                .collect();
            Some(TensorRow {
                name: name.to_string(),
                ggml_type: ggml_type.to_string(),
                size_bytes: size_bytes
                    .parse()
                    .unwrap_or_else(|error| panic!("{name}: size {size_bytes}: {error}")),
                dimensions,
            })
        })
        .collect()
}

/// `<metadata key> = [true, false, ...]`, one entry per block.
pub(crate) fn parse_layer_patterns(text: &str) -> BTreeMap<String, Vec<bool>> {
    text.lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
        .filter_map(|line| {
            let (key, list) = line.split_once(" = ")?;
            let list = list.strip_prefix('[')?.strip_suffix(']')?;
            let values = list
                .split(", ")
                .map(|entry| match entry.trim() {
                    "true" => true,
                    "false" => false,
                    other => panic!("{key}: {other} is not a boolean"),
                })
                .collect();
            Some((key.to_string(), values))
        })
        .collect()
}

fn first_record<'a>(records: &'a Value, fixture: &str) -> &'a Value {
    records
        .as_array()
        .and_then(|array| array.first())
        .unwrap_or_else(|| panic!("{fixture} holds no record"))
}

fn record_ids(record: &Value, field: &str, fixture: &str) -> Vec<u32> {
    record[field]
        .as_array()
        .unwrap_or_else(|| panic!("{fixture}: record has no {field} array"))
        .iter()
        .map(|id| {
            let id = id
                .as_u64()
                .unwrap_or_else(|| panic!("{fixture}: {field} holds a non-integer id"));
            u32::try_from(id).unwrap_or_else(|error| panic!("{fixture}: {field} id {id}: {error}"))
        })
        .collect()
}

fn json_fixture(name: &str) -> Value {
    serde_json::from_str(&read_fixture(name))
        .unwrap_or_else(|error| panic!("{name} is not json: {error}"))
}

/// The greedy-parity comparator: the first position where the two id runs
/// disagree, or `None` when they are identical. A length difference diverges
/// at the shorter run's length, so a truncated run is never mistaken for a
/// match.
pub(crate) fn first_id_divergence(expected: &[u32], actual: &[u32]) -> Option<usize> {
    let shared = expected.len().min(actual.len());
    (0..shared)
        .find(|&index| expected[index] != actual[index])
        .or_else(|| (expected.len() != actual.len()).then_some(shared))
}

/// Every header value R1 names, as one predicate so the control test can prove
/// the same comparison rejects a mutated capture.
fn header_problem(entries: &[HeaderEntry]) -> Option<String> {
    if entries.len() != 52 {
        return Some(format!("{} header entries, expected 52", entries.len()));
    }
    for (key, expected) in [
        ("general.architecture", "flint"),
        ("flint.block_count", "49"),
        ("flint.expert_count", "256"),
        ("flint.expert_used_count", "6"),
        ("flint.expert_shared_count", "2"),
        ("flint.leading_dense_block_count", "2"),
        ("flint.attention.sliding_window", "513"),
    ] {
        let found = header_scalar(entries, key);
        if found != expected {
            return Some(format!("{key} = {found}, expected {expected}"));
        }
    }
    for key in [
        "flint.attention.sliding_window_pattern",
        "flint.attention.rope_pattern",
    ] {
        match header_entry(entries, key).array_len() {
            Some(49) => {}
            other => return Some(format!("{key} array length {other:?}, expected 49")),
        }
    }
    None
}

fn tensor_problem(rows: &[TensorRow]) -> Option<String> {
    if rows.len() != 779 {
        return Some(format!("{} tensor rows, expected 779", rows.len()));
    }
    if let Some(row) = rows.iter().find(|row| {
        row.name.is_empty()
            || row.ggml_type.is_empty()
            || row.size_bytes == 0
            || row.dimensions.is_empty()
    }) {
        return Some(format!(
            "tensor {} is missing its name, type, byte size or dimensions",
            row.name
        ));
    }
    if rows[0].name != "output.weight" {
        return Some(format!(
            "first tensor is {}, expected output.weight",
            rows[0].name
        ));
    }
    if rows[778].name != "blk.48.ffn_up_shexp.weight" {
        return Some(format!(
            "last tensor is {}, expected blk.48.ffn_up_shexp.weight",
            rows[778].name
        ));
    }
    None
}

#[test]
fn flint_fixture_header_matches_the_captured_checkpoint() {
    verify_fixture_digests();
    let header = parse_header(&read_fixture(HEADER_FIXTURE));
    if let Some(problem) = header_problem(&header) {
        panic!("{HEADER_FIXTURE}: {problem}");
    }

    let tensors = parse_tensor_rows(&read_fixture(TENSOR_FIXTURE));
    if let Some(problem) = tensor_problem(&tensors) {
        panic!("{TENSOR_FIXTURE}: {problem}");
    }

    let patterns = parse_layer_patterns(&read_fixture(LAYER_PATTERN_FIXTURE));
    for key in [
        "flint.attention.sliding_window_pattern",
        "flint.attention.rope_pattern",
    ] {
        let pattern = patterns
            .get(key)
            .unwrap_or_else(|| panic!("{LAYER_PATTERN_FIXTURE} has no {key}"));
        assert_eq!(pattern.len(), 49, "{key} carries one entry per block");
        let captured_prefix = header_scalar(&header, key);
        let rendered: String = pattern
            .iter()
            .take(5)
            .map(|value| value.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        assert!(
            captured_prefix.starts_with(&format!("[{rendered}")),
            "{key}: the loader's captured prefix {captured_prefix} does not start with the \
             full array's first five entries [{rendered}"
        );
    }

    let llama = json_fixture(LLAMA_ID_FIXTURE);
    let llama_record = first_record(&llama, LLAMA_ID_FIXTURE);
    assert_eq!(
        record_ids(llama_record, "prompt_ids", LLAMA_ID_FIXTURE).len(),
        5,
        "the captured prompt tokenizes to five ids"
    );
    assert_eq!(
        record_ids(llama_record, "generated_ids", LLAMA_ID_FIXTURE).len(),
        16,
        "the capture ran sixteen greedy steps"
    );

    let tokenizer = json_fixture(TOKENIZER_ID_FIXTURE);
    let tokenizer_record = first_record(&tokenizer, TOKENIZER_ID_FIXTURE);
    let sample_ids = record_ids(tokenizer_record, "ids", TOKENIZER_ID_FIXTURE);
    assert_eq!(
        sample_ids.len(),
        22,
        "the captured sample text tokenizes to twenty-two ids"
    );

    let vocab = header_entry(&header, "tokenizer.ggml.tokens")
        .array_len()
        .unwrap_or_else(|| panic!("tokenizer.ggml.tokens is not an array"));
    assert_eq!(vocab, 150_144);
    assert!(
        sample_ids
            .iter()
            .all(|&id| u32::try_from(vocab).is_ok_and(|vocab| id < vocab)),
        "every captured tokenizer id indexes the captured vocabulary"
    );
}

#[test]
fn flint_oracle_control_ids_pass_the_parity_comparator() {
    let llama = json_fixture(LLAMA_ID_FIXTURE);
    let record = first_record(&llama, LLAMA_ID_FIXTURE);
    let prompt_ids = record_ids(record, "prompt_ids", LLAMA_ID_FIXTURE);
    let generated_ids = record_ids(record, "generated_ids", LLAMA_ID_FIXTURE);

    assert_eq!(
        first_id_divergence(&prompt_ids, &prompt_ids),
        None,
        "the oracle control feeds the captured ids back through the comparator"
    );
    assert_eq!(
        first_id_divergence(&generated_ids, &generated_ids),
        None,
        "the captured greedy run must match itself"
    );

    let mut mutated = generated_ids.clone();
    mutated[3] ^= 1;
    assert_eq!(
        first_id_divergence(&generated_ids, &mutated),
        Some(3),
        "a comparator that cannot report a planted divergence proves nothing"
    );
    assert_eq!(
        first_id_divergence(&generated_ids, &generated_ids[..generated_ids.len() - 1]),
        Some(generated_ids.len() - 1),
        "a truncated run diverges at the shorter length, not at a false match"
    );
}

#[test]
fn flint_fixture_control_rejects_a_mutated_capture() {
    let header = parse_header(&read_fixture(HEADER_FIXTURE));
    assert_eq!(
        header_problem(&header),
        None,
        "the captured header passes its own comparison"
    );
    let mut mutated_header = header.clone();
    for entry in &mut mutated_header {
        if entry.key == "flint.block_count" {
            entry.value = String::from("48");
        }
    }
    let problem = header_problem(&mutated_header)
        .unwrap_or_else(|| panic!("a mutated block_count must fail the header comparison"));
    assert!(
        problem.contains("flint.block_count"),
        "the failure names the mutated key, got: {problem}"
    );

    let rows = parse_tensor_rows(&read_fixture(TENSOR_FIXTURE));
    assert_eq!(
        tensor_problem(&rows),
        None,
        "the captured tensor directory passes its own comparison"
    );
    let mut mutated_rows = rows.clone();
    mutated_rows.pop();
    assert!(
        tensor_problem(&mutated_rows).is_some(),
        "a directory one tensor short must fail the tensor comparison"
    );

    let mut renamed_rows = rows.clone();
    if let Some(last) = renamed_rows.last_mut() {
        last.name = String::from("blk.48.ffn_up_exps.weight");
    }
    let problem = tensor_problem(&renamed_rows)
        .unwrap_or_else(|| panic!("a renamed last tensor must fail the tensor comparison"));
    assert!(
        problem.contains("blk.48.ffn_up_shexp.weight"),
        "the failure names the expected last tensor, got: {problem}"
    );

    let mut short_array = header.clone();
    for entry in &mut short_array {
        if entry.key == "flint.attention.rope_pattern" {
            entry.kind = String::from("arr[bool,48]");
        }
    }
    let problem = header_problem(&short_array)
        .unwrap_or_else(|| panic!("a 48-entry pattern array must fail the header comparison"));
    assert!(
        problem.contains("flint.attention.rope_pattern"),
        "the failure names the short array, got: {problem}"
    );
}
