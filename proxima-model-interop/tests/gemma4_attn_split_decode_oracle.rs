//! llama.cpp greedy-id gate for the split-KV decode attention form. The
//! oracle is llama.cpp's own greedy token ids for the same prompts on the same
//! blob (recorded by the recipe in `ORACLE.md`, stored in
//! `tests/fixtures/gemma4_e2b_llama_greedy_ids.json`), never this crate's CPU
//! path or its feature-off Metal output.
//!
//! Admission, with `metal-attn-split-decode` on:
//! - the four gate prompts (`gemma4_correctness_gate.rs`) match the oracle
//!   exactly over their full recorded length;
//! - on every long-context record the first divergence from the oracle is no
//!   earlier than the feature-off build's, read from
//!   `tests/fixtures/gemma4_e2b_feature_off_first_divergence.json`, which the
//!   feature-off recorder below writes from the same oracle records;
//! - the answer substring holds for every record that names one.
//!
//! Every absent fixture fails loudly, naming the recipe: an `#[ignore]`d gate
//! that returned success having compared nothing would be indistinguishable
//! from a pass. A gate that decodes zero records or zero tokens is red.

#![cfg(all(feature = "metal", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::fs::File;
use std::path::PathBuf;

use memmap2::Mmap;
use proxima_gguf::parse_complete;
use proxima_gguf::types::GgmlType;
use proxima_model_interop::{GPU_LAYERS_ALL, LoadedModel, ServingConfig};
use serde_json::Value;

const GEMMA4_E2B_DEFAULT_PATH: &str = "/Users/brianbruggeman/.ollama/models/blobs/\
     sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd";

/// End-of-generation ids of the gemma4 vocabulary: `<eos>`, `<turn|>` and
/// the tool-response closer. llama.cpp records the id that stopped it as the
/// last oracle id; this crate excludes the stop token from returned ids, so
/// both streams are cut at the first of these before they are compared.
const EOG_IDS: [u32; 3] = [1, 106, 50];

const GATE_PROMPTS: [&str; 4] = [
    "paris",
    "soliloquy",
    "ant_vs_briefcase",
    "hippo_vs_building",
];

struct OracleRecord {
    name: String,
    prompt: String,
    prompt_ids: Vec<u32>,
    oracle_ids: Vec<u32>,
    #[cfg(feature = "metal-attn-split-decode")]
    expected_substring: Option<String>,
}

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn oracle_path() -> PathBuf {
    fixtures_dir().join("gemma4_e2b_llama_greedy_ids.json")
}

fn baseline_path() -> PathBuf {
    fixtures_dir().join("gemma4_e2b_feature_off_first_divergence.json")
}

fn read_fixture(path: &PathBuf, recipe: &str) -> Value {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("no fixture at {}: {error}; {recipe}", path.display()));
    serde_json::from_str(&text)
        .unwrap_or_else(|error| panic!("{} is not valid json: {error}", path.display()))
}

fn id_array(record: &Value, field: &str) -> Vec<u32> {
    record[field]
        .as_array()
        .unwrap_or_else(|| panic!("record.{field} is a json array"))
        .iter()
        .map(|id| u32::try_from(id.as_u64().expect("an id is an integer")).expect("an id fits u32"))
        .collect()
}

fn cut_at_eog(ids: &[u32]) -> &[u32] {
    let end = ids
        .iter()
        .position(|id| EOG_IDS.contains(id))
        .unwrap_or(ids.len());
    &ids[..end]
}

fn load_oracle_records() -> Vec<OracleRecord> {
    let document = read_fixture(
        &oracle_path(),
        "record it with the llama-server recipe in ORACLE.md",
    );
    let records: Vec<OracleRecord> = document
        .as_array()
        .expect("the oracle fixture is a json array of records")
        .iter()
        .map(|record| OracleRecord {
            name: record["name"].as_str().expect("record.name").to_string(),
            prompt: record["prompt"]
                .as_str()
                .expect("record.prompt")
                .to_string(),
            prompt_ids: id_array(record, "prompt_ids"),
            oracle_ids: id_array(record, "oracle_ids"),
            #[cfg(feature = "metal-attn-split-decode")]
            expected_substring: record["expected_substring"].as_str().map(str::to_string),
        })
        .collect();
    assert!(
        !records.is_empty(),
        "an oracle fixture with zero records compares nothing"
    );
    for gate in GATE_PROMPTS {
        assert!(
            records.iter().any(|record| record.name == gate),
            "the oracle fixture must carry the gate prompt {gate}"
        );
    }
    records
}

/// Prints, per record, whether this crate's tokenizer turns the record's
/// prompt text into llama.cpp's recorded prompt ids, and where it first
/// differs: a tokenizer disagreement would confound every comparison below.
fn report_prompt_ids(bytes: &[u8], records: &[OracleRecord]) {
    let parsed = parse_complete(bytes).expect("parse the real gemma4-E2B checkpoint header");
    let vocab =
        proxima_tokenizer::gguf::vocab_from_metadata(&parsed).expect("vocab from gguf metadata");
    for record in records {
        let ids = proxima_tokenizer::encode_with_bos_eos(
            &record.prompt,
            &vocab,
            true,
            vocab.add_eos_token().unwrap_or(false),
        )
        .expect("the oracle prompt tokenizes");
        println!(
            "{}: prompt_ids_match={} first_prompt_divergence={:?} proxima_prompt_len={} llama_prompt_len={}",
            record.name,
            ids == record.prompt_ids,
            first_divergence(&ids, &record.prompt_ids),
            ids.len(),
            record.prompt_ids.len()
        );
    }
}

fn load_model(bytes: &[u8]) -> LoadedModel<'_> {
    let parsed = parse_complete(bytes).expect("parse the real gemma4-E2B checkpoint header");
    LoadedModel::load(&parsed, bytes).expect("bind the real gemma4-E2B checkpoint")
}

fn serving_config() -> ServingConfig<'static> {
    ServingConfig {
        gpu_layers: GPU_LAYERS_ALL,
        kv_cache_key_quant: GgmlType::F32,
        kv_cache_value_quant: GgmlType::F32,
        flash_attention: false,
        batch_size: 0,
        ubatch_size: 0,
        reasoning_budget: 0,
        ..ServingConfig::default()
    }
}

/// Index of the first token where the streams differ; the shorter length when
/// one is a prefix of the other; `None` when they are equal.
fn first_divergence(proxima_ids: &[u32], oracle_ids: &[u32]) -> Option<usize> {
    let common = proxima_ids.len().min(oracle_ids.len());
    (0..common)
        .find(|&index| proxima_ids[index] != oracle_ids[index])
        .or((proxima_ids.len() != oracle_ids.len()).then_some(common))
}

/// First divergence between a decode and the oracle once both are cut at
/// their first end-of-generation id, and the length of the cut oracle.
fn compare_until_eog(proxima_ids: &[u32], oracle_ids: &[u32]) -> (Option<usize>, usize) {
    let oracle_cut = cut_at_eog(oracle_ids);
    (
        first_divergence(cut_at_eog(proxima_ids), oracle_cut),
        oracle_cut.len(),
    )
}

/// Prints one record's decode against the oracle (match length, first
/// divergence, both raw and cut lengths) and returns the same comparison.
fn report_decode(record: &OracleRecord, ids: &[u32], text: &str) -> (Option<usize>, usize) {
    let proxima_cut = cut_at_eog(ids);
    let oracle_cut = cut_at_eog(&record.oracle_ids);
    let (divergence, oracle_cut_len) = compare_until_eog(ids, &record.oracle_ids);
    println!(
        "{}: first_divergence={divergence:?} match_len={} oracle_cut_len={oracle_cut_len} proxima_raw_len={} proxima_cut_len={} oracle_raw_len={} text={text:?}",
        record.name,
        divergence.unwrap_or(oracle_cut_len),
        ids.len(),
        proxima_cut.len(),
        record.oracle_ids.len()
    );
    if let Some(index) = divergence {
        println!(
            "{}: divergent step {index}: proxima={:?} oracle={:?}",
            record.name,
            proxima_cut.get(index),
            oracle_cut.get(index)
        );
    }
    (divergence, oracle_cut_len)
}

fn decode(model: &LoadedModel<'_>, record: &OracleRecord) -> (Vec<u32>, String) {
    let (ids, text, _stopped_by_eos) = model
        .generate_with_serving_config(&record.prompt, record.oracle_ids.len(), serving_config())
        .unwrap_or_else(|error| panic!("{} greedy decode failed: {error}", record.name));
    assert!(
        !ids.is_empty(),
        "{}: a decode that produced zero tokens compares nothing",
        record.name
    );
    (ids, text)
}

fn with_mapped_blob<T>(body: impl FnOnce(&[u8]) -> T) -> T {
    let path = std::env::var("PROXIMA_GEMMA4_E2B_GGUF")
        .unwrap_or_else(|_| GEMMA4_E2B_DEFAULT_PATH.to_string());
    assert!(
        std::path::Path::new(&path).exists(),
        "no host-local gguf fixture at {path}: set PROXIMA_GEMMA4_E2B_GGUF to a valid checkpoint path, or stage one at this default path"
    );
    let file = File::open(&path).unwrap_or_else(|error| panic!("open {path}: {error}"));
    // SAFETY: the checkpoint file is not written or truncated by any other
    // process for the duration of this read-only mapping.
    let mapping = unsafe { Mmap::map(&file) }.expect("mmap the real gemma4-E2B checkpoint");
    body(&mapping)
}

#[cfg(feature = "metal-attn-split-decode")]
#[proxima::test]
#[ignore = "requires the real gemma4 E2B blob and the llama.cpp oracle fixture (ORACLE.md)"]
async fn the_split_decode_form_matches_the_llama_cpp_greedy_ids() {
    let records = load_oracle_records();
    let baseline = read_fixture(
        &baseline_path(),
        "record it by running `record_feature_off_first_divergences` on a build without metal-attn-split-decode",
    );
    let mut failures = Vec::new();
    let mut decoded_tokens = 0_usize;

    with_mapped_blob(|bytes| {
        report_prompt_ids(bytes, &records);
        let model = load_model(bytes);
        for record in &records {
            let (ids, text) = decode(&model, record);
            decoded_tokens += ids.len();
            let (divergence, oracle_cut_len) = report_decode(record, &ids, &text);

            if GATE_PROMPTS.contains(&record.name.as_str()) && divergence.is_some() {
                failures.push(format!(
                    "{}: gate prompt diverged from the oracle at {divergence:?}",
                    record.name
                ));
            }
            if let Some(expected) = &record.expected_substring
                && !text.to_lowercase().contains(expected.as_str())
            {
                failures.push(format!(
                    "{}: expected substring {expected:?} not found in {text:?}",
                    record.name
                ));
            }
            if record.name.starts_with("long_") {
                let off = baseline[record.name.as_str()].as_u64().unwrap_or_else(|| {
                    panic!(
                        "{}: no feature-off baseline divergence recorded",
                        record.name
                    )
                });
                let on = divergence.unwrap_or(oracle_cut_len) as u64;
                if on < off {
                    failures.push(format!(
                        "{}: first divergence {on} is earlier than the feature-off {off}",
                        record.name
                    ));
                }
            }
        }
    });

    assert!(
        decoded_tokens > 0,
        "zero decoded tokens across {} records is a red gate",
        records.len()
    );
    assert!(
        failures.is_empty(),
        "split-decode oracle gate failed:\n{}",
        failures.join("\n")
    );
}

#[cfg(not(feature = "metal-attn-split-decode"))]
#[proxima::test]
#[ignore = "recorder: run on a build WITHOUT metal-attn-split-decode to write the feature-off baseline"]
async fn record_feature_off_first_divergences() {
    let records = load_oracle_records();
    let mut baseline = serde_json::Map::new();
    with_mapped_blob(|bytes| {
        report_prompt_ids(bytes, &records);
        let model = load_model(bytes);
        for record in &records {
            let (ids, text) = decode(&model, record);
            let (divergence, oracle_cut_len) = report_decode(record, &ids, &text);
            if record.name.starts_with("long_") {
                baseline.insert(
                    record.name.clone(),
                    Value::from(divergence.unwrap_or(oracle_cut_len) as u64),
                );
            }
        }
    });
    assert!(
        !baseline.is_empty(),
        "the oracle fixture carries no long_ records to baseline"
    );
    std::fs::write(
        baseline_path(),
        serde_json::to_string_pretty(&Value::Object(baseline)).expect("serializes"),
    )
    .expect("the baseline fixture is writable");
}
