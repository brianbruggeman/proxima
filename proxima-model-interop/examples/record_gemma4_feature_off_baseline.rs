//! Records, for every `long_` record of the llama.cpp oracle fixture, the
//! first token where this build's greedy gemma4-E2B decode diverges from the
//! oracle, into `tests/fixtures/gemma4_e2b_feature_off_first_divergence.json`.
//! The split-decode gate (`tests/gemma4_attn_split_decode_oracle.rs`) compares
//! its own divergences against this file, so it must be recorded on a build
//! WITHOUT `metal-attn-split-decode`.
//!
//! `PROXIMA_GEMMA4_E2B_GGUF` names the checkpoint.
//!
//! `cargo run -p proxima-model-interop --release --features std,metal --example record_gemma4_feature_off_baseline`

#![allow(clippy::unwrap_used, clippy::expect_used)]

#[cfg(all(
    feature = "metal",
    target_os = "macos",
    not(feature = "metal-attn-split-decode")
))]
mod recorder {
    use std::fs::File;
    use std::path::PathBuf;

    use memmap2::Mmap;
    use proxima_gguf::parse_complete;
    use proxima_gguf::types::GgmlType;
    use proxima_model_interop::{GPU_LAYERS_ALL, LoadedModel, ServingConfig};
    use serde_json::Value;

    const GEMMA4_E2B_DEFAULT_PATH: &str = "/Users/brianbruggeman/.ollama/models/blobs/\
         sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd";

    const EOG_IDS: [u32; 3] = [1, 106, 50];

    struct OracleRecord {
        name: String,
        prompt: String,
        prompt_ids: Vec<u32>,
        oracle_ids: Vec<u32>,
    }

    fn fixtures_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
    }

    fn id_array(record: &Value, field: &str) -> Result<Vec<u32>, String> {
        record[field]
            .as_array()
            .ok_or_else(|| format!("record.{field} is not a json array"))?
            .iter()
            .map(|id| {
                id.as_u64()
                    .and_then(|value| u32::try_from(value).ok())
                    .ok_or_else(|| format!("record.{field} holds a non-u32 id"))
            })
            .collect()
    }

    fn cut_at_eog(ids: &[u32]) -> &[u32] {
        let end = ids
            .iter()
            .position(|id| EOG_IDS.contains(id))
            .unwrap_or(ids.len());
        &ids[..end]
    }

    fn first_divergence(proxima_ids: &[u32], oracle_ids: &[u32]) -> Option<usize> {
        let common = proxima_ids.len().min(oracle_ids.len());
        (0..common)
            .find(|&index| proxima_ids[index] != oracle_ids[index])
            .or((proxima_ids.len() != oracle_ids.len()).then_some(common))
    }

    fn load_oracle_records() -> Result<Vec<OracleRecord>, String> {
        let path = fixtures_dir().join("gemma4_e2b_llama_greedy_ids.json");
        let text = std::fs::read_to_string(&path).map_err(|error| {
            format!(
                "no fixture at {}: {error}; record it with the llama-server recipe in ORACLE.md",
                path.display()
            )
        })?;
        let document: Value = serde_json::from_str(&text)
            .map_err(|error| format!("{} is not valid json: {error}", path.display()))?;
        let records = document
            .as_array()
            .ok_or("the oracle fixture is not a json array of records")?
            .iter()
            .map(|record| {
                Ok(OracleRecord {
                    name: record["name"].as_str().ok_or("record.name")?.to_string(),
                    prompt: record["prompt"]
                        .as_str()
                        .ok_or("record.prompt")?
                        .to_string(),
                    prompt_ids: id_array(record, "prompt_ids")?,
                    oracle_ids: id_array(record, "oracle_ids")?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        if records.is_empty() {
            return Err("an oracle fixture with zero records compares nothing".to_string());
        }
        Ok(records)
    }

    fn report_prompt_ids(bytes: &[u8], records: &[OracleRecord]) -> Result<(), String> {
        let parsed = parse_complete(bytes).map_err(|error| format!("parse header: {error}"))?;
        let vocab = proxima_tokenizer::gguf::vocab_from_metadata(&parsed)
            .map_err(|error| format!("vocab from gguf metadata: {error}"))?;
        for record in records {
            let ids = proxima_tokenizer::encode_with_bos_eos(
                &record.prompt,
                &vocab,
                true,
                vocab.add_eos_token().unwrap_or(false),
            )
            .map_err(|error| format!("{}: the oracle prompt tokenizes: {error}", record.name))?;
            println!(
                "{}: prompt_ids_match={} first_prompt_divergence={:?} proxima_prompt_len={} llama_prompt_len={}",
                record.name,
                ids == record.prompt_ids,
                first_divergence(&ids, &record.prompt_ids),
                ids.len(),
                record.prompt_ids.len()
            );
        }
        Ok(())
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

    fn decode(
        model: &LoadedModel<'_>,
        record: &OracleRecord,
    ) -> Result<(Vec<u32>, String), String> {
        let (ids, text, _stopped_by_eos) = model
            .generate_with_serving_config(&record.prompt, record.oracle_ids.len(), serving_config())
            .map_err(|error| format!("{} greedy decode failed: {error}", record.name))?;
        if ids.is_empty() {
            return Err(format!(
                "{}: a decode that produced zero tokens compares nothing",
                record.name
            ));
        }
        Ok((ids, text))
    }

    fn report_decode(record: &OracleRecord, ids: &[u32], text: &str) -> (Option<usize>, usize) {
        let proxima_cut = cut_at_eog(ids);
        let oracle_cut = cut_at_eog(&record.oracle_ids);
        let divergence = first_divergence(proxima_cut, oracle_cut);
        println!(
            "{}: first_divergence={divergence:?} match_len={} oracle_cut_len={} proxima_raw_len={} proxima_cut_len={} oracle_raw_len={} text={text:?}",
            record.name,
            divergence.unwrap_or(oracle_cut.len()),
            oracle_cut.len(),
            ids.len(),
            proxima_cut.len(),
            record.oracle_ids.len()
        );
        (divergence, oracle_cut.len())
    }

    pub fn run() -> Result<(), String> {
        let records = load_oracle_records()?;
        let path = std::env::var("PROXIMA_GEMMA4_E2B_GGUF")
            .unwrap_or_else(|_| GEMMA4_E2B_DEFAULT_PATH.to_string());
        let file = File::open(&path).map_err(|error| {
            format!("no host-local gguf fixture at {path}: set PROXIMA_GEMMA4_E2B_GGUF to a valid checkpoint path ({error})")
        })?;
        // SAFETY: the checkpoint file is not written or truncated by any other
        // process for the duration of this read-only mapping.
        let mapping = unsafe { Mmap::map(&file) }
            .map_err(|error| format!("mmap the real gemma4-E2B checkpoint: {error}"))?;
        let bytes: &[u8] = &mapping;

        report_prompt_ids(bytes, &records)?;
        let parsed = parse_complete(bytes).map_err(|error| format!("parse header: {error}"))?;
        let model = LoadedModel::load(&parsed, bytes)
            .map_err(|error| format!("bind the real gemma4-E2B checkpoint: {error}"))?;

        let mut baseline = serde_json::Map::new();
        for record in &records {
            let (ids, text) = decode(&model, record)?;
            let (divergence, oracle_cut_len) = report_decode(record, &ids, &text);
            if record.name.starts_with("long_") {
                baseline.insert(
                    record.name.clone(),
                    Value::from(divergence.unwrap_or(oracle_cut_len) as u64),
                );
            }
        }
        if baseline.is_empty() {
            return Err("the oracle fixture carries no long_ records to baseline".to_string());
        }
        let output = fixtures_dir().join("gemma4_e2b_feature_off_first_divergence.json");
        let body = serde_json::to_string_pretty(&Value::Object(baseline))
            .map_err(|error| format!("serialize the baseline: {error}"))?;
        std::fs::write(&output, body)
            .map_err(|error| format!("write {}: {error}", output.display()))
    }
}

#[cfg(all(
    feature = "metal",
    target_os = "macos",
    not(feature = "metal-attn-split-decode")
))]
fn main() -> std::process::ExitCode {
    match recorder::run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("record_gemma4_feature_off_baseline: {message}");
            std::process::ExitCode::FAILURE
        }
    }
}

#[cfg(not(all(
    feature = "metal",
    target_os = "macos",
    not(feature = "metal-attn-split-decode")
)))]
fn main() {
    println!(
        "record_gemma4_feature_off_baseline requires --features std,metal on macOS and a build without metal-attn-split-decode; skipped"
    );
}
