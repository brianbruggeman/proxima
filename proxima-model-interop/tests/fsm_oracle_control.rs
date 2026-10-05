#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::Path;

const CHECKPOINTS: [&str; 3] = ["gemma4_e2b", "gemma4_26b", "granite_moe"];

fn tolerance(checkpoint: &str) -> f64 {
    match checkpoint {
        "gemma4_e2b" => 4.673e-3,
        "gemma4_26b" => 3.719e-3,
        "granite_moe" => 2.289e-3,
        other => panic!("no readout tolerance recorded for checkpoint {other}"),
    }
}

fn fixture(checkpoint: &str, name: &str) -> serde_json::Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/llama-parity")
        .join(checkpoint)
        .join(name);
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("{checkpoint}: cannot read {}: {error}", path.display()));
    serde_json::from_str(&text)
        .unwrap_or_else(|error| panic!("{checkpoint}: {} is not json: {error}", path.display()))
}

fn ids_of(value: &serde_json::Value) -> Vec<u32> {
    value
        .as_array()
        .expect("ids are a json array")
        .iter()
        .map(|id| {
            u32::try_from(id.as_u64().expect("an id is an unsigned integer"))
                .expect("an id fits u32")
        })
        .collect()
}

fn first_divergence(expected: &[u32], actual: &[u32]) -> Option<usize> {
    let shared = expected.len().min(actual.len());
    (0..shared)
        .find(|&index| expected[index] != actual[index])
        .or_else(|| (expected.len() != actual.len()).then_some(shared))
}

fn top_logprobs(record: &serde_json::Value) -> Vec<f64> {
    record["steps"]
        .as_array()
        .expect("a record carries a steps array")
        .iter()
        .flat_map(|step| {
            step["top"]
                .as_array()
                .expect("a step carries a top array")
                .iter()
                .map(|entry| entry["logprob"].as_f64().expect("a top entry carries a logprob"))
        })
        .collect()
}

fn max_logprob_gap(left: &serde_json::Value, right: &serde_json::Value) -> f64 {
    let left_values = top_logprobs(left);
    let right_values = top_logprobs(right);
    assert_eq!(
        left_values.len(),
        right_values.len(),
        "records differ in top entry count"
    );
    left_values
        .iter()
        .zip(&right_values)
        .map(|(left_value, right_value)| (left_value - right_value).abs())
        .fold(0.0, f64::max)
}

fn first_record<'a>(value: &'a serde_json::Value, checkpoint: &str, name: &str) -> &'a serde_json::Value {
    value
        .as_array()
        .and_then(|records| records.first())
        .unwrap_or_else(|| panic!("{checkpoint}: {name} holds no record"))
}

fn self_match_ids(name: &str, request: &str) -> usize {
    CHECKPOINTS
        .iter()
        .map(|checkpoint| {
            let value = fixture(checkpoint, name);
            let ids = ids_of(&first_record(&value, checkpoint, name)[request]["generated_ids"]);
            assert!(!ids.is_empty(), "{checkpoint}: {name} {request} has no generated ids");
            assert_eq!(first_divergence(&ids, &ids), None, "{checkpoint}: {name}");
        })
        .count()
}

#[test]
fn fsm_oracle_control_followup_ids_match_themselves() {
    assert_eq!(self_match_ids("followup_ids.json", "turn2"), 3);
}

#[test]
fn fsm_oracle_control_cache_reuse_ids_match_themselves() {
    assert_eq!(self_match_ids("cache_reuse_ids.json", "request2"), 3);
}

#[test]
fn fsm_oracle_control_n_probs_match_themselves() {
    let processed: usize = CHECKPOINTS
        .iter()
        .map(|checkpoint| {
            let records = fixture(checkpoint, "n_probs.json");
            let records = records.as_array().expect("n_probs is a json array of records");
            records
                .iter()
                .map(|record| {
                    assert!(
                        !record["steps"].as_array().expect("steps array").is_empty(),
                        "{checkpoint}: record has no steps"
                    );
                    let gap = max_logprob_gap(record, record);
                    assert!(gap <= tolerance(checkpoint), "{checkpoint}: self gap {gap}");
                })
                .count()
        })
        .sum();
    assert_eq!(processed, 9);
}

#[test]
fn fsm_oracle_control_flipped_id_is_rejected() {
    let value = fixture("gemma4_e2b", "followup_ids.json");
    let original = ids_of(&first_record(&value, "gemma4_e2b", "followup_ids.json")["turn2"]["generated_ids"]);
    let mut flipped = original.clone();
    flipped[7] ^= 1;
    assert_eq!(first_divergence(&original, &flipped), Some(7));
}

#[test]
fn fsm_oracle_control_perturbed_probability_is_rejected() {
    let records = fixture("gemma4_e2b", "n_probs.json");
    let original = &records.as_array().expect("n_probs is a json array")[0];
    let mut perturbed = original.clone();
    let shifted = perturbed["steps"][0]["top"][0]["logprob"].as_f64().expect("logprob")
        + 2.0 * tolerance("gemma4_e2b");
    perturbed["steps"][0]["top"][0]["logprob"] = serde_json::json!(shifted);
    assert!(max_logprob_gap(original, &perturbed) > tolerance("gemma4_e2b"));
}
