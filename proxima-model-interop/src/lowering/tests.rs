use alloc::string::ToString;

use proxima_gguf::value::MetadataValue as Value;

use super::*;
use crate::bind::SlidingRope;

fn gemma4_shaped_hparams(sliding_rope: Option<SlidingRope>) -> ModelHparams {
    ModelHparams {
        vocab: 262_144,
        embedding: 1536,
        feed_forward: 6144,
        query_heads: 8,
        kv_heads: 1,
        kv_heads_by_layer: vec![1; 35],
        head_dim: 512,
        block_count: 35,
        expert_count: 0,
        expert_used_count: 0,
        rope_freq_base: 1_000_000.0,
        rms_epsilon: 1e-6,
        tied_embeddings: true,
        family: "gemma4".to_string(),
        sliding_rope,
    }
}

#[test]
fn bind_symbols_rejects_new_count_above_one_when_single_position_step_is_set() {
    match bind_symbols(2, 8, &[], true) {
        Err(InteropError::MultiPositionStepUnsupported { new_count }) => {
            assert_eq!(new_count, 2);
        }
        other => panic!("expected MultiPositionStepUnsupported, got {other:?}"),
    }
}

#[test]
fn bind_symbols_allows_new_count_one_when_single_position_step_is_set() {
    let bound = bind_symbols(1, 8, &[], true).expect("new_count == 1 is always allowed");
    assert_eq!(bound[symbols::NEW_COUNT as usize], 1);
    assert_eq!(bound[symbols::KV_BOUND as usize], 8);
}

#[test]
fn bind_symbols_allows_new_count_above_one_when_single_position_step_is_unset() {
    let bound = bind_symbols(4, 8, &[], false)
        .expect("batched prefill is unaffected by the flag when it is false");
    assert_eq!(bound[symbols::NEW_COUNT as usize], 4);
}

#[test]
fn a_family_with_no_profile_is_refused_by_name_before_anything_binds() {
    let parsed = crate::test_support::parsed_header(vec![(
        "general.architecture",
        Value::String("phi9".to_string()),
    )]);

    match bind_checkpoint(&parsed, &[]) {
        Err(InteropError::MissingFamilyProfile { family }) => assert_eq!(family, "phi9"),
        Ok(_) => panic!("a family with no profile must not bind"),
        Err(other) => panic!("expected MissingFamilyProfile, got {other}"),
    }
}

#[test]
fn a_sliding_rope_adds_a_cos_and_sin_row_per_new_position() {
    let hparams = gemma4_shaped_hparams(Some(SlidingRope { freq_base: 10_000.0, dimension_count: 256 }));
    let mut inputs = Vec::new();

    sliding_rope_inputs(&hparams, 40, 3, &mut inputs);

    let names: Vec<&str> = inputs.iter().map(|input| input.name).collect();
    assert_eq!(names, ["rope_cos_swa", "rope_sin_swa"]);
    assert!(inputs.iter().all(|input| input.values.len() == 3 * 128));
    assert!(inputs.iter().all(|input| input.symbol.is_none()));
}

#[test]
fn no_sliding_rope_adds_no_step_inputs() {
    let hparams = gemma4_shaped_hparams(None);
    let mut inputs = Vec::new();

    sliding_rope_inputs(&hparams, 0, 1, &mut inputs);

    assert!(inputs.is_empty());
}

#[test]
fn rope_freq_factors_read_the_bound_tensor_by_name_and_none_when_absent() {
    let mut weights = BoundWeights::new(&[]);
    assert!(rope_freq_factors(&weights).is_none());

    weights.owned.push(("rope_freqs.weight".to_string(), vec![1.0, 1.0, 1.0e30]));

    assert_eq!(rope_freq_factors(&weights), Some([1.0, 1.0, 1.0e30].as_slice()));
}

/// The measured `context_length` of each real checkpoint (`ollama
/// /api/show`, 2026-09-29), each written into a GGUF header by the real
/// encoder and read back.
#[proxima::test]
#[case::gemma4_e2b_reads_131072("gemma4", 131_072)]
#[case::qwen35moe_a3b_reads_262144("qwen35moe", 262_144)]
#[case::qwen3_8b_dense_reads_40960("qwen3", 40_960)]
async fn trained_context_read(#[case] family: &'static str, #[case] expected: u32) {
    let context_key = alloc::format!("{family}.context_length");
    let parsed = crate::test_support::parsed_header(vec![
        ("general.architecture", Value::String(family.to_string())),
        (context_key.as_str(), Value::U32(expected)),
    ]);

    assert_eq!(trained_context_length(&parsed), Some(expected));
}

#[test]
fn trained_context_absent_key_reads_none() {
    let parsed = crate::test_support::parsed_header(vec![(
        "general.architecture",
        Value::String("qwen3".to_string()),
    )]);

    assert_eq!(
        trained_context_length(&parsed),
        None,
        "a header without {{arch}}.context_length must read None, not a default"
    );
}
