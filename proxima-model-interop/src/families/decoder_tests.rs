//! The decoder-only family's dialects and profile inference on synthetic
//! tensor lists; the host-checkpoint differential is `tests/local_checkpoints.rs`.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use proxima_tensor::spec::{Activation, FfnCombination};
use serde_json::json;

use super::*;

#[test]
fn hf_dialect_reads_qk_norm_and_shared_expert_from_names_and_activation_from_config() {
    let family = DecoderFamily::builtin().expect("embedded family parses");
    let templates = family.templates().expect("templates expand");
    let names = [
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.q_norm.weight",
        "model.layers.0.mlp.shared_expert.gate_proj.weight",
        "model.layers.0.mlp.shared_expert.up_proj.weight",
        "model.layers.0.mlp.shared_expert.down_proj.weight",
        "model.layers.0.mlp.shared_expert_gate.weight",
    ];
    let tensors: Vec<TensorShape> = names
        .iter()
        .map(|name| TensorShape {
            name: (*name).into(),
            shape: vec![8, 8],
        })
        .collect();
    let sha = CheckpointSha::parse(&"0".repeat(64)).expect("64 hex digits");
    let config = json!({"hidden_act": "silu"});

    let inferred = infer(&tensors, &config, &templates, sha).expect("hf dialect matches");
    let facts = infer_profile(&family, &inferred, &config).expect("profile reads");

    assert_eq!(inferred.family, "decoder_only.hf");
    assert_eq!(facts.qk_norm, Slot::Known(true));
    assert_eq!(
        facts.layer_combination,
        Slot::Known(FfnCombination::RoutedWithSharedExpert)
    );
    assert_eq!(facts.shared_expert_gate, Slot::Known(true));
    assert_eq!(facts.activation, Slot::Known(Activation::Silu));
    assert_eq!(facts.rope_layout, Slot::Unresolved);
}

#[test]
fn a_checkpoint_with_no_attention_projection_is_not_a_decoder() {
    let family = DecoderFamily::builtin().expect("embedded family parses");
    let templates = family.templates().expect("templates expand");
    let tensors = [TensorShape {
        name: "blk.0.ssm_a".into(),
        shape: vec![4],
    }];
    let sha = CheckpointSha::parse(&"0".repeat(64)).expect("64 hex digits");

    let error = infer(&tensors, &json!({}), &templates, sha).expect_err("no attn_q anywhere");

    assert!(matches!(error, InferError::NoFamily { .. }));
}

#[test]
fn another_familys_tensors_in_the_file_are_counted_not_hidden() {
    let family = DecoderFamily::builtin().expect("embedded family parses");
    let templates = family.templates().expect("templates expand");
    let names = [
        "token_embd.weight",
        "output_norm.weight",
        "blk.0.attn_q.weight",
        "v.blk.0.attn_q.weight",
        "mm.mm_input_projection.weight",
    ];
    let tensors: Vec<TensorShape> = names
        .iter()
        .map(|name| TensorShape {
            name: (*name).into(),
            shape: vec![8, 8],
        })
        .collect();
    let sha = CheckpointSha::parse(&"0".repeat(64)).expect("64 hex digits");

    let inferred = infer(&tensors, &json!({}), &templates, sha).expect("companions are claimed");

    assert_eq!(inferred.facts["companion_tensors"].value, Value::Int(2));
}

#[test]
fn the_embedded_facts_table_parses_and_every_section_states_residue_only() {
    let table = FactsTable::builtin().expect("embedded facts parse");
    let sections = table.iter().count();

    assert!(sections >= 16, "one section per architecture, got {sections}");
    assert!(table.iter().all(|(_, facts)| facts.field_count() >= 4));
}

#[test]
fn a_fact_that_disagrees_with_a_known_slot_is_refused_naming_field_and_architecture() {
    let family = DecoderFamily::builtin().expect("embedded family parses");
    let templates = family.templates().expect("templates expand");
    let tensors = [
        "token_embd.weight",
        "output_norm.weight",
        "blk.0.attn_q.weight",
    ]
    .map(|name| TensorShape {
        name: name.into(),
        shape: vec![8, 8],
    });
    let sha = CheckpointSha::parse(&"0".repeat(64)).expect("64 hex digits");
    let config = json!({"embedding_scale": 12.0});
    let inferred = infer(&tensors, &config, &templates, sha).expect("gguf dialect matches");
    let facts = infer_profile(&family, &inferred, &config).expect("profile reads");
    let stated = ArchFacts {
        embedding_scale: Some(EmbeddingFact::None),
        ..ArchFacts::default()
    };

    let refused = facts.settle(&stated, "invented").expect_err("a contradiction is refused");

    assert!(
        matches!(&refused, FactsError::Contradicts { field: "embedding_scale", architecture, .. } if architecture == "invented"),
        "{refused}"
    );
}

#[test]
fn an_open_field_after_the_facts_is_a_typed_error_naming_it() {
    let family = DecoderFamily::builtin().expect("embedded family parses");
    let templates = family.templates().expect("templates expand");
    let layer = "blk.0.";
    let projection = format!("{layer}attn_q.weight");
    let tensors = ["token_embd.weight", projection.as_str()].map(|name| TensorShape {
        name: name.into(),
        shape: vec![8, 8],
    });
    let sha = CheckpointSha::parse(&"0".repeat(64)).expect("64 hex digits");
    let config = json!({});
    let inferred = infer(&tensors, &config, &templates, sha).expect("gguf dialect matches");
    let facts = infer_profile(&family, &inferred, &config).expect("profile reads");

    let open = facts.into_profile("invented").expect_err("rope_layout is stated nowhere");

    assert!(
        matches!(&open, FactsError::Unresolved { architecture, fields } if architecture == "invented" && fields.contains("rope_layout")),
        "{open}"
    );
}
