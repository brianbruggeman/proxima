use super::*;
use alloc::vec;
use arrayvec::ArrayVec;
use proxima_gguf::{GgmlType, MetadataValue, TensorInfo};

fn parsed(architecture: &str, task: Option<&str>, head: Option<&str>) -> ParsedGguf {
    let mut metadata = vec![
        (
            "general.architecture".to_string(),
            MetadataValue::String(architecture.to_string()),
        ),
        (
            format!("{architecture}.block_count"),
            MetadataValue::U32(28),
        ),
    ];
    if let Some(task) = task {
        metadata.push((
            "general.task".to_string(),
            MetadataValue::String(task.to_string()),
        ));
    }
    let tensors: Vec<TensorInfo> = head
        .map(|name| {
            let mut dims = ArrayVec::<u64, 4>::new();
            dims.push(1);
            TensorInfo {
                name: name.to_string(),
                dims,
                ggml_type: GgmlType::F32,
                offset: 0,
            }
        })
        .into_iter()
        .collect();
    ParsedGguf {
        version: 3,
        tensor_count: tensors.len() as u64,
        kv_count: metadata.len() as u64,
        metadata,
        tensors,
        data_offset: 0,
        alignment: 32,
    }
}

#[test]
fn explicit_reranker_wins() {
    let profile = classify_task(&parsed("bert", Some("reranking"), None));
    assert_eq!(profile.task, ModelTask::Reranker);
    assert!(!profile.generation_supported);
}

#[test]
fn model_name_distinguishes_qwen_embedding_from_decoder() {
    let mut checkpoint = parsed("qwen3", None, None);
    checkpoint.metadata.push((
        "general.name".to_string(),
        MetadataValue::String("Qwen3 Embedding 0.6B".to_string()),
    ));
    let profile = classify_task(&checkpoint);
    assert_eq!(profile.task, ModelTask::Embedding);
}

#[test]
fn decoder_family_is_generation() {
    let profile = classify_task(&parsed("qwen3", None, None));
    assert_eq!(profile.task, ModelTask::CausalGeneration);
    assert!(profile.generation_supported);
}

#[test]
fn a_future_family_with_block_keys_and_no_pooling_is_a_decoder() {
    let profile = classify_task(&parsed("some_future_family", None, None));
    assert_eq!(profile.task, ModelTask::CausalGeneration);
}

#[test]
fn pooling_type_key_marks_an_encoder_whatever_the_family_is_called() {
    let mut checkpoint = parsed("qwen3", None, None);
    checkpoint.metadata.push((
        "qwen3.pooling_type".to_string(),
        MetadataValue::U32(3),
    ));
    assert_eq!(classify_task(&checkpoint).task, ModelTask::Embedding);
}

#[test]
fn non_causal_attention_key_is_an_embedding_encoder() {
    let mut checkpoint = parsed("llama", None, None);
    checkpoint.metadata.push((
        "llama.attention.causal".to_string(),
        MetadataValue::Bool(false),
    ));
    assert_eq!(classify_task(&checkpoint).task, ModelTask::Embedding);
}

#[test]
fn a_checkpoint_with_no_block_keys_is_unknown() {
    let mut checkpoint = parsed("llama", None, None);
    checkpoint.metadata.retain(|(key, _)| key != "llama.block_count");
    assert_eq!(classify_task(&checkpoint).task, ModelTask::Unknown);
}

#[test]
fn classifier_head_is_not_a_decoder() {
    let profile = classify_task(&parsed("unknown", None, Some("classifier.weight")));
    assert_eq!(profile.task, ModelTask::SequenceClassification);
}

fn encoder(architecture: &str, block_count: u32, pooling_type: u32) -> ParsedGguf {
    let mut checkpoint = parsed(architecture, None, None);
    checkpoint.metadata.retain(|(key, _)| !key.ends_with(".block_count"));
    checkpoint.metadata.extend([
        (
            format!("{architecture}.block_count"),
            MetadataValue::U32(block_count),
        ),
        (
            format!("{architecture}.attention.causal"),
            MetadataValue::Bool(false),
        ),
        (
            format!("{architecture}.pooling_type"),
            MetadataValue::U32(pooling_type),
        ),
    ]);
    checkpoint
}

#[test]
fn a_bert_header_with_mean_pooling_is_an_embedding_model() {
    let profile = classify_task(&encoder("bert", 6, 1));
    assert_eq!(profile.task, ModelTask::Embedding);
    assert!(!profile.generation_supported);
    assert_eq!(profile.evidence, ["encoder keys: bert"]);
}

#[test]
fn a_nomic_bert_header_with_mean_pooling_is_an_embedding_model() {
    assert_eq!(classify_task(&encoder("nomic-bert", 12, 1)).task, ModelTask::Embedding);
}

#[test]
fn rank_pooling_marks_a_reranker_whatever_the_family_is_called() {
    assert_eq!(classify_task(&encoder("bert", 12, 4)).task, ModelTask::Reranker);
    assert_eq!(classify_task(&encoder("some_future_family", 24, 4)).task, ModelTask::Reranker);
}

#[test]
fn an_architecture_named_like_an_encoder_with_decoder_keys_is_a_decoder() {
    let profile = classify_task(&parsed("bert_style_decoder", None, None));
    assert_eq!(profile.task, ModelTask::CausalGeneration);
}
