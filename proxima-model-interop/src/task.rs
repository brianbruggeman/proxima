//! Task-family detection before a checkpoint enters the decoder loader.
//!
//! GGUF does not guarantee one universal task key, so detection is evidence
//! based and conservative. It is a routing result, not a claim that an
//! encoder graph is already executable by [`crate::generate::LoadedModel`].

use alloc::string::{String, ToString};
use alloc::vec::Vec;

use proxima_gguf::pipe::ParsedGguf;

/// The model-level task family a serving harness must choose.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelTask {
    CausalGeneration,
    Embedding,
    Reranker,
    SequenceClassification,
    Unknown,
}

impl ModelTask {
    #[must_use]
    pub const fn is_generation(self) -> bool {
        matches!(self, Self::CausalGeneration)
    }

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::CausalGeneration => "causal-generation",
            Self::Embedding => "embedding",
            Self::Reranker => "reranker",
            Self::SequenceClassification => "sequence-classification",
            Self::Unknown => "unknown",
        }
    }
}

/// Detection result retained for benchmark output and routing diagnostics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskProfile {
    pub task: ModelTask,
    pub architecture: Option<String>,
    pub evidence: Vec<String>,
    pub generation_supported: bool,
}

fn normalized(value: &str) -> String {
    value.to_ascii_lowercase().replace(['_', '-', ' '], "")
}

fn explicit_task(parsed: &ParsedGguf) -> Option<&str> {
    ["general.task", "general.model_task", "general.pipeline_tag"]
        .iter()
        .find_map(|key| parsed.metadata_value(key).and_then(|value| value.as_str()))
}

fn model_identity(parsed: &ParsedGguf) -> Option<&str> {
    ["general.name", "general.basename", "general.model_name"]
        .iter()
        .find_map(|key| parsed.metadata_value(key).and_then(|value| value.as_str()))
}

fn task_from_word(value: &str) -> Option<ModelTask> {
    let value = normalized(value);
    if value.contains("rerank") || value.contains("crossencoder") {
        Some(ModelTask::Reranker)
    } else if value.contains("embedding") || value.contains("featureextraction") {
        Some(ModelTask::Embedding)
    } else if value.contains("classification") || value.contains("sequenceclass") {
        Some(ModelTask::SequenceClassification)
    } else if value.contains("textgeneration") || value.contains("causallm") {
        Some(ModelTask::CausalGeneration)
    } else {
        None
    }
}

fn architecture_task(architecture: &str) -> Option<ModelTask> {
    let value = normalized(architecture);
    if value.contains("reranker") || value.contains("crossencoder") {
        Some(ModelTask::Reranker)
    } else if value.contains("bert") || value.contains("encoder") {
        Some(ModelTask::Embedding)
    } else {
        None
    }
}

/// Classifies a parsed checkpoint without reading tensor payload bytes.
#[must_use]
pub fn classify_task(parsed: &ParsedGguf) -> TaskProfile {
    let architecture = parsed
        .metadata_value("general.architecture")
        .and_then(|value| value.as_str())
        .map(ToString::to_string);
    let mut evidence = Vec::new();

    let task = if let Some(value) = explicit_task(parsed) {
        if let Some(task) = task_from_word(value) {
            evidence.push(format!("explicit task metadata: {value}"));
            task
        } else {
            evidence.push(format!("unrecognized task metadata: {value}"));
            ModelTask::Unknown
        }
    } else if let Some(value) = model_identity(parsed).and_then(task_from_word) {
        evidence.push("model identity metadata".to_string());
        value
    } else if let Some(value) = architecture.as_deref().and_then(architecture_task) {
        evidence.push(format!(
            "architecture family: {}",
            architecture.as_deref().unwrap_or("")
        ));
        value
    } else if parsed.tensors.iter().any(|tensor| {
        let name = normalized(&tensor.name);
        name.contains("classifier") || name == "scoreweight" || name == "scorebias"
    }) {
        evidence.push("classification/reranking head tensor".to_string());
        ModelTask::SequenceClassification
    } else if architecture.as_deref().is_some_and(|value| {
        let value = normalized(value);
        value.contains("llama")
            || value.contains("mistral")
            || value.contains("qwen")
            || value.contains("mixtral")
            || value.contains("lfm")
    }) {
        evidence.push(format!(
            "decoder architecture family: {}",
            architecture.as_deref().unwrap_or("")
        ));
        ModelTask::CausalGeneration
    } else {
        evidence.push("no task or known architecture evidence".to_string());
        ModelTask::Unknown
    };

    TaskProfile {
        task,
        architecture,
        evidence,
        generation_supported: task.is_generation(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrayvec::ArrayVec;
    use proxima_gguf::{GgmlType, MetadataValue, TensorInfo};

    fn parsed(architecture: &str, task: Option<&str>, head: Option<&str>) -> ParsedGguf {
        let mut metadata = vec![(
            "general.architecture".to_string(),
            MetadataValue::String(architecture.to_string()),
        )];
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
    fn classifier_head_is_not_a_decoder() {
        let profile = classify_task(&parsed("unknown", None, Some("classifier.weight")));
        assert_eq!(profile.task, ModelTask::SequenceClassification);
    }
}
