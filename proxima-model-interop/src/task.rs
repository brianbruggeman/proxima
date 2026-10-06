//! Task-family detection before a checkpoint enters the decoder loader.
//!
//! GGUF does not guarantee one universal task key, so detection is evidence
//! based and conservative. It is a routing result, not a claim that an
//! encoder graph is already executable by `crate::generate::LoadedModel`.

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use proxima_gguf::pipe::ParsedGguf;
use proxima_gguf::MetadataValue;

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

// llama.h LLAMA_POOLING_TYPE_RANK: the pooling a reranking checkpoint declares
// to attach its classification head
const POOLING_TYPE_RANK: u32 = 4;

fn encoder_task(parsed: &ParsedGguf, architecture: &str) -> Option<ModelTask> {
    let key = |suffix: &str| format!("{architecture}.{suffix}");
    let bidirectional = matches!(
        parsed.metadata_value(&key("attention.causal")),
        Some(MetadataValue::Bool(false))
    );
    match parsed.metadata_value(&key("pooling_type")) {
        Some(pooling) if pooling.as_u32() == Some(POOLING_TYPE_RANK) => Some(ModelTask::Reranker),
        Some(_) => Some(ModelTask::Embedding),
        None if bidirectional => Some(ModelTask::Embedding),
        None => None,
    }
}

fn declares_causal_decoder(parsed: &ParsedGguf, architecture: &str) -> bool {
    let key = |suffix: &str| format!("{architecture}.{suffix}");
    let has_blocks = parsed.metadata_value(&key("block_count")).is_some();
    let pools = parsed.metadata_value(&key("pooling_type")).is_some();
    let bidirectional = matches!(
        parsed.metadata_value(&key("attention.causal")),
        Some(MetadataValue::Bool(false))
    );
    has_blocks && !pools && !bidirectional
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
    } else if let Some(value) = architecture
        .as_deref()
        .and_then(|name| encoder_task(parsed, name))
    {
        evidence.push(format!(
            "encoder keys: {}",
            architecture.as_deref().unwrap_or("")
        ));
        value
    } else if parsed.tensors.iter().any(|tensor| {
        let name = normalized(&tensor.name);
        name.contains("classifier") || name == "scoreweight" || name == "scorebias"
    }) {
        evidence.push("classification/reranking head tensor".to_string());
        ModelTask::SequenceClassification
    } else if architecture
        .as_deref()
        .is_some_and(|value| declares_causal_decoder(parsed, value))
    {
        evidence.push(format!(
            "causal decoder keys: {}",
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
mod tests;
