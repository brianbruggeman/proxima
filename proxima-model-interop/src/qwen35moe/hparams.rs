use alloc::format;
use alloc::vec::Vec;

use proxima_gguf::pipe::ParsedGguf;
use proxima_gguf::value::{MetadataArray, MetadataValue};

use crate::bind::{metadata_f32_optional, metadata_str, metadata_u32, metadata_u32_optional_or};
use crate::error::InteropError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    Gdn,
    Attention,
}

impl LayerKind {
    #[must_use]
    pub fn from_interval(layer: u32, interval: u32) -> Self {
        if interval != 0 && (layer + 1).is_multiple_of(interval) {
            Self::Attention
        } else {
            Self::Gdn
        }
    }
}

#[derive(Debug, Clone)]
pub struct Architecture {
    pub vocab: u32,
    pub embedding: u32,
    pub query_heads: u32,
    pub kv_heads_by_layer: Vec<u32>,
    pub attn_head_dim: u32,
    pub rope_dims: u32,
    pub rope_dimension_sections: Vec<u32>,
    pub rope_mrope_interleaved: bool,
    pub block_count: u32,
    pub full_attention_interval: u32,
    pub rope_freq_base: f32,
    pub rms_epsilon: f32,
    pub ssm_conv_kernel: u32,
    pub ssm_state_size: u32,
    pub ssm_group_count: u32,
    pub ssm_time_step_rank: u32,
    pub ssm_inner_size: u32,
    pub v_head_reordered: bool,
    pub expert_count: u32,
    pub expert_used_count: u32,
    pub expert_feed_forward: u32,
    pub expert_shared_feed_forward: u32,
    pub layer_kinds: Vec<LayerKind>,
}

fn per_layer_kv(
    parsed: &ParsedGguf,
    key: &str,
    block_count: u32,
) -> Result<Vec<u32>, InteropError> {
    match parsed.metadata_value(key) {
        Some(MetadataValue::U32(value)) => Ok(vec![*value; block_count as usize]),
        Some(MetadataValue::I32(value)) => Ok(vec![*value as u32; block_count as usize]),
        Some(MetadataValue::Array(MetadataArray::U32(values))) => (values.len()
            == block_count as usize)
            .then_some(values.clone())
            .ok_or_else(|| InteropError::MetadataArrayLengthMismatch {
                key: key.to_owned(),
                expected: block_count as usize,
                found: values.len(),
            }),
        Some(MetadataValue::Array(MetadataArray::I32(values))) => (values.len()
            == block_count as usize)
            .then_some(values.iter().map(|value| *value as u32).collect())
            .ok_or_else(|| InteropError::MetadataArrayLengthMismatch {
                key: key.to_owned(),
                expected: block_count as usize,
                found: values.len(),
            }),
        _ => Err(InteropError::MissingMetadataKey {
            key: key.to_owned(),
        }),
    }
}

pub fn from_metadata(parsed: &ParsedGguf) -> Result<Architecture, InteropError> {
    let family = metadata_str(parsed, "general.architecture")?;
    let prefix = |name: &str| format!("{family}.{name}");
    let embedding = metadata_u32(parsed, &prefix("embedding_length"))?;
    let block_count = metadata_u32(parsed, &prefix("block_count"))?;
    let interval = metadata_u32(parsed, &prefix("full_attention_interval"))?;
    let kv_heads_by_layer = per_layer_kv(parsed, &prefix("attention.head_count_kv"), block_count)?;
    let layer_kinds = (0..block_count)
        .map(|layer| LayerKind::from_interval(layer, interval))
        .collect();
    let v_head_reordered = match parsed.metadata_value(&prefix("ssm.v_head_reordered")) {
        Some(MetadataValue::Bool(value)) => *value,
        None => false,
        _ => {
            return Err(InteropError::UnsupportedServingConfig(format!(
                "{family}.ssm.v_head_reordered must be bool"
            )));
        }
    };
    let rope_dimension_sections = match parsed.metadata_value(&prefix("rope.dimension_sections")) {
        Some(MetadataValue::Array(MetadataArray::I32(values))) => {
            values.iter().map(|value| *value as u32).collect()
        }
        Some(MetadataValue::Array(MetadataArray::U32(values))) => values.clone(),
        _ => Vec::new(),
    };
    let rope_mrope_interleaved = matches!(
        parsed.metadata_value(&prefix("rope.mrope_interleaved")),
        Some(MetadataValue::Bool(true))
    );
    Ok(Architecture {
        vocab: crate::bind::vocab_from_token_embedding(parsed, embedding)?,
        embedding,
        query_heads: metadata_u32(parsed, &prefix("attention.head_count"))?,
        kv_heads_by_layer,
        attn_head_dim: metadata_u32(parsed, &prefix("attention.key_length"))?,
        rope_dims: metadata_u32_optional_or(
            parsed,
            &prefix("rope.dimension_count"),
            metadata_u32(parsed, &prefix("attention.key_length"))?,
        ),
        rope_dimension_sections,
        rope_mrope_interleaved,
        block_count,
        full_attention_interval: interval,
        rope_freq_base: metadata_f32_optional(
            parsed,
            &prefix("rope.freq_base"),
            proxima_tensor::sized::ROPE_FREQ_BASE_DEFAULT,
        ),
        rms_epsilon: metadata_f32_optional(
            parsed,
            &prefix("attention.layer_norm_rms_epsilon"),
            1e-6,
        ),
        ssm_conv_kernel: metadata_u32(parsed, &prefix("ssm.conv_kernel"))?,
        ssm_state_size: metadata_u32(parsed, &prefix("ssm.state_size"))?,
        ssm_group_count: metadata_u32(parsed, &prefix("ssm.group_count"))?,
        ssm_time_step_rank: metadata_u32(parsed, &prefix("ssm.time_step_rank"))?,
        ssm_inner_size: metadata_u32(parsed, &prefix("ssm.inner_size"))?,
        v_head_reordered,
        expert_count: metadata_u32(parsed, &prefix("expert_count"))?,
        expert_used_count: metadata_u32(parsed, &prefix("expert_used_count"))?,
        expert_feed_forward: metadata_u32(parsed, &prefix("expert_feed_forward_length"))?,
        expert_shared_feed_forward: metadata_u32(
            parsed,
            &prefix("expert_shared_feed_forward_length"),
        )?,
        layer_kinds,
    })
}
