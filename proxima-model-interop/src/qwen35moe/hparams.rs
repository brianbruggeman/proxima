use alloc::format;
use alloc::vec::Vec;

use proxima_gguf::pipe::ParsedGguf;
use proxima_gguf::value::{MetadataArray, MetadataValue};

use crate::bind::{
    find_tensor, metadata_f32_optional, metadata_str, metadata_u32, metadata_u32_optional_or,
};
use crate::error::InteropError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    Gdn,
    Attention,
}

impl LayerKind {
    #[must_use]
    pub fn from_kv_heads(kv_heads: u32) -> Self {
        if kv_heads == 0 {
            Self::Gdn
        } else {
            Self::Attention
        }
    }

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
    // The tensor directory is the authoritative layer-kind declaration.  The
    // interval is only a useful hint: current Qwen3.5 headers contain an
    // irregular final attention layer (`blk.40`) that the interval alone
    // misclassifies.  `attn_q.weight` is present only on attention layers;
    // GDN layers instead carry `attn_qkv.weight`, so this remains correct for
    // future schedules without baking another architecture exception into
    // the parser.
    let layer_kinds = (0..block_count)
        .map(|layer| {
            let attention_name = format!("blk.{layer}.attn_q.weight");
            if find_tensor(parsed, &attention_name).is_ok() {
                LayerKind::Attention
            } else {
                LayerKind::Gdn
            }
        })
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

/// One `(kv_heads, attn_head_dim, None)` entry per layer whose
/// `head_count_kv` is nonzero: the GDN layers (`head_count_kv == 0`) keep a
/// recurrent state, not a KV cache, and are charged nothing here.
///
/// # Errors
///
/// [`InteropError::MissingMetadataKey`] or
/// [`InteropError::MetadataArrayLengthMismatch`] from the underlying reads.
pub fn kv_layers_from_metadata(
    parsed: &ParsedGguf,
) -> Result<Vec<(u32, u32, Option<u32>)>, InteropError> {
    let family = metadata_str(parsed, "general.architecture")?;
    let prefix = |name: &str| format!("{family}.{name}");

    let block_count = metadata_u32(parsed, &prefix("block_count"))?;
    let kv_heads_by_layer = per_layer_kv(parsed, &prefix("attention.head_count_kv"), block_count)?;
    let head_dim = metadata_u32(parsed, &prefix("attention.key_length"))?;

    Ok(kv_heads_by_layer
        .into_iter()
        .filter(|&kv_heads| kv_heads != 0)
        .map(|kv_heads| (kv_heads, head_dim, None))
        .collect())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use alloc::string::ToString;
    use alloc::vec;

    use proxima_gguf::value::{MetadataArray, MetadataValue as Value};

    use crate::architecture::Architecture as _;
    use crate::memory_fit::{MemoryBudget, WeightClassBytes};
    use crate::qwen35moe::QWEN35MOE;
    use crate::test_support::parsed_header;

    /// `qwen3.6:35b-a3b`'s KV-relevant header (`ollama /api/show`,
    /// 2026-09-29): 40 blocks, `head_count_kv` of 2 on every fourth layer
    /// and 0 on the 30 GDN layers, key length 256.
    #[test]
    fn memory_budget_qwen35moe() {
        let parsed = parsed_header(vec![
            (
                "general.architecture",
                Value::String("qwen35moe".to_string()),
            ),
            ("qwen35moe.block_count", Value::U32(40)),
            (
                "qwen35moe.attention.head_count_kv",
                Value::Array(MetadataArray::U32(
                    (0..40u32)
                        .map(|layer| if (layer + 1) % 4 == 0 { 2 } else { 0 })
                        .collect(),
                )),
            ),
            ("qwen35moe.attention.key_length", Value::U32(256)),
        ]);

        let layers = QWEN35MOE
            .kv_layers(&parsed)
            .expect("the header carries every key kv_layers reads");
        let budget = MemoryBudget::derive(WeightClassBytes::default(), &layers, 262_144, 0, 0);
        let every_layer = vec![(2u32, 256u32, None); 40];
        let today = MemoryBudget::derive(WeightClassBytes::default(), &every_layer, 262_144, 0, 0);

        assert_eq!(layers.len(), 10, "only layers with head_count_kv != 0");
        assert_eq!(budget.kv_cache_bytes, 10_737_418_240);
        assert_eq!(today.kv_cache_bytes, 42_949_672_960);
    }
}
