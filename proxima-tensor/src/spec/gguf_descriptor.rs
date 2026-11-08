use alloc::format;

use proxima_gguf::pipe::ParsedGguf;
use proxima_gguf::value::{MetadataArray, MetadataValue};

use super::*;

/// Builds a sliding-pattern-family [`ModelDescriptor`] from a parsed GGUF header: the
/// per-layer sliding/full split, matformer dense-FFN widths, PLE, shared-KV
/// and the dense-vs-MoE FFN shape all come from the checkpoint's own
/// `{general.architecture}.*` keys and the `token_embd.weight` directory
/// entry (vocab). The per-family values GGUF does not carry (score scale,
/// value-norm, activation, embedding scale, RoPE pairing, norm placement)
/// come from the [`FamilyProfile`] the caller loaded for
/// `general.architecture`; only the table names stay structural.
///
/// This is the one place a sliding-pattern descriptor is built; the bind in
/// `proxima-model-interop` and the tests in `proxima-tensor` both feed it a
/// real header and hand the result to [`build_forward`]. `sliding_kv_ring`
/// is the caller's KV layout choice, not checkpoint data
/// ([`ModelDescriptor::sliding_kv_ring`]).
///
/// # Errors
///
/// [`TensorError::MissingGgufEntry`] when a required key or `token_embd.weight`
/// is absent or has an unsupported wire type;
/// [`TensorError::GgufPerLayerLengthMismatch`] when a per-layer array is not
/// `block_count` long; [`TensorError::GgufVocabShapeMismatch`] when the
/// embedding table does not divide by `embedding_length`;
/// [`TensorError::GgufUnequalValueWidth`] when a layer kind's
/// `attention.value_length` differs from its `attention.key_length`, which this
/// engine's single head width cannot express; and
/// [`TensorError::GgufUnsupportedExpertGating`] when `expert_gating_func` names
/// a routing function this engine has no implementation for.
pub fn sliding_pattern_descriptor_from_gguf(
    parsed: &ParsedGguf,
    sliding_kv_ring: bool,
    profile: &FamilyProfile,
) -> Result<ModelDescriptor, TensorError> {
    let family = metadata_str(parsed, "general.architecture")?;
    let key = |name: &str| format!("{family}.{name}");

    let embedding = metadata_u32(parsed, &key("embedding_length"))?;
    let block_count = metadata_u32(parsed, &key("block_count"))?;
    let kv_heads_by_layer =
        metadata_u32_per_layer(parsed, &key("attention.head_count_kv"), block_count)?;
    let sliding_window_pattern = metadata_bool_per_layer(
        parsed,
        &key("attention.sliding_window_pattern"),
        block_count,
    )?;
    let feed_forward_by_layer =
        metadata_u32_per_layer(parsed, &key("feed_forward_length"), block_count)?;
    let expert_count = metadata_u32_optional(parsed, &key("expert_count"));
    // llama.cpp's `leading_dense_block_count` selects the dense branch per
    // layer; a dense-only checkpoint keeps the old "every layer dense" default
    let leading_dense_block_count = metadata_u32_option(parsed, &key("leading_dense_block_count"))
        .unwrap_or(if expert_count > 0 { 0 } else { block_count });
    let expert_shared_feed_forward =
        metadata_u32_optional(parsed, &key("expert_shared_feed_forward_length"));
    let expert_gating =
        expert_gating_func(metadata_u32_optional(parsed, &key("expert_gating_func")))?;
    // sinks are a per-layer tensor, not a header scalar: the leaf's presence is
    // the schedule
    let sink_layers: Vec<bool> = (0..block_count)
        .map(|layer| {
            let name = format!("blk.{layer}.attn_sinks.weight");
            parsed.tensors.iter().any(|tensor| tensor.name == name)
        })
        .collect();
    let ple_dim = metadata_u32_optional(parsed, &key("embedding_length_per_layer_input"));
    let shared_kv_layers = metadata_u32_optional(parsed, &key("attention.shared_kv_layers"));
    let query_heads = metadata_u32(parsed, &key("attention.head_count"))?;
    // llama.cpp seeds both head widths with `n_embd / n_head` before the keys
    // override them (`llama-model.cpp`, `n_embd_head_k_full` / `_v_full`)
    let default_head_width = embedding.checked_div(query_heads).unwrap_or(0);
    let key_length = match metadata_u32_option(parsed, &key("attention.key_length")) {
        Some(width) => width,
        // llama.cpp's zero-head branch leaves every width at 0, but this engine
        // has no zero-width attention to build, so a header with no head count
        // and no key width is still missing the key
        None if default_head_width == 0 => return Err(missing(&key("attention.key_length"))),
        None => default_head_width,
    };
    // a checkpoint whose sliding layers attend at the same head width as its
    // full layers omits the swa override; the key only exists for checkpoints
    // that differ (the flint capture carries neither, gemma4 carries both)
    let key_length_swa =
        metadata_u32_option(parsed, &key("attention.key_length_swa")).unwrap_or(key_length);
    // this engine's attention graph carries a single head width for K and V
    let value_length =
        metadata_u32_option(parsed, &key("attention.value_length")).unwrap_or(default_head_width);
    let value_length_swa =
        metadata_u32_option(parsed, &key("attention.value_length_swa")).unwrap_or(value_length);
    for (kind, key_width, value_width) in [
        ("full", key_length, value_length),
        ("sliding", key_length_swa, value_length_swa),
    ] {
        if key_width != value_width {
            return Err(TensorError::GgufUnequalValueWidth {
                kind,
                key_length: key_width,
                value_length: value_width,
            });
        }
    }
    let sliding_window = metadata_u32(parsed, &key("attention.sliding_window"))?;
    let softcap = metadata_f32_optional(parsed, &key("final_logit_softcapping"), 0.0);
    let rotary_dim = rotary_or_head_dim(
        metadata_u32_option(parsed, &key("rope.dimension_count")),
        key_length,
    );
    // llama.cpp carries one rotary width per layer kind and derives the swa one
    // from the full one (`llama-model.cpp`: `n_rot_swa = n_rot_full` before
    // `rope.dimension_count_swa` overrides it), never from the swa head width
    let rotary_dim_swa = rotary_or_head_dim(
        metadata_u32_option(parsed, &key("rope.dimension_count_swa")),
        rotary_dim,
    );

    let profile_ffn = profile.layer_ffn(expert_count);
    let ffn = LayerFfnConfig {
        ple: ple_dim > 0,
        // the header's own gating value wins over the profile's when it names one
        routed_gating: expert_gating.unwrap_or(profile_ffn.routed_gating),
        ..profile_ffn
    };
    let sliding = LayerAttentionConfig {
        head_dim: key_length_swa,
        kv_heads: 0,
        mask_window: Some(sliding_window),
        value_source_kind: ValueSourceKind::ProjectedV,
        key_source_kind: KeySourceKind::ProjectedK,
        rope_table: RopeTableSel {
            cos_name: "rope_cos_swa".into(),
            sin_name: "rope_sin_swa".into(),
        },
        rope_pairing: profile.rope_pairing(rotary_dim_swa),
        score_scale: profile.score_scale(key_length_swa),
        value_norm: profile.value_norm,
        qk_norm: profile.qk_norm,
        attention_sinks: false,
    };
    let full = LayerAttentionConfig {
        head_dim: key_length,
        mask_window: None,
        // moe checkpoints (shared_kv_layers == 0) have no attn_v on full layers
        value_source_kind: if shared_kv_layers > 0 {
            ValueSourceKind::ProjectedV
        } else {
            ValueSourceKind::SharedWithKey
        },
        rope_table: RopeTableSel {
            cos_name: "rope_cos".into(),
            sin_name: "rope_sin".into(),
        },
        rope_pairing: profile.rope_pairing(rotary_dim),
        score_scale: profile.score_scale(key_length),
        ..sliding
    };

    let first_shared_idx = block_count.saturating_sub(shared_kv_layers);
    let layers: Vec<LayerSchedule> = sliding_window_pattern
        .iter()
        .zip(&kv_heads_by_layer)
        .zip(&feed_forward_by_layer)
        .enumerate()
        .map(|(layer, ((&is_sliding, &kv_heads), &feed_forward))| {
            let template = if is_sliding {
                sliding.clone()
            } else {
                full.clone()
            };
            let attention = LayerAttentionConfig {
                kv_heads,
                attention_sinks: sink_layers[layer],
                ..template
            };
            // a shared layer's V is already post-norm, re-norming would double-apply
            let attention = if layer as u32 >= first_shared_idx {
                let source =
                    shared_kv_source_layer(&sliding_window_pattern, first_shared_idx, layer);
                LayerAttentionConfig {
                    key_source_kind: KeySourceKind::SharedFromLayer(source),
                    value_source_kind: ValueSourceKind::SharedFromLayer(source),
                    value_norm: false,
                    ..attention
                }
            } else {
                attention
            };
            LayerSchedule {
                kind: LayerKind::Attention,
                attention,
                ffn: LayerFfnConfig {
                    dense_feed_forward: Some(feed_forward),
                    ..ffn
                },
            }
        })
        .collect();

    let cache_strategy = if layers
        .iter()
        .all(|entry| entry.kind == LayerKind::Attention)
    {
        CacheStrategy::Cached
    } else {
        CacheStrategy::Cacheless
    };

    Ok(ModelDescriptor {
        vocab: vocab_from_token_embedding(parsed, embedding)?,
        embedding,
        feed_forward: feed_forward_by_layer.first().copied().unwrap_or(0),
        expert_feed_forward: metadata_u32_optional(parsed, &key("expert_feed_forward_length")),
        query_heads,
        block_count,
        expert_count,
        expert_used_count: metadata_u32_optional(parsed, &key("expert_used_count")),
        // an exclusive (dense-only) layer takes the dense branch when layer < this, so
        // block_count keeps the unbound routed branch from ever being built
        leading_dense_block_count,
        l_cache: 0,
        embedding_scale: profile.embedding_scale,
        logit_softcap: (softcap > 0.0).then_some(softcap),
        logit_scale: None,
        residual_scale: None,
        layers,
        cache_strategy,
        cache_mask: CacheMask::Padded,
        ple_dim: (ple_dim > 0).then_some(ple_dim),
        sliding_kv_ring,
        qk_norm: false,
        qkv_biases: false,
        paired_gate_up_reduce: false,
        fused_qkv_reduce: false,
        head_repeats: 1,
        last_row_only: true,
        speculative_verify: profile.speculative_verify,
        ssm_conv_kernel: 0,
        ssm_state_size: 0,
        ssm_group_count: 0,
        ssm_time_step_rank: 0,
        ssm_inner_size: 0,
        ssm_epsilon: 0.0,
        v_head_reordered: false,
        ssm_out_head_major: false,
        hadamard_block: 0,
        hadamard_gdn_v_grouped: false,
        hadamard_latent_embedding: false,
        expert_shared_feed_forward,
        prefill_width: None,
        gated_attention: false,
        recurrence: None,
        loopcd: LoopCdConfig::default(),
        network: None,
    })
}

/// The own-KV layer a trailing shared layer reads: the last layer before
/// `first_shared_idx` with the same sliding-vs-full kind (ollama
/// the reference Go source, lines 590-611). Falls back to `first_shared_idx - 1` when no layer
/// of that kind precedes it, which no real checkpoint reaches.
fn shared_kv_source_layer(
    sliding_window_pattern: &[bool],
    first_shared_idx: u32,
    layer: usize,
) -> u32 {
    let is_sliding = sliding_window_pattern[layer];
    (0..first_shared_idx as usize)
        .rev()
        .find(|&candidate| sliding_window_pattern[candidate] == is_sliding)
        .map_or_else(|| first_shared_idx.saturating_sub(1), |index| index as u32)
}

fn missing(name: &str) -> TensorError {
    TensorError::MissingGgufEntry { name: name.into() }
}

fn metadata_str<'parsed>(
    parsed: &'parsed ParsedGguf,
    key: &str,
) -> Result<&'parsed str, TensorError> {
    parsed
        .metadata_value(key)
        .and_then(MetadataValue::as_str)
        .ok_or_else(|| missing(key))
}

fn metadata_u32(parsed: &ParsedGguf, key: &str) -> Result<u32, TensorError> {
    parsed
        .metadata_value(key)
        .and_then(MetadataValue::as_u32)
        .ok_or_else(|| missing(key))
}

fn metadata_u32_optional(parsed: &ParsedGguf, key: &str) -> u32 {
    metadata_u32_option(parsed, key).unwrap_or(0)
}

/// A present key wins even when it is `0` -- llama.cpp's `get_key` reports
/// absent and zero differently (`llama-model-loader.cpp`), and an explicit
/// rotary width of zero is a checkpoint asking for no rotation.
fn metadata_u32_option(parsed: &ParsedGguf, key: &str) -> Option<u32> {
    parsed.metadata_value(key).and_then(MetadataValue::as_u32)
}

/// The header's own gating value, llama.cpp's `llama_expert_gating_func_type`
/// (`1` softmax, `2` sigmoid; `0` none, absent). `3` (`SOFTMAX_WEIGHT`) and `4`
/// (`SQRT_SOFTPLUS`) are real values this engine has no function for, so they
/// are refused by name rather than served with the profile's default.
fn expert_gating_func(value: u32) -> Result<Option<ExpertGatingFunc>, TensorError> {
    match value {
        0 => Ok(None),
        1 => Ok(Some(ExpertGatingFunc::Softmax)),
        2 => Ok(Some(ExpertGatingFunc::Sigmoid)),
        other => Err(TensorError::GgufUnsupportedExpertGating { value: other }),
    }
}

const fn rotary_or_head_dim(rope_dimension_count: Option<u32>, head_dim: u32) -> u32 {
    match rope_dimension_count {
        Some(count) => count,
        None => head_dim,
    }
}

fn metadata_f32_optional(parsed: &ParsedGguf, key: &str, default: f32) -> f32 {
    match parsed.metadata_value(key) {
        Some(MetadataValue::F32(value)) => *value,
        Some(MetadataValue::F64(value)) => *value as f32,
        _ => default,
    }
}

fn per_layer_len(key: &str, block_count: u32, found: usize) -> Result<(), TensorError> {
    if found == block_count as usize {
        return Ok(());
    }
    Err(TensorError::GgufPerLayerLengthMismatch {
        key: key.into(),
        expected: block_count as usize,
        found,
    })
}

fn metadata_u32_per_layer(
    parsed: &ParsedGguf,
    key: &str,
    block_count: u32,
) -> Result<Vec<u32>, TensorError> {
    match parsed.metadata_value(key) {
        Some(MetadataValue::U32(value)) => Ok(alloc::vec![*value; block_count as usize]),
        Some(MetadataValue::I32(value)) => u32::try_from(*value)
            .map(|value| alloc::vec![value; block_count as usize])
            .map_err(|_| missing(key)),
        Some(MetadataValue::Array(MetadataArray::U32(values))) => {
            per_layer_len(key, block_count, values.len())?;
            Ok(values.clone())
        }
        Some(MetadataValue::Array(MetadataArray::I32(values))) => {
            per_layer_len(key, block_count, values.len())?;
            values
                .iter()
                .map(|&value| u32::try_from(value).map_err(|_| missing(key)))
                .collect()
        }
        _ => Err(missing(key)),
    }
}

fn metadata_bool_per_layer(
    parsed: &ParsedGguf,
    key: &str,
    block_count: u32,
) -> Result<Vec<bool>, TensorError> {
    match parsed.metadata_value(key) {
        Some(MetadataValue::Array(MetadataArray::Bool(values))) => {
            per_layer_len(key, block_count, values.len())?;
            Ok(values.clone())
        }
        _ => Err(missing(key)),
    }
}

fn vocab_from_token_embedding(parsed: &ParsedGguf, embedding: u32) -> Result<u32, TensorError> {
    let name = "token_embd.weight";
    let tensor = parsed
        .tensors
        .iter()
        .find(|tensor| tensor.name == name)
        .ok_or_else(|| missing(name))?;
    let elements = tensor.element_count();
    let divisor = u64::from(embedding);
    if divisor == 0 || !elements.is_multiple_of(divisor) {
        return Err(TensorError::GgufVocabShapeMismatch {
            elements,
            embedding,
        });
    }
    Ok((elements / divisor) as u32)
}
