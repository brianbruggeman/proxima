//! The `uniform` header reader
//! ([`proxima_tensor::spec::ScheduleSource::Uniform`]): every checkpoint whose
//! scalar `{family}.*` keys describe every layer alike (every dense decoder,
//! the dense mixture-of-experts decoders, and the short-convolution
//! hybrid, whose zero-KV-head layers the tensor directory resolves). Composes
//! [`crate::bind::architecture_from_metadata`],
//! [`crate::bind::checkpoint_has_qk_norm`], and
//! [`proxima_tensor::spec::gqa_descriptor_from_shape`] built straight off
//! this checkpoint's own parsed hyperparameters; [`build_forward`]'s
//! `CacheStrategy::Cached`, `CacheMask::Bounded` arm lowers the result
//! (`expert_count`/`expert_used_count` off the checkpoint's own metadata is
//! what already selects a dense vs. mixture-of-experts program inside that one
//! builder).
//!
//! RoPE pairing is profile data (`rope_layout`), never inferred from QK-norm
//! tensors. This reader compares no family name:
//! [`crate::bind::ModelHparams::family`] keys
//! [`crate::profiles::family_profile`], and the profile's `rope_layout` rides
//! into the one generic [`build_forward`] call every family takes. A family with
//! no profile is an error, never a default.

use proxima_gguf::pipe::ParsedGguf;
use proxima_tensor::spec::{
    AttentionScoreScale, CacheStrategy, EmbeddingScale, LayerAttentionConfig, LayerKind, LayerSchedule,
    ModelDescriptor, gqa_descriptor_from_shape,
};

use crate::bind::{
    ModelHparams, architecture_from_metadata, checkpoint_has_qk_norm, checkpoint_qkv_biases,
    metadata_f32_optional, metadata_u32_optional,
};
use crate::error::InteropError;
use crate::profiles::family_profile;
use crate::task::{ModelTask, classify_task};

/// This header's descriptor and hyperparameters, the two values
/// [`crate::lowering`] lowers and binds from.
///
/// # Errors
///
/// Whatever [`architecture_from_metadata`] and [`descriptor_from_gguf`] can
/// fail with.
pub(crate) fn header(parsed: &ParsedGguf) -> Result<(ModelDescriptor, ModelHparams), InteropError> {
    let architecture = architecture_from_metadata(parsed)?;
    let descriptor = descriptor_from_gguf(parsed, &architecture)?;
    Ok((descriptor, architecture))
}

/// The dense checkpoint's whole pre-lowering program as one config:
/// [`gqa_descriptor_from_shape`] over the family profile
/// `general.architecture` names, the header's own scales layered on top, and
/// the logits row count the checkpoint's task needs. [`build_forward`] over
/// the result is the entire lowering; serialize the descriptor and a restored
/// copy lowers to the same program.
///
/// # Errors
///
/// The family has no profile, the header declares partial rotary, or the
/// per-layer KV head counts differ (this program has one cache shape).
pub fn descriptor_from_gguf(
    parsed: &ParsedGguf,
    architecture: &ModelHparams,
) -> Result<ModelDescriptor, InteropError> {
    let profile = family_profile(&architecture.family)?;
    require_full_rotary(parsed, architecture)?;
    // This builder has one KV cache shape for every attention layer. Preserve a
    // checkpoint's per-layer configuration in `ModelHparams`, but
    // do not silently select a representative value for this uniform
    // program.
    let kv_heads = architecture.uniform_attention_kv_heads()?;
    let descriptor = gqa_descriptor_from_shape(
        architecture.vocab,
        architecture.embedding,
        architecture.feed_forward,
        architecture.query_heads,
        kv_heads,
        architecture.head_dim,
        architecture.block_count,
        architecture.expert_count,
        architecture.expert_used_count,
        checkpoint_has_qk_norm(parsed),
        checkpoint_qkv_biases(parsed, architecture)?,
        false,
        false,
        &profile,
    );
    // Encoder-style tasks consume the full hidden sequence for pooling or a
    // task head; decoder generation samples only the final row, so the vocab
    // projection stays narrow there (`proxima-tensor/docs/discipline.md`
    // ROW 418/421 measured the cost of computing every row instead).
    let last_row_only = !matches!(classify_task(parsed).task, ModelTask::Embedding);
    let descriptor = with_conv_layers(descriptor, parsed, architecture)?;
    let descriptor = with_header_window(with_header_scales(descriptor, parsed, &architecture.family), parsed, &architecture.family);
    Ok(ModelDescriptor { last_row_only, ..descriptor })
}

/// A checkpoint whose header declares zero KV heads on some layers is a hybrid
/// of attention and short-convolution layers: the tensor directory says which
/// kind each zero-KV layer is, the dense and routed FFN widths and the leading
/// dense block count come off the header, and the program is lowered without a
/// cache because a convolution layer carries state a KV cache cannot hold
/// ([`CacheStrategy::Cacheless`]). A checkpoint with an attention layer at
/// every index is returned unchanged.
fn with_conv_layers(
    descriptor: ModelDescriptor,
    parsed: &ParsedGguf,
    architecture: &ModelHparams,
) -> Result<ModelDescriptor, InteropError> {
    let kinds = layer_kinds(parsed, architecture)?;
    if kinds.iter().all(|kind| *kind == LayerKind::Attention) {
        return Ok(descriptor);
    }
    let family = &architecture.family;
    let layers = descriptor
        .layers
        .iter()
        .zip(kinds)
        .map(|(layer, kind)| LayerSchedule { kind, ..layer.clone() })
        .collect();
    Ok(ModelDescriptor {
        layers,
        feed_forward: metadata_u32_optional(parsed, &format!("{family}.feed_forward_length")),
        leading_dense_block_count: metadata_u32_optional(parsed, &format!("{family}.leading_dense_block_count")),
        l_cache: metadata_u32_optional(parsed, &format!("{family}.shortconv.l_cache")),
        cache_strategy: CacheStrategy::Cacheless,
        ..descriptor
    })
}

fn layer_kinds(parsed: &ParsedGguf, architecture: &ModelHparams) -> Result<Vec<LayerKind>, InteropError> {
    let names: Vec<&str> = parsed.tensors.iter().map(|tensor| tensor.name.as_str()).collect();
    architecture
        .kv_heads_by_layer
        .iter()
        .enumerate()
        .map(|(layer, &kv_heads)| {
            if kv_heads != 0 {
                return Ok(LayerKind::Attention);
            }
            match LayerKind::from_tensor_names(names.iter().copied(), layer as u32)? {
                LayerKind::ShortConv => Ok(LayerKind::ShortConv),
                other => Err(InteropError::UnsupportedServingConfig(format!(
                    "layer {layer} declares zero kv heads but its tensors say {other:?}"
                ))),
            }
        })
        .collect()
}

/// `<family>.attention.sliding_window`, when the header carries it, is the
/// window of every layer; absent or zero leaves every layer on the full
/// causal mask, the same reading llama.cpp's `n_swa` gets.
fn with_header_window(descriptor: ModelDescriptor, parsed: &ParsedGguf, family: &str) -> ModelDescriptor {
    let window = metadata_u32_optional(parsed, &format!("{family}.attention.sliding_window"));
    if window == 0 {
        return descriptor;
    }
    let layers = descriptor
        .layers
        .iter()
        .map(|layer| LayerSchedule {
            attention: LayerAttentionConfig {
                mask_window: Some(window),
                ..layer.attention.clone()
            },
            ..layer.clone()
        })
        .collect();
    ModelDescriptor { layers, ..descriptor }
}

fn header_scale(parsed: &ParsedGguf, family: &str, key: &str) -> Option<f32> {
    let value = metadata_f32_optional(parsed, &format!("{family}.{key}"), 0.0);
    (value != 0.0).then_some(value)
}

fn with_header_scales(descriptor: ModelDescriptor, parsed: &ParsedGguf, family: &str) -> ModelDescriptor {
    let layers = match header_scale(parsed, family, "attention.scale") {
        Some(scale) => descriptor
            .layers
            .iter()
            .map(|layer| LayerSchedule {
                attention: LayerAttentionConfig {
                    score_scale: AttentionScoreScale::Factor(scale),
                    ..layer.attention.clone()
                },
                ..layer.clone()
            })
            .collect(),
        None => descriptor.layers.clone(),
    };
    ModelDescriptor {
        embedding_scale: header_scale(parsed, family, "embedding_scale")
            .map(EmbeddingScale::Factor)
            .or(descriptor.embedding_scale),
        logit_scale: header_scale(parsed, family, "logit_scale"),
        residual_scale: header_scale(parsed, family, "residual_scale"),
        layers,
        ..descriptor
    }
}

/// The single-range builder rotates `head_dim` channels. A header whose
/// `<arch>.rope.dimension_count` says otherwise is partial rotary, which this
/// program cannot lower; refuse it rather than rotate the wrong width.
fn require_full_rotary(parsed: &ParsedGguf, architecture: &ModelHparams) -> Result<(), InteropError> {
    let key = format!("{}.rope.dimension_count", architecture.family);
    match metadata_u32_optional(parsed, &key) {
        0 => Ok(()),
        rope_dimension_count if rope_dimension_count == architecture.head_dim => Ok(()),
        rope_dimension_count => Err(InteropError::PartialRotaryUnsupported {
            family: architecture.family.clone(),
            rope_dimension_count,
            head_dim: architecture.head_dim,
        }),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
