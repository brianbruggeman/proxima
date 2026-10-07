//! The `sliding_pattern` header reader ([`proxima_tensor::spec::ScheduleSource::SlidingPattern`]):
//! the parsed header goes to [`proxima_tensor::spec::gemma4_descriptor_from_gguf`],
//! which builds the whole [`proxima_tensor::spec::ModelDescriptor`] (per-layer
//! [`proxima_tensor::spec::LayerAttentionConfig`]/
//! [`proxima_tensor::spec::LayerFfnConfig`] schedule included). The lowering
//! and the weight bind are the generic ones in [`crate::lowering`]: there is no
//! bespoke sliding-pattern forward-graph builder, no sliding-pattern schedule and no sliding-pattern
//! tensor-name table in this crate.
//! Teaching pointer: read `proxima_tensor::spec::attention_forward`'s own doc
//! on `scheduled_forward_program_with_experts` before touching this file -- every
//! knob the descriptor sets is documented there, not here.


use proxima_gguf::pipe::ParsedGguf;
use proxima_tensor::spec::{ModelDescriptor, gemma4_descriptor_from_gguf};

use crate::bind::{ModelHparams, SlidingRope, find_tensor, metadata_str};
use crate::error::InteropError;
use crate::lowering::KvLayout;
use crate::profiles::family_profile;

use super::hparams::from_metadata;

/// The checkpoint's descriptor: [`gemma4_descriptor_from_gguf`] over the family
/// profile `general.architecture` names, so the values GGUF does not carry come
/// from `crate::profiles` and never from this module.
pub fn descriptor_from_gguf(
    parsed: &ParsedGguf,
    sliding_kv_ring: bool,
) -> Result<ModelDescriptor, InteropError> {
    let profile = family_profile(metadata_str(parsed, "general.architecture")?)?;
    Ok(gemma4_descriptor_from_gguf(parsed, sliding_kv_ring, &profile)?)
}

/// `PROXIMA_HEAD_REPEATS=1|2|3` (unset or unparsable reads as `1`): the
/// head-cost measurement knob, layered into [`proxima_tensor::spec::ModelDescriptor::head_repeats`]
/// here so the op-graph builders stay pure functions of the descriptor.
#[cfg(feature = "instrument")]
fn head_repeats_from_env() -> u32 {
    std::env::var("PROXIMA_HEAD_REPEATS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1)
}

/// This header's descriptor and hyperparameters, the two values
/// [`crate::lowering`] lowers and binds from. Every layer is
/// [`proxima_tensor::spec::LayerKind::Attention`] (this family has no
/// `ShortConv` layers), routed through the padded-mask cached engine: the
/// two-range one, not single-range, because the first step processes the whole
/// prompt as one `cached_len=0` call (`scheduled_cached.rs`'s own
/// module doc).
pub(crate) fn header(
    parsed: &ParsedGguf,
    layout: KvLayout,
) -> Result<(ModelDescriptor, ModelHparams), InteropError> {
    let hparams = from_metadata(parsed)?;
    let descriptor = descriptor_from_gguf(parsed, layout == KvLayout::SlidingRing)?;
    #[cfg(feature = "instrument")]
    let descriptor = ModelDescriptor {
        head_repeats: head_repeats_from_env(),
        ..descriptor
    };
    let architecture = ModelHparams {
        vocab: hparams.vocab,
        embedding: hparams.embedding,
        feed_forward: hparams.feed_forward,
        query_heads: hparams.head_count,
        kv_heads: *hparams.kv_heads_by_layer.last().unwrap_or(&0),
        kv_heads_by_layer: hparams.kv_heads_by_layer.clone(),
        head_dim: hparams.key_length,
        block_count: hparams.block_count,
        expert_count: hparams.expert_count,
        expert_used_count: hparams.expert_used_count,
        rope_freq_base: hparams.rope_freq_base,
        rms_epsilon: hparams.rms_epsilon,
        tied_embeddings: find_tensor(parsed, "output.weight").is_err(),
        family: metadata_str(parsed, "general.architecture")?.into(),
        sliding_rope: Some(SlidingRope {
            freq_base: hparams.rope_freq_base_swa,
            dimension_count: hparams.rope_dimension_count_swa,
        }),
    };
    Ok((descriptor, architecture))
}

/// Regression coverage for the bug this crate shipped once: the binder and the
/// forward program each independently gated `attn_k.weight`/
/// `attn_k_norm.weight`/`attn_v.weight` per layer, and nothing forced the
/// two gates to agree -- `gemma4_descriptor_from_gguf` used to gate
/// `ValueSourceKind::ProjectedV` (and therefore the forward program's own
/// `attn_v.weight` [`proxima_tensor::op::Op::Input`] leaf) on `is_sliding`
/// alone, the MoE convention, while E2B/E4B's own-KV FULL layers (`blk.4`,
/// `blk.9`, `blk.14` on the real `e2b-it-qat` checkpoint) carry a
/// real `attn_v.weight` regardless of sliding vs full. This test builds the
/// ACTUAL forward program `crate::lowering::bind_checkpoint` builds (not a hand-simulated
/// stand-in) for a synthetic E2B-shaped [`Gemma4Hparams`] whose
/// `sliding_window_pattern`/`shared_kv_layers`/`block_count` are the real
/// checkpoint's own measured values (a real header dump against
/// `~/.ollama/models/blobs/sha256-3646b4c...` on 2026-09-20), then asserts
/// the forward program's own declared `Input` leaf names for these three
/// per-layer weights equal exactly the layers that own those tensors in the
/// real checkpoint (derived below from the header pattern, never from the
/// lowering under test).
#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod declared_leaves_match_bound_leaves_tests;
