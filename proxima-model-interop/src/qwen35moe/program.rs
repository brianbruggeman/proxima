//! `qwen35moe`'s forward program, lowered by
//! [`proxima_tensor::spec::build_forward`] from one [`ModelDescriptor`]: the
//! hybrid stack of gated-DeltaNet and gated attention layers over a routed FFN
//! plus a sigmoid-gated shared expert is the recurrent-hybrid engine's routed
//! arm, so this file holds no graph construction -- only the map from the
//! checkpoint's header ([`Qwen35MoeHparams`]) and the family profile to that
//! descriptor.
//!
//! The real checkpoint also carries `{architecture}.rope.dimension_sections`/
//! `rope.mrope_section` (`[11, 11, 10]`, summing to `rope_dims / 2` pairs)
//! and `rope.mrope_interleaved` -- Qwen2-VL/Qwen3-VL-style multi-axis
//! (text/height/width) RoPE. The lowered program still applies one uniform
//! split-half rotation across the `rope_dims` width; the multi-axis position
//! table is a step input the decode loop builds, not part of the lowering.
//!
//! # Teaching pointer
//!
//! A new variant of this family is a config file: serialize
//! [`descriptor_from_architecture`]'s result, edit the layer schedule or the
//! expert counts, and hand it to [`proxima_tensor::spec::build_forward`] (or
//! `LoadedModel::load_with_descriptor`) with no Rust change.

use proxima_tensor::spec::{
    ForwardProgram, ForwardRoots, LayerAttentionConfig, LayerKind, LayerSchedule, ModelDescriptor,
    LayerCacheRoots, MoeSites, build_forward, CacheMask, CacheStrategy, KeySourceKind, ValueSourceKind,
    RopeTableSel,
};
use proxima_tensor::{Op, TensorError};

pub use proxima_tensor::spec::MoeLayerDiagnostics;

use super::hparams::{Qwen35MoeHparams, LayerKind as HeaderLayerKind};
use crate::error::InteropError;
use crate::profiles::family_profile;

const FAMILY: &str = "qwen35moe";

/// [`qwen35moe_forward_program`]'s own return shape: the built `Vec<Op>`,
/// its logits/hidden roots, each layer's production cache-root tag, every
/// routed site, and each layer's own [`MoeLayerDiagnostics`] side table.
pub type MoeForwardProgram = (
    Vec<Op>,
    ForwardRoots,
    Vec<LayerCacheRoots>,
    MoeSites,
    Vec<MoeLayerDiagnostics>,
);

/// The checkpoint's whole pre-lowering program as one config: the header's
/// layer kinds, per-layer KV heads and recurrence and expert shapes over the
/// family profile's FFN, score scale and rope pairing. `width` pins the
/// position axis ([`ModelDescriptor::prefill_width`]).
///
/// # Errors
///
/// The family has no embedded profile.
pub fn descriptor_from_architecture(
    architecture: &Qwen35MoeHparams,
    width: Option<u32>,
) -> Result<ModelDescriptor, InteropError> {
    let profile = family_profile(FAMILY)?;
    let attention = LayerAttentionConfig {
        head_dim: architecture.attn_head_dim,
        kv_heads: 0,
        mask_window: None,
        value_source_kind: ValueSourceKind::ProjectedV,
        key_source_kind: KeySourceKind::ProjectedK,
        rope_table: RopeTableSel {
            cos_name: "rope_cos".into(),
            sin_name: "rope_sin".into(),
        },
        rope_pairing: profile.rope_pairing(architecture.rope_dims),
        score_scale: profile.score_scale(architecture.attn_head_dim),
        value_norm: profile.value_norm,
    };
    let ffn = profile.layer_ffn(architecture.expert_count);
    let layers = architecture
        .layer_kinds
        .iter()
        .zip(&architecture.kv_heads_by_layer)
        .map(|(kind, &kv_heads)| LayerSchedule {
            kind: match kind {
                HeaderLayerKind::Attention => LayerKind::Attention,
                HeaderLayerKind::Gdn => LayerKind::Gdn,
            },
            attention: LayerAttentionConfig {
                kv_heads,
                ..attention.clone()
            },
            ffn,
        })
        .collect();
    Ok(ModelDescriptor {
        vocab: architecture.vocab,
        embedding: architecture.embedding,
        feed_forward: 0,
        expert_feed_forward: architecture.expert_feed_forward,
        query_heads: architecture.query_heads,
        block_count: architecture.block_count,
        expert_count: architecture.expert_count,
        expert_used_count: architecture.expert_used_count,
        leading_dense_block_count: 0,
        l_cache: 0,
        embedding_scale: profile.embedding_scale,
        logit_softcap: None,
        logit_scale: None,
        residual_scale: None,
        layers,
        cache_strategy: CacheStrategy::Cached,
        cache_mask: CacheMask::Bounded,
        ple_dim: None,
        sliding_kv_ring: false,
        qk_norm: false,
        qkv_biases: false,
        paired_gate_up_reduce: false,
        fused_qkv_reduce: false,
        head_repeats: 1,
        last_row_only: true,
        speculative_verify: profile.speculative_verify,
        ssm_conv_kernel: architecture.ssm_conv_kernel,
        ssm_state_size: architecture.ssm_state_size,
        ssm_group_count: architecture.ssm_group_count,
        ssm_time_step_rank: architecture.ssm_time_step_rank,
        ssm_inner_size: architecture.ssm_inner_size,
        ssm_epsilon: architecture.rms_epsilon,
        v_head_reordered: architecture.v_head_reordered,
        expert_shared_feed_forward: architecture.expert_shared_feed_forward,
        prefill_width: width,
        gated_attention: true,
    })
}

/// Builds `qwen35moe`'s whole-model forward program.
///
/// # Errors
///
/// [`InteropError::Tensor`] if the descriptor does not lower.
pub fn qwen35moe_forward_program(architecture: &Qwen35MoeHparams) -> Result<MoeForwardProgram, InteropError> {
    qwen35moe_forward_program_at_width(architecture, None)
}

/// [`qwen35moe_forward_program`], with the prompt-position axis pinned to a
/// literal `Extent::Static(width)` instead of the ordinary
/// `Extent::Symbolic(0)` (`crate::generate::symbols::NEW_COUNT`) every
/// per-step decode call resolves dynamically. A caller that already knows
/// the whole prompt's width up front builds THIS variant once for that
/// width instead: the gated-DeltaNet mixer's M > 1 branch only ever fires for
/// a literal `Extent::Static` leading axis (its recurrence is unrolled in
/// Rust, so its length has to be known when the program is lowered), so this
/// is the ONE seam that reaches it.
pub fn qwen35moe_forward_program_at_width(
    architecture: &Qwen35MoeHparams,
    width: Option<u32>,
) -> Result<MoeForwardProgram, InteropError> {
    let ForwardProgram {
        program,
        logits,
        hidden,
        layer_roots,
        moe_sites,
        layer_diagnostics,
        ..
    } = build_forward(&descriptor_from_architecture(architecture, width)?)?;
    let hidden = hidden.ok_or(TensorError::UnsupportedInBuilder {
        builder: "qwen35moe_forward_program_at_width",
        feature: "a program with no hidden root",
    })?;
    Ok((program, ForwardRoots { logits, hidden }, layer_roots, moe_sites, layer_diagnostics))
}
