//! Header configuration for the `gemma4` mixture-of-experts checkpoint
//! family -- `qwen35moe::hparams`'s own shape (family-prefixed metadata
//! reads, a per-layer array read into `Vec`) applied to Gemma 4's own
//! alternating sliding-window/full-attention schedule instead of Qwen 3.6's
//! GDN/attention schedule. Layer kind (sliding vs full) lives entirely in
//! `sliding_window_pattern` -- there is no bespoke `LayerKind` here, since
//! [`crate::gemma4::bind::Gemma4Arch::bind`] reads that array directly to
//! build [`proxima_tensor::spec::LayerAttentionConfig`] per layer for the
//! generic [`proxima_tensor::spec::lfm2_forward_program_with_experts`]
//! engine, and every gemma4 layer is
//! [`proxima_tensor::spec::LayerKind::Attention`] (this family has no
//! `ShortConv` layers).

use alloc::format;
use alloc::vec::Vec;

use proxima_gguf::pipe::ParsedGguf;

use crate::bind::{
    metadata_bool_per_layer, metadata_f32_optional, metadata_str, metadata_u32,
    metadata_u32_optional, metadata_u32_optional_or, metadata_u32_per_layer,
    vocab_from_token_embedding,
};
use crate::error::InteropError;

#[derive(Debug, Clone)]
pub struct Architecture {
    pub vocab: u32,
    pub embedding: u32,
    pub block_count: u32,
    /// `{family}.feed_forward_length`'s first entry -- kept for callers that
    /// still want a single representative dense-FFN width (e.g. parity
    /// logging). The authoritative per-layer widths this matformer-style
    /// checkpoint (Gemma 4 E2B/E4B) actually carries live in
    /// [`Self::feed_forward_by_layer`]; a uniform checkpoint (12B/26B/31B)
    /// has every entry equal to this one.
    pub feed_forward: u32,
    /// `{family}.feed_forward_length`, one entry per block --
    /// [`metadata_u32_per_layer`] broadcasts a scalar (12B/26B/31B) to
    /// `block_count` equal entries, and preserves an array (E2B/E4B, whose
    /// matformer variable-width dense FFN this key stores per layer) as-is.
    pub feed_forward_by_layer: Vec<u32>,
    pub expert_feed_forward: u32,
    pub expert_count: u32,
    pub expert_used_count: u32,
    pub head_count: u32,
    pub kv_heads_by_layer: Vec<u32>,
    pub rms_epsilon: f32,
    pub key_length: u32,
    pub value_length: u32,
    pub sliding_window: u32,
    pub key_length_swa: u32,
    pub value_length_swa: u32,
    pub sliding_window_pattern: Vec<bool>,
    /// `{family}.attention.shared_kv_layers` -- the count of TRAILING
    /// layers (`block_count - shared_kv_layers` through `block_count - 1`)
    /// that carry no `attn_k.weight`/`attn_v.weight`/`attn_k_norm.weight`
    /// tensors of their own at all (confirmed against the real
    /// `gemma4:e2b-it-qat` header: an `UnknownTensor` load error on
    /// `blk.15.attn_k_norm.weight` when `block_count=35` and this key reads
    /// `20`). `0` (the default `metadata_u32_optional` returns when the key
    /// is absent, e.g. E4B/12B/26B/31B) means every layer owns its own
    /// K/V -- this crate's prior behaviour, unchanged.
    pub shared_kv_layers: u32,
    pub rope_freq_base: f32,
    pub rope_freq_base_swa: f32,
    pub rope_dimension_count: u32,
    pub rope_dimension_count_swa: u32,
    pub final_logit_softcapping: f32,
}

pub fn from_metadata(parsed: &ParsedGguf) -> Result<Architecture, InteropError> {
    let family = metadata_str(parsed, "general.architecture")?;
    let prefix = |name: &str| format!("{family}.{name}");

    let embedding = metadata_u32(parsed, &prefix("embedding_length"))?;
    let block_count = metadata_u32(parsed, &prefix("block_count"))?;
    let kv_heads_by_layer =
        metadata_u32_per_layer(parsed, &prefix("attention.head_count_kv"), block_count)?;
    let sliding_window_pattern = metadata_bool_per_layer(
        parsed,
        &prefix("attention.sliding_window_pattern"),
        block_count,
    )?;
    let rope_freq_base = metadata_f32_optional(
        parsed,
        &prefix("rope.freq_base"),
        proxima_tensor::sized::ROPE_FREQ_BASE_DEFAULT,
    );
    let rope_dimension_count = metadata_u32(parsed, &prefix("rope.dimension_count"))?;
    let feed_forward_by_layer =
        metadata_u32_per_layer(parsed, &prefix("feed_forward_length"), block_count)?;
    let feed_forward = *feed_forward_by_layer.first().unwrap_or(&0);

    Ok(Architecture {
        vocab: vocab_from_token_embedding(parsed, embedding)?,
        embedding,
        block_count,
        feed_forward,
        feed_forward_by_layer,
        // MoE-only keys (12B/26B/31B carry all three; E2B/E4B are dense and
        // carry none) -- absent means "this checkpoint is not
        // mixture-of-experts", not a malformed file, the same shape
        // `crate::bind`'s own `metadata_u32_optional` already uses for
        // every other family's dense-vs-MoE split.
        expert_feed_forward: metadata_u32_optional(parsed, &prefix("expert_feed_forward_length")),
        expert_count: metadata_u32_optional(parsed, &prefix("expert_count")),
        expert_used_count: metadata_u32_optional(parsed, &prefix("expert_used_count")),
        head_count: metadata_u32(parsed, &prefix("attention.head_count"))?,
        kv_heads_by_layer,
        rms_epsilon: metadata_f32_optional(
            parsed,
            &prefix("attention.layer_norm_rms_epsilon"),
            1e-6,
        ),
        key_length: metadata_u32(parsed, &prefix("attention.key_length"))?,
        value_length: metadata_u32(parsed, &prefix("attention.value_length"))?,
        sliding_window: metadata_u32(parsed, &prefix("attention.sliding_window"))?,
        key_length_swa: metadata_u32(parsed, &prefix("attention.key_length_swa"))?,
        value_length_swa: metadata_u32(parsed, &prefix("attention.value_length_swa"))?,
        sliding_window_pattern,
        shared_kv_layers: metadata_u32_optional(parsed, &prefix("attention.shared_kv_layers")),
        rope_freq_base,
        rope_freq_base_swa: metadata_f32_optional(
            parsed,
            &prefix("rope.freq_base_swa"),
            rope_freq_base,
        ),
        rope_dimension_count,
        rope_dimension_count_swa: metadata_u32_optional_or(
            parsed,
            &prefix("rope.dimension_count_swa"),
            rope_dimension_count,
        ),
        final_logit_softcapping: metadata_f32_optional(
            parsed,
            &prefix("final_logit_softcapping"),
            0.0,
        ),
    })
}
