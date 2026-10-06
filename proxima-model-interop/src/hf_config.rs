//! `config.json` -> [`ModelHparams`], the HuggingFace counterpart to
//! [`crate::bind::architecture_from_metadata`]'s GGUF-metadata reader. Same
//! output type, same four hyperparameter groups (embedding/attention/
//! feed-forward/MoE), different wire container: GGUF keys every dimension
//! under `{architecture}.*`; HF's `config.json` names every field directly,
//! no per-architecture prefix, so [`HfConfig`] is a flat `serde`-derived
//! struct rather than the string-keyed metadata lookups `bind.rs` needs.
//!
//! Confirmed against the real
//! `~/.lmstudio/models/lmstudio-community/Qwen3-30B-A3B-MLX-4bit/config.json`
//! on this host (a real Qwen3 MoE checkpoint's own file, not a synthetic
//! fixture): every field below is present there under exactly the name
//! read here, including the MoE-only `num_experts`/`num_experts_per_tok`
//! pair and the separate `moe_intermediate_size` (see
//! [`architecture_from_hf_config`]'s own doc for why that one is NOT the
//! same field as `intermediate_size`).
//!
//! Alloc-tier, like [`crate::bind::ModelHparams`] itself: parsing JSON
//! text into a struct needs an allocator (`String`/`Vec` for the
//! deserialized fields) but nothing from the platform, so this module never
//! gates on `std` -- `serde`/`serde_json` are both built here with
//! `default-features = false, features = ["alloc"]` for exactly that
//! reason (see `proxima-model-interop/Cargo.toml`).

use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use serde::Deserialize;

use crate::bind::ModelHparams;
use crate::error::InteropError;

/// A HuggingFace `config.json`, exactly the fields
/// [`architecture_from_hf_config`] needs -- not a full mirror of every key
/// a real `config.json` carries (tokenizer/generation knobs like
/// `bos_token_id`, `torch_dtype`, `rope_scaling`, ... have no
/// [`ModelHparams`] field to land in, and `serde`'s default "ignore
/// unknown fields" behavior means this struct is forward-compatible with
/// them rather than needing to enumerate them).
///
/// `model_type` is stored on [`ModelHparams::family`], the key of the
/// family profile the HF path reads (see [`architecture_from_hf_config`]);
/// `architectures` is read for diagnostics only.
#[derive(Debug, Clone, Deserialize)]
pub struct HfConfig {
    /// e.g. `"qwen3_moe"` -- the family profile key,
    /// carried to [`ModelHparams::family`] by [`architecture_from_hf_config`].
    #[serde(default)]
    pub model_type: String,
    /// e.g. `["Qwen3MoeForCausalLM"]` -- same status as `model_type`.
    #[serde(default)]
    pub architectures: Vec<String>,
    pub hidden_size: u32,
    pub num_attention_heads: u32,
    /// Absent on a plain multi-head-attention checkpoint (no GQA), in which
    /// case KV heads equal query heads -- mirrors
    /// [`crate::bind::architecture_from_metadata`]'s GGUF read, which has a
    /// real required key here (`{architecture}.attention.head_count_kv`)
    /// because llama.cpp's own GGUF writer always emits it; HF's own
    /// `config.json` schema does not guarantee the key for a non-GQA model.
    #[serde(default)]
    pub num_key_value_heads: Option<u32>,
    pub num_hidden_layers: u32,
    /// Per-layer (dense) feed-forward width. For a MoE checkpoint this is
    /// NOT the per-expert width -- see [`architecture_from_hf_config`].
    pub intermediate_size: u32,
    /// Per-expert feed-forward width, MoE-only. Absent on a dense
    /// checkpoint.
    #[serde(default)]
    pub moe_intermediate_size: Option<u32>,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    pub vocab_size: u32,
    /// Explicit per-head rotary dimension. Absent when it equals
    /// `hidden_size / num_attention_heads`, the common case
    /// [`architecture_from_hf_config`] derives when this is `None`.
    #[serde(default)]
    pub head_dim: Option<u32>,
    /// Total expert count per MoE layer. Named `num_experts` on Qwen's own
    /// `config.json` (confirmed on the real Qwen3-30B-A3B-MLX-4bit file);
    /// `num_local_experts` is Mixtral's name for the identical field, so
    /// this reads either.
    #[serde(alias = "num_local_experts", default)]
    pub num_experts: Option<u32>,
    /// How many of `num_experts` each token routes to, MoE-only.
    #[serde(default)]
    pub num_experts_per_tok: Option<u32>,
    /// `true` when the checkpoint reuses `model.embed_tokens.weight` as its
    /// LM head rather than shipping a separate `lm_head.weight` tensor.
    /// Confirmed on the real
    /// `~/.lmstudio/models/HuggingFaceTB/SmolLM2-135M-Instruct/config.json`
    /// (`"tie_word_embeddings": true`, and that checkpoint's own
    /// `model.safetensors` manifest carries no `lm_head.weight` entry at
    /// all). Defaults to `false` for a `config.json` that omits the key,
    /// matching HF's own schema default.
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

fn default_rms_norm_eps() -> f32 {
    1e-5
}

fn default_rope_theta() -> f32 {
    proxima_tensor::sized::ROPE_FREQ_BASE_DEFAULT
}

/// Parses `bytes` (a `config.json` file's own bytes, however the caller
/// read them -- this crate stays sans-IO and never opens a file itself) as
/// an [`HfConfig`].
///
/// # Errors
///
/// [`InteropError::MalformedHfConfig`] if `bytes` is not valid JSON, or is
/// missing/mis-typing one of [`HfConfig`]'s required fields.
pub fn parse_hf_config(bytes: &[u8]) -> Result<HfConfig, InteropError> {
    serde_json::from_slice(bytes).map_err(|error| InteropError::MalformedHfConfig {
        reason: alloc::string::ToString::to_string(&error),
    })
}

/// Reads [`ModelHparams`] out of `config` -- the HF counterpart to
/// [`crate::bind::architecture_from_metadata`]'s GGUF read. Every field maps
/// straight across except two, both because HF's schema does not carry
/// GGUF's exact shape:
///
/// - `kv_heads` falls back to `num_attention_heads` when
///   `num_key_value_heads` is absent (no GQA), rather than
///   [`InteropError::MissingMetadataKey`] the way a truly-absent required
///   GGUF key would -- HF's own schema does not require this key for a
///   non-GQA checkpoint, so treating absence as "equal to query heads" reads
///   what the format actually promises instead of inventing a stricter
///   contract than HF's own spec has.
/// - `feed_forward` reads `moe_intermediate_size` when `config.expert_count`
///   (derived just above it) is nonzero, falling back to `intermediate_size`
///   only if that MoE-specific field is itself absent. Confirmed against the
///   real Qwen3-30B-A3B-MLX-4bit `config.json`: it declares BOTH
///   `intermediate_size: 6144` (unused by this checkpoint's own MoE forward
///   pass, kept only for architecture-family compatibility) and
///   `moe_intermediate_size: 768` (the real per-expert FFN width) -- reading
///   the wrong one would silently build a program with the wrong `feed_forward`
///   dimension for every `ffn_gate`/`ffn_up`/`ffn_down` expert weight.
///   `crate::bind_leaves::bind_program_leaves`'s GGUF path has no such ambiguity:
///   llama.cpp's own GGUF writer already folds a MoE checkpoint's per-expert
///   width into the one `{architecture}.feed_forward_length` key.
///
/// `model_type` becomes [`ModelHparams::family`], so the HF path keys the
/// same family profile the GGUF path keys by `general.architecture`; a family
/// with no profile is an error at bind time, not a default. `architectures` is
/// read by [`parse_hf_config`] for diagnostics and not consulted here.
#[must_use]
pub fn architecture_from_hf_config(config: &HfConfig) -> ModelHparams {
    let kv_heads = config
        .num_key_value_heads
        .unwrap_or(config.num_attention_heads);
    let head_dim = config
        .head_dim
        .unwrap_or_else(|| config.hidden_size / config.num_attention_heads.max(1));
    let expert_count = config.num_experts.unwrap_or(0);
    let expert_used_count = if expert_count == 0 {
        0
    } else {
        config.num_experts_per_tok.unwrap_or(0)
    };
    let feed_forward = if expert_count == 0 {
        config.intermediate_size
    } else {
        config
            .moe_intermediate_size
            .unwrap_or(config.intermediate_size)
    };

    ModelHparams {
        vocab: config.vocab_size,
        embedding: config.hidden_size,
        feed_forward,
        query_heads: config.num_attention_heads,
        kv_heads,
        kv_heads_by_layer: vec![kv_heads; config.num_hidden_layers as usize],
        head_dim,
        block_count: config.num_hidden_layers,
        expert_count,
        expert_used_count,
        rope_freq_base: config.rope_theta,
        rms_epsilon: config.rms_norm_eps,
        tied_embeddings: config.tie_word_embeddings,
        family: config.model_type.clone(),
        sliding_rope: None,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
