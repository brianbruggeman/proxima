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
use proxima_safetensors::Manifest;
use proxima_tensor::spec::LayerKind;

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
    #[serde(alias = "tie_embedding", default)]
    pub tie_word_embeddings: bool,
    /// Ordered mixer kind for architectures that alternate layer operators.
    #[serde(default)]
    pub layer_types: Vec<String>,
    /// LFM short-convolution cache width, serialized as `conv_L_cache`.
    #[serde(rename = "conv_L_cache", default)]
    pub conv_l_cache: Option<u32>,
    /// lfm2 scales its SwiGLU width by two thirds before optional alignment.
    #[serde(default)]
    pub block_auto_adjust_ff_dim: bool,
    /// optional multiplier applied after lfm2's two-thirds adjustment.
    #[serde(default)]
    pub block_ffn_dim_multiplier: Option<f64>,
    /// lfm2 rounds an adjusted FFN width up to this multiple.
    #[serde(default)]
    pub block_multiple_of: Option<u32>,
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
/// - For `lfm2` with a complete `layer_types` list, convolution layers carry
///   zero per-layer KV heads, the existing short-convolution architecture
///   marker; attention layers keep the configured KV-head count.
///
/// `model_type` becomes [`ModelHparams::family`], so the HF path keys the
/// same family profile the GGUF path keys by `general.architecture`; a family
/// with no profile is an error at bind time, not a default. `architectures` is
/// read by [`parse_hf_config`] for diagnostics and not consulted here.
///
/// Returns [`InteropError::MalformedHfConfig`] when an LFM2 layer schedule is
/// incomplete, contains an unsupported mixer, or requires a missing cache width.
#[must_use]
pub fn architecture_from_hf_config(config: &HfConfig) -> Result<ModelHparams, InteropError> {
    let configured_kv_heads = config
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
    let feed_forward = if config.model_type == "lfm2" && config.block_auto_adjust_ff_dim {
        lfm_feed_forward_width(config)?
    } else if expert_count == 0 {
        config.intermediate_size
    } else {
        config
            .moe_intermediate_size
            .unwrap_or(config.intermediate_size)
    };
    let kv_heads_by_layer = if config.model_type == "lfm2" {
        lfm_layer_kinds_from_config(config)?
            .into_iter()
            .map(|layer_kind| match layer_kind {
                LayerKind::ShortConv => Ok(0),
                LayerKind::Attention => Ok(configured_kv_heads),
                LayerKind::Gdn => Err(malformed_lfm("GDN is not a valid LFM2 layer type")),
            })
            .collect::<Result<Vec<_>, _>>()?
    } else {
        vec![configured_kv_heads; config.num_hidden_layers as usize]
    };
    let kv_heads = kv_heads_by_layer
        .first()
        .copied()
        .filter(|first| kv_heads_by_layer.iter().all(|heads| heads == first))
        .unwrap_or(0);

    Ok(ModelHparams {
        vocab: config.vocab_size,
        embedding: config.hidden_size,
        feed_forward,
        query_heads: config.num_attention_heads,
        kv_heads,
        kv_heads_by_layer,
        head_dim,
        block_count: config.num_hidden_layers,
        expert_count,
        expert_used_count,
        rope_freq_base: config.rope_theta,
        rms_epsilon: config.rms_norm_eps,
        tied_embeddings: config.tie_word_embeddings,
        family: config.model_type.clone(),
        sliding_rope: None,
    })
}

fn lfm_feed_forward_width(config: &HfConfig) -> Result<u32, InteropError> {
    let adjusted = u64::from(config.intermediate_size)
        .checked_mul(2)
        .map(|width| width / 3)
        .ok_or_else(|| malformed_lfm("intermediate_size overflows the LFM2 FFN calculation"))?;
    let Some(multiplier) = config.block_ffn_dim_multiplier else {
        return u32::try_from(adjusted).map_err(|_| malformed_lfm("LFM2 FFN width exceeds u32"));
    };
    if !multiplier.is_finite() || multiplier < 0.0 {
        return Err(malformed_lfm(
            "block_ffn_dim_multiplier must be finite and nonnegative",
        ));
    }
    let width = (adjusted as f64 * multiplier).trunc();
    if width > u64::MAX as f64 {
        return Err(malformed_lfm(
            "block_ffn_dim_multiplier overflows the LFM2 FFN width",
        ));
    }
    let adjusted = width as u64;
    let multiple = u64::from(config.block_multiple_of.unwrap_or(1));
    if multiple == 0 {
        return Err(malformed_lfm("block_multiple_of must be positive"));
    }
    let rounded = adjusted
        .checked_add(multiple - 1)
        .map(|width| width / multiple * multiple)
        .ok_or_else(|| malformed_lfm("aligned LFM2 FFN width overflows"))?;
    u32::try_from(rounded).map_err(|_| malformed_lfm("LFM2 FFN width exceeds u32"))
}

/// Resolves LFM's ordered mixer schedule and verifies each layer against the
/// checkpoint's safetensors directory. A config label alone cannot establish
/// that the corresponding operator weights exist in this checkpoint.
pub fn lfm_layer_kinds_from_manifest(
    config: &HfConfig,
    manifest: &Manifest,
) -> Result<Vec<LayerKind>, InteropError> {
    let layer_kinds = lfm_layer_kinds_from_config(config)?;

    let conv_width = config.conv_l_cache.unwrap_or(0);
    let head_dim = config
        .head_dim
        .unwrap_or_else(|| config.hidden_size / config.num_attention_heads.max(1));
    let query_width = u64::from(config.num_attention_heads) * u64::from(head_dim);
    let kv_width = u64::from(
        config
            .num_key_value_heads
            .unwrap_or(config.num_attention_heads),
    ) * u64::from(head_dim);
    let hidden_width = u64::from(config.hidden_size);

    for (layer_index, layer_kind) in layer_kinds.iter().enumerate() {
        let layer = layer_index as u32;
        let prefix = alloc::format!("model.layers.{layer}.");
        match layer_kind {
            LayerKind::ShortConv => {
                require_lfm_shape(
                    manifest,
                    &alloc::format!("{prefix}conv.conv.weight"),
                    &[hidden_width, 1, u64::from(conv_width)],
                )?;
                require_lfm_shape(
                    manifest,
                    &alloc::format!("{prefix}conv.in_proj.weight"),
                    &[hidden_width * 3, hidden_width],
                )?;
                require_lfm_shape(
                    manifest,
                    &alloc::format!("{prefix}conv.out_proj.weight"),
                    &[hidden_width, hidden_width],
                )?;
                for suffix in [
                    "self_attn.q_proj.weight",
                    "self_attn.k_proj.weight",
                    "self_attn.v_proj.weight",
                    "self_attn.out_proj.weight",
                    "self_attn.q_layernorm.weight",
                    "self_attn.k_layernorm.weight",
                ] {
                    reject_lfm_marker(manifest, &alloc::format!("{prefix}{suffix}"))?;
                }
            }
            LayerKind::Attention => {
                for (suffix, shape) in [
                    ("self_attn.q_proj.weight", vec![query_width, hidden_width]),
                    ("self_attn.k_proj.weight", vec![kv_width, hidden_width]),
                    ("self_attn.v_proj.weight", vec![kv_width, hidden_width]),
                    (
                        "self_attn.out_proj.weight",
                        vec![hidden_width, hidden_width],
                    ),
                    ("self_attn.q_layernorm.weight", vec![u64::from(head_dim)]),
                    ("self_attn.k_layernorm.weight", vec![u64::from(head_dim)]),
                ] {
                    require_lfm_shape(manifest, &alloc::format!("{prefix}{suffix}"), &shape)?;
                }
                for suffix in [
                    "conv.conv.weight",
                    "conv.in_proj.weight",
                    "conv.out_proj.weight",
                ] {
                    reject_lfm_marker(manifest, &alloc::format!("{prefix}{suffix}"))?;
                }
            }
            LayerKind::Gdn => {
                return Err(malformed_lfm("GDN is not a valid LFM2 layer type"));
            }
        }
    }

    Ok(layer_kinds)
}

fn lfm_layer_kinds_from_config(config: &HfConfig) -> Result<Vec<LayerKind>, InteropError> {
    if config.model_type != "lfm2" || config.layer_types.len() != config.num_hidden_layers as usize
    {
        return Err(malformed_lfm(
            "model type or layer_types length is inconsistent",
        ));
    }

    let conv_width = config.conv_l_cache.unwrap_or(0);
    let mut layer_kinds = Vec::with_capacity(config.layer_types.len());

    for layer_type in &config.layer_types {
        match layer_type.as_str() {
            "conv" => {
                if conv_width == 0 {
                    return Err(malformed_lfm(
                        "conv_L_cache must be positive for conv layers",
                    ));
                }
                layer_kinds.push(LayerKind::ShortConv);
            }
            "full_attention" => {
                layer_kinds.push(LayerKind::Attention);
            }
            _ => return Err(malformed_lfm("layer_types contains an unknown mixer kind")),
        }
    }

    Ok(layer_kinds)
}

fn require_lfm_shape(
    manifest: &Manifest,
    name: &str,
    expected: &[u64],
) -> Result<(), InteropError> {
    let Some(tensor) = manifest.tensor(name) else {
        return Err(InteropError::MalformedHfConfig {
            reason: alloc::format!(
                "invalid lfm2 layer schedule: required tensor {name} is missing"
            ),
        });
    };
    if tensor.shape != expected {
        return Err(InteropError::MalformedHfConfig {
            reason: alloc::format!(
                "invalid lfm2 layer schedule: tensor {name} has shape {:?}, expected {expected:?}",
                tensor.shape
            ),
        });
    }
    Ok(())
}

fn reject_lfm_marker(manifest: &Manifest, name: &str) -> Result<(), InteropError> {
    if manifest.tensor(name).is_some() {
        return Err(InteropError::MalformedHfConfig {
            reason: alloc::format!(
                "invalid lfm2 layer schedule: conflicting mixer tensor {name} is present"
            ),
        });
    }
    Ok(())
}

fn malformed_lfm(reason: &str) -> InteropError {
    InteropError::MalformedHfConfig {
        reason: alloc::format!("invalid lfm2 layer schedule: {reason}"),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
