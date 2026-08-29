//! `config.json` -> [`ModelArchitecture`], the HuggingFace counterpart to
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
//! Alloc-tier, like [`crate::bind::ModelArchitecture`] itself: parsing JSON
//! text into a struct needs an allocator (`String`/`Vec` for the
//! deserialized fields) but nothing from the platform, so this module never
//! gates on `std` -- `serde`/`serde_json` are both built here with
//! `default-features = false, features = ["alloc"]` for exactly that
//! reason (see `proxima-model-interop/Cargo.toml`).

use alloc::string::String;
use alloc::vec::Vec;

use serde::Deserialize;

use crate::bind::ModelArchitecture;
use crate::error::InteropError;

/// A HuggingFace `config.json`, exactly the fields
/// [`architecture_from_hf_config`] needs -- not a full mirror of every key
/// a real `config.json` carries (tokenizer/generation knobs like
/// `bos_token_id`, `torch_dtype`, `rope_scaling`, ... have no
/// [`ModelArchitecture`] field to land in, and `serde`'s default "ignore
/// unknown fields" behavior means this struct is forward-compatible with
/// them rather than needing to enumerate them).
///
/// `model_type`/`architectures` are read but not stored on
/// [`ModelArchitecture`] -- see [`architecture_from_hf_config`]'s doc for
/// why: neither this loader nor the GGUF one keeps its own architecture
/// family string past parsing, and the one runtime branch that exists
/// today (dense vs. mixture-of-experts) already reads off `expert_count`,
/// which this struct derives from the MoE-only fields below.
#[derive(Debug, Clone, Deserialize)]
pub struct HfConfig {
    /// e.g. `"qwen3_moe"`, `"llama"`, `"mistral"` -- read for
    /// completeness/diagnostics, not consumed by [`architecture_from_hf_config`].
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

/// Reads [`ModelArchitecture`] out of `config` -- the HF counterpart to
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
///   `crate::bind::bind_all_weights`'s GGUF path has no such ambiguity:
///   llama.cpp's own GGUF writer already folds a MoE checkpoint's per-expert
///   width into the one `{architecture}.feed_forward_length` key.
///
/// `model_type`/`architectures` are read by [`parse_hf_config`] but not
/// consulted here, matching [`crate::bind::architecture_from_metadata`]'s own
/// GGUF read: `general.architecture` is used there only to build
/// `{architecture}.*` metadata key names and is then dropped, never stored on
/// [`ModelArchitecture`]. HF's config carries the same fact under
/// `model_type`/`architectures`, and it goes unused here for the identical
/// reason -- the one branch this crate's forward-program selection makes
/// (dense vs. mixture-of-experts, `crate::generate::LoadedModel::load`) is
/// already `architecture.expert_count == 0`, and `expert_count` is exactly
/// what `num_experts`/`num_local_experts` derive below. Introducing an
/// explicit family enum was considered and rejected: writing the call site
/// both ways (`if architecture.expert_count == 0 { .. } else { .. }` vs. a
/// hypothetical `match architecture.family { Family::Dense => .., Family::Moe
/// => .. }`) produces the identical branch under a new name -- no family
/// this crate's one forward-program template could route on beyond
/// dense/MoE actually exists in the checkpoints available to test this
/// against, so the enum would carry a fact nothing reads.
#[must_use]
pub fn architecture_from_hf_config(config: &HfConfig) -> ModelArchitecture {
    let kv_heads = config.num_key_value_heads.unwrap_or(config.num_attention_heads);
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
        config.moe_intermediate_size.unwrap_or(config.intermediate_size)
    };

    ModelArchitecture {
        vocab: config.vocab_size,
        embedding: config.hidden_size,
        feed_forward,
        query_heads: config.num_attention_heads,
        kv_heads,
        head_dim,
        block_count: config.num_hidden_layers,
        expert_count,
        expert_used_count,
        rope_freq_base: config.rope_theta,
        tied_embeddings: config.tie_word_embeddings,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// The real, on-disk `config.json` this task's own evidence points at --
    /// `~/.lmstudio/models/lmstudio-community/Qwen3-30B-A3B-MLX-4bit/config.json`,
    /// copied verbatim (checked byte-for-byte against `cat` on the real
    /// file), not synthesized. A Qwen3 mixture-of-experts checkpoint: this
    /// is the fixture that proves `num_experts`/`num_experts_per_tok`/
    /// `moe_intermediate_size` are read, not just the dense fields.
    const REAL_QWEN3_MOE_CONFIG_JSON: &str = r#"{
        "architectures": ["Qwen3MoeForCausalLM"],
        "attention_bias": false,
        "attention_dropout": 0.0,
        "bos_token_id": 151643,
        "decoder_sparse_step": 1,
        "eos_token_id": 151645,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 2048,
        "initializer_range": 0.02,
        "intermediate_size": 6144,
        "max_position_embeddings": 40960,
        "max_window_layers": 48,
        "mlp_only_layers": [],
        "model_type": "qwen3_moe",
        "moe_intermediate_size": 768,
        "norm_topk_prob": true,
        "num_attention_heads": 32,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "num_hidden_layers": 48,
        "num_key_value_heads": 4,
        "output_router_logits": false,
        "quantization": {"group_size": 64, "bits": 4},
        "quantization_config": {"group_size": 64, "bits": 4},
        "rms_norm_eps": 1e-06,
        "rope_scaling": null,
        "rope_theta": 1000000.0,
        "router_aux_loss_coef": 0.001,
        "sliding_window": null,
        "tie_word_embeddings": false,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.51.0",
        "use_cache": true,
        "use_sliding_window": false,
        "vocab_size": 151936
    }"#;

    #[test]
    fn real_qwen3_moe_config_json_parses_every_field_this_crate_reads() {
        let config = parse_hf_config(REAL_QWEN3_MOE_CONFIG_JSON.as_bytes())
            .expect("real qwen3 moe config.json parses");

        assert_eq!(config.model_type, "qwen3_moe");
        assert_eq!(config.architectures, alloc::vec![String::from("Qwen3MoeForCausalLM")]);
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.num_key_value_heads, Some(4));
        assert_eq!(config.num_hidden_layers, 48);
        assert_eq!(config.intermediate_size, 6144);
        assert_eq!(config.moe_intermediate_size, Some(768));
        assert!((config.rms_norm_eps - 1e-6).abs() < 1e-12);
        assert!((config.rope_theta - 1_000_000.0).abs() < 1e-6);
        assert_eq!(config.vocab_size, 151_936);
        assert_eq!(config.head_dim, Some(128));
        assert_eq!(config.num_experts, Some(128));
        assert_eq!(config.num_experts_per_tok, Some(8));
    }

    #[test]
    fn real_qwen3_moe_config_json_derives_the_real_moe_architecture() {
        let config = parse_hf_config(REAL_QWEN3_MOE_CONFIG_JSON.as_bytes())
            .expect("real qwen3 moe config.json parses");

        let architecture = architecture_from_hf_config(&config);

        assert_eq!(
            architecture,
            ModelArchitecture {
                vocab: 151_936,
                embedding: 2048,
                feed_forward: 768,
                query_heads: 32,
                kv_heads: 4,
                head_dim: 128,
                block_count: 48,
                expert_count: 128,
                expert_used_count: 8,
                rope_freq_base: 1_000_000.0,
                tied_embeddings: false,
            },
            "feed_forward must read moe_intermediate_size (768), not intermediate_size (6144), \
             once expert_count is nonzero"
        );
    }

    /// A dense (non-MoE) config with none of `num_key_value_heads`/
    /// `num_experts`/`num_experts_per_tok`/`moe_intermediate_size`/`head_dim`
    /// present -- proves every optional field's fallback, not just the
    /// MoE-populated real fixture above.
    #[test]
    fn dense_config_with_every_optional_field_absent_derives_via_fallbacks() {
        let json = r#"{
            "model_type": "llama",
            "hidden_size": 8,
            "num_attention_heads": 2,
            "num_hidden_layers": 4,
            "intermediate_size": 32,
            "vocab_size": 100
        }"#;
        let config = parse_hf_config(json.as_bytes()).expect("minimal dense config.json parses");
        let architecture = architecture_from_hf_config(&config);

        assert_eq!(
            architecture,
            ModelArchitecture {
                vocab: 100,
                embedding: 8,
                feed_forward: 32,
                query_heads: 2,
                kv_heads: 2,
                head_dim: 4,
                block_count: 4,
                expert_count: 0,
                expert_used_count: 0,
                rope_freq_base: proxima_tensor::sized::ROPE_FREQ_BASE_DEFAULT,
                tied_embeddings: false,
            },
            "kv_heads falls back to query_heads, head_dim to hidden_size/num_attention_heads, \
             expert_count/expert_used_count to 0, rope_theta to the sizing-config default"
        );
    }

    /// Mixtral's own field name (`num_local_experts`, not Qwen's
    /// `num_experts`) must resolve to the SAME [`HfConfig::num_experts`]
    /// field via `#[serde(alias = ..)]` -- this is the "family destroyed"
    /// case the alias exists for: two real checkpoint families spell the
    /// identical fact two different ways.
    #[test]
    fn mixtral_style_num_local_experts_alias_reads_the_same_field_as_qwens_num_experts() {
        let json = r#"{
            "hidden_size": 8,
            "num_attention_heads": 2,
            "num_hidden_layers": 4,
            "intermediate_size": 32,
            "vocab_size": 100,
            "num_local_experts": 8,
            "num_experts_per_tok": 2
        }"#;
        let config = parse_hf_config(json.as_bytes()).expect("mixtral-style config.json parses");

        assert_eq!(config.num_experts, Some(8), "num_local_experts must alias into num_experts");
        assert_eq!(architecture_from_hf_config(&config).expert_count, 8);
    }

    /// The real, on-disk `config.json` this session downloaded --
    /// `~/.lmstudio/models/HuggingFaceTB/SmolLM2-135M-Instruct/config.json`,
    /// copied verbatim -- a dense, tied-embedding Llama-family checkpoint:
    /// the fixture that proves `tie_word_embeddings` is read into
    /// [`ModelArchitecture::tied_embeddings`], not just the MoE fields the
    /// Qwen3 fixture above exercises.
    const REAL_SMOLLM2_CONFIG_JSON: &str = r#"{
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": 576,
        "intermediate_size": 1536,
        "num_hidden_layers": 30,
        "num_attention_heads": 9,
        "num_key_value_heads": 3,
        "rms_norm_eps": 1e-05,
        "rope_theta": 100000,
        "vocab_size": 49152,
        "tie_word_embeddings": true,
        "torch_dtype": "bfloat16",
        "max_position_embeddings": 8192,
        "hidden_act": "silu",
        "attention_bias": false,
        "mlp_bias": false,
        "rope_scaling": null
    }"#;

    #[test]
    fn real_smollm2_config_json_derives_a_tied_dense_architecture() {
        let config = parse_hf_config(REAL_SMOLLM2_CONFIG_JSON.as_bytes()).expect("real smollm2 config.json parses");

        assert!(config.tie_word_embeddings, "smollm2 ships tie_word_embeddings: true");

        let architecture = architecture_from_hf_config(&config);
        assert_eq!(
            architecture,
            ModelArchitecture {
                vocab: 49_152,
                embedding: 576,
                feed_forward: 1536,
                query_heads: 9,
                kv_heads: 3,
                head_dim: 64,
                block_count: 30,
                expert_count: 0,
                expert_used_count: 0,
                rope_freq_base: 100_000.0,
                tied_embeddings: true,
            },
            "head_dim must derive as hidden_size/num_attention_heads (576/9=64) since no explicit \
             head_dim key is present, and tied_embeddings must read config's tie_word_embeddings"
        );
    }

    #[test]
    fn missing_required_field_is_a_typed_error_not_a_panic() {
        let json = r#"{"hidden_size": 8}"#;
        let outcome = parse_hf_config(json.as_bytes());
        assert!(
            matches!(outcome, Err(InteropError::MalformedHfConfig { .. })),
            "missing required field must surface as a typed error, got {outcome:?}"
        );
    }

    #[test]
    fn malformed_json_is_a_typed_error_not_a_panic() {
        let outcome = parse_hf_config(b"not json at all");
        assert!(
            matches!(outcome, Err(InteropError::MalformedHfConfig { .. })),
            "malformed json must surface as a typed error, got {outcome:?}"
        );
    }
}
