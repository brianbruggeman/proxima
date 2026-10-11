//! The `config.json` fields an MLX export takes from GGUF metadata and tensor
//! directory rather than from [`ModelHparams`]: the per-family multipliers,
//! token ids, context length and the Gemma 4 sliding-window block.
//!
//! Each field is a pair of an HF key and the GGUF key it is read from, so a
//! key with no GGUF source is absent from the table and from the output.
//! [`ModelHparams`] covers the dimensions; this covers what mlx-lm reads on
//! top of them. Not derivable from a GGUF, and so never written: `torch_dtype`
//! and `dtype` (a GGUF records the quantization of the file, not the dtype of
//! the checkpoint it was converted from), `transformers_version`,
//! `initializer_range`, `attention_dropout`, `use_cache`,
//! `output_router_logits` and `router_aux_loss_coef` (HF training and
//! generation defaults with no GGUF key and no family constant),
//! `attention_k_eq_v`, `use_bidirectional_attention`,
//! `num_global_key_value_heads` and the extra stop ids of Gemma's top-level
//! `eos_token_id`. `audio_config`, `vision_config` and the media token ids are
//! not written either: the GGUF is text-only, so the export has no tower
//! weights for them to describe.

use alloc::format;
use alloc::vec::Vec;

use proxima_gguf::pipe::ParsedGguf;
use serde_json::{Value, json};

use super::mlx_config::{float_value, malformed};
use crate::bind::{ModelHparams, metadata_bool_per_layer, metadata_u32, metadata_u32_per_layer};
use crate::error::InteropError;
use crate::rope_scaling::{RopeScaling, optional_f32};

pub(super) type Field = (&'static str, Value);

pub(super) type FamilyFields = fn(&ParsedGguf, &ModelHparams) -> Result<Vec<Field>, InteropError>;

pub(super) type KeyTable = &'static [(&'static str, &'static str)];

const TOKEN_IDS: KeyTable = &[
    ("bos_token_id", "tokenizer.ggml.bos_token_id"),
    ("eos_token_id", "tokenizer.ggml.eos_token_id"),
    ("pad_token_id", "tokenizer.ggml.padding_token_id"),
];

pub(super) const GRANITE_FLOATS: KeyTable = &[
    ("attention_multiplier", "attention.scale"),
    ("embedding_multiplier", "embedding_scale"),
    ("residual_multiplier", "residual_scale"),
    ("logits_scaling", "logit_scale"),
];

pub(super) const GRANITE_INTS: KeyTable = &[("max_position_embeddings", "context_length")];

/// The rotated fraction of a full-attention head under proportional `RoPE`:
/// the GGUF keeps it only in the `rope_freqs.weight` tensor data, which this
/// sans-IO writer does not read, so it is the family constant mlx-lm itself
/// defaults to (`gemma4_text.py` `ModelArgs.__post_init__`) and the vendored
/// reference carries. Omitting it makes mlx-lm rotate the whole head.
const FULL_ATTENTION_PARTIAL_ROTARY_FACTOR: f64 = 0.25;

pub(super) const GEMMA_FLOATS: KeyTable = &[("final_logit_softcapping", "final_logit_softcapping")];

pub(super) const GEMMA_INTS: KeyTable = &[
    ("max_position_embeddings", "context_length"),
    ("global_head_dim", "attention.key_length"),
    (
        "hidden_size_per_layer_input",
        "embedding_length_per_layer_input",
    ),
    ("num_kv_shared_layers", "attention.shared_kv_layers"),
    ("sliding_window", "attention.sliding_window"),
];

pub(super) fn table_fields(
    parsed: &ParsedGguf,
    architecture: &str,
    floats: KeyTable,
    ints: KeyTable,
) -> Result<Vec<Field>, InteropError> {
    let float_fields = floats.iter().filter_map(|(hf_key, suffix)| {
        optional_f32(parsed, &format!("{architecture}.{suffix}"))
            .map(|value| float_value(value).map(|json| (*hf_key, json)))
    });
    let int_fields = ints.iter().filter_map(|(hf_key, suffix)| {
        optional_u32(parsed, &format!("{architecture}.{suffix}"))
            .map(|value| Ok((*hf_key, json!(value))))
    });
    let token_fields = TOKEN_IDS.iter().filter_map(|(hf_key, key)| {
        optional_u32(parsed, key).map(|value| Ok((*hf_key, json!(value))))
    });
    float_fields.chain(int_fields).chain(token_fields).collect()
}

fn optional_u32(parsed: &ParsedGguf, key: &str) -> Option<u32> {
    parsed
        .metadata_value(key)
        .and_then(proxima_gguf::value::MetadataValue::as_u32)
}

pub(super) fn granite_fields(
    parsed: &ParsedGguf,
    _hparams: &ModelHparams,
) -> Result<Vec<Field>, InteropError> {
    let mut fields = alloc::vec![("hidden_act", json!("silu")), attention_bias(parsed),];
    fields.extend(rope_scaling_field(&RopeScaling::from_gguf(parsed)?)?);
    Ok(fields)
}

fn attention_bias(parsed: &ParsedGguf) -> Field {
    let has_bias = parsed
        .tensors
        .iter()
        .any(|tensor| tensor.name.ends_with(".attn_q.bias"));
    ("attention_bias", json!(has_bias))
}

fn rope_scaling_field(scaling: &RopeScaling) -> Result<Option<Field>, InteropError> {
    let value = match scaling {
        RopeScaling::None => return Ok(None),
        RopeScaling::Linear { factor } => {
            json!({"rope_type": "linear", "factor": float_value(*factor)?})
        }
        RopeScaling::Yarn {
            factor,
            original_context,
            attention_factor,
            beta_fast,
            beta_slow,
            ..
        } => json!({
            "rope_type": "yarn",
            "factor": float_value(*factor)?,
            "original_max_position_embeddings": original_context,
            "attention_factor": float_value(*attention_factor)?,
            "beta_fast": float_value(*beta_fast)?,
            "beta_slow": float_value(*beta_slow)?,
        }),
    };
    Ok(Some(("rope_scaling", value)))
}

pub(super) fn gemma_fields(
    parsed: &ParsedGguf,
    hparams: &ModelHparams,
) -> Result<Vec<Field>, InteropError> {
    let mut fields = alloc::vec![
        ("hidden_activation", json!("gelu_pytorch_tanh")),
        ("enable_moe_block", json!(hparams.expert_count > 0)),
        attention_bias(parsed),
        ("layer_types", layer_types(parsed, hparams)?),
        ("rope_parameters", rope_parameters(parsed, hparams)?),
        ("use_double_wide_mlp", json!(double_wide(parsed, hparams)?)),
    ];
    fields.extend(per_layer_vocab(parsed, hparams));
    Ok(fields)
}

fn layer_types(parsed: &ParsedGguf, hparams: &ModelHparams) -> Result<Value, InteropError> {
    let key = format!("{}.attention.sliding_window_pattern", hparams.family);
    let pattern = metadata_bool_per_layer(parsed, &key, hparams.block_count)?;
    Ok(Value::Array(
        pattern
            .iter()
            .map(|sliding| {
                json!(if *sliding {
                    "sliding_attention"
                } else {
                    "full_attention"
                })
            })
            .collect(),
    ))
}

fn rope_parameters(parsed: &ParsedGguf, hparams: &ModelHparams) -> Result<Value, InteropError> {
    let sliding_key = format!("{}.rope.freq_base_swa", hparams.family);
    let sliding = optional_f32(parsed, &sliding_key).unwrap_or(hparams.rope_freq_base);
    Ok(json!({
        "full_attention": {
            "partial_rotary_factor": FULL_ATTENTION_PARTIAL_ROTARY_FACTOR,
            "rope_theta": float_value(hparams.rope_freq_base)?,
            "rope_type": "proportional",
        },
        "sliding_attention": {
            "rope_theta": float_value(sliding)?,
            "rope_type": "default",
        },
    }))
}

fn double_wide(parsed: &ParsedGguf, hparams: &ModelHparams) -> Result<bool, InteropError> {
    let key = format!("{}.feed_forward_length", hparams.family);
    let widths = metadata_u32_per_layer(parsed, &key, hparams.block_count)?;
    let first = widths
        .first()
        .copied()
        .ok_or_else(|| malformed("no layers"))?;
    Ok(widths.iter().any(|width| *width != first))
}

fn per_layer_vocab(parsed: &ParsedGguf, hparams: &ModelHparams) -> Option<Field> {
    let width = metadata_u32(
        parsed,
        &format!("{}.embedding_length_per_layer_input", hparams.family),
    )
    .ok()?;
    let table = parsed
        .tensors
        .iter()
        .find(|tensor| tensor.name == "per_layer_token_embd.weight")?;
    let per_row = u64::from(width) * u64::from(hparams.block_count);
    let vocab = table.element_count().checked_div(per_row)?;
    Some(("vocab_size_per_layer_input", json!(vocab)))
}

pub(super) fn tied_embeddings(hparams: &ModelHparams, parsed: &ParsedGguf) -> bool {
    hparams.tied_embeddings
        || parsed
            .tensors
            .iter()
            .all(|tensor| tensor.name != "output.weight")
}
