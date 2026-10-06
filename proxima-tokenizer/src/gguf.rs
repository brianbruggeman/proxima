//! Builds a [`Vocab`] from a [`proxima_gguf::ParsedGguf`]'s metadata.
//! Feature-gated (`gguf`) so the tokenizer core has no hard dependency on
//! the GGUF reader -- a caller who already has tokens/merges from
//! somewhere else (a plain JSON vocab, a hand-built test fixture) never
//! pulls this module in.
//!
//! # Real metadata keys
//!
//! Dumped directly from
//! `~/repos/others/llama.cpp/models/ggml-vocab-llama-bpe.gguf` (a
//! vocab-only fixture, `tensor_count: 0`) rather than assumed from memory
//! -- every key below is one this crate has actually seen on the wire:
//!
//! | key | type | on the real fixture |
//! |---|---|---|
//! | `tokenizer.ggml.model` | string | `"gpt2"` (byte-level BPE) on the llama-bpe fixture; the SentencePiece/SPM value (`MODEL_UNIGRAM`) on the openchat-3.5-1210 fixture below -- this is the key [`vocab_from_metadata`] dispatches the encoder on |
//! | `tokenizer.ggml.pre` | string | `"llama-bpe"` on the llama-bpe fixture, the digit-per-pretoken names on the Qwen checkpoints (see [`PreType::from_gguf_name`]); [`vocab_from_metadata`] maps it to a [`PreType`] exactly as llama.cpp does (`llama-vocab.cpp:2168-2271`), and a `"gpt2"` vocab with a missing value uses the default pre-split with a warning, and an unmapped value is an error |
//! | `tokenizer.ggml.tokens` | array\<string\> | 128256 entries, index == token id |
//! | `tokenizer.ggml.token_type` | array\<i32\> | 128256 entries, parallel to `tokens` (see [`crate::vocab::TokenType::from_raw`]) |
//! | `tokenizer.ggml.merges` | array\<string\> | 280147 entries, each `"left right"` space-separated, priority order -- `"gpt2"` vocabs only |
//! | `tokenizer.ggml.scores` | array\<f32\> | one per token, `MODEL_UNIGRAM` vocabs only -- dumped from the real 32002-token openchat-3.5-1210 fixture (a scores-driven model, no `merges` key at all) |
//! | `tokenizer.ggml.bos_token_id` | u32 | `128000` |
//! | `tokenizer.ggml.eos_token_id` | u32 | `128001` |
//! | `tokenizer.ggml.add_bos_token` | bool | present (`true`) on the real openchat-3.5-1210 and Nous-Hermes-2-Mixtral fixtures; **absent** on the real deepseek-coder-33b-instruct and 8B-A1B short-conv fixtures -- this is llama.cpp's own conversion-time heuristic, not guaranteed present, so [`vocab_from_metadata`] threads it through as `Option<bool>` ([`crate::vocab::Vocab::with_bos_eos_policy`]) rather than defaulting it |
//! | `tokenizer.ggml.add_eos_token` | bool | `false` on the same two fixtures that carry `add_bos_token`; absent on the same two that lack it |
//!
//! `tokenizer.ggml.unknown_token_id` and `tokenizer.ggml.padding_token_id`
//! are read too, but absent on the llama-bpe fixture -- byte-level BPE has
//! no OOV case (every byte has a base token), so llama.cpp's own gguf
//! writer omits `unknown_token_id` for this vocab family.

use alloc::string::String;
use alloc::vec::Vec;

use proxima_gguf::{MetadataArray, MetadataValue, ParsedGguf};

use crate::error::TokenizerError;
use crate::gguf_names::{MODEL_BYTE_LEVEL_BPE, MODEL_UNIGRAM};
use crate::pretokenize::PreType;
use crate::vocab::{TokenType, Vocab};

const MODEL_KEY: &str = "tokenizer.ggml.model";
const TOKENS_KEY: &str = "tokenizer.ggml.tokens";
const PRE_KEY: &str = "tokenizer.ggml.pre";
const ARCHITECTURE_KEY: &str = "general.architecture";
const MERGES_KEY: &str = "tokenizer.ggml.merges";
const SCORES_KEY: &str = "tokenizer.ggml.scores";
const TOKEN_TYPE_KEY: &str = "tokenizer.ggml.token_type";
const BOS_KEY: &str = "tokenizer.ggml.bos_token_id";
const EOS_KEY: &str = "tokenizer.ggml.eos_token_id";
const UNKNOWN_KEY: &str = "tokenizer.ggml.unknown_token_id";
const ADD_BOS_KEY: &str = "tokenizer.ggml.add_bos_token";
const ADD_EOS_KEY: &str = "tokenizer.ggml.add_eos_token";

/// Builds a [`Vocab`] from `metadata`'s tokenizer keys. The engine is read
/// off the companion arrays the file carries, never a caller flag and never a
/// model name: a `tokenizer.ggml.merges` array selects the merges-driven
/// constructor ([`Vocab::new`]), a scores-driven (`MODEL_UNIGRAM`) model with a
/// `tokenizer.ggml.scores` array and no merges selects the scores-driven one
/// ([`Vocab::new_unigram`]). A merges-driven vocab whose own tokens are
/// char-level (see [`Vocab::is_char_level_bpe`]) carries its own splitting and
/// reads no `tokenizer.ggml.pre`; every other merges-driven vocab maps
/// `tokenizer.ggml.pre` through [`PreType::from_gguf_name`].
///
/// # Errors
///
/// [`TokenizerError::MissingMetadataKey`] if `tokenizer.ggml.tokens` or
/// `tokenizer.ggml.model` is absent, or if the declared model's required
/// companion array (`merges` for `MODEL_BYTE_LEVEL_BPE`, `scores` for `MODEL_UNIGRAM`) is
/// missing, named by which key came up empty.
/// [`TokenizerError::UnsupportedTokenizerModel`] for any other model value
/// that carries no merges. [`TokenizerError::UnsupportedPreTokenizer`] for a byte-level vocab whose `tokenizer.ggml.pre` has no exact mapping ([`PreType::from_gguf_name`]); [`TokenizerError::WrongMetadataType`] if a
/// present key has the wrong GGUF value type. Anything [`Vocab::new`]/
/// [`Vocab::new_unigram`] can fail with otherwise (a malformed merge rule,
/// a missing base byte token, a scores/tokens length mismatch).
pub fn vocab_from_metadata(metadata: &ParsedGguf) -> Result<Vocab, TokenizerError> {
    let tokens = string_array(metadata, TOKENS_KEY)?
        .ok_or(TokenizerError::MissingMetadataKey { key: TOKENS_KEY })?;
    let bos_token_id = u32_scalar(metadata, BOS_KEY)?;
    let eos_token_id = u32_scalar(metadata, EOS_KEY)?;
    let unknown_token_id = u32_scalar(metadata, UNKNOWN_KEY)?;
    let model = string_scalar(metadata, MODEL_KEY)?
        .ok_or(TokenizerError::MissingMetadataKey { key: MODEL_KEY })?;
    let token_types = token_type_array(metadata, tokens.len())?;
    let add_bos_token = bool_scalar(metadata, ADD_BOS_KEY)?;
    let add_eos_token = bool_scalar(metadata, ADD_EOS_KEY)?;

    let merges = string_array(metadata, MERGES_KEY)?;
    let vocab = match (model.as_str(), merges) {
        (MODEL_UNIGRAM, None) => {
            let scores = f32_array(metadata, SCORES_KEY)?
                .ok_or(TokenizerError::MissingMetadataKey { key: SCORES_KEY })?;
            Vocab::new_unigram(tokens, scores, bos_token_id, eos_token_id, unknown_token_id)?
        }
        (_, Some(merges)) => {
            let vocab = Vocab::new(
                tokens,
                &merges,
                bos_token_id,
                eos_token_id,
                unknown_token_id,
            )?;
            if vocab.is_char_level_bpe() {
                vocab
            } else {
                vocab.with_pre_type(pre_type_from_metadata(metadata)?)
            }
        }
        (MODEL_BYTE_LEVEL_BPE, None) => return Err(TokenizerError::MissingMetadataKey { key: MERGES_KEY }),
        (other, None) => {
            return Err(TokenizerError::UnsupportedTokenizerModel {
                model: String::from(other),
            });
        }
    };

    let vocab = match token_types {
        Some(token_types) => vocab.with_token_types(token_types)?,
        None => vocab,
    };
    Ok(vocab.with_bos_eos_policy(add_bos_token, add_eos_token))
}

/// Selects the byte-level pre-split from `tokenizer.ggml.pre`, mapped as
/// llama.cpp maps it ([`PreType::from_gguf_name`]).
///
/// A missing or empty value is llama.cpp's `default` pre type with a warning
/// (`llama-vocab.cpp:2159-2167`), so it is [`PreType::Default`] here with the
/// same warning, naming the file's `general.architecture`. An unmapped value
/// is an error carrying that value: llama.cpp throws on a name it does not
/// know (`llama-vocab.cpp:2423`), and for the names it does know but this
/// crate has no exact rule for, any stand-in would be silent wrong output.
///
/// # Errors
///
/// [`TokenizerError::UnsupportedPreTokenizer`] for a value with no exact
/// mapping; [`TokenizerError::WrongMetadataType`] for a non-string value.
fn pre_type_from_metadata(metadata: &ParsedGguf) -> Result<PreType, TokenizerError> {
    let Some(name) = string_scalar(metadata, PRE_KEY)?.filter(|name| !name.is_empty()) else {
        let architecture = string_scalar(metadata, ARCHITECTURE_KEY)?;
        proxima_telemetry::warn!(
            architecture = ?architecture,
            "missing tokenizer.ggml.pre, using the default pre-tokenizer; generation quality may be degraded, consider regenerating the model"
        );
        return Ok(PreType::Default);
    };
    PreType::from_gguf_name(&name).ok_or(TokenizerError::UnsupportedPreTokenizer { pre: name })
}

/// Reads `tokenizer.ggml.token_type` (`array<i32>`, parallel to
/// `tokenizer.ggml.tokens`) if present, mapping each raw value through
/// [`TokenType::from_raw`]. `None` when the key is absent -- not every
/// vocab family carries it (byte-level BPE vocabs work fine without it,
/// since [`Vocab::with_token_types`] is purely additive), so this is not
/// [`TokenizerError::MissingMetadataKey`].
///
/// # Errors
///
/// [`TokenizerError::TokenArrayLengthMismatch`] if the array's length
/// disagrees with `tokens_len`; [`TokenizerError::WrongMetadataType`] if
/// the key is present with the wrong GGUF value type.
fn token_type_array(
    metadata: &ParsedGguf,
    tokens_len: usize,
) -> Result<Option<Vec<TokenType>>, TokenizerError> {
    let raw = match metadata.metadata_value(TOKEN_TYPE_KEY) {
        None => return Ok(None),
        Some(MetadataValue::Array(MetadataArray::I32(values))) => values,
        Some(_) => {
            return Err(TokenizerError::WrongMetadataType {
                key: TOKEN_TYPE_KEY,
            });
        }
    };
    if raw.len() != tokens_len {
        return Err(TokenizerError::TokenArrayLengthMismatch {
            tokens_len,
            token_type_len: raw.len(),
        });
    }
    Ok(Some(raw.iter().copied().map(TokenType::from_raw).collect()))
}

fn string_array(
    metadata: &ParsedGguf,
    key: &'static str,
) -> Result<Option<Vec<String>>, TokenizerError> {
    match metadata.metadata_value(key) {
        None => Ok(None),
        Some(MetadataValue::Array(MetadataArray::String(values))) => Ok(Some(values.clone())),
        Some(_) => Err(TokenizerError::WrongMetadataType { key }),
    }
}

fn f32_array(metadata: &ParsedGguf, key: &'static str) -> Result<Option<Vec<f32>>, TokenizerError> {
    match metadata.metadata_value(key) {
        None => Ok(None),
        Some(MetadataValue::Array(MetadataArray::F32(values))) => Ok(Some(values.clone())),
        Some(_) => Err(TokenizerError::WrongMetadataType { key }),
    }
}

fn string_scalar(
    metadata: &ParsedGguf,
    key: &'static str,
) -> Result<Option<String>, TokenizerError> {
    match metadata.metadata_value(key) {
        None => Ok(None),
        Some(MetadataValue::String(value)) => Ok(Some(value.clone())),
        Some(_) => Err(TokenizerError::WrongMetadataType { key }),
    }
}

fn u32_scalar(metadata: &ParsedGguf, key: &'static str) -> Result<Option<u32>, TokenizerError> {
    match metadata.metadata_value(key) {
        None => Ok(None),
        Some(MetadataValue::U32(value)) => Ok(Some(*value)),
        Some(_) => Err(TokenizerError::WrongMetadataType { key }),
    }
}

/// `None` when `key` is absent -- not every checkpoint's own GGUF
/// conversion carries an opinion (confirmed against real, on-disk
/// `deepseek-coder-33b-instruct` and 8B-A1B short-conv fixtures, both of which
/// omit `tokenizer.ggml.add_bos_token`/`add_eos_token` entirely), distinct
/// from `Some(false)` (openchat-3.5-1210's real fixture carries
/// `add_bos_token = true`, `add_eos_token = false` explicitly). See
/// [`crate::vocab::Vocab::with_bos_eos_policy`].
fn bool_scalar(metadata: &ParsedGguf, key: &'static str) -> Result<Option<bool>, TokenizerError> {
    match metadata.metadata_value(key) {
        None => Ok(None),
        Some(MetadataValue::Bool(value)) => Ok(Some(*value)),
        Some(_) => Err(TokenizerError::WrongMetadataType { key }),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
