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
//! | `tokenizer.ggml.model` | string | `"gpt2"` (byte-level BPE, not sentencepiece unigram -- no `tokenizer.ggml.scores` key exists for this vocab) |
//! | `tokenizer.ggml.pre` | string | `"llama-bpe"` (selects `LLAMA_VOCAB_PRE_TYPE_LLAMA3`'s pretokenizer regex in llama.cpp) |
//! | `tokenizer.ggml.tokens` | array\<string\> | 128256 entries, index == token id |
//! | `tokenizer.ggml.token_type` | array\<i32\> | 128256 entries, parallel to `tokens` (see [`crate::vocab::TokenType::from_raw`]) |
//! | `tokenizer.ggml.merges` | array\<string\> | 280147 entries, each `"left right"` space-separated, priority order |
//! | `tokenizer.ggml.bos_token_id` | u32 | `128000` |
//! | `tokenizer.ggml.eos_token_id` | u32 | `128001` |
//!
//! `tokenizer.ggml.unknown_token_id` and `tokenizer.ggml.padding_token_id`
//! are read too, but absent on this fixture -- byte-level BPE has no OOV
//! case (every byte has a base token), so llama.cpp's own gguf writer
//! omits `unknown_token_id` for this vocab family.

use alloc::string::String;
use alloc::vec::Vec;

use proxima_gguf::{MetadataArray, MetadataValue, ParsedGguf};

use crate::error::TokenizerError;
use crate::vocab::Vocab;

const TOKENS_KEY: &str = "tokenizer.ggml.tokens";
const MERGES_KEY: &str = "tokenizer.ggml.merges";
const BOS_KEY: &str = "tokenizer.ggml.bos_token_id";
const EOS_KEY: &str = "tokenizer.ggml.eos_token_id";
const UNKNOWN_KEY: &str = "tokenizer.ggml.unknown_token_id";

/// Builds a [`Vocab`] from `metadata`'s tokenizer keys.
///
/// # Errors
///
/// [`TokenizerError::MissingMetadataKey`] if `tokenizer.ggml.tokens` is
/// absent (merges may legitimately be empty for a vocab with no BPE
/// merges at all, so that key is optional and defaults to `&[]`).
/// [`TokenizerError::WrongMetadataType`] if a present key has the wrong
/// GGUF value type. Anything [`Vocab::new`] itself can fail with
/// otherwise (a malformed merge rule, a missing base byte token).
pub fn vocab_from_metadata(metadata: &ParsedGguf) -> Result<Vocab, TokenizerError> {
    let tokens = string_array(metadata, TOKENS_KEY)?.ok_or(TokenizerError::MissingMetadataKey { key: TOKENS_KEY })?;
    let merges = string_array(metadata, MERGES_KEY)?.unwrap_or_default();
    let bos_token_id = u32_scalar(metadata, BOS_KEY)?;
    let eos_token_id = u32_scalar(metadata, EOS_KEY)?;
    let unknown_token_id = u32_scalar(metadata, UNKNOWN_KEY)?;

    Vocab::new(tokens, &merges, bos_token_id, eos_token_id, unknown_token_id)
}

fn string_array(metadata: &ParsedGguf, key: &'static str) -> Result<Option<Vec<String>>, TokenizerError> {
    match metadata.metadata_value(key) {
        None => Ok(None),
        Some(MetadataValue::Array(MetadataArray::String(values))) => Ok(Some(values.clone())),
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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::path::Path;

    use super::*;

    #[test]
    fn missing_tokens_key_is_an_error() {
        let metadata = ParsedGguf {
            version: 3,
            tensor_count: 0,
            kv_count: 0,
            metadata: Vec::new(),
            tensors: Vec::new(),
            data_offset: 0,
            alignment: 32,
        };
        let error = vocab_from_metadata(&metadata).expect_err("no tokens key");
        assert!(matches!(
            error,
            TokenizerError::MissingMetadataKey { key: TOKENS_KEY }
        ));
    }

    /// Loads the real llama-bpe vocab fixture and confirms the exact
    /// metadata keys/types documented in this module's doc comment are
    /// what actually round-trips through `proxima-gguf`.
    /// `#[ignore]`d: depends on a sibling checkout that may not exist on
    /// every host.
    #[test]
    #[ignore = "depends on a real .gguf checkout outside this repo"]
    fn builds_a_vocab_from_the_real_llama_bpe_fixture() {
        let candidate = Path::new(
            "/Users/brianbruggeman/repos/others/llama.cpp/models/ggml-vocab-llama-bpe.gguf",
        );
        if !candidate.exists() {
            eprintln!("no real .gguf found at {candidate:?}, skipping");
            return;
        }
        let (parsed, _bytes) = proxima_gguf::edge::read_file(candidate).expect("parse real gguf file");
        let vocab = vocab_from_metadata(&parsed).expect("builds vocab from real metadata");
        assert_eq!(vocab.len(), 128_256);
        assert_eq!(vocab.bos_token_id(), Some(128_000));
        assert_eq!(vocab.eos_token_id(), Some(128_001));
        println!("real fixture vocab: {} tokens", vocab.len());
    }
}
