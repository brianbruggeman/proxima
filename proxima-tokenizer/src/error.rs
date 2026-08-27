//! Typed failures. Every malformed-input or malformed-vocab path returns
//! one of these — never a panic, never a silent drop.

use alloc::string::String;

use thiserror::Error;

/// Everything that can go wrong building a [`crate::vocab::Vocab`] or
/// running encode/decode over one.
#[derive(Debug, Error, PartialEq, Eq, Clone)]
pub enum TokenizerError {
    #[error("vocab is missing the base byte token for byte {byte} (display char {display:?})")]
    MissingBaseByteToken { byte: u8, display: char },

    #[error("merge rule {index} ('{left} {right}') references a token not in the vocab")]
    UnresolvedMerge {
        index: usize,
        left: String,
        right: String,
    },

    #[error("merge rule {index} produces token '{merged}', which is not in the vocab")]
    UnresolvedMergeResult { index: usize, merged: String },

    #[error("merge rule {index} ('{merge}') is not a single space-separated pair")]
    MalformedMergeRule { index: usize, merge: String },

    #[error("token id {token_id} is out of range for a vocab of size {vocab_len}")]
    TokenIdOutOfRange { token_id: u32, vocab_len: usize },

    #[error("decoded bytes are not valid utf-8")]
    InvalidUtf8,

    #[error("input is {len} bytes, which exceeds this tokenizer's configured limit of {limit}")]
    InputTooLarge { len: usize, limit: usize },

    #[error("gguf metadata is missing required tokenizer key '{key}'")]
    MissingMetadataKey { key: &'static str },

    #[error("gguf metadata key '{key}' has the wrong type for a tokenizer field")]
    WrongMetadataType { key: &'static str },

    #[error(
        "gguf metadata arrays 'tokenizer.ggml.tokens' (len {tokens_len}) and 'tokenizer.ggml.token_type' (len {token_type_len}) disagree in length"
    )]
    TokenArrayLengthMismatch {
        tokens_len: usize,
        token_type_len: usize,
    },

    #[error(
        "gguf metadata arrays 'tokenizer.ggml.tokens' (len {tokens_len}) and 'tokenizer.ggml.scores' (len {scores_len}) disagree in length"
    )]
    ScoreArrayLengthMismatch { tokens_len: usize, scores_len: usize },

    #[error("gguf tokenizer.ggml.model '{model}' is not a tokenizer family this crate supports")]
    UnsupportedTokenizerModel { model: String },

    #[error("hf tokenizer.json is not valid json, or is missing model.vocab/model.merges: {reason}")]
    MalformedHfTokenizerJson { reason: String },
}
