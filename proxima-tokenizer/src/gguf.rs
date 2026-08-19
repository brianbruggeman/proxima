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

    /// The decode path end to end at the real openchat-3.5-1210 vocab
    /// scale (32002 tokens): a synthetic logits vector with a known peak
    /// runs through [`crate::sample::greedy_pick`], and the resulting id
    /// decodes ([`crate::decode`]) back to the exact expected string --
    /// including a multi-byte UTF-8 token, so the byte-level decode path
    /// is exercised, not just the id lookup. Only the metadata region is
    /// read (growing-buffer `parse_complete` loop, matching
    /// `proxima-gguf/src/restack.rs`'s `real_mixtral_file` module) -- the
    /// 3.9 GB tensor payload is never touched. `#[ignore]`d: depends on a
    /// host-local model cache outside this repo.
    #[test]
    #[ignore = "depends on a host-local openchat gguf checkout outside this repo"]
    fn greedy_decode_at_real_openchat_vocab_scale() {
        use std::io::{Read, Seek, SeekFrom};

        use crate::sample::greedy_pick;

        let candidate = Path::new(
            "/Users/brianbruggeman/.lmstudio/models/TheBloke/openchat-3.5-1210-GGUF/openchat-3.5-1210.Q4_K_S.gguf",
        );
        if !candidate.exists() {
            eprintln!("no real openchat .gguf found at {candidate:?}, skipping");
            return;
        }

        let mut file = std::fs::File::open(candidate).expect("open host-local openchat gguf fixture");

        let mut header_buf = Vec::new();
        let parsed = 'grow: {
            for cap in [4usize << 20, 16 << 20, 64 << 20] {
                header_buf.resize(cap, 0);
                file.seek(SeekFrom::Start(0)).expect("seek to file start");
                let read = file.read(&mut header_buf).expect("read gguf header region");
                header_buf.truncate(read);
                if let Ok(parsed) = proxima_gguf::pipe::parse_complete(&header_buf) {
                    break 'grow parsed;
                }
            }
            panic!("gguf metadata region did not fit in 64 MiB");
        };

        let vocab = vocab_from_metadata(&parsed).expect("builds vocab from real openchat metadata");
        assert_eq!(vocab.len(), 32_002, "real openchat-3.5-1210 vocab must have exactly 32002 tokens");

        // three distinct tokens, picked by inspecting the real vocab
        // (`tokenizer.ggml.model = "llama"`, a SentencePiece/unigram
        // vocab, confirmed via `tokenizer.ggml.scores` present and
        // `tokenizer.ggml.merges` absent): the BOS control token, a
        // plain ASCII subword, and a multi-byte UTF-8 katakana token.
        let cases: [(u32, &str); 3] = [(1, "<s>"), (450, "de"), (30_000, "ァ")];

        for (token_id, expected_text) in cases {
            let mut logits = alloc::vec![0.0f32; vocab.len()];
            logits[token_id as usize] = 100.0;

            let picked = greedy_pick(&logits).expect("logits are non-empty");
            assert_eq!(picked, token_id, "greedy pick must recover the peaked token id");

            let decoded = decode_ids_for_test(&vocab, &[picked]);
            assert_eq!(decoded, expected_text, "decode must recover the exact expected text");
        }

        // degenerate control: a flat, constant logits vector carries no
        // signal. it must NOT resolve to any of the peaked ids above --
        // otherwise a broken argmax (e.g. one that always returns a fixed
        // index) could pass the assertions above by coincidence.
        let flat_logits = alloc::vec![1.0f32; vocab.len()];
        let flat_pick = greedy_pick(&flat_logits).expect("flat logits are non-empty");
        assert_eq!(flat_pick, 0, "ties resolve deterministically to the lowest id");
        for (token_id, _) in cases {
            assert_ne!(flat_pick, token_id, "a flat vector must not coincidentally hit a real peak's id");
        }
    }

    #[cfg(test)]
    fn decode_ids_for_test(vocab: &Vocab, ids: &[u32]) -> String {
        crate::decode(ids, vocab).expect("decodes real vocab ids")
    }
}
