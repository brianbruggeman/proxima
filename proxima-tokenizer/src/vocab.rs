//! A constructed byte-level BPE vocabulary: sans-IO, plain owned data. A
//! caller builds one directly ([`Vocab::new`]) or from parsed GGUF metadata
//! (`crate::gguf::vocab_from_metadata`, behind the `gguf` feature) and hands
//! it to [`crate::bpe`]/[`crate::pipe`] — this type never touches a file.

use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use crate::byte_level::{byte_to_char, char_to_byte};
use crate::error::TokenizerError;

/// Mirrors llama.cpp's `llama_token_type` (`include/llama.h:131-137`) --
/// the GGUF `tokenizer.ggml.token_type` array tags every vocab entry with
/// one of these, and BOS/EOS/UNK handling reads it back to decide whether
/// a token is ordinary text or a control/special marker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenType {
    Undefined,
    Normal,
    Unknown,
    Control,
    UserDefined,
    Unused,
    Byte,
}

impl TokenType {
    /// Maps the raw `i32` GGUF stores (`tokenizer.ggml.token_type`) to a
    /// [`TokenType`]. Unrecognized values fall back to `Undefined` rather
    /// than erroring — the type tag is advisory (it steers whether a token
    /// participates in ordinary BPE merging), not load-bearing for
    /// correctness the way a missing token or merge would be.
    #[must_use]
    pub fn from_raw(raw: i32) -> Self {
        match raw {
            1 => Self::Normal,
            2 => Self::Unknown,
            3 => Self::Control,
            4 => Self::UserDefined,
            5 => Self::Unused,
            6 => Self::Byte,
            _ => Self::Undefined,
        }
    }
}

/// One fully-resolved merge rule: the pair of token ids it fires on, its
/// priority (lower fires first -- the position it held in
/// `tokenizer.ggml.merges`), and the token id the pair collapses into.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MergeRule {
    rank: u32,
    merged_id: u32,
}

/// A byte-level BPE vocabulary: every token's display-domain string, its
/// decoded raw bytes, the merge table, and the handful of special token
/// ids callers care about (BOS/EOS/unknown).
#[derive(Debug, Clone, PartialEq)]
pub struct Vocab {
    id_to_token: Vec<String>,
    id_to_bytes: Vec<Vec<u8>>,
    token_to_id: BTreeMap<String, u32>,
    merge_ranks: BTreeMap<(u32, u32), MergeRule>,
    scores: Vec<f32>,
    base_byte_token_id: [Option<u32>; 256],
    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,
    unknown_token_id: Option<u32>,
}

impl Vocab {
    /// Builds a vocab from its three GGUF-shaped constituents: the token
    /// list (index == token id), the merge rules in priority order
    /// (`"left right"`, space-separated, display-domain strings exactly as
    /// GGUF stores them), and the optional special token ids.
    ///
    /// # Errors
    ///
    /// [`TokenizerError::MissingBaseByteToken`] if any of the 256
    /// single-byte display tokens is absent (every byte-level BPE vocab
    /// must carry all 256 — that is the alphabet encode/decode is built
    /// on). [`TokenizerError::MalformedMergeRule`],
    /// [`TokenizerError::UnresolvedMerge`], or
    /// [`TokenizerError::UnresolvedMergeResult`] if a merge rule doesn't
    /// resolve against `tokens`.
    pub fn new(
        tokens: Vec<String>,
        merges: &[String],
        bos_token_id: Option<u32>,
        eos_token_id: Option<u32>,
        unknown_token_id: Option<u32>,
    ) -> Result<Self, TokenizerError> {
        Self::assemble(tokens, merges, Vec::new(), bos_token_id, eos_token_id, unknown_token_id)
    }

    /// Builds a SentencePiece-unigram vocab (`tokenizer.ggml.model =
    /// "llama"`) from its per-token unigram scores instead of an explicit
    /// merge list -- hands to [`crate::unigram::encode_fragment`], which
    /// greedily merges the highest-`token_score` adjacent pair (both
    /// crate-private) (mirroring llama.cpp's `llm_tokenizer_spm_session`)
    /// rather than walking `Vocab::merge_rule`'s precomputed rank table the
    /// way [`crate::bpe::encode_pretoken`] does.
    ///
    /// # Errors
    ///
    /// [`TokenizerError::ScoreArrayLengthMismatch`] if `scores.len() !=
    /// tokens.len()`; anything [`Vocab::new`] itself can fail with
    /// otherwise (a missing base byte token, most likely -- SentencePiece
    /// byte-fallback vocabs spell their 256-byte alphabet as `<0xXX>`
    /// tokens, checked by `hex_fallback_token`, crate-private).
    pub fn new_unigram(
        tokens: Vec<String>,
        scores: Vec<f32>,
        bos_token_id: Option<u32>,
        eos_token_id: Option<u32>,
        unknown_token_id: Option<u32>,
    ) -> Result<Self, TokenizerError> {
        if scores.len() != tokens.len() {
            return Err(TokenizerError::ScoreArrayLengthMismatch {
                tokens_len: tokens.len(),
                scores_len: scores.len(),
            });
        }
        Self::assemble(tokens, &[], scores, bos_token_id, eos_token_id, unknown_token_id)
    }

    fn assemble(
        tokens: Vec<String>,
        merges: &[String],
        scores: Vec<f32>,
        bos_token_id: Option<u32>,
        eos_token_id: Option<u32>,
        unknown_token_id: Option<u32>,
    ) -> Result<Self, TokenizerError> {
        let token_to_id: BTreeMap<String, u32> = tokens
            .iter()
            .enumerate()
            .map(|(id, token)| (token.clone(), id as u32))
            .collect();

        let id_to_bytes: Vec<Vec<u8>> = tokens.iter().map(|token| token_bytes_for(token)).collect();

        let mut base_byte_token_id = [None; 256];
        for byte in 0..=255u8 {
            let display = byte_to_char(byte);
            let mut single_char = String::new();
            single_char.push(display);
            let token_id = token_to_id
                .get(&single_char)
                .copied()
                .or_else(|| token_to_id.get(hex_fallback_token(byte).as_str()).copied());
            if token_id.is_none() {
                return Err(TokenizerError::MissingBaseByteToken { byte, display });
            }
            base_byte_token_id[byte as usize] = token_id;
        }

        let mut merge_ranks = BTreeMap::new();
        for (rank, merge) in merges.iter().enumerate() {
            let mut halves = merge.split(' ');
            let (Some(left), Some(right), None) = (halves.next(), halves.next(), halves.next()) else {
                return Err(TokenizerError::MalformedMergeRule {
                    index: rank,
                    merge: merge.clone(),
                });
            };
            let left_id = token_to_id
                .get(left)
                .copied()
                .ok_or_else(|| TokenizerError::UnresolvedMerge {
                    index: rank,
                    left: String::from(left),
                    right: String::from(right),
                })?;
            let right_id = token_to_id
                .get(right)
                .copied()
                .ok_or_else(|| TokenizerError::UnresolvedMerge {
                    index: rank,
                    left: String::from(left),
                    right: String::from(right),
                })?;
            let mut merged = String::with_capacity(left.len() + right.len());
            merged.push_str(left);
            merged.push_str(right);
            let merged_id =
                token_to_id
                    .get(merged.as_str())
                    .copied()
                    .ok_or(TokenizerError::UnresolvedMergeResult {
                        index: rank,
                        merged,
                    })?;
            merge_ranks.insert(
                (left_id, right_id),
                MergeRule {
                    rank: rank as u32,
                    merged_id,
                },
            );
        }

        Ok(Self {
            id_to_token: tokens,
            id_to_bytes,
            token_to_id,
            merge_ranks,
            scores,
            base_byte_token_id,
            bos_token_id,
            eos_token_id,
            unknown_token_id,
        })
    }

    /// Number of tokens in the vocab.
    #[must_use]
    pub fn len(&self) -> usize {
        self.id_to_token.len()
    }

    /// Whether the vocab has zero tokens.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.id_to_token.is_empty()
    }

    /// The base single-byte token id for a raw byte -- the seed sequence
    /// BPE merging starts from.
    #[must_use]
    pub(crate) fn base_byte_token(&self, byte: u8) -> u32 {
        // built in `new`: every byte has an entry, or construction failed.
        self.base_byte_token_id[byte as usize].unwrap_or(0)
    }

    /// The merge rule for an adjacent token id pair, if one exists.
    #[must_use]
    pub(crate) fn merge_rule(&self, left: u32, right: u32) -> Option<(u32, u32)> {
        self.merge_ranks
            .get(&(left, right))
            .map(|rule| (rule.rank, rule.merged_id))
    }

    /// Whether this vocab was built via [`Vocab::new_unigram`] (carries
    /// per-token scores) rather than [`Vocab::new`] (carries merge rules).
    /// [`crate::pipe::encode`]/[`crate::pipe::decode`] read this to select
    /// [`crate::unigram::encode_fragment`] over
    /// [`crate::bpe::encode_pretoken`] -- the vocab's own shape decides,
    /// never a caller flag.
    #[must_use]
    pub fn is_unigram(&self) -> bool {
        !self.scores.is_empty()
    }

    /// The unigram log-probability score for a token id, if this is a
    /// scores-driven vocab. Higher (less negative) means "merge this pair
    /// first" in [`crate::unigram::encode_fragment`]'s greedy loop.
    #[must_use]
    pub(crate) fn token_score(&self, token_id: u32) -> Option<f32> {
        self.scores.get(token_id as usize).copied()
    }

    /// The display-domain string for a token id.
    #[must_use]
    pub fn token_str(&self, token_id: u32) -> Option<&str> {
        self.id_to_token.get(token_id as usize).map(String::as_str)
    }

    /// The raw decoded bytes a token id represents.
    #[must_use]
    pub fn token_bytes(&self, token_id: u32) -> Option<&[u8]> {
        self.id_to_bytes.get(token_id as usize).map(Vec::as_slice)
    }

    /// The token id for an exact display-domain string, if present.
    #[must_use]
    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }

    #[must_use]
    pub fn bos_token_id(&self) -> Option<u32> {
        self.bos_token_id
    }

    #[must_use]
    pub fn eos_token_id(&self) -> Option<u32> {
        self.eos_token_id
    }

    #[must_use]
    pub fn unknown_token_id(&self) -> Option<u32> {
        self.unknown_token_id
    }
}

/// The SentencePiece byte-fallback token spelling for a raw byte
/// (`"<0x1A>"`, uppercase hex, zero-padded) -- the convention llama.cpp's
/// SentencePiece/unigram vocabs (`tokenizer.ggml.model = "llama"`) use for
/// their base byte alphabet instead of the GPT-2 display alphabet
/// ([`byte_to_char`]). Checked as a fallback so [`Vocab::new`] accepts
/// either family's base-byte spelling.
fn hex_fallback_token(byte: u8) -> String {
    format!("<0x{byte:02X}>")
}

/// A token's raw byte representation: [`parse_hex_fallback_byte`]'s single
/// byte if `token` is a `"<0xXX>"` byte-fallback spelling (its whole point
/// is naming exactly one raw byte, not six literal display chars),
/// otherwise [`decode_display_string`].
fn token_bytes_for(token: &str) -> Vec<u8> {
    match parse_hex_fallback_byte(token) {
        Some(byte) => alloc::vec![byte],
        None => decode_display_string(token),
    }
}

/// Parses a SentencePiece byte-fallback token's spelling ([`hex_fallback_token`]'s
/// `"<0x1A>"` shape) back to the single raw byte it names, if `token`
/// matches that exact shape.
fn parse_hex_fallback_byte(token: &str) -> Option<u8> {
    let hex = token.strip_prefix("<0x")?.strip_suffix('>')?;
    if hex.len() != 2 {
        return None;
    }
    u8::from_str_radix(hex, 16).ok()
}

/// Converts a token's display-domain string back to the raw bytes it
/// represents. A char outside the byte-level alphabet (shouldn't occur for
/// well-formed GGUF vocabs, but special/added tokens are free-form) passes
/// through as its own UTF-8 encoding — the same fallback llama.cpp's
/// `token_to_piece` uses for `USER_DEFINED`/`CONTROL` tokens.
fn decode_display_string(token: &str) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(token.len());
    for character in token.chars() {
        match char_to_byte(character) {
            Some(byte) => bytes.push(byte),
            None => {
                let mut buffer = [0u8; 4];
                bytes.extend_from_slice(character.encode_utf8(&mut buffer).as_bytes());
            }
        }
    }
    bytes
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
pub(crate) mod tests {
    use alloc::vec;

    use super::*;
    use crate::byte_level::byte_to_char;

    /// A tiny vocab covering the base 256 byte tokens plus enough merges
    /// to combine "h" + "i" -> "hi", used across this crate's tests.
    pub(crate) fn tiny_vocab() -> Vocab {
        let mut tokens: Vec<String> = (0..=255u8).map(|byte| String::from(byte_to_char(byte))).collect();
        tokens.push(String::from("hi"));
        let mut space_hi = String::new();
        space_hi.push(byte_to_char(b' '));
        space_hi.push_str("hi");
        tokens.push(space_hi);

        let mut merge_space_hi = String::new();
        merge_space_hi.push(byte_to_char(b' '));
        merge_space_hi.push_str(" hi");

        let merges = vec![String::from("h i"), merge_space_hi];
        Vocab::new(tokens, &merges, Some(1), Some(2), None).expect("tiny vocab builds")
    }

    /// A tiny SentencePiece-unigram vocab: all 256 `<0xXX>` byte-fallback
    /// pieces plus a chained-merge case ("▁"+"h" -> "▁h", "h"+"i" -> "hi"
    /// scored higher so it wins the first pass, then "▁"+"hi" -> "▁hi"
    /// scored highest of all) -- used across this crate's unigram tests.
    pub(crate) fn tiny_unigram_vocab() -> Vocab {
        let mut tokens: Vec<String> = (0..=255u8).map(hex_fallback_token).collect();
        let mut scores = vec![0.0f32; tokens.len()];
        for (word, score) in [("\u{2581}", -3.0), ("\u{2581}h", -5.0), ("hi", -2.0), ("\u{2581}hi", -1.0)] {
            tokens.push(String::from(word));
            scores.push(score);
        }
        Vocab::new_unigram(tokens, scores, Some(1), Some(2), None).expect("tiny unigram vocab builds")
    }

    #[test]
    fn missing_base_byte_token_is_an_error() {
        let tokens: Vec<String> = (0..=254u8).map(|byte| String::from(byte_to_char(byte))).collect();
        let error = Vocab::new(tokens, &[], None, None, None).expect_err("missing byte 255");
        assert!(matches!(error, TokenizerError::MissingBaseByteToken { byte: 255, .. }));
    }

    #[test]
    fn malformed_merge_rule_is_an_error() {
        let tokens: Vec<String> = (0..=255u8).map(|byte| String::from(byte_to_char(byte))).collect();
        let merges = vec![String::from("only-one-half")];
        let error = Vocab::new(tokens, &merges, None, None, None).expect_err("malformed merge");
        assert!(matches!(error, TokenizerError::MalformedMergeRule { index: 0, .. }));
    }

    #[test]
    fn unresolved_merge_is_an_error() {
        let tokens: Vec<String> = (0..=255u8).map(|byte| String::from(byte_to_char(byte))).collect();
        let merges = vec![String::from("h nonexistent-token")];
        let error = Vocab::new(tokens, &merges, None, None, None).expect_err("unresolved merge");
        assert!(matches!(error, TokenizerError::UnresolvedMerge { index: 0, .. }));
    }

    #[test]
    fn unresolved_merge_result_is_an_error() {
        let tokens: Vec<String> = (0..=255u8).map(|byte| String::from(byte_to_char(byte))).collect();
        let merges = vec![String::from("h i")]; // "hi" never added to `tokens`
        let error = Vocab::new(tokens, &merges, None, None, None).expect_err("unresolved merge result");
        assert!(matches!(error, TokenizerError::UnresolvedMergeResult { index: 0, .. }));
    }

    #[test]
    fn tiny_vocab_resolves_its_merges() {
        let vocab = tiny_vocab();
        assert_eq!(vocab.len(), 258);
        assert_eq!(vocab.bos_token_id(), Some(1));
        assert_eq!(vocab.eos_token_id(), Some(2));
    }

    #[test]
    fn merges_driven_vocab_is_not_unigram() {
        assert!(!tiny_vocab().is_unigram());
    }

    #[test]
    fn scores_driven_vocab_is_unigram() {
        let vocab = tiny_unigram_vocab();
        assert!(vocab.is_unigram());
        let hi_id = vocab.token_id("\u{2581}hi").expect("piece exists");
        assert_eq!(vocab.token_score(hi_id), Some(-1.0));
    }

    #[test]
    fn score_array_length_mismatch_is_an_error() {
        let tokens: Vec<String> = (0..=255u8).map(hex_fallback_token).collect();
        let error = Vocab::new_unigram(tokens, vec![0.0; 10], None, None, None).expect_err("length mismatch");
        assert!(matches!(
            error,
            TokenizerError::ScoreArrayLengthMismatch { tokens_len: 256, scores_len: 10 }
        ));
    }
}
