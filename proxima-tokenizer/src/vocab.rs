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
        let token_to_id: BTreeMap<String, u32> = tokens
            .iter()
            .enumerate()
            .map(|(id, token)| (token.clone(), id as u32))
            .collect();

        let id_to_bytes: Vec<Vec<u8>> = tokens.iter().map(|token| decode_display_string(token)).collect();

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
}
