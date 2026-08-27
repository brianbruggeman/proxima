//! SentencePiece-family segmentation for `tokenizer.ggml.model = "llama"`
//! vocabs. Despite the "unigram" name this crate's module doc-comments used
//! before this was read against source, the encoder here is NOT the
//! literature unigram-Viterbi lattice: `tokenizer.ggml.model == "llama"`
//! dispatches to `LLAMA_VOCAB_TYPE_SPM` in llama.cpp
//! (`llama-vocab.cpp:1405-1406`), whose tokenizer
//! (`llm_tokenizer_spm_session::tokenize`, `llama-vocab.cpp:116-172`)
//! greedily merges the highest-scoring adjacent symbol pair until none
//! resolve -- the same shape as [`crate::bpe::encode_pretoken`]'s rescan
//! loop, just keyed by a dynamic vocab lookup of the merged text (via
//! [`crate::vocab::Vocab::token_id`]/`Vocab::token_score`, the latter
//! crate-private) instead of a precomputed merge-rank table. Literature
//! unigram-Viterbi is
//! `LLAMA_VOCAB_TYPE_UGM` (`tokenizer.ggml.model == "t5"`), a different
//! checkpoint family this crate does not target. Per this workspace's
//! incumbent-wins-on-correctness rule, the incumbent's actual dispatch is
//! what this module matches.

use alloc::string::String;
use alloc::vec::Vec;
use core::ops::Range;

use crate::error::TokenizerError;
use crate::vocab::Vocab;

/// SentencePiece's escape marker for a literal space (`▁`, U+2581).
/// `tokenizer.ggml.model = "llama"` vocabs store every space-containing
/// piece with spaces already substituted for this codepoint
/// (`llama_escape_whitespace`, `llama-vocab.cpp:2372-2374`).
const SPACE_MARKER: char = '\u{2581}';

/// Normalizes raw text the way llama.cpp's SPM path does before segmenting:
/// prepends one literal space (`add_space_prefix`, the default for
/// `LLAMA_VOCAB_TYPE_SPM`, `llama-vocab.cpp:1664`, unless overridden by
/// `tokenizer.ggml.add_space_prefix` -- not read by this crate, matching the
/// most common checkpoint shape) then substitutes every space for
/// `SPACE_MARKER` (crate-private). `""` stays `""`: llama.cpp only ever
/// tokenizes a non-empty fragment (`llama-vocab.cpp:2409`).
#[must_use]
pub fn escape(text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }
    let mut escaped = String::with_capacity(text.len() + SPACE_MARKER.len_utf8());
    escaped.push(SPACE_MARKER);
    for character in text.chars() {
        escaped.push(if character == ' ' { SPACE_MARKER } else { character });
    }
    escaped
}

/// Reverses [`escape`]: every `SPACE_MARKER` (crate-private) back to a
/// literal space, then strips exactly the one leading space [`escape`]
/// added (matching `remove_space`, `llama-vocab.cpp:2684-2685`).
#[must_use]
pub fn unescape(text: &str) -> String {
    let mut unescaped = String::with_capacity(text.len());
    for character in text.chars() {
        unescaped.push(if character == SPACE_MARKER { ' ' } else { character });
    }
    if unescaped.starts_with(' ') {
        unescaped.remove(0);
    }
    unescaped
}

/// Encodes one already-[`escape`]d fragment into token ids. Unlike
/// [`crate::bpe::encode_pretoken`], SentencePiece's SPM tokenizer has no
/// separate pretokenizer stage -- the whole input is one symbol chain, not
/// split into pretokens first (`llama-vocab.cpp:2430-2446` calls
/// `llm_tokenizer_spm_session::tokenize` once per whole-text fragment).
///
/// Seeds one symbol per `char` (SentencePiece operates on codepoints, not
/// raw bytes), then repeatedly merges the adjacent pair whose concatenated
/// text resolves to a vocab token with the highest `token_score`
/// (crate-private) (ties keep the leftmost position, matching the
/// priority-queue comparator at `llama-vocab.cpp:96-100`) until no pair
/// resolves. Any symbol that still doesn't name a vocab token falls back to
/// one base byte token per raw byte (`Vocab::base_byte_token`,
/// crate-private) -- the `<0xXX>` alphabet every byte-fallback SentencePiece
/// vocab carries.
///
/// # Errors
///
/// Never in practice: [`Vocab::new_unigram`]'s construction check
/// guarantees every byte has a fallback token, so byte resolution here
/// cannot fail. Returns `Result` to match this crate's other encode
/// functions and leave room for a future non-byte-fallback vocab shape.
pub fn encode_fragment(text: &str, vocab: &Vocab) -> Result<Vec<u32>, TokenizerError> {
    let mut symbols: Vec<Range<usize>> =
        text.char_indices().map(|(offset, character)| offset..offset + character.len_utf8()).collect();

    loop {
        let mut best: Option<(usize, f32, u32)> = None; // (position, score, merged_id)
        for position in 0..symbols.len().saturating_sub(1) {
            let span = symbols[position].start..symbols[position + 1].end;
            let Some(token_id) = vocab.token_id(&text[span]) else { continue };
            let Some(score) = vocab.token_score(token_id) else { continue };
            if best.is_none_or(|(_, best_score, _)| score > best_score) {
                best = Some((position, score, token_id));
            }
        }
        let Some((position, _, _)) = best else { break };
        symbols[position] = symbols[position].start..symbols[position + 1].end;
        symbols.remove(position + 1);
    }

    let mut ids = Vec::new();
    for span in symbols {
        let piece = &text[span];
        if let Some(token_id) = vocab.token_id(piece) {
            ids.push(token_id);
        } else {
            ids.extend(piece.bytes().map(|byte| vocab.base_byte_token(byte)));
        }
    }
    Ok(ids)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::vocab::tests::tiny_unigram_vocab;

    #[test]
    fn escape_prepends_and_substitutes_spaces() {
        assert_eq!(escape("hi"), "\u{2581}hi");
        assert_eq!(escape(" hi there"), "\u{2581}\u{2581}hi\u{2581}there");
    }

    #[test]
    fn escape_of_empty_text_stays_empty() {
        assert_eq!(escape(""), "");
    }

    #[test]
    fn unescape_reverses_escape_for_a_plain_word() {
        assert_eq!(unescape(&escape("hi")), "hi");
    }

    #[test]
    fn encode_fragment_chains_merges_to_the_highest_scoring_piece() {
        let vocab = tiny_unigram_vocab();
        let normalized = escape("hi");
        let ids = encode_fragment(&normalized, &vocab).expect("encodes");
        let hi_id = vocab.token_id("\u{2581}hi").expect("merged piece exists");
        assert_eq!(ids, [hi_id], "should chain through the highest-score bigram at every step");
    }

    #[test]
    fn encode_fragment_falls_back_to_bytes_for_unmatched_symbols() {
        let vocab = tiny_unigram_vocab();
        let normalized = escape("xz"); // no piece for "x", "z", or "xz" in the tiny vocab
        let ids = encode_fragment(&normalized, &vocab).expect("encodes");
        // "▁" resolves as its own piece, "x" and "z" fall back to base bytes.
        assert_eq!(ids.len(), 3, "▁ as one piece, x and z as individual byte fallbacks");
    }

    #[test]
    fn encode_fragment_of_empty_text_is_empty() {
        let vocab = tiny_unigram_vocab();
        assert!(encode_fragment("", &vocab).expect("encodes").is_empty());
    }
}
