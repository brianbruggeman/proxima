//! [`encode`]/[`decode`]: "I already have the whole input as one contiguous
//! value, transform all of it." Both directions are stateless given a
//! vocab.

use alloc::string::String;
use alloc::vec::Vec;

use crate::bpe::{decode_ids, encode_pretoken};
use crate::error::TokenizerError;
use crate::pretokenize::pretokenize;
use crate::unigram;
use crate::vocab::Vocab;

/// Scans `text` for literal occurrences of an "added token" marker
/// (`Vocab::with_token_types`'s `Control`/`UserDefined` entries -- e.g.
/// `<|end_of_turn|>` in a chat template) via
/// [`Vocab::longest_added_token_match`], emitting each marker's id
/// directly and running [`encode_ordinary`] only on the plain-text spans
/// between markers. Longest-match wins when two markers share a prefix
/// (the trie walk in [`Vocab::longest_added_token_match`] always returns
/// the deepest/longest node with an id, never the first). A vocab that
/// never called `with_token_types` has an empty trie, so this degenerates
/// to exactly one call to [`encode_ordinary`] over the whole input --
/// identical to this function's behavior before markers existed.
///
/// Splits `text` into pretokens ([`crate::pretokenize::pretokenize`]) and
/// BPE-merges each independently, concatenating the resulting ids in order
/// -- for a merges-driven vocab. For a scores-driven ([`Vocab::is_unigram`])
/// vocab, normalizes the whole input ([`unigram::escape`]) and segments it
/// as one fragment ([`unigram::encode_fragment`]) instead: SentencePiece has
/// no separate pretokenizer stage, see [`crate::unigram`]'s module doc.
/// Never adds BOS/EOS -- see [`encode_with_bos_eos`] for that, explicitly.
///
/// # Errors
///
/// Any [`TokenizerError`] [`encode_pretoken`]/[`unigram::encode_fragment`]
/// surfaces.
pub fn encode(text: &str, vocab: &Vocab) -> Result<Vec<u32>, TokenizerError> {
    let mut ids = Vec::new();
    let bytes = text.as_bytes();
    let mut ordinary_start = 0usize;
    let mut position = 0usize;
    while position < bytes.len() {
        match vocab.longest_added_token_match(&bytes[position..]) {
            Some((token_id, matched_len)) => {
                if ordinary_start < position {
                    ids.extend(encode_ordinary(&text[ordinary_start..position], vocab)?);
                }
                ids.push(token_id);
                position += matched_len;
                ordinary_start = position;
            }
            None => {
                let char_len = text[position..].chars().next().map_or(1, char::len_utf8);
                position += char_len;
            }
        }
    }
    if ordinary_start < bytes.len() {
        ids.extend(encode_ordinary(&text[ordinary_start..], vocab)?);
    }
    Ok(ids)
}

/// [`encode`]'s per-span encoder, run on the plain text between added-token
/// markers (or the whole input, when there are none).
fn encode_ordinary(text: &str, vocab: &Vocab) -> Result<Vec<u32>, TokenizerError> {
    if vocab.is_unigram() {
        let normalized = unigram::escape(text);
        return unigram::encode_fragment(&normalized, vocab);
    }
    let mut ids = Vec::new();
    for span in pretokenize(text) {
        let piece = &text[span];
        ids.extend(encode_pretoken(piece.as_bytes(), vocab)?);
    }
    Ok(ids)
}

/// [`encode`], additionally prepending/appending the vocab's BOS/EOS ids
/// when present and requested. Explicit opt-in on both ends: special
/// tokens are never silently added or dropped.
///
/// # Errors
///
/// [`TokenizerError::MissingMetadataKey`] if `add_bos`/`add_eos` is
/// requested but the vocab has no such token id; any error [`encode`]
/// surfaces otherwise.
pub fn encode_with_bos_eos(
    text: &str,
    vocab: &Vocab,
    add_bos: bool,
    add_eos: bool,
) -> Result<Vec<u32>, TokenizerError> {
    let mut ids = Vec::new();
    if add_bos {
        let bos = vocab
            .bos_token_id()
            .ok_or(TokenizerError::MissingMetadataKey { key: "tokenizer.ggml.bos_token_id" })?;
        ids.push(bos);
    }
    ids.extend(encode(text, vocab)?);
    if add_eos {
        let eos = vocab
            .eos_token_id()
            .ok_or(TokenizerError::MissingMetadataKey { key: "tokenizer.ggml.eos_token_id" })?;
        ids.push(eos);
    }
    Ok(ids)
}

/// Concatenates every token id's raw bytes ([`decode_ids`]) and interprets
/// the result as UTF-8.
///
/// # Errors
///
/// [`TokenizerError::TokenIdOutOfRange`] for an id absent from `vocab`;
/// [`TokenizerError::InvalidUtf8`] if the concatenated bytes are not
/// valid UTF-8 (possible when `ids` did not come from this crate's own
/// [`encode`] -- an arbitrary id sequence is not guaranteed to land on
/// UTF-8 boundaries).
pub fn decode(ids: &[u32], vocab: &Vocab) -> Result<String, TokenizerError> {
    let bytes = decode_ids(ids, vocab)?;
    let text = String::from_utf8(bytes).map_err(|_| TokenizerError::InvalidUtf8)?;
    if vocab.is_unigram() {
        return Ok(unigram::unescape(&text));
    }
    Ok(text)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::vocab::tests::{tiny_unigram_vocab, tiny_vocab};

    #[test]
    fn encode_with_bos_eos_prepends_and_appends() {
        let vocab = tiny_vocab();
        let ids = encode_with_bos_eos(" hi", &vocab, true, true).expect("encodes with bos/eos");
        assert_eq!(ids.first().copied(), vocab.bos_token_id());
        assert_eq!(ids.last().copied(), vocab.eos_token_id());
        assert_eq!(ids.len(), encode(" hi", &vocab).expect("plain encode").len() + 2);
    }

    #[test]
    fn round_trip_arbitrary_ascii_and_multibyte_utf8() {
        let vocab = tiny_vocab();
        for text in [" hi", "hi hi", "xyz", "\u{1F600} hi", ""] {
            let ids = encode(text, &vocab).expect("encodes");
            let decoded = decode(&ids, &vocab).expect("decodes");
            assert_eq!(decoded, text, "round trip failed for {text:?}");
        }
    }

    #[test]
    fn unigram_round_trip_arbitrary_ascii_and_multibyte_utf8() {
        let vocab = tiny_unigram_vocab();
        for text in [" hi", "hi hi", "xyz", "\u{1F600} hi", ""] {
            let ids = encode(text, &vocab).expect("encodes");
            let decoded = decode(&ids, &vocab).expect("decodes");
            assert_eq!(decoded, text, "round trip failed for {text:?} (ids: {ids:?})");
        }
    }

    #[test]
    fn unigram_vocab_dispatches_to_the_unigram_encoder_not_bpe() {
        // degenerate control: a merges-driven vocab given the same input
        // must NOT collapse "hi" the same way a scores-driven vocab does --
        // proves `encode` actually branches on `Vocab::is_unigram` rather
        // than always running one encoder.
        let unigram_vocab = tiny_unigram_vocab();
        let bpe_vocab = tiny_vocab();
        let unigram_ids = encode("hi", &unigram_vocab).expect("unigram encodes");
        let bpe_ids = encode("hi", &bpe_vocab).expect("bpe encodes");
        assert_eq!(unigram_ids.len(), 1, "unigram vocab merges \u{2581}hi to one piece");
        assert_eq!(bpe_ids.len(), 1, "bpe vocab merges h+i to one piece");
        assert_ne!(unigram_ids, bpe_ids, "the two encoders assign different ids for the same text");
        assert_eq!(decode(&unigram_ids, &unigram_vocab).expect("unigram decodes"), "hi");
        assert_eq!(decode(&bpe_ids, &bpe_vocab).expect("bpe decodes"), "hi");
    }
}
