//! The merge engine: turns one pretoken's raw bytes into a sequence of
//! token ids by repeatedly collapsing the lowest-rank adjacent pair, and
//! the inverse (token ids back to bytes).

use alloc::collections::BinaryHeap;
use alloc::vec::Vec;
use core::cmp::Reverse;

use crate::error::TokenizerError;
use crate::vocab::Vocab;

/// Encodes one pretoken (a contiguous slice of raw bytes -- already split
/// out by `crate::pretokenize`) into token ids. Seeds one token per byte
/// via `Vocab::base_byte_token`, then repeatedly merges the adjacent
/// pair with the lowest rank until none remain.
///
/// # Errors
///
/// [`TokenizerError::MissingBaseByteToken`] should never surface here
/// (the vocab that built successfully already proved every byte has a
/// base token) but is threaded through in case a future vocab
/// construction path relaxes that guarantee.
pub fn encode_pretoken(bytes: &[u8], vocab: &Vocab) -> Result<Vec<u32>, TokenizerError> {
    let mut ids: Vec<u32> = bytes
        .iter()
        .map(|&byte| vocab.base_byte_token(byte))
        .collect();
    if ids.len() < 2 {
        return Ok(ids);
    }

    loop {
        let mut best: Option<(usize, u32, u32)> = None; // (position, rank, merged_id)
        for position in 0..ids.len() - 1 {
            if let Some((rank, merged_id)) = vocab.merge_rule(ids[position], ids[position + 1])
                && best.is_none_or(|(_, best_rank, _)| rank < best_rank)
            {
                best = Some((position, rank, merged_id));
            }
        }
        let Some((position, _, merged_id)) = best else {
            break;
        };
        ids[position] = merged_id;
        ids.remove(position + 1);
    }

    Ok(ids)
}

/// One live symbol in [`encode_char_pretoken`]'s doubly linked chain: a byte
/// range of the pre-token, its token id (`None` when the text is not a vocab
/// token), and its neighbours by index.
struct Symbol {
    start: usize,
    end: usize,
    token_id: Option<u32>,
    previous: Option<usize>,
    next: Option<usize>,
    alive: bool,
}

/// A candidate merge: ordered by rank (lowest fires first), then left index
/// (leftmost on ties), matching llama.cpp's `llm_bigram_bpe` comparator.
type Bigram = Reverse<(u32, usize, usize, u32)>;

fn push_bigram(
    heap: &mut BinaryHeap<Bigram>,
    symbols: &[Symbol],
    left: Option<usize>,
    right: Option<usize>,
    vocab: &Vocab,
) {
    let (Some(left), Some(right)) = (left, right) else {
        return;
    };
    let (Some(left_id), Some(right_id)) = (symbols[left].token_id, symbols[right].token_id) else {
        return;
    };
    if let Some((rank, merged_id)) = vocab.merge_rule(left_id, right_id) {
        heap.push(Reverse((rank, left, right, merged_id)));
    }
}

/// Char-level BPE over one pre-token of gemma4's `[^\n]+|[\n]+` split, with
/// spaces already spelled `▁`. Seeds one symbol per UTF-8 character, then
/// merges by rank (lowest first, leftmost on ties) using a min-heap over a
/// linked list of symbols -- `O(n log n)`, since a pre-token is a whole
/// line, not a word. A symbol that still is not a vocab token falls back to
/// its `<0xXX>` byte tokens ([`Vocab::byte_fallback_token`]). An all-newline
/// pre-token that is itself a token maps straight to it.
///
/// # Errors
///
/// [`TokenizerError::MissingBaseByteToken`] if a fallback byte has no
/// `<0xXX>` token in the vocab.
pub fn encode_char_pretoken(piece: &str, vocab: &Vocab) -> Result<Vec<u32>, TokenizerError> {
    if piece.bytes().all(|byte| byte == b'\n')
        && let Some(token_id) = vocab.token_id(piece)
    {
        return Ok(alloc::vec![token_id]);
    }

    let mut symbols: Vec<Symbol> = piece
        .char_indices()
        .map(|(start, character)| {
            let end = start + character.len_utf8();
            Symbol {
                start,
                end,
                token_id: vocab.token_id(&piece[start..end]),
                previous: None,
                next: None,
                alive: true,
            }
        })
        .collect();
    let count = symbols.len();
    for (index, symbol) in symbols.iter_mut().enumerate() {
        symbol.previous = index.checked_sub(1);
        symbol.next = (index + 1 < count).then_some(index + 1);
    }

    let mut heap = BinaryHeap::new();
    for index in 1..count {
        push_bigram(&mut heap, &symbols, Some(index - 1), Some(index), vocab);
    }
    while let Some(Reverse((rank, left, right, merged_id))) = heap.pop() {
        if !symbols[left].alive || !symbols[right].alive || symbols[left].next != Some(right) {
            continue;
        }
        let current = symbols[left]
            .token_id
            .zip(symbols[right].token_id)
            .and_then(|(left_id, right_id)| vocab.merge_rule(left_id, right_id));
        if current != Some((rank, merged_id)) {
            continue;
        }
        let right_end = symbols[right].end;
        let right_next = symbols[right].next;
        symbols[right].alive = false;
        symbols[left].end = right_end;
        symbols[left].token_id = Some(merged_id);
        symbols[left].next = right_next;
        if let Some(after) = right_next {
            symbols[after].previous = Some(left);
        }
        let before = symbols[left].previous;
        push_bigram(&mut heap, &symbols, before, Some(left), vocab);
        push_bigram(&mut heap, &symbols, Some(left), right_next, vocab);
    }

    let mut ids = Vec::new();
    for symbol in symbols.iter().filter(|symbol| symbol.alive) {
        match symbol.token_id {
            Some(token_id) => ids.push(token_id),
            None => {
                for &byte in &piece.as_bytes()[symbol.start..symbol.end] {
                    ids.push(vocab.byte_fallback_token(byte).ok_or(
                        TokenizerError::MissingBaseByteToken {
                            byte,
                            display: char::from(byte),
                        },
                    )?);
                }
            }
        }
    }
    Ok(ids)
}

/// Decodes a sequence of token ids back to raw bytes, concatenating each
/// token's byte representation in order.
///
/// # Errors
///
/// [`TokenizerError::TokenIdOutOfRange`] if any id has no entry in
/// `vocab`.
pub fn decode_ids(ids: &[u32], vocab: &Vocab) -> Result<Vec<u8>, TokenizerError> {
    let mut bytes = Vec::new();
    for &id in ids {
        let piece = vocab
            .token_bytes(id)
            .ok_or(TokenizerError::TokenIdOutOfRange {
                token_id: id,
                vocab_len: vocab.len(),
            })?;
        bytes.extend_from_slice(piece);
    }
    Ok(bytes)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::vocab::tests::tiny_vocab;

    #[test]
    fn encode_pretoken_merges_h_and_i_into_hi() {
        let vocab = tiny_vocab();
        let ids = encode_pretoken(b"hi", &vocab).expect("encodes");
        let hi_id = vocab.token_id("hi").expect("hi token exists");
        assert_eq!(ids, [hi_id]);
    }

    #[test]
    fn encode_pretoken_prefers_the_lowest_rank_merge_first() {
        let vocab = tiny_vocab();
        let ids = encode_pretoken(" hi".as_bytes(), &vocab).expect("encodes");
        let space_hi_id = vocab.token_id("\u{0120}hi").expect("space-hi token exists");
        assert_eq!(ids, [space_hi_id]);
    }

    #[test]
    fn encode_pretoken_leaves_unmergeable_bytes_as_base_tokens() {
        let vocab = tiny_vocab();
        let ids = encode_pretoken(b"xz", &vocab).expect("encodes");
        assert_eq!(ids.len(), 2, "no merge rule for x-z, stays two base tokens");
    }

    #[test]
    fn encode_then_decode_round_trips() {
        let vocab = tiny_vocab();
        let ids = encode_pretoken(" hi".as_bytes(), &vocab).expect("encodes");
        let bytes = decode_ids(&ids, &vocab).expect("decodes");
        assert_eq!(bytes, b" hi");
    }

    #[test]
    fn decode_out_of_range_id_is_an_error() {
        let vocab = tiny_vocab();
        let error = decode_ids(&[u32::MAX], &vocab).expect_err("out of range");
        assert!(matches!(error, TokenizerError::TokenIdOutOfRange { .. }));
    }

    #[test]
    fn encode_pretoken_handles_empty_input() {
        let vocab = tiny_vocab();
        let ids = encode_pretoken(b"", &vocab).expect("encodes");
        assert!(ids.is_empty());
    }

    fn char_level_vocab() -> Vocab {
        let mut tokens: Vec<alloc::string::String> = (0..=255u8)
            .map(|byte| alloc::format!("<0x{byte:02X}>"))
            .collect();
        for piece in [
            "\u{2581}", "h", "i", "hi", "\u{2581}hi", "\u{e9}", "\n", "\n\n",
        ] {
            tokens.push(alloc::string::String::from(piece));
        }
        let merges = alloc::vec![
            alloc::string::String::from("h i"),
            alloc::string::String::from("\u{2581} hi"),
        ];
        Vocab::new(tokens, &merges, None, None, None).expect("char-level vocab builds")
    }

    #[test]
    fn char_level_probe_fires_on_merges_marker_and_hex_newline() {
        assert!(char_level_vocab().is_char_level_bpe());
        assert!(!tiny_vocab().is_char_level_bpe());
    }

    #[test]
    fn char_level_merges_chain_across_the_space_marker() {
        let vocab = char_level_vocab();
        let ids = encode_char_pretoken("\u{2581}hi", &vocab).expect("encodes");
        assert_eq!(ids, [vocab.token_id("\u{2581}hi").expect("token")]);
    }

    #[test]
    fn char_level_all_newline_run_maps_to_its_whole_token() {
        let vocab = char_level_vocab();
        let ids = encode_char_pretoken("\n\n", &vocab).expect("encodes");
        assert_eq!(ids, [vocab.token_id("\n\n").expect("token")]);
    }

    #[test]
    fn char_level_unknown_character_falls_back_to_hex_byte_tokens() {
        let vocab = char_level_vocab();
        let ids = encode_char_pretoken("\u{4eca}", &vocab).expect("encodes");
        let expected: Vec<u32> = [0xE4u8, 0xBB, 0x8A]
            .iter()
            .map(|byte| vocab.token_id(&alloc::format!("<0x{byte:02X}>")).expect("hex"))
            .collect();
        assert_eq!(ids, expected);
    }

    #[test]
    fn char_level_literal_latin1_token_decodes_as_utf8_not_the_gpt2_inverse_remap() {
        let vocab = char_level_vocab();
        let id = vocab.token_id("\u{e9}").expect("token");
        assert_eq!(decode_ids(&[id], &vocab).expect("decodes"), "\u{e9}".as_bytes());
    }

    #[test]
    fn gpt2_byte_level_path_is_unchanged_by_the_char_level_probe() {
        let vocab = tiny_vocab();
        let ids = encode_pretoken(b"hi", &vocab).expect("encodes");
        assert_eq!(ids, [vocab.token_id("hi").expect("hi")]);
        assert_eq!(crate::encode("\n", &vocab).expect("encodes").len(), 1);
    }
}
