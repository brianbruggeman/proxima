//! The merge engine: turns one pretoken's raw bytes into a sequence of
//! token ids by repeatedly collapsing the lowest-rank adjacent pair, and
//! the inverse (token ids back to bytes).

use alloc::vec::Vec;

use crate::error::TokenizerError;
use crate::vocab::Vocab;

/// Encodes one pretoken (a contiguous slice of raw bytes -- already split
/// out by `crate::pretokenize`) into token ids. Seeds one token per byte
/// via [`Vocab::base_byte_token`], then repeatedly merges the adjacent
/// pair with the lowest rank until none remain.
///
/// # Errors
///
/// [`TokenizerError::MissingBaseByteToken`] should never surface here
/// (the vocab that built successfully already proved every byte has a
/// base token) but is threaded through in case a future vocab
/// construction path relaxes that guarantee.
pub fn encode_pretoken(bytes: &[u8], vocab: &Vocab) -> Result<Vec<u32>, TokenizerError> {
    let mut ids: Vec<u32> = bytes.iter().map(|&byte| vocab.base_byte_token(byte)).collect();
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
            .ok_or(TokenizerError::TokenIdOutOfRange { token_id: id, vocab_len: vocab.len() })?;
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
}
