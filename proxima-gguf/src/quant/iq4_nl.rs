//! `IQ4_NL`: 4-bit non-linear (table-indexed) weights in 32-element blocks,
//! one `f16` scale per block, no sub-block structure, no minimum term.
//! `value = scale * kvalues_iq4nl[nibble]` per element
//! (`dequantize_row_iq4_nl`, `ggml-quants.c:2428-2444`, cited on
//! [`dequantize_block`]).
//!
//! Layout, from llama.cpp (`/Users/brianbruggeman/repos/others/llama.cpp`):
//! `ggml/src/ggml-common.h:405-410`:
//! ```c
//! #define QK4_NL 32
//! typedef struct {
//!     ggml_half d;
//!     uint8_t qs[QK4_NL/2];
//! } block_iq4_nl;
//! static_assert(sizeof(block_iq4_nl) == sizeof(ggml_half) + QK4_NL/2, "wrong iq4_nl block size/padding");
//! ```
//! 18 bytes per 32 elements -- byte-identical layout to `Q4_0`
//! ([`crate::quant::q4_0`]), the difference is entirely in how a nibble maps
//! to a value: `Q4_0` recenters the raw nibble by a fixed midpoint (`nibble
//! - 8`), `IQ4_NL` looks the nibble up in a 16-entry non-linear codebook
//! (`kvalues_iq4nl`, `ggml-common.h:1077-1079`) trained to fit weight
//! distributions better than a uniform 4-bit ladder. Cross-checked here at
//! compile time against [`crate::types::GgmlType::Iq4Nl`]'s already-landed
//! [`crate::types::GgmlType::block_layout`] rather than re-typed by hand.

use crate::quant::QuantError;
use crate::types::GgmlType;

/// This codec's name as it appears in a rendered [`QuantError`] message.
const CODEC: &str = "iq4_nl";

/// Elements per block (`ggml-common.h:405`, `#define QK4_NL 32`).
pub const QK4_NL: usize = 32;

/// Bytes per block. Derived from the already-landed
/// [`crate::types::GgmlType::block_layout`] (`types.rs`), not re-typed --
/// that table was itself checked against `ggml-common.h`'s
/// `static_assert(sizeof(block_iq4_nl) == sizeof(ggml_half) + QK4_NL/2,
/// ...)` at `ggml-common.h:410`.
pub const BLOCK_BYTES: usize = {
    let layout = GgmlType::Iq4Nl.block_layout();
    assert!(
        layout.block_elements as usize == QK4_NL,
        "GgmlType::Iq4Nl block_elements drifted from QK4_NL"
    );
    layout.block_bytes as usize
};

const D_OFFSET: usize = 0;
const QS_OFFSET: usize = 2;
/// Half the block's elements -- each packed byte carries two nibbles, one
/// per half of the block (`dequantize_row_iq4_nl`, `ggml-quants.c:2437-2440`).
const HALF_BLOCK: usize = QK4_NL / 2;

/// The non-linear 4-bit codebook every nibble indexes into
/// (`ggml-common.h:1077-1079`, `GGML_TABLE_BEGIN(int8_t, kvalues_iq4nl,
/// 16)`), copied verbatim -- not derived, not approximated. Trained offline
/// by llama.cpp to fit typical weight distributions better than a uniform
/// `nibble - 8` ladder; this crate only decodes, never retrains it.
pub const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

/// Number of whole `IQ4_NL` blocks a byte run decodes to, or `None` if
/// `byte_len` is not an exact multiple of [`BLOCK_BYTES`].
#[must_use]
pub const fn blocks_for_bytes(byte_len: usize) -> Option<usize> {
    if byte_len.is_multiple_of(BLOCK_BYTES) {
        Some(byte_len / BLOCK_BYTES)
    } else {
        None
    }
}

/// Exact packed byte length for `block_count` `IQ4_NL` blocks.
#[must_use]
pub const fn bytes_for_blocks(block_count: usize) -> usize {
    block_count * BLOCK_BYTES
}

/// Exact `f32` element count for `block_count` `IQ4_NL` blocks.
#[must_use]
pub const fn elements_for_blocks(block_count: usize) -> usize {
    block_count * QK4_NL
}

fn f16_at(block: &[u8], offset: usize) -> half::f16 {
    let mut bytes = [0u8; 2];
    bytes.copy_from_slice(&block[offset..offset + 2]);
    half::f16::from_le_bytes(bytes)
}

/// Dequantizes one 32-element `IQ4_NL` block. `block` must be exactly
/// [`BLOCK_BYTES`] bytes and `output` exactly [`QK4_NL`] elements -- callers
/// go through [`dequantize`], which validates both.
///
/// Ports `dequantize_row_iq4_nl` (`ggml-quants.c:2428-2444`) exactly: each
/// packed byte carries two 4-bit codebook indices, low nibble at
/// `output[j]`, high nibble at `output[HALF_BLOCK + j]`, each looked up in
/// [`KVALUES_IQ4NL`] then scaled by the block's single `f16` delta -- no
/// recenter, the table itself is signed.
pub fn dequantize_block(block: &[u8], output: &mut [f32]) {
    let delta = f16_at(block, D_OFFSET).to_f32();
    let qs = &block[QS_OFFSET..QS_OFFSET + HALF_BLOCK];
    for (index, &byte) in qs.iter().enumerate() {
        let low = KVALUES_IQ4NL[usize::from(byte & 0x0F)];
        let high = KVALUES_IQ4NL[usize::from(byte >> 4)];
        output[index] = f32::from(low) * delta;
        output[HALF_BLOCK + index] = f32::from(high) * delta;
    }
}

/// Dequantizes a run of `IQ4_NL` blocks. `data` is borrowed, `output` is
/// caller-provided -- no allocation on this path.
///
/// # Errors
/// [`QuantError::InputNotBlockMultiple`] if `data.len()` is not a multiple
/// of [`BLOCK_BYTES`]; [`QuantError::OutputSizeMismatch`] if `output.len()`
/// does not exactly match the decoded element count.
pub fn dequantize(data: &[u8], output: &mut [f32]) -> Result<(), QuantError> {
    let block_count = blocks_for_bytes(data.len()).ok_or(QuantError::InputNotBlockMultiple {
        codec: CODEC,
        found: data.len(),
        block_bytes: BLOCK_BYTES,
    })?;
    let expected = elements_for_blocks(block_count);
    if output.len() != expected {
        return Err(QuantError::OutputSizeMismatch {
            found: output.len(),
            expected,
        });
    }
    for (block, out_chunk) in data
        .as_chunks::<BLOCK_BYTES>()
        .0
        .iter()
        .zip(output.as_chunks_mut::<QK4_NL>().0)
    {
        dequantize_block(block, out_chunk);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use alloc::vec;
    use alloc::vec::Vec;

    use super::{
        BLOCK_BYTES, CODEC, HALF_BLOCK, KVALUES_IQ4NL, QK4_NL, QS_OFFSET, QuantError, dequantize,
    };

    /// One block, hand-packed and hand-decoded, checked against the
    /// `x = kvalues_iq4nl[nibble] * d` formula computed by hand -- not by
    /// calling a quantize routine (this codec is decode-only). Every one of
    /// the 16 table entries is exercised, split across the low and high
    /// nibble halves so both packing paths are covered. `d=0.0123` is a
    /// realistic real-checkpoint-scale delta.
    #[test]
    fn dequantize_block_matches_hand_computed_fixture() {
        let low_nibbles: [u8; HALF_BLOCK] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let high_nibbles: [u8; HALF_BLOCK] = [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0];
        let delta = 0.0123f32;

        let mut block = [0u8; BLOCK_BYTES];
        block[0..2].copy_from_slice(&half::f16::from_f32(delta).to_le_bytes());
        for index in 0..HALF_BLOCK {
            block[QS_OFFSET + index] = low_nibbles[index] | (high_nibbles[index] << 4);
        }

        let exact_delta = half::f16::from_f32(delta).to_f32();
        let expected: Vec<f32> = low_nibbles
            .iter()
            .chain(high_nibbles.iter())
            .map(|&nibble| f32::from(KVALUES_IQ4NL[usize::from(nibble)]) * exact_delta)
            .collect();

        let mut output = [0.0f32; QK4_NL];
        dequantize(&block, &mut output).expect("well-formed single block");
        assert_eq!(output.as_slice(), expected.as_slice());
    }

    /// Boundary nibbles: `0` indexes the table's most negative entry
    /// (`-127`), `15` indexes its most positive (`113`) -- the two extremes
    /// of the codebook, not the extremes of the raw 4-bit range.
    #[test]
    fn dequantize_block_boundary_min_and_max_nibble() {
        let mut block = [0u8; BLOCK_BYTES];
        let delta = 0.5f32;
        block[0..2].copy_from_slice(&half::f16::from_f32(delta).to_le_bytes());
        block[QS_OFFSET] = 0xF0; // low nibble 0, high nibble 15.

        let mut output = [0.0f32; QK4_NL];
        dequantize(&block, &mut output).expect("well-formed single block");
        assert_eq!(output[0], f32::from(KVALUES_IQ4NL[0]) * 0.5);
        assert_eq!(output[HALF_BLOCK], f32::from(KVALUES_IQ4NL[15]) * 0.5);
    }

    /// A real 160-element embedding row -- 5 whole `IQ4_NL` blocks, exactly
    /// the row width the target checkpoint's `token_embd` ngram table
    /// stores (`dims: [160, 320001536]`, 90 bytes per row = 5 blocks * 18
    /// bytes). Proves the multi-block path composes the per-block formula
    /// correctly across a boundary a single-block test cannot exercise.
    #[test]
    fn dequantize_five_block_embedding_row_matches_hand_computed_fixture() {
        let blocks = 5;
        let elements = QK4_NL * blocks;
        let mut packed = vec![0u8; BLOCK_BYTES * blocks];
        let mut expected = vec![0.0f32; elements];

        for block_index in 0..blocks {
            let delta = 0.0123f32 * (block_index as f32 + 1.0);
            let exact_delta = half::f16::from_f32(delta).to_f32();
            let block = &mut packed[block_index * BLOCK_BYTES..(block_index + 1) * BLOCK_BYTES];
            block[0..2].copy_from_slice(&half::f16::from_f32(delta).to_le_bytes());
            for index in 0..HALF_BLOCK {
                let low_nibble = (index % 16) as u8;
                let high_nibble = ((index + 8) % 16) as u8;
                block[QS_OFFSET + index] = low_nibble | (high_nibble << 4);
                let element_base = block_index * QK4_NL;
                expected[element_base + index] =
                    f32::from(KVALUES_IQ4NL[usize::from(low_nibble)]) * exact_delta;
                expected[element_base + HALF_BLOCK + index] =
                    f32::from(KVALUES_IQ4NL[usize::from(high_nibble)]) * exact_delta;
            }
        }

        let mut output = vec![0.0f32; elements];
        dequantize(&packed, &mut output).expect("five well-formed blocks");
        assert_eq!(output, expected);
    }

    #[test]
    fn dequantize_rejects_non_block_multiple_length() {
        let data = vec![0u8; BLOCK_BYTES - 1];
        let mut output = vec![0.0f32; QK4_NL];
        let error = dequantize(&data, &mut output).unwrap_err();
        assert_eq!(
            error,
            QuantError::InputNotBlockMultiple {
                codec: CODEC,
                found: BLOCK_BYTES - 1,
                block_bytes: BLOCK_BYTES,
            }
        );
    }

    #[test]
    fn dequantize_rejects_output_size_mismatch() {
        let data = vec![0u8; BLOCK_BYTES];
        let mut output = vec![0.0f32; QK4_NL - 1];
        let error = dequantize(&data, &mut output).unwrap_err();
        assert_eq!(
            error,
            QuantError::OutputSizeMismatch {
                found: QK4_NL - 1,
                expected: QK4_NL,
            }
        );
    }

    /// A truncated block run that is neither a whole block nor empty --
    /// typed error, never a panic or an out-of-bounds read.
    #[test]
    fn dequantize_rejects_truncated_partial_block() {
        let partial_bytes = BLOCK_BYTES + BLOCK_BYTES / 2;
        let data = vec![0u8; partial_bytes];
        let mut output = vec![0.0f32; QK4_NL];
        let error = dequantize(&data, &mut output).unwrap_err();
        assert_eq!(
            error,
            QuantError::InputNotBlockMultiple {
                codec: CODEC,
                found: partial_bytes,
                block_bytes: BLOCK_BYTES,
            }
        );
    }
}
