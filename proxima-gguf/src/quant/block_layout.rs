//! The GGUF adapter over the one block walker, [`proxima_numeric::decode`].
//!
//! Teaching pointer: `proxima_numeric::BlockLayout` is the data and
//! [`proxima_numeric::decode_blocks`] is the walker; this module only maps a [`QuantError`] onto
//! the numeric crate's typed failures so a caller of `proxima_gguf::quant` keeps its error shape.
//! The per-codec decoders (`q4_0.rs`, `q5_1.rs`, `q4_k.rs`, ...) are the incumbents the shared
//! walker is checked against bit for bit in `tests/numeric_parity.rs`.

use proxima_numeric::BlockLayout;
use proxima_numeric::NumericError;
use proxima_numeric::NumericFormat;
use proxima_numeric::decode_blocks;

use crate::quant::QuantError;

const CODEC: &str = "block_layout";

/// One block of `layout` decoded into `output`.
///
/// # Errors
/// [`QuantError::InputNotBlockMultiple`] when `block` is not the layout's own byte size and
/// [`QuantError::OutputSizeMismatch`] when `output` is not its element count.
pub fn dequantize_block(
    layout: &BlockLayout,
    block: &[u8],
    output: &mut [f32],
) -> Result<(), QuantError> {
    dequantize(layout, block, output)
}

/// A run of blocks decoded into caller-owned storage.
///
/// # Errors
/// [`QuantError::InputNotBlockMultiple`] when `data` is not a whole number of blocks and
/// [`QuantError::OutputSizeMismatch`] when `output` is not exactly the decoded element count.
pub fn dequantize(layout: &BlockLayout, data: &[u8], output: &mut [f32]) -> Result<(), QuantError> {
    decode_blocks(&NumericFormat::Block(*layout), data, output).map(|_| ()).map_err(|error| {
        match error {
            NumericError::InputNotBlockMultiple {
                found, block_bytes, ..
            } => QuantError::InputNotBlockMultiple {
                codec: CODEC,
                found,
                block_bytes,
            },
            NumericError::OutputSizeMismatch { found, expected } => {
                QuantError::OutputSizeMismatch { found, expected }
            }
            other => QuantError::UnsupportedBlockLayout {
                reason: numeric_reason(&other),
            },
        }
    })
}

fn numeric_reason(error: &NumericError) -> &'static str {
    match error {
        NumericError::WidthUnsupported { .. } => "a layout field is wider than the 32-bit codec",
        NumericError::LayoutUnsupported { .. } => "a layout outside the scalar codec's domain",
        NumericError::InputNotBlockMultiple { .. } => "input is not a whole number of blocks",
        NumericError::OutputSizeMismatch { .. } => "output size does not match the element count",
        NumericError::BlockSizeMismatch { .. } => "a block is not the format's own size",
        NumericError::NoDecoder { .. } => "no decoder for this format",
        NumericError::LevelOutOfTable { .. } => "a code outside the level table",
        NumericError::InvalidGeometry { .. } => "a layout that cannot address its own block",
        NumericError::PolicyTooStrict { .. } => "numeric policy too strict",
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use crate::quant::q4_0;

    #[test]
    fn the_shared_walker_reproduces_the_per_codec_q4_0_decoder() {
        let mut block = [0_u8; q4_0::BLOCK_BYTES];
        for (index, byte) in block.iter_mut().enumerate() {
            *byte = (index as u8).wrapping_mul(37).wrapping_add(11);
        }
        let mut decoded = [0.0_f32; q4_0::QK4_0];
        dequantize_block(&BlockLayout::Q4_0, &block, &mut decoded).expect("one q4_0 block decodes");
        let mut expected = [0.0_f32; q4_0::QK4_0];
        q4_0::dequantize_block(&block, &mut expected);
        assert_eq!(decoded.map(f32::to_bits), expected.map(f32::to_bits));
    }

    #[test]
    fn a_mis_sized_payload_is_the_quant_error_callers_already_match_on() {
        let error = dequantize(&BlockLayout::Q4_0, &[0_u8; 17], &mut [0.0; 32])
            .expect_err("17 bytes is not a whole q4_0 block");
        assert_eq!(
            error,
            QuantError::InputNotBlockMultiple {
                codec: CODEC,
                found: 17,
                block_bytes: 18,
            }
        );
    }
}
