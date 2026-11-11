//! The encoder for [`super::block_layout`]'s layouts, kept as an adapter over
//! [`proxima_numeric::encode::encode_blocks`].
//!
//! The quantizer lives in `proxima-numeric` because it is numeric data, not file parsing. A layout
//! with a monomorphic [`proxima_numeric::FormatCodec`] encoder (Q4_0, Q5_0, Q8_0, Q8_1, Q1_0, Q2_0,
//! PQ2_0, TQ2_0) is encoded by that codec, whose bytes match llama.cpp's `quantize_row_*_ref`;
//! every other layout runs the shared per-run walker, which fits a shared absolute-maximum scale and
//! writes codes back through the layout's own planes. This module keeps the [`QuantError`] surface
//! every existing call site matches on, and maps the shared error onto it.
//!
//! Teaching pointer: [`proxima_numeric::encode::reach`] is the test of whether a layout is one a
//! shared scale can express, [`proxima_numeric::encode::encode_blocks`] is the encoder, and
//! [`proxima_numeric::Numeric`] is the value-plus-format surface for converting between formats.

use proxima_numeric::NumericFormat;
use proxima_numeric::encode::{encode_blocks, reach as shared_reach};
use proxima_numeric::error::NumericError;

use crate::quant::QuantError;

const CODEC: &str = "block_layout_encode";

/// The largest value magnitude `layout` reaches on both sides of zero, in units of its scale: the
/// divisor of the block maximum. Also the test of whether [`quantize`] encodes `layout` at all.
///
/// # Errors
/// [`QuantError::UnsupportedBlockLayout`] naming what the layout has that a shared absolute-maximum
/// scale cannot express.
pub fn reach(layout: &proxima_numeric::BlockLayout) -> Result<f32, QuantError> {
    shared_reach(layout).map_err(map_error)
}

/// Encodes whole blocks of `f32` into `layout`'s packed bytes.
///
/// # Errors
/// [`QuantError::UnsupportedBlockLayout`] for a layout [`reach`] refuses;
/// [`QuantError::InputNotElementMultiple`] if `input` is not a whole number of blocks;
/// [`QuantError::OutputSizeMismatch`] if `output` is not exactly the packed byte count;
/// [`QuantError::NonFiniteInput`] for a NaN or infinite element.
pub fn quantize(
    layout: &proxima_numeric::BlockLayout,
    input: &[f32],
    output: &mut [u8],
) -> Result<(), QuantError> {
    encode_blocks(&NumericFormat::Block(*layout), input, output)
        .map(|_blocks| ())
        .map_err(map_error)
}

fn map_error(error: NumericError) -> QuantError {
    match error {
        NumericError::UnsupportedEncode { reason, .. }
        | NumericError::InvalidGeometry { reason, .. } => {
            QuantError::UnsupportedBlockLayout { reason }
        }
        NumericError::InputNotElementMultiple { found, elements, .. } => {
            QuantError::InputNotElementMultiple {
                codec: CODEC,
                unit: "block",
                found,
                block_elements: elements,
            }
        }
        NumericError::OutputSizeMismatch { found, expected } => {
            QuantError::OutputSizeMismatch { found, expected }
        }
        NumericError::NonFinite { .. } => QuantError::NonFiniteInput { codec: CODEC },
        // the shared encoder reaches these only through its scalar arm, which a `BlockLayout`
        // cannot select; naming them keeps the match exhaustive so a new variant is a build error
        NumericError::WidthUnsupported { .. }
        | NumericError::LayoutUnsupported { .. }
        | NumericError::InputNotBlockMultiple { .. }
        | NumericError::BlockSizeMismatch { .. }
        | NumericError::NoDecoder { .. }
        | NumericError::LevelOutOfTable { .. }
        | NumericError::PolicyTooStrict { .. } => QuantError::UnsupportedBlockLayout {
            reason: "the shared encoder reported an error a block layout cannot cause",
        },
    }
}
