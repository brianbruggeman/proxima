//! The policies a conversion names, and the one narrowing primitive every arm composes.
//!
//! [`crate::value::Numeric`] is the conversion surface: a value plus the format it is stored in,
//! with [`crate::value::Numeric::to`] writing it into any other format. This module holds the two
//! decisions that surface cannot infer — how a value that is not exactly representable is rounded,
//! and what a value outside the target's range becomes — plus the scalar narrowing they parameterize.
//!
//! The one thing deliberately *not* copied from `num-traits` is `as`: every narrowing here names
//! its [`Rounding`] and its [`Overflow`], because both are numeric decisions the caller owns and a
//! silently inferred one is a silently wrong answer.
//!
//! Teaching pointer: [`narrow_to_layout`] is the arithmetic; [`crate::float_layout_codec::encode_with`]
//! is what it calls; [`Rounding`] and [`Overflow`] are the two arguments a caller must have an
//! opinion about.

use crate::error::NumericError;
use crate::float_layout::FloatLayout;
use crate::format::NumericFormat;

/// How a value that is not exactly representable is rounded into a format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Rounding {
    /// Round to the nearest representable value; a tie goes to the even code.
    NearestTiesEven,
    /// Drop the bits below the target's precision, toward zero.
    TowardZero,
    /// Round away from zero on any discarded bit.
    AwayFromZero,
}

/// What a value outside the target format's finite range becomes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Overflow {
    /// Clamp to the format's own maximum finite magnitude, whatever its infinity encoding is.
    Saturate,
    /// Follow the format's own special-value policy: infinity where it has an infinity encoding,
    /// saturation where it does not. The storage policy is the format's decision, not the
    /// caller's.
    Format,
}

/// What a widened value becomes in a scalar layout, under an explicit rounding and overflow policy.
///
/// This is the scalar *query* — the f32 the format would hold — for a caller deciding whether a
/// value survives a format. Writing a whole value, including block and level-set formats, is
/// [`crate::value::Numeric::to`].
///
/// # Errors
/// [`NumericError::WidthUnsupported`] when the layout is wider than the 32-bit scalar codec, and
/// [`NumericError::NoDecoder`] when the format is not scalar.
pub fn narrow_to_format(
    value: f32,
    format: &NumericFormat,
    rounding: Rounding,
    overflow: Overflow,
) -> Result<f32, NumericError> {
    let NumericFormat::Scalar(layout) = format else {
        return Err(NumericError::NoDecoder {
            format: format.name().unwrap_or("unnamed"),
        });
    };
    narrow_to_layout(value, *layout, rounding, overflow)
}

/// [`narrow_to_format`] for a layout already in hand.
///
/// # Errors
/// [`NumericError::WidthUnsupported`] when the layout is wider than the 32-bit scalar codec.
pub fn narrow_to_layout(
    value: f32,
    layout: FloatLayout,
    rounding: Rounding,
    overflow: Overflow,
) -> Result<f32, NumericError> {
    crate::float_layout_codec::layout_supported(&layout)?;
    let mut nan_count = 0;
    let code = crate::float_layout_codec::encode_with(value, layout, rounding, overflow, &mut nan_count);
    Ok(crate::float_layout_codec::decode(code, layout))
}
