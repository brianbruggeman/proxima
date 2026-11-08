//! Conversion between a value and a format, modelled on `num_traits::{NumCast, ToPrimitive,
//! FromPrimitive}` but over the layout rather than a Rust primitive, and `no_std` first.
//!
//! The one thing deliberately *not* copied from `num-traits` is `as`: every narrowing here names
//! its [`Rounding`] and its [`Overflow`], because both are numeric decisions the caller owns and
//! a silently inferred one is a silently wrong answer.
//!
//! Teaching pointer: [`NumericValue`] is the value an instantiation carries,
//! [`narrow_to_format`] is the one narrowing primitive everything else composes, and
//! [`crate::float_layout_codec::encode_with`] is the arithmetic beneath it.

use crate::error::NumericError;
use crate::float_layout::FloatLayout;
use crate::float_layout_codec::{decode, encode_with};
use crate::format::NumericFormat;

/// The one value another type sees: a format's own value widened exactly. Widening never rounds
/// and never saturates, so it takes no policy.
pub trait ToValue<T> {
    /// # Errors-free by construction
    /// Widening into `f32` cannot fail for any format this crate decodes.
    #[must_use]
    fn to_value(self) -> T;
}

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

/// The value one format instantiation carries, and the format it carries it in.
pub trait NumericValue: Copy {
    /// The format this value is stored as.
    const FORMAT: NumericFormat;
}

/// Narrow a widened value into a scalar layout under an explicit rounding and overflow policy.
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
    let code = encode_with(value, layout, rounding, overflow, &mut nan_count);
    Ok(decode(code, layout))
}

/// Narrow `T` into `Self` under an explicit rounding and overflow policy.
pub trait FromValue<T>: Sized {
    /// # Errors
    /// [`NumericError::WidthUnsupported`] when the target format is wider than the scalar codec.
    fn from_value(value: T, rounding: Rounding, overflow: Overflow) -> Result<Self, NumericError>;
}

macro_rules! numeric_value {
    ($($value:ty => $format:expr;)*) => {
        $(
            impl NumericValue for $value {
                const FORMAT: NumericFormat = $format;
            }
        )*
    };
}

numeric_value! {
    f32 => NumericFormat::Scalar(FloatLayout::BF32);
    i8 => NumericFormat::Scalar(FloatLayout::INT8);
    u8 => NumericFormat::Scalar(FloatLayout::UINT8);
    i16 => NumericFormat::Scalar(FloatLayout::INT16);
    u16 => NumericFormat::Scalar(FloatLayout::UINT16);
    i32 => NumericFormat::Scalar(FloatLayout::INT32);
    u32 => NumericFormat::Scalar(FloatLayout::UINT32);
}

macro_rules! from_f32 {
    ($($value:ty => $layout:expr;)*) => {
        $(
            impl FromValue<f32> for $value {
                fn from_value(
                    value: f32,
                    rounding: Rounding,
                    overflow: Overflow,
                ) -> Result<Self, NumericError> {
                    let narrowed = narrow_to_layout(value, $layout, rounding, overflow)?;
                    Ok(<$value as FromFormat>::from_format(narrowed))
                }
            }
        )*
    };
}

/// A value reconstructed from a format's exact `f32` projection.
trait FromFormat: Sized {
    fn from_format(value: f32) -> Self;
}

impl FromFormat for f32 {
    fn from_format(value: f32) -> Self {
        value
    }
}

impl FromFormat for i8 {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the value already came out of an 8-bit integer layout, so the cast is exact"
    )]
    fn from_format(value: f32) -> Self {
        value as i8
    }
}

impl FromFormat for u8 {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the value already came out of an 8-bit integer layout, so the cast is exact"
    )]
    fn from_format(value: f32) -> Self {
        value as u8
    }
}

impl FromFormat for i16 {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the value already came out of a 16-bit integer layout, so the cast is exact"
    )]
    fn from_format(value: f32) -> Self {
        value as i16
    }
}

impl FromFormat for u16 {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the value already came out of a 16-bit integer layout, so the cast is exact"
    )]
    fn from_format(value: f32) -> Self {
        value as u16
    }
}

impl FromFormat for i32 {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the value already came out of a 32-bit integer layout, so the cast is exact"
    )]
    fn from_format(value: f32) -> Self {
        value as i32
    }
}

impl FromFormat for u32 {
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the value already came out of an unsigned 32-bit integer layout"
    )]
    fn from_format(value: f32) -> Self {
        value as u32
    }
}

from_f32! {
    f32 => FloatLayout::BF32;
    i8 => FloatLayout::INT8;
    u8 => FloatLayout::UINT8;
    i16 => FloatLayout::INT16;
    u16 => FloatLayout::UINT16;
    i32 => FloatLayout::INT32;
    u32 => FloatLayout::UINT32;
}

macro_rules! to_f32 {
    ($($value:ty;)*) => {
        $(
            impl ToValue<f32> for $value {
                fn to_value(self) -> f32 {
                    <$value as Widen>::widen(self)
                }
            }
        )*
    };
}

/// Exact widening into `f32`; every implementor's range sits inside `f32`'s significand.
trait Widen {
    fn widen(self) -> f32;
}

impl Widen for f32 {
    fn widen(self) -> f32 {
        self
    }
}

impl Widen for i8 {
    fn widen(self) -> f32 {
        f32::from(self)
    }
}

impl Widen for u8 {
    fn widen(self) -> f32 {
        f32::from(self)
    }
}

impl Widen for i16 {
    fn widen(self) -> f32 {
        f32::from(self)
    }
}

impl Widen for u16 {
    fn widen(self) -> f32 {
        f32::from(self)
    }
}

to_f32! {
    f32;
    i8;
    u8;
    i16;
    u16;
}
