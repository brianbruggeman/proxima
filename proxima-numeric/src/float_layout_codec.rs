//! Generic scalar float codec over [`FloatLayout`]: `f32 <-> code` by integer field arithmetic.
//!
//! Teaching pointer: [`FloatLayout`] (in this crate) is the data; this module is the one piece of
//! float arithmetic, and `bf16`, `bf8_e5m2` and `bf4_e2m1` are instances of it. Round to nearest
//! with ties to the even code by default, [`encode_with`] takes any [`Rounding`] and
//! [`Overflow`], no `f64`, no allocation.
//!
//! [`encode`] and [`decode`] are unchecked: they assume a layout inside the codec's domain
//! ([`layout_supported`]), because the arithmetic shifts by `23 - mantissa_bits` and by
//! `exponent_bits`, which leaves the domain is an underflow. Every public entry point in this
//! crate asks [`layout_supported`] first, so an out-of-domain layout is a typed error rather than
//! a panic.

use crate::convert::{Overflow, Rounding};
use crate::float_layout::{
    FloatLayout, NanNarrow, NanWiden, SignEncoding, Specials, named_layout_table,
};

use crate::error::NumericError;

/// Exponent bits the codec's `1u32 << exponent_bits` field arithmetic can address.
pub const MAX_EXPONENT_BITS: u8 = 8;

/// Mantissa bits the codec's `23 - mantissa_bits` shift can address.
pub const MAX_MANTISSA_BITS: u8 = 23;

/// Whether [`encode`] and [`decode`] can execute `layout`: at most 32 bits wide, at most 23
/// mantissa bits for a float, and at most 63 mantissa bits for an integer (the integer path
/// accumulates in `u64`).
///
/// # Errors
/// [`NumericError::WidthUnsupported`] past 32 bits, [`NumericError::LayoutUnsupported`] for a
/// float wider in mantissa than binary32 or an integer past 64 bits.
pub const fn layout_supported(layout: &FloatLayout) -> Result<(), NumericError> {
    if layout.width() > 32 {
        return Err(NumericError::WidthUnsupported {
            width: layout.width(),
            max: 32,
        });
    }
    if layout.exponent_bits <= MAX_EXPONENT_BITS && layout.mantissa_bits <= MAX_MANTISSA_BITS {
        return Ok(());
    }
    if layout.exponent_bits == 0 && layout.mantissa_bits <= 63 {
        return Ok(());
    }
    Err(NumericError::LayoutUnsupported {
        exponent_bits: layout.exponent_bits,
        mantissa_bits: layout.mantissa_bits,
        max_exponent_bits: MAX_EXPONENT_BITS,
        max_mantissa_bits: MAX_MANTISSA_BITS,
    })
}

// the one validity predicate of the slice codec: supported and with at least one bit to store
#[inline(always)]
pub(crate) const fn is_storable(layout: &FloatLayout) -> bool {
    layout_supported(layout).is_ok() && layout.width() != 0
}

const F32_FRACTION_BITS: u32 = 23;
const F32_FRACTION_MASK: u32 = 0x7f_ffff;
const F32_BIAS: i32 = 127;
const F32_INFINITY_BITS: u32 = 0x7f80_0000;
const F32_QUIET_BIT: u32 = 1 << 22;

const fn exponent_ones(layout: &FloatLayout) -> u32 {
    (1u32 << layout.exponent_bits) - 1
}

const fn mantissa_mask(layout: &FloatLayout) -> u32 {
    (1u32 << layout.mantissa_bits) - 1
}

const fn sign_shift(layout: &FloatLayout) -> u32 {
    layout.exponent_bits as u32 + layout.mantissa_bits as u32
}

const fn infinity_code(layout: &FloatLayout) -> u32 {
    exponent_ones(layout) << layout.mantissa_bits
}

const fn max_normal_code(layout: &FloatLayout) -> u32 {
    match layout.specials {
        Specials::None => infinity_code(layout) | mantissa_mask(layout),
        Specials::NanOnly if layout.mantissa_bits == 0 => infinity_code(layout) - 1,
        Specials::NanOnly => infinity_code(layout) | (mantissa_mask(layout) - 1),
        Specials::SaturateOnOverflow | Specials::InfinityOnOverflow => {
            (infinity_code(layout) - (1u32 << layout.mantissa_bits)) | mantissa_mask(layout)
        }
    }
}

const fn overflow_code(layout: &FloatLayout) -> u32 {
    match layout.specials {
        Specials::InfinityOnOverflow => infinity_code(layout),
        Specials::None | Specials::NanOnly | Specials::SaturateOnOverflow => {
            max_normal_code(layout)
        }
    }
}

const fn quiet_bit(layout: &FloatLayout) -> u32 {
    (1u32 << layout.mantissa_bits) >> 1
}

#[inline(always)]
const fn nan_payload(fraction: u32, layout: &FloatLayout) -> u32 {
    match layout.nan.narrow {
        NanNarrow::Canonical => quiet_bit(layout),
        NanNarrow::Preserve => {
            (fraction >> (F32_FRACTION_BITS - layout.mantissa_bits as u32)) | quiet_bit(layout)
        }
    }
}

#[inline(always)]
const fn round_shift(value: u32, shift: u32, rounding: Rounding) -> u32 {
    if shift == 0 {
        return value;
    }
    if shift > F32_FRACTION_BITS + 1 {
        return match rounding {
            Rounding::AwayFromZero if value != 0 => 1,
            Rounding::NearestTiesEven | Rounding::TowardZero | Rounding::AwayFromZero => 0,
        };
    }
    let quotient = value >> shift;
    let remainder = value & ((1u32 << shift) - 1);
    let halfway = 1u32 << (shift - 1);
    match rounding {
        Rounding::NearestTiesEven => {
            if remainder > halfway || (remainder == halfway && quotient & 1 == 1) {
                quotient + 1
            } else {
                quotient
            }
        }
        Rounding::TowardZero => quotient,
        Rounding::AwayFromZero => {
            if remainder != 0 {
                quotient + 1
            } else {
                quotient
            }
        }
    }
}

#[inline(always)]
const fn overflow_for(layout: &FloatLayout, overflow: Overflow) -> u32 {
    match overflow {
        Overflow::Format => overflow_code(layout),
        Overflow::Saturate => max_normal_code(layout),
    }
}

#[inline(always)]
const fn finite_magnitude_code(
    abs_bits: u32,
    layout: &FloatLayout,
    rounding: Rounding,
    overflow: Overflow,
) -> u32 {
    let exponent_field = (abs_bits >> F32_FRACTION_BITS) as i32;
    let (significand, source_exponent) = if exponent_field == 0 {
        (abs_bits, 1)
    } else {
        (
            (abs_bits & F32_FRACTION_MASK) | (1 << F32_FRACTION_BITS),
            exponent_field,
        )
    };
    let target = source_exponent - F32_BIAS + layout.bias() as i32;
    if target > exponent_ones(layout) as i32 {
        return overflow_for(layout, overflow);
    }
    let (base_exponent, extra_shift) = if target < 1 {
        (0, (1 - target) as u32)
    } else {
        ((target - 1) as u32, 0)
    };
    let shift = F32_FRACTION_BITS - layout.mantissa_bits as u32 + extra_shift;
    let code = (base_exponent << layout.mantissa_bits) + round_shift(significand, shift, rounding);
    let limit = overflow_for(layout, overflow);
    if code > limit { limit } else { code }
}

#[inline(always)]
const fn magnitude_code(
    abs_bits: u32,
    layout: &FloatLayout,
    rounding: Rounding,
    overflow: Overflow,
) -> u32 {
    if abs_bits >> F32_FRACTION_BITS == 0xff {
        overflow_for(layout, overflow)
    } else {
        finite_magnitude_code(abs_bits, layout, rounding, overflow)
    }
}

#[inline(always)]
const fn nan_code(bits: u32, layout: &FloatLayout) -> u32 {
    match layout.specials {
        Specials::None => max_normal_code(layout),
        Specials::NanOnly => {
            ((bits >> 31) << sign_shift(layout)) | infinity_code(layout) | mantissa_mask(layout)
        }
        Specials::SaturateOnOverflow | Specials::InfinityOnOverflow => {
            ((bits >> 31) << sign_shift(layout))
                | infinity_code(layout)
                | nan_payload(bits & F32_FRACTION_MASK, layout)
        }
    }
}

const fn scale_by_power_of_two(mut value: f32, exponent: i32) -> f32 {
    let mut remaining = exponent;
    while remaining > 0 {
        value *= 2.0;
        remaining -= 1;
    }
    while remaining < 0 {
        value *= 0.5;
        remaining += 1;
    }
    value
}

const fn integer_scale_exponent(layout: &FloatLayout) -> i32 {
    1 - layout.bias() as i32 - layout.mantissa_bits as i32
}

const fn integer_mantissa_mask(layout: &FloatLayout) -> u64 {
    (1u64 << layout.mantissa_bits) - 1
}

const fn integer_sign_bit(code: u32, layout: &FloatLayout) -> u64 {
    match layout.sign {
        SignEncoding::Absent => 0,
        SignEncoding::Magnitude | SignEncoding::TwosComplement => {
            ((code as u64) >> layout.mantissa_bits) & 1
        }
    }
}

#[inline(always)]
const fn integer_decode(code: u32, layout: &FloatLayout) -> f32 {
    let mantissa = (code as u64) & integer_mantissa_mask(layout);
    let sign = integer_sign_bit(code, layout);
    let exponent = integer_scale_exponent(layout);
    match layout.sign {
        SignEncoding::Absent => scale_by_power_of_two(mantissa as f32, exponent),
        SignEncoding::Magnitude => {
            let magnitude = scale_by_power_of_two(mantissa as f32, exponent);
            if sign == 1 { -magnitude } else { magnitude }
        }
        SignEncoding::TwosComplement => {
            let value = mantissa as i64 - ((sign as i64) << layout.mantissa_bits);
            scale_by_power_of_two(value as f32, exponent)
        }
    }
}

const fn round_magnitude(magnitude: f32, rounding: Rounding) -> u64 {
    let truncated = magnitude as u64;
    let fraction = magnitude - truncated as f32;
    match rounding {
        Rounding::NearestTiesEven => {
            if fraction > 0.5 || (fraction == 0.5 && truncated & 1 == 1) {
                truncated.saturating_add(1)
            } else {
                truncated
            }
        }
        Rounding::TowardZero => truncated,
        Rounding::AwayFromZero => {
            if fraction > 0.0 {
                truncated.saturating_add(1)
            } else {
                truncated
            }
        }
    }
}

#[inline(always)]
const fn integer_encode(value: f32, layout: &FloatLayout, rounding: Rounding) -> u32 {
    if value.is_nan() {
        return 0;
    }
    let scaled = scale_by_power_of_two(value, -integer_scale_exponent(layout));
    let negative = scaled.is_sign_negative();
    let magnitude = round_magnitude(if negative { -scaled } else { scaled }, rounding);
    let mask = integer_mantissa_mask(layout);
    let top = layout.mantissa_bits;
    let code = match layout.sign {
        SignEncoding::Absent => {
            if negative {
                0
            } else {
                min_u64(magnitude, mask)
            }
        }
        SignEncoding::Magnitude => min_u64(magnitude, mask) | ((negative as u64) << top),
        SignEncoding::TwosComplement => {
            if negative {
                let clamped = min_u64(magnitude, 1u64 << top);
                ((1u64 << (top + 1)) - clamped) & ((1u64 << (top + 1)) - 1)
            } else {
                min_u64(magnitude, mask)
            }
        }
    };
    code as u32
}

const fn min_u64(left: u64, right: u64) -> u64 {
    if left < right { left } else { right }
}

/// Encodes `value` into `layout`, counting NaN inputs for layouts that have no NaN encoding.
///
/// Rounds to nearest, ties to the even code. Overflow clamps to the signed max normal for
/// `Specials::None` and `SaturateOnOverflow` and becomes infinity for `InfinityOnOverflow`;
/// values below half the minimum subnormal flush to signed zero. NaN becomes the positive max
/// normal and increments `nan_count` for `Specials::None`; `NanOnly` uses its one reserved NaN
/// code, and layouts with IEEE reserved encodings keep the sign and top payload bits, set quiet,
/// and leave the counter alone.
#[must_use]
#[inline(always)]
pub const fn encode_counting(value: f32, layout: FloatLayout, nan_count: &mut u64) -> u32 {
    encode_with(
        value,
        layout,
        Rounding::NearestTiesEven,
        Overflow::Format,
        nan_count,
    )
}

/// Encodes `value` into `layout` under an explicit rounding mode and overflow policy.
///
/// [`Overflow::Format`] follows the layout's own special-value policy; [`Overflow::Saturate`]
/// clamps to the largest finite magnitude whatever the layout's infinity encoding is. NaN and
/// subnormal flushing follow [`encode_counting`].
#[must_use]
#[inline(always)]
pub const fn encode_with(
    value: f32,
    layout: FloatLayout,
    rounding: Rounding,
    overflow: Overflow,
    nan_count: &mut u64,
) -> u32 {
    if layout.exponent_bits == 0 {
        if value.is_nan() {
            *nan_count += 1;
        }
        return integer_encode(value, &layout, rounding);
    }
    let bits = value.to_bits();
    if layout.is_identity_of(&SOURCE_LAYOUT) {
        return bits;
    }
    let bits = without_unrepresentable_sign(bits, &layout);
    if truncates_f32(&layout, rounding, overflow) {
        return prefix_code(bits, &layout);
    }
    if rounds_by_magic_add(&layout, rounding, overflow) {
        return magic_add_code(bits, &layout);
    }
    let abs_bits = bits & 0x7fff_ffff;
    let is_nan = abs_bits > F32_INFINITY_BITS;
    if matches!(layout.specials, Specials::None) {
        *nan_count += is_nan as u64;
    }
    let magnitude = if straight_line(&layout, rounding) {
        straight_line_magnitude_code(abs_bits, &layout, overflow_for(&layout, overflow))
    } else {
        magnitude_code(abs_bits, &layout, rounding, overflow)
    };
    let sign = (bits >> 31) << sign_shift(&layout);
    if is_nan {
        nan_code(bits, &layout)
    } else {
        sign | magnitude
    }
}

// a layout with no sign bit stores the magnitude only, the code the slice route already stored for a negative
// once its byte was cut to the layout width; or-ing the sign in would set a bit above the width
#[inline(always)]
const fn without_unrepresentable_sign(bits: u32, layout: &FloatLayout) -> u32 {
    match layout.sign {
        SignEncoding::Absent => bits & 0x7fff_ffff,
        SignEncoding::Magnitude | SignEncoding::TwosComplement => bits,
    }
}

// a prefix layout needs one rounding and nothing else: the carry into the exponent already yields its infinity
#[inline(always)]
const fn truncates_f32(layout: &FloatLayout, rounding: Rounding, overflow: Overflow) -> bool {
    matches!(rounding, Rounding::NearestTiesEven)
        && matches!(overflow, Overflow::Format)
        && layout.is_mantissa_prefix_of(&SOURCE_LAYOUT)
}

// same default rounding and overflow as the prefix route, for the layouts that are not a prefix
#[inline(always)]
const fn rounds_by_magic_add(layout: &FloatLayout, rounding: Rounding, overflow: Overflow) -> bool {
    matches!(rounding, Rounding::NearestTiesEven)
        && matches!(overflow, Overflow::Format)
        && magic_add_selected(layout)
}

// the one decision of which layouts take the magic add; the per-element and the slice route both ask it
#[inline(always)]
const fn magic_add_selected(layout: &FloatLayout) -> bool {
    !layout.is_mantissa_prefix_of(&SOURCE_LAYOUT) && magic_add_applies(layout)
}

// a round-up carry never reaches bit 31, so rounding the whole pattern converts sign and magnitude at once;
// with no bits dropped and every nan bit kept the pattern is returned as it came, so a signalling nan survives
#[inline(always)]
const fn prefix_code(bits: u32, layout: &FloatLayout) -> u32 {
    let shift = F32_FRACTION_BITS - layout.mantissa_bits as u32;
    if shift == 0 {
        return match layout.nan.narrow {
            NanNarrow::Preserve => bits,
            NanNarrow::Canonical if (bits << 1) > F32_INFINITY_BITS << 1 => {
                (bits & (1 << 31)) | F32_INFINITY_BITS | F32_QUIET_BIT
            }
            NanNarrow::Canonical => bits,
        };
    }
    let rounded = rounded_shifted(bits, shift, false);
    let kept = match layout.nan.narrow {
        NanNarrow::Preserve => u32::MAX,
        NanNarrow::Canonical => 1 << 31,
    };
    let nan = ((bits & kept) | F32_INFINITY_BITS | F32_QUIET_BIT) >> shift;
    if (bits << 1) > F32_INFINITY_BITS << 1 { nan } else { rounded }
}

// a layout with no mantissa bit rounds every normal tie up, which the fpu's tie-to-even add does not; every built constant must be a normal f32
#[inline(always)]
const fn magic_add_applies(layout: &FloatLayout) -> bool {
    let bias = layout.bias() as i32;
    let shift = F32_FRACTION_BITS as i32 - layout.mantissa_bits as i32;
    matches!(layout.specials, Specials::InfinityOnOverflow)
        && matches!(layout.sign, SignEncoding::Magnitude)
        && layout.exponent_bits != 0
        && layout.mantissa_bits >= 1
        && layout.mantissa_bits < 23
        && bias >= 0
        && bias <= 127
        && F32_BIAS + exponent_ones(layout) as i32 - bias + shift <= 254
}

// adding a power of two whose ulp is the target's lets the fpu do the tie-to-even rounding, subnormals included, with no data-dependent branch
macro_rules! magic_add_lane {
    ($name:ident, $lane:ty) => {
        #[inline(always)]
        const fn $name(bits: u32, layout: &FloatLayout) -> $lane {
            let bias = layout.bias() as i32;
            let mantissa = layout.mantissa_bits as u32;
            let shift = F32_FRACTION_BITS - mantissa;
            let magnitude_width = layout.exponent_bits as u32 + mantissa;
            let cap = ((F32_BIAS + exponent_ones(layout) as i32 - bias) as u32) << F32_FRACTION_BITS;
            let min_normal = ((F32_BIAS + 1 - bias) as u32) << F32_FRACTION_BITS;
            let rebias = ((F32_BIAS + 1 - bias) as u32 + shift) << mantissa;
            let magnitude = bits & 0x7fff_ffff;
            let capped = if magnitude < cap { magnitude } else { cap };
            let exponent =
                (if capped > min_normal { capped } else { min_normal }) & F32_INFINITY_BITS;
            let magic = exponent + (shift << F32_FRACTION_BITS);
            let sum = (f32::from_bits(capped) + f32::from_bits(magic)).to_bits();
            let code = (sum as $lane)
                .wrapping_sub(magic as $lane)
                .wrapping_add((magic >> shift) as $lane)
                .wrapping_sub(rebias as $lane);
            let nan_kept = match layout.nan.narrow {
                NanNarrow::Preserve => (magnitude >> shift) as $lane,
                NanNarrow::Canonical => 0,
            };
            let nan = nan_kept | infinity_code(layout) as $lane | quiet_bit(layout) as $lane;
            let selected = if magnitude > F32_INFINITY_BITS { nan } else { code };
            let magnitude_mask = ((1u32 << magnitude_width) - 1) as $lane;
            (selected & magnitude_mask) | ((bits >> (31 - magnitude_width)) as $lane & !magnitude_mask)
        }
    };
}

// the lane is the type everything after the one fpu add is held in: a layout of at most 16 bits
// fits u16, and the subtract/add chain is arithmetic modulo 2^16, so a vectorised loop runs it at
// twice the u32 lane count and stores 16-bit lanes with no change to any result bit
magic_add_lane!(magic_add_code, u32);
magic_add_lane!(magic_add_code_narrow, u16);

// any other layout takes the field-by-field path, so every layout is admitted and only the named ones fold
#[inline(always)]
const fn straight_line(layout: &FloatLayout, rounding: Rounding) -> bool {
    let bias = layout.bias() as i32;
    let mantissa = layout.mantissa_bits as i32;
    matches!(rounding, Rounding::NearestTiesEven)
        && layout.exponent_bits != 0
        && mantissa <= 22
        && bias >= 0
        && bias <= 127
        && bias + mantissa >= -103
        && bias + mantissa <= 150
}

// both the normal and the subnormal result are computed and one selected, so a loop over it vectorizes
#[inline(always)]
const fn straight_line_magnitude_code(abs_bits: u32, layout: &FloatLayout, limit: u32) -> u32 {
    let bias = layout.bias() as i32;
    let mantissa = layout.mantissa_bits as i32;
    let shift = F32_FRACTION_BITS - layout.mantissa_bits as u32;
    let rebias = ((F32_BIAS - bias) as u32) << F32_FRACTION_BITS;
    let min_normal_bits = ((F32_BIAS + 1 - bias) as u32) << F32_FRACTION_BITS;
    let implicit_one = mantissa == 0 && abs_bits >= min_normal_bits;
    let normal = rounded_shifted(abs_bits.wrapping_sub(rebias), shift, implicit_one);
    let code = if bias == F32_BIAS {
        normal
    } else {
        let magic = f32::from_bits(((151 - bias - mantissa) as u32) << F32_FRACTION_BITS);
        let subnormal = (f32::from_bits(abs_bits) + magic)
            .to_bits()
            .wrapping_sub(magic.to_bits());
        if abs_bits < min_normal_bits { subnormal } else { normal }
    };
    let reaches_limit =
        rounded_shifted(F32_INFINITY_BITS.wrapping_sub(rebias), shift, mantissa == 0) > limit;
    if reaches_limit && code > limit { limit } else { code }
}

// with no mantissa bits a normal value's implicit one is the code's low bit, so every tie rounds up, as the field-by-field path does
#[inline(always)]
const fn rounded_shifted(shifted: u32, shift: u32, implicit_one_is_low_bit: bool) -> u32 {
    let tie = if implicit_one_is_low_bit { 1 } else { (shifted >> shift) & 1 };
    shifted.wrapping_add((1u32 << (shift - 1)) - 1 + tie) >> shift
}

/// Encodes `value` into `layout`; the same rounding as [`encode_counting`] without the counter.
#[must_use]
#[inline(always)]
pub const fn encode(value: f32, layout: FloatLayout) -> u32 {
    let mut ignored_nan_count = 0;
    encode_counting(value, layout, &mut ignored_nan_count)
}

#[inline(always)]
const fn f32_bits_from_scaled(significand: u32, scale: i32) -> u32 {
    if significand == 0 {
        return 0;
    }
    let lead = 31 - significand.leading_zeros();
    let biased = lead as i32 + scale + F32_BIAS;
    if biased >= 255 {
        F32_INFINITY_BITS
    } else if biased >= 1 {
        ((biased as u32) << F32_FRACTION_BITS)
            | ((significand << (F32_FRACTION_BITS - lead)) & F32_FRACTION_MASK)
    } else if scale + 149 >= 0 {
        significand << (scale + 149) as u32
    } else {
        significand >> (-(scale + 149)) as u32
    }
}

#[inline(always)]
const fn finite_magnitude_bits(exponent: u32, mantissa: u32, layout: &FloatLayout) -> u32 {
    let mantissa_bits = layout.mantissa_bits as i32;
    if exponent == 0 {
        f32_bits_from_scaled(mantissa, 1 - layout.bias() as i32 - mantissa_bits)
    } else {
        let significand = (1u32 << layout.mantissa_bits) | mantissa;
        f32_bits_from_scaled(
            significand,
            exponent as i32 - layout.bias() as i32 - mantissa_bits,
        )
    }
}

/// Decodes `code` from `layout` exactly; the sign bit sits at `exponent_bits + mantissa_bits`.
/// A reserved NaN code keeps its payload, shifted to the top of the `f32` fraction, and widens as
/// its [`NanPolicy`](crate::float_layout::NanPolicy) says; a layout equal to `f32` in format and NaN
/// policy returns the pattern unchanged.
#[must_use]
#[inline(always)]
pub const fn decode(code: u32, layout: FloatLayout) -> f32 {
    if layout.exponent_bits == 0 {
        return integer_decode(code, &layout);
    }
    if layout.is_identity_of(&SOURCE_LAYOUT) {
        return f32::from_bits(code);
    }
    if layout.is_mantissa_prefix_of(&SOURCE_LAYOUT) {
        return embedded_f32(code, &layout);
    }
    let mantissa = code & mantissa_mask(&layout);
    let exponent = (code >> layout.mantissa_bits) & exponent_ones(&layout);
    let reserved = match layout.specials {
        Specials::None => false,
        Specials::NanOnly => {
            exponent == exponent_ones(&layout) && mantissa == mantissa_mask(&layout)
        }
        Specials::SaturateOnOverflow | Specials::InfinityOnOverflow => {
            exponent == exponent_ones(&layout)
        }
    };
    let reserved_bits = if mantissa == 0 {
        F32_INFINITY_BITS
    } else if drops_nan_payload(&layout) {
        F32_INFINITY_BITS | F32_QUIET_BIT
    } else {
        F32_INFINITY_BITS
            | F32_QUIET_BIT
            | (mantissa << (F32_FRACTION_BITS - layout.mantissa_bits as u32))
    };
    let finite_bits = if straight_line(&layout, Rounding::NearestTiesEven) {
        straight_line_finite_bits(code, &layout)
    } else {
        finite_magnitude_bits(exponent, mantissa, &layout)
    };
    let magnitude_bits = if reserved { reserved_bits } else { finite_bits };
    let sign = (code >> sign_shift(&layout)) & 1;
    f32::from_bits(magnitude_bits | (sign << 31))
}

const fn drops_nan_payload(layout: &FloatLayout) -> bool {
    matches!(layout.nan.widen, NanWiden::Canonical)
}

// the code shifted into place is the exact f32, so the nan policy decides only whether the quiet bit is set
#[inline(always)]
const fn embedded_f32(code: u32, layout: &FloatLayout) -> f32 {
    let widen = F32_FRACTION_BITS - layout.mantissa_bits as u32;
    if layout.width() <= 16 {
        f32::from_bits(embedded_bits_narrow(code as u16, widen, layout))
    } else {
        f32::from_bits(embedded_bits(code, widen, layout))
    }
}

macro_rules! embedded_lane {
    ($name:ident, $lane:ty) => {
        #[inline(always)]
        const fn $name(code: $lane, widen: u32, layout: &FloatLayout) -> u32 {
            let code = code & (((1u64 << layout.width()) - 1) as $lane);
            let magnitude = code & (((1u64 << sign_shift(layout)) - 1) as $lane);
            let is_nan = magnitude > (F32_INFINITY_BITS >> widen) as $lane;
            let quiet_code = (F32_QUIET_BIT >> widen) as $lane;
            let canonical_code = (code ^ magnitude) | ((F32_INFINITY_BITS | F32_QUIET_BIT) >> widen) as $lane;
            let quieted = match (is_nan, drops_nan_payload(layout)) {
                (false, _) => code,
                (true, false) => code | quiet_code,
                (true, true) => canonical_code,
            };
            (quieted as u32) << widen
        }
    };
}

// the lane is the type the nan test and the quiet bit run in: a code of at most 16 bits fits u16,
// so a vectorised loop does them at twice the lane count and widens once, by the shift
embedded_lane!(embedded_bits, u32);
embedded_lane!(embedded_bits_narrow, u16);

// one exact multiply by a power of two re-biases a normal code and renormalizes a subnormal one, with no branch
#[inline(always)]
const fn straight_line_finite_bits(code: u32, layout: &FloatLayout) -> u32 {
    let bias = layout.bias() as i32;
    let shift = F32_FRACTION_BITS - layout.mantissa_bits as u32;
    let field_bits = layout.exponent_bits as u32 + layout.mantissa_bits as u32;
    let fields = code & ((1u32 << field_bits) - 1);
    let rebias = f32::from_bits(((2 * F32_BIAS - bias) as u32) << F32_FRACTION_BITS);
    (f32::from_bits(fields << shift) * rebias).to_bits()
}

// the f32 the graph computes in; every slice converts from or to it
const SOURCE_LAYOUT: FloatLayout = FloatLayout::BF32;

// plain data, so the hot path builds no error text; the caller makes a NumericError on the cold path
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SliceFault {
    /// The layout is outside the codec's domain or has no bits; [`require_storage`] names which.
    Layout,
    /// The output holds `found` units where the input needs `expected`.
    Output { found: usize, expected: usize },
    /// The packed input is not a whole number of `block_bytes` elements.
    NotMultiple { found: usize, block_bytes: usize },
}

// the one runtime-layout to instantiation bridge; a const, not a binding, so an arm's closure still coerces to a fn pointer
macro_rules! dispatch_named_layout {
    ($layout:expr, $named:ident => $arm:expr, else $fallback:expr, [ $(($name:ident, $text:literal)),* $(,)? ]) => {
        match $layout.fingerprint() {
            $(
                found if found == FloatLayout::$name.fingerprint() && is_storable(&FloatLayout::$name) =>
                {
                    const $named: FloatLayout = FloatLayout::$name;
                    $arm
                }
            )*
            _ => $fallback,
        }
    };
}

pub(crate) use dispatch_named_layout;

#[inline(always)]
pub(crate) const fn element_bytes(layout: &FloatLayout) -> usize {
    layout.width().div_ceil(8) as usize
}

// inline(always) is what turns a literal layout into a compile-time constant
#[inline(always)]
fn encode_run(layout: FloatLayout, values: &[f32], out: &mut [u8]) -> Result<(), SliceFault> {
    let expected = values.len() * element_bytes(&layout);
    if out.len() != expected {
        return Err(SliceFault::Output {
            found: out.len(),
            expected,
        });
    }
    encode_body(layout, values, out);
    Ok(())
}

// one count bounds both views, so the loop is a single counted pass instead of a zip clamping to the shorter side
#[inline(always)]
fn exact_chunks<const WIDTH: usize>(bytes: &[u8], count: usize) -> Option<&[[u8; WIDTH]]> {
    bytes.as_chunks::<WIDTH>().0.get(..count)
}

#[inline(always)]
fn exact_chunks_mut<const WIDTH: usize>(bytes: &mut [u8], count: usize) -> Option<&mut [[u8; WIDTH]]> {
    bytes.as_chunks_mut::<WIDTH>().0.get_mut(..count)
}

// the callers have already tied out.len() to values.len(); a shorter out converts nothing
#[inline(always)]
pub(crate) fn encode_body(layout: FloatLayout, values: &[f32], out: &mut [u8]) {
    let bytes = element_bytes(&layout);
    if layout.is_identity_of(&SOURCE_LAYOUT) {
        let Some(chunks) = exact_chunks_mut::<4>(out, values.len()) else {
            return;
        };
        for (value, chunk) in values.iter().zip(chunks) {
            *chunk = value.to_le_bytes();
        }
        return;
    }
    if bytes == 2 && magic_add_selected(&layout) {
        let Some(chunks) = exact_chunks_mut::<2>(out, values.len()) else {
            return;
        };
        for (value, slot) in values.iter().zip(chunks) {
            *slot = magic_add_code_narrow(value.to_bits(), &layout).to_le_bytes();
        }
        return;
    }
    for (value, slot) in values.iter().zip(out.chunks_exact_mut(bytes)) {
        slot.copy_from_slice(&encode(*value, layout).to_le_bytes()[..bytes]);
    }
}

#[inline(always)]
fn packed_elements(layout: &FloatLayout, data: &[u8]) -> Result<usize, SliceFault> {
    let bytes = element_bytes(layout);
    if !data.len().is_multiple_of(bytes) {
        return Err(SliceFault::NotMultiple {
            found: data.len(),
            block_bytes: bytes,
        });
    }
    Ok(data.len() / bytes)
}

#[inline(always)]
fn decode_run(layout: FloatLayout, data: &[u8], out: &mut [f32]) -> Result<(), SliceFault> {
    let count = packed_elements(&layout, data)?;
    if out.len() != count {
        return Err(SliceFault::Output {
            found: out.len(),
            expected: count,
        });
    }
    decode_body(layout, data, out, |slot, value| *slot = value);
    Ok(())
}

// generic over the slot so one loop fills both an f32 and a caller-owned MaybeUninit<f32>
#[inline(always)]
pub(crate) fn decode_body<Slot>(
    layout: FloatLayout,
    data: &[u8],
    out: &mut [Slot],
    write: impl Fn(&mut Slot, f32) + Copy,
) {
    let bytes = element_bytes(&layout);
    if layout.is_identity_of(&SOURCE_LAYOUT) {
        let Some(chunks) = exact_chunks::<4>(data, out.len()) else {
            return;
        };
        for (chunk, slot) in chunks.iter().zip(out.iter_mut()) {
            write(slot, f32::from_le_bytes(*chunk));
        }
        return;
    }
    if bytes == 2 && layout.is_mantissa_prefix_of(&SOURCE_LAYOUT) {
        let widen = F32_FRACTION_BITS - layout.mantissa_bits as u32;
        let Some(pairs) = exact_chunks::<2>(data, out.len()) else {
            return;
        };
        for (pair, value) in pairs.iter().zip(out.iter_mut()) {
            let bits = embedded_bits_narrow(u16::from_le_bytes(*pair), widen, &layout);
            write(value, f32::from_bits(bits));
        }
        return;
    }
    for (slot, value) in data.chunks_exact(bytes).zip(out.iter_mut()) {
        let mut word = [0u8; 4];
        word[..bytes].copy_from_slice(slot);
        write(value, decode(u32::from_le_bytes(word), layout));
    }
}

fn require_storage(layout: &FloatLayout) -> Result<(), NumericError> {
    layout_supported(layout)?;
    if !is_storable(layout) {
        return Err(NumericError::InvalidGeometry {
            format: "scalar",
            reason: "a layout with no bits",
        });
    }
    Ok(())
}

impl SliceFault {
    // cold, so it may resolve the format's name and re-ask for the layout error
    pub(crate) fn into_error(self, layout: &FloatLayout, format: &'static str) -> NumericError {
        match self {
            Self::Layout => layout_supported(layout).err().unwrap_or(NumericError::InvalidGeometry {
                format,
                reason: "a scalar with no bits",
            }),
            Self::Output { found, expected } => NumericError::OutputSizeMismatch { found, expected },
            Self::NotMultiple { found, block_bytes } => NumericError::InputNotBlockMultiple {
                format,
                found,
                block_bytes,
            },
        }
    }
}

/// Encodes `values` into `out`, one element of `layout` per value, little-endian. A layout equal to
/// `f32` is a byte copy, so signalling NaN payloads survive. A named layout runs no checks beyond the
/// output length; any other layout pays [`require_storage`] once.
///
/// # Errors
/// [`SliceFault`]: an unsupported or zero-width layout, or an `out` that is not `values.len()`
/// elements.
pub(crate) fn encode_slice(
    layout: &FloatLayout,
    values: &[f32],
    out: &mut [u8],
) -> Result<(), SliceFault> {
    named_layout_table!(dispatch_named_layout!(
        *layout,
        NAMED => encode_run(NAMED, values, out),
        else {
            require_storage(layout).map_err(|_| SliceFault::Layout)?;
            encode_run(*layout, values, out)
        },
    ))
}

/// The inverse of [`encode_slice`]: `data` holds exactly `out.len()` elements of `layout`.
///
/// # Errors
/// As [`encode_slice`], and a `data` that is not a whole number of elements.
pub(crate) fn decode_slice(
    layout: &FloatLayout,
    data: &[u8],
    out: &mut [f32],
) -> Result<(), SliceFault> {
    named_layout_table!(dispatch_named_layout!(
        *layout,
        NAMED => decode_run(NAMED, data, out),
        else {
            require_storage(layout).map_err(|_| SliceFault::Layout)?;
            decode_run(*layout, data, out)
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::float_layout::{NanNarrow, NanPolicy, NanWiden};

    fn bit_patterns(count: usize) -> Vec<f32> {
        let mut state = 0x9e37_79b9_u32;
        (0..count)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                f32::from_bits(state ^ (state << 13))
            })
            .collect()
    }

    fn encoded(layout: &FloatLayout, values: &[f32]) -> Vec<u8> {
        let mut out = vec![0u8; values.len() * element_bytes(layout)];
        encode_slice(layout, values, &mut out).expect("a supported layout encodes");
        out
    }

    #[test]
    fn e5m2_nan_is_the_canonical_quiet_nan_whatever_the_payload() {
        let inputs = [0x7ff0_8983, 0x7f80_0001, 0xff80_0001, 0x7fc5_0737].map(f32::from_bits);

        let codes = encoded(&FloatLayout::E5M2, &inputs);

        assert_eq!(codes, [0x7e, 0x7e, 0xfe, 0x7e]);
    }

    #[test]
    fn f16_nan_keeps_its_top_payload_bits() {
        let input = f32::from_bits(0x7f85_0737);

        let code = encode(input, FloatLayout::F16);

        assert_eq!(code, 0x7e00 | (0x0005_0737 >> 13));
    }

    #[test]
    fn equal_layouts_copy_signalling_nan_payloads_bit_for_bit() {
        let inputs = [0x7f85_0737, 0xff80_0001, 0x7fc0_0000].map(f32::from_bits);
        let bytes = encoded(&FloatLayout::BF32, &inputs);
        let mut back = [0.0f32; 3];

        decode_slice(&FloatLayout::BF32, &bytes, &mut back).expect("binary32 decodes");

        assert_eq!(back.map(f32::to_bits), inputs.map(f32::to_bits));
    }

    #[test]
    fn identity_layout_returns_a_signalling_nan_unchanged_through_every_entry_point() {
        let signalling = 0x7f80_0001u32;
        let mut sliced = [0.0f32];
        let mut stored = [0u8; 4];

        decode_slice(&FloatLayout::BF32, &signalling.to_le_bytes(), &mut sliced)
            .expect("binary32 decodes");
        encode_slice(&FloatLayout::BF32, &[f32::from_bits(signalling)], &mut stored)
            .expect("binary32 encodes");

        assert_eq!(decode(signalling, FloatLayout::BF32).to_bits(), signalling);
        assert_eq!(sliced[0].to_bits(), signalling);
        assert_eq!(encode(f32::from_bits(signalling), FloatLayout::BF32), signalling);
        assert_eq!(u32::from_le_bytes(stored), signalling);
    }

    #[test]
    fn a_float8_nan_policy_on_the_binary32_layout_still_canonicalises_narrowing() {
        let layout = FloatLayout {
            nan: crate::float_layout::NanPolicy::FLOAT8,
            ..FloatLayout::BF32
        };
        let signalling = f32::from_bits(0x7f80_0001);
        let mut stored = [0u8; 4];

        encode_slice(&layout, &[signalling], &mut stored).expect("binary32 encodes");

        assert_eq!(encode(signalling, layout), 0x7fc0_0000);
        assert_eq!(u32::from_le_bytes(stored), 0x7fc0_0000);
    }

    #[test]
    fn named_instantiation_matches_the_runtime_layout_path_bit_for_bit() {
        let inputs = bit_patterns(65_536);
        for layout in [
            FloatLayout::F16,
            FloatLayout::BF16,
            FloatLayout::E5M2,
            FloatLayout::E5M2_SATURATING,
            FloatLayout::E4M3FN,
            FloatLayout::E3M2,
            FloatLayout::E2M1,
        ] {
            let runtime = FloatLayout {
                bias_override: Some(layout.bias()),
                ..layout
            };
            assert_ne!(runtime, layout);

            let from_instantiation = encoded(&layout, &inputs);
            let from_runtime = encoded(&runtime, &inputs);

            assert_eq!(from_instantiation, from_runtime, "{layout:?}");
        }
    }

    const FLOAT_LAYOUTS: [FloatLayout; 10] = [
        FloatLayout::F16,
        FloatLayout::BF16,
        FloatLayout::E5M2,
        FloatLayout::E5M2_SATURATING,
        FloatLayout::E4M3FN,
        FloatLayout::E3M2,
        FloatLayout::E2M3,
        FloatLayout::E2M2,
        FloatLayout::E2M1,
        FloatLayout::F8_E8M0,
    ];

    #[test]
    fn straight_line_encode_matches_the_field_by_field_path_on_every_stride_of_f32() {
        for layout in FLOAT_LAYOUTS {
            let limit = overflow_for(&layout, Overflow::Format);
            for abs_bits in (0..=F32_INFINITY_BITS).step_by(257).chain([F32_INFINITY_BITS]) {
                let fast = straight_line_magnitude_code(abs_bits, &layout, limit);
                let reference =
                    magnitude_code(abs_bits, &layout, Rounding::NearestTiesEven, Overflow::Format);
                assert_eq!(fast, reference, "{layout:?} {abs_bits:#010x}");
            }
        }
    }

    #[test]
    fn straight_line_encode_matches_the_field_by_field_path_at_every_rounding_boundary() {
        for layout in FLOAT_LAYOUTS {
            let limit = overflow_for(&layout, Overflow::Format);
            let step = 1u32 << (F32_FRACTION_BITS - u32::from(layout.mantissa_bits).min(22));
            for code in 0..=infinity_code(&layout).min(0x1ff) + 2 {
                let representable = encode_to_f32_bits(code, &layout);
                for offset in [0, 1, step / 2 - 1, step / 2, step / 2 + 1, step - 1] {
                    let abs_bits = representable.saturating_add(offset).min(F32_INFINITY_BITS);
                    let fast = straight_line_magnitude_code(abs_bits, &layout, limit);
                    let reference = magnitude_code(
                        abs_bits,
                        &layout,
                        Rounding::NearestTiesEven,
                        Overflow::Format,
                    );
                    assert_eq!(fast, reference, "{layout:?} {abs_bits:#010x}");
                }
            }
        }
    }

    fn encode_to_f32_bits(code: u32, layout: &FloatLayout) -> u32 {
        let exponent = (code >> layout.mantissa_bits) & exponent_ones(layout);
        let mantissa = code & mantissa_mask(layout);
        finite_magnitude_bits(exponent, mantissa, layout)
    }

    #[test]
    fn straight_line_decode_matches_the_field_by_field_path_on_every_code() {
        for layout in FLOAT_LAYOUTS {
            for code in 0..1u32 << (layout.exponent_bits + layout.mantissa_bits) {
                let exponent = (code >> layout.mantissa_bits) & exponent_ones(&layout);
                let mantissa = code & mantissa_mask(&layout);
                let reserved = !matches!(layout.specials, Specials::None)
                    && exponent == exponent_ones(&layout);
                if reserved {
                    continue;
                }
                let fast = straight_line_finite_bits(code, &layout);
                let reference = finite_magnitude_bits(exponent, mantissa, &layout);
                assert_eq!(fast, reference, "{layout:?} {code:#06x}");
            }
        }
    }

    fn infinity_layouts() -> [FloatLayout; 5] {
        let infinity = |exponent_bits, mantissa_bits, bias_override| FloatLayout {
            exponent_bits,
            mantissa_bits,
            specials: Specials::InfinityOnOverflow,
            bias_override,
            sign: SignEncoding::Magnitude,
            nan: crate::float_layout::NanPolicy::IEEE,
        };
        [
            FloatLayout::F16,
            FloatLayout::E5M2,
            infinity(5, 2, Some(10)),
            infinity(4, 3, None),
            infinity(6, 9, Some(20)),
        ]
    }

    #[test]
    fn magic_add_narrowing_matches_the_field_by_field_path_on_every_stride_of_all_f32_patterns() {
        for layout in infinity_layouts() {
            assert!(magic_add_applies(&layout), "{layout:?}");
            for bits in (0..=u32::MAX).step_by(4099).chain([0x7f80_0000, 0xff80_0000, 0x7f80_0001]) {
                let abs_bits = bits & 0x7fff_ffff;
                let expected = if abs_bits > F32_INFINITY_BITS {
                    nan_code(bits, &layout)
                } else {
                    let magnitude =
                        magnitude_code(abs_bits, &layout, Rounding::NearestTiesEven, Overflow::Format);
                    ((bits >> 31) << sign_shift(&layout)) | magnitude
                };
                assert_eq!(magic_add_code(bits, &layout), expected, "{layout:?} {bits:#010x}");
            }
        }
    }

    #[test]
    fn sixteen_bit_lane_narrowing_matches_the_u32_pipeline_on_every_stride_of_all_f32_patterns() {
        for layout in infinity_layouts().into_iter().filter(|layout| layout.width() <= 16) {
            for bits in (0..=u32::MAX).step_by(997).chain([0x7f80_0000, 0xff80_0000, 0x7f80_0001, 0xffc0_0000]) {
                assert_eq!(
                    u32::from(magic_add_code_narrow(bits, &layout)),
                    magic_add_code(bits, &layout),
                    "{layout:?} {bits:#010x}"
                );
            }
        }
    }

    #[test]
    fn magic_add_narrowing_matches_at_every_rounding_boundary_of_f16() {
        let layout = FloatLayout::F16;
        for code in 0..=0x7c01u32 {
            let representable = encode_to_f32_bits(code & 0x7fff, &layout);
            for offset in [0u32, 1, 0xfff, 0x1000, 0x1001, 0x1fff] {
                let abs_bits = representable.saturating_add(offset).min(F32_INFINITY_BITS);
                let expected =
                    magnitude_code(abs_bits, &layout, Rounding::NearestTiesEven, Overflow::Format);
                assert_eq!(magic_add_code(abs_bits, &layout), expected, "{abs_bits:#010x}");
            }
        }
    }

    #[test]
    fn mantissa_prefix_property_holds_for_bf16_and_binary32_only() {
        let prefixes = FLOAT_LAYOUTS.map(|layout| layout.is_mantissa_prefix_of(&SOURCE_LAYOUT));

        assert_eq!(
            prefixes,
            [false, true, false, false, false, false, false, false, false, false]
        );
        assert!(FloatLayout::BF32.is_mantissa_prefix_of(&SOURCE_LAYOUT), "binary32 is its own prefix");
    }

    #[test]
    fn an_ieee_nan_policy_on_e5m2_keeps_the_top_payload_bits_per_element_and_per_slice() {
        let layout = FloatLayout { nan: NanPolicy::IEEE, ..FloatLayout::E5M2 };
        let input = f32::from_bits(0x7ff0_8983);
        let mut stored = [0u8; 1];

        encode_slice(&layout, &[input], &mut stored).expect("e5m2 encodes");

        assert_eq!(encode(input, layout), 0x7f);
        assert_eq!(stored, [0x7f]);
    }

    #[test]
    fn a_float8_nan_policy_on_binary16_widens_to_the_canonical_quiet_nan() {
        let layout = FloatLayout { nan: NanPolicy::FLOAT8, ..FloatLayout::F16 };

        let widened = [0x7c01, 0xfe55].map(|code| decode(code, layout).to_bits());

        assert_eq!(widened, [0x7fc0_0000, 0xffc0_0000]);
    }

    #[test]
    fn layouts_that_differ_only_in_nan_policy_have_different_fingerprints() {
        let policies: Vec<NanPolicy> = [NanNarrow::Preserve, NanNarrow::Canonical]
            .into_iter()
            .flat_map(|narrow| {
                [NanWiden::Quiet, NanWiden::Canonical]
                    .into_iter()
                    .map(move |widen| NanPolicy { narrow, widen })
            })
            .collect();

        let fingerprints: Vec<u64> = policies
            .iter()
            .map(|nan| FloatLayout { nan: *nan, ..FloatLayout::F16 }.fingerprint())
            .collect();

        for (index, fingerprint) in fingerprints.iter().enumerate() {
            assert_eq!(fingerprints.iter().filter(|other| *other == fingerprint).count(), 1, "policy {index}");
        }
    }

    /// Narrowing: ml_dtypes `float8.h:1010` `quiet_NaN()` is `0b0_11111_10`, so every NaN stores as
    /// `sign | 0x7e` (onnxruntime `float8.h:398-399` stores `sign | 0x7f`; the sources disagree and the
    /// current behaviour stays). Widening: onnxruntime `float8.h:454-462`, canonical quiet NaN with the
    /// sign. The saturating layout is the one named `bfloat8`.
    #[test]
    fn the_saturating_e5m2_layout_narrows_to_0x7e_and_widens_to_the_canonical_quiet_nan() {
        let positive = f32::from_bits(0x7ff0_8983);
        let negative = f32::from_bits(0xff80_0001);
        let layout = FloatLayout::E5M2_SATURATING;

        assert_eq!(encode(positive, layout), 0x7e);
        assert_eq!(encode(negative, layout), 0xfe);
        assert_eq!(decode(0x7e, layout).to_bits(), 0x7fc0_0000);
        assert_eq!(decode(0x7d, layout).to_bits(), 0x7fc0_0000);
        assert_eq!(decode(0xfd, layout).to_bits(), 0xffc0_0000);
    }

    #[test]
    fn truncating_narrow_matches_the_field_by_field_path_on_every_stride_of_f32() {
        let layout = FloatLayout::BF16;
        for abs_bits in (0..=F32_INFINITY_BITS).step_by(251).chain([F32_INFINITY_BITS]) {
            let truncated = prefix_code(abs_bits, &layout);
            let reference =
                magnitude_code(abs_bits, &layout, Rounding::NearestTiesEven, Overflow::Format);
            assert_eq!(truncated, reference, "{abs_bits:#010x}");
        }
    }

    #[test]
    fn embedded_widen_matches_the_field_by_field_path_on_every_finite_bf16_code() {
        let layout = FloatLayout::BF16;
        for code in 0..0x7f80u32 {
            let exponent = code >> 7;
            let reference = finite_magnitude_bits(exponent, code & 0x7f, &layout);
            assert_eq!(embedded_f32(code, &layout).to_bits(), reference, "{code:#06x}");
        }
    }

    #[test]
    fn float8_policy_widens_every_nan_code_to_the_canonical_quiet_nan_keeping_the_sign() {
        let widened = [0x7d, 0x7e, 0x7f, 0xfd, 0xfe, 0xff].map(|code| decode(code, FloatLayout::E5M2).to_bits());

        assert_eq!(widened, [0x7fc0_0000, 0x7fc0_0000, 0x7fc0_0000, 0xffc0_0000, 0xffc0_0000, 0xffc0_0000]);
    }

    #[test]
    fn binary32_identity_holds_for_every_rounding_and_overflow_argument() {
        let signalling = f32::from_bits(0x7f80_0001);

        for rounding in [Rounding::NearestTiesEven, Rounding::TowardZero, Rounding::AwayFromZero] {
            for overflow in [Overflow::Saturate, Overflow::Format] {
                let code = encode_with(signalling, FloatLayout::BF32, rounding, overflow, &mut 0);
                assert_eq!(code, 0x7f80_0001, "{rounding:?} {overflow:?}");
            }
        }
    }

    #[test]
    fn narrowing_binary32_through_binary32_is_the_identity_under_a_non_default_rounding() {
        let signalling = f32::from_bits(0x7f80_0001);

        let narrowed = crate::convert::narrow_to_layout(signalling, FloatLayout::BF32, Rounding::TowardZero, Overflow::Saturate)
            .expect("binary32 is within the codec domain");

        assert_eq!(narrowed.to_bits(), 0x7f80_0001);
    }

    #[test]
    fn a_wrong_sized_output_is_a_fault_not_a_panic() {
        let mut out = [0u8; 3];

        let fault = encode_slice(&FloatLayout::BF16, &[1.0, 2.0], &mut out);

        assert_eq!(fault, Err(SliceFault::Output { found: 3, expected: 4 }));
    }

    #[test]
    fn a_layout_outside_the_codec_domain_is_refused_not_instantiated() {
        let mut out = [0u8; 16];

        let refused = encode_slice(&FloatLayout::INT64, &[1.0, 2.0], &mut out);

        assert!(refused.is_err());
    }
}
