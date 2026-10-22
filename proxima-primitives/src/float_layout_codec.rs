//! Generic scalar float codec over [`FloatLayout`]: `f32 <-> code` by integer field arithmetic.
//!
//! Teaching pointer: [`FloatLayout`] (in `proxima-primitives`) is the data; this module is the one
//! piece of float arithmetic, and `bf16`, `bf8_e5m2` and `bf4_e2m1` are instances of it. Round to nearest with
//! ties to the even code, no `f64`, no allocation. Layouts are at most 32 bits wide with at most
//! 23 mantissa bits, so every intermediate fits in `u32`.

use crate::float_layout::{FloatLayout, SignEncoding, Specials};

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

const fn nan_payload(fraction: u32, layout: &FloatLayout) -> u32 {
    (fraction >> (F32_FRACTION_BITS - layout.mantissa_bits as u32)) | quiet_bit(layout)
}

const fn round_shift_ties_even(value: u32, shift: u32) -> u32 {
    if shift == 0 {
        return value;
    }
    if shift > F32_FRACTION_BITS + 1 {
        return 0;
    }
    let quotient = value >> shift;
    let remainder = value & ((1u32 << shift) - 1);
    let halfway = 1u32 << (shift - 1);
    if remainder > halfway || (remainder == halfway && quotient & 1 == 1) {
        quotient + 1
    } else {
        quotient
    }
}

const fn finite_magnitude_code(abs_bits: u32, layout: &FloatLayout) -> u32 {
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
        return overflow_code(layout);
    }
    let (base_exponent, extra_shift) = if target < 1 {
        (0, (1 - target) as u32)
    } else {
        ((target - 1) as u32, 0)
    };
    let shift = F32_FRACTION_BITS - layout.mantissa_bits as u32 + extra_shift;
    let code = (base_exponent << layout.mantissa_bits) + round_shift_ties_even(significand, shift);
    let limit = overflow_code(layout);
    if code > limit { limit } else { code }
}

const fn magnitude_code(abs_bits: u32, layout: &FloatLayout) -> u32 {
    if abs_bits >> F32_FRACTION_BITS == 0xff {
        overflow_code(layout)
    } else {
        finite_magnitude_code(abs_bits, layout)
    }
}

const fn nan_code(bits: u32, layout: &FloatLayout, nan_count: &mut u64) -> u32 {
    match layout.specials {
        Specials::None => {
            *nan_count += 1;
            max_normal_code(layout)
        }
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

const fn round_magnitude_ties_even(magnitude: f32) -> u64 {
    let truncated = magnitude as u64;
    let fraction = magnitude - truncated as f32;
    if fraction > 0.5 || (fraction == 0.5 && truncated & 1 == 1) {
        truncated.saturating_add(1)
    } else {
        truncated
    }
}

const fn integer_encode(value: f32, layout: &FloatLayout) -> u32 {
    if value.is_nan() {
        return 0;
    }
    let scaled = scale_by_power_of_two(value, -integer_scale_exponent(layout));
    let negative = scaled.is_sign_negative();
    let magnitude = round_magnitude_ties_even(if negative { -scaled } else { scaled });
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
pub const fn encode_counting(value: f32, layout: FloatLayout, nan_count: &mut u64) -> u32 {
    if layout.exponent_bits == 0 {
        if value.is_nan() {
            *nan_count += 1;
        }
        return integer_encode(value, &layout);
    }
    if value.is_nan() {
        return nan_code(value.to_bits(), &layout, nan_count);
    }
    let bits = value.to_bits();
    let sign = (bits >> 31) << sign_shift(&layout);
    sign | magnitude_code(bits & 0x7fff_ffff, &layout)
}

/// Encodes `value` into `layout`; the same rounding as [`encode_counting`] without the counter.
#[must_use]
pub const fn encode(value: f32, layout: FloatLayout) -> u32 {
    let mut ignored_nan_count = 0;
    encode_counting(value, layout, &mut ignored_nan_count)
}

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
/// A reserved NaN code keeps its payload, shifted to the top of the `f32` fraction, and widens quiet.
#[must_use]
pub const fn decode(code: u32, layout: FloatLayout) -> f32 {
    if layout.exponent_bits == 0 {
        return integer_decode(code, &layout);
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
    let magnitude_bits = match (reserved, mantissa) {
        (true, 0) => F32_INFINITY_BITS,
        (true, _) => {
            F32_INFINITY_BITS
                | F32_QUIET_BIT
                | (mantissa << (F32_FRACTION_BITS - layout.mantissa_bits as u32))
        }
        (false, _) => finite_magnitude_bits(exponent, mantissa, &layout),
    };
    let sign = (code >> sign_shift(&layout)) & 1;
    f32::from_bits(magnitude_bits | (sign << 31))
}
