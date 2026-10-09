//! Scalar E5M2 brain-float conversion used by the low-precision training path.

/// The E5M2 encoding for one value.
#[must_use]
pub fn encode(value: f32) -> u8 {
    let bits = value.to_bits();
    let sign = ((bits >> 24) & 0x80) as u8;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let fraction = bits & 0x7f_ffff;

    if exponent == 0xff {
        return if fraction == 0 { sign | 0x7b } else { 0x7e };
    }

    if exponent == 0 {
        return sign;
    }

    let unbiased_exponent = exponent - 127;
    if unbiased_exponent > 15 {
        return sign | 0x7b;
    }

    if unbiased_exponent < -14 {
        let shift = (7 - unbiased_exponent) as u32;
        let significand = (1 << 23) | fraction;
        let rounded = round_shift_ties_even(significand, shift);
        return sign | (rounded.min(4) as u8);
    }

    let rounded_fraction = round_shift_ties_even(fraction, 21);
    let mut target_exponent = unbiased_exponent + 15;
    let mut target_fraction = rounded_fraction;
    if target_fraction == 4 {
        target_fraction = 0;
        target_exponent += 1;
    }
    if target_exponent >= 31 {
        return sign | 0x7b;
    }
    sign | ((target_exponent as u8) << 2) | (target_fraction as u8)
}

/// Decodes one E5M2 byte, including its infinity and NaN patterns.
#[must_use]
pub fn decode(encoded: u8) -> f32 {
    let sign = if encoded & 0x80 == 0 { 1.0 } else { -1.0 };
    let exponent = (encoded >> 2) & 0x1f;
    let fraction = encoded & 0x03;

    if exponent == 0x1f {
        return if fraction == 0 {
            sign * f32::INFINITY
        } else {
            f32::NAN
        };
    }

    if exponent == 0 {
        return sign * f32::from(fraction) * f32::from_bits(0x3780_0000);
    }

    let significand = 1.0 + f32::from(fraction) * 0.25;
    sign * significand * power_of_two(i32::from(exponent) - 15)
}

fn power_of_two(exponent: i32) -> f32 {
    f32::from_bits(((exponent + 127) as u32) << 23)
}

fn round_shift_ties_even(value: u32, shift: u32) -> u32 {
    if shift == 0 {
        return value;
    }
    if shift > 32 {
        return 0;
    }
    if shift == 32 {
        return u32::from(value > (1 << 31));
    }

    let quotient = value >> shift;
    let remainder_mask = (1 << shift) - 1;
    let remainder = value & remainder_mask;
    let halfway = 1 << (shift - 1);
    quotient + u32::from(remainder > halfway || (remainder == halfway && quotient & 1 == 1))
}
