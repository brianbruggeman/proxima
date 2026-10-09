//! Proxima's scalar E5M2 brain-float representation.

/// One E5M2 byte using the scalar contract in the Granite numeric matrix.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct BFloat8(u8);

impl BFloat8 {
    /// Constructs a value from its stored E5M2 bits.
    #[must_use]
    pub const fn from_bits(bits: u8) -> Self {
        Self(bits)
    }

    /// Returns the stored E5M2 bits.
    #[must_use]
    pub const fn to_bits(self) -> u8 {
        self.0
    }

    /// Encodes an F32 value using round-to-nearest, ties-to-even.
    #[must_use]
    pub fn from_f32(value: f32) -> Self {
        Self(encode_f32_bits(value.to_bits()))
    }

    /// Decodes this E5M2 value exactly into F32.
    #[must_use]
    pub fn to_f32(self) -> f32 {
        f32::from_bits(decode_f32_bits(self.0))
    }
}

fn round_shift_to_even(significand: u32, shift: u32) -> u32 {
    if shift == 0 {
        return significand;
    }
    if shift > 24 {
        return 0;
    }

    let quotient = significand >> shift;
    let remainder = significand & ((1 << shift) - 1);
    let halfway = 1 << (shift - 1);
    if remainder > halfway || (remainder == halfway && quotient & 1 != 0) {
        quotient + 1
    } else {
        quotient
    }
}

fn encode_f32_bits(source: u32) -> u8 {
    let sign = ((source >> 24) & 0x80) as u8;
    let exponent = (source >> 23) & 0xff;
    let mantissa = source & 0x7f_ffff;

    if exponent == 0xff {
        return if mantissa == 0 {
            sign | 0x7c
        } else {
            sign | 0x7e
        };
    }
    if exponent == 0 && mantissa == 0 {
        return sign;
    }

    let (significand, binary_power, leading_exponent) = if exponent == 0 {
        (
            mantissa,
            -149,
            (u32::BITS - mantissa.leading_zeros() - 1) as i32 - 149,
        )
    } else {
        (
            0x80_0000 | mantissa,
            exponent as i32 - 150,
            exponent as i32 - 127,
        )
    };

    if leading_exponent < -14 {
        let shift = -(binary_power + 16) as u32;
        let units = round_shift_to_even(significand, shift);
        return sign | units.min(4) as u8;
    }

    let mut rounded_significand = round_shift_to_even(significand, 21);
    let mut rounded_exponent = leading_exponent;
    if rounded_significand == 8 {
        rounded_exponent += 1;
        rounded_significand = 4;
    }
    if rounded_exponent > 15 {
        return sign | 0x7c;
    }

    sign | (((rounded_exponent + 15) as u8) << 2) | (rounded_significand - 4) as u8
}

fn decode_f32_bits(source: u8) -> u32 {
    let sign = u32::from(source & 0x80) << 24;
    let exponent = (source >> 2) & 0x1f;
    let mantissa = u32::from(source & 0x03);

    match exponent {
        0 if mantissa == 0 => sign,
        0 => {
            let leading_bit = u32::from(mantissa >= 2);
            let f32_exponent = leading_bit + 111;
            let f32_mantissa = (mantissa - (1 << leading_bit)) << (23 - leading_bit);
            sign | (f32_exponent << 23) | f32_mantissa
        }
        31 => sign | 0x7f80_0000 | (mantissa << 21),
        _ => sign | ((u32::from(exponent) + 112) << 23) | (mantissa << 21),
    }
}

#[cfg(test)]
mod tests {
    use super::BFloat8;
    use crate::convert::Convert;
    use proxima_primitives::block_on;
    use proxima_primitives::pipe::Pipe;

    #[test]
    fn card_01_bf8_convert_decodes_every_byte_exactly() {
        for bits in u8::MIN..=u8::MAX {
            let sign = u32::from(bits & 0x80) << 24;
            let exponent = (bits >> 2) & 0x1f;
            let mantissa = u32::from(bits & 0x03);
            let expected = match exponent {
                0 if mantissa == 0 => sign,
                0 => {
                    let leading_bit = u32::from(mantissa >= 2);
                    sign | ((leading_bit + 111) << 23)
                        | ((mantissa - (1 << leading_bit)) << (23 - leading_bit))
                }
                31 => sign | 0x7f80_0000 | (mantissa << 21),
                _ => sign | ((u32::from(exponent) + 112) << 23) | (mantissa << 21),
            };
            assert_eq!(
                BFloat8::from_bits(bits).to_f32().to_bits(),
                expected,
                "byte {bits:#04x}"
            );
        }
    }

    #[test]
    fn card_01_bf8_convert_matches_specified_golden_vectors() {
        let fixture = include_str!("../specs/granite-attention-numeric-matrix/bf8_vectors.csv");
        let mut lines = fixture.lines();
        assert_eq!(lines.next(), Some("case,f32_bits,bf8_bits"));

        let mut vector_count = 0;
        for line in lines {
            let mut fields = line.split(',');
            let case_name = fields.next().expect("case field is present");
            let source_bits = fields
                .next()
                .expect("F32 bits field is present")
                .trim_start_matches("0x");
            let expected_bits = fields
                .next()
                .expect("BF8 bits field is present")
                .trim_start_matches("0x");
            assert!(fields.next().is_none(), "unexpected field for {case_name}");

            let source =
                f32::from_bits(u32::from_str_radix(source_bits, 16).expect("valid F32 bits"));
            let expected = u8::from_str_radix(expected_bits, 16).expect("valid BF8 bits");
            let encoded = block_on(Convert::<f32, BFloat8>::new().call(source))
                .expect("F32 to BF8 conversion is infallible");
            assert_eq!(encoded.to_bits(), expected, "case {case_name}");
            let decoded =
                block_on(Convert::<BFloat8, f32>::new().call(BFloat8::from_bits(expected)))
                    .expect("BF8 to F32 conversion is infallible");
            assert_eq!(
                decoded.to_bits(),
                BFloat8::from_bits(expected).to_f32().to_bits(),
                "decoded case {case_name}"
            );
            vector_count += 1;
        }
        assert_eq!(vector_count, 16);
    }
}
