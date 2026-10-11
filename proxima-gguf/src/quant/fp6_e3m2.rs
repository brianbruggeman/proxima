//! Finite-only E3M2 conversion.
//!
//! Teaching pointer: wraps the generic [`float_layout`] codec with [`FloatLayout::E3M2`]; only the
//! non-finite rejection lives here. Packed storage is [`crate::quant::cold_float`].

use proxima_primitives::float_layout::FloatLayout;

use crate::quant::QuantError;
use crate::quant::float_layout;

const CODEC: &str = "fp6_e3m2";
const CODE_MASK: u8 = 0x3f;

/// Encodes one value as sign(1), exponent(3), mantissa(2), bias 3.
///
/// Finite values saturate to the finite range and round to nearest, ties to
/// even. Non-finite inputs are rejected because the format has no NaN/Inf.
pub fn encode(value: f32) -> Result<u8, QuantError> {
    if !value.is_finite() {
        return Err(QuantError::NonFiniteInput { codec: CODEC });
    }
    Ok(float_layout::encode(value, FloatLayout::E3M2) as u8)
}

/// Decodes one six-bit E3M2 value. The upper two bits are ignored.
#[must_use]
pub fn decode(encoded: u8) -> f32 {
    float_layout::decode(u32::from(encoded & CODE_MASK), FloatLayout::E3M2)
}

#[cfg(test)]
mod tests {
    use super::{decode, encode};
    use crate::quant::QuantError;

    #[test]
    fn e3m2_golden_values_and_rounding_ties_are_explicit() {
        assert_eq!(encode(0.0), Ok(0x00));
        assert_eq!(encode(1.0), Ok(0x0c));
        assert_eq!(encode(28.0), Ok(0x1f));
        assert_eq!(encode(-1.0), Ok(0x2c));
        assert_eq!(encode(1.125), Ok(0x0c));
        assert_eq!(encode(1.375), Ok(0x0e));
        assert_eq!(decode(0x1f), 28.0);
        assert_eq!(
            encode(f32::INFINITY),
            Err(QuantError::NonFiniteInput { codec: "fp6_e3m2" })
        );
        assert_eq!(encode(1000.0), Ok(0x1f));
        assert_eq!(encode(-1000.0), Ok(0x3f));
    }

    #[test]
    fn every_e3m2_codeword_decodes_and_reencodes_identically() {
        for codeword in 0..64_u8 {
            assert_eq!(
                encode(decode(codeword)),
                Ok(codeword),
                "codeword {codeword:#04x}"
            );
        }
    }

    #[test]
    fn e3m2_midpoint_ties_select_the_even_codeword() {
        for lower_code in 0..31_u8 {
            let lower = decode(lower_code);
            let upper = decode(lower_code + 1);
            let midpoint = lower + (upper - lower) * 0.5;
            let expected = if lower_code & 1 == 0 {
                lower_code
            } else {
                lower_code + 1
            };
            assert_eq!(encode(midpoint), Ok(expected));
        }
    }
}
