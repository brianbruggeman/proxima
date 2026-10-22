//! A float layout is a tuple of exponent width, mantissa width and special-value policy; the
//! scalar codecs `proxima-gguf/src/quant/bf8_e5m2.rs` and `bf4_e2m1.rs` will instantiate it
//! instead of each hand-coding the same arithmetic.

/// How a layout treats the all-ones exponent and finite overflow.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Specials {
    /// No inf or nan encodings; the all-ones exponent is a normal value and overflow saturates
    /// to the signed max normal (E3M2, E2M1).
    None,
    /// IEEE inf and nan encodings are reserved; finite overflow and inf input clamp to the max
    /// normal, nan still encodes as nan (E5M2_SATURATING, the GGUF bf8 storage policy).
    SaturateOnOverflow,
    /// IEEE inf and nan encodings are reserved; overflow rounds to inf (binary32, BF16).
    InfinityOnOverflow,
}

/// How the sign of a stored number is encoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum SignEncoding {
    /// No sign bit: the layout is unsigned and `width()` has no sign bit (`uint`).
    Absent,
    /// A sign bit above the other fields selects the sign of the magnitude, so zero has two
    /// encodings (every IEEE-family float; sign-magnitude integers).
    Magnitude,
    /// A sign bit of weight `-2^(mantissa_bits)` over the mantissa field: the range is
    /// asymmetric and zero has one encoding (`int8` and the other native integers).
    TwosComplement,
}

/// A sign encoding, `exponent_bits`, `mantissa_bits`; numeric layout metadata can describe up to
/// 128-bit scalars, while the current generic scalar codecs execute widths up to 32 bits. An integer is the
/// layout with `exponent_bits` 0 and a `bias_override` of `1 - mantissa_bits`, which makes its
/// scale exactly 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct FloatLayout {
    /// Width of the exponent field; 3 for E3M2, 8 for binary32.
    pub exponent_bits: u8,
    /// Width of the stored mantissa field, excluding the implicit leading one; 2 for E3M2, 23
    /// for binary32.
    pub mantissa_bits: u8,
    /// Policy for the all-ones exponent and for overflow; `Specials::None` for E3M2.
    pub specials: Specials,
    /// Replaces the IEEE bias `2^(exponent_bits - 1) - 1` when set; `Some(4)` on E3M2 shifts
    /// its max normal from 28.0 to 14.0.
    pub bias_override: Option<i16>,
    /// How the sign is encoded; `Magnitude` for IEEE floats, `TwosComplement` for `int8`.
    pub sign: SignEncoding,
}

const fn pow2(exponent: i32) -> f32 {
    match exponent {
        -126..=127 => f32::from_bits(((exponent + 127) as u32) << 23),
        -149..=-127 => f32::from_bits(1 << (exponent + 149)),
        ..=-150 => 0.0,
        _ => f32::INFINITY,
    }
}

impl FloatLayout {
    pub const E5M2: Self = Self::ieee_like(5, 2, Specials::InfinityOnOverflow);
    /// E5M2 widths with saturation: the GGUF bf8 storage policy, which keeps infinities out of stored weights.
    pub const E5M2_SATURATING: Self = Self::ieee_like(5, 2, Specials::SaturateOnOverflow);
    /// IEEE 754 binary16: bias 15, inf and nan reserved, max normal 65504.0.
    pub const F16: Self = Self::ieee_like(5, 10, Specials::InfinityOnOverflow);
    pub const E2M1: Self = Self::ieee_like(2, 1, Specials::None);
    /// Proxima cold-storage E2M2: bias 1, no inf or nan, max normal 7.0.
    pub const E2M2: Self = Self::ieee_like(2, 2, Specials::None);
    /// OCP MX v1.0 FP6 E3M2: bias 3, no inf or nan, max normal 28.0 (page 12, section 5.3.2, Table 4).
    pub const E3M2: Self = Self::ieee_like(3, 2, Specials::None);
    pub const E2M3: Self = Self::ieee_like(2, 3, Specials::None);
    pub const BF32: Self = Self::ieee_like(8, 23, Specials::InfinityOnOverflow);
    pub const BF16: Self = Self::ieee_like(8, 7, Specials::InfinityOnOverflow);
    pub const BF8: Self = Self::E5M2;
    pub const BF4: Self = Self::E2M1;
    pub const INT8: Self = Self::integer(8, SignEncoding::TwosComplement);
    pub const UINT8: Self = Self::integer(8, SignEncoding::Absent);
    pub const INT16: Self = Self::integer(16, SignEncoding::TwosComplement);
    pub const UINT16: Self = Self::integer(16, SignEncoding::Absent);
    pub const INT32: Self = Self::integer(32, SignEncoding::TwosComplement);
    pub const UINT32: Self = Self::integer(32, SignEncoding::Absent);
    pub const INT64: Self = Self::integer(64, SignEncoding::TwosComplement);
    pub const UINT64: Self = Self::integer(64, SignEncoding::Absent);
    pub const INT128: Self = Self::integer(128, SignEncoding::TwosComplement);
    pub const UINT128: Self = Self::integer(128, SignEncoding::Absent);
    pub const F64: Self = Self::ieee_like(11, 52, Specials::InfinityOnOverflow);
    pub const UINT1: Self = Self::integer(1, SignEncoding::Absent);
    pub const UINT2: Self = Self::integer(2, SignEncoding::Absent);
    pub const UINT3: Self = Self::integer(3, SignEncoding::Absent);
    pub const UINT4: Self = Self::integer(4, SignEncoding::Absent);
    pub const UINT5: Self = Self::integer(5, SignEncoding::Absent);
    pub const UINT6: Self = Self::integer(6, SignEncoding::Absent);
    /// Sign-magnitude eight bits: range -127..=127 with a negative zero.
    pub const SIGN_MAGNITUDE_INT8: Self = Self::integer(8, SignEncoding::Magnitude);

    const fn ieee_like(exponent_bits: u8, mantissa_bits: u8, specials: Specials) -> Self {
        Self {
            exponent_bits,
            mantissa_bits,
            specials,
            bias_override: None,
            sign: SignEncoding::Magnitude,
        }
    }

    const fn integer(width: u8, sign: SignEncoding) -> Self {
        let mantissa_bits = match sign {
            SignEncoding::Absent => width,
            SignEncoding::Magnitude | SignEncoding::TwosComplement => width - 1,
        };
        Self {
            exponent_bits: 0,
            mantissa_bits,
            specials: Specials::None,
            bias_override: Some(1 - mantissa_bits as i16),
            sign,
        }
    }

    #[must_use]
    pub const fn bias(&self) -> i16 {
        match self.bias_override {
            Some(bias) => bias,
            None => (1i16 << self.exponent_bits.saturating_sub(1)) - 1,
        }
    }

    #[must_use]
    pub const fn width(&self) -> u8 {
        let sign_bits = match self.sign {
            SignEncoding::Absent => 0,
            SignEncoding::Magnitude | SignEncoding::TwosComplement => 1,
        };
        sign_bits + self.exponent_bits + self.mantissa_bits
    }

    /// Whether two layouts are the same format: equal exponent and mantissa widths, equal
    /// specials, and equal effective bias. `==` also compares the raw `bias_override`, so a
    /// layout spelled with `Some(15)` and one with `None` differ under `==` and agree here.
    #[must_use]
    pub fn same_format(&self, other: &Self) -> bool {
        self.exponent_bits == other.exponent_bits
            && self.mantissa_bits == other.mantissa_bits
            && self.specials == other.specials
            && self.bias() == other.bias()
            && self.sign == other.sign
    }

    /// Bytes `elements` values occupy packed densely at `width()` bits each, plus one `scale` code
    /// per `block` elements when `scale_block` is `Some((scale, block))`, rounded up to whole
    /// bytes: 32 elements of E3M2 under an f16 scale are 32 * 6 + 16 = 208 bits, 26 bytes.
    #[must_use]
    pub const fn packed_bytes(
        &self,
        elements: u64,
        scale_block: Option<(FloatLayout, u64)>,
    ) -> u64 {
        let element_bits = elements * self.width() as u64;
        let scale_bits = match scale_block {
            Some((scale, block)) if block > 0 => elements.div_ceil(block) * scale.width() as u64,
            _ => 0,
        };
        (element_bits + scale_bits).div_ceil(8)
    }

    /// Binary32 projection of the maximum normal float; not an exact range for wider formats or
    /// an integer bound.
    #[must_use]
    pub const fn max_normal(&self) -> f32 {
        let exponent_codes = (1i32 << self.exponent_bits) - 1;
        let top_code = match self.specials {
            Specials::None => exponent_codes,
            Specials::SaturateOnOverflow | Specials::InfinityOnOverflow => exponent_codes - 1,
        };
        let significand = 2.0 - pow2(-(self.mantissa_bits as i32));
        significand * pow2(top_code - self.bias() as i32)
    }

    /// Binary32 projection of the minimum normal float; wider formats can underflow to zero.
    #[must_use]
    pub const fn min_normal(&self) -> f32 {
        pow2(1 - self.bias() as i32)
    }

    /// Binary32 projection of the minimum subnormal float; wider formats can underflow to zero.
    #[must_use]
    pub const fn min_subnormal(&self) -> f32 {
        pow2(1 - self.bias() as i32 - self.mantissa_bits as i32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CASES: [(FloatLayout, i16, u8, f32, f32, f32); 6] = [
        (
            FloatLayout::E5M2,
            15,
            8,
            57344.0,
            6.103_515_6e-5,
            1.525_878_9e-5,
        ),
        (FloatLayout::E2M1, 1, 4, 6.0, 1.0, 0.5),
        (FloatLayout::E3M2, 3, 6, 28.0, 0.25, 0.0625),
        (FloatLayout::E2M3, 1, 6, 7.5, 1.0, 0.125),
        (
            FloatLayout::BF16,
            127,
            16,
            3.389_531_4e38,
            f32::MIN_POSITIVE,
            f32::from_bits(1 << 16),
        ),
        (
            FloatLayout::BF32,
            127,
            32,
            f32::MAX,
            f32::MIN_POSITIVE,
            1.0e-45,
        ),
    ];

    #[test]
    fn float_layout_derived_bias_per_layout() {
        let biases = CASES.map(|case| case.0.bias());
        assert_eq!(biases, CASES.map(|case| case.1));
    }

    #[test]
    fn float_layout_derived_width_per_layout() {
        let widths = CASES.map(|case| case.0.width());
        assert_eq!(widths, CASES.map(|case| case.2));
    }

    #[test]
    fn float_layout_derived_max_normal_per_layout() {
        let maxima = CASES.map(|case| case.0.max_normal());
        assert_eq!(maxima, CASES.map(|case| case.3));
    }

    #[test]
    fn float_layout_derived_min_normal_per_layout() {
        let minima = CASES.map(|case| case.0.min_normal());
        assert_eq!(minima, CASES.map(|case| case.4));
    }

    #[test]
    fn float_layout_derived_min_subnormal_per_layout() {
        let minima = CASES.map(|case| case.0.min_subnormal());
        assert_eq!(minima, CASES.map(|case| case.5));
    }

    #[test]
    fn float_layout_derived_bias_override_shifts_range() {
        let shifted = FloatLayout {
            bias_override: Some(4),
            ..FloatLayout::E3M2
        };

        assert_eq!(shifted.bias(), 4);
        assert_eq!(shifted.max_normal(), 14.0);
    }

    #[test]
    fn float_layout_constants_bf8_equals_e5m2() {
        assert_eq!(FloatLayout::BF8, FloatLayout::E5M2);
    }

    #[test]
    fn float_layout_constants_bf4_equals_e2m1() {
        assert_eq!(FloatLayout::BF4, FloatLayout::E2M1);
    }

    #[test]
    fn float_layout_constants_bf32_is_e8m23() {
        let layout = FloatLayout::BF32;

        assert_eq!((layout.exponent_bits, layout.mantissa_bits), (8, 23));
        assert_eq!(layout.max_normal(), f32::MAX);
        assert_eq!(layout.min_subnormal().to_bits(), 1);
    }

    #[test]
    fn float_layout_constants_f16_is_e5m10() {
        let layout = FloatLayout::F16;

        assert_eq!((layout.exponent_bits, layout.mantissa_bits), (5, 10));
        assert_eq!(layout.max_normal(), 65504.0);
        assert_eq!(layout.min_subnormal(), 2.0f32.powi(-24));
    }

    #[test]
    fn float_layout_constants_bf16_is_e8m7() {
        let layout = FloatLayout::BF16;

        assert_eq!((layout.exponent_bits, layout.mantissa_bits), (8, 7));
        assert_eq!(layout.max_normal().to_bits(), 0x7f7f_0000);
    }

    #[test]
    fn float_layout_constants_specials_and_widths_per_layout() {
        let observed = [
            FloatLayout::E5M2,
            FloatLayout::BF8,
            FloatLayout::E5M2_SATURATING,
            FloatLayout::BF16,
            FloatLayout::BF32,
            FloatLayout::E3M2,
            FloatLayout::E2M3,
            FloatLayout::E2M1,
            FloatLayout::BF4,
        ]
        .map(|layout| (layout.specials, layout.width()));

        let saturate = Specials::SaturateOnOverflow;
        let infinity = Specials::InfinityOnOverflow;
        let expected = [
            (infinity, 8),
            (infinity, 8),
            (saturate, 8),
            (infinity, 16),
            (infinity, 32),
            (Specials::None, 6),
            (Specials::None, 6),
            (Specials::None, 4),
            (Specials::None, 4),
        ];
        assert_eq!(observed, expected);
    }
}
