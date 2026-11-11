//! A float layout is a tuple of exponent width, mantissa width and special-value policy; the
//! shared scalar codec instantiates formats such as E5M2, E4M3FN and E2M1 without per-format math.
//!
//! Teaching pointer: [`FloatLayout`] is this workspace's `llvm::fltSemantics` — precision plus
//! exponent range, with the special-value policy and the bias as separate decisions rather than
//! baked into an enum. [`crate::float_layout_codec`] is the one piece of arithmetic over it.

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
    /// The top mantissa code under the all-ones exponent is NaN; other codes under that exponent
    /// remain finite (OCP E4M3FN), and overflow saturates to the largest finite value.
    NanOnly,
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

/// What a NaN becomes when `f32` is narrowed into a layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, serde::Serialize, serde::Deserialize)]
pub enum NanNarrow {
    /// Sign, top payload bits and a set quiet bit survive (IEEE 754-2019 6.2.3, "preserve the
    /// payload to the extent possible"; `half` `f32_to_bf16` convert.rs:13, `f32_to_f16_fallback`
    /// binary16/arch.rs:569).
    #[default]
    Preserve,
    /// The canonical quiet NaN with an empty payload (ml_dtypes `float8.h:1010` `quiet_NaN()` for
    /// E5M2, `0b0_11111_10`, IEEE 754-2019 6.2.1: the first trailing-significand bit is the quiet bit).
    Canonical,
}

/// What a reserved NaN code becomes when a layout is widened to `f32`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, serde::Serialize, serde::Deserialize)]
pub enum NanWiden {
    /// The payload is kept and the quiet bit is set (`half` `bf16_to_f32` convert.rs:96-98,
    /// `f16_to_f32_fallback` binary16/arch.rs:702-704).
    #[default]
    Quiet,
    /// The canonical quiet NaN `0x7fc0_0000` with the sign kept and the payload dropped (onnxruntime
    /// `include/onnxruntime/core/common/float8.h:454-462`, `Float8E5M2::ToFloat`: codes 125..=127 give
    /// `0x7fc0_0000`, codes from 253 give `0xffc0_0000`).
    Canonical,
}

/// The NaN payload policy of one layout: the data the codec reads instead of a threshold on width.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, serde::Serialize, serde::Deserialize)]
pub struct NanPolicy {
    /// What narrowing `f32` into the layout does to a NaN.
    pub narrow: NanNarrow,
    /// What widening the layout's NaN code to `f32` does to its payload.
    pub widen: NanWiden,
}

impl NanPolicy {
    /// IEEE 754-2019 6.2.3 on both directions: binary16, bfloat16 and the layouts without a reference.
    pub const IEEE: Self = Self {
        narrow: NanNarrow::Preserve,
        widen: NanWiden::Quiet,
    };
    /// The float8 reference: canonical quiet NaN out, canonical quiet NaN in. Widening follows
    /// onnxruntime `float8.h:454-462` (`Float8E5M2::ToFloat`, sign kept, payload dropped); narrowing
    /// stays at ml_dtypes' `0b0_11111_10` because onnxruntime `float8.h:398-399` narrows to `sign | 0x7f`
    /// and the two external sources disagree there.
    pub const FLOAT8: Self = Self {
        narrow: NanNarrow::Canonical,
        widen: NanWiden::Canonical,
    };
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
    /// NaN payload policy for narrowing into and widening out of the layout; `NanPolicy::FLOAT8`
    /// for E5M2, `NanPolicy::IEEE` for binary16.
    #[serde(default)]
    pub nan: NanPolicy,
}

const fn specials_index(specials: Specials) -> u8 {
    match specials {
        Specials::None => 0,
        Specials::SaturateOnOverflow => 1,
        Specials::InfinityOnOverflow => 2,
        Specials::NanOnly => 3,
    }
}

const fn sign_index(sign: SignEncoding) -> u8 {
    match sign {
        SignEncoding::Absent => 0,
        SignEncoding::Magnitude => 1,
        SignEncoding::TwosComplement => 2,
    }
}

const fn specials_equal(left: Specials, right: Specials) -> bool {
    specials_index(left) == specials_index(right)
}

const fn sign_equal(left: SignEncoding, right: SignEncoding) -> bool {
    sign_index(left) == sign_index(right)
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
    pub const E5M2: Self =
        Self::ieee_like(5, 2, Specials::InfinityOnOverflow).with_nan(NanPolicy::FLOAT8);
    /// OCP E4M3FN: bias 7, one NaN code per sign, no infinities, max finite 448.
    pub const E4M3FN: Self = Self::ieee_like(4, 3, Specials::NanOnly);
    /// E5M2 widths with saturation: the GGUF bf8 storage policy, which keeps infinities out of stored weights.
    pub const E5M2_SATURATING: Self =
        Self::ieee_like(5, 2, Specials::SaturateOnOverflow).with_nan(NanPolicy::FLOAT8);
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
    /// OCP MX v1.0 E8M0 block scale: eight exponent bits, no mantissa and no sign bit; the all-ones
    /// code is NaN and every other code is `2^(code - 127)`.
    pub const F8_E8M0: Self = Self {
        exponent_bits: 8,
        mantissa_bits: 0,
        specials: Specials::NanOnly,
        bias_override: None,
        sign: SignEncoding::Absent,
        nan: NanPolicy::IEEE,
    };

    const fn ieee_like(exponent_bits: u8, mantissa_bits: u8, specials: Specials) -> Self {
        Self {
            exponent_bits,
            mantissa_bits,
            specials,
            bias_override: None,
            sign: SignEncoding::Magnitude,
            nan: NanPolicy::IEEE,
        }
    }

    const fn with_nan(mut self, nan: NanPolicy) -> Self {
        self.nan = nan;
        self
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
            nan: NanPolicy::IEEE,
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
    pub const fn same_format(&self, other: &Self) -> bool {
        self.shares_exponent_with(other) && self.mantissa_bits == other.mantissa_bits
    }

    /// Whether converting between the two layouts changes no bit of a code: the same format and the
    /// same NaN policy in both directions, so a signalling NaN keeps its payload and its quiet bit.
    #[must_use]
    pub const fn is_identity_of(&self, other: &Self) -> bool {
        self.same_format(other)
            && self.nan.narrow as u8 == other.nan.narrow as u8
            && self.nan.widen as u8 == other.nan.widen as u8
    }

    /// Whether two layouts agree on everything but the mantissa width: the same exponent width,
    /// effective bias, special-value policy and sign encoding, so every exponent field and every
    /// special value means the same number in both. Only layouts with an exponent field can share one.
    #[must_use]
    pub const fn shares_exponent_with(&self, other: &Self) -> bool {
        self.exponent_bits == other.exponent_bits
            && self.bias() == other.bias()
            && specials_equal(self.specials, other.specials)
            && sign_equal(self.sign, other.sign)
    }

    /// Whether this layout is `wider` with trailing mantissa bits dropped: they share an exponent
    /// ([`Self::shares_exponent_with`]) and this one has no more mantissa bits. Widening to `wider`
    /// is then an exact bit embedding (a shift) and narrowing from it is one rounding of the dropped
    /// bits, with a carry into the exponent field already yielding the overflow value.
    #[must_use]
    pub const fn is_mantissa_prefix_of(&self, wider: &Self) -> bool {
        self.exponent_bits != 0
            && self.shares_exponent_with(wider)
            && self.mantissa_bits <= wider.mantissa_bits
    }

    /// A value that is equal for two layouts exactly when every field of the description is equal,
    /// `bias_override` and the NaN policy included. A `u64` so a dispatch over named layouts is one
    /// integer compare per row, which the compiler turns into a switch.
    #[must_use]
    pub const fn fingerprint(&self) -> u64 {
        let (bias_present, bias) = match self.bias_override {
            Some(bias) => (1u64, bias as u16 as u64),
            None => (0, 0),
        };
        (self.exponent_bits as u64)
            | (self.mantissa_bits as u64) << 8
            | (specials_index(self.specials) as u64) << 16
            | (sign_index(self.sign) as u64) << 18
            | (self.nan.narrow as u64) << 20
            | (self.nan.widen as u64) << 21
            | bias_present << 23
            | bias << 24
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
            Specials::None | Specials::NanOnly => exponent_codes,
            Specials::SaturateOnOverflow | Specials::InfinityOnOverflow => exponent_codes - 1,
        };
        let significand = match self.specials {
            // the top mantissa code under the all-ones exponent is the NaN code, so the largest
            // finite significand stops one code short of 2.0
            Specials::NanOnly if self.mantissa_bits == 0 => 1.0,
            Specials::NanOnly => 2.0 - 2.0 * pow2(-(self.mantissa_bits as i32)),
            _ => 2.0 - pow2(-(self.mantissa_bits as i32)),
        };
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

    /// The canonical lowercase name of exactly this layout. `None` for a layout nobody named.
    ///
    /// Exact equality, not [`Self::same_format`]: a name has to survive a round trip through
    /// [`Self::from_name`], and `==` compares the raw `bias_override`, so a layout spelled with an
    /// explicit bias that happens to equal the default is a different value and serializes
    /// structurally rather than collapsing onto the constant it agrees with numerically.
    #[must_use]
    pub fn name(&self) -> Option<&'static str> {
        NAMED_LAYOUTS
            .iter()
            .find_map(|(named, name)| (named == self).then_some(*name))
    }

    /// The name of the format this layout is numerically the same as, by [`Self::same_format`].
    /// For a reader-facing label; never for a round-tripped wire or config value.
    #[must_use]
    pub fn format_name(&self) -> Option<&'static str> {
        NAMED_LAYOUTS
            .iter()
            .find_map(|(named, name)| named.same_format(self).then_some(*name))
    }

    /// The layout a canonical name denotes; the inverse of [`Self::name`].
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        NAMED_LAYOUTS
            .iter()
            .find_map(|(layout, named)| (*named == name).then_some(*layout))
    }
}

/// The one list of named layouts, as constant names and spellings. A macro rather than a `const`
/// because the scalar codec instantiates itself once per row at compile time
/// ([`crate::float_layout_codec`]): `named_layout_table!(callback!(prefix...))` expands to
/// `callback! { prefix... [ (NAME, "spelling"), ... ] }`, so a new named format is one row here and
/// both [`NAMED_LAYOUTS`] and the codec's instantiations follow.
macro_rules! named_layout_table {
    ($callback:ident ! ( $($prefix:tt)* )) => {
        $callback! {
            $($prefix)*
            [
                (INT8, "int8"),
                (UINT8, "uint8"),
                (INT16, "int16"),
                (UINT16, "uint16"),
                (INT32, "int32"),
                (UINT32, "uint32"),
                (INT64, "int64"),
                (UINT64, "uint64"),
                (INT128, "int128"),
                (UINT128, "uint128"),
                (SIGN_MAGNITUDE_INT8, "sign_magnitude_int8"),
                (BF32, "float32"),
                (F64, "float64"),
                (F16, "float16"),
                (BF16, "bfloat16"),
                (E5M2_SATURATING, "bfloat8"),
                (E5M2, "float8_e5m2"),
                (E4M3FN, "float8_e4m3fn"),
                (E3M2, "fp6_e3m2"),
                (E2M3, "e2m3"),
                (E2M2, "e2m2"),
                (E2M1, "e2m1"),
                (UINT1, "uint1"),
                (UINT2, "uint2"),
                (UINT3, "uint3"),
                (UINT4, "uint4"),
                (UINT5, "uint5"),
                (UINT6, "uint6"),
                (F8_E8M0, "float8_e8m0"),
            ]
        }
    };
}

pub(crate) use named_layout_table;

macro_rules! named_layouts_array {
    ([$(($name:ident, $text:literal)),* $(,)?]) => {
        /// Every layout with a spelling. One table serves serde for both [`FloatLayout`] and
        /// [`crate::DType`], so a new named format is one row in `named_layout_table!` and nothing
        /// else.
        pub const NAMED_LAYOUTS: [(FloatLayout, &str); [$($text),*].len()] =
            [$((FloatLayout::$name, $text)),*];
    };
}

named_layout_table!(named_layouts_array!());

#[cfg(test)]
mod tests {
    use super::*;

    const CASES: [(FloatLayout, i16, u8, f32, f32, f32); 7] = [
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
        (FloatLayout::E4M3FN, 7, 8, 448.0, 0.015625, 0.001953125),
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
            FloatLayout::E4M3FN,
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
            (Specials::NanOnly, 8),
            (Specials::None, 4),
            (Specials::None, 4),
        ];
        assert_eq!(observed, expected);
    }
}
