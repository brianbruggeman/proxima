#[cfg(feature = "float-layout-dtype")]
use proxima_primitives::float_layout::FloatLayout;

#[cfg(all(feature = "float-layout-dtype", feature = "config"))]
use crate::dtype_spelling::DTypeSpelling;

#[cfg(not(feature = "float-layout-dtype"))]
/// Element type of a tensor operand.
///
/// Carried as a field on every [`Op`](crate::Op) rather than as a type
/// parameter. A single node routinely mixes three of these — quantized matmul
/// is `i8 x i8 -> i32 -> f32` — so one `T` could not describe it, and a leaf's
/// element type is not known until the weights are opened anyway.
///
/// Every variant is a fixed-width machine scalar, plain and fieldless.
/// Decimal is deliberately not here: it is not a machine scalar (its
/// arithmetic rescales rather than mapping onto one CPU instruction), and the
/// intended shape for it is an ordinary integer `DType` plus a separate
/// pipe-shaped conversion trait that carries the scale outside this enum —
/// not a payload-carrying variant of it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "config", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "config", serde(rename_all = "snake_case"))]
pub enum DType {
    Bool,
    Int8,
    UInt8,
    Int16,
    UInt16,
    Int32,
    UInt32,
    Int64,
    UInt64,
    Int128,
    UInt128,
    BFloat16,
    BFloat8,
    /// 6-bit E3M2 codes under a per-block scale; a block, not an element, is the
    /// addressable unit, so `size_bytes` is the whole-byte floor of one code.
    Fp6E3M2,
    Float16,
    Float32,
    Float64,
}

#[cfg(feature = "float-layout-dtype")]
/// Element type of a tensor operand.
///
/// Carried as a field on every [`Op`](crate::Op) rather than as a type
/// parameter. A single node routinely mixes three of these — quantized matmul
/// is `i8 x i8 -> i32 -> f32` — so one `T` could not describe it, and a leaf's
/// element type is not known until the weights are opened anyway.
///
/// Every float up to 32 bits is one [`Float`](Self::Float) variant carrying a
/// [`FloatLayout`]: the layout is an open set (E4M3, E2M1, any future format)
/// that a fieldless enum cannot name, so `Float32`, `Float16`, `BFloat16`,
/// `BFloat8` and `Fp6E3M2` are associated constants of that variant, not
/// variants. Float64 and the 64 and 128-bit integers stay named because the
/// generic codec's intermediates are 32 bits wide.
/// Decimal is deliberately not here: it is not a machine scalar, and the
/// intended shape for it is an ordinary integer `DType` plus a separate
/// pipe-shaped conversion trait that carries the scale outside this enum.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(
    feature = "config",
    derive(serde::Serialize, serde::Deserialize),
    serde(into = "DTypeSpelling", from = "DTypeSpelling")
)]
pub enum DType {
    Bool,
    Int8,
    UInt8,
    Int16,
    UInt16,
    Int32,
    UInt32,
    Int64,
    UInt64,
    Int128,
    UInt128,
    Float(FloatLayout),
    Float64,
}

#[cfg(feature = "float-layout-dtype")]
impl core::fmt::Debug for DType {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let name = match *self {
            Self::Bool => "Bool",
            Self::Int8 => "Int8",
            Self::UInt8 => "UInt8",
            Self::Int16 => "Int16",
            Self::UInt16 => "UInt16",
            Self::Int32 => "Int32",
            Self::UInt32 => "UInt32",
            Self::Int64 => "Int64",
            Self::UInt64 => "UInt64",
            Self::Int128 => "Int128",
            Self::UInt128 => "UInt128",
            Self::Float64 => "Float64",
            Self::Float(layout) => {
                return match Self::former_name(layout) {
                    Some(former) => formatter.write_str(former),
                    None => formatter.debug_tuple("Float").field(&layout).finish(),
                };
            }
        };
        formatter.write_str(name)
    }
}

#[cfg(feature = "float-layout-dtype")]
impl DType {
    /// The name a layout carried as a variant before it became a constant of [`Float`](Self::Float),
    /// by comparing the whole layout, never matching on a name.
    pub(crate) fn former_name(layout: FloatLayout) -> Option<&'static str> {
        [
            (FloatLayout::BF32, "Float32"),
            (FloatLayout::F16, "Float16"),
            (FloatLayout::BF16, "BFloat16"),
            (FloatLayout::E5M2_SATURATING, "BFloat8"),
            (FloatLayout::E3M2, "Fp6E3M2"),
        ]
        .into_iter()
        .find_map(|(named, name)| (named == layout).then_some(name))
    }
}

#[cfg(feature = "float-layout-dtype")]
// the former variant names stay as constants so expression sites keep reading the same
#[allow(non_upper_case_globals)]
impl DType {
    pub const Float32: Self = Self::Float(FloatLayout::BF32);
    pub const Float16: Self = Self::Float(FloatLayout::F16);
    pub const BFloat16: Self = Self::Float(FloatLayout::BF16);
    pub const BFloat8: Self = Self::Float(FloatLayout::E5M2_SATURATING);
    pub const Fp6E3M2: Self = Self::Float(FloatLayout::E3M2);
}

#[cfg(feature = "float-layout-dtype")]
impl DType {
    #[must_use]
    pub const fn size_bytes(self) -> usize {
        match self {
            Self::Bool | Self::Int8 | Self::UInt8 => 1,
            Self::Int16 | Self::UInt16 => 2,
            Self::Int32 | Self::UInt32 => 4,
            Self::Int64 | Self::UInt64 | Self::Float64 => 8,
            Self::Int128 | Self::UInt128 => 16,
            Self::Float(layout) => (layout.width() as usize).div_ceil(8),
        }
    }

    #[must_use]
    pub const fn bits(self) -> usize {
        match self {
            Self::Float(layout) => layout.width() as usize,
            other => other.size_bytes() * 8,
        }
    }

    #[must_use]
    pub const fn is_float(self) -> bool {
        matches!(self, Self::Float(_) | Self::Float64)
    }

    /// Whether a fold over this type can accumulate into itself without
    /// widening. Narrow integers cannot: a sum of `i8` overflows almost
    /// immediately, which is why quantized contraction accumulates in `Int32`.
    /// Every width from 32 bits up is wide enough to accumulate in place,
    /// signed or unsigned, integer or float; a float is wide enough from 16.
    #[must_use]
    pub const fn accumulates_in_place(self) -> bool {
        match self {
            Self::Int32
            | Self::UInt32
            | Self::Int64
            | Self::UInt64
            | Self::Int128
            | Self::UInt128
            | Self::Float64 => true,
            Self::Float(layout) => layout.width() >= 16,
            Self::Bool | Self::Int8 | Self::UInt8 | Self::Int16 | Self::UInt16 => false,
        }
    }
}

#[cfg(not(feature = "float-layout-dtype"))]
impl DType {
    #[must_use]
    pub const fn size_bytes(self) -> usize {
        match self {
            Self::Bool | Self::Int8 | Self::UInt8 | Self::BFloat8 | Self::Fp6E3M2 => 1,
            Self::Int16 | Self::UInt16 | Self::BFloat16 | Self::Float16 => 2,
            Self::Int32 | Self::UInt32 | Self::Float32 => 4,
            Self::Int64 | Self::UInt64 | Self::Float64 => 8,
            Self::Int128 | Self::UInt128 => 16,
        }
    }

    #[must_use]
    pub const fn is_float(self) -> bool {
        matches!(
            self,
            Self::BFloat16
                | Self::BFloat8
                | Self::Fp6E3M2
                | Self::Float16
                | Self::Float32
                | Self::Float64
        )
    }

    /// Whether a fold over this type can accumulate into itself without
    /// widening. Narrow integers cannot: a sum of `i8` overflows almost
    /// immediately, which is why quantized contraction accumulates in `Int32`.
    /// Every width from 32 bits up is wide enough to accumulate in place,
    /// signed or unsigned, integer or float.
    #[must_use]
    pub const fn accumulates_in_place(self) -> bool {
        matches!(
            self,
            Self::Int32
                | Self::UInt32
                | Self::Int64
                | Self::UInt64
                | Self::Int128
                | Self::UInt128
                | Self::Float32
                | Self::Float64
                | Self::BFloat16
                | Self::Float16
        )
    }
}

impl DType {
    /// Whether this type is a legal gather index type. Only whole-number
    /// types can index a dimension; `Bool` and the floats cannot.
    #[must_use]
    pub const fn is_integer(self) -> bool {
        matches!(
            self,
            Self::Int8
                | Self::UInt8
                | Self::Int16
                | Self::UInt16
                | Self::Int32
                | Self::UInt32
                | Self::Int64
                | Self::UInt64
                | Self::Int128
                | Self::UInt128
        )
    }

    /// Whether this integer type has a sign bit. Only meaningful for
    /// [`is_integer`](Self::is_integer) types — a float's sign is not gated
    /// by this, and `Bool` has none at all.
    #[must_use]
    pub const fn is_signed_integer(self) -> bool {
        matches!(
            self,
            Self::Int8 | Self::Int16 | Self::Int32 | Self::Int64 | Self::Int128
        )
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[proxima::test]
    #[case::boolean(DType::Bool, 1)]
    #[case::narrow_int(DType::Int8, 1)]
    #[case::narrow_uint(DType::UInt8, 1)]
    #[case::int16(DType::Int16, 2)]
    #[case::uint16(DType::UInt16, 2)]
    #[case::half(DType::Float16, 2)]
    #[case::brain_half(DType::BFloat16, 2)]
    #[case::single(DType::Float32, 4)]
    #[case::int32(DType::Int32, 4)]
    #[case::uint32(DType::UInt32, 4)]
    #[case::int64(DType::Int64, 8)]
    #[case::uint64(DType::UInt64, 8)]
    #[case::double(DType::Float64, 8)]
    #[case::int128(DType::Int128, 16)]
    #[case::uint128(DType::UInt128, 16)]
    async fn size_bytes_matches_width(#[case] dtype: DType, #[case] expected: usize) {
        assert_eq!(dtype.size_bytes(), expected);
    }

    #[test]
    fn card_02_bf8_dtype_has_one_byte_float_identity_without_self_accumulation() {
        assert_eq!(DType::BFloat8.size_bytes(), 1);
        assert!(DType::BFloat8.is_float());
        assert!(!DType::BFloat8.accumulates_in_place());
    }

    #[test]
    fn card_02_bf8_dtype_preserves_existing_half_and_single_widths() {
        assert_eq!(DType::BFloat16.size_bytes(), 2);
        assert_eq!(DType::Float16.size_bytes(), 2);
        assert_eq!(DType::Float32.size_bytes(), 4);
        assert!(DType::BFloat16.is_float());
        assert!(DType::Float16.is_float());
        assert!(DType::Float32.is_float());
    }

    #[proxima::test]
    #[case::narrow_int_widens(DType::Int8, false)]
    #[case::narrow_uint_widens(DType::UInt8, false)]
    #[case::int16_widens(DType::Int16, false)]
    #[case::uint16_widens(DType::UInt16, false)]
    #[case::bool_widens(DType::Bool, false)]
    #[case::int32_in_place(DType::Int32, true)]
    #[case::float32_in_place(DType::Float32, true)]
    #[case::int64_in_place(DType::Int64, true)]
    #[case::uint64_in_place(DType::UInt64, true)]
    #[case::int128_in_place(DType::Int128, true)]
    #[case::float64_in_place(DType::Float64, true)]
    async fn narrow_integers_require_a_wider_accumulator(
        #[case] dtype: DType,
        #[case] in_place: bool,
    ) {
        assert_eq!(dtype.accumulates_in_place(), in_place);
    }

    #[test]
    fn only_floating_types_report_float() {
        assert!(DType::Float32.is_float());
        assert!(DType::Float64.is_float());
        assert!(DType::BFloat16.is_float());
        assert!(DType::Float16.is_float());
        assert!(!DType::Int32.is_float());
        assert!(!DType::Bool.is_float());
    }

    #[proxima::test]
    #[case::int8(DType::Int8, true)]
    #[case::uint8(DType::UInt8, true)]
    #[case::int16(DType::Int16, true)]
    #[case::uint16(DType::UInt16, true)]
    #[case::int32(DType::Int32, true)]
    #[case::uint32(DType::UInt32, true)]
    #[case::int64(DType::Int64, true)]
    #[case::uint64(DType::UInt64, true)]
    #[case::int128(DType::Int128, true)]
    #[case::uint128(DType::UInt128, true)]
    #[case::bool_is_not_integer(DType::Bool, false)]
    #[case::float32_is_not_integer(DType::Float32, false)]
    async fn only_whole_number_types_report_integer(#[case] dtype: DType, #[case] integer: bool) {
        assert_eq!(dtype.is_integer(), integer);
    }

    #[proxima::test]
    #[case::int8_signed(DType::Int8, true)]
    #[case::uint8_unsigned(DType::UInt8, false)]
    #[case::int64_signed(DType::Int64, true)]
    #[case::uint64_unsigned(DType::UInt64, false)]
    #[case::int128_signed(DType::Int128, true)]
    #[case::uint128_unsigned(DType::UInt128, false)]
    async fn signedness_matches_the_variant_name(#[case] dtype: DType, #[case] signed: bool) {
        assert_eq!(dtype.is_signed_integer(), signed);
    }

    #[cfg(feature = "config")]
    #[derive(serde::Serialize, serde::Deserialize)]
    struct Holder {
        dtype: DType,
    }

    #[cfg(feature = "config")]
    fn spelled(dtype: DType) -> String {
        toml::to_string(&Holder { dtype }).expect("a dtype must serialize to toml")
    }

    #[cfg(feature = "config")]
    #[proxima::test]
    #[case::boolean(DType::Bool, "dtype = \"bool\"\n")]
    #[case::uint8(DType::UInt8, "dtype = \"u_int8\"\n")]
    #[case::int128(DType::Int128, "dtype = \"int128\"\n")]
    #[case::brain_half(DType::BFloat16, "dtype = \"b_float16\"\n")]
    #[case::brain_byte(DType::BFloat8, "dtype = \"b_float8\"\n")]
    #[case::fp6(DType::Fp6E3M2, "dtype = \"fp6_e3_m2\"\n")]
    #[case::half(DType::Float16, "dtype = \"float16\"\n")]
    #[case::single(DType::Float32, "dtype = \"float32\"\n")]
    #[case::double(DType::Float64, "dtype = \"float64\"\n")]
    async fn former_spellings_are_pinned(#[case] dtype: DType, #[case] expected: &str) {
        assert_eq!(spelled(dtype), expected);
    }

    #[cfg(feature = "float-layout-dtype")]
    mod layout_carrier {
        use proxima_primitives::float_layout::{FloatLayout, SignEncoding, Specials};

        use super::*;

        const E4M3: FloatLayout = FloatLayout {
            exponent_bits: 4,
            mantissa_bits: 3,
            specials: Specials::SaturateOnOverflow,
            bias_override: None,
            sign: SignEncoding::Magnitude,
        };

        #[proxima::test]
        #[case::single(FloatLayout::BF32, 32)]
        #[case::half(FloatLayout::F16, 16)]
        #[case::brain_half(FloatLayout::BF16, 16)]
        #[case::brain_byte_saturating(FloatLayout::E5M2_SATURATING, 8)]
        #[case::fp4_e2m1(FloatLayout::E2M1, 4)]
        #[case::fp6_e3m2(FloatLayout::E3M2, 6)]
        #[case::fp6_e2m3(FloatLayout::E2M3, 6)]
        #[case::fp8_e4m3_from_fields(E4M3, 8)]
        async fn float_layout_dtype_bits_equal_layout_width(
            #[case] layout: FloatLayout,
            #[case] width: usize,
        ) {
            let dtype = DType::Float(layout);

            assert_eq!(dtype.bits(), usize::from(layout.width()));
            assert_eq!(dtype.bits(), width);
            assert_eq!(dtype.size_bytes(), width.div_ceil(8));
        }

        #[test]
        fn float_layout_dtype_named_constants_are_layout_instances() {
            assert_eq!(DType::Float32, DType::Float(FloatLayout::BF32));
            assert_eq!(DType::Float16, DType::Float(FloatLayout::F16));
            assert_eq!(DType::BFloat16, DType::Float(FloatLayout::BF16));
            assert_eq!(DType::BFloat8, DType::Float(FloatLayout::E5M2_SATURATING));
            assert_eq!(DType::Fp6E3M2, DType::Float(FloatLayout::E3M2));
        }

        #[cfg(feature = "config")]
        #[test]
        fn float_layout_dtype_former_spellings_round_trip_through_serde() {
            let layouts = [
                DType::Float32,
                DType::Float16,
                DType::BFloat16,
                DType::BFloat8,
                DType::Fp6E3M2,
                DType::Float(E4M3),
            ];

            let restored = layouts.map(|dtype| {
                toml::from_str::<Holder>(&spelled(dtype))
                    .expect("a spelling written by the serializer must deserialize")
                    .dtype
            });

            assert_eq!(restored, layouts);
        }

        #[test]
        fn float_layout_dtype_wide_types_stay_named_float64_int64() {
            let wide = [
                (DType::Float64, 64),
                (DType::Int64, 64),
                (DType::UInt64, 64),
                (DType::Int128, 128),
                (DType::UInt128, 128),
            ];

            for (dtype, bits) in wide {
                assert!(
                    !matches!(dtype, DType::Float(_)),
                    "{dtype:?} must not carry a layout"
                );
                assert_eq!(dtype.bits(), bits);
            }
            assert!(DType::Float64.is_float());
        }

        #[test]
        fn debug_spelling_keeps_the_former_names_and_prints_unnamed_layouts_by_fields() {
            assert_eq!(format!("{:?}", DType::BFloat16), "BFloat16");
            assert_eq!(format!("{:?}", DType::Fp6E3M2), "Fp6E3M2");
            assert!(format!("{:?}", DType::Float(E4M3)).starts_with("Float(FloatLayout"));
        }

        #[test]
        fn float_layout_dtype_control_equal_width_layouts_are_distinct_dtypes() {
            let fp6_e3m2 = DType::Float(FloatLayout::E3M2);
            let fp6_e2m3 = DType::Float(FloatLayout::E2M3);

            assert_eq!(fp6_e3m2.bits(), fp6_e2m3.bits());
            assert_ne!(fp6_e3m2, fp6_e2m3);
        }
    }
}
