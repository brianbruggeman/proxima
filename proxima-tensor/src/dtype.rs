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
    Float16,
    Float32,
    Float64,
}

impl DType {
    #[must_use]
    pub const fn size_bytes(self) -> usize {
        match self {
            Self::Bool | Self::Int8 | Self::UInt8 => 1,
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
            Self::BFloat16 | Self::Float16 | Self::Float32 | Self::Float64
        )
    }

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
}
