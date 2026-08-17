/// Element type of a tensor operand.
///
/// Carried as a field on every [`Op`](crate::Op) rather than as a type
/// parameter. A single node routinely mixes three of these — quantized matmul
/// is `i8 x i8 -> i32 -> f32` — so one `T` could not describe it, and a leaf's
/// element type is not known until the weights are opened anyway.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "config", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "config", serde(rename_all = "snake_case"))]
pub enum DType {
    Bool,
    Int8,
    UInt8,
    Int32,
    UInt32,
    BFloat16,
    Float16,
    Float32,
}

impl DType {
    #[must_use]
    pub const fn size_bytes(self) -> usize {
        match self {
            Self::Bool | Self::Int8 | Self::UInt8 => 1,
            Self::BFloat16 | Self::Float16 => 2,
            Self::Int32 | Self::UInt32 | Self::Float32 => 4,
        }
    }

    #[must_use]
    pub const fn is_float(self) -> bool {
        matches!(self, Self::BFloat16 | Self::Float16 | Self::Float32)
    }

    /// Whether this type is a legal gather index type. Only whole-number
    /// types can index a dimension; `Bool` and the floats cannot.
    #[must_use]
    pub const fn is_integer(self) -> bool {
        matches!(self, Self::Int8 | Self::UInt8 | Self::Int32 | Self::UInt32)
    }

    /// Whether a fold over this type can accumulate into itself without
    /// widening. Narrow integers cannot: a sum of `i8` overflows almost
    /// immediately, which is why quantized contraction accumulates in `Int32`.
    #[must_use]
    pub const fn accumulates_in_place(self) -> bool {
        matches!(
            self,
            Self::Int32 | Self::UInt32 | Self::Float32 | Self::BFloat16 | Self::Float16
        )
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case::boolean(DType::Bool, 1)]
    #[case::narrow_int(DType::Int8, 1)]
    #[case::half(DType::Float16, 2)]
    #[case::brain_half(DType::BFloat16, 2)]
    #[case::single(DType::Float32, 4)]
    fn size_bytes_matches_width(#[case] dtype: DType, #[case] expected: usize) {
        assert_eq!(dtype.size_bytes(), expected);
    }

    #[rstest]
    #[case::narrow_int_widens(DType::Int8, false)]
    #[case::narrow_uint_widens(DType::UInt8, false)]
    #[case::bool_widens(DType::Bool, false)]
    #[case::int32_in_place(DType::Int32, true)]
    #[case::float32_in_place(DType::Float32, true)]
    fn narrow_integers_require_a_wider_accumulator(#[case] dtype: DType, #[case] in_place: bool) {
        assert_eq!(dtype.accumulates_in_place(), in_place);
    }

    #[test]
    fn only_floating_types_report_float() {
        assert!(DType::Float32.is_float());
        assert!(DType::BFloat16.is_float());
        assert!(!DType::Int32.is_float());
        assert!(!DType::Bool.is_float());
    }

    #[rstest]
    #[case::int8(DType::Int8, true)]
    #[case::uint8(DType::UInt8, true)]
    #[case::int32(DType::Int32, true)]
    #[case::uint32(DType::UInt32, true)]
    #[case::bool_is_not_integer(DType::Bool, false)]
    #[case::float32_is_not_integer(DType::Float32, false)]
    fn only_whole_number_types_report_integer(#[case] dtype: DType, #[case] integer: bool) {
        assert_eq!(dtype.is_integer(), integer);
    }
}
