use proxima_primitives::float_layout::{FloatLayout, SignEncoding, Specials};

#[cfg(feature = "config")]
use crate::dtype_spelling::DTypeSpelling;

/// Element encoding carried by a tensor operation.
///
/// Every numeric type is the shared structural [`FloatLayout`]. The descriptor supports
/// 8–128-bit integer and 4–64-bit floating formats; generic codecs may support a narrower range.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(
    feature = "config",
    derive(serde::Serialize, serde::Deserialize),
    serde(into = "DTypeSpelling", from = "DTypeSpelling")
)]
pub enum DType {
    Bool,
    Layout(FloatLayout),
}

impl core::fmt::Debug for DType {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match *self {
            Self::Bool => formatter.write_str("Bool"),
            Self::Layout(layout) => match Self::former_name(layout) {
                Some(former) => formatter.write_str(former),
                None => formatter.debug_tuple("Layout").field(&layout).finish(),
            },
        }
    }
}

impl DType {
    pub const INT8: Self = Self::Layout(FloatLayout::INT8);
    pub const UINT8: Self = Self::Layout(FloatLayout::UINT8);
    pub const INT16: Self = Self::Layout(FloatLayout::INT16);
    pub const UINT16: Self = Self::Layout(FloatLayout::UINT16);
    pub const INT32: Self = Self::Layout(FloatLayout::INT32);
    pub const UINT32: Self = Self::Layout(FloatLayout::UINT32);
    pub const INT64: Self = Self::Layout(FloatLayout::INT64);
    pub const UINT64: Self = Self::Layout(FloatLayout::UINT64);
    pub const INT128: Self = Self::Layout(FloatLayout::INT128);
    pub const UINT128: Self = Self::Layout(FloatLayout::UINT128);
    pub const BFLOAT16: Self = Self::Layout(FloatLayout::BF16);
    pub const BFLOAT8: Self = Self::Layout(FloatLayout::E5M2_SATURATING);
    pub const FLOAT8_E4M3FN: Self = Self::Layout(FloatLayout::E4M3FN);
    pub const FLOAT8_E5M2: Self = Self::Layout(FloatLayout::E5M2);
    pub const FP6_E3_M2: Self = Self::Layout(FloatLayout::E3M2);
    pub const FLOAT16: Self = Self::Layout(FloatLayout::F16);
    pub const FLOAT32: Self = Self::Layout(FloatLayout::BF32);
    pub const FLOAT64: Self = Self::Layout(FloatLayout::F64);

    /// The former spelling for a canonical layout, compared by every field.
    pub(crate) fn former_name(layout: FloatLayout) -> Option<&'static str> {
        [
            (FloatLayout::INT8, "Int8"),
            (FloatLayout::UINT8, "UInt8"),
            (FloatLayout::INT16, "Int16"),
            (FloatLayout::UINT16, "UInt16"),
            (FloatLayout::INT32, "Int32"),
            (FloatLayout::UINT32, "UInt32"),
            (FloatLayout::INT64, "Int64"),
            (FloatLayout::UINT64, "UInt64"),
            (FloatLayout::INT128, "Int128"),
            (FloatLayout::UINT128, "UInt128"),
            (FloatLayout::BF32, "Float32"),
            (FloatLayout::F64, "Float64"),
            (FloatLayout::F16, "Float16"),
            (FloatLayout::BF16, "BFloat16"),
            (FloatLayout::E5M2_SATURATING, "BFloat8"),
            (FloatLayout::E3M2, "Fp6E3M2"),
        ]
        .into_iter()
        .find_map(|(named, name)| named.same_format(&layout).then_some(name))
    }

    #[must_use]
    pub const fn size_bytes(self) -> usize {
        match self {
            Self::Bool => 1,
            Self::Layout(layout) => (layout.width() as usize).div_ceil(8),
        }
    }

    #[must_use]
    pub const fn bits(self) -> usize {
        match self {
            Self::Bool => 8,
            Self::Layout(layout) => layout.width() as usize,
        }
    }

    #[must_use]
    pub const fn is_float(self) -> bool {
        match self {
            Self::Layout(layout) => layout.exponent_bits > 0,
            Self::Bool => false,
        }
    }

    /// Whether a fold can accumulate in the same encoding without widening.
    #[must_use]
    pub const fn accumulates_in_place(self) -> bool {
        match self {
            Self::Layout(layout) => {
                if layout.exponent_bits == 0 {
                    layout.width() >= 32
                } else {
                    layout.width() >= 16
                }
            }
            Self::Bool => false,
        }
    }

    /// Whether this encoding is a whole-number type suitable for gather indices.
    #[must_use]
    pub const fn is_integer(self) -> bool {
        match self {
            Self::Layout(layout) => {
                layout.exponent_bits == 0
                    && matches!(layout.specials, Specials::None)
                    && layout.bias() == 1 - layout.mantissa_bits as i16
            }
            Self::Bool => false,
        }
    }

    /// Whether an integer layout has a signed representation.
    #[must_use]
    pub const fn is_signed_integer(self) -> bool {
        match self {
            Self::Layout(layout) => {
                layout.exponent_bits == 0
                    && !matches!(layout.sign, SignEncoding::Absent)
                    && matches!(layout.specials, Specials::None)
                    && layout.bias() == 1 - layout.mantissa_bits as i16
            }
            Self::Bool => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use proxima_primitives::float_layout::FloatLayout;

    use super::DType;

    #[test]
    fn numeric_names_are_structural_layout_values() {
        let cases = [
            (DType::INT8, FloatLayout::INT8),
            (DType::UINT8, FloatLayout::UINT8),
            (DType::INT16, FloatLayout::INT16),
            (DType::UINT16, FloatLayout::UINT16),
            (DType::INT32, FloatLayout::INT32),
            (DType::UINT32, FloatLayout::UINT32),
            (DType::INT64, FloatLayout::INT64),
            (DType::UINT64, FloatLayout::UINT64),
            (DType::INT128, FloatLayout::INT128),
            (DType::UINT128, FloatLayout::UINT128),
            (DType::FLOAT16, FloatLayout::F16),
            (DType::FLOAT8_E4M3FN, FloatLayout::E4M3FN),
            (DType::FLOAT8_E5M2, FloatLayout::E5M2),
            (DType::BFLOAT16, FloatLayout::BF16),
            (DType::FLOAT32, FloatLayout::BF32),
            (DType::FLOAT64, FloatLayout::F64),
        ];

        for (dtype, expected_layout) in cases {
            assert_eq!(dtype, DType::Layout(expected_layout));
        }
    }

    #[test]
    fn integer_layout_fields_derive_width_and_signedness() {
        let cases = [
            (DType::INT8, 8, true),
            (DType::UINT8, 8, false),
            (DType::INT16, 16, true),
            (DType::UINT16, 16, false),
            (DType::INT32, 32, true),
            (DType::UINT32, 32, false),
            (DType::INT64, 64, true),
            (DType::UINT64, 64, false),
            (DType::INT128, 128, true),
            (DType::UINT128, 128, false),
        ];

        for (dtype, width, signed) in cases {
            assert_eq!(dtype.bits(), width);
            assert_eq!(dtype.size_bytes(), width.div_ceil(8));
            assert!(dtype.is_integer());
            assert!(!dtype.is_float());
            assert_eq!(dtype.is_signed_integer(), signed);
        }
    }

    #[test]
    fn structural_layout_classification_preserves_accumulation_widths() {
        assert!(!DType::INT16.accumulates_in_place());
        assert!(DType::INT32.accumulates_in_place());
        assert!(DType::INT64.accumulates_in_place());
        assert!(DType::FLOAT16.accumulates_in_place());
        assert!(DType::FLOAT32.accumulates_in_place());
        assert!(DType::FLOAT64.accumulates_in_place());
        assert!(!DType::BFLOAT8.accumulates_in_place());
        assert!(!DType::FP6_E3_M2.accumulates_in_place());
        assert!(DType::FLOAT8_E4M3FN.is_float());
        assert_eq!(DType::FLOAT8_E4M3FN.bits(), 8);
    }

    #[test]
    fn debug_keeps_legacy_names_for_canonical_layouts() {
        assert_eq!(format!("{:?}", DType::INT8), "Int8");
        assert_eq!(format!("{:?}", DType::UINT32), "UInt32");
        assert_eq!(format!("{:?}", DType::BFLOAT16), "BFloat16");
        assert_eq!(format!("{:?}", DType::FLOAT64), "Float64");
    }

    #[cfg(feature = "config")]
    #[test]
    fn serde_keeps_numeric_spellings_while_restoring_structural_layouts() {
        let dtypes = [
            DType::INT8,
            DType::UINT32,
            DType::INT64,
            DType::UINT128,
            DType::BFLOAT16,
            DType::FLOAT8_E4M3FN,
            DType::FLOAT8_E5M2,
            DType::FLOAT64,
            DType::Layout(FloatLayout {
                exponent_bits: 4,
                mantissa_bits: 3,
                specials: proxima_primitives::float_layout::Specials::NanOnly,
                bias_override: Some(7),
                sign: proxima_primitives::float_layout::SignEncoding::Magnitude,
            }),
        ];

        #[derive(serde::Serialize, serde::Deserialize)]
        struct TypeHolder {
            dtype: DType,
        }

        for dtype in dtypes {
            let serialized = toml::to_string(&TypeHolder { dtype }).expect("dtype serializes");
            let restored: TypeHolder = toml::from_str(&serialized).expect("dtype deserializes");
            assert_eq!(restored.dtype, dtype);
            if let DType::Layout(layout) = restored.dtype {
                assert_eq!(restored.dtype, DType::Layout(layout));
            }
        }
    }
}
