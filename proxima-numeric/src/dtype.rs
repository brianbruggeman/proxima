use crate::float_layout::{FloatLayout, SignEncoding, Specials};

/// Element encoding carried by a tensor operation.
///
/// Every numeric type is the shared structural [`FloatLayout`], so a new element format is one
/// layout constant, never a new dtype variant. `Bool` is the one element that is not a number.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    Bool,
    Layout(FloatLayout),
}

impl core::fmt::Debug for DType {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match *self {
            Self::Bool => formatter.write_str("Bool"),
            Self::Layout(layout) => match debug_name(layout) {
                Some(name) => formatter.write_str(name),
                None => formatter.debug_tuple("Layout").field(&layout).finish(),
            },
        }
    }
}

/// The spelling a `dtype` has always carried in `Debug`. A plan's structural fingerprint is
/// derived from the rendered form, so these names are load-bearing for every recorded graph
/// hash; the canonical config spelling ([`FloatLayout::name`]) is lowercase and separate.
fn debug_name(layout: FloatLayout) -> Option<&'static str> {
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
    .find_map(|(named, name)| (named == layout).then_some(name))
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

    /// The number this dtype is stored as; `Bool` has no numeric format.
    #[must_use]
    pub const fn format(self) -> Option<crate::NumericFormat> {
        match self {
            Self::Layout(layout) => Some(crate::NumericFormat::Scalar(layout)),
            Self::Bool => None,
        }
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

#[cfg(feature = "config")]
impl serde::Serialize for DType {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Bool => serializer.serialize_str("bool"),
            Self::Layout(layout) => match layout.name() {
                Some(name) => serializer.serialize_str(name),
                None => serializer.serialize_newtype_struct("float", layout),
            },
        }
    }
}

#[cfg(feature = "config")]
impl<'de> serde::Deserialize<'de> for DType {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct DTypeVisitor;

        impl<'de> serde::de::Visitor<'de> for DTypeVisitor {
            type Value = DType;

            fn expecting(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                formatter.write_str("a dtype name such as \"float32\", or a structural layout")
            }

            fn visit_str<E: serde::de::Error>(self, name: &str) -> Result<DType, E> {
                if name == "bool" {
                    return Ok(DType::Bool);
                }
                FloatLayout::from_name(name)
                    .map(DType::Layout)
                    .ok_or_else(|| E::unknown_variant(name, &[]))
            }

            fn visit_map<A: serde::de::MapAccess<'de>>(self, map: A) -> Result<DType, A::Error> {
                <FloatLayout as serde::Deserialize>::deserialize(
                    serde::de::value::MapAccessDeserializer::new(map),
                )
                .map(DType::Layout)
            }
        }

        deserializer.deserialize_any(DTypeVisitor)
    }
}

#[cfg(all(test, feature = "config"))]
mod tests {
    use crate::float_layout::{FloatLayout, SignEncoding, Specials};

    use super::DType;

    #[test]
    fn every_named_dtype_serializes_to_its_canonical_name() {
        let cases = [
            (DType::INT8, "\"int8\""),
            (DType::UINT32, "\"uint32\""),
            (DType::INT128, "\"int128\""),
            (DType::BFLOAT16, "\"bfloat16\""),
            (DType::BFLOAT8, "\"bfloat8\""),
            (DType::FLOAT8_E4M3FN, "\"float8_e4m3fn\""),
            (DType::FLOAT8_E5M2, "\"float8_e5m2\""),
            (DType::FP6_E3_M2, "\"fp6_e3m2\""),
            (DType::FLOAT16, "\"float16\""),
            (DType::FLOAT32, "\"float32\""),
            (DType::FLOAT64, "\"float64\""),
            (DType::Bool, "\"bool\""),
        ];

        for (dtype, expected) in cases {
            let serialized = serde_json::to_string(&dtype).expect("dtype serializes");
            assert_eq!(serialized, expected);
        }
    }

    #[test]
    fn named_and_structural_dtypes_round_trip() {
        let unnamed = DType::Layout(FloatLayout {
            exponent_bits: 4,
            mantissa_bits: 3,
            specials: Specials::NanOnly,
            bias_override: Some(7),
            sign: SignEncoding::Magnitude,
            nan: Default::default(),
        });

        for dtype in [
            DType::INT8,
            DType::UINT128,
            DType::FLOAT16,
            DType::FLOAT8_E4M3FN,
            DType::FLOAT64,
            DType::Bool,
            unnamed,
        ] {
            let serialized = serde_json::to_string(&dtype).expect("dtype serializes");
            let restored: DType = serde_json::from_str(&serialized).expect("dtype deserializes");
            assert_eq!(restored, dtype);
        }
    }

    #[test]
    fn a_layout_spelled_with_an_explicit_bias_round_trips_structurally() {
        let explicitly_biased = FloatLayout {
            bias_override: Some(FloatLayout::E4M3FN.bias()),
            ..FloatLayout::E4M3FN
        };

        assert_eq!(explicitly_biased.format_name(), Some("float8_e4m3fn"));
        assert_eq!(
            explicitly_biased.name(),
            None,
            "a name must round-trip, so it denotes an exact layout, not a numerically equal one"
        );
        let serialized =
            serde_json::to_string(&DType::Layout(explicitly_biased)).expect("serializes");
        let restored: DType = serde_json::from_str(&serialized).expect("deserializes");
        assert_eq!(restored, DType::Layout(explicitly_biased));
    }

    #[test]
    fn debug_keeps_legacy_names_for_canonical_layouts() {
        assert_eq!(format!("{:?}", DType::INT8), "Int8");
        assert_eq!(format!("{:?}", DType::BFLOAT16), "BFloat16");
        assert_eq!(format!("{:?}", DType::FLOAT64), "Float64");
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
}
