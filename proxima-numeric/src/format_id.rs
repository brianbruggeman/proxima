use crate::dtype::DType;
use crate::format::NumericFormat;

/// A two-byte handle naming one entry of the one numeric format table.
///
/// [`NumericFormat`] is 520 bytes because its block arm carries a whole layout; a caller that
/// only needs to hold, hash, order or compare a format in a `const` context holds this instead
/// and calls [`Self::format`] when it needs the layout. Index order is the table order: the
/// wire-tagged formats of [`NumericFormat::ALL`] first, so `id.index() == format.tag()` for every
/// one of them, then [`NumericFormat::UNTAGGED`], then the scalar twins of the `DType` constants
/// that are in neither table. Appending to the later tables never shifts an existing index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FormatId(u16);

const _: () = assert!(core::mem::size_of::<FormatId>() == 2);

const TAGGED: usize = NumericFormat::ALL.len();
const UNTAGGED: usize = NumericFormat::UNTAGGED.len();

/// `DType` constants whose `NumericFormat` twin has neither a wire tag nor an untagged row.
const SCALAR_TWINS: [NumericFormat; 13] = [
    NumericFormat::Int8,
    NumericFormat::UInt8,
    NumericFormat::Int16,
    NumericFormat::UInt16,
    NumericFormat::Int32,
    NumericFormat::UInt32,
    NumericFormat::Int64,
    NumericFormat::UInt64,
    NumericFormat::Int128,
    NumericFormat::UInt128,
    NumericFormat::Float64,
    NumericFormat::Float8E4M3Fn,
    NumericFormat::Fp6E3M2Scalar,
];

impl FormatId {
    /// Number of formats the table names.
    pub const COUNT: usize = TAGGED + UNTAGGED + SCALAR_TWINS.len();

    /// The position in the table; for a wire-tagged format this equals [`NumericFormat::tag`].
    #[must_use]
    pub const fn index(self) -> u16 {
        self.0
    }

    /// The id at a table position, `None` past the end of the table.
    #[must_use]
    pub const fn from_index(index: u16) -> Option<Self> {
        if (index as usize) < Self::COUNT {
            Some(Self(index))
        } else {
            None
        }
    }

    /// The format this id names; total, because an id is only built from a table position.
    #[must_use]
    pub const fn format(self) -> NumericFormat {
        let index = self.0 as usize;
        if index < TAGGED {
            NumericFormat::ALL[index]
        } else if index < TAGGED + UNTAGGED {
            NumericFormat::UNTAGGED[index - TAGGED].0
        } else {
            SCALAR_TWINS[index - TAGGED - UNTAGGED]
        }
    }

    /// The format's canonical lowercase name, the spelling [`NumericFormat::name`] gives.
    #[must_use]
    pub fn name(self) -> Option<&'static str> {
        self.format().name()
    }

    /// The `DType` describing the same elements; `None` for a block or level format.
    #[must_use]
    pub fn to_dtype(self) -> Option<DType> {
        self.format().to_dtype()
    }
}

impl NumericFormat {
    /// The two-byte handle for this format, `None` for a layout the table has no row for.
    #[must_use]
    pub fn id(&self) -> Option<FormatId> {
        let position = Self::ALL
            .iter()
            .position(|named| named == self)
            .or_else(|| {
                Self::UNTAGGED
                    .iter()
                    .position(|(named, _)| named == self)
                    .map(|found| TAGGED + found)
            })
            .or_else(|| {
                SCALAR_TWINS
                    .iter()
                    .position(|named| named == self)
                    .map(|found| TAGGED + UNTAGGED + found)
            })?;
        u16::try_from(position).ok().map(FormatId)
    }
}

impl DType {
    /// The two-byte handle for this dtype, `None` only for an unnamed custom layout.
    #[must_use]
    pub fn id(&self) -> Option<FormatId> {
        self.numeric_format().id()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::float_layout::FloatLayout;

    fn dtype_twins() -> [(DType, NumericFormat); 19] {
        [
            (DType::Bool, NumericFormat::Bool),
            (DType::INT8, NumericFormat::Int8),
            (DType::UINT8, NumericFormat::UInt8),
            (DType::INT16, NumericFormat::Int16),
            (DType::UINT16, NumericFormat::UInt16),
            (DType::INT32, NumericFormat::Int32),
            (DType::UINT32, NumericFormat::UInt32),
            (DType::INT64, NumericFormat::Int64),
            (DType::UINT64, NumericFormat::UInt64),
            (DType::INT128, NumericFormat::Int128),
            (DType::UINT128, NumericFormat::UInt128),
            (DType::BFLOAT16, NumericFormat::BFloat16),
            (DType::BFLOAT8, NumericFormat::Bf8E5M2),
            (DType::FLOAT8_E4M3FN, NumericFormat::Float8E4M3Fn),
            (DType::FLOAT8_E5M2, NumericFormat::BFloat8),
            (DType::FP6_E3_M2, NumericFormat::Fp6E3M2Scalar),
            (DType::FLOAT16, NumericFormat::Float16),
            (DType::FLOAT32, NumericFormat::Float32),
            (DType::FLOAT64, NumericFormat::Float64),
        ]
    }

    #[test]
    fn every_dtype_constant_round_trips_through_its_format_id_with_identical_methods() {
        let twins = dtype_twins();
        assert_eq!(twins.len(), 19, "pinned: every DType constant plus Bool");

        for (dtype, format) in twins {
            let id = dtype.id().expect("a dtype constant is named in the table");
            assert_eq!(id.to_dtype(), Some(dtype), "{dtype:?} round trip");
            assert_eq!(id.format(), dtype.numeric_format(), "{dtype:?} numeric_format");
            assert_eq!(id.format(), format, "{dtype:?} twin");
            assert_eq!(Some(id), format.id(), "{dtype:?} format id");
            let through_id = id.format();
            assert_eq!(through_id.size_bytes(), dtype.size_bytes(), "{dtype:?} size_bytes");
            assert_eq!(through_id.bits(), dtype.bits(), "{dtype:?} bits");
            assert_eq!(through_id.is_float(), dtype.is_float(), "{dtype:?} is_float");
            assert_eq!(
                through_id.accumulates_in_place(),
                dtype.accumulates_in_place(),
                "{dtype:?} accumulates_in_place"
            );
            assert_eq!(through_id.is_integer(), dtype.is_integer(), "{dtype:?} is_integer");
            assert_eq!(
                through_id.is_signed_integer(),
                dtype.is_signed_integer(),
                "{dtype:?} is_signed_integer"
            );
        }
    }

    #[test]
    fn every_table_position_names_a_distinct_format_that_maps_back() {
        let mut seen: std::vec::Vec<NumericFormat> = std::vec::Vec::new();

        for index in 0..FormatId::COUNT {
            let id = FormatId::from_index(index as u16).expect("index is inside the table");
            let format = id.format();
            assert_eq!(format.id(), Some(id), "position {index}");
            assert!(!seen.contains(&format), "position {index} repeats a format");
            seen.push(format);
        }
        assert_eq!(FormatId::COUNT, 51, "pinned: 36 tagged, 2 untagged, 13 scalar twins");
        assert_eq!(FormatId::from_index(51), None);
    }

    #[test]
    fn a_wire_tagged_format_has_the_id_index_equal_to_its_tag() {
        for (position, format) in NumericFormat::ALL.iter().enumerate() {
            let id = format.id().expect("a tagged format is in the table");
            assert_eq!(usize::from(id.index()), position);
            assert_eq!(format.tag(), Some(id.index() as u8));
            assert_eq!(id.name(), Some(NumericFormat::NAMES[position]));
        }
        assert_eq!(NumericFormat::Float32.id().map(FormatId::index), Some(36));
        assert_eq!(NumericFormat::Bool.id().map(FormatId::index), Some(37));
        assert_eq!(NumericFormat::Int8.id().map(FormatId::index), Some(38));
    }

    #[test]
    fn an_unnamed_custom_layout_has_no_id() {
        let custom = FloatLayout {
            bias_override: Some(5),
            ..FloatLayout::E4M3FN
        };

        assert_eq!(DType::Layout(custom).id(), None);
        assert_eq!(NumericFormat::Scalar(custom).id(), None);
    }

    #[test]
    fn a_block_format_has_an_id_but_no_dtype() {
        let id = NumericFormat::Q4_0.id().expect("Q4_0 is wire-tagged");

        assert_eq!(id.to_dtype(), None);
        assert_eq!(id.name(), Some("q4_0"));
    }

    #[test]
    fn a_format_id_is_const_comparable() {
        const FLOAT32: Option<FormatId> = FormatId::from_index(36);
        const FORMAT: NumericFormat = match FLOAT32 {
            Some(id) => id.format(),
            None => NumericFormat::Bool,
        };

        assert!(FORMAT == NumericFormat::Float32);
    }
}
