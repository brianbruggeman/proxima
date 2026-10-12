use proxima_primitives::float_layout::FloatLayout;

use crate::dtype::DType;

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum DTypeSpelling {
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
    Fp6E3M2,
    Float16,
    Float32,
    Float64,
    Float(FloatLayout),
}

impl From<DType> for DTypeSpelling {
    fn from(dtype: DType) -> Self {
        match dtype {
            DType::Bool => Self::Bool,
            DType::Int8 => Self::Int8,
            DType::UInt8 => Self::UInt8,
            DType::Int16 => Self::Int16,
            DType::UInt16 => Self::UInt16,
            DType::Int32 => Self::Int32,
            DType::UInt32 => Self::UInt32,
            DType::Int64 => Self::Int64,
            DType::UInt64 => Self::UInt64,
            DType::Int128 => Self::Int128,
            DType::UInt128 => Self::UInt128,
            DType::Float64 => Self::Float64,
            DType::Float(layout) => Self::former_spelling(layout),
        }
    }
}

impl DTypeSpelling {
    /// The spelling a layout kept from when it was a variant, by comparing the whole layout; any
    /// other layout is spelled by its fields.
    fn former_spelling(layout: FloatLayout) -> Self {
        [
            (FloatLayout::BF32, Self::Float32),
            (FloatLayout::F16, Self::Float16),
            (FloatLayout::BF16, Self::BFloat16),
            (FloatLayout::E5M2_SATURATING, Self::BFloat8),
            (FloatLayout::E3M2, Self::Fp6E3M2),
        ]
        .into_iter()
        .find_map(|(named, spelling)| (named == layout).then_some(spelling))
        .unwrap_or(Self::Float(layout))
    }
}

impl From<DTypeSpelling> for DType {
    fn from(spelling: DTypeSpelling) -> Self {
        match spelling {
            DTypeSpelling::Bool => Self::Bool,
            DTypeSpelling::Int8 => Self::Int8,
            DTypeSpelling::UInt8 => Self::UInt8,
            DTypeSpelling::Int16 => Self::Int16,
            DTypeSpelling::UInt16 => Self::UInt16,
            DTypeSpelling::Int32 => Self::Int32,
            DTypeSpelling::UInt32 => Self::UInt32,
            DTypeSpelling::Int64 => Self::Int64,
            DTypeSpelling::UInt64 => Self::UInt64,
            DTypeSpelling::Int128 => Self::Int128,
            DTypeSpelling::UInt128 => Self::UInt128,
            DTypeSpelling::Float64 => Self::Float64,
            DTypeSpelling::BFloat16 => Self::BFloat16,
            DTypeSpelling::BFloat8 => Self::BFloat8,
            DTypeSpelling::Fp6E3M2 => Self::Fp6E3M2,
            DTypeSpelling::Float16 => Self::Float16,
            DTypeSpelling::Float32 => Self::Float32,
            DTypeSpelling::Float(layout) => Self::Float(layout),
        }
    }
}
