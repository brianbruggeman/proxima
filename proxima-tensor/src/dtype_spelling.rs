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
            DType::Layout(layout) => Self::former_spelling(layout),
        }
    }
}

impl DTypeSpelling {
    /// Preserve a former scalar spelling when every layout field matches; otherwise serialize
    /// the descriptor fields.
    fn former_spelling(layout: FloatLayout) -> Self {
        [
            (FloatLayout::INT8, Self::Int8),
            (FloatLayout::UINT8, Self::UInt8),
            (FloatLayout::INT16, Self::Int16),
            (FloatLayout::UINT16, Self::UInt16),
            (FloatLayout::INT32, Self::Int32),
            (FloatLayout::UINT32, Self::UInt32),
            (FloatLayout::INT64, Self::Int64),
            (FloatLayout::UINT64, Self::UInt64),
            (FloatLayout::INT128, Self::Int128),
            (FloatLayout::UINT128, Self::UInt128),
            (FloatLayout::BF32, Self::Float32),
            (FloatLayout::F64, Self::Float64),
            (FloatLayout::F16, Self::Float16),
            (FloatLayout::BF16, Self::BFloat16),
            (FloatLayout::E5M2_SATURATING, Self::BFloat8),
            (FloatLayout::E3M2, Self::Fp6E3M2),
        ]
        .into_iter()
        .find_map(|(named, spelling)| named.same_format(&layout).then_some(spelling))
        .unwrap_or(Self::Float(layout))
    }
}

impl From<DTypeSpelling> for DType {
    fn from(spelling: DTypeSpelling) -> Self {
        match spelling {
            DTypeSpelling::Bool => Self::Bool,
            DTypeSpelling::Int8 => Self::INT8,
            DTypeSpelling::UInt8 => Self::UINT8,
            DTypeSpelling::Int16 => Self::INT16,
            DTypeSpelling::UInt16 => Self::UINT16,
            DTypeSpelling::Int32 => Self::INT32,
            DTypeSpelling::UInt32 => Self::UINT32,
            DTypeSpelling::Int64 => Self::INT64,
            DTypeSpelling::UInt64 => Self::UINT64,
            DTypeSpelling::Int128 => Self::INT128,
            DTypeSpelling::UInt128 => Self::UINT128,
            DTypeSpelling::Float64 => Self::FLOAT64,
            DTypeSpelling::BFloat16 => Self::BFLOAT16,
            DTypeSpelling::BFloat8 => Self::BFLOAT8,
            DTypeSpelling::Fp6E3M2 => Self::FP6_E3_M2,
            DTypeSpelling::Float16 => Self::FLOAT16,
            DTypeSpelling::Float32 => Self::FLOAT32,
            DTypeSpelling::Float(layout) => Self::Layout(layout),
        }
    }
}
