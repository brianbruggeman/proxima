//! A parsed metadata value: either a scalar or a homogeneous array of
//! scalars, per `gguf.h:8-16`. Nested arrays are not part of the format
//! (the writer only emits `GGUF_TYPE_ARRAY` as an outer tag, never an
//! array element type — `gguf.cpp:462` rejects it) so `MetadataArray`
//! has no `Array` variant.

use alloc::string::String;
use alloc::vec::Vec;

use crate::types::MetadataType;

/// One metadata KV value, tagged by [`MetadataType`].
#[derive(Debug, Clone, PartialEq)]
pub enum MetadataValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    U64(u64),
    I64(i64),
    F64(f64),
    Array(MetadataArray),
}

impl MetadataValue {
    /// The wire type tag this value would round-trip through.
    #[must_use]
    pub fn metadata_type(&self) -> MetadataType {
        match self {
            Self::U8(_) => MetadataType::U8,
            Self::I8(_) => MetadataType::I8,
            Self::U16(_) => MetadataType::U16,
            Self::I16(_) => MetadataType::I16,
            Self::U32(_) => MetadataType::U32,
            Self::I32(_) => MetadataType::I32,
            Self::F32(_) => MetadataType::F32,
            Self::Bool(_) => MetadataType::Bool,
            Self::String(_) => MetadataType::String,
            Self::U64(_) => MetadataType::U64,
            Self::I64(_) => MetadataType::I64,
            Self::F64(_) => MetadataType::F64,
            Self::Array(_) => MetadataType::Array,
        }
    }

    /// Convenience accessor for `general.alignment` and similar u32
    /// scalar fields the caller wants without a full `match`.
    #[must_use]
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Self::U32(value) => Some(*value),
            _ => None,
        }
    }

    /// Convenience accessor for string-valued fields (`general.architecture`
    /// and friends).
    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(value) => Some(value.as_str()),
            _ => None,
        }
    }
}

/// A homogeneous array value. One variant per element [`MetadataType`],
/// mirroring [`MetadataValue`] minus the nested `Array` case.
#[derive(Debug, Clone, PartialEq)]
pub enum MetadataArray {
    U8(Vec<u8>),
    I8(Vec<i8>),
    U16(Vec<u16>),
    I16(Vec<i16>),
    U32(Vec<u32>),
    I32(Vec<i32>),
    F32(Vec<f32>),
    Bool(Vec<bool>),
    String(Vec<String>),
    U64(Vec<u64>),
    I64(Vec<i64>),
    F64(Vec<f64>),
}

impl MetadataArray {
    /// Number of elements, regardless of element type.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::U8(values) => values.len(),
            Self::I8(values) => values.len(),
            Self::U16(values) => values.len(),
            Self::I16(values) => values.len(),
            Self::U32(values) => values.len(),
            Self::I32(values) => values.len(),
            Self::F32(values) => values.len(),
            Self::Bool(values) => values.len(),
            Self::String(values) => values.len(),
            Self::U64(values) => values.len(),
            Self::I64(values) => values.len(),
            Self::F64(values) => values.len(),
        }
    }

    /// Whether the array has zero elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
