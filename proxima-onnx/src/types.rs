//! Enum discriminants defined by `onnx.proto`. Field numbers and variant
//! values sourced from
//! `https://raw.githubusercontent.com/onnx/onnx/main/onnx/onnx.proto3`
//! (fetched 2026-08-18, `main` branch).
//!
//! Both enums are decoded from a raw wire `int32`/`i32` and kept alongside
//! that raw value on the owning message (`TensorProto::data_type_raw`,
//! `AttributeProto::type_raw`) -- converting the discriminant to a typed
//! Rust enum is not the same operation as converting tensor *values*, which
//! this crate never does (raw_data stays bytes).

/// `TensorProto.DataType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum DataType {
    Undefined = 0,
    Float = 1,
    Uint8 = 2,
    Int8 = 3,
    Uint16 = 4,
    Int16 = 5,
    Int32 = 6,
    Int64 = 7,
    String = 8,
    Bool = 9,
    Float16 = 10,
    Double = 11,
    Uint32 = 12,
    Uint64 = 13,
    Complex64 = 14,
    Complex128 = 15,
    Bfloat16 = 16,
    Float8E4M3Fn = 17,
    Float8E4M3Fnuz = 18,
    Float8E5M2 = 19,
    Float8E5M2Fnuz = 20,
    Uint4 = 21,
    Int4 = 22,
    Float4E2M1 = 23,
    Float8E8M0 = 24,
    Uint2 = 25,
    Int2 = 26,
}

impl DataType {
    #[must_use]
    pub fn from_wire(raw: i32) -> Option<Self> {
        Some(match raw {
            0 => Self::Undefined,
            1 => Self::Float,
            2 => Self::Uint8,
            3 => Self::Int8,
            4 => Self::Uint16,
            5 => Self::Int16,
            6 => Self::Int32,
            7 => Self::Int64,
            8 => Self::String,
            9 => Self::Bool,
            10 => Self::Float16,
            11 => Self::Double,
            12 => Self::Uint32,
            13 => Self::Uint64,
            14 => Self::Complex64,
            15 => Self::Complex128,
            16 => Self::Bfloat16,
            17 => Self::Float8E4M3Fn,
            18 => Self::Float8E4M3Fnuz,
            19 => Self::Float8E5M2,
            20 => Self::Float8E5M2Fnuz,
            21 => Self::Uint4,
            22 => Self::Int4,
            23 => Self::Float4E2M1,
            24 => Self::Float8E8M0,
            25 => Self::Uint2,
            26 => Self::Int2,
            _ => return None,
        })
    }
}

/// `AttributeProto.AttributeType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum AttributeType {
    Undefined = 0,
    Float = 1,
    Int = 2,
    String = 3,
    Tensor = 4,
    Graph = 5,
    SparseTensor = 11,
    TypeProto = 13,
    Floats = 6,
    Ints = 7,
    Strings = 8,
    Tensors = 9,
    Graphs = 10,
    SparseTensors = 12,
    TypeProtos = 14,
}

impl AttributeType {
    #[must_use]
    pub fn from_wire(raw: i32) -> Option<Self> {
        Some(match raw {
            0 => Self::Undefined,
            1 => Self::Float,
            2 => Self::Int,
            3 => Self::String,
            4 => Self::Tensor,
            5 => Self::Graph,
            11 => Self::SparseTensor,
            13 => Self::TypeProto,
            6 => Self::Floats,
            7 => Self::Ints,
            8 => Self::Strings,
            9 => Self::Tensors,
            10 => Self::Graphs,
            12 => Self::SparseTensors,
            14 => Self::TypeProtos,
            _ => return None,
        })
    }
}
