//! Faithful Rust structs for the nine `onnx.proto` messages this crate
//! covers. Field numbers sourced from
//! `https://raw.githubusercontent.com/onnx/onnx/main/onnx/onnx.proto3`
//! (fetched 2026-08-18, `main` branch, IR_VERSION 14 in progress).
//!
//! Every string/bytes field borrows from the input buffer -- `&'a str` /
//! `&'a [u8]`, never `String`/`Vec<u8>`. [`TensorProto::raw_data`] in
//! particular is exposed as a borrowed byte range with its declared
//! [`crate::types::DataType`] and `dims`; this crate never reinterprets
//! those bytes.
//!
//! # Scope
//!
//! Only the nine messages named in this module's doc are decoded.
//! Everything else `onnx.proto` defines --
//! `SparseTensorProto`/`FunctionProto`/`TrainingInfoProto`/
//! `DeviceConfigurationProto`/multi-device sharding/`metadata_props` on any
//! message/`TypeProto.Opaque` -- is unrecognized field traffic to this
//! parser and is skipped exactly like any other unknown field number
//! (protobuf forward-compatibility, not a parsing gap).

use alloc::boxed::Box;
use alloc::vec::Vec;

/// `OperatorSetIdProto` (`onnx.proto3:928`).
#[derive(Debug, Clone, PartialEq)]
pub struct OperatorSetIdProto<'a> {
    pub domain: &'a str,
    pub version: i64,
}

/// `TensorShapeProto.Dimension`'s `oneof value` (`onnx.proto3:821-824`).
#[derive(Debug, Clone, PartialEq)]
pub enum DimensionValue<'a> {
    Value(i64),
    Param(&'a str),
}

/// `TensorShapeProto.Dimension` (`onnx.proto3:820-831`).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Dimension<'a> {
    pub value: Option<DimensionValue<'a>>,
    pub denotation: &'a str,
}

/// `TensorShapeProto` (`onnx.proto3:819-833`).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TensorShapeProto<'a> {
    pub dim: Vec<Dimension<'a>>,
}

/// `TypeProto.Tensor` and the structurally identical `TypeProto.SparseTensor`
/// (`onnx.proto3:840-846`, `874-880`).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TypeProtoTensor<'a> {
    pub elem_type: i32,
    pub shape: Option<TensorShapeProto<'a>>,
}

/// `TypeProto.Map` (`onnx.proto3:856-863`).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TypeProtoMap<'a> {
    pub key_type: i32,
    pub value_type: Option<Box<TypeProto<'a>>>,
}

/// `TypeProto`'s `oneof value` (`onnx.proto3:892-916`). `Sequence`,
/// `Optional`, and `Map::value_type` all nest a `TypeProto` directly (no
/// `repeated`/`Vec` indirection), so those arms box it to give `TypeProto`
/// a finite size.
#[derive(Debug, Clone, PartialEq)]
pub enum TypeValue<'a> {
    Tensor(TypeProtoTensor<'a>),
    Sequence(Box<TypeProto<'a>>),
    Map(TypeProtoMap<'a>),
    Optional(Box<TypeProto<'a>>),
    SparseTensor(TypeProtoTensor<'a>),
}

/// `TypeProto` (`onnx.proto3:838-923`).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TypeProto<'a> {
    pub value: Option<TypeValue<'a>>,
    pub denotation: &'a str,
}

/// `ValueInfoProto` (`onnx.proto3:205-215`).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ValueInfoProto<'a> {
    pub name: &'a str,
    pub r#type: Option<TypeProto<'a>>,
    pub doc_string: &'a str,
}

/// `AttributeProto` (`onnx.proto3:138-201`). Exactly one of `f`/`i`/`s`/`t`/
/// `g`/`sparse_tensor`/`tp`, or one of the `repeated` fields, is meaningful
/// per instance -- `type_raw` (wire field 20) is the discriminator, decoded
/// via [`crate::types::AttributeType::from_wire`].
#[derive(Debug, Clone, PartialEq, Default)]
pub struct AttributeProto<'a> {
    pub name: &'a str,
    pub ref_attr_name: &'a str,
    pub doc_string: &'a str,
    pub type_raw: i32,
    pub f: f32,
    pub i: i64,
    pub s: &'a [u8],
    pub t: Option<TensorProto<'a>>,
    pub g: Option<GraphProto<'a>>,
    pub tp: Option<TypeProto<'a>>,
    pub floats: Vec<f32>,
    pub ints: Vec<i64>,
    pub strings: Vec<&'a [u8]>,
    pub tensors: Vec<TensorProto<'a>>,
    pub graphs: Vec<GraphProto<'a>>,
    pub type_protos: Vec<TypeProto<'a>>,
}

/// `NodeProto` (`onnx.proto3:224-250`).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct NodeProto<'a> {
    pub input: Vec<&'a str>,
    pub output: Vec<&'a str>,
    pub name: &'a str,
    pub op_type: &'a str,
    pub domain: &'a str,
    pub overload: &'a str,
    pub attribute: Vec<AttributeProto<'a>>,
    pub doc_string: &'a str,
}

/// `TensorProto` (`onnx.proto3:608-790`). `raw_data` is the field that
/// forces every string/bytes field in this crate to borrow: exposing it as
/// `&'a [u8]` costs nothing extra once every sibling field already borrows,
/// and copying it into an owned `Vec<u8>` would mean copying model weights.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TensorProto<'a> {
    pub dims: Vec<i64>,
    pub data_type: i32,
    pub float_data: Vec<f32>,
    pub int32_data: Vec<i32>,
    pub string_data: Vec<&'a [u8]>,
    pub int64_data: Vec<i64>,
    pub name: &'a str,
    pub doc_string: &'a str,
    pub raw_data: Option<&'a [u8]>,
    pub double_data: Vec<f64>,
    pub uint64_data: Vec<u64>,
}

/// `GraphProto` (`onnx.proto3:565-603`).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct GraphProto<'a> {
    pub node: Vec<NodeProto<'a>>,
    pub name: &'a str,
    pub initializer: Vec<TensorProto<'a>>,
    pub doc_string: &'a str,
    pub input: Vec<ValueInfoProto<'a>>,
    pub output: Vec<ValueInfoProto<'a>>,
    pub value_info: Vec<ValueInfoProto<'a>>,
}

/// `ModelProto` (`onnx.proto3:450-527`), the top-level message an `.onnx`
/// file is one instance of, with no outer framing.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ModelProto<'a> {
    pub ir_version: i64,
    pub opset_import: Vec<OperatorSetIdProto<'a>>,
    pub producer_name: &'a str,
    pub producer_version: &'a str,
    pub domain: &'a str,
    pub model_version: i64,
    pub doc_string: &'a str,
    pub graph: Option<GraphProto<'a>>,
}
