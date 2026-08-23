//! Recursive decoders over an already-complete message slice.
//!
//! Every function here assumes its whole `buf` argument is present in
//! memory -- no `NeedMore` concept exists at this layer, unlike
//! [`crate::parser::OnnxParser`], which buffers a top-level
//! [`ModelProto`] field until it is fully
//! present *before* handing its bytes to [`decode_model_field`]. Any
//! [`proxima_protocols::protobuf_wire::ParseError`] raised here (a length
//! prefix claiming more bytes than the slice actually has, an overflowed
//! varint, a deprecated wire type) is therefore a genuine malformed-input
//! error, never a chunking artifact.
//!
//! [`decode_model_field`] is the seam the incremental FSM and the
//! whole-slice [`decode_model_proto`] both drive -- one switch statement,
//! not two copies of it.

use alloc::boxed::Box;
use alloc::vec::Vec;

use proxima_protocols::protobuf_wire::{Field, Fields, ParseError as WireError, decode_varint};

use crate::error::OnnxError;
use crate::messages::{
    AttributeProto, Dimension, DimensionValue, GraphProto, ModelProto, NodeProto,
    OperatorSetIdProto, TensorProto, TensorShapeProto, TypeProto, TypeProtoMap, TypeProtoTensor,
    TypeValue, ValueInfoProto,
};

/// One decoded top-level [`ModelProto`] field. This is also the FSM's event
/// type, carried as `Some(ModelField)` from [`crate::parser::OnnxParser::poll`]
/// -- see this module's doc.
#[derive(Debug, Clone, PartialEq)]
pub enum ModelField<'a> {
    IrVersion(i64),
    OpsetImport(OperatorSetIdProto<'a>),
    ProducerName(&'a str),
    ProducerVersion(&'a str),
    Domain(&'a str),
    ModelVersion(i64),
    DocString(&'a str),
    Graph(GraphProto<'a>),
    /// A field number this crate does not decode (out of scope, or a future
    /// opset ONNX adds after this parser was written). Never an error --
    /// protobuf forward-compatibility requires unrecognized fields to be
    /// skipped, not rejected.
    Unknown,
}

fn wire_name(field: &Field<'_>) -> &'static str {
    match field {
        Field::Varint { .. } => "varint",
        Field::I64 { .. } => "i64",
        Field::Len { .. } => "len",
        Field::I32 { .. } => "i32",
    }
}

fn expect_str<'a>(field: Field<'a>) -> Result<&'a str, OnnxError> {
    let field_number = field.field_number();
    match field {
        Field::Len { payload, .. } => {
            core::str::from_utf8(payload).map_err(|_| OnnxError::InvalidUtf8 { field: field_number })
        }
        other => Err(OnnxError::WireTypeMismatch {
            field: field_number,
            expected: "len",
            found: wire_name(&other),
        }),
    }
}

fn expect_bytes<'a>(field: Field<'a>) -> Result<&'a [u8], OnnxError> {
    let field_number = field.field_number();
    match field {
        Field::Len { payload, .. } => Ok(payload),
        other => Err(OnnxError::WireTypeMismatch {
            field: field_number,
            expected: "len",
            found: wire_name(&other),
        }),
    }
}

fn expect_i64(field: Field<'_>) -> Result<i64, OnnxError> {
    let field_number = field.field_number();
    match field {
        Field::Varint { value, .. } => Ok(value as i64),
        other => Err(OnnxError::WireTypeMismatch {
            field: field_number,
            expected: "varint",
            found: wire_name(&other),
        }),
    }
}

fn expect_i32(field: Field<'_>) -> Result<i32, OnnxError> {
    let field_number = field.field_number();
    match field {
        Field::Varint { value, .. } => Ok(value as i32),
        other => Err(OnnxError::WireTypeMismatch {
            field: field_number,
            expected: "varint",
            found: wire_name(&other),
        }),
    }
}

fn expect_f32(field: Field<'_>) -> Result<f32, OnnxError> {
    let field_number = field.field_number();
    match field {
        Field::I32 { value, .. } => Ok(f32::from_bits(u32::from_le_bytes(value))),
        other => Err(OnnxError::WireTypeMismatch {
            field: field_number,
            expected: "i32",
            found: wire_name(&other),
        }),
    }
}

macro_rules! packed_varint_pusher {
    ($name:ident, $ty:ty) => {
        fn $name(field: Field<'_>, out: &mut Vec<$ty>) -> Result<(), OnnxError> {
            let field_number = field.field_number();
            match field {
                Field::Varint { value, .. } => {
                    out.push(value as $ty);
                    Ok(())
                }
                Field::Len { payload, .. } => {
                    let mut cursor = payload;
                    while !cursor.is_empty() {
                        let (value, used) = decode_varint(cursor)
                            .map_err(|source| OnnxError::Wire { field: field_number, source })?;
                        out.push(value as $ty);
                        cursor = &cursor[used..];
                    }
                    Ok(())
                }
                other => Err(OnnxError::WireTypeMismatch {
                    field: field_number,
                    expected: "varint or len (packed varint)",
                    found: wire_name(&other),
                }),
            }
        }
    };
}

packed_varint_pusher!(push_i64_packed, i64);
packed_varint_pusher!(push_i32_packed, i32);
packed_varint_pusher!(push_u64_packed, u64);

fn push_f32_packed(field: Field<'_>, out: &mut Vec<f32>) -> Result<(), OnnxError> {
    let field_number = field.field_number();
    match field {
        Field::I32 { value, .. } => {
            out.push(f32::from_bits(u32::from_le_bytes(value)));
            Ok(())
        }
        Field::Len { payload, .. } => {
            if payload.len() % 4 != 0 {
                return Err(OnnxError::WireTypeMismatch {
                    field: field_number,
                    expected: "packed f32 (multiple of 4 bytes)",
                    found: "len (unaligned)",
                });
            }
            for chunk in payload.as_chunks::<4>().0 {
                out.push(f32::from_bits(u32::from_le_bytes(*chunk)));
            }
            Ok(())
        }
        other => Err(OnnxError::WireTypeMismatch {
            field: field_number,
            expected: "i32 or len (packed)",
            found: wire_name(&other),
        }),
    }
}

fn push_f64_packed(field: Field<'_>, out: &mut Vec<f64>) -> Result<(), OnnxError> {
    let field_number = field.field_number();
    match field {
        Field::I64 { value, .. } => {
            out.push(f64::from_bits(u64::from_le_bytes(value)));
            Ok(())
        }
        Field::Len { payload, .. } => {
            if payload.len() % 8 != 0 {
                return Err(OnnxError::WireTypeMismatch {
                    field: field_number,
                    expected: "packed f64 (multiple of 8 bytes)",
                    found: "len (unaligned)",
                });
            }
            for chunk in payload.as_chunks::<8>().0 {
                out.push(f64::from_bits(u64::from_le_bytes(*chunk)));
            }
            Ok(())
        }
        other => Err(OnnxError::WireTypeMismatch {
            field: field_number,
            expected: "i64 or len (packed)",
            found: wire_name(&other),
        }),
    }
}

fn each_field<'a>(buf: &'a [u8], mut visit: impl FnMut(Field<'a>) -> Result<(), OnnxError>) -> Result<(), OnnxError> {
    for field in Fields::new(buf) {
        let field: Field<'a> = field.map_err(|source: WireError| OnnxError::Wire { field: 0, source })?;
        visit(field)?;
    }
    Ok(())
}

fn decode_operator_set_id(buf: &[u8]) -> Result<OperatorSetIdProto<'_>, OnnxError> {
    let mut out = OperatorSetIdProto { domain: "", version: 0 };
    each_field(buf, |field| {
        match field.field_number() {
            1 => out.domain = expect_str(field)?,
            2 => out.version = expect_i64(field)?,
            _ => {}
        }
        Ok(())
    })?;
    Ok(out)
}

fn decode_dimension(buf: &[u8]) -> Result<Dimension<'_>, OnnxError> {
    let mut out = Dimension::default();
    each_field(buf, |field| {
        match field.field_number() {
            1 => out.value = Some(DimensionValue::Value(expect_i64(field)?)),
            2 => out.value = Some(DimensionValue::Param(expect_str(field)?)),
            3 => out.denotation = expect_str(field)?,
            _ => {}
        }
        Ok(())
    })?;
    Ok(out)
}

fn decode_tensor_shape(buf: &[u8]) -> Result<TensorShapeProto<'_>, OnnxError> {
    let mut out = TensorShapeProto::default();
    each_field(buf, |field| {
        if field.field_number() == 1 {
            out.dim.push(decode_dimension(expect_bytes(field)?)?);
        }
        Ok(())
    })?;
    Ok(out)
}

fn decode_type_proto_tensor(buf: &[u8]) -> Result<TypeProtoTensor<'_>, OnnxError> {
    let mut out = TypeProtoTensor::default();
    each_field(buf, |field| {
        match field.field_number() {
            1 => out.elem_type = expect_i32(field)?,
            2 => out.shape = Some(decode_tensor_shape(expect_bytes(field)?)?),
            _ => {}
        }
        Ok(())
    })?;
    Ok(out)
}

fn decode_type_proto_map(buf: &[u8]) -> Result<TypeProtoMap<'_>, OnnxError> {
    let mut out = TypeProtoMap::default();
    each_field(buf, |field| {
        match field.field_number() {
            1 => out.key_type = expect_i32(field)?,
            2 => out.value_type = Some(Box::new(decode_type_proto(expect_bytes(field)?)?)),
            _ => {}
        }
        Ok(())
    })?;
    Ok(out)
}

fn decode_type_proto(buf: &[u8]) -> Result<TypeProto<'_>, OnnxError> {
    let mut out = TypeProto::default();
    each_field(buf, |field| {
        match field.field_number() {
            1 => out.value = Some(TypeValue::Tensor(decode_type_proto_tensor(expect_bytes(field)?)?)),
            4 => out.value = Some(TypeValue::Sequence(Box::new(decode_type_proto(expect_bytes(field)?)?))),
            5 => out.value = Some(TypeValue::Map(decode_type_proto_map(expect_bytes(field)?)?)),
            9 => out.value = Some(TypeValue::Optional(Box::new(decode_type_proto(expect_bytes(field)?)?))),
            8 => out.value = Some(TypeValue::SparseTensor(decode_type_proto_tensor(expect_bytes(field)?)?)),
            6 => out.denotation = expect_str(field)?,
            _ => {}
        }
        Ok(())
    })?;
    Ok(out)
}

fn decode_value_info(buf: &[u8]) -> Result<ValueInfoProto<'_>, OnnxError> {
    let mut out = ValueInfoProto::default();
    each_field(buf, |field| {
        match field.field_number() {
            1 => out.name = expect_str(field)?,
            2 => out.r#type = Some(decode_type_proto(expect_bytes(field)?)?),
            3 => out.doc_string = expect_str(field)?,
            _ => {}
        }
        Ok(())
    })?;
    Ok(out)
}

fn decode_attribute(buf: &[u8]) -> Result<AttributeProto<'_>, OnnxError> {
    let mut out = AttributeProto::default();
    each_field(buf, |field| {
        match field.field_number() {
            1 => out.name = expect_str(field)?,
            21 => out.ref_attr_name = expect_str(field)?,
            13 => out.doc_string = expect_str(field)?,
            20 => out.type_raw = expect_i32(field)?,
            2 => out.f = expect_f32(field)?,
            3 => out.i = expect_i64(field)?,
            4 => out.s = expect_bytes(field)?,
            5 => out.t = Some(decode_tensor(expect_bytes(field)?)?),
            6 => out.g = Some(decode_graph_proto(expect_bytes(field)?)?),
            14 => out.tp = Some(decode_type_proto(expect_bytes(field)?)?),
            7 => push_f32_packed(field, &mut out.floats)?,
            8 => push_i64_packed(field, &mut out.ints)?,
            9 => out.strings.push(expect_bytes(field)?),
            10 => out.tensors.push(decode_tensor(expect_bytes(field)?)?),
            11 => out.graphs.push(decode_graph_proto(expect_bytes(field)?)?),
            15 => out.type_protos.push(decode_type_proto(expect_bytes(field)?)?),
            _ => {}
        }
        Ok(())
    })?;
    Ok(out)
}

fn decode_node(buf: &[u8]) -> Result<NodeProto<'_>, OnnxError> {
    let mut out = NodeProto::default();
    each_field(buf, |field| {
        match field.field_number() {
            1 => out.input.push(expect_str(field)?),
            2 => out.output.push(expect_str(field)?),
            3 => out.name = expect_str(field)?,
            4 => out.op_type = expect_str(field)?,
            7 => out.domain = expect_str(field)?,
            8 => out.overload = expect_str(field)?,
            5 => out.attribute.push(decode_attribute(expect_bytes(field)?)?),
            6 => out.doc_string = expect_str(field)?,
            _ => {}
        }
        Ok(())
    })?;
    Ok(out)
}

fn decode_tensor(buf: &[u8]) -> Result<TensorProto<'_>, OnnxError> {
    let mut out = TensorProto::default();
    each_field(buf, |field| {
        match field.field_number() {
            1 => push_i64_packed(field, &mut out.dims)?,
            2 => out.data_type = expect_i32(field)?,
            4 => push_f32_packed(field, &mut out.float_data)?,
            5 => push_i32_packed(field, &mut out.int32_data)?,
            6 => out.string_data.push(expect_bytes(field)?),
            7 => push_i64_packed(field, &mut out.int64_data)?,
            8 => out.name = expect_str(field)?,
            12 => out.doc_string = expect_str(field)?,
            9 => out.raw_data = Some(expect_bytes(field)?),
            10 => push_f64_packed(field, &mut out.double_data)?,
            11 => push_u64_packed(field, &mut out.uint64_data)?,
            _ => {}
        }
        Ok(())
    })?;
    Ok(out)
}

fn decode_graph_proto(buf: &[u8]) -> Result<GraphProto<'_>, OnnxError> {
    let mut out = GraphProto::default();
    each_field(buf, |field| {
        match field.field_number() {
            1 => out.node.push(decode_node(expect_bytes(field)?)?),
            2 => out.name = expect_str(field)?,
            5 => out.initializer.push(decode_tensor(expect_bytes(field)?)?),
            10 => out.doc_string = expect_str(field)?,
            11 => out.input.push(decode_value_info(expect_bytes(field)?)?),
            12 => out.output.push(decode_value_info(expect_bytes(field)?)?),
            13 => out.value_info.push(decode_value_info(expect_bytes(field)?)?),
            _ => {}
        }
        Ok(())
    })?;
    Ok(out)
}

/// Decode one top-level `ModelProto` field. Shared by [`decode_model_proto`]
/// (the whole-slice path) and [`crate::parser::OnnxParser::poll`] (the
/// chunked FSM path) -- one decode table, driven two ways.
pub fn decode_model_field(field: Field<'_>) -> Result<ModelField<'_>, OnnxError> {
    Ok(match field.field_number() {
        1 => ModelField::IrVersion(expect_i64(field)?),
        8 => ModelField::OpsetImport(decode_operator_set_id(expect_bytes(field)?)?),
        2 => ModelField::ProducerName(expect_str(field)?),
        3 => ModelField::ProducerVersion(expect_str(field)?),
        4 => ModelField::Domain(expect_str(field)?),
        5 => ModelField::ModelVersion(expect_i64(field)?),
        6 => ModelField::DocString(expect_str(field)?),
        7 => ModelField::Graph(decode_graph_proto(expect_bytes(field)?)?),
        _ => ModelField::Unknown,
    })
}

/// Decode a complete, already-assembled `ModelProto` byte slice in one
/// pass. Every string/bytes field in the result borrows from `buf`.
///
/// # Errors
///
/// Any [`OnnxError`] the field decoders above surface for malformed wire
/// content.
pub fn decode_model_proto(buf: &[u8]) -> Result<ModelProto<'_>, OnnxError> {
    let mut model = ModelProto::default();
    each_field(buf, |field| {
        match decode_model_field(field)? {
            ModelField::IrVersion(value) => model.ir_version = value,
            ModelField::OpsetImport(value) => model.opset_import.push(value),
            ModelField::ProducerName(value) => model.producer_name = value,
            ModelField::ProducerVersion(value) => model.producer_version = value,
            ModelField::Domain(value) => model.domain = value,
            ModelField::ModelVersion(value) => model.model_version = value,
            ModelField::DocString(value) => model.doc_string = value,
            ModelField::Graph(value) => model.graph = Some(value),
            ModelField::Unknown => {}
        }
        Ok(())
    })?;
    Ok(model)
}
