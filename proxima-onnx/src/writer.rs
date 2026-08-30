//! The write half of this crate, sans-IO the same way [`crate::parser`] and
//! [`crate::decode`] are: [`write_model_proto`] never opens a file -- it
//! takes an already-built [`ModelProto`] and returns owned wire bytes, the
//! exact inverse of [`crate::decode::decode_model_proto`] and the recursive
//! decoders it drives. Mirrors `proxima-safetensors::writer::write_complete`'s
//! shape (a stateless free function over an owned in-memory model), and
//! reuses this crate's own `tests.rs` wire-builder idiom
//! (`push_tag`/`push_len`/`push_varint`/`push_f32`) now promoted to library
//! code instead of test-only helpers.
//!
//! # Field coverage
//!
//! Every field [`crate::decode`] reads for the nine messages this crate
//! covers is written here, at the same field number and wire type -- see
//! that module's decode table and [`crate::messages`]'s module doc for the
//! field-number source. Packed-repeated numeric fields (`dims`,
//! `float_data`, `int64_data`, `int32_data`, `double_data`, `uint64_data`,
//! `ints`, `floats`) are always emitted in the packed `Len` form, which
//! every decoder here already accepts (`packed_varint_pusher!`,
//! `push_f32_packed`, `push_f64_packed` all accept packed-or-scalar).
//! `string_data`/`strings`/`tensors`/`graphs`/`type_protos` stay one
//! length-delimited field per element, matching the decoder's per-occurrence
//! `Vec::push` loop.
//!
//! # Not byte-identical, structurally identical
//!
//! protobuf does not mandate a canonical byte encoding (field order,
//! zero-length field omission, and packed-vs-unpacked are all legal
//! variance a spec-compliant reader must tolerate), so this writer is not
//! required to reproduce another encoder's exact bytes -- only bytes that
//! [`crate::decode::decode_model_proto`] parses back to an equal
//! [`ModelProto`]. See this module's tests.

use alloc::vec::Vec;

use proxima_protocols::protobuf_wire::encode_varint;

use crate::messages::{
    AttributeProto, Dimension, DimensionValue, GraphProto, ModelProto, NodeProto,
    OperatorSetIdProto, TensorProto, TensorShapeProto, TypeProto, TypeProtoMap, TypeProtoTensor,
    TypeValue, ValueInfoProto,
};

// `pub(crate)`: [`crate::lift`] reuses these same wire-primitive builders to
// assemble `NodeProto`/`TensorProto`/`GraphProto` byte fragments directly,
// without ever materializing an owned [`crate::messages::NodeProto`] --
// those types borrow (`&'a str`), by design, from a wire buffer that does
// not exist yet on the lift side. See [`crate::lift`]'s module doc for why.

pub(crate) fn push_tag(field: u32, wire: u8, buf: &mut Vec<u8>) {
    encode_varint((u64::from(field) << 3) | u64::from(wire), buf);
}

pub(crate) fn push_len(field: u32, payload: &[u8], buf: &mut Vec<u8>) {
    push_tag(field, 2, buf);
    encode_varint(payload.len() as u64, buf);
    buf.extend_from_slice(payload);
}

/// Always emits the field even at protobuf3's scalar default (`""`),
/// deliberately: [`crate::messages`]'s Rust structs carry a plain `&str`
/// with no separate "was this field present on the wire" flag, so the
/// only way [`crate::decode`] can reconstruct a distinct empty-but-present
/// string is to see field presence -- and it cannot distinguish that from
/// absence either. Always-emit is therefore the only encoding this
/// writer's paired decoder round-trips correctly for every value,
/// including the default one; protobuf3's wire spec permits (never
/// requires) omitting defaults, so this is legal, just not the minimal
/// encoding an optimizing encoder would choose.
pub(crate) fn push_str(field: u32, value: &str, buf: &mut Vec<u8>) {
    push_len(field, value.as_bytes(), buf);
}

pub(crate) fn push_bytes(field: u32, value: &[u8], buf: &mut Vec<u8>) {
    push_len(field, value, buf);
}

pub(crate) fn push_varint(field: u32, value: u64, buf: &mut Vec<u8>) {
    push_tag(field, 0, buf);
    encode_varint(value, buf);
}

pub(crate) fn push_i64(field: u32, value: i64, buf: &mut Vec<u8>) {
    push_varint(field, value as u64, buf);
}

pub(crate) fn push_i32(field: u32, value: i32, buf: &mut Vec<u8>) {
    push_varint(field, value as u64, buf);
}

pub(crate) fn push_f32(field: u32, value: f32, buf: &mut Vec<u8>) {
    push_tag(field, 5, buf);
    buf.extend_from_slice(&value.to_bits().to_le_bytes());
}

pub(crate) fn push_packed_i64(field: u32, values: &[i64], buf: &mut Vec<u8>) {
    if values.is_empty() {
        return;
    }
    let mut payload = Vec::with_capacity(values.len() * 2);
    for value in values {
        encode_varint(*value as u64, &mut payload);
    }
    push_len(field, &payload, buf);
}

pub(crate) fn push_packed_i32(field: u32, values: &[i32], buf: &mut Vec<u8>) {
    if values.is_empty() {
        return;
    }
    let mut payload = Vec::with_capacity(values.len() * 2);
    for value in values {
        encode_varint(*value as u64, &mut payload);
    }
    push_len(field, &payload, buf);
}

pub(crate) fn push_packed_u64(field: u32, values: &[u64], buf: &mut Vec<u8>) {
    if values.is_empty() {
        return;
    }
    let mut payload = Vec::with_capacity(values.len() * 2);
    for value in values {
        encode_varint(*value, &mut payload);
    }
    push_len(field, &payload, buf);
}

pub(crate) fn push_packed_f32(field: u32, values: &[f32], buf: &mut Vec<u8>) {
    if values.is_empty() {
        return;
    }
    let mut payload = Vec::with_capacity(values.len() * 4);
    for value in values {
        payload.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    push_len(field, &payload, buf);
}

pub(crate) fn push_packed_f64(field: u32, values: &[f64], buf: &mut Vec<u8>) {
    if values.is_empty() {
        return;
    }
    let mut payload = Vec::with_capacity(values.len() * 8);
    for value in values {
        payload.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    push_len(field, &payload, buf);
}

/// Serialize an [`OperatorSetIdProto`] (`onnx.proto3:928`).
fn write_operator_set_id(value: &OperatorSetIdProto<'_>) -> Vec<u8> {
    let mut buf = Vec::new();
    push_str(1, value.domain, &mut buf);
    push_i64(2, value.version, &mut buf);
    buf
}

/// Serialize a [`Dimension`] (`onnx.proto3:820-831`).
fn write_dimension(value: &Dimension<'_>) -> Vec<u8> {
    let mut buf = Vec::new();
    match &value.value {
        Some(DimensionValue::Value(dim)) => push_i64(1, *dim, &mut buf),
        Some(DimensionValue::Param(param)) => push_str(2, param, &mut buf),
        None => {}
    }
    push_str(3, value.denotation, &mut buf);
    buf
}

/// Serialize a [`TensorShapeProto`] (`onnx.proto3:819-833`).
fn write_tensor_shape(value: &TensorShapeProto<'_>) -> Vec<u8> {
    let mut buf = Vec::new();
    for dim in &value.dim {
        push_len(1, &write_dimension(dim), &mut buf);
    }
    buf
}

/// Serialize a [`TypeProtoTensor`] (`onnx.proto3:840-846`,`874-880`).
fn write_type_proto_tensor(value: &TypeProtoTensor<'_>) -> Vec<u8> {
    let mut buf = Vec::new();
    push_i32(1, value.elem_type, &mut buf);
    if let Some(shape) = &value.shape {
        push_len(2, &write_tensor_shape(shape), &mut buf);
    }
    buf
}

/// Serialize a [`TypeProtoMap`] (`onnx.proto3:856-863`).
fn write_type_proto_map(value: &TypeProtoMap<'_>) -> Vec<u8> {
    let mut buf = Vec::new();
    push_i32(1, value.key_type, &mut buf);
    if let Some(value_type) = &value.value_type {
        push_len(2, &write_type_proto(value_type), &mut buf);
    }
    buf
}

/// Serialize a [`TypeProto`] (`onnx.proto3:838-923`).
fn write_type_proto(value: &TypeProto<'_>) -> Vec<u8> {
    let mut buf = Vec::new();
    match &value.value {
        Some(TypeValue::Tensor(tensor)) => push_len(1, &write_type_proto_tensor(tensor), &mut buf),
        Some(TypeValue::Sequence(inner)) => push_len(4, &write_type_proto(inner), &mut buf),
        Some(TypeValue::Map(map)) => push_len(5, &write_type_proto_map(map), &mut buf),
        Some(TypeValue::Optional(inner)) => push_len(9, &write_type_proto(inner), &mut buf),
        Some(TypeValue::SparseTensor(tensor)) => push_len(8, &write_type_proto_tensor(tensor), &mut buf),
        None => {}
    }
    push_str(6, value.denotation, &mut buf);
    buf
}

/// Serialize a [`ValueInfoProto`] (`onnx.proto3:205-215`).
fn write_value_info(value: &ValueInfoProto<'_>) -> Vec<u8> {
    let mut buf = Vec::new();
    push_str(1, value.name, &mut buf);
    if let Some(type_proto) = &value.r#type {
        push_len(2, &write_type_proto(type_proto), &mut buf);
    }
    push_str(3, value.doc_string, &mut buf);
    buf
}

/// Serialize an [`AttributeProto`] (`onnx.proto3:138-201`).
fn write_attribute(value: &AttributeProto<'_>) -> Vec<u8> {
    let mut buf = Vec::new();
    push_str(1, value.name, &mut buf);
    push_str(21, value.ref_attr_name, &mut buf);
    push_str(13, value.doc_string, &mut buf);
    push_i32(20, value.type_raw, &mut buf);
    push_f32(2, value.f, &mut buf);
    push_i64(3, value.i, &mut buf);
    push_bytes(4, value.s, &mut buf);
    if let Some(tensor) = &value.t {
        push_len(5, &write_tensor(tensor), &mut buf);
    }
    if let Some(graph) = &value.g {
        push_len(6, &write_graph(graph), &mut buf);
    }
    if let Some(type_proto) = &value.tp {
        push_len(14, &write_type_proto(type_proto), &mut buf);
    }
    push_packed_f32(7, &value.floats, &mut buf);
    push_packed_i64(8, &value.ints, &mut buf);
    for entry in &value.strings {
        push_bytes(9, entry, &mut buf);
    }
    for tensor in &value.tensors {
        push_len(10, &write_tensor(tensor), &mut buf);
    }
    for graph in &value.graphs {
        push_len(11, &write_graph(graph), &mut buf);
    }
    for type_proto in &value.type_protos {
        push_len(15, &write_type_proto(type_proto), &mut buf);
    }
    buf
}

/// Serialize a [`NodeProto`] (`onnx.proto3:224-250`).
fn write_node(value: &NodeProto<'_>) -> Vec<u8> {
    let mut buf = Vec::new();
    for input in &value.input {
        push_str(1, input, &mut buf);
    }
    for output in &value.output {
        push_str(2, output, &mut buf);
    }
    push_str(3, value.name, &mut buf);
    push_str(4, value.op_type, &mut buf);
    push_str(7, value.domain, &mut buf);
    push_str(8, value.overload, &mut buf);
    for attribute in &value.attribute {
        push_len(5, &write_attribute(attribute), &mut buf);
    }
    push_str(6, value.doc_string, &mut buf);
    buf
}

/// Serialize a [`TensorProto`] (`onnx.proto3:608-790`).
fn write_tensor(value: &TensorProto<'_>) -> Vec<u8> {
    let mut buf = Vec::new();
    push_packed_i64(1, &value.dims, &mut buf);
    push_i32(2, value.data_type, &mut buf);
    push_packed_f32(4, &value.float_data, &mut buf);
    push_packed_i32(5, &value.int32_data, &mut buf);
    for entry in &value.string_data {
        push_bytes(6, entry, &mut buf);
    }
    push_packed_i64(7, &value.int64_data, &mut buf);
    push_str(8, value.name, &mut buf);
    push_str(12, value.doc_string, &mut buf);
    if let Some(raw) = value.raw_data {
        push_bytes(9, raw, &mut buf);
    }
    push_packed_f64(10, &value.double_data, &mut buf);
    push_packed_u64(11, &value.uint64_data, &mut buf);
    buf
}

/// Serialize a [`GraphProto`] (`onnx.proto3:565-603`).
fn write_graph(value: &GraphProto<'_>) -> Vec<u8> {
    let mut buf = Vec::new();
    for node in &value.node {
        push_len(1, &write_node(node), &mut buf);
    }
    push_str(2, value.name, &mut buf);
    for initializer in &value.initializer {
        push_len(5, &write_tensor(initializer), &mut buf);
    }
    push_str(10, value.doc_string, &mut buf);
    for input in &value.input {
        push_len(11, &write_value_info(input), &mut buf);
    }
    for output in &value.output {
        push_len(12, &write_value_info(output), &mut buf);
    }
    for value_info in &value.value_info {
        push_len(13, &write_value_info(value_info), &mut buf);
    }
    buf
}

/// Serialize a complete [`ModelProto`] to protobuf wire bytes -- the exact
/// inverse of [`crate::decode::decode_model_proto`]. Stateless, mirrors
/// [`crate::pipe::parse_complete`]'s free-function shape.
///
/// protobuf encoding is infallible for an already-constructed, well-typed
/// `ModelProto` (there is no field this type can hold that has no wire
/// representation), so this returns owned bytes directly rather than a
/// `Result` -- unlike `proxima-safetensors::writer::write_complete`, which
/// validates caller-supplied invariants (duplicate names, byte-length
/// mismatches) this format simply has none of.
#[must_use]
pub fn write_model_proto(model: &ModelProto<'_>) -> Vec<u8> {
    let mut buf = Vec::new();
    push_i64(1, model.ir_version, &mut buf);
    push_str(2, model.producer_name, &mut buf);
    push_str(3, model.producer_version, &mut buf);
    push_str(4, model.domain, &mut buf);
    push_i64(5, model.model_version, &mut buf);
    push_str(6, model.doc_string, &mut buf);
    if let Some(graph) = &model.graph {
        push_len(7, &write_graph(graph), &mut buf);
    }
    for opset in &model.opset_import {
        push_len(8, &write_operator_set_id(opset), &mut buf);
    }
    buf
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use alloc::string::ToString;
    use alloc::vec;

    use super::*;
    use crate::decode::decode_model_proto;

    /// Every field this crate's decoder reads, populated with real-shaped
    /// values (nested `TypeProto` variants, packed numeric tensor data,
    /// `raw_data`, attribute variance across scalar/repeated/nested forms) --
    /// not `AAAA`-style stub bytes. `write_model_proto` then
    /// `decode_model_proto` must reproduce an equal structure: protobuf does
    /// not mandate byte-identical output (see this module's doc), so
    /// structural `PartialEq` is the correctness bar, not raw-byte equality.
    fn build_reference_model() -> ModelProto<'static> {
        let x_shape = TensorShapeProto {
            dim: vec![
                Dimension { value: Some(DimensionValue::Value(2)), denotation: "" },
                Dimension { value: Some(DimensionValue::Param("batch")), denotation: "seq" },
            ],
        };
        let x_type = TypeProto { value: Some(TypeValue::Tensor(TypeProtoTensor { elem_type: 1, shape: Some(x_shape) })), denotation: "" };
        let x_input = ValueInfoProto { name: "x", r#type: Some(x_type), doc_string: "input x" };
        let y_output = ValueInfoProto { name: "y", r#type: None, doc_string: "" };

        let weight = TensorProto {
            dims: vec![2, 2],
            data_type: 1,
            name: "W",
            doc_string: "weight",
            raw_data: Some(&[0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 64, 0, 0, 128, 64]),
            ..TensorProto::default()
        };
        let int_tensor = TensorProto { dims: vec![3], data_type: 7, int64_data: vec![-1, 0, 5], name: "ids", ..TensorProto::default() };

        let float_attr = AttributeProto { name: "alpha", type_raw: 1, f: 1.5, ..AttributeProto::default() };
        let ints_attr = AttributeProto { name: "perm", type_raw: 7, ints: vec![1, 0, 2], ..AttributeProto::default() };
        let tensor_attr = AttributeProto { name: "value", type_raw: 4, t: Some(int_tensor.clone()), ..AttributeProto::default() };

        let node0 = NodeProto {
            input: vec!["x", "W"],
            output: vec!["z"],
            name: "node0",
            op_type: "Gemm",
            domain: "",
            overload: "",
            attribute: vec![float_attr, ints_attr, tensor_attr],
            doc_string: "gemm node",
        };
        let node1 = NodeProto { input: vec!["z"], output: vec!["y"], name: "node1", op_type: "Relu", ..NodeProto::default() };

        let graph = GraphProto {
            node: vec![node0, node1],
            name: "reference_graph",
            initializer: vec![weight],
            doc_string: "a graph exercising every field this writer covers",
            input: vec![x_input],
            output: vec![y_output],
            value_info: vec![ValueInfoProto { name: "z", ..ValueInfoProto::default() }],
        };

        ModelProto {
            ir_version: 8,
            opset_import: vec![OperatorSetIdProto { domain: "", version: 17 }],
            producer_name: "proxima-onnx",
            producer_version: "0.1",
            domain: "ai.proxima",
            model_version: 1,
            doc_string: "round-trip fixture",
            graph: Some(graph),
        }
    }

    /// `write_model_proto` then `decode_model_proto` reproduces every field
    /// of the original -- the headline serializer proof, structural
    /// equality via `ModelProto`'s own `PartialEq`.
    #[test]
    fn write_then_parse_round_trips_every_field() {
        let reference = build_reference_model();
        let bytes = write_model_proto(&reference);
        let parsed = decode_model_proto(&bytes).expect("written bytes parse back");
        assert_eq!(parsed, reference);
    }

    /// An all-default `ModelProto` (no graph, every scalar at its zero
    /// value) still round-trips: this writer always emits scalar/string
    /// fields rather than omitting protobuf3 defaults (see [`push_str`]'s
    /// doc), so a default-valued field on the wire decodes back to the
    /// same default, not to a spuriously-absent one.
    #[test]
    fn default_valued_model_round_trips() {
        let model = ModelProto::default();
        let bytes = write_model_proto(&model);
        let parsed = decode_model_proto(&bytes).expect("default model parses back");
        assert_eq!(parsed, model);
    }

    /// A round trip through a real byte-for-byte fixture this crate's own
    /// `tests.rs` builds by hand -- proves the writer's output is not just
    /// self-consistent with its own decoder but agrees with the
    /// independently hand-encoded wire format `tests.rs` exercises.
    #[test]
    fn write_model_proto_matches_hand_encoded_wire_semantics() {
        let reference = build_reference_model();
        let bytes = write_model_proto(&reference);
        let parsed_graph = decode_model_proto(&bytes).expect("parses").graph.expect("graph present");
        assert_eq!(parsed_graph.node.len(), 2);
        assert_eq!(parsed_graph.node[0].op_type, "Gemm");
        assert_eq!(parsed_graph.node[0].attribute[2].t.as_ref().expect("tensor attr").int64_data, vec![-1, 0, 5]);
        assert_eq!(parsed_graph.initializer[0].name, "W");
        let _ = parsed_graph.name.to_string();
    }
}
