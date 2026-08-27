//! Crate-level integration tests: a hand-built synthetic `ModelProto`
//! (two nodes, attributes of several types including a repeated field, an
//! initializer with `raw_data`, graph inputs/outputs), driven through both
//! the single-shot [`crate::parse_complete`] path and the raw
//! [`crate::parser::OnnxParser`] FSM split at arbitrary chunk boundaries.
//! Every byte is hand-encoded protobuf wire format, not `AAAA`-style stub
//! bytes.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use alloc::vec::Vec;

use proxima_protocols::protobuf_wire::encode_varint;

use crate::decode::{ModelField, decode_model_field};
use crate::error::OnnxError;
use crate::messages::{DimensionValue, TypeValue};
use crate::parser::OnnxParser;
use crate::pipe::parse_complete;

// -- wire-format byte builders (hand-rolled, mirrors what a real ONNX
// writer emits: tag = (field << 3) | wire_type, LEB128 varints throughout).

fn push_tag(field: u32, wire: u8, buf: &mut Vec<u8>) {
    encode_varint((u64::from(field) << 3) | u64::from(wire), buf);
}

fn push_len(field: u32, payload: &[u8], buf: &mut Vec<u8>) {
    push_tag(field, 2, buf);
    encode_varint(payload.len() as u64, buf);
    buf.extend_from_slice(payload);
}

fn push_str(field: u32, value: &str, buf: &mut Vec<u8>) {
    push_len(field, value.as_bytes(), buf);
}

fn push_varint(field: u32, value: u64, buf: &mut Vec<u8>) {
    push_tag(field, 0, buf);
    encode_varint(value, buf);
}

fn push_f32(field: u32, value: f32, buf: &mut Vec<u8>) {
    push_tag(field, 5, buf);
    buf.extend_from_slice(&value.to_bits().to_le_bytes());
}

fn push_packed_i64(field: u32, values: &[i64], buf: &mut Vec<u8>) {
    let mut payload = Vec::new();
    for value in values {
        encode_varint(*value as u64, &mut payload);
    }
    push_len(field, &payload, buf);
}

// -- message builders, one per onnx.proto message this crate decodes.

fn build_operator_set_id(domain: &str, version: i64) -> Vec<u8> {
    let mut buf = Vec::new();
    push_str(1, domain, &mut buf);
    push_varint(2, version as u64, &mut buf);
    buf
}

fn build_dimension_value(value: i64) -> Vec<u8> {
    let mut buf = Vec::new();
    push_varint(1, value as u64, &mut buf);
    buf
}

fn build_dimension_param(param: &str) -> Vec<u8> {
    let mut buf = Vec::new();
    push_str(2, param, &mut buf);
    buf
}

fn build_tensor_shape(dims: &[Vec<u8>]) -> Vec<u8> {
    let mut buf = Vec::new();
    for dim in dims {
        push_len(1, dim, &mut buf);
    }
    buf
}

fn build_type_proto_tensor(elem_type: i32, shape: &[u8]) -> Vec<u8> {
    let mut buf = Vec::new();
    push_varint(1, elem_type as u64, &mut buf);
    push_len(2, shape, &mut buf);
    buf
}

fn build_type_proto(tensor_type: &[u8]) -> Vec<u8> {
    let mut buf = Vec::new();
    push_len(1, tensor_type, &mut buf);
    buf
}

fn build_value_info(name: &str, type_proto: &[u8], doc_string: &str) -> Vec<u8> {
    let mut buf = Vec::new();
    push_str(1, name, &mut buf);
    push_len(2, type_proto, &mut buf);
    push_str(3, doc_string, &mut buf);
    buf
}

struct TensorFixture<'a> {
    dims: &'a [i64],
    data_type: i32,
    name: &'a str,
    doc_string: &'a str,
    raw_data: &'a [u8],
}

fn build_tensor(fixture: &TensorFixture<'_>) -> Vec<u8> {
    let mut buf = Vec::new();
    push_packed_i64(1, fixture.dims, &mut buf);
    push_varint(2, fixture.data_type as u64, &mut buf);
    push_str(8, fixture.name, &mut buf);
    push_str(12, fixture.doc_string, &mut buf);
    push_len(9, fixture.raw_data, &mut buf);
    buf
}

fn build_attribute_float(name: &str, value: f32) -> Vec<u8> {
    let mut buf = Vec::new();
    push_str(1, name, &mut buf);
    push_varint(20, 1, &mut buf); // AttributeType::FLOAT
    push_f32(2, value, &mut buf);
    buf
}

fn build_attribute_int(name: &str, value: i64) -> Vec<u8> {
    let mut buf = Vec::new();
    push_str(1, name, &mut buf);
    push_varint(20, 2, &mut buf); // AttributeType::INT
    push_varint(3, value as u64, &mut buf);
    buf
}

fn build_attribute_string(name: &str, value: &[u8]) -> Vec<u8> {
    let mut buf = Vec::new();
    push_str(1, name, &mut buf);
    push_varint(20, 3, &mut buf); // AttributeType::STRING
    push_len(4, value, &mut buf);
    buf
}

fn build_attribute_tensor(name: &str, tensor: &[u8]) -> Vec<u8> {
    let mut buf = Vec::new();
    push_str(1, name, &mut buf);
    push_varint(20, 4, &mut buf); // AttributeType::TENSOR
    push_len(5, tensor, &mut buf);
    buf
}

fn build_attribute_ints(name: &str, values: &[i64]) -> Vec<u8> {
    let mut buf = Vec::new();
    push_str(1, name, &mut buf);
    push_varint(20, 7, &mut buf); // AttributeType::INTS
    push_packed_i64(8, values, &mut buf);
    buf
}

struct NodeFixture<'a> {
    input: &'a [&'a str],
    output: &'a [&'a str],
    name: &'a str,
    op_type: &'a str,
    doc_string: &'a str,
    attributes: &'a [Vec<u8>],
}

fn build_node(fixture: &NodeFixture<'_>) -> Vec<u8> {
    let mut buf = Vec::new();
    for value in fixture.input {
        push_str(1, value, &mut buf);
    }
    for value in fixture.output {
        push_str(2, value, &mut buf);
    }
    push_str(3, fixture.name, &mut buf);
    push_str(4, fixture.op_type, &mut buf);
    for attribute in fixture.attributes {
        push_len(5, attribute, &mut buf);
    }
    push_str(6, fixture.doc_string, &mut buf);
    buf
}

fn build_graph(
    nodes: &[Vec<u8>],
    name: &str,
    initializers: &[Vec<u8>],
    doc_string: &str,
    inputs: &[Vec<u8>],
    outputs: &[Vec<u8>],
) -> Vec<u8> {
    let mut buf = Vec::new();
    for node in nodes {
        push_len(1, node, &mut buf);
    }
    push_str(2, name, &mut buf);
    for initializer in initializers {
        push_len(5, initializer, &mut buf);
    }
    push_str(10, doc_string, &mut buf);
    for input in inputs {
        push_len(11, input, &mut buf);
    }
    for output in outputs {
        push_len(12, output, &mut buf);
    }
    buf
}

/// Everything the synthetic model asserts about itself after parsing, kept
/// alongside the bytes so the round-trip test and the chunk-boundary test
/// can both check against one source of truth.
struct Fixture {
    bytes: Vec<u8>,
    weight_raw_data: [u8; 16],
    initializer_tensor_raw_data: [u8; 4],
}

fn build_fixture() -> Fixture {
    let weight_raw_data: [u8; 16] = {
        let mut bytes = [0u8; 16];
        for (index, value) in [1.0f32, 2.0, 3.0, 4.0].iter().enumerate() {
            bytes[index * 4..index * 4 + 4].copy_from_slice(&value.to_le_bytes());
        }
        bytes
    };
    let initializer_tensor_raw_data: [u8; 4] = 0.5f32.to_le_bytes();

    let x_shape = build_tensor_shape(&[build_dimension_value(1), build_dimension_param("batch")]);
    let x_type = build_type_proto(&build_type_proto_tensor(1, &x_shape));
    let x_input = build_value_info("x", &x_type, "input x");

    let y_shape = build_tensor_shape(&[build_dimension_value(1)]);
    let y_type = build_type_proto(&build_type_proto_tensor(1, &y_shape));
    let y_output = build_value_info("y", &y_type, "output y");

    let weight_tensor = build_tensor(&TensorFixture {
        dims: &[2, 2],
        data_type: 1,
        name: "W",
        doc_string: "weight",
        raw_data: &weight_raw_data,
    });

    let const_tensor = build_tensor(&TensorFixture {
        dims: &[1],
        data_type: 1,
        name: "scale",
        doc_string: "",
        raw_data: &initializer_tensor_raw_data,
    });

    let node0 = build_node(&NodeFixture {
        input: &["x", "W"],
        output: &["z"],
        name: "node0",
        op_type: "MatMul",
        doc_string: "matmul node",
        attributes: &[build_attribute_float("alpha", 1.5)],
    });

    let node1 = build_node(&NodeFixture {
        input: &["z"],
        output: &["y"],
        name: "node1",
        op_type: "Relu",
        doc_string: "",
        attributes: &[
            build_attribute_int("axis", -1),
            build_attribute_string("mode", b"clip"),
            build_attribute_tensor("const_tensor", &const_tensor),
            build_attribute_ints("perm", &[0, 2, 1, 3]),
        ],
    });

    let graph = build_graph(
        &[node0, node1],
        "main_graph",
        &[weight_tensor],
        "graph doc",
        &[x_input],
        &[y_output],
    );

    let mut bytes = Vec::new();
    push_varint(1, 8, &mut bytes); // ir_version
    push_len(8, &build_operator_set_id("", 17), &mut bytes); // opset_import
    push_str(2, "proxima-test", &mut bytes); // producer_name
    push_str(3, "0.1.0", &mut bytes); // producer_version
    push_str(4, "ai.proxima.test", &mut bytes); // domain
    push_varint(5, 1, &mut bytes); // model_version
    push_str(6, "synthetic test model", &mut bytes); // doc_string
    // an unrecognized field (100) folded in right before `graph` -- proves
    // unknown field numbers are skipped, not rejected (protobuf
    // forward-compatibility).
    push_varint(100, 424_242, &mut bytes);
    push_len(7, &graph, &mut bytes); // graph

    Fixture { bytes, weight_raw_data, initializer_tensor_raw_data }
}

fn assert_fixture_parsed(model: &crate::messages::ModelProto<'_>, fixture: &Fixture) {
    assert_eq!(model.ir_version, 8);
    assert_eq!(model.opset_import.len(), 1);
    assert_eq!(model.opset_import[0].domain, "");
    assert_eq!(model.opset_import[0].version, 17);
    assert_eq!(model.producer_name, "proxima-test");
    assert_eq!(model.producer_version, "0.1.0");
    assert_eq!(model.domain, "ai.proxima.test");
    assert_eq!(model.model_version, 1);
    assert_eq!(model.doc_string, "synthetic test model");

    let graph = model.graph.as_ref().expect("graph field present");
    assert_eq!(graph.name, "main_graph");
    assert_eq!(graph.doc_string, "graph doc");

    assert_eq!(graph.input.len(), 1);
    assert_eq!(graph.input[0].name, "x");
    assert_eq!(graph.input[0].doc_string, "input x");
    let x_tensor_type = match &graph.input[0].r#type.as_ref().unwrap().value {
        Some(TypeValue::Tensor(tensor)) => tensor,
        other => panic!("expected tensor_type, got {other:?}"),
    };
    assert_eq!(x_tensor_type.elem_type, 1);
    let x_dims = &x_tensor_type.shape.as_ref().unwrap().dim;
    assert_eq!(x_dims.len(), 2);
    assert_eq!(x_dims[0].value, Some(DimensionValue::Value(1)));
    assert_eq!(x_dims[1].value, Some(DimensionValue::Param("batch")));

    assert_eq!(graph.output.len(), 1);
    assert_eq!(graph.output[0].name, "y");
    assert_eq!(graph.output[0].doc_string, "output y");

    assert_eq!(graph.initializer.len(), 1);
    let weight = &graph.initializer[0];
    assert_eq!(weight.name, "W");
    assert_eq!(weight.doc_string, "weight");
    assert_eq!(weight.data_type, 1);
    assert_eq!(weight.dims, alloc::vec![2, 2]);
    let raw_data = weight.raw_data.expect("raw_data present");
    assert_eq!(raw_data, fixture.weight_raw_data.as_slice());
    // zero-copy: the returned slice must be a sub-range of the caller's own
    // input buffer, never a fresh allocation.
    let buffer_range = fixture.bytes.as_ptr_range();
    let raw_data_range = raw_data.as_ptr_range();
    assert!(buffer_range.start <= raw_data_range.start && raw_data_range.end <= buffer_range.end);

    assert_eq!(graph.node.len(), 2);

    let node0 = &graph.node[0];
    assert_eq!(node0.name, "node0");
    assert_eq!(node0.op_type, "MatMul");
    assert_eq!(node0.input, alloc::vec!["x", "W"]);
    assert_eq!(node0.output, alloc::vec!["z"]);
    assert_eq!(node0.doc_string, "matmul node");
    assert_eq!(node0.attribute.len(), 1);
    assert_eq!(node0.attribute[0].name, "alpha");
    assert_eq!(node0.attribute[0].type_raw, 1);
    assert!((node0.attribute[0].f - 1.5).abs() < f32::EPSILON);

    let node1 = &graph.node[1];
    assert_eq!(node1.name, "node1");
    assert_eq!(node1.op_type, "Relu");
    assert_eq!(node1.attribute.len(), 4);

    let axis = &node1.attribute[0];
    assert_eq!(axis.name, "axis");
    assert_eq!(axis.type_raw, 2);
    assert_eq!(axis.i, -1);

    let mode = &node1.attribute[1];
    assert_eq!(mode.name, "mode");
    assert_eq!(mode.type_raw, 3);
    assert_eq!(mode.s, b"clip");

    let const_tensor_attr = &node1.attribute[2];
    assert_eq!(const_tensor_attr.name, "const_tensor");
    assert_eq!(const_tensor_attr.type_raw, 4);
    let inner_tensor = const_tensor_attr.t.as_ref().expect("nested tensor present");
    assert_eq!(inner_tensor.name, "scale");
    assert_eq!(inner_tensor.dims, alloc::vec![1]);
    assert_eq!(
        inner_tensor.raw_data.expect("nested raw_data present"),
        fixture.initializer_tensor_raw_data.as_slice()
    );

    let perm = &node1.attribute[3];
    assert_eq!(perm.name, "perm");
    assert_eq!(perm.type_raw, 7);
    assert_eq!(perm.ints, alloc::vec![0, 2, 1, 3]);
}

#[test]
fn synthetic_model_round_trips_every_field() {
    let fixture = build_fixture();
    let model = parse_complete(&fixture.bytes).expect("parse synthetic model");
    assert_fixture_parsed(&model, &fixture);
}

// -- chunk-boundary property test: the same bytes, fed to `OnnxParser` in
// many differently-sized pieces (including splits that land mid-varint and
// mid-length-header), must produce the identical event sequence as the
// same bytes decoded whole. Each event is compared and dropped immediately
// -- `OnnxParser::poll`'s borrow of `&mut self` means a caller cannot hold
// one event while requesting the next, which the test structure below
// respects rather than fighting.

fn expected_events(bytes: &[u8]) -> Vec<ModelField<'_>> {
    let mut events = Vec::new();
    for field in proxima_protocols::protobuf_wire::Fields::new(bytes) {
        let field = field.expect("well-formed synthetic bytes");
        events.push(decode_model_field(field).expect("decode top-level field"));
    }
    events
}

fn run_parser_and_compare(bytes: &[u8], chunk_bounds: &[usize]) {
    let expected = expected_events(bytes);
    let mut parser = OnnxParser::new();
    let mut start = 0usize;
    let mut expected_index = 0usize;
    let mut bounds = chunk_bounds.to_vec();
    bounds.push(bytes.len());

    for &end in &bounds {
        parser.feed(&bytes[start..end]);
        start = end;
        loop {
            match parser.poll().expect("poll well-formed synthetic bytes") {
                None => break,
                Some(event) => {
                    assert_eq!(
                        event, expected[expected_index],
                        "event {expected_index} mismatched at chunk boundary {end}"
                    );
                    expected_index += 1;
                }
            }
        }
    }
    parser.finish().expect("parser reached a clean field boundary");
    assert_eq!(expected_index, expected.len(), "not every top-level field was observed");
}

#[test]
fn chunk_boundary_matches_whole_buffer_fed_in_one_piece() {
    let fixture = build_fixture();
    run_parser_and_compare(&fixture.bytes, &[]);
}

#[test]
fn chunk_boundary_matches_whole_buffer_at_awkward_splits() {
    let fixture = build_fixture();
    let len = fixture.bytes.len();
    // 1/3/7/13-byte feeds, one byte at a time, and splits landing mid-varint
    // (the ir_version/model_version tag+value are single bytes each, so a
    // 1-byte-at-a-time feed schedule already forces many mid-varint and
    // mid-length-header splits across the rest of the message).
    let schedules: [&[usize]; 5] = [
        &[1, 3, 7, 13],
        &[2, 5, 11, 17, 23],
        &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        &[len / 2],
        &[len - 1],
    ];
    for schedule in schedules {
        run_parser_and_compare(&fixture.bytes, schedule);
    }
    // one byte at a time -- the maximally awkward schedule.
    let one_byte_at_a_time: Vec<usize> = (1..len).collect();
    run_parser_and_compare(&fixture.bytes, &one_byte_at_a_time);
}

// -- sad paths: never a panic, never an out-of-bounds read, always a typed
// error (or `NeedMore`, never a false positive).

#[test]
fn truncation_at_every_prefix_length_never_panics() {
    // A top-level protobuf message carries no outer length prefix, so a
    // truncation that happens to land exactly on a field boundary is
    // syntactically indistinguishable from "that's the whole message" --
    // `finish()` can only detect a cut that lands *inside* a field, never
    // a stream that stops early but on a clean boundary (the same
    // limitation any top-level protobuf reader has). What every prefix
    // length must guarantee, and what this sweeps for, is that driving the
    // parser to `None`-or-error and then calling `finish()` never panics
    // and never reads out of bounds.
    let fixture = build_fixture();
    for offset in 0..fixture.bytes.len() {
        let mut parser = OnnxParser::new();
        parser.feed(&fixture.bytes[..offset]);
        loop {
            match parser.poll() {
                Ok(None) => break,
                Ok(Some(_)) => {}
                Err(_) => break,
            }
        }
        let _ = parser.finish();
    }
    // the positive control: the full buffer must report a clean finish.
    let mut parser = OnnxParser::new();
    parser.feed(&fixture.bytes);
    while parser.poll().expect("full buffer parses").is_some() {}
    parser.finish().expect("full buffer reaches a clean field boundary");
}

#[test]
fn fsm_rejects_a_declared_length_past_the_sanity_cap() {
    // field 7 (graph), len-delimited, declares 2^41 bytes -- comfortably
    // past the FSM's sanity cap, so this must fail fast as a typed error
    // rather than waiting forever for bytes that will never arrive.
    let mut bytes = Vec::new();
    push_tag(7, 2, &mut bytes);
    encode_varint(1u64 << 41, &mut bytes);
    let mut parser = OnnxParser::new();
    parser.feed(&bytes);
    let outcome = parser.poll();
    assert!(matches!(outcome, Err(OnnxError::DeclaredLengthTooLarge { .. })));
}

#[test]
fn bad_wire_type_is_a_typed_error_not_a_panic() {
    // field 1, deprecated wire type 3 (group start): `Fields` rejects this
    // before a `Field` value can even be constructed, both through the raw
    // wire walker and through this crate's own whole-slice decode path.
    let bytes = [(1u8 << 3) | 3];
    let mut iter = proxima_protocols::protobuf_wire::Fields::new(&bytes);
    match iter.next() {
        Some(Err(_)) => {}
        other => panic!("expected a wire-format error for a deprecated wire type, got {other:?}"),
    }
    let outcome = parse_complete(&bytes);
    assert!(matches!(outcome, Err(OnnxError::Wire { .. })));
}

#[test]
fn declared_length_exceeding_buffer_is_a_typed_error() {
    // field 7 (graph), len-delimited, declares 1000 bytes but supplies none.
    let mut bytes = Vec::new();
    push_tag(7, 2, &mut bytes);
    encode_varint(1000, &mut bytes);
    let outcome = parse_complete(&bytes);
    assert!(matches!(outcome, Err(OnnxError::Wire { .. })));
}

#[test]
fn unknown_field_numbers_are_skipped_gracefully() {
    let fixture = build_fixture();
    // field 100 was folded into the synthetic model's top-level bytes
    // (see `build_fixture`); a successful, fully-correct parse is itself
    // the proof it was skipped rather than rejected or mis-consumed.
    let model = parse_complete(&fixture.bytes).expect("unknown field must not fail the parse");
    assert_fixture_parsed(&model, &fixture);
}

// -- real-world file: best effort, never fails the suite if absent.

#[cfg(feature = "std")]
#[test]
#[ignore = "depends on a real .onnx checkout outside this repo"]
fn parses_a_real_onnx_file_if_one_is_present() {
    let candidate = std::path::Path::new(
        "/Users/brianbruggeman/repos/others/burn/examples/onnx-inference/src/model/mnist.onnx",
    );
    if !candidate.exists() {
        eprintln!("no real .onnx found at {candidate:?}, skipping");
        return;
    }
    let bytes = std::fs::read(candidate).expect("read real onnx file");
    let model = parse_complete(&bytes).expect("parse real onnx file");
    let graph = model.graph.as_ref().expect("real model has a graph");
    println!("producer: {} {}", model.producer_name, model.producer_version);
    println!(
        "opset: {}",
        model
            .opset_import
            .first()
            .map(|opset| opset.version)
            .unwrap_or_default()
    );
    println!("node_count: {}", graph.node.len());
    for node in graph.node.iter().take(5) {
        println!("op: {}", node.op_type);
    }
}
