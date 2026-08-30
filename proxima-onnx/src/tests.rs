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

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|value| value.to_le_bytes()).collect()
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

// -- ONNX bytes -> Op lowering -> evaluation, end to end.

/// `x -> Gemm(W1, b1) -> Relu -> Gemm(W2, b2) -> Softmax -> y`, a real
/// 2-layer MLP: encoded as genuine wire-format ONNX bytes (the same
/// builders [`build_fixture`] uses), parsed, lowered via
/// [`crate::lower::lower_graph`], and evaluated via
/// [`proxima_tensor::evaluate_named`]. Expected softmax outputs are a hand
/// computation (`bc -l`, independent of this crate's own math) of
/// `softmax(relu(x @ W1 + b1) @ W2 + b2)` -- see the row-by-row arithmetic
/// in this test's own comments, not a re-derivation through the code under
/// test.
#[test]
fn onnx_bytes_lower_to_op_and_evaluate_a_two_layer_mlp() {
    // x: [2, 3] row-major -- [[1.0, 0.5, -1.0], [0.0, 2.0, 1.0]].
    let x_data: [f32; 6] = [1.0, 0.5, -1.0, 0.0, 2.0, 1.0];
    // W1: [3, 4].
    let w1_data: [f32; 12] = [0.1, 0.2, -0.1, 0.05, 0.3, -0.2, 0.4, 0.1, -0.5, 0.1, 0.2, -0.3];
    let b1_data: [f32; 4] = [0.1, -0.1, 0.05, 0.0];
    // W2: [4, 2].
    let w2_data: [f32; 8] = [0.2, -0.3, 0.1, 0.4, -0.2, 0.05, 0.3, 0.1];
    let b2_data: [f32; 2] = [0.0, 0.1];

    let x_shape = build_tensor_shape(&[build_dimension_value(2), build_dimension_value(3)]);
    let x_type = build_type_proto(&build_type_proto_tensor(1, &x_shape));
    let x_input = build_value_info("x", &x_type, "");
    let y_output = build_value_info("y", &[], "");

    let w1_tensor = build_tensor(&TensorFixture { dims: &[3, 4], data_type: 1, name: "W1", doc_string: "", raw_data: &f32_bytes(&w1_data) });
    let b1_tensor = build_tensor(&TensorFixture { dims: &[4], data_type: 1, name: "b1", doc_string: "", raw_data: &f32_bytes(&b1_data) });
    let w2_tensor = build_tensor(&TensorFixture { dims: &[4, 2], data_type: 1, name: "W2", doc_string: "", raw_data: &f32_bytes(&w2_data) });
    let b2_tensor = build_tensor(&TensorFixture { dims: &[2], data_type: 1, name: "b2", doc_string: "", raw_data: &f32_bytes(&b2_data) });

    let gemm1 = build_node(&NodeFixture { input: &["x", "W1", "b1"], output: &["h"], name: "gemm1", op_type: "Gemm", doc_string: "", attributes: &[] });
    let relu = build_node(&NodeFixture { input: &["h"], output: &["hr"], name: "relu", op_type: "Relu", doc_string: "", attributes: &[] });
    let gemm2 = build_node(&NodeFixture { input: &["hr", "W2", "b2"], output: &["logits"], name: "gemm2", op_type: "Gemm", doc_string: "", attributes: &[] });
    let softmax = build_node(&NodeFixture {
        input: &["logits"],
        output: &["y"],
        name: "softmax",
        op_type: "Softmax",
        doc_string: "",
        attributes: &[build_attribute_int("axis", 1)],
    });

    let graph = build_graph(&[gemm1, relu, gemm2, softmax], "mlp", &[w1_tensor, b1_tensor, w2_tensor, b2_tensor], "", &[x_input], &[y_output]);

    let mut bytes = Vec::new();
    push_varint(1, 8, &mut bytes); // ir_version
    push_len(7, &graph, &mut bytes); // graph

    let model = parse_complete(&bytes).expect("parse the mlp model bytes");
    let onnx_graph = model.graph.as_ref().expect("mlp graph present");

    let lowered = crate::lower::lower_graph(onnx_graph).expect("lower the mlp graph to Op");
    assert_eq!(lowered.graph_inputs, alloc::vec!["x".to_string()]);

    let mut named: Vec<(&str, &[f32])> = lowered.initializers.iter().map(|(name, data)| (name.as_str(), data.as_slice())).collect();
    named.push(("x", &x_data));

    let output_node = lowered
        .graph_outputs
        .iter()
        .find(|(name, _)| name.as_str() == "y")
        .expect("y is a declared graph output")
        .1;
    let evaluated =
        proxima_tensor::cpu::evaluate_named(&lowered.program, &[], &named, &[output_node]).expect("evaluate the lowered mlp program");
    let (data, shape) = evaluated.get(output_node).expect("y result present");

    assert_eq!(shape, &[2, 2]);
    // hand-computed via `bc -l`: softmax(relu(x @ W1 + b1) @ W2 + b2).
    let expected = [0.599_888_4_f32, 0.400_111_6, 0.434_749_25, 0.565_250_74];
    for (actual, expected) in data.iter().zip(expected.iter()) {
        assert!((actual - expected).abs() < 1e-4, "softmax output {actual} does not match hand-computed reference {expected}");
    }

    let row0_sum = data[0] + data[1];
    let row1_sum = data[2] + data[3];
    assert!((row0_sum - 1.0).abs() < 1e-5, "softmax row 0 sums to {row0_sum}, not 1.0");
    assert!((row1_sum - 1.0).abs() < 1e-5, "softmax row 1 sums to {row1_sum}, not 1.0");
}

// -- the writable round trip: Op program -> ONNX wire bytes -> Op program.

/// The full write-side loop this crate's ONNX support now closes:
/// `onnx bytes -> lower -> Op` (the read half, already proven above by
/// [`onnx_bytes_lower_to_op_and_evaluate_a_two_layer_mlp`]) followed by
/// `Op -> lift -> onnx bytes -> lower -> Op` (the write half). Both `Op`
/// programs are evaluated via [`proxima_tensor::cpu::evaluate_named`] and
/// must agree to `1e-4`, proving [`crate::lift::lift_model`] plus
/// [`crate::lower::lower_graph`] round-trip the same 2-layer MLP
/// (`Gemm -> Relu -> Gemm -> Softmax`, already decomposed by the first
/// lowering into the primitive `Op` vocabulary [`crate::lift`]'s own doc
/// says it lifts faithfully -- `Mul`/`Add`/`Max`/`Sub`/`Exp`/`Div`
/// elementwise nodes and `ReduceSum`/`ReduceMax` reduce nodes, never
/// `Gemm`/`Softmax` themselves) without drifting numerically.
#[test]
fn op_program_lifts_to_onnx_bytes_and_lowers_back_to_an_equivalent_program() {
    let x_data: [f32; 6] = [1.0, 0.5, -1.0, 0.0, 2.0, 1.0];
    let w1_data: [f32; 12] = [0.1, 0.2, -0.1, 0.05, 0.3, -0.2, 0.4, 0.1, -0.5, 0.1, 0.2, -0.3];
    let b1_data: [f32; 4] = [0.1, -0.1, 0.05, 0.0];
    let w2_data: [f32; 8] = [0.2, -0.3, 0.1, 0.4, -0.2, 0.05, 0.3, 0.1];
    let b2_data: [f32; 2] = [0.0, 0.1];

    let x_shape = build_tensor_shape(&[build_dimension_value(2), build_dimension_value(3)]);
    let x_type = build_type_proto(&build_type_proto_tensor(1, &x_shape));
    let x_input = build_value_info("x", &x_type, "");
    let y_output = build_value_info("y", &[], "");

    let w1_tensor = build_tensor(&TensorFixture { dims: &[3, 4], data_type: 1, name: "W1", doc_string: "", raw_data: &f32_bytes(&w1_data) });
    let b1_tensor = build_tensor(&TensorFixture { dims: &[4], data_type: 1, name: "b1", doc_string: "", raw_data: &f32_bytes(&b1_data) });
    let w2_tensor = build_tensor(&TensorFixture { dims: &[4, 2], data_type: 1, name: "W2", doc_string: "", raw_data: &f32_bytes(&w2_data) });
    let b2_tensor = build_tensor(&TensorFixture { dims: &[2], data_type: 1, name: "b2", doc_string: "", raw_data: &f32_bytes(&b2_data) });

    let gemm1 = build_node(&NodeFixture { input: &["x", "W1", "b1"], output: &["h"], name: "gemm1", op_type: "Gemm", doc_string: "", attributes: &[] });
    let relu = build_node(&NodeFixture { input: &["h"], output: &["hr"], name: "relu", op_type: "Relu", doc_string: "", attributes: &[] });
    let gemm2 = build_node(&NodeFixture { input: &["hr", "W2", "b2"], output: &["logits"], name: "gemm2", op_type: "Gemm", doc_string: "", attributes: &[] });
    let softmax = build_node(&NodeFixture {
        input: &["logits"],
        output: &["y"],
        name: "softmax",
        op_type: "Softmax",
        doc_string: "",
        attributes: &[build_attribute_int("axis", 1)],
    });

    let graph = build_graph(&[gemm1, relu, gemm2, softmax], "mlp", &[w1_tensor, b1_tensor, w2_tensor, b2_tensor], "", &[x_input], &[y_output]);
    let mut bytes = Vec::new();
    push_varint(1, 8, &mut bytes);
    push_len(7, &graph, &mut bytes);

    // -- read half: onnx bytes -> Op, and a baseline evaluation.
    let original_model = parse_complete(&bytes).expect("parse the mlp model bytes");
    let original_graph = original_model.graph.as_ref().expect("mlp graph present");
    let original = crate::lower::lower_graph(original_graph).expect("lower the original mlp graph");
    let mut original_named: Vec<(&str, &[f32])> = original.initializers.iter().map(|(name, data)| (name.as_str(), data.as_slice())).collect();
    original_named.push(("x", &x_data));
    let original_output = original.graph_outputs.iter().find(|(name, _)| name.as_str() == "y").expect("y is declared").1;
    let baseline = proxima_tensor::cpu::evaluate_named(&original.program, &[], &original_named, &[original_output]).expect("evaluate the original program");
    let (baseline_data, baseline_shape) = baseline.get(original_output).expect("baseline y present");

    // -- write half: Op -> lift -> onnx bytes.
    let lift_input = crate::lift::LiftInput {
        program: &original.program,
        initializers: &original.initializers,
        graph_inputs: &original.graph_inputs,
        graph_outputs: &original.graph_outputs,
        graph_name: "mlp_lifted",
    };
    let lifted_bytes = crate::lift::lift_model(lift_input).expect("lift the lowered mlp program to onnx bytes");

    // -- re-parse: the lifted bytes are a structurally valid ModelProto.
    let reparsed_model = parse_complete(&lifted_bytes).expect("lifted bytes parse back to a ModelProto");
    let reparsed_graph = reparsed_model.graph.as_ref().expect("lifted graph present");
    assert!(!reparsed_graph.node.is_empty(), "lifted graph carries its primitive-op nodes");

    // -- read half again: onnx bytes -> Op, over the LIFTED graph this time.
    let reloaded = crate::lower::lower_graph(reparsed_graph).expect("lower the lifted mlp graph back to Op");
    let mut reloaded_named: Vec<(&str, &[f32])> = reloaded.initializers.iter().map(|(name, data)| (name.as_str(), data.as_slice())).collect();
    reloaded_named.push(("x", &x_data));
    let reloaded_output = reloaded.graph_outputs.iter().find(|(name, _)| name.as_str() == "y").expect("y is declared on the lifted graph").1;
    let evaluated = proxima_tensor::cpu::evaluate_named(&reloaded.program, &[], &reloaded_named, &[reloaded_output]).expect("evaluate the round-tripped program");
    let (data, shape) = evaluated.get(reloaded_output).expect("round-tripped y present");

    assert_eq!(shape, baseline_shape, "round trip preserves y's shape");
    for (actual, expected) in data.iter().zip(baseline_data.iter()) {
        assert!((actual - expected).abs() < 1e-4, "round-tripped output {actual} does not match original-program baseline {expected}");
    }
}

/// `Gemm(transA=1)`: `lower_gemm` reads `A`'s real `[K, M]` shape through a
/// *permuted* pattern (`lift::try_matmul_shape`'s own doc: `lhs_axis0 == k`
/// names `trans_a`). Round-trips through [`crate::lift::lift_model`] as a
/// NAMED `Gemm` node carrying `transA=1` -- never `MatMul` (which has no
/// transpose attribute in ONNX) and never a primitive `Mul`/`ReduceSum`
/// spray -- then lowers back and evaluates identically to the original.
#[test]
fn gemm_transposed_a_round_trips_through_lift_as_a_named_gemm() {
    let a_data: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data: [f32; 12] = [0.1, 0.2, -0.1, 0.05, 0.3, -0.2, 0.4, 0.1, -0.5, 0.1, 0.2, -0.3];

    let a_shape = build_tensor_shape(&[build_dimension_value(3), build_dimension_value(2)]);
    let a_type = build_type_proto(&build_type_proto_tensor(1, &a_shape));
    let a_input = build_value_info("a", &a_type, "");
    let y_output = build_value_info("y", &[], "");

    let b_tensor = build_tensor(&TensorFixture { dims: &[3, 4], data_type: 1, name: "b", doc_string: "", raw_data: &f32_bytes(&b_data) });
    let gemm = build_node(&NodeFixture { input: &["a", "b"], output: &["y"], name: "gemm", op_type: "Gemm", doc_string: "", attributes: &[build_attribute_int("transA", 1)] });

    let graph = build_graph(&[gemm], "transposed_gemm", &[b_tensor], "", &[a_input], &[y_output]);
    let mut bytes = Vec::new();
    push_varint(1, 8, &mut bytes);
    push_len(7, &graph, &mut bytes);

    let original_model = parse_complete(&bytes).expect("parse the transposed-Gemm model bytes");
    let original_graph = original_model.graph.as_ref().expect("transposed-Gemm graph present");
    let original = crate::lower::lower_graph(original_graph).expect("lower the original transposed-Gemm graph");
    let mut original_named: Vec<(&str, &[f32])> = original.initializers.iter().map(|(name, data)| (name.as_str(), data.as_slice())).collect();
    original_named.push(("a", &a_data));
    let original_output = original.graph_outputs.iter().find(|(name, _)| name.as_str() == "y").expect("y is declared").1;
    let baseline = proxima_tensor::cpu::evaluate_named(&original.program, &[], &original_named, &[original_output]).expect("evaluate the original transposed-Gemm program");
    let (baseline_data, baseline_shape) = baseline.get(original_output).expect("baseline y present");

    let lift_input = crate::lift::LiftInput {
        program: &original.program,
        initializers: &original.initializers,
        graph_inputs: &original.graph_inputs,
        graph_outputs: &original.graph_outputs,
        graph_name: "transposed_gemm_lifted",
    };
    let lifted_bytes = crate::lift::lift_model(lift_input).expect("lift the lowered transposed-Gemm program to onnx bytes");

    let reparsed_model = parse_complete(&lifted_bytes).expect("lifted transposed-Gemm bytes parse back to a ModelProto");
    let reparsed_graph = reparsed_model.graph.as_ref().expect("lifted transposed-Gemm graph present");
    // `Gemm` always applies its `alpha` scale (even at the 1.0 default,
    // this crate's own `lower_gemm` doc), so the contraction itself is an
    // intermediate node -- the graph's declared output `"y"` names that
    // scale's `Mul`, not the `Gemm` node underneath it.
    let gemm_node = reparsed_graph.node.iter().find(|node| node.op_type == "Gemm").expect("the lifted graph carries a NAMED Gemm node, not a primitive Mul/ReduceSum spray");
    let trans_a_attribute = gemm_node.attribute.iter().find(|attribute| attribute.name == "transA").expect("the lifted Gemm node carries transA");
    assert_eq!(trans_a_attribute.i, 1, "the lifted Gemm node's transA matches the original graph's");
    assert!(!reparsed_graph.node.iter().any(|node| node.op_type == "ReduceSum"), "a raised Gemm never leaves behind a primitive ReduceSum");

    let reloaded = crate::lower::lower_graph(reparsed_graph).expect("lower the lifted transposed-Gemm graph back to Op");
    let mut reloaded_named: Vec<(&str, &[f32])> = reloaded.initializers.iter().map(|(name, data)| (name.as_str(), data.as_slice())).collect();
    reloaded_named.push(("a", &a_data));
    let reloaded_output = reloaded.graph_outputs.iter().find(|(name, _)| name.as_str() == "y").expect("y is declared on the lifted graph").1;
    let evaluated = proxima_tensor::cpu::evaluate_named(&reloaded.program, &[], &reloaded_named, &[reloaded_output]).expect("evaluate the round-tripped transposed-Gemm program");
    let (data, shape) = evaluated.get(reloaded_output).expect("round-tripped y present");

    assert_eq!(shape, baseline_shape, "round trip preserves y's shape");
    for (actual, expected) in data.iter().zip(baseline_data.iter()) {
        assert!((actual - expected).abs() < 1e-4, "round-tripped output {actual} does not match original-program baseline {expected}");
    }
}

/// `Conv` (unpadded, stride 1, single channel/group): `lower::conv2d_core`'s
/// two-level materialized-window shape round-trips through
/// [`crate::lift::lift_model`] as a NAMED `Conv` node -- never a primitive
/// `Mul`/`ReduceSum` spray, and never the intermediate `Elementwise` levels
/// [`crate::lift::try_conv_shape`]'s own doc says it folds -- then lowers
/// back and evaluates identically to the original.
#[test]
fn conv_round_trips_through_lift_as_a_named_conv() {
    let x_data: [f32; 9] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let w_data: [f32; 4] = [1.0, 0.0, 0.0, -1.0];

    let x_shape = build_tensor_shape(&[build_dimension_value(1), build_dimension_value(1), build_dimension_value(3), build_dimension_value(3)]);
    let x_type = build_type_proto(&build_type_proto_tensor(1, &x_shape));
    let x_input = build_value_info("x", &x_type, "");
    let y_output = build_value_info("y", &[], "");

    let w_tensor = build_tensor(&TensorFixture { dims: &[1, 1, 2, 2], data_type: 1, name: "w", doc_string: "", raw_data: &f32_bytes(&w_data) });
    let conv = build_node(&NodeFixture {
        input: &["x", "w"],
        output: &["y"],
        name: "conv",
        op_type: "Conv",
        doc_string: "",
        attributes: &[build_attribute_ints("kernel_shape", &[2, 2])],
    });

    let graph = build_graph(&[conv], "conv_graph", &[w_tensor], "", &[x_input], &[y_output]);
    let mut bytes = Vec::new();
    push_varint(1, 8, &mut bytes);
    push_len(7, &graph, &mut bytes);

    let original_model = parse_complete(&bytes).expect("parse the Conv model bytes");
    let original_graph = original_model.graph.as_ref().expect("Conv graph present");
    let original = crate::lower::lower_graph(original_graph).expect("lower the original Conv graph");
    let mut original_named: Vec<(&str, &[f32])> = original.initializers.iter().map(|(name, data)| (name.as_str(), data.as_slice())).collect();
    original_named.push(("x", &x_data));
    let original_output = original.graph_outputs.iter().find(|(name, _)| name.as_str() == "y").expect("y is declared").1;
    let baseline = proxima_tensor::cpu::evaluate_named(&original.program, &[], &original_named, &[original_output]).expect("evaluate the original Conv program");
    let (baseline_data, baseline_shape) = baseline.get(original_output).expect("baseline y present");

    let lift_input = crate::lift::LiftInput {
        program: &original.program,
        initializers: &original.initializers,
        graph_inputs: &original.graph_inputs,
        graph_outputs: &original.graph_outputs,
        graph_name: "conv_lifted",
    };
    let lifted_bytes = crate::lift::lift_model(lift_input).expect("lift the lowered Conv program to onnx bytes");

    let reparsed_model = parse_complete(&lifted_bytes).expect("lifted Conv bytes parse back to a ModelProto");
    let reparsed_graph = reparsed_model.graph.as_ref().expect("lifted Conv graph present");
    let conv_node = reparsed_graph.node.iter().find(|node| node.op_type == "Conv").expect("the lifted graph carries a NAMED Conv node, not a primitive Mul/ReduceSum spray");
    let kernel_shape = conv_node.attribute.iter().find(|attribute| attribute.name == "kernel_shape").expect("the lifted Conv node carries kernel_shape");
    assert_eq!(kernel_shape.ints, alloc::vec![2, 2], "recovered kernel_shape matches the original 2x2 kernel");
    assert!(!reparsed_graph.node.iter().any(|node| node.op_type == "ReduceSum"), "a raised Conv never leaves behind a primitive ReduceSum");
    assert!(!reparsed_graph.node.iter().any(|node| node.op_type == "Gather"), "a raised Conv never leaves behind the primitive window-axis Gather");

    let reloaded = crate::lower::lower_graph(reparsed_graph).expect("lower the lifted Conv graph back to Op");
    let mut reloaded_named: Vec<(&str, &[f32])> = reloaded.initializers.iter().map(|(name, data)| (name.as_str(), data.as_slice())).collect();
    reloaded_named.push(("x", &x_data));
    let reloaded_output = reloaded.graph_outputs.iter().find(|(name, _)| name.as_str() == "y").expect("y is declared on the lifted graph").1;
    let evaluated = proxima_tensor::cpu::evaluate_named(&reloaded.program, &[], &reloaded_named, &[reloaded_output]).expect("evaluate the round-tripped Conv program");
    let (data, shape) = evaluated.get(reloaded_output).expect("round-tripped y present");

    assert_eq!(shape, baseline_shape, "round trip preserves y's shape");
    for (actual, expected) in data.iter().zip(baseline_data.iter()) {
        assert!((actual - expected).abs() < 1e-4, "round-tripped output {actual} does not match original-program baseline {expected}");
    }
}
