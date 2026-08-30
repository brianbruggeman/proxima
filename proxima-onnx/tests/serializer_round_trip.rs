//! The write-side codec exercised as a genuine multi-op model builder:
//! a `Gemm -> Relu -> Gemm -> Softmax` two-layer MLP is constructed
//! entirely through this crate's own typed message structs
//! ([`proxima_onnx::messages::{ModelProto, GraphProto, NodeProto, ...}`])
//! and serialized via [`proxima_onnx::writer::write_model_proto`] -- the
//! crate's own bidirectional codec, schema-complete like
//! `proxima-model-interop/tests/support/mod.rs`'s fixture builders, never
//! hand-pushed protobuf tag bytes (`proxima-onnx/src/tests.rs`'s
//! `push_tag`/`push_len` builders are the thing this test intentionally
//! does NOT use). The resulting real wire bytes are re-parsed, lowered,
//! and evaluated, and the output is checked against the same
//! hand-computed `softmax(relu(x @ W1 + b1) @ W2 + b2)` reference
//! `proxima-onnx/src/tests.rs::onnx_bytes_lower_to_op_and_evaluate_a_two_layer_mlp`
//! uses, proving the write path alone (not `crate::lift`) produces bytes
//! this crate's own read path accepts and lowers correctly.

#![allow(clippy::unwrap_used, clippy::expect_used)]

extern crate alloc;

use proxima_onnx::messages::{
    AttributeProto, Dimension, DimensionValue, GraphProto, ModelProto, NodeProto, OperatorSetIdProto, TensorProto, TensorShapeProto, TypeProto,
    TypeProtoTensor, TypeValue, ValueInfoProto,
};
use proxima_onnx::writer::write_model_proto;

fn f32_bytes(values: &[f32]) -> alloc::vec::Vec<u8> {
    values.iter().flat_map(|value| value.to_le_bytes()).collect()
}

fn tensor<'a>(name: &'a str, dims: &[i64], raw_data: &'a [u8]) -> TensorProto<'a> {
    TensorProto { dims: dims.to_vec(), data_type: 1, name, raw_data: Some(raw_data), ..TensorProto::default() }
}

fn tensor_input(name: &'static str, dims: &[i64]) -> ValueInfoProto<'static> {
    let shape = TensorShapeProto { dim: dims.iter().map(|&value| Dimension { value: Some(DimensionValue::Value(value)), denotation: "" }).collect() };
    ValueInfoProto { name, r#type: Some(TypeProto { value: Some(TypeValue::Tensor(TypeProtoTensor { elem_type: 1, shape: Some(shape) })), denotation: "" }), doc_string: "" }
}

/// Builds the model through [`proxima_onnx::messages`] structs and
/// [`write_model_proto`], serializes it to real wire bytes, re-parses
/// those bytes, lowers, and evaluates -- asserting against the same
/// hand-computed reference the synthetic-bytes MLP test uses, so a
/// divergence here means the serializer (not the hand-computed math)
/// is wrong.
#[test]
fn serializer_builds_a_two_layer_mlp_that_reparses_lowers_and_evaluates_correctly() {
    let x_data: [f32; 6] = [1.0, 0.5, -1.0, 0.0, 2.0, 1.0];
    let w1_data: [f32; 12] = [0.1, 0.2, -0.1, 0.05, 0.3, -0.2, 0.4, 0.1, -0.5, 0.1, 0.2, -0.3];
    let b1_data: [f32; 4] = [0.1, -0.1, 0.05, 0.0];
    let w2_data: [f32; 8] = [0.2, -0.3, 0.1, 0.4, -0.2, 0.05, 0.3, 0.1];
    let b2_data: [f32; 2] = [0.0, 0.1];

    let w1_bytes = f32_bytes(&w1_data);
    let b1_bytes = f32_bytes(&b1_data);
    let w2_bytes = f32_bytes(&w2_data);
    let b2_bytes = f32_bytes(&b2_data);

    let w1 = tensor("W1", &[3, 4], &w1_bytes);
    let b1 = tensor("b1", &[4], &b1_bytes);
    let w2 = tensor("W2", &[4, 2], &w2_bytes);
    let b2 = tensor("b2", &[2], &b2_bytes);

    let gemm1 = NodeProto { input: vec!["x", "W1", "b1"], output: vec!["h"], name: "gemm1", op_type: "Gemm", ..NodeProto::default() };
    let relu = NodeProto { input: vec!["h"], output: vec!["hr"], name: "relu", op_type: "Relu", ..NodeProto::default() };
    let gemm2 = NodeProto { input: vec!["hr", "W2", "b2"], output: vec!["logits"], name: "gemm2", op_type: "Gemm", ..NodeProto::default() };
    let softmax_axis = AttributeProto { name: "axis", i: 1, type_raw: 2, ..AttributeProto::default() };
    let softmax = NodeProto { input: vec!["logits"], output: vec!["y"], name: "softmax", op_type: "Softmax", attribute: vec![softmax_axis], ..NodeProto::default() };

    let graph = GraphProto {
        node: vec![gemm1, relu, gemm2, softmax],
        name: "serializer_mlp",
        initializer: vec![w1, b1, w2, b2],
        input: vec![tensor_input("x", &[2, 3])],
        output: vec![ValueInfoProto { name: "y", ..ValueInfoProto::default() }],
        ..GraphProto::default()
    };
    let model = ModelProto {
        ir_version: 8,
        opset_import: vec![OperatorSetIdProto { domain: "", version: 18 }],
        graph: Some(graph),
        ..ModelProto::default()
    };

    let bytes = write_model_proto(&model);
    assert!(!bytes.is_empty(), "the serializer produces nonempty wire bytes");

    let reparsed_model = proxima_onnx::pipe::parse_complete(&bytes).expect("re-parse the serializer's own wire bytes");
    let reparsed_graph = reparsed_model.graph.as_ref().expect("re-parsed model carries a graph");
    assert_eq!(reparsed_graph.node.len(), 4, "all four nodes round-trip through the serializer");

    let lowered = proxima_onnx::lower::lower_graph(reparsed_graph).expect("lower the serializer-built graph");
    let mut named: Vec<(&str, &[f32])> = lowered.initializers.iter().map(|(name, data)| (name.as_str(), data.as_slice())).collect();
    named.push(("x", &x_data));

    let output_node = lowered.graph_outputs.iter().find(|(name, _)| name.as_str() == "y").expect("y is a declared graph output").1;
    let evaluated = proxima_tensor::cpu::evaluate_named(&lowered.program, &[], &named, &[output_node]).expect("evaluate the serializer-built program");
    let (data, shape) = evaluated.get(output_node).expect("y result present");

    assert_eq!(shape, &[2, 2]);
    // same hand-computed reference as tests.rs's synthetic-bytes MLP test.
    let expected = [0.599_888_4_f32, 0.400_111_6, 0.434_749_25, 0.565_250_74];
    for (actual, expected) in data.iter().zip(expected.iter()) {
        assert!((actual - expected).abs() < 1e-4, "softmax output {actual} does not match hand-computed reference {expected}");
    }
}
