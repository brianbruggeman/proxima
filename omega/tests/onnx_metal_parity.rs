//! Proves an ONNX-lowered program runs on `Backend::Metal` at parity with
//! `Backend::Cpu`, through the SAME `omega::backend` wrapper
//! `backend_parity.rs` already proves is backend-agnostic — this file only
//! changes where the `Vec<Op>` program comes from: `proxima_onnx::lower`
//! instead of `proxima_tensor::spec`.
//!
//! Fixture mirrors `proxima-onnx/src/tests.rs`'s
//! `onnx_bytes_lower_to_op_and_evaluate_a_two_layer_mlp`: a real 2-layer MLP
//! (`Gemm -> Relu -> Gemm -> Softmax`) encoded as genuine wire-format ONNX
//! bytes, parsed, lowered, then run on both backends. `Gemm` lowers to
//! `matmul` + broadcast `Add` and `Relu`/`Softmax` lower to elementwise/
//! reduce compositions already covered individually by
//! `metal_parity.rs`'s `softmax_parity_matches_within_epsilon` and
//! `embedding_matmul_parity_matches_within_epsilon` — this test is the first
//! place an onnx-lowered program is the thing that reaches
//! `omega::backend::execute_plan_named(.., Backend::Metal)`.

#![cfg(all(feature = "cpu", feature = "metal", target_os = "macos"))]
// every expect below runs against onnx bytes this test hand-encodes or a
// real device call; a failure there IS the test failing, not a case to
// recover.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use omega::backend::{Backend, execute_plan_named, plan_named};
use proxima_onnx::lower::lower_graph;
use proxima_onnx::pipe::parse_complete;
use proxima_tensor::QuantizedBlock;

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|value| value.to_le_bytes()).collect()
}

/// LEB128 varint, same wire encoding `proxima_protocols::protobuf_wire`'s
/// own `encode_varint` produces — hand-rolled here rather than pulling in
/// `proxima-protocols` as a second dev-dependency solely for this one
/// primitive, mirroring `proxima-onnx/src/tests.rs`'s own byte builders in
/// shape (this file's tag/len/str helpers are the same composition, just
/// self-contained).
fn encode_varint(mut value: u64, buf: &mut Vec<u8>) {
    loop {
        let byte = (value & 0x7f) as u8;
        value >>= 7;
        if value == 0 {
            buf.push(byte);
            break;
        }
        buf.push(byte | 0x80);
    }
}

fn tag(field: u32, wire: u8, buf: &mut Vec<u8>) {
    encode_varint((u64::from(field) << 3) | u64::from(wire), buf);
}

fn push_str(field: u32, value: &str, buf: &mut Vec<u8>) {
    push_len(field, value.as_bytes(), buf);
}

fn push_len(field: u32, payload: &[u8], buf: &mut Vec<u8>) {
    tag(field, 2, buf);
    encode_varint(payload.len() as u64, buf);
    buf.extend_from_slice(payload);
}

fn push_varint(field: u32, value: u64, buf: &mut Vec<u8>) {
    tag(field, 0, buf);
    encode_varint(value, buf);
}

fn build_dimension_value(value: i64) -> Vec<u8> {
    let mut buf = Vec::new();
    push_varint(1, value as u64, &mut buf);
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

fn build_value_info(name: &str, type_proto: &[u8]) -> Vec<u8> {
    let mut buf = Vec::new();
    push_str(1, name, &mut buf);
    push_len(2, type_proto, &mut buf);
    buf
}

struct TensorFixture<'a> {
    dims: &'a [i64],
    name: &'a str,
    raw_data: &'a [u8],
}

fn build_tensor(fixture: &TensorFixture<'_>) -> Vec<u8> {
    let mut buf = Vec::new();
    let mut dims_payload = Vec::new();
    for dim in fixture.dims {
        encode_varint(*dim as u64, &mut dims_payload);
    }
    push_len(1, &dims_payload, &mut buf);
    push_varint(2, 1, &mut buf); // FLOAT
    push_str(8, fixture.name, &mut buf);
    push_len(9, fixture.raw_data, &mut buf);
    buf
}

fn build_attribute_int(name: &str, value: i64) -> Vec<u8> {
    let mut buf = Vec::new();
    push_str(1, name, &mut buf);
    push_varint(20, 2, &mut buf); // AttributeType::INT
    push_varint(3, value as u64, &mut buf);
    buf
}

struct NodeFixture<'a> {
    input: &'a [&'a str],
    output: &'a [&'a str],
    name: &'a str,
    op_type: &'a str,
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
    buf
}

fn build_graph(nodes: &[Vec<u8>], name: &str, initializers: &[Vec<u8>], inputs: &[Vec<u8>], outputs: &[Vec<u8>]) -> Vec<u8> {
    let mut buf = Vec::new();
    for node in nodes {
        push_len(1, node, &mut buf);
    }
    push_str(2, name, &mut buf);
    for initializer in initializers {
        push_len(5, initializer, &mut buf);
    }
    for input in inputs {
        push_len(11, input, &mut buf);
    }
    for output in outputs {
        push_len(12, output, &mut buf);
    }
    buf
}

/// Hand-encoded ONNX model bytes for `x -> Gemm(W1, b1) -> Relu ->
/// Gemm(W2, b2) -> Softmax -> y`, plus the `x` input data the test binds to
/// the lowered program.
fn two_layer_mlp_model_bytes() -> (Vec<u8>, [f32; 6]) {
    let x_data: [f32; 6] = [1.0, 0.5, -1.0, 0.0, 2.0, 1.0];
    let w1_data: [f32; 12] = [0.1, 0.2, -0.1, 0.05, 0.3, -0.2, 0.4, 0.1, -0.5, 0.1, 0.2, -0.3];
    let b1_data: [f32; 4] = [0.1, -0.1, 0.05, 0.0];
    let w2_data: [f32; 8] = [0.2, -0.3, 0.1, 0.4, -0.2, 0.05, 0.3, 0.1];
    let b2_data: [f32; 2] = [0.0, 0.1];

    let x_shape = build_tensor_shape(&[build_dimension_value(2), build_dimension_value(3)]);
    let x_type = build_type_proto(&build_type_proto_tensor(1, &x_shape));
    let x_input = build_value_info("x", &x_type);
    let y_output = build_value_info("y", &[]);

    let w1_tensor = build_tensor(&TensorFixture { dims: &[3, 4], name: "W1", raw_data: &f32_bytes(&w1_data) });
    let b1_tensor = build_tensor(&TensorFixture { dims: &[4], name: "b1", raw_data: &f32_bytes(&b1_data) });
    let w2_tensor = build_tensor(&TensorFixture { dims: &[4, 2], name: "W2", raw_data: &f32_bytes(&w2_data) });
    let b2_tensor = build_tensor(&TensorFixture { dims: &[2], name: "b2", raw_data: &f32_bytes(&b2_data) });

    let gemm1 = build_node(&NodeFixture { input: &["x", "W1", "b1"], output: &["h"], name: "gemm1", op_type: "Gemm", attributes: &[] });
    let relu = build_node(&NodeFixture { input: &["h"], output: &["hr"], name: "relu", op_type: "Relu", attributes: &[] });
    let gemm2 = build_node(&NodeFixture { input: &["hr", "W2", "b2"], output: &["logits"], name: "gemm2", op_type: "Gemm", attributes: &[] });
    let softmax = build_node(&NodeFixture {
        input: &["logits"],
        output: &["y"],
        name: "softmax",
        op_type: "Softmax",
        attributes: &[build_attribute_int("axis", 1)],
    });

    let graph = build_graph(&[gemm1, relu, gemm2, softmax], "mlp", &[w1_tensor, b1_tensor, w2_tensor, b2_tensor], &[x_input], &[y_output]);

    let mut bytes = Vec::new();
    push_varint(1, 8, &mut bytes); // ir_version
    push_len(7, &graph, &mut bytes); // graph

    (bytes, x_data)
}

#[test]
fn an_onnx_lowered_mlp_runs_on_metal_at_cpu_parity_through_the_backend_wrapper() {
    let (bytes, x_data) = two_layer_mlp_model_bytes();

    let model = parse_complete(&bytes).expect("parse the mlp model bytes");
    let onnx_graph = model.graph.as_ref().expect("mlp graph present");
    let lowered = lower_graph(onnx_graph).expect("lower the mlp graph to Op");

    let mut owned: Vec<(String, Vec<f32>)> = lowered.initializers.clone();
    owned.push(("x".to_string(), x_data.to_vec()));
    let named: Vec<(&str, QuantizedBlock<'_>)> = owned
        .iter()
        .map(|(name, data)| (name.as_str(), QuantizedBlock::Float32(data.as_slice())))
        .collect();

    let output_node = lowered
        .graph_outputs
        .iter()
        .find(|(name, _)| name.as_str() == "y")
        .expect("y is a declared graph output")
        .1;
    let roots = [output_node];

    let mut cpu_plan = plan_named(Backend::Cpu, &lowered.program, &[], &named, &roots)
        .expect("omega::backend plans the onnx-lowered mlp on cpu");
    let cpu = execute_plan_named(&mut cpu_plan, &named)
        .expect("omega::backend runs the onnx-lowered mlp on cpu");

    let mut metal_plan = plan_named(Backend::Metal, &lowered.program, &[], &named, &roots)
        .expect("omega::backend plans the onnx-lowered mlp on metal");
    let metal = execute_plan_named(&mut metal_plan, &named)
        .expect("omega::backend runs the onnx-lowered mlp on a real device");

    let (cpu_data, cpu_shape) = cpu.get(output_node).expect("cpu y present");
    let (metal_data, metal_shape) = metal.get(output_node).expect("metal y present");

    assert_eq!(cpu_shape, &[2, 2]);
    assert_eq!(metal_shape, cpu_shape, "metal preserves the cpu-inferred output shape");
    assert_eq!(metal_data.len(), cpu_data.len());

    // hand-computed via `bc -l`, same reference `proxima-onnx`'s own
    // `onnx_bytes_lower_to_op_and_evaluate_a_two_layer_mlp` asserts against:
    // softmax(relu(x @ W1 + b1) @ W2 + b2).
    let expected = [0.599_888_4_f32, 0.400_111_6, 0.434_749_25, 0.565_250_74];
    for (actual, expected) in metal_data.iter().zip(expected.iter()) {
        assert!((actual - expected).abs() < 1e-3, "metal softmax output {actual} does not match hand-computed reference {expected}");
    }

    // GPU-appropriate tolerance (`metal_parity.rs`'s widened arms use 5e-3
    // for a matmul-into-softmax-into-matmul chain on f16; this chain is f32
    // throughout, so 1e-3 covers the reorder-driven float error without
    // hiding a real divergence).
    let max_abs_diff = cpu_data
        .iter()
        .zip(metal_data.iter())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0f32, f32::max);
    println!("onnx mlp cpu-vs-metal: {} elements compared, max abs diff = {max_abs_diff:e}", cpu_data.len());
    assert!(
        max_abs_diff <= 1e-3,
        "omega's cpu and metal arms disagree on the onnx-lowered mlp: max_abs_diff={max_abs_diff}"
    );

    let row0_sum = metal_data[0] + metal_data[1];
    let row1_sum = metal_data[2] + metal_data[3];
    assert!((row0_sum - 1.0).abs() < 1e-3, "metal softmax row 0 sums to {row0_sum}, not 1.0");
    assert!((row1_sum - 1.0).abs() < 1e-3, "metal softmax row 1 sums to {row1_sum}, not 1.0");
}
