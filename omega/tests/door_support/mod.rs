//! The two-layer MLP the door tests share, built as `proxima_onnx` message
//! structs and encoded with `write_model_proto`, so a door parses real wire
//! bytes.

use std::collections::BTreeMap;

use omega::door::DoorBatch;
use proxima_onnx::messages::{
    AttributeProto, Dimension, DimensionValue, GraphProto, ModelProto, NodeProto, TensorProto,
    TensorShapeProto, TypeProto, TypeProtoTensor, TypeValue, ValueInfoProto,
};
use proxima_onnx::write_model_proto;

pub const MLP_INPUT: [f32; 6] = [1.0, 0.5, -1.0, 0.0, 2.0, 1.0];

fn float_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

pub fn fixed(extent: i64) -> Dimension<'static> {
    Dimension {
        value: Some(DimensionValue::Value(extent)),
        denotation: "",
    }
}

pub fn float_input(name: &'static str, dims: Vec<Dimension<'static>>) -> ValueInfoProto<'static> {
    let tensor = TypeProtoTensor {
        elem_type: 1,
        shape: Some(TensorShapeProto { dim: dims }),
    };
    ValueInfoProto {
        name,
        r#type: Some(TypeProto {
            value: Some(TypeValue::Tensor(tensor)),
            denotation: "",
        }),
        doc_string: "",
    }
}

pub fn weights<'a>(name: &'a str, dims: &[i64], raw: &'a [u8]) -> TensorProto<'a> {
    TensorProto {
        dims: dims.to_vec(),
        data_type: 1,
        name,
        raw_data: Some(raw),
        ..TensorProto::default()
    }
}

pub fn node<'a>(op_type: &'a str, inputs: &[&'a str], output: &'a str) -> NodeProto<'a> {
    NodeProto {
        input: inputs.to_vec(),
        output: vec![output],
        name: output,
        op_type,
        ..NodeProto::default()
    }
}

pub fn two_layer_mlp_bytes() -> Vec<u8> {
    let w1 = float_bytes(&[
        0.1, 0.2, -0.1, 0.05, 0.3, -0.2, 0.4, 0.1, -0.5, 0.1, 0.2, -0.3,
    ]);
    let b1 = float_bytes(&[0.1, -0.1, 0.05, 0.0]);
    let w2 = float_bytes(&[0.2, -0.3, 0.1, 0.4, -0.2, 0.05, 0.3, 0.1]);
    let b2 = float_bytes(&[0.0, 0.1]);
    let softmax = NodeProto {
        attribute: vec![AttributeProto {
            name: "axis",
            type_raw: 2,
            i: 1,
            ..AttributeProto::default()
        }],
        ..node("Softmax", &["logits"], "y")
    };
    let graph = GraphProto {
        node: vec![
            node("Gemm", &["x", "W1", "b1"], "h"),
            node("Relu", &["h"], "hr"),
            node("Gemm", &["hr", "W2", "b2"], "logits"),
            softmax,
        ],
        name: "mlp",
        initializer: vec![
            weights("W1", &[3, 4], &w1),
            weights("b1", &[4], &b1),
            weights("W2", &[4, 2], &w2),
            weights("b2", &[2], &b2),
        ],
        input: vec![float_input("x", vec![fixed(2), fixed(3)])],
        output: vec![ValueInfoProto {
            name: "y",
            ..ValueInfoProto::default()
        }],
        ..GraphProto::default()
    };
    write_model_proto(&ModelProto {
        ir_version: 8,
        graph: Some(graph),
        ..ModelProto::default()
    })
}

pub fn mlp_batch() -> DoorBatch {
    DoorBatch {
        pins: BTreeMap::new(),
        inputs: vec![("x".to_owned(), MLP_INPUT.to_vec())],
    }
}
