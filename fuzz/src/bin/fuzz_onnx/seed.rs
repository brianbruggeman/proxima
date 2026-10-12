use proxima_onnx::messages::{
    AttributeProto, Dimension, DimensionValue, GraphProto, ModelProto, NodeProto,
    OperatorSetIdProto, TensorProto, TensorShapeProto, TypeProto, TypeProtoTensor, TypeValue,
    ValueInfoProto,
};
use proxima_onnx::writer::write_model_proto;

const FIRST_WEIGHTS: [f32; 12] = [
    0.1, 0.2, -0.1, 0.05, 0.3, -0.2, 0.4, 0.1, -0.5, 0.1, 0.2, -0.3,
];
const FIRST_BIAS: [f32; 4] = [0.1, -0.1, 0.05, 0.0];
const SECOND_WEIGHTS: [f32; 8] = [0.2, -0.3, 0.1, 0.4, -0.2, 0.05, 0.3, 0.1];
const SECOND_BIAS: [f32; 2] = [0.0, 0.1];

pub fn seed_model_bytes() -> Vec<u8> {
    let first_weights_raw = f32_le_bytes(&FIRST_WEIGHTS);
    let model = ModelProto {
        ir_version: 8,
        opset_import: vec![OperatorSetIdProto {
            domain: "",
            version: 18,
        }],
        producer_name: "pytorch",
        graph: Some(mlp_graph(&first_weights_raw)),
        ..ModelProto::default()
    };
    write_model_proto(&model)
}

fn mlp_graph(first_weights_raw: &[u8]) -> GraphProto<'_> {
    GraphProto {
        node: mlp_nodes(),
        name: "seed_mlp",
        initializer: vec![
            raw_initializer("W1", &[3, 4], first_weights_raw),
            float_initializer("b1", &[4], &FIRST_BIAS),
            float_initializer("W2", &[4, 2], &SECOND_WEIGHTS),
            float_initializer("b2", &[2], &SECOND_BIAS),
        ],
        input: vec![tensor_input("x", &[2, 3])],
        output: vec![ValueInfoProto {
            name: "y",
            ..ValueInfoProto::default()
        }],
        ..GraphProto::default()
    }
}

fn mlp_nodes() -> Vec<NodeProto<'static>> {
    let softmax_axis = AttributeProto {
        name: "axis",
        i: 1,
        type_raw: 2,
        ..AttributeProto::default()
    };
    vec![
        node("gemm1", "Gemm", &["x", "W1", "b1"], "h"),
        node("relu", "Relu", &["h"], "hr"),
        node("gemm2", "Gemm", &["hr", "W2", "b2"], "logits"),
        NodeProto {
            attribute: vec![softmax_axis],
            ..node("softmax", "Softmax", &["logits"], "y")
        },
    ]
}

fn node(
    name: &'static str,
    op_type: &'static str,
    inputs: &[&'static str],
    output: &'static str,
) -> NodeProto<'static> {
    NodeProto {
        input: inputs.to_vec(),
        output: vec![output],
        name,
        op_type,
        ..NodeProto::default()
    }
}

fn f32_le_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn raw_initializer<'a>(name: &'a str, dims: &[i64], raw_data: &'a [u8]) -> TensorProto<'a> {
    TensorProto {
        dims: dims.to_vec(),
        data_type: 1,
        name,
        raw_data: Some(raw_data),
        ..TensorProto::default()
    }
}

fn float_initializer<'a>(name: &'a str, dims: &[i64], values: &[f32]) -> TensorProto<'a> {
    TensorProto {
        dims: dims.to_vec(),
        data_type: 1,
        name,
        float_data: values.to_vec(),
        ..TensorProto::default()
    }
}

fn tensor_input(name: &'static str, dims: &[i64]) -> ValueInfoProto<'static> {
    let dim = dims
        .iter()
        .map(|&value| Dimension {
            value: Some(DimensionValue::Value(value)),
            denotation: "",
        })
        .collect();
    ValueInfoProto {
        name,
        r#type: Some(TypeProto {
            value: Some(TypeValue::Tensor(TypeProtoTensor {
                elem_type: 1,
                shape: Some(TensorShapeProto { dim }),
            })),
            denotation: "",
        }),
        doc_string: "",
    }
}
