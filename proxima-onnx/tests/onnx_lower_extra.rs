//! `LayerNormalization`, `Gelu`, `Squeeze` and `Expand` lowered through
//! [`proxima_onnx::lower::lower_graph`] and evaluated on the CPU backend.
//! Numeric expectations are hand-derived from the formulas in the ONNX
//! `Operators.md` (cited per case) and written as literals; each numeric
//! case has a control that alters the expected value and must fail.

// the harness asserts on a successful lowering and evaluation, so a failure there is the test failure
#![allow(clippy::expect_used)]

use proxima_onnx::lower::{LowerError, lower_graph};
use proxima_onnx::messages::{AttributeProto, GraphProto, NodeProto, TensorProto, ValueInfoProto};
use proxima_tensor::cpu::evaluate_named;

const TOLERANCE: f32 = 1e-5;
const ALTERATION: f32 = 1e-3;

fn float_tensor(name: &'static str, dims: &[i64], data: &[f32]) -> TensorProto<'static> {
    TensorProto {
        dims: dims.to_vec(),
        data_type: 1,
        float_data: data.to_vec(),
        name,
        ..TensorProto::default()
    }
}

fn int64_tensor(name: &'static str, values: &[i64]) -> TensorProto<'static> {
    TensorProto {
        dims: vec![values.len() as i64],
        data_type: 7,
        int64_data: values.to_vec(),
        name,
        ..TensorProto::default()
    }
}

fn node(
    op_type: &'static str,
    input: &[&'static str],
    output: &'static str,
    attribute: Vec<AttributeProto<'static>>,
) -> NodeProto<'static> {
    NodeProto {
        input: input.to_vec(),
        output: vec![output],
        op_type,
        name: op_type,
        attribute,
        ..NodeProto::default()
    }
}

fn float_attribute(name: &'static str, value: f32) -> AttributeProto<'static> {
    AttributeProto {
        name,
        f: value,
        ..AttributeProto::default()
    }
}

fn int_attribute(name: &'static str, value: i64) -> AttributeProto<'static> {
    AttributeProto {
        name,
        i: value,
        ..AttributeProto::default()
    }
}

fn ints_attribute(name: &'static str, values: &[i64]) -> AttributeProto<'static> {
    AttributeProto {
        name,
        ints: values.to_vec(),
        ..AttributeProto::default()
    }
}

fn string_attribute(name: &'static str, value: &'static [u8]) -> AttributeProto<'static> {
    AttributeProto {
        name,
        s: value,
        ..AttributeProto::default()
    }
}

fn run(
    nodes: Vec<NodeProto<'static>>,
    initializer: Vec<TensorProto<'static>>,
) -> Result<(Vec<f32>, Vec<u64>), LowerError> {
    let graph = GraphProto {
        node: nodes,
        name: "extra_graph",
        initializer,
        output: vec![ValueInfoProto {
            name: "y",
            ..ValueInfoProto::default()
        }],
        ..GraphProto::default()
    };
    let lowered = lower_graph(&graph)?;
    let named: Vec<(&str, &[f32])> = lowered
        .initializers
        .iter()
        .map(|(name, data)| (name.as_str(), data.as_slice()))
        .collect();
    let output = lowered.graph_outputs[0].1;
    let evaluated = evaluate_named(&lowered.program, &[], &named, &[output])
        .expect("the lowered program evaluates on the cpu backend");
    let (data, shape) = evaluated.get(output).expect("the graph output is present");
    Ok((data.to_vec(), shape.to_vec()))
}

fn max_abs_difference(actual: &[f32], expected: &[f32]) -> f32 {
    assert_eq!(actual.len(), expected.len(), "element counts differ");
    actual
        .iter()
        .zip(expected)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f32::max)
}

fn altered(expected: &[f32]) -> Vec<f32> {
    let mut copy = expected.to_vec();
    copy[0] += ALTERATION;
    copy
}

const LAYER_NORM_INPUT: [f32; 8] = [1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0];
const LAYER_NORM_SCALE: [f32; 4] = [1.0, 0.5, 2.0, 1.5];
const LAYER_NORM_BIAS: [f32; 4] = [0.1, -0.2, 0.3, 0.0];

// y = (x - mean) / sqrt(variance + 1e-5) * scale + bias per row: row one has
// mean 2.5 and variance 1.25, row two mean 5 and variance 5.
const LAYER_NORM_WITH_BIAS_EXPECTED: [f32; 8] = [
    -1.241_635_4,
    -0.423_605_9,
    1.194_423_6,
    2.012_453,
    -1.241_639_4,
    -0.423_606_6,
    1.194_426_3,
    2.012_459_2,
];

// epsilon = 0 makes row two a doubling of row one, so both rows normalize
// to the same values: centered / sqrt(variance) = [-1.3416408, -0.4472136,
// 0.4472136, 1.3416408], times scale.
const LAYER_NORM_NO_BIAS_EXPECTED: [f32; 8] = [
    -1.341_640_8,
    -0.223_606_8,
    0.894_427_2,
    2.012_461_2,
    -1.341_640_8,
    -0.223_606_8,
    0.894_427_2,
    2.012_461_2,
];

fn layer_normalization_with_scale_and_bias() -> Vec<f32> {
    let layer_norm = node(
        "LayerNormalization",
        &["x", "scale", "bias"],
        "y",
        vec![int_attribute("axis", -1)],
    );
    let (data, shape) = run(
        vec![layer_norm],
        vec![
            float_tensor("x", &[2, 4], &LAYER_NORM_INPUT),
            float_tensor("scale", &[4], &LAYER_NORM_SCALE),
            float_tensor("bias", &[4], &LAYER_NORM_BIAS),
        ],
    )
    .expect("lower LayerNormalization with scale and bias");
    assert_eq!(shape, vec![2, 4]);
    data
}

fn layer_normalization_without_bias() -> Vec<f32> {
    let layer_norm = node(
        "LayerNormalization",
        &["x", "scale"],
        "y",
        vec![float_attribute("epsilon", 0.0)],
    );
    let (data, shape) = run(
        vec![layer_norm],
        vec![
            float_tensor("x", &[2, 4], &LAYER_NORM_INPUT),
            float_tensor("scale", &[4], &LAYER_NORM_SCALE),
        ],
    )
    .expect("lower LayerNormalization without bias");
    assert_eq!(shape, vec![2, 4]);
    data
}

#[test]
fn onnx_lower_extra_layer_normalization_axis_last_with_scale_and_bias() {
    let actual = layer_normalization_with_scale_and_bias();

    assert!(
        max_abs_difference(&actual, &LAYER_NORM_WITH_BIAS_EXPECTED) <= TOLERANCE,
        "{actual:?}"
    );
}

#[test]
fn onnx_lower_extra_layer_normalization_axis_last_with_scale_and_bias_altered_expected_fails() {
    let actual = layer_normalization_with_scale_and_bias();

    assert!(max_abs_difference(&actual, &altered(&LAYER_NORM_WITH_BIAS_EXPECTED)) > TOLERANCE);
}

#[test]
fn onnx_lower_extra_layer_normalization_without_bias() {
    let actual = layer_normalization_without_bias();

    assert!(
        max_abs_difference(&actual, &LAYER_NORM_NO_BIAS_EXPECTED) <= TOLERANCE,
        "{actual:?}"
    );
}

#[test]
fn onnx_lower_extra_layer_normalization_without_bias_altered_expected_fails() {
    let actual = layer_normalization_without_bias();

    assert!(max_abs_difference(&actual, &altered(&LAYER_NORM_NO_BIAS_EXPECTED)) > TOLERANCE);
}

const GELU_INPUT: [f32; 5] = [-1.0, 0.0, 0.5, 1.0, 2.0];

// 0.5 * x * (1 + erf(x / sqrt(2))) = x * Phi(x), Phi from the standard
// normal table: Phi(-1) = 0.158655254, Phi(0.5) = 0.691462461,
// Phi(1) = 0.841344746, Phi(2) = 0.977249868.
const GELU_ERF_EXPECTED: [f32; 5] = [-0.158_655_25, 0.0, 0.345_731_23, 0.841_344_8, 1.954_499_7];

// 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3))), evaluated in f64.
const GELU_TANH_EXPECTED: [f32; 5] = [-0.158_808_01, 0.0, 0.345_714, 0.841_192, 1.954_597_7];

fn gelu(attribute: Vec<AttributeProto<'static>>) -> Vec<f32> {
    let (data, shape) = run(
        vec![node("Gelu", &["x"], "y", attribute)],
        vec![float_tensor("x", &[5], &GELU_INPUT)],
    )
    .expect("lower Gelu");
    assert_eq!(shape, vec![5]);
    data
}

#[test]
fn onnx_lower_extra_gelu_erf_default() {
    let actual = gelu(Vec::new());

    assert!(
        max_abs_difference(&actual, &GELU_ERF_EXPECTED) <= TOLERANCE,
        "{actual:?}"
    );
}

#[test]
fn onnx_lower_extra_gelu_erf_default_altered_expected_fails() {
    let actual = gelu(Vec::new());

    assert!(max_abs_difference(&actual, &altered(&GELU_ERF_EXPECTED)) > TOLERANCE);
}

#[test]
fn onnx_lower_extra_gelu_tanh_approximation() {
    let actual = gelu(vec![string_attribute("approximate", b"tanh")]);

    assert!(
        max_abs_difference(&actual, &GELU_TANH_EXPECTED) <= TOLERANCE,
        "{actual:?}"
    );
}

#[test]
fn onnx_lower_extra_gelu_tanh_approximation_altered_expected_fails() {
    let actual = gelu(vec![string_attribute("approximate", b"tanh")]);

    assert!(max_abs_difference(&actual, &altered(&GELU_TANH_EXPECTED)) > TOLERANCE);
}

const SQUEEZE_INPUT: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

#[test]
fn onnx_lower_extra_squeeze_axes_input() {
    let squeeze = node("Squeeze", &["x", "axes"], "squeezed", Vec::new());
    let transpose = node(
        "Transpose",
        &["squeezed"],
        "y",
        vec![ints_attribute("perm", &[1, 0])],
    );

    let (data, shape) = run(
        vec![squeeze, transpose],
        vec![
            float_tensor("x", &[1, 3, 1, 2], &SQUEEZE_INPUT),
            int64_tensor("axes", &[0, 2]),
        ],
    )
    .expect("lower Squeeze with an axes input");

    assert_eq!(shape, vec![2, 3]);
    assert_eq!(data, vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
}

#[test]
fn onnx_lower_extra_squeeze_without_axes_drops_every_unit_dimension() {
    let (data, shape) = run(
        vec![node("Squeeze", &["x"], "y", Vec::new())],
        vec![float_tensor("x", &[1, 3, 1, 2], &SQUEEZE_INPUT)],
    )
    .expect("lower Squeeze without axes");

    assert_eq!(shape, vec![3, 2]);
    assert_eq!(data, SQUEEZE_INPUT.to_vec());
}

#[test]
fn onnx_lower_extra_expand_broadcasts_to_a_larger_shape() {
    let unsqueeze = node(
        "Unsqueeze",
        &["x"],
        "lifted",
        vec![ints_attribute("axes", &[0])],
    );
    let expand = node("Expand", &["lifted", "target"], "y", Vec::new());

    let (data, shape) = run(
        vec![unsqueeze, expand],
        vec![
            float_tensor("x", &[3, 1], &[1.0, 2.0, 3.0]),
            int64_tensor("target", &[2, 3, 4]),
        ],
    )
    .expect("lower Expand over an Unsqueeze view");

    let one_batch = [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0];
    assert_eq!(shape, vec![2, 3, 4]);
    assert_eq!(data, [one_batch, one_batch].concat());
}

#[test]
fn onnx_lower_extra_expand_rejects_a_shape_that_cannot_broadcast() {
    let outcome = run(
        vec![node("Expand", &["x", "target"], "y", Vec::new())],
        vec![
            float_tensor("x", &[3], &[1.0, 2.0, 3.0]),
            int64_tensor("target", &[4]),
        ],
    );

    assert!(
        matches!(&outcome, Err(LowerError::UnsupportedShape { op_type, .. }) if op_type == "Expand"),
        "{outcome:?}"
    );
}
