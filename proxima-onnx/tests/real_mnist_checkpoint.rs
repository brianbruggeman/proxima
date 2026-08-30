//! Real, on-disk MNIST classifier (`mnist.onnx`, a LeNet-style
//! `Conv/BatchNormalization/Relu/Flatten/Gemm/LogSoftmax` stack exported
//! from a real training run, not a synthetic fixture), run through
//! [`proxima_onnx::pipe::parse_complete`] -> [`proxima_onnx::lower::lower_graph`]
//! -> [`proxima_tensor::cpu::evaluate_named`]. `#[ignore]`d and skips
//! cleanly when the host-local checkout is absent, the same convention
//! `proxima-model-interop/tests/real_lfm2_checkpoint.rs` uses for its own
//! host-local `.gguf` fixture.

#![cfg(feature = "std")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::vec::Vec;

const MODEL_PATH: &str = "/Users/brianbruggeman/repos/others/burn/examples/onnx-inference/src/model/mnist.onnx";

fn checkpoint_present() -> bool {
    std::path::Path::new(MODEL_PATH).exists()
}

/// Parses, lowers, AND evaluates the real checkpoint end to end against an
/// all-zero `28x28` input. This real file surfaced two op types this crate
/// never lowered before (`BatchNormalization`, `LogSoftmax`, closed
/// alongside this test's own first landing) and, at evaluation time, a
/// third: `Flatten` aliases its output onto the producing
/// `Conv`/`BatchNormalization` node via [`crate::lower::Value`]'s view
/// mechanism rather than materializing a real reshape `Op`, sound only
/// while the aliased value's logical rank still matches the real node's
/// rank -- broken here, where it feeds `Gemm` as the contraction LHS with
/// logical rank 2 against a real producing rank of 4.
///
/// Closed not by materializing a new reshaped node (the *read*-side
/// algebra has no primitive that merges several real axes into one --
/// `IndexMap::Computed` gathers exactly one axis per operand reference, see
/// `proxima-tensor/src/map.rs`'s own doc), but by widening `Gemm`'s own
/// matmul iteration space: `Value::flatten_source` records which real axes
/// `Flatten` merged, and `lower_gemm` gives the contracted `K` axis one
/// iteration axis per real axis the merge covers, addressing the real
/// `Conv`/`BatchNormalization` node directly and reproducing the flat
/// index on the OTHER (plain, single-real-axis) operand as a genuine
/// [`proxima_tensor::AxisTerm`] sum -- the same multi-term-axis machinery
/// convolution's own `h*stride + r*dilation` already uses, so
/// `IndexMap::Affine`'s no-div/mod restriction is never touched.
#[test]
#[ignore = "depends on a real .onnx checkout outside this repo"]
fn real_mnist_onnx_parses_lowers_and_evaluates_end_to_end() {
    if !checkpoint_present() {
        eprintln!("skipping: no host-local mnist.onnx checkout at {MODEL_PATH}");
        return;
    }
    let bytes = std::fs::read(MODEL_PATH).expect("read the real mnist.onnx checkpoint");
    let model = proxima_onnx::pipe::parse_complete(&bytes).expect("parse the real mnist.onnx checkpoint");
    let graph = model.graph.as_ref().expect("real mnist model has a graph");

    let mut op_counts: std::collections::BTreeMap<&str, usize> = std::collections::BTreeMap::new();
    for node in &graph.node {
        *op_counts.entry(node.op_type).or_insert(0) += 1;
    }
    std::println!("real_mnist op histogram: {op_counts:?}");

    let lowered =
        proxima_onnx::lower::lower_graph(graph).expect("lower the real mnist.onnx graph to Op (BatchNormalization + LogSoftmax now supported)");
    std::println!("real_mnist lowered program length: {}", lowered.program.len());
    assert!(!lowered.program.is_empty(), "a real model lowers to a nonempty Op program");

    let graph_input_name = lowered.graph_inputs.first().expect("real mnist model declares at least one input").clone();
    let zero_input: Vec<f32> = std::vec![0.0_f32; 28 * 28];
    let mut named: Vec<(&str, &[f32])> = lowered.initializers.iter().map(|(name, data)| (name.as_str(), data.as_slice())).collect();
    named.push((graph_input_name.as_str(), zero_input.as_slice()));

    let output_node = lowered.graph_outputs.first().expect("real mnist model declares at least one output").1;
    let evaluated = proxima_tensor::cpu::evaluate_named(&lowered.program, &[], &named, &[output_node])
        .expect("real mnist evaluation succeeds now that Flatten-into-Gemm addresses the real producing node directly");
    let (data, shape) = evaluated.get(output_node).expect("real mnist output present");
    std::println!("real_mnist evaluation SUCCEEDED: output shape={shape:?} values={data:?}");

    assert_eq!(shape, &std::vec![1_u64, 10], "LogSoftmax over 10 MNIST classes");
    assert_eq!(data.len(), 10, "one log-probability per class");
    assert!(data.iter().all(|value| value.is_finite()), "every log-probability is finite, got {data:?}");
    let probability_mass: f32 = data.iter().map(|log_probability| log_probability.exp()).sum();
    assert!((probability_mass - 1.0).abs() < 1e-3, "exp(log-softmax) sums to 1 (a valid probability distribution), got {probability_mass}");
}
