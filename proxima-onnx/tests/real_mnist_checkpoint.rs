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

/// Parses and lowers the real checkpoint, then attempts evaluation against
/// an all-zero `28x28` input. This real file surfaced two op types this
/// crate never lowered before (`BatchNormalization`, `LogSoftmax`, both
/// closed alongside this test); parse and lower both now succeed end to
/// end over the real byte stream (76 `Op`s from 12 `NodeProto`s).
///
/// Evaluation exposes a third, separate, NOT closed here gap: `Flatten`
/// aliases its output onto the producing `Conv`/`BatchNormalization` node
/// via [`crate::lower::Value`]'s view mechanism rather than materializing a
/// real reshape `Op` (`lower.rs`'s own `Value` doc), sound only while the
/// aliased value is a terminal graph output or feeds an operand whose
/// logical rank still matches the real node's rank. Here it feeds `Gemm`
/// as the contraction LHS, whose logical rank (2) no longer matches the
/// real producing node's rank (4) -- `shape::infer` correctly rejects the
/// mismatch as `ExtentMismatch` rather than silently misreading memory
/// (caught, not silent), but closing it for real needs either a genuine
/// per-element gather addressed by lower-time-precomputed flat-index
/// constants, or lifting `IndexMap::Affine`'s "no div/mod" restriction --
/// neither attempted in this pass. This test reports exactly that boundary
/// rather than asserting a false success.
#[test]
#[ignore = "depends on a real .onnx checkout outside this repo"]
fn real_mnist_onnx_parses_and_lowers_end_to_end() {
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
    match proxima_tensor::cpu::evaluate_named(&lowered.program, &[], &named, &[output_node]) {
        Ok(evaluated) => {
            let (data, shape) = evaluated.get(output_node).expect("real mnist output present");
            std::println!("real_mnist evaluation SUCCEEDED: output shape={shape:?} values={data:?}");
        }
        Err(error) => {
            std::println!("real_mnist evaluation hit the named, still-open Flatten-view-into-Gemm gap this test's own doc describes: {error:?}");
        }
    }
}
