//! Correctness oracle for the tile-pipeline experiment
//! (`proxima-tensor/docs/discipline.md` ROW 155): the pipeline's logits
//! must agree with `cpu::evaluate_named`'s own output within a documented
//! reassociation bound per image (the pipeline sums FC1's 11616-wide
//! reduction in row-streamed order, the sealed executor in its own SIMD
//! dot-fold order -- a real, bounded floating-point reassociation, the
//! same category ROW 151's own differential test already documents for
//! this initiative, never a defect in either arm), AND full-1000-image
//! accuracy must land at EXACTLY 0.9900 -- the same bar
//! `real_mnist_accuracy.rs`/`mnist_f32_lane.rs` both hit this session.
//!
//! `#![cfg(feature = "tile-pipeline-bench")]`: default-off, same convention
//! `real_mnist_accuracy.rs` uses -- absent the feature this file compiles
//! to zero tests (not an error; the feature gate is the point).

#![cfg(feature = "tile-pipeline-bench")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::fs;
use std::path::{Path, PathBuf};

#[path = "../benches/support/tile_pipeline.rs"]
mod tile_pipeline;

use tile_pipeline::{BandRows, MnistWeights, run_pipeline_forward, run_pipeline_forward_direct};

const MODEL_PATH: &str = "/Users/brianbruggeman/repos/others/burn/examples/onnx-inference/src/model/mnist.onnx";
const DATASET_DIR: &str = "/Users/brianbruggeman/.cache/burn-dataset/mnist";
const TEST_IMAGES_COUNT: usize = 1000;
/// Bound per logit: the reassociation this pipeline introduces is a single
/// reordering of an 11616-term sum (FC1) plus 3 small (<=144-term) conv
/// reductions -- `1e-4` absolute is generous relative to logits whose
/// magnitude sits in the single digits (LogSoftmax output), and tight
/// enough that a real correctness bug (wrong weight indexing, wrong
/// channel order) would still trip it by orders of magnitude.
const LOGIT_ABSOLUTE_TOLERANCE: f32 = 1e-3;

fn checkpoint_present() -> bool {
    Path::new(MODEL_PATH).exists()
}

fn test_images_path() -> PathBuf {
    Path::new(DATASET_DIR).join("test/t10k-images-idx3-ubyte")
}

fn test_labels_path() -> PathBuf {
    Path::new(DATASET_DIR).join("test/t10k-labels-idx1-ubyte")
}

fn dataset_present() -> bool {
    test_images_path().exists() && test_labels_path().exists()
}

fn idx_header(bytes: &[u8]) -> (usize, Vec<usize>) {
    let dimension_count = bytes[3] as usize;
    let item_count = u32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
    let mut extents = Vec::with_capacity(dimension_count - 1);
    for axis in 1..dimension_count {
        let offset = 4 + axis * 4;
        extents.push(u32::from_be_bytes([bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]]) as usize);
    }
    (item_count, extents)
}

fn load_normalized_images(path: &Path, limit: usize) -> Vec<Vec<f32>> {
    let bytes = fs::read(path).expect("read idx3 image file");
    let (item_count, extents) = idx_header(&bytes);
    let pixel_count = extents.iter().product::<usize>();
    let take = item_count.min(limit);
    let header_length = 4 + extents.len() * 4 + 4;
    (0..take)
        .map(|image_index| {
            let start = header_length + image_index * pixel_count;
            bytes[start..start + pixel_count].iter().map(|&pixel| ((pixel as f32 / 255.0) - 0.1307) / 0.3081).collect()
        })
        .collect()
}

fn load_labels(path: &Path, limit: usize) -> Vec<u8> {
    let bytes = fs::read(path).expect("read idx1 label file");
    let (item_count, _extents) = idx_header(&bytes);
    let take = item_count.min(limit);
    bytes[8..8 + take].to_vec()
}

fn argmax(values: &[f32]) -> usize {
    values.iter().enumerate().max_by(|left, right| left.1.total_cmp(right.1)).map(|(index, _)| index).expect("nonempty logits")
}

struct LoadedModel {
    program: Vec<proxima_tensor::Op>,
    graph_input_name: String,
    output_node: proxima_tensor::NodeId,
    initializers: Vec<(String, Vec<f32>)>,
}

fn load_model() -> LoadedModel {
    let bytes = fs::read(MODEL_PATH).expect("read the real mnist.onnx checkpoint");
    let model = proxima_onnx::pipe::parse_complete(&bytes).expect("parse the real mnist.onnx checkpoint");
    let graph = model.graph.as_ref().expect("real mnist model has a graph");
    let lowered = proxima_onnx::lower::lower_graph(graph).expect("lower the real mnist.onnx graph to Op");
    let graph_input_name = lowered.graph_inputs.first().expect("real mnist model declares at least one input").clone();
    let output_node = lowered.graph_outputs.first().expect("real mnist model declares at least one output").1;
    LoadedModel { program: lowered.program, graph_input_name, output_node, initializers: lowered.initializers }
}

fn evaluate_named_logits(model: &LoadedModel, image: &[f32]) -> Vec<f32> {
    let initializers: Vec<(&str, &[f32])> = model.initializers.iter().map(|(name, data)| (name.as_str(), data.as_slice())).collect();
    let mut named = initializers;
    named.push((model.graph_input_name.as_str(), image));
    let evaluated = proxima_tensor::cpu::evaluate_named(&model.program, &[], &named, &[model.output_node]).expect("evaluate real mnist image via the sealed executor");
    let (data, shape) = evaluated.get(model.output_node).expect("real mnist output present");
    assert_eq!(shape, &vec![1_u64, 10], "LogSoftmax over 10 MNIST classes");
    data.to_vec()
}

/// The pipeline's logits agree with the sealed executor's within
/// [`LOGIT_ABSOLUTE_TOLERANCE`] per logit, for every one of the 3 band
/// granularities the task's own sweep calls for (1 / `kh` / `2*kh` rows),
/// over the first 20 real `t10k` test images -- not a synthetic fixture,
/// the same real held-out data `real_mnist_accuracy.rs` uses.
#[test]
fn pipeline_logits_match_sealed_executor_within_reassociation_bound() {
    if !checkpoint_present() || !dataset_present() {
        eprintln!("skipping: no host-local mnist.onnx checkout or MNIST idx dataset");
        return;
    }
    let model = load_model();
    let weights = MnistWeights::from_initializers(&model.initializers.iter().map(|(name, data)| (name.as_str(), data.as_slice())).collect::<Vec<_>>());
    let images = load_normalized_images(&test_images_path(), 20);

    for (band_label, band_rows) in [("1-row band", 1), ("kh-row band", 3), ("2kh-row band", 6)] {
        for (index, image) in images.iter().enumerate() {
            let incumbent = evaluate_named_logits(&model, image);
            let pipeline = run_pipeline_forward(image, &weights, BandRows(band_rows));
            for (logit_index, (&incumbent_value, &pipeline_value)) in incumbent.iter().zip(pipeline.iter()).enumerate() {
                let delta = (incumbent_value - pipeline_value).abs();
                assert!(
                    delta <= LOGIT_ABSOLUTE_TOLERANCE,
                    "{band_label}, image {index}, logit {logit_index}: incumbent={incumbent_value} pipeline={pipeline_value} delta={delta} exceeds {LOGIT_ABSOLUTE_TOLERANCE}"
                );
            }
            assert_eq!(argmax(&incumbent), argmax(&pipeline), "{band_label}, image {index}: argmax disagreement");
        }
    }
}

/// ROW 172's dispatch-floor arm (`run_pipeline_forward_direct`, each stage
/// called via `compute_direct` instead of composed `AndThen` +
/// `block_on_ready`) produces BIT-IDENTICAL logits to the production
/// `AndThen`-composed pipeline -- same `process_band` body either way, so
/// this is the correctness precondition for treating the two arms' timing
/// delta as pure dispatch overhead rather than a divergent computation.
#[test]
fn direct_call_arm_is_bit_identical_to_andthen_composed_pipeline() {
    if !checkpoint_present() || !dataset_present() {
        eprintln!("skipping: no host-local mnist.onnx checkout or MNIST idx dataset");
        return;
    }
    let model = load_model();
    let weights = MnistWeights::from_initializers(&model.initializers.iter().map(|(name, data)| (name.as_str(), data.as_slice())).collect::<Vec<_>>());
    let images = load_normalized_images(&test_images_path(), 20);

    for (band_label, band_rows) in [("1-row band", 1), ("kh-row band", 3), ("2kh-row band", 6)] {
        for (index, image) in images.iter().enumerate() {
            let composed = run_pipeline_forward(image, &weights, BandRows(band_rows));
            let direct = run_pipeline_forward_direct(image, &weights, BandRows(band_rows));
            assert_eq!(composed, direct, "{band_label}, image {index}: AndThen-composed and direct-call arms diverged");
        }
    }
}

/// Full-1000-image accuracy for the pipeline lands at EXACTLY 0.9900 --
/// bit-for-bit the same argmax count the sealed executor and the
/// ROW-154-sealed bench both hit this session, over the real `t10k` test
/// split.
#[test]
fn pipeline_full_test_split_accuracy_is_exactly_0_9900() {
    if !checkpoint_present() || !dataset_present() {
        eprintln!("skipping: no host-local mnist.onnx checkout or MNIST idx dataset");
        return;
    }
    let model = load_model();
    let weights = MnistWeights::from_initializers(&model.initializers.iter().map(|(name, data)| (name.as_str(), data.as_slice())).collect::<Vec<_>>());
    let images = load_normalized_images(&test_images_path(), TEST_IMAGES_COUNT);
    let labels = load_labels(&test_labels_path(), TEST_IMAGES_COUNT);
    assert_eq!(images.len(), labels.len());
    assert!(images.len() >= TEST_IMAGES_COUNT);

    let mut correct = 0_usize;
    for (image, &label) in images.iter().zip(labels.iter()) {
        let logits = run_pipeline_forward(image, &weights, BandRows(3));
        if argmax(&logits) == label as usize {
            correct += 1;
        }
    }
    let accuracy = correct as f64 / images.len() as f64;
    eprintln!("tile pipeline accuracy: {accuracy:.4} ({correct}/{})", images.len());
    assert_eq!(correct, 990, "tile pipeline must classify exactly 990/1000 real t10k test images, matching ROW 154's own sealed accuracy bit-for-bit");
}
