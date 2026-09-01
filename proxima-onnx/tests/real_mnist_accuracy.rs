//! Real classification accuracy for the real, on-disk `mnist.onnx`
//! checkpoint (see `tests/real_mnist_checkpoint.rs` for the parse/lower
//! provenance) against the real MNIST `t10k` test split — not "the output
//! is a finite distribution" but "the argmax matches the label" over real
//! held-out images, downloaded to `~/.cache/burn-dataset/mnist` the same
//! `~/repos/others/burn/examples/onnx-inference` example itself reads.
//!
//! Normalization is read from that example's own
//! `src/bin/mnist_inference.rs` (`(x/255 - 0.1307) / 0.3081` over the raw
//! `u8` pixel, never predivided — confirmed against `burn-dataset`'s own
//! `MnistItem::image`, a `[[f32; 28]; 28]` of the *raw* `0..=255` pixel
//! value, `BytesToImage::map` in `burn-dataset-0.21.0/src/vision/mnist.rs`).
//! `#[ignore]`d and skips cleanly when either the checkpoint or the idx
//! dataset files are absent, the same convention
//! `real_mnist_checkpoint.rs`/`proxima-model-interop/tests/real_lfm2_checkpoint.rs`
//! use for their own host-local fixtures.

#![cfg(feature = "std")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;
use std::vec::Vec;

const MODEL_PATH: &str = "/Users/brianbruggeman/repos/others/burn/examples/onnx-inference/src/model/mnist.onnx";
const DATASET_DIR: &str = "/Users/brianbruggeman/.cache/burn-dataset/mnist";
const TEST_IMAGES_COUNT: usize = 1000;

fn checkpoint_present() -> bool {
    Path::new(MODEL_PATH).exists()
}

fn dataset_present() -> bool {
    test_images_path().exists() && test_labels_path().exists()
}

fn test_images_path() -> PathBuf {
    Path::new(DATASET_DIR).join("test/t10k-images-idx3-ubyte")
}

fn test_labels_path() -> PathBuf {
    Path::new(DATASET_DIR).join("test/t10k-labels-idx1-ubyte")
}

/// Parses an idx3 (`u8` image) or idx1 (`u8` label) file's big-endian
/// header: a magic number, an item count, then `dimension_count - 1`
/// per-axis extents (idx3 carries `[items, rows, cols]`; idx1 carries just
/// `[items]`) — [Yann LeCun's idx format](http://yann.lecun.com/exdb/mnist/),
/// trivial enough this crate's DE-CISC boundary keeps it inline rather than
/// pulling in a parsing dependency for two call sites.
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

/// Every test image, normalized exactly as `mnist_inference.rs` normalizes
/// its single image: `(pixel / 255 - 0.1307) / 0.3081`, one `Vec<f32>` of
/// length `28*28` per image, row-major (idx3 already stores rows then
/// columns, the same order `Tensor::reshape([1, 1, 28, 28])` expects).
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

/// Parses and lowers the real `mnist.onnx` checkpoint once, then evaluates
/// it over [`TEST_IMAGES_COUNT`] real `t10k` test images, asserting top-1
/// accuracy against the real labels — the upgrade `real_mnist_checkpoint.rs`'s
/// own doc calls for: "only checks the output is a finite distribution"
/// becomes "matches the label".
#[test]
#[ignore = "depends on a real .onnx checkout and the real MNIST idx dataset outside this repo"]
fn real_mnist_onnx_classifies_real_test_images_at_reference_accuracy() {
    if !checkpoint_present() {
        eprintln!("skipping: no host-local mnist.onnx checkout at {MODEL_PATH}");
        return;
    }
    if !dataset_present() {
        eprintln!("skipping: no host-local MNIST idx dataset under {DATASET_DIR}");
        return;
    }

    let bytes = fs::read(MODEL_PATH).expect("read the real mnist.onnx checkpoint");
    let model = proxima_onnx::pipe::parse_complete(&bytes).expect("parse the real mnist.onnx checkpoint");
    let graph = model.graph.as_ref().expect("real mnist model has a graph");
    let lowered = proxima_onnx::lower::lower_graph(graph).expect("lower the real mnist.onnx graph to Op");

    let graph_input_name = lowered.graph_inputs.first().expect("real mnist model declares at least one input").clone();
    let output_node = lowered.graph_outputs.first().expect("real mnist model declares at least one output").1;
    let initializers: Vec<(&str, &[f32])> = lowered.initializers.iter().map(|(name, data)| (name.as_str(), data.as_slice())).collect();

    let images = load_normalized_images(&test_images_path(), TEST_IMAGES_COUNT);
    let labels = load_labels(&test_labels_path(), TEST_IMAGES_COUNT);
    assert_eq!(images.len(), labels.len(), "same number of images and labels");
    assert!(images.len() >= TEST_IMAGES_COUNT, "expected at least {TEST_IMAGES_COUNT} real test images, got {}", images.len());

    let start = Instant::now();
    let mut correct = 0_usize;
    let mut sample_rows: Vec<(usize, usize, u8)> = Vec::new();
    for (index, (image, &label)) in images.iter().zip(labels.iter()).enumerate() {
        let mut named = initializers.clone();
        named.push((graph_input_name.as_str(), image.as_slice()));
        let evaluated = proxima_tensor::cpu::evaluate_named(&lowered.program, &[], &named, &[output_node])
            .unwrap_or_else(|error| panic!("evaluate real mnist image {index}: {error}"));
        let (data, shape) = evaluated.get(output_node).expect("real mnist output present");
        assert_eq!(shape, &std::vec![1_u64, 10], "LogSoftmax over 10 MNIST classes");
        let predicted = argmax(data);
        if predicted == label as usize {
            correct += 1;
        }
        if index < 5 {
            sample_rows.push((index, predicted, label));
        }
    }
    let elapsed = start.elapsed();
    let accuracy = correct as f64 / images.len() as f64;

    eprintln!("real_mnist accuracy: {accuracy:.4} ({correct}/{} images) in {elapsed:?}", images.len());
    #[cfg(feature = "epilogue-fuse-diag")]
    {
        let (hits, elements, nanos) = proxima_tensor::cpu::epilogue_fuse_totals();
        eprintln!(
            "real_mnist epilogue_fuse: hits={hits} elements={elements} nanos={nanos} ns_per_element={:.4}",
            nanos as f64 / elements as f64
        );
    }
    for (index, predicted, label) in &sample_rows {
        eprintln!("real_mnist sample[{index}]: predicted={predicted} label={label}");
    }

    assert!(accuracy >= 0.95, "expected real mnist.onnx to classify at least 95% of {} real test images, got {accuracy:.4}", images.len());
}
