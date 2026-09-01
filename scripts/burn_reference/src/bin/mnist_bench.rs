//! ROW 187 burn arm: burn's own `onnx-inference` mnist example, adapted from
//! `~/repos/others/burn/examples/onnx-inference/src/bin/mnist_inference.rs`,
//! run against the SAME `mnist.onnx` file and the SAME t10k idx3/idx1 fixture
//! `proxima-onnx/benches/mnist_f32_lane.rs` uses (`MODEL_PATH`/`DATASET_DIR`
//! below are copied verbatim from that file). The idx3/idx1 parse and
//! normalization formula are duplicated rather than shared, same reason
//! `mnist_f32_lane.rs` gives: a standalone binary here cannot depend on a
//! sibling crate's test helpers.
//!
//! `NdArray<f32>` backend, `burn` built with `default-features = false,
//! features = ["std", "ndarray", "dataset", "vision"]` (see Cargo.toml) --
//! this disables burn-ndarray's own `default` feature set
//! (`["std", "simd", "multi-threads"]`), so this arm is single-threaded, no
//! rayon, no macerator SIMD. That is a deliberate, recorded choice, not the
//! upstream default: `cargo run --bin mnist_inference` in burn's own
//! `examples/onnx-inference` crate (which does not disable burn's own
//! default features) would pull in rayon + SIMD.

use std::env::args;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use burn::backend::ndarray::NdArray;
use burn::tensor::Tensor;

use burn_reference::mnist::Model;

const MODEL_PATH: &str = "/Users/brianbruggeman/repos/others/burn/examples/onnx-inference/src/model/mnist.onnx";
const DATASET_DIR: &str = "/Users/brianbruggeman/.cache/burn-dataset/mnist";
const DEFAULT_IMAGE_COUNT: usize = 1000;

fn test_images_path() -> PathBuf {
    Path::new(DATASET_DIR).join("test/t10k-images-idx3-ubyte")
}

fn test_labels_path() -> PathBuf {
    Path::new(DATASET_DIR).join("test/t10k-labels-idx1-ubyte")
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

fn load_raw_images(path: &Path, limit: usize) -> Vec<Vec<u8>> {
    let bytes = fs::read(path).expect("read idx3 image file");
    let (item_count, extents) = idx_header(&bytes);
    let pixel_count = extents.iter().product::<usize>();
    let take = item_count.min(limit);
    let header_length = 4 + extents.len() * 4 + 4;
    (0..take)
        .map(|image_index| {
            let start = header_length + image_index * pixel_count;
            bytes[start..start + pixel_count].to_vec()
        })
        .collect()
}

fn load_labels(path: &Path, limit: usize) -> Vec<u8> {
    let bytes = fs::read(path).expect("read idx1 label file");
    let (item_count, _extents) = idx_header(&bytes);
    let take = item_count.min(limit);
    bytes[8..8 + take].to_vec()
}

fn percentile(sorted_ns: &[u64], fraction: f64) -> u64 {
    let index = ((sorted_ns.len() as f64 - 1.0) * fraction).round() as usize;
    sorted_ns[index]
}

fn mean_and_cov(samples: &[f64]) -> (f64, f64) {
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let variance = samples.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / samples.len() as f64;
    (mean, variance.sqrt() / mean * 100.0)
}

fn main() {
    type Backend = NdArray<f32>;

    let image_count = args().nth(1).and_then(|value| value.parse::<usize>().ok()).unwrap_or(DEFAULT_IMAGE_COUNT);
    let round_count = args().nth(2).and_then(|value| value.parse::<usize>().ok()).unwrap_or(5);

    assert!(Path::new(MODEL_PATH).exists(), "expected mnist.onnx at {MODEL_PATH}");
    assert!(test_images_path().exists() && test_labels_path().exists(), "expected t10k idx dataset under {DATASET_DIR}");

    let device = <Backend as burn::tensor::backend::Backend>::Device::default();
    let model: Model<Backend> = Model::default();

    let raw_images = load_raw_images(&test_images_path(), image_count);
    let labels = load_labels(&test_labels_path(), image_count);
    assert_eq!(raw_images.len(), labels.len(), "same number of images and labels");
    assert!(raw_images.len() >= image_count.min(10_000), "expected {image_count} real t10k test images, got {}", raw_images.len());

    // warm-up pass, discarded, mirrors mnist_f32_lane.rs's own warm-up.
    {
        let image_data: Vec<f32> = raw_images[0].iter().map(|&pixel| pixel as f32).collect();
        let input = Tensor::<Backend, 1>::from_floats(image_data.as_slice(), &device).reshape([1, 1, 28, 28]);
        let input = ((input / 255) - 0.1307) / 0.3081;
        let _ = model.forward(input);
    }

    // accuracy pass, once, outside the timed rounds -- same gate
    // `mnist_f32_lane.rs` runs, same normalization formula burn's own
    // `mnist_inference.rs` uses verbatim.
    let mut correct = 0usize;
    for (raw_image, &label) in raw_images.iter().zip(labels.iter()) {
        let image_data: Vec<f32> = raw_image.iter().map(|&pixel| pixel as f32).collect();
        let input = Tensor::<Backend, 1>::from_floats(image_data.as_slice(), &device).reshape([1, 1, 28, 28]);
        let input = ((input / 255) - 0.1307) / 0.3081;
        let output = model.forward(input);
        let predicted = output.argmax(1).into_scalar() as u8;
        if predicted == label {
            correct += 1;
        }
    }
    let accuracy = correct as f64 / raw_images.len() as f64;
    eprintln!("burn_reference: accuracy={accuracy:.4} ({correct}/{})", raw_images.len());

    // per-round latency sweep: one full pass over the image set per round.
    let mut round_mean_ms = Vec::with_capacity(round_count);
    let mut round_p50_ms = Vec::with_capacity(round_count);
    let mut round_p95_ms = Vec::with_capacity(round_count);
    for round in 0..round_count {
        let mut per_image_ns: Vec<u64> = Vec::with_capacity(raw_images.len());
        for raw_image in &raw_images {
            let image_data: Vec<f32> = raw_image.iter().map(|&pixel| pixel as f32).collect();
            let start = Instant::now();
            let input = Tensor::<Backend, 1>::from_floats(image_data.as_slice(), &device).reshape([1, 1, 28, 28]);
            let input = ((input / 255) - 0.1307) / 0.3081;
            let output = model.forward(input);
            let _ = std::hint::black_box(output.into_data());
            per_image_ns.push(start.elapsed().as_nanos() as u64);
        }
        per_image_ns.sort_unstable();
        let mean_ns = per_image_ns.iter().sum::<u64>() as f64 / per_image_ns.len() as f64;
        let p50_ns = percentile(&per_image_ns, 0.50);
        let p95_ns = percentile(&per_image_ns, 0.95);
        eprintln!(
            "burn_reference: round={round} images={} mean={:.4}ms p50={:.4}ms p95={:.4}ms",
            per_image_ns.len(),
            mean_ns / 1e6,
            p50_ns as f64 / 1e6,
            p95_ns as f64 / 1e6,
        );
        round_mean_ms.push(mean_ns / 1e6);
        round_p50_ms.push(p50_ns as f64 / 1e6);
        round_p95_ms.push(p95_ns as f64 / 1e6);
    }

    let (mean_of_means, cov_of_means) = mean_and_cov(&round_mean_ms);
    let (mean_of_p50, cov_of_p50) = mean_and_cov(&round_p50_ms);
    let (mean_of_p95, cov_of_p95) = mean_and_cov(&round_p95_ms);
    eprintln!(
        "burn_reference: SUMMARY rounds={round_count} images/round={} mean={:.4}ms (CoV={cov_of_means:.2}%) p50={:.4}ms (CoV={cov_of_p50:.2}%) p95={:.4}ms (CoV={cov_of_p95:.2}%) accuracy={accuracy:.4} round_means={round_mean_ms:?}",
        raw_images.len(),
        mean_of_means,
        mean_of_p50,
        mean_of_p95,
    );
}
