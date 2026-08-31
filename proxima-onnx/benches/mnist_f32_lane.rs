//! Sealed incumbent baseline for the mnist f32 inference lane (proxima
//! discipline-log campaign, `proxima-tensor/docs/discipline.md`). Measures
//! per-image `cpu::evaluate_named` latency over the real, on-disk
//! `mnist.onnx` checkpoint against the real MNIST `t10k` test split -- the
//! same fixtures `proxima-onnx/tests/real_mnist_accuracy.rs` uses, loaded
//! here a second time rather than shared, since a bench crate cannot depend
//! on a sibling test binary's helpers.
//!
//! `mnist-f32-bench`, default-off: presence-guarded on
//! `~/repos/others/burn/examples/onnx-inference/src/model/mnist.onnx` +
//! `~/.cache/burn-dataset/mnist`, clean skip (prints and returns) when
//! either is absent. Deliberately does NOT enable `proxima-tensor/instrument`
//! -- that feature adds ~30-40% overhead to every `run_reduce` call
//! (measured this session), and this bench's whole point is the clean,
//! unperturbed number. The instrumented per-node breakdown and the exact
//! analytic MAC count live in `examples/mnist_diag.rs` (`mnist-diag`
//! feature) instead, run once, non-timed, as a companion.
//!
//! Re-prove with (host must be quiet; see the discipline log row this bench
//! seeds for the loadout it was actually measured under):
//! `CARGO_TARGET_DIR=<scratch> cargo bench -p proxima-onnx --bench mnist_f32_lane --features mnist-f32-bench -- --save-baseline mnist-f32-lane`

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::cell::Cell;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use criterion::Criterion;

const MODEL_PATH: &str = "/Users/brianbruggeman/repos/others/burn/examples/onnx-inference/src/model/mnist.onnx";
const DATASET_DIR: &str = "/Users/brianbruggeman/.cache/burn-dataset/mnist";
const TEST_IMAGES_COUNT: usize = 1000;
// measured this session (see discipline.md): 990/1000 = 0.9900 exactly.
// asserted with margin below the exact ratio so an f64 rounding wobble on a
// different host can never fail this gate for a reason unrelated to
// correctness; a real accuracy regression (wrong function, not just slow)
// still trips it well before 0.989.
const MIN_ACCURACY: f64 = 0.989;

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

/// Verbatim of `real_mnist_accuracy.rs`'s own idx3/idx1 header parse -- see
/// that file's doc for the format citation. Duplicated rather than shared:
/// a criterion bench binary and a `#[test]` binary do not share a crate.
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

/// Register-blocked FMA accumulator chain (8 independent lanes so the
/// latency of one `f32` FMA never becomes the loop's own bottleneck) --
/// the roofline denominator: single-performance-core sustained MAC/s on
/// THIS host, measured fresh in THIS bench rather than only cited from a
/// prior session's log row (`docs/discipline.md` ROW 20 measured 97 GFLOPS
/// achievable pure-register FMA on this same M1 Max; this is the
/// re-provable, in-artifact twin of that number). One fma = one MAC = two
/// FLOPs.
fn fma_roofline_macs_per_sec() -> f64 {
    const LANES: usize = 8;
    const ITERS: u64 = 200_000_000;
    let mut lane = [1.0000001f32; LANES];
    let multiplier = [1.0000002f32; LANES];
    let start = Instant::now();
    for _ in 0..ITERS {
        for index in 0..LANES {
            lane[index] = lane[index].mul_add(multiplier[index], lane[index]);
        }
    }
    let elapsed = start.elapsed();
    // prevents the whole loop being optimized away as dead code; the
    // checksum is never asserted against, only forced live.
    std::hint::black_box(lane);
    (ITERS * LANES as u64) as f64 / elapsed.as_secs_f64()
}

fn main() {
    if !checkpoint_present() {
        eprintln!("mnist_f32_lane: skipping, no host-local mnist.onnx checkout at {MODEL_PATH}");
        return;
    }
    if !dataset_present() {
        eprintln!("mnist_f32_lane: skipping, no host-local MNIST idx dataset under {DATASET_DIR}");
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

    // the accuracy gate: this bench can never speed up by computing the
    // wrong function. Evaluated once, outside criterion's own timed loop.
    let mut correct = 0usize;
    for (image, &label) in images.iter().zip(labels.iter()) {
        let mut named = initializers.clone();
        named.push((graph_input_name.as_str(), image.as_slice()));
        let evaluated = proxima_tensor::cpu::evaluate_named(&lowered.program, &[], &named, &[output_node]).expect("evaluate real mnist image");
        let (data, _shape) = evaluated.get(output_node).expect("real mnist output present");
        if argmax(data) == label as usize {
            correct += 1;
        }
    }
    let accuracy = correct as f64 / images.len() as f64;
    eprintln!("mnist_f32_lane: accuracy={accuracy:.4} ({correct}/{})", images.len());
    assert!(accuracy >= MIN_ACCURACY, "expected >= {MIN_ACCURACY} accuracy on {} real mnist test images, got {accuracy:.4}", images.len());

    // roofline: measured fresh, mean of 3, CoV reported (bench-metrics
    // discipline -- never a point estimate above 5% CoV without the range).
    let roofline_samples: Vec<f64> = (0..3).map(|_| fma_roofline_macs_per_sec()).collect();
    let roofline_mean = roofline_samples.iter().sum::<f64>() / roofline_samples.len() as f64;
    let roofline_variance = roofline_samples.iter().map(|value| (value - roofline_mean).powi(2)).sum::<f64>() / roofline_samples.len() as f64;
    let roofline_cov = roofline_variance.sqrt() / roofline_mean * 100.0;
    eprintln!(
        "mnist_f32_lane: single-core FMA roofline = {:.2} GMAC/s ({:.2} GFLOP/s), CoV={roofline_cov:.2}% over {} runs, samples={roofline_samples:?}",
        roofline_mean / 1e9,
        roofline_mean * 2.0 / 1e9,
        roofline_samples.len()
    );

    // manual latency sweep for percentiles (bench-metrics discipline: p50/
    // p95, not just criterion's mean+CI) over one full pass of the 1000
    // real images, one warm-up pass first.
    let mut named = initializers.clone();
    named.push((graph_input_name.as_str(), images[0].as_slice()));
    let _ = proxima_tensor::cpu::evaluate_named(&lowered.program, &[], &named, &[output_node]).expect("warm-up eval");

    let mut per_image_ns: Vec<u64> = Vec::with_capacity(images.len());
    for image in &images {
        let mut named = initializers.clone();
        named.push((graph_input_name.as_str(), image.as_slice()));
        let start = Instant::now();
        let evaluated = proxima_tensor::cpu::evaluate_named(&lowered.program, &[], &named, &[output_node]).expect("evaluate");
        std::hint::black_box(&evaluated);
        per_image_ns.push(start.elapsed().as_nanos() as u64);
    }
    per_image_ns.sort_unstable();
    let sweep_mean_ns = per_image_ns.iter().sum::<u64>() as f64 / per_image_ns.len() as f64;
    let sweep_p50_ns = per_image_ns[per_image_ns.len() / 2];
    let sweep_p95_ns = per_image_ns[(per_image_ns.len() * 95) / 100];
    let sweep_variance = per_image_ns.iter().map(|&value| (value as f64 - sweep_mean_ns).powi(2)).sum::<f64>() / per_image_ns.len() as f64;
    let sweep_cov = sweep_variance.sqrt() / sweep_mean_ns * 100.0;
    eprintln!(
        "mnist_f32_lane: manual sweep over {} real images: mean={:.3}ms p50={:.3}ms p95={:.3}ms CoV={sweep_cov:.2}%",
        per_image_ns.len(),
        sweep_mean_ns / 1e6,
        sweep_p50_ns as f64 / 1e6,
        sweep_p95_ns as f64 / 1e6,
    );

    let mut criterion = Criterion::default();
    let mut group = criterion.benchmark_group("mnist_f32_lane");
    group.sample_size(30);
    let index = Cell::new(0usize);
    group.bench_function("evaluate_named_per_image", |bencher| {
        bencher.iter(|| {
            let current = index.get();
            index.set((current + 1) % images.len());
            let mut named = initializers.clone();
            named.push((graph_input_name.as_str(), images[current].as_slice()));
            proxima_tensor::cpu::evaluate_named(&lowered.program, &[], &named, &[output_node]).expect("evaluate")
        });
    });
    group.finish();
    criterion.final_summary();
}
