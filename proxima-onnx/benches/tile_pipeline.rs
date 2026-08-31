//! Tile-pipeline bench (`proxima-tensor/docs/discipline.md` ROW 155):
//! measures the line-buffer row-band streaming forward
//! (`benches/support/tile_pipeline.rs`) against the sealed
//! `cpu::evaluate_named` executor (`benches/mnist_f32_lane.rs`'s own
//! incumbent) on the real, on-disk `mnist.onnx` checkpoint and the real
//! MNIST `t10k` test images.
//!
//! Arms, each labeled `design-favors` per the disciplined-component gate:
//! - `incumbent_executor` (`design-favors: incumbent`): `cpu::evaluate_named`
//!   over the full lowered program, unmodified -- the SAME call
//!   `mnist_f32_lane.rs` measures, re-measured here on the SAME host/run so
//!   the comparison never crosses a session boundary.
//! - `tile_pipeline_band1` / `_band_kh` / `_band_2kh` (`design-favors:
//!   ours`): the pipeline at 3 row-band granularities, the task's own
//!   sweep axis.
//! - `single_layer_band_vs_materialized` (`design-favors: neutral`): one
//!   conv+relu layer alone, banded vs a single whole-layer materialize,
//!   isolating the between-op-traffic thesis from the other two layers'
//!   own cost.
//!
//! Whole forward IS the 100% frequency-weighted path (one call per real
//! image, exactly once) -- there is no warmer or colder path in this
//! model, so every arm below is reported at its true (100%) frequency; no
//! separate frequency-weighted scorecard table is needed the way a
//! multi-path component would require.
//!
//! Re-prove: `CARGO_TARGET_DIR=<scratch> cargo bench -p proxima-onnx
//! --bench tile_pipeline --features tile-pipeline-bench -- --save-baseline
//! <name>`

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use criterion::Criterion;

#[path = "support/tile_pipeline.rs"]
mod tile_pipeline;

/// Counting wrapper over the system allocator -- gate 8's "allocation
/// count per arm", measured, not assumed. Only consulted in the dedicated
/// pre-criterion pass below (`report_allocation_counts`); criterion's own
/// timed samples run with the SAME allocator underneath (it wraps, never
/// replaces, every allocation), so counting does not perturb the timing
/// arms above or below this struct.
struct CountingAllocator;

static ALLOCATION_COUNT: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATION_COUNT.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL_ALLOCATOR: CountingAllocator = CountingAllocator;

/// One-shot, non-timed allocation count for a single forward pass at each
/// band granularity -- printed once before the timed criterion arms run,
/// the same "non-timed diagnostic, timed bench kept clean" split
/// `mnist_diag.rs` already uses for its own instrument-gated counters.
fn report_allocation_counts(image: &[f32], weights: &tile_pipeline::MnistWeights<'_>) {
    for (label, band_rows) in [("band1", 1), ("band_kh", 3), ("band_2kh", 6)] {
        ALLOCATION_COUNT.store(0, Ordering::Relaxed);
        let logits = tile_pipeline::run_pipeline_forward(image, weights, tile_pipeline::BandRows(band_rows));
        std::hint::black_box(logits);
        let count = ALLOCATION_COUNT.load(Ordering::Relaxed);
        println!("tile_pipeline allocation count, single forward, {label}: {count} allocations");
    }
}

use tile_pipeline::{BandRows, BatchNormAffine, ConvReluStage, FcAccumulateStage, MnistWeights, RowBand, block_on_ready, run_pipeline_forward};
use proxima_primitives::pipe::Pipe;

/// Wall-clock share per named forward-pass stage -- gate 18 (a perf claim
/// needs a measurement artifact in the same breath): `run_pipeline_forward`
/// composes its stages through `AndThen`, which hides per-stage cost behind
/// one opaque `Future`. This driver reimplements the SAME sequential call
/// order `AndThen` performs (verified against `primitives.rs`'s own
/// `AndThen::call`: `first.call(input).await?; second.call(intermediate)`)
/// by hand, so each stage's own `Instant` window can be read independently.
/// Bench-only duplication (never library code), local to this file.
#[derive(Default, Clone, Copy)]
struct StageTimings {
    band_bookkeeping: Duration,
    conv1: Duration,
    conv2: Duration,
    conv3: Duration,
    fc1: Duration,
    fc2: Duration,
    softmax: Duration,
}

impl StageTimings {
    fn add(&mut self, other: &StageTimings) {
        self.band_bookkeeping += other.band_bookkeeping;
        self.conv1 += other.conv1;
        self.conv2 += other.conv2;
        self.conv3 += other.conv3;
        self.fc1 += other.fc1;
        self.fc2 += other.fc2;
        self.softmax += other.softmax;
    }

    fn total(&self) -> Duration {
        self.band_bookkeeping + self.conv1 + self.conv2 + self.conv3 + self.fc1 + self.fc2 + self.softmax
    }
}

/// One forward pass, timed stage-by-stage. Duplicates `matvec_bias` /
/// `apply_batch_norm` / `log_softmax` inline (all three are private to
/// `support/tile_pipeline.rs` and under 5 lines each) rather than widening
/// that module's visibility for a bench-only profiling pass.
fn run_pipeline_forward_profiled(image: &[f32], weights: &MnistWeights<'_>, band: BandRows) -> ([f32; 10], StageTimings) {
    let batch_norm1 = BatchNormAffine::new(weights.norm1_weight, weights.norm1_bias, weights.norm1_running_mean, weights.norm1_running_var, 1e-5);
    // ROW 170: per-call-site dot form selection, same as
    // `support::tile_pipeline::run_pipeline_forward` -- conv1 unblocked,
    // conv2/conv3 blocked, fc1 always unblocked.
    let stage1 = ConvReluStage::<false>::new(1, 8, 3, 3, 28, weights.conv1_weight, weights.conv1_bias, None);
    let stage2 = ConvReluStage::<true>::new(8, 16, 3, 3, 26, weights.conv2_weight, weights.conv2_bias, None);
    let stage3 = ConvReluStage::<true>::new(16, 24, 3, 3, 24, weights.conv3_weight, weights.conv3_bias, Some(batch_norm1));
    let fc_stage = FcAccumulateStage::new(24, 22, 22, 32, weights.fc1_weight, weights.fc1_bias);

    let mut timings = StageTimings::default();
    let mut row = 0;
    while row < 28 {
        let take = band.0.min(28 - row);

        let start = Instant::now();
        let data = image[row * 28..(row + take) * 28].to_vec();
        let input_band = RowBand { channels: 1, width: 28, rows: take, data };
        timings.band_bookkeeping += start.elapsed();

        let start = Instant::now();
        let out1 = block_on_ready(stage1.call(input_band)).expect("stage1 infallible");
        timings.conv1 += start.elapsed();

        let start = Instant::now();
        let out2 = block_on_ready(stage2.call(out1)).expect("stage2 infallible");
        timings.conv2 += start.elapsed();

        let start = Instant::now();
        let out3 = block_on_ready(stage3.call(out2)).expect("stage3 infallible");
        timings.conv3 += start.elapsed();

        let start = Instant::now();
        block_on_ready(fc_stage.call(out3)).expect("fc1 infallible");
        timings.fc1 += start.elapsed();

        row += take;
    }

    let start = Instant::now();
    let fc1_out = fc_stage.finalize();
    timings.fc1 += start.elapsed();

    let start = Instant::now();
    let fc2_out: Vec<f32> = (0..10)
        .map(|output_index| {
            let row = &weights.fc2_weight[output_index * 32..(output_index + 1) * 32];
            let dot: f32 = row.iter().zip(&fc1_out).fold(0.0_f32, |accumulator, (&weight_value, &input_value)| input_value.mul_add(weight_value, accumulator));
            dot + weights.fc2_bias[output_index]
        })
        .collect();
    timings.fc2 += start.elapsed();

    let start = Instant::now();
    let bn2_out: Vec<f32> = fc2_out
        .iter()
        .enumerate()
        .map(|(index, &value)| (value - weights.norm2_running_mean[index]) / (weights.norm2_running_var[index] + 1e-5_f32).sqrt() * weights.norm2_weight[index] + weights.norm2_bias[index])
        .collect();
    let max = bn2_out.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = bn2_out.iter().map(|&value| (value - max).exp()).sum();
    let log_sum = sum.ln();
    let mut logits = [0.0_f32; 10];
    for (destination, &value) in logits.iter_mut().zip(&bn2_out) {
        *destination = value - max - log_sum;
    }
    timings.softmax += start.elapsed();

    (logits, timings)
}

/// Per-stage µs, averaged over `iterations` real forward passes (rotating
/// through the real `t10k` images, not one image repeated) -- printed once,
/// non-timed relative to the criterion arms below (the SAME "diagnostic
/// pass, then clean timed arms" split `report_allocation_counts` uses).
fn report_stage_profile(images: &[Vec<f32>], weights: &MnistWeights<'_>, band: BandRows, label: &str, iterations: usize) {
    let mut total = StageTimings::default();
    for index in 0..iterations {
        let (logits, timings) = run_pipeline_forward_profiled(&images[index % images.len()], weights, band);
        std::hint::black_box(logits);
        total.add(&timings);
    }
    let scale = iterations as f64;
    let stage_us = |duration: Duration| duration.as_secs_f64() * 1_000_000.0 / scale;
    println!(
        "tile_pipeline stage profile [{label}], {iterations} forward passes: \
band_bookkeeping={:.3}us conv1={:.3}us conv2={:.3}us conv3={:.3}us fc1={:.3}us fc2={:.3}us softmax={:.3}us total={:.3}us",
        stage_us(total.band_bookkeeping),
        stage_us(total.conv1),
        stage_us(total.conv2),
        stage_us(total.conv3),
        stage_us(total.fc1),
        stage_us(total.fc2),
        stage_us(total.softmax),
        stage_us(total.total()),
    );
}

const MODEL_PATH: &str = "/Users/brianbruggeman/repos/others/burn/examples/onnx-inference/src/model/mnist.onnx";
const DATASET_DIR: &str = "/Users/brianbruggeman/.cache/burn-dataset/mnist";
const BENCH_IMAGES: usize = 50;

fn checkpoint_present() -> bool {
    Path::new(MODEL_PATH).exists()
}

fn test_images_path() -> PathBuf {
    Path::new(DATASET_DIR).join("test/t10k-images-idx3-ubyte")
}

fn dataset_present() -> bool {
    test_images_path().exists()
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

fn bench_whole_forward(criterion: &mut Criterion) {
    if !checkpoint_present() || !dataset_present() {
        eprintln!("tile_pipeline bench: skipping, no host-local mnist.onnx checkout or MNIST idx dataset");
        return;
    }

    let bytes = fs::read(MODEL_PATH).expect("read the real mnist.onnx checkpoint");
    let model = proxima_onnx::pipe::parse_complete(&bytes).expect("parse the real mnist.onnx checkpoint");
    let graph = model.graph.as_ref().expect("real mnist model has a graph");
    let lowered = proxima_onnx::lower::lower_graph(graph).expect("lower the real mnist.onnx graph to Op");
    let graph_input_name = lowered.graph_inputs.first().expect("input").clone();
    let output_node = lowered.graph_outputs.first().expect("output").1;
    let owned_initializers: Vec<(String, Vec<f32>)> = lowered.initializers.clone();
    let initializer_slices: Vec<(&str, &[f32])> = owned_initializers.iter().map(|(name, data)| (name.as_str(), data.as_slice())).collect();
    let weights = MnistWeights::from_initializers(&initializer_slices);

    let images = load_normalized_images(&test_images_path(), BENCH_IMAGES);

    report_allocation_counts(&images[0], &weights);
    for (label, band_rows) in [("band1", 1), ("band_kh", 3), ("band_2kh", 6)] {
        report_stage_profile(&images, &weights, BandRows(band_rows), label, 200);
    }

    let mut group = criterion.benchmark_group("mnist_forward_per_image");
    group.sample_size(20);

    // ONE image per criterion iteration, index-rotated -- matching
    // `mnist_f32_lane.rs`'s own sealed-bench convention exactly (not the
    // 50-images-per-iteration shape this arm used on its first attempt,
    // which measurably diverged from that sealed bench's own same-session
    // cross-check number; see ROW 155's own honest note on this).
    let incumbent_index = std::cell::Cell::new(0usize);
    group.bench_function("incumbent_executor", |bencher| {
        bencher.iter(|| {
            let current = incumbent_index.get();
            incumbent_index.set((current + 1) % images.len());
            let mut named = initializer_slices.clone();
            named.push((graph_input_name.as_str(), images[current].as_slice()));
            let evaluated = proxima_tensor::cpu::evaluate_named(&lowered.program, &[], &named, &[output_node]).expect("evaluate");
            let (data, _shape) = evaluated.get(output_node).expect("output present");
            std::hint::black_box(data);
        });
    });

    for (label, band_rows) in [("tile_pipeline_band1", 1), ("tile_pipeline_band_kh", 3), ("tile_pipeline_band_2kh", 6)] {
        let pipeline_index = std::cell::Cell::new(0usize);
        group.bench_function(label, |bencher| {
            bencher.iter(|| {
                let current = pipeline_index.get();
                pipeline_index.set((current + 1) % images.len());
                let logits = run_pipeline_forward(&images[current], &weights, BandRows(band_rows));
                std::hint::black_box(logits);
            });
        });
    }

    group.finish();

    let mut neutral_group = criterion.benchmark_group("single_conv_layer_band_vs_materialized");
    neutral_group.sample_size(20);
    bench_single_layer_neutral_arm(&mut neutral_group, &weights, &images);
    neutral_group.finish();
}

/// `design-favors: neutral` -- isolates ONE conv+relu layer's own
/// between-op-traffic cost from the other two layers': the banded FSM
/// stage (`ConvReluStage`, `kh`-row band) vs the SAME arithmetic run as a
/// single whole-layer call (one call, `28` rows all at once -- forces the
/// stage to hold its own full input in the ring rather than stream it,
/// the closest same-code-path proxy for "materialize the whole layer" this
/// module can express without a second kernel).
fn bench_single_layer_neutral_arm(group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>, weights: &MnistWeights<'_>, images: &[Vec<f32>]) {
    group.bench_function("layer1_banded_kh_rows", |bencher| {
        bencher.iter(|| {
            for image in images {
                let stage = ConvReluStage::<false>::new(1, 8, 3, 3, 28, weights.conv1_weight, weights.conv1_bias, None);
                let mut row = 0;
                while row < 28 {
                    let take = 3.min(28 - row);
                    let data = image[row * 28..(row + take) * 28].to_vec();
                    let band = tile_pipeline::RowBand { channels: 1, width: 28, rows: take, data };
                    let out = block_on_ready(stage.call(band)).expect("infallible");
                    std::hint::black_box(out);
                    row += take;
                }
            }
        });
    });

    group.bench_function("layer1_single_whole_layer_call", |bencher| {
        bencher.iter(|| {
            for image in images {
                let stage = ConvReluStage::<false>::new(1, 8, 3, 3, 28, weights.conv1_weight, weights.conv1_bias, None);
                let band = tile_pipeline::RowBand { channels: 1, width: 28, rows: 28, data: image.clone() };
                let out = block_on_ready(stage.call(band)).expect("infallible");
                std::hint::black_box(out);
            }
        });
    });
}

fn main() {
    let mut criterion = Criterion::default();
    bench_whole_forward(&mut criterion);
    criterion.final_summary();
}
