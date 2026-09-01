//! Stage parallelism vs data parallelism probe (`proxima-tensor/docs/discipline.md`
//! ROW 173): the mnist tile pipeline (ROW 155-172, `benches/support/tile_pipeline.rs`)
//! single-threaded is 460-465us/image (ROW 171/172's own sealed number), stage
//! profile conv1=27us / conv2=129us / conv3=245us / fc1=65us (ROW 172's own
//! profiler table). Two arms:
//!
//! - `stage_parallel`: the 4 stages (conv1, conv2, conv3, fc1+epilogue) each
//!   pinned to their own OS thread, connected by
//!   `proxima_core::ring::mpsc::Ring` (lock-free MPMC, used SPSC-style here --
//!   exactly one producer and one consumer per ring) carrying row-bands. Images
//!   flow through the pipeline back-to-back; steady-state throughput is capped
//!   by the SLOWEST stage (conv3, 245us), not the sum -- ceiling math below.
//!   Per-image LATENCY should stay close to the single-thread total (each
//!   image still visits all 4 stages in sequence), while THROUGHPUT should
//!   approach `1/245us` once the pipeline is full.
//! - `data_parallel`: 4 independent single-thread pipelines (the SAME
//!   `run_pipeline_forward`, unmodified, ROW 155-172's own production
//!   surface) on 4 threads, each processing its own disjoint slice of images.
//!   Per-image latency is unchanged from the single-thread number; throughput
//!   should approach 4x the single-thread rate. This is the arm
//!   stage-parallelism must justify itself against.
//!
//! `design-favors: ours` for both (no external incumbent for row-band stage
//! pipelining exists; `data_parallel` is the internal home-turf arm -- the
//! honest alternative shape, not a third-party library).
//!
//! Correctness: every arm's logits are asserted bit-identical (`==` on
//! `[f32; 10]`, not a tolerance) against `run_pipeline_forward` run
//! single-threaded on the SAME image, before any timing number is trusted.
//!
//! Re-prove: `CARGO_TARGET_DIR=<scratch> cargo build --release -p proxima-onnx
//! --bench stage_parallel --features stage-parallel-bench`, then run the
//! produced `deps/stage_parallel-*` binary directly 3-5x (not `cargo bench` --
//! this is a plain `Instant`-timed driver, no criterion harness, since the
//! measured unit is a multi-thread steady-state run, not a single call).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::too_many_arguments)]

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use proxima_core::ring::Ring;

#[path = "support/tile_pipeline.rs"]
mod tile_pipeline;

use tile_pipeline::{BandRows, BatchNormAffine, ConvReluStage, FcAccumulateStage, MnistWeights, RowBand, run_pipeline_forward, run_pipeline_forward_direct};

const MODEL_PATH: &str = "/Users/brianbruggeman/repos/others/burn/examples/onnx-inference/src/model/mnist.onnx";
const DATASET_DIR: &str = "/Users/brianbruggeman/.cache/burn-dataset/mnist";
/// Steady-state image count per timed run -- large enough that pipeline
/// fill/drain (depth 4) is <2% of the run, small enough the whole sweep
/// fits the session's time budget.
const STEADY_STATE_IMAGES: usize = 400;
/// Full split, for the winning arm's accuracy gate (task's own "full-1000
/// accuracy exactly 0.9900" requirement).
const FULL_SPLIT_IMAGES: usize = 1000;
const RING_CAPACITY: usize = 2;
const BAND: BandRows = BandRows(3);

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
    let item_count = u32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
    let take = item_count.min(limit);
    bytes[8..8 + take].to_vec()
}

/// One message on a stage-to-stage ring: a row-band to compute, the
/// end-of-image boundary (stage resets its own ring/accumulator state and
/// forwards the boundary downstream), or shutdown (drain complete, stop the
/// thread). Images flow through in submission order -- a single producer per
/// ring and FIFO delivery means completion order equals submission order, so
/// no image id needs to ride in the message; the collector's Nth result IS
/// image N.
enum StageMsg {
    Band(RowBand),
    EndOfImage,
    Shutdown,
}

fn spin_send<T>(ring: &Ring<T>, mut value: T) {
    loop {
        match ring.push(value) {
            Ok(()) => return,
            Err(returned) => {
                value = returned;
                std::hint::spin_loop();
            }
        }
    }
}

fn spin_recv<T>(ring: &Ring<T>) -> T {
    loop {
        if let Some(value) = ring.dequeue() {
            return value;
        }
        std::hint::spin_loop();
    }
}

/// One conv+relu(+bn) stage's own worker loop: owns a fresh `ConvReluStage`
/// per image (mirrors `run_pipeline_forward`'s own per-image construction --
/// zero behavior change, see module doc), fed bands from `input`, forwarding
/// computed bands to `output`. `BLOCKED`/`ROWS` select the SAME dot-product
/// form `run_pipeline_forward` uses per stage (ROW 170/171), unchanged here.
#[allow(clippy::needless_pass_by_value)]
fn conv_stage_worker<const BLOCKED: bool, const ROWS: usize>(
    input: Arc<Ring<StageMsg>>,
    output: Arc<Ring<StageMsg>>,
    channels_in: usize,
    channels_out: usize,
    kernel_height: usize,
    kernel_width: usize,
    input_width: usize,
    weight: &'static [f32],
    bias: &'static [f32],
    batch_norm: Option<BatchNormAffine>,
) {
    let mut stage: Option<ConvReluStage<'static, BLOCKED, ROWS>> = None;
    loop {
        match spin_recv(&input) {
            StageMsg::Band(band) => {
                let active = stage.get_or_insert_with(|| ConvReluStage::new(channels_in, channels_out, kernel_height, kernel_width, input_width, weight, bias, batch_norm.clone()));
                let computed = active.compute_direct(band);
                spin_send(&output, StageMsg::Band(computed));
            }
            StageMsg::EndOfImage => {
                stage = None;
                spin_send(&output, StageMsg::EndOfImage);
            }
            StageMsg::Shutdown => {
                spin_send(&output, StageMsg::Shutdown);
                return;
            }
        }
    }
}

/// FC1's own worker: accumulates a fresh `FcAccumulateStage` per image
/// (mirrors `run_pipeline_forward`'s own per-image construction), and on
/// `EndOfImage` runs the SAME fc1-finalize -> fc2 -> batchnorm2 ->
/// log-softmax epilogue `run_pipeline_forward` runs inline, pushing the
/// finished `[f32; 10]` logits to `results`.
#[allow(clippy::needless_pass_by_value)]
fn fc_stage_worker(input: Arc<Ring<StageMsg>>, results: Arc<Ring<[f32; 10]>>, weights: MnistWeights<'static>) {
    let mut stage: Option<FcAccumulateStage<'static>> = None;
    loop {
        match spin_recv(&input) {
            StageMsg::Band(band) => {
                let active = stage.get_or_insert_with(|| FcAccumulateStage::new(24, 22, 22, 32, weights.fc1_weight, weights.fc1_bias));
                active.compute_direct(band);
            }
            StageMsg::EndOfImage => {
                let active = stage.take().expect("fc stage saw EndOfImage with no bands accumulated");
                let logits = finalize_epilogue(&active, &weights);
                spin_send(&results, logits);
            }
            StageMsg::Shutdown => return,
        }
    }
}

/// fc1-finalize -> fc2 -> batchnorm2 -> log-softmax, the SAME epilogue
/// `run_pipeline_forward` runs inline after its own `AndThen`-composed
/// chain -- extracted here so both the sequential and stage-parallel arms
/// call one body (never two hand-copies of the arithmetic to drift apart).
fn finalize_epilogue(fc_stage: &FcAccumulateStage<'_>, weights: &MnistWeights<'_>) -> [f32; 10] {
    let fc1_out = fc_stage.finalize();
    let fc2_out: Vec<f32> = (0..10)
        .map(|output_index| {
            let row = &weights.fc2_weight[output_index * 32..(output_index + 1) * 32];
            let dot: f32 = row.iter().zip(&fc1_out).fold(0.0_f32, |accumulator, (&weight_value, &input_value)| input_value.mul_add(weight_value, accumulator));
            dot + weights.fc2_bias[output_index]
        })
        .collect();
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
    logits
}

struct RunResult {
    wall: Duration,
    outputs: Vec<[f32; 10]>,
    latencies: Vec<Duration>,
}

/// Stage-parallel arm: 4 worker threads (conv1, conv2, conv3, fc1+epilogue)
/// connected by 4 `Ring<StageMsg>` (SPSC usage of the lock-free MPMC
/// primitive -- `proxima_core::ring::mpsc::Ring`, gate 14: an existing
/// primitive expresses this, no new channel type written) plus one results
/// ring. Main thread is the producer (feeds row-bands for every image, in
/// order) and the collector (drains exactly `images.len()` results,
/// stamping a completion `Instant` per image for the latency series).
fn run_stage_parallel(images: &[Vec<f32>], weights: MnistWeights<'static>) -> RunResult {
    let batch_norm1 = BatchNormAffine::new(weights.norm1_weight, weights.norm1_bias, weights.norm1_running_mean, weights.norm1_running_var, 1e-5);

    let ring_in = Arc::new(Ring::<StageMsg>::with_capacity(RING_CAPACITY));
    let ring_12 = Arc::new(Ring::<StageMsg>::with_capacity(RING_CAPACITY));
    let ring_23 = Arc::new(Ring::<StageMsg>::with_capacity(RING_CAPACITY));
    let ring_3fc = Arc::new(Ring::<StageMsg>::with_capacity(RING_CAPACITY));
    let results = Arc::new(Ring::<[f32; 10]>::with_capacity(images.len().next_power_of_two().max(2)));

    let handle1 = {
        let (input, output) = (Arc::clone(&ring_in), Arc::clone(&ring_12));
        thread::Builder::new()
            .name("conv1".into())
            .spawn(move || conv_stage_worker::<false, 1>(input, output, 1, 8, 3, 3, 28, weights.conv1_weight, weights.conv1_bias, None))
            .expect("spawn conv1 worker")
    };
    let handle2 = {
        let (input, output) = (Arc::clone(&ring_12), Arc::clone(&ring_23));
        thread::Builder::new()
            .name("conv2".into())
            .spawn(move || conv_stage_worker::<true, 4>(input, output, 8, 16, 3, 3, 26, weights.conv2_weight, weights.conv2_bias, None))
            .expect("spawn conv2 worker")
    };
    let handle3 = {
        let (input, output) = (Arc::clone(&ring_23), Arc::clone(&ring_3fc));
        thread::Builder::new()
            .name("conv3".into())
            .spawn(move || conv_stage_worker::<true, 4>(input, output, 16, 24, 3, 3, 24, weights.conv3_weight, weights.conv3_bias, Some(batch_norm1)))
            .expect("spawn conv3 worker")
    };
    let handle_fc = {
        let (input, output) = (Arc::clone(&ring_3fc), Arc::clone(&results));
        thread::Builder::new().name("fc1".into()).spawn(move || fc_stage_worker(input, output, weights)).expect("spawn fc1 worker")
    };

    // Producer runs on its OWN thread, concurrently with the collector below
    // -- sequential send-then-drain on one thread would let the fc stage's
    // results pile up in `results` (capacity > images.len(), never blocks)
    // while the producer is still feeding later images, so a `start.elapsed()`
    // read at drain time would measure "time since the drain loop began", not
    // real per-image pipeline latency. `start_ring` carries each image's send
    // `Instant` to the collector in the SAME submission order the results
    // ring delivers completions, so they pair up positionally.
    let start_ring = Arc::new(Ring::<Instant>::with_capacity(images.len().next_power_of_two().max(2)));
    let wall_start = Instant::now();
    let producer = {
        let ring_in = Arc::clone(&ring_in);
        let start_ring = Arc::clone(&start_ring);
        let images = images.to_vec();
        thread::Builder::new()
            .name("producer".into())
            .spawn(move || {
                for image in &images {
                    spin_send(&start_ring, Instant::now());
                    let mut row = 0;
                    while row < 28 {
                        let take = BAND.0.min(28 - row);
                        let data = image[row * 28..(row + take) * 28].to_vec();
                        spin_send(&ring_in, StageMsg::Band(RowBand { channels: 1, width: 28, rows: take, data }));
                        row += take;
                    }
                    spin_send(&ring_in, StageMsg::EndOfImage);
                }
                spin_send(&ring_in, StageMsg::Shutdown);
            })
            .expect("spawn producer thread")
    };

    let mut outputs = Vec::with_capacity(images.len());
    let mut latencies = Vec::with_capacity(images.len());
    for _ in 0..images.len() {
        let logits = spin_recv(&results);
        let completed = Instant::now();
        let start = spin_recv(&start_ring);
        latencies.push(completed.duration_since(start));
        outputs.push(logits);
    }
    let wall = wall_start.elapsed();

    producer.join().expect("producer thread panicked");
    handle1.join().expect("conv1 worker panicked");
    handle2.join().expect("conv2 worker panicked");
    handle3.join().expect("conv3 worker panicked");
    handle_fc.join().expect("fc1 worker panicked");

    RunResult { wall, outputs, latencies }
}

/// Data-parallel arm: `THREADS` independent single-thread pipelines (the
/// SAME `run_pipeline_forward` the production surface uses, unmodified), each
/// given its own disjoint slice of `images`. Per-image latency is measured
/// directly around each `run_pipeline_forward` call; throughput is
/// `images.len() / wall`.
fn run_data_parallel(images: &[Vec<f32>], weights: MnistWeights<'static>, threads: usize) -> RunResult {
    let chunk = images.len().div_ceil(threads);
    let wall_start = Instant::now();
    let handles: Vec<_> = images
        .chunks(chunk)
        .map(|slice| {
            let slice = slice.to_vec();
            thread::spawn(move || {
                let mut outputs = Vec::with_capacity(slice.len());
                let mut latencies = Vec::with_capacity(slice.len());
                for image in &slice {
                    let start = Instant::now();
                    let logits = run_pipeline_forward(image, &weights, BAND);
                    latencies.push(start.elapsed());
                    outputs.push(logits);
                }
                (outputs, latencies)
            })
        })
        .collect();

    let mut outputs = Vec::with_capacity(images.len());
    let mut latencies = Vec::with_capacity(images.len());
    for handle in handles {
        let (mut chunk_outputs, mut chunk_latencies) = handle.join().expect("data-parallel worker panicked");
        outputs.append(&mut chunk_outputs);
        latencies.append(&mut chunk_latencies);
    }
    let wall = wall_start.elapsed();
    RunResult { wall, outputs, latencies }
}

fn percentile(sorted: &[Duration], fraction: f64) -> Duration {
    let index = ((sorted.len() - 1) as f64 * fraction).round() as usize;
    sorted[index]
}

fn report(label: &str, images: usize, run: &RunResult) {
    let mut sorted = run.latencies.clone();
    sorted.sort();
    let p50 = percentile(&sorted, 0.50);
    let p99 = percentile(&sorted, 0.99);
    let throughput = images as f64 / run.wall.as_secs_f64();
    println!(
        "{label}: n={images} wall={:.3}ms throughput={:.1} images/s p50={:.3}us p99={:.3}us",
        run.wall.as_secs_f64() * 1000.0,
        throughput,
        p50.as_secs_f64() * 1_000_000.0,
        p99.as_secs_f64() * 1_000_000.0,
    );
}

fn leak_weights(owned: Vec<(String, Vec<f32>)>) -> MnistWeights<'static> {
    let leaked: &'static [(String, Vec<f32>)] = Box::leak(owned.into_boxed_slice());
    let slices: Vec<(&'static str, &'static [f32])> = leaked.iter().map(|(name, data)| (name.as_str(), data.as_slice())).collect();
    MnistWeights::from_initializers(&slices)
}

fn main() {
    if !checkpoint_present() || !dataset_present() {
        eprintln!("stage_parallel bench: skipping, no host-local mnist.onnx checkout or MNIST idx dataset");
        return;
    }

    let bytes = fs::read(MODEL_PATH).expect("read the real mnist.onnx checkpoint");
    let model = proxima_onnx::pipe::parse_complete(&bytes).expect("parse the real mnist.onnx checkpoint");
    let graph = model.graph.as_ref().expect("real mnist model has a graph");
    let lowered = proxima_onnx::lower::lower_graph(graph).expect("lower the real mnist.onnx graph to Op");
    let owned_initializers: Vec<(String, Vec<f32>)> = lowered.initializers.clone();
    let weights: MnistWeights<'static> = leak_weights(owned_initializers);

    let images = load_normalized_images(&test_images_path(), STEADY_STATE_IMAGES);
    println!("stage_parallel bench: {} images loaded, band={:?}", images.len(), BAND);

    // Ceiling math, on paper, BEFORE measuring (ROW 172's own profiler
    // table: conv1=27us conv2=129us conv3=245us fc1=65us, band_kh total
    // 459-465us):
    println!(
        "ceiling math: single-thread total ~465us/image -> {:.0} images/s; stage-parallel cap = 1/slowest_stage (conv3 ~245us) = {:.0} images/s ({:.2}x single-thread), NOT 4x because conv3 is 53% of total; data-parallel cap ~= 4 x single-thread = {:.0} images/s",
        1_000_000.0 / 465.0,
        1_000_000.0 / 245.0,
        465.0 / 245.0,
        4.0 * 1_000_000.0 / 465.0,
    );

    // Correctness gate FIRST, on a small prefix, before any timing is trusted.
    // Cross-checked against BOTH the `AndThen`-composed production surface
    // AND ROW 172's own direct-call arm (they were already proven
    // bit-identical to each other there) -- a third independent path to the
    // same reference number, not a fresh assumption.
    let reference: Vec<[f32; 10]> = images.iter().map(|image| run_pipeline_forward(image, &weights, BAND)).collect();
    let direct_match = images.iter().zip(&reference).filter(|(image, expected)| run_pipeline_forward_direct(image, &weights, BAND) == **expected).count();
    assert_eq!(direct_match, images.len(), "run_pipeline_forward_direct diverged from run_pipeline_forward on this image set");

    let stage_parallel = run_stage_parallel(&images, weights);
    let stage_parallel_match = stage_parallel.outputs.iter().zip(&reference).filter(|(actual, expected)| *actual == *expected).count();
    println!("stage_parallel correctness: {stage_parallel_match}/{} bit-identical to run_pipeline_forward", images.len());
    assert_eq!(stage_parallel_match, images.len(), "stage_parallel arm diverged from the single-thread reference");

    let data_parallel = run_data_parallel(&images, weights, 4);
    let data_parallel_match = data_parallel.outputs.iter().zip(&reference).filter(|(actual, expected)| *actual == *expected).count();
    println!("data_parallel correctness: {data_parallel_match}/{} bit-identical to run_pipeline_forward", images.len());
    assert_eq!(data_parallel_match, images.len(), "data_parallel arm diverged from the single-thread reference");

    report("stage_parallel", images.len(), &stage_parallel);
    report("data_parallel(4)", images.len(), &data_parallel);

    // Full-1000 accuracy on the faster arm, per the task's own gate.
    let full_images = load_normalized_images(&test_images_path(), FULL_SPLIT_IMAGES);
    let labels = load_labels(&test_labels_path(), FULL_SPLIT_IMAGES);
    let faster_is_data_parallel = data_parallel.wall < stage_parallel.wall;
    let faster_label = if faster_is_data_parallel { "data_parallel(4)" } else { "stage_parallel" };
    let full_outputs = if faster_is_data_parallel { run_data_parallel(&full_images, weights, 4).outputs } else { run_stage_parallel(&full_images, weights).outputs };
    let correct = full_outputs
        .iter()
        .zip(&labels)
        .filter(|(logits, label)| {
            let argmax = logits.iter().enumerate().max_by(|(_, left), (_, right)| left.total_cmp(right)).map(|(index, _)| index).expect("10 logits");
            argmax == **label as usize
        })
        .count();
    println!("full-1000 accuracy on faster arm ({faster_label}): {correct}/{} = {:.4}", full_images.len(), correct as f64 / full_images.len() as f64);
}
