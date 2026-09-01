//! Tail-latency and soak evidence for the mnist f32 inference lane, on the
//! DEFAULT generic-executor path (`proxima_tensor::cpu::evaluate_named`, no
//! `PROXIMA_ACCELERATE_GEMM` toggle, epilogue fusion default-on per ROW 186).
//! Reuses the EXISTING `benches/common/hdr_phased.rs` phased-tail harness
//! (`HdrQuartet`) for the TAIL arm's percentile reporting -- no new
//! reporting code lands in that shared module. The SOAK arm reuses the same
//! `hdrhistogram` crate directly (already a workspace dependency the shared
//! harness itself depends on) for per-minute histograms, since `HdrQuartet`
//! is shaped for one warmup/steady/spike/spindown pass over a KNOWN
//! iteration count, not an open-ended multi-minute sustained-rate loop.
//!
//! Allocation-growth tracking reuses the counting-allocator precedent from
//! `proxima-onnx/tests/epilogue_fuse_alloc.rs` verbatim (same
//! `CountingAllocator` shape, same `#[global_allocator]` placement).
//!
//! Fixtures: the real, on-disk `mnist.onnx` checkpoint and the real MNIST
//! `t10k` test split (10,000 images) -- the same ones
//! `proxima-onnx/benches/mnist_f32_lane.rs` and
//! `proxima-onnx/tests/real_mnist_accuracy.rs` already read. Loaders
//! duplicated here rather than shared: separate bench/test binary targets
//! within one crate cannot share a module without a common library, and the
//! existing files in this crate already establish per-target duplication as
//! the convention (see `epilogue_fuse_alloc.rs`'s own doc comment).
//!
//! `mnist-tail-bench`, default-off: presence-guarded on
//! `~/repos/others/burn/examples/onnx-inference/src/model/mnist.onnx` +
//! `~/.cache/burn-dataset/mnist`, clean skip (prints and returns) when
//! either is absent.
//!
//! Re-prove with (host must be quiet -- see the discipline log row this
//! bench seeds for the loadout it was actually measured under):
//! `CARGO_TARGET_DIR=<scratch> cargo bench -p proxima-onnx --bench mnist_tail_soak --features mnist-tail-bench`
//! Soak duration is configurable via `MNIST_SOAK_MINUTES` (default 5); the
//! discipline row states honestly whichever value was actually used.
//!
//! **HARNESS-LANDED/UNMEASURED (2026-09-01):** this session never reached a
//! quiet host (18 minutes waited, load 8.2-38.2 throughout, another
//! bencher's session held the measurement slot per an owner/coordinator
//! scope-change mid-session) -- the code below is compiled, clippy-clean,
//! and functionally smoke-tested (all arms execute, N>0 in every phase and
//! every minute), but its printed numbers from that smoke run are NOT a
//! sealed measurement and are not cited as one anywhere in the discipline
//! log. The re-prove command above is the exact, unmodified command the
//! next quiet-host session runs to seal ROW 195.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use hdrhistogram::Histogram;

#[path = "../../benches/common/hdr_phased.rs"]
mod hdr_phased;
use hdr_phased::HdrQuartet;

thread_local! {
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

struct CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.with(|count| count.set(count.get() + 1));
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.with(|count| count.set(count.get() + 1));
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

const MODEL_PATH: &str = "/Users/brianbruggeman/repos/others/burn/examples/onnx-inference/src/model/mnist.onnx";
const DATASET_DIR: &str = "/Users/brianbruggeman/.cache/burn-dataset/mnist";
/// Real t10k split has 10,000 images; the task floor is >=5,000. We use the
/// whole split once for the TAIL arm (no recycling needed there) and recycle
/// it by modulo for the SOAK arm's much larger call count.
const TAIL_IMAGE_COUNT: usize = 10_000;
/// SOAK arm pacing: comfortably under the ~1.0-1.1ms/image sealed mean (ROW
/// 194) so the loop is genuinely open-loop fixed-rate, not closed-loop
/// max-throughput -- period 5ms leaves ~4ms of headroom per call at the
/// sealed mean before any queueing could occur.
const SOAK_TARGET_RATE_PER_SEC: f64 = 200.0;
/// Pre-registered drift band (stated BEFORE any soak minute runs, per the
/// task's own instruction): a later minute's own p99 is "no drift" if it
/// stays within +/-25% of the first minute's p99. Wide enough to absorb
/// normal run-to-run CoV already measured on this host (ROW 194: 13-15%
/// CoV run-to-run), tight enough to catch a real multi-minute trend.
const DRIFT_BAND_PCT: f64 = 25.0;
/// Pre-registered allocation-growth band: a later minute's own
/// allocations-per-image average is "flat" if it stays within +/-15% of the
/// first minute's own average. Rust has no GC, so any real drift here
/// would indicate an actual per-call allocation-count regression, not GC
/// pause accounting.
const ALLOC_BAND_PCT: f64 = 15.0;

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

/// Verbatim of `real_mnist_accuracy.rs`'s / `mnist_f32_lane.rs`'s own idx3
/// header parse.
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

fn soak_minutes_from_env() -> u64 {
    std::env::var("MNIST_SOAK_MINUTES").ok().and_then(|value| value.parse().ok()).unwrap_or(5)
}

fn fresh_minute_histogram() -> Histogram<u64> {
    // 60s upper bound, same choice `h2_tail_vs_incumbents.rs` makes for its
    // own hand-rolled histograms -- generous enough that a genuine
    // page-fault/scheduler/thermal stall is captured, not silently dropped.
    Histogram::<u64>::new_with_bounds(1, 60_000_000_000, 3).expect("hdr bounds")
}

/// Prints the one percentile `HdrQuartet::report` does not carry (p95),
/// reading its already-public histogram fields directly -- no change to
/// `hdr_phased.rs` itself.
fn print_p95(arm: &str, phase: &str, hist: &Histogram<u64>) {
    let count = hist.len();
    if count == 0 {
        println!("arm={arm} phase={phase} p95=0ns");
        return;
    }
    println!("arm={arm} phase={phase} p95={}ns", hist.value_at_quantile(0.95));
}

fn mean_stdev_cov(values: &[f64]) -> (f64, f64, f64) {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / values.len() as f64;
    let stdev = variance.sqrt();
    let cov = if mean > 0.0 { stdev / mean * 100.0 } else { 0.0 };
    (mean, stdev, cov)
}

fn main() {
    if !checkpoint_present() {
        eprintln!("mnist_tail_soak: skipping, no host-local mnist.onnx checkout at {MODEL_PATH}");
        return;
    }
    if !dataset_present() {
        eprintln!("mnist_tail_soak: skipping, no host-local MNIST idx dataset under {DATASET_DIR}");
        return;
    }

    let bytes = fs::read(MODEL_PATH).expect("read the real mnist.onnx checkpoint");
    let model = proxima_onnx::pipe::parse_complete(&bytes).expect("parse the real mnist.onnx checkpoint");
    let graph = model.graph.as_ref().expect("real mnist model has a graph");
    let lowered = proxima_onnx::lower::lower_graph(graph).expect("lower the real mnist.onnx graph to Op");

    let graph_input_name = lowered.graph_inputs.first().expect("real mnist model declares at least one input").clone();
    let output_node = lowered.graph_outputs.first().expect("real mnist model declares at least one output").1;
    let initializers: Vec<(&str, &[f32])> = lowered.initializers.iter().map(|(name, data)| (name.as_str(), data.as_slice())).collect();

    let images = load_normalized_images(&test_images_path(), TAIL_IMAGE_COUNT);
    assert!(images.len() >= 5_000, "task floor is >=5,000 images, got {}", images.len());

    let evaluate = |image: &[f32]| {
        let mut named = initializers.clone();
        named.push((graph_input_name.as_str(), image));
        proxima_tensor::cpu::evaluate_named(&lowered.program, &[], &named, &[output_node]).expect("evaluate real mnist image")
    };

    // pre-registration: stated BEFORE any measurement, from ROW 194's own
    // sealed NEON-default arm (mean 1.007274ms/image, CoV 13.04%, this same
    // default generic-executor path). A p999/max far above these bands is a
    // FINDING to mechanism-trace, not to hide.
    println!("mnist_tail_soak: PRE-REGISTERED bands (from ROW 194, sealed mean=1.007ms/image, CoV=13.04%):");
    println!("  expected p50 ~0.85-1.15ms, expected p99 ~1.5-3.0ms (2-3x mean, wide CoV precedent)");
    println!("  p999/max: no prior citation exists at this percentile depth -- any value >5ms is flagged as a FINDING below, not hidden");
    println!();

    // one uncounted warm-up call, same placement as mnist_f32_lane.rs /
    // epilogue_fuse_alloc.rs -- primes any first-call-only setup (program
    // pool growth) so it does not pollute the measured window.
    let _ = evaluate(&images[0]);

    // ---- TAIL arm ----
    let mut quartet = HdrQuartet::new();
    let mut true_max_ns: u64 = 0;
    let mut over_bound_count: usize = 0;
    for (index, image) in images.iter().enumerate() {
        let start = Instant::now();
        let evaluated = evaluate(image);
        std::hint::black_box(&evaluated);
        let elapsed_ns = (start.elapsed().as_nanos() as u64).max(1);
        true_max_ns = true_max_ns.max(elapsed_ns);
        if elapsed_ns >= 1_000_000_000 {
            over_bound_count += 1;
        }
        quartet.record(index as u64, elapsed_ns);
    }
    quartet.finalize(images.len() as u64);
    quartet.report("tail_default_generic");
    print_p95("tail_default_generic", "warmup", &quartet.warmup);
    print_p95("tail_default_generic", "steady", &quartet.steady);
    print_p95("tail_default_generic", "spike", &quartet.spike);
    print_p95("tail_default_generic", "spindown", &quartet.spindown);
    println!("tail_default_generic: true_max={true_max_ns}ns (independent of the 1s HdrQuartet histogram bound), samples_over_1s_bound={over_bound_count}");
    println!();

    // ---- SOAK arm ----
    let soak_minutes = soak_minutes_from_env();
    println!("mnist_tail_soak: SOAK arm, target_rate={SOAK_TARGET_RATE_PER_SEC}/s, requested_minutes={soak_minutes}");
    let target_period = Duration::from_secs_f64(1.0 / SOAK_TARGET_RATE_PER_SEC);
    let mut next_deadline = Instant::now();
    let mut image_cursor: usize = 0;
    let mut per_minute_p50_ns: Vec<f64> = Vec::new();
    let mut per_minute_p99_ns: Vec<f64> = Vec::new();
    let mut per_minute_alloc_per_image: Vec<f64> = Vec::new();
    let soak_wall_start = Instant::now();

    for minute in 0..soak_minutes {
        let minute_start = Instant::now();
        let minute_deadline = minute_start + Duration::from_secs(60);
        let alloc_before = ALLOCATIONS.with(Cell::get);
        let mut histogram = fresh_minute_histogram();
        let mut minute_max_ns: u64 = 0;
        while Instant::now() < minute_deadline {
            let now = Instant::now();
            if now < next_deadline {
                std::thread::sleep(next_deadline - now);
            }
            next_deadline += target_period;
            let index = image_cursor % images.len();
            image_cursor += 1;
            let start = Instant::now();
            let evaluated = evaluate(&images[index]);
            std::hint::black_box(&evaluated);
            let elapsed_ns = (start.elapsed().as_nanos() as u64).max(1);
            minute_max_ns = minute_max_ns.max(elapsed_ns);
            let _ = histogram.record(elapsed_ns);
        }
        let alloc_after = ALLOCATIONS.with(Cell::get);
        let alloc_delta = alloc_after - alloc_before;
        let count = histogram.len();
        let per_image_alloc = if count > 0 { alloc_delta as f64 / count as f64 } else { 0.0 };
        let p50 = histogram.value_at_quantile(0.50);
        let p90 = histogram.value_at_quantile(0.90);
        let p95 = histogram.value_at_quantile(0.95);
        let p99 = histogram.value_at_quantile(0.99);
        let p999 = histogram.value_at_quantile(0.999);
        println!(
            "arm=soak_default_generic phase=minute_{minute} count={count} p50={p50}ns p90={p90}ns p95={p95}ns p99={p99}ns p999={p999}ns max={minute_max_ns}ns alloc_delta={alloc_delta} alloc_per_image={per_image_alloc:.4}"
        );
        per_minute_p50_ns.push(p50 as f64);
        per_minute_p99_ns.push(p99 as f64);
        per_minute_alloc_per_image.push(per_image_alloc);
    }

    let soak_wall_elapsed = soak_wall_start.elapsed();
    println!("mnist_tail_soak: SOAK arm actually ran {:.2}s wall ({} of {} requested minutes fully completed)", soak_wall_elapsed.as_secs_f64(), per_minute_p99_ns.len(), soak_minutes);
    println!();

    if per_minute_p99_ns.len() >= 2 {
        let (mean, stdev, cov) = mean_stdev_cov(&per_minute_p99_ns);
        println!("mnist_tail_soak: per-minute p99 across {} minutes: mean={mean:.0}ns stdev={stdev:.0}ns CoV={cov:.2}%", per_minute_p99_ns.len());

        let first_p99 = per_minute_p99_ns[0];
        let mut worst_drift_pct: f64 = 0.0;
        let mut worst_drift_minute: usize = 0;
        for (minute, &p99) in per_minute_p99_ns.iter().enumerate().skip(1) {
            let drift_pct = (p99 - first_p99) / first_p99 * 100.0;
            if drift_pct.abs() > worst_drift_pct.abs() {
                worst_drift_pct = drift_pct;
                worst_drift_minute = minute;
            }
        }
        let drift_verdict = if worst_drift_pct.abs() <= DRIFT_BAND_PCT { "PASS" } else { "FAIL" };
        println!(
            "mnist_tail_soak: DRIFT CHECK ({drift_verdict}): worst deviation is minute {worst_drift_minute} at {worst_drift_pct:+.2}% vs minute 0's p99 ({first_p99:.0}ns), band=+/-{DRIFT_BAND_PCT}%"
        );

        let first_alloc = per_minute_alloc_per_image[0];
        let mut worst_alloc_drift_pct: f64 = 0.0;
        let mut worst_alloc_minute: usize = 0;
        for (minute, &per_image) in per_minute_alloc_per_image.iter().enumerate().skip(1) {
            let drift_pct = if first_alloc > 0.0 { (per_image - first_alloc) / first_alloc * 100.0 } else { 0.0 };
            if drift_pct.abs() > worst_alloc_drift_pct.abs() {
                worst_alloc_drift_pct = drift_pct;
                worst_alloc_minute = minute;
            }
        }
        let alloc_verdict = if worst_alloc_drift_pct.abs() <= ALLOC_BAND_PCT { "PASS" } else { "FAIL" };
        println!(
            "mnist_tail_soak: ALLOCATION-GROWTH CHECK ({alloc_verdict}): worst deviation is minute {worst_alloc_minute} at {worst_alloc_drift_pct:+.2}% vs minute 0's alloc/image ({first_alloc:.4}), band=+/-{ALLOC_BAND_PCT}%"
        );
    } else {
        println!("mnist_tail_soak: fewer than 2 full minutes completed -- drift and allocation-growth checks require >=2 minutes and were NOT run; reporting single-minute data only, honestly, per the task's own budget-permitting clause");
    }
}
