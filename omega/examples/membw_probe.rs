//! The memory-bandwidth ceiling every weight-throughput ratio in this file's
//! log divides by, MEASURED rather than derived or taken from a spec sheet.
//!
//! Every prior row's "kernel achieved X GB/s" number lacked a denominator:
//! nothing in this crate had ever measured "how fast can this device's
//! memory system stream bytes, full stop." A derived ceiling was struck from
//! the log once already for exactly that reason (guiding-principles #18: a
//! DERIVED number may never carry a mechanism claim) — this probe exists so
//! the fraction-of-ceiling arithmetic has a real numerator AND a real
//! denominator.
//!
//! Two arms, run independently because they can plausibly hit different
//! ceilings even on Apple Silicon's unified memory (different cache paths,
//! different access-granularity, different driver overhead):
//!
//!   - CPU: a plain Rust sum-reduce over a buffer far larger than any cache
//!     level on this host (M1 Max: 12 MB/core-cluster L2, ~48 MB SLC), single-
//!     threaded and multi-threaded (`std::thread::available_parallelism`).
//!     One add per element — as close to zero arithmetic-per-byte as a
//!     reduction can get without becoming a no-op the compiler deletes.
//!   - Metal: the same shape (one full reduce-to-scalar, Add, over one big
//!     f32 buffer), run through `omega::execute_plan` at two buffer sizes so
//!     the marginal difference cancels the per-call fixed cost — the SAME
//!     two-size technique `q4k_matvec_probe` uses and this file's own ROW 71
//!     established as mandatory once a single-size number was shown to
//!     conflate kernel-bandwidth with per-call driver overhead.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::hint::black_box;
use std::time::Instant;

// 512 Mi f32 elements = 2 GiB, ~40x this host's ~48 MB SLC.
const CPU_ELEMENTS: usize = 512 * 1024 * 1024;
const CPU_RUNS: usize = 5;

fn fill(buffer: &mut [f32]) {
    for (index, slot) in buffer.iter_mut().enumerate() {
        // index-derived, not all-equal or all-zero, so no SIMD reduction
        // shortcut collapses the sum to a closed form the compiler can hoist.
        *slot = (index as f32 * 0.000_001).sin();
    }
}

fn sum_single_threaded(buffer: &[f32]) -> f32 {
    buffer.iter().copied().sum()
}

fn sum_multi_threaded(buffer: &[f32], workers: usize) -> f32 {
    let chunk_len = buffer.len().div_ceil(workers);
    std::thread::scope(|scope| {
        let handles: Vec<_> = buffer
            .chunks(chunk_len)
            .map(|chunk| scope.spawn(move || chunk.iter().copied().sum::<f32>()))
            .collect();
        handles.into_iter().map(|handle| handle.join().expect("worker thread panicked")).sum()
    })
}

fn run_cpu_arm() {
    let bytes = CPU_ELEMENTS * std::mem::size_of::<f32>();
    let mut buffer = vec![0.0_f32; CPU_ELEMENTS];
    fill(&mut buffer);

    let mut single_ms = Vec::with_capacity(CPU_RUNS);
    for _ in 0..CPU_RUNS {
        let started = Instant::now();
        let total = sum_single_threaded(&buffer);
        single_ms.push(started.elapsed().as_secs_f64() * 1000.0);
        black_box(total);
    }
    single_ms.sort_by(f64::total_cmp);

    let workers = std::thread::available_parallelism().map(std::num::NonZero::get).unwrap_or(1);
    let mut multi_ms = Vec::with_capacity(CPU_RUNS);
    for _ in 0..CPU_RUNS {
        let started = Instant::now();
        let total = sum_multi_threaded(&buffer, workers);
        multi_ms.push(started.elapsed().as_secs_f64() * 1000.0);
        black_box(total);
    }
    multi_ms.sort_by(f64::total_cmp);

    let gbs = |ms: f64| (bytes as f64 / 1e9) / (ms / 1000.0);
    let cov = |samples: &[f64]| {
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let variance =
            samples.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / samples.len() as f64;
        (variance.sqrt() / mean) * 100.0
    };

    println!("membw_probe CPU arm: buffer={} MiB ({bytes} bytes), pattern=single-pass streaming sum-reduce, {CPU_RUNS} runs, min reported (sibling-process interference inflates, never deflates)", bytes / (1024 * 1024));
    println!(
        "  single-thread: min={:.3} ms  {:.1} GB/s   CoV={:.2}%   all_ms={:?}",
        single_ms[0],
        gbs(single_ms[0]),
        cov(&single_ms),
        single_ms
    );
    println!(
        "  multi-thread (workers={workers}): min={:.3} ms  {:.1} GB/s   CoV={:.2}%   all_ms={:?}",
        multi_ms[0],
        gbs(multi_ms[0]),
        cov(&multi_ms),
        multi_ms
    );
}

#[cfg(all(feature = "metal", feature = "cpu", target_os = "macos"))]
fn run_metal_arm() {
    use proxima_tensor::{
        DType, Extent, IndexMap, Keep, NodeId, Op, QuantizedBlock, Reduce, ReduceInit, ScalarOp, append,
        map,
    };

    fn full_reduce_program(elements: u32) -> (Vec<Op>, NodeId) {
        let mut program = Vec::new();
        let source = append(
            &mut program,
            Op::Input { dtype: DType::Float32, shape: vec![Extent::Static(elements)], name: None },
        );
        let sum = append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: source,
                in_map: IndexMap::Affine(map::projection(1, &[0])),
                out_map: IndexMap::Affine(map::projection(1, &[])),
                keep: Keep::Reduce,
                name: None,
            }),
        );
        (program, sum)
    }

    fn measure(elements: u32, runs: usize) -> (f64, f64) {
        let bytes = f64::from(elements) * 4.0;
        let data: Vec<f32> = (0..elements).map(|index| (f64::from(index) * 1e-6).sin() as f32).collect();
        let (program, sum) = full_reduce_program(elements);
        let blocks = [QuantizedBlock::Float32(&data)];
        let resolved = omega::plan(&program, &[], &blocks, &[sum]).expect("membw probe plans");
        omega::execute_plan(&resolved, &blocks).expect("membw probe warms up");
        let mut samples = Vec::with_capacity(runs);
        for _ in 0..runs {
            let started = Instant::now();
            let out = omega::execute_plan(&resolved, &blocks).expect("membw probe executes");
            samples.push(started.elapsed().as_secs_f64() * 1000.0);
            black_box(out);
        }
        samples.sort_by(f64::total_cmp);
        (samples[0], bytes)
    }

    const RUNS: usize = 21;
    // Sizes chosen so the multi-hundred-MB sweep dominates the ~0.2-0.4 ms
    // per-call fixed cost (compile-once via `plan`, but still one command
    // buffer + waitUntilCompleted + readback per `execute_plan` call).
    let (small_ms, small_bytes) = measure(64 * 1024 * 1024, RUNS);
    let (large_ms, large_bytes) = measure(256 * 1024 * 1024, RUNS);

    let delta_ms = large_ms - small_ms;
    let delta_bytes = large_bytes - small_bytes;
    let marginal_gbs = (delta_bytes / 1e9) / (delta_ms / 1000.0);

    println!(
        "membw_probe Metal arm: pattern=full reduce-to-scalar (Add), one add per element, {RUNS} runs, min per size"
    );
    println!(
        "  small  buffer={:.1} MB  min={small_ms:.3} ms  single-size={:.1} GB/s",
        small_bytes / 1e6,
        (small_bytes / 1e9) / (small_ms / 1000.0)
    );
    println!(
        "  large  buffer={:.1} MB  min={large_ms:.3} ms  single-size={:.1} GB/s",
        large_bytes / 1e6,
        (large_bytes / 1e9) / (large_ms / 1000.0)
    );
    println!(
        "  MARGINAL (large-small, cancels per-call fixed cost): {:.1} MB in {delta_ms:.3} ms = {marginal_gbs:.1} GB/s",
        delta_bytes / 1e6
    );
}

#[cfg(not(all(feature = "metal", feature = "cpu", target_os = "macos")))]
fn run_metal_arm() {
    println!("membw_probe Metal arm requires --features metal,cpu on macOS; skipped");
}

fn main() {
    run_cpu_arm();
    run_metal_arm();
}
