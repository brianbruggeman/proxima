//! Async load sweep — the production shape, not single-thread microbench.
//!
//! Run: `cargo bench -p proxima-telemetry --bench bench_trace_async_sweep`
//!
//! Real telemetry is emitted from MANY concurrent async tasks on a FIXED worker
//! pool, not from raw OS threads in a hot loop. Concurrency comes from tasks
//! (can be thousands); parallelism is the worker pool (≈ cores). Those are
//! different axes, and conflating them is what made the earlier sweep useless.
//!
//! Each task models a request handler: emit a span, then `yield_now().await`
//! (the await every real async handler has — without it one task starves the
//! whole worker). Offered load is paced per task (Hz) so we can cross the
//! drain-capacity line deliberately instead of always running flat-out.
//!
//! Four sub-sweeps, each isolating one variable:
//!   A. concurrency      — tasks 1..4096 on workers=cores (does emit scale with
//!                         concurrency without OS oversubscription? yes — tasks
//!                         are not threads).
//!   B. worker oversub   — fix tasks, sweep worker_threads cores/2..4×cores. THE
//!                         "how do we know we oversubscribed?" answer: the p99/max
//!                         cliff appears when WORKERS exceed cores, not when tasks do.
//!   C. offered vs drain — pace offered rate below/at/above sink drain capacity.
//!                         the leading signal is assist% (records the producer had
//!                         to drain itself), which climbs BEFORE any latency cliff
//!                         while dropped stays 0.
//!   D. drain provisioning — sweep background drainer threads; more drain => lower
//!                         assist% and higher sustained rate.
//!
//! Latency is captured in a lock-free log2 histogram (no per-sample memory, no
//! warmup bias, scales to any task count). Harness, not a test: it prints.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::hint::black_box;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use proxima_telemetry::config::OverflowPolicy;
use proxima_telemetry::out::native::{FrameSink, NATIVE_FRAME_SIZE};
use proxima_telemetry::pipes::NativePipe;
use proxima_telemetry::recorder::Recorder;
use tokio::runtime::Builder;

// per-record sink cost, modeled by a busy-spin (real CPU time the export costs).
struct SpinSink {
    spin_ns: u64,
}

impl FrameSink for SpinSink {
    fn write_frame(&self, frame: &[u8; NATIVE_FRAME_SIZE]) {
        black_box(frame);
        if self.spin_ns == 0 {
            return;
        }
        let started = Instant::now();
        while (started.elapsed().as_nanos() as u64) < self.spin_ns {
            core::hint::spin_loop();
        }
    }
}

// lock-free log2-ns latency histogram. bucket b holds emits with latency in
// [2^(b-1), 2^b) ns; quantile() returns that bucket's upper bound (2^b). Coarse
// (power-of-two) but the story here is orders of magnitude (µs vs ms), and it
// costs one atomic add per emit with zero per-sample storage at any task count.
struct LatencyHist {
    buckets: [AtomicU64; 48],
    max_ns: AtomicU64,
}

impl LatencyHist {
    fn new() -> Self {
        Self {
            buckets: core::array::from_fn(|_| AtomicU64::new(0)),
            max_ns: AtomicU64::new(0),
        }
    }

    fn record(&self, ns: u64) {
        let bucket = (64 - ns.max(1).leading_zeros()) as usize;
        self.buckets[bucket.min(47)].fetch_add(1, Ordering::Relaxed);
        self.max_ns.fetch_max(ns, Ordering::Relaxed);
    }

    fn quantile(&self, quant: f64) -> u64 {
        let total: u64 = self
            .buckets
            .iter()
            .map(|bucket| bucket.load(Ordering::Relaxed))
            .sum();
        if total == 0 {
            return 0;
        }
        let target = (total as f64 * quant) as u64;
        let mut cumulative = 0u64;
        for (index, bucket) in self.buckets.iter().enumerate() {
            cumulative += bucket.load(Ordering::Relaxed);
            if cumulative >= target {
                return 1u64 << index;
            }
        }
        1u64 << 47
    }

    fn max(&self) -> u64 {
        self.max_ns.load(Ordering::Relaxed)
    }
}

struct Run {
    offered_per_s: f64,
    p50: u64,
    p99: u64,
    p999: u64,
    max: u64,
    dropped: u64,
    assist_pct: f64,
}

#[allow(clippy::too_many_arguments)]
fn async_load(
    workers: usize,
    tasks: usize,
    per_task_hz: f64,
    core_count: usize,
    sink_ns: u64,
    drainers: usize,
    ring_cap: usize,
    window: Duration,
) -> Run {
    let recorder = Arc::new(
        Recorder::builder()
            .pipe(NativePipe::new(SpinSink { spin_ns: sink_ns }))
            .core_count(core_count)
            .ring_capacity(ring_cap)
            .metric_capacity(ring_cap)
            .overflow(OverflowPolicy::Block)
            .start()
            .expect("recorder"),
    );

    // background drain: `drainers` OS threads over disjoint core ranges.
    let stop_drain = Arc::new(AtomicBool::new(false));
    let chunk = core_count.div_ceil(drainers.max(1));
    let drain_handles: Vec<_> = (0..drainers)
        .map(|index| {
            let recorder = Arc::clone(&recorder);
            let stop = Arc::clone(&stop_drain);
            let start = index * chunk;
            let end = ((index + 1) * chunk).min(core_count);
            thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    if recorder.drain_range(start, end) == 0 {
                        thread::yield_now();
                    }
                }
            })
        })
        .collect();

    let hist = Arc::new(LatencyHist::new());
    let emitted = Arc::new(AtomicU64::new(0));
    let stop_tasks = Arc::new(AtomicBool::new(false));
    let dt = if per_task_hz > 0.0 {
        Some(Duration::from_secs_f64(1.0 / per_task_hz))
    } else {
        None
    };

    let runtime = Builder::new_multi_thread()
        .worker_threads(workers)
        .enable_time()
        .build()
        .expect("runtime");

    let started = Instant::now();
    runtime.block_on(async {
        let mut handles = Vec::with_capacity(tasks);
        for _ in 0..tasks {
            let recorder = Arc::clone(&recorder);
            let hist = Arc::clone(&hist);
            let emitted = Arc::clone(&emitted);
            let stop = Arc::clone(&stop_tasks);
            handles.push(tokio::spawn(async move {
                let mut local = 0u64;
                while !stop.load(Ordering::Relaxed) {
                    let one = Instant::now();
                    drop(
                        recorder
                            .span(black_box("process"))
                            .tag("route", black_box("/v1"))
                            .start(),
                    );
                    hist.record(one.elapsed().as_nanos() as u64);
                    local += 1;
                    match dt {
                        // paced: the await models the handler's think/IO time.
                        Some(interval) => tokio::time::sleep(interval).await,
                        // flat-out: still yield once so C tasks actually interleave
                        // on the worker pool (a non-awaiting task starves its worker).
                        None => tokio::task::yield_now().await,
                    }
                }
                emitted.fetch_add(local, Ordering::Relaxed);
            }));
        }
        tokio::time::sleep(window).await;
        stop_tasks.store(true, Ordering::Relaxed);
        for handle in handles {
            let _ = handle.await;
        }
    });
    let elapsed = started.elapsed().as_secs_f64();

    stop_drain.store(true, Ordering::Relaxed);
    for handle in drain_handles {
        handle.join().expect("drainer");
    }
    while recorder.drain() > 0 {}

    let emitted_n = emitted.load(Ordering::Relaxed);
    let assisted = recorder.assisted();
    Run {
        offered_per_s: emitted_n as f64 / elapsed,
        p50: hist.quantile(0.50),
        p99: hist.quantile(0.99),
        p999: hist.quantile(0.999),
        max: hist.max(),
        dropped: recorder.dropped(),
        assist_pct: 100.0 * assisted as f64 / emitted_n.max(1) as f64,
    }
}

fn row(label: String, run: &Run) {
    println!(
        "  {label:<22} {:>10.2}M {:>9} {:>9} {:>9} {:>9} {:>7} {:>8.1}%",
        millions(run.offered_per_s),
        fmt_ns(run.p50),
        fmt_ns(run.p99),
        fmt_ns(run.p999),
        fmt_ns(run.max),
        run.dropped,
        run.assist_pct,
    );
}

fn header() {
    println!(
        "  {:<22} {:>11} {:>9} {:>9} {:>9} {:>9} {:>7} {:>9}",
        "config", "throughput", "p50", "p99", "p999", "max", "dropped", "assist%"
    );
}

fn main() {
    let cores = thread::available_parallelism()
        .map(|value| value.get())
        .unwrap_or(8);
    let ring_cap = 4096;
    let window = Duration::from_millis(1500);
    let local = 2_000u64;
    let network = 50_000u64;

    println!(
        "# async load sweep — host cores={cores}, ring_cap={ring_cap}, {}ms/run",
        window.as_millis()
    );
    println!("tasks = concurrent async emitters (tokio); workers = OS worker pool. latency is a");
    println!("log2 histogram (quantile = bucket upper bound). 'network' sink = 50µs/record.\n");

    // A. concurrency: tasks scale on workers=cores. does emit hold up as
    // concurrency rises far past cores WITHOUT oversubscribing the OS? (tasks≠threads)
    println!("## A. concurrency — workers=cores={cores}, 2 drainers, local 2µs sink, flat-out");
    header();
    for tasks in [1usize, cores, 64, 512, 4096] {
        let run = async_load(cores, tasks, 0.0, cores, local, 2, ring_cap, window);
        row(format!("tasks={tasks}"), &run);
    }
    println!(
        "  -> concurrency is a task count, not a thread count: thousands of tasks stay on the"
    );
    println!(
        "     fixed pool. push-CAS contention rises with tasks but the OS is not oversubscribed.\n"
    );

    // B. worker oversubscription: THE answer to "how do we know we oversub?"
    println!("## B. worker oversubscription — 512 tasks, 2 drainers, local 2µs sink, flat-out");
    println!(
        "   sweep WORKER threads around cores={cores}: the cliff is at workers>cores, not tasks."
    );
    header();
    for workers in [(cores / 2).max(1), cores, cores * 2, cores * 4] {
        let run = async_load(workers, 512, 0.0, cores, local, 2, ring_cap, window);
        row(format!("workers={workers}"), &run);
    }
    println!(
        "  -> p99/max climb once workers>cores (runnable threads exceed cores => OS preemption)."
    );
    println!(
        "     rule: worker_threads ≈ cores. you do NOT control oversubscription with task count.\n"
    );

    // C. offered load vs drain capacity: assist% is the leading signal. pace the
    // aggregate offered rate across the single-drainer capacity to show assist%
    // cross 0 -> nonzero BEFORE any drop (dropped stays 0 throughout).
    let load_tasks = 512;
    println!(
        "## C. offered vs drain capacity — workers=cores, {load_tasks} tasks, 1 drainer, network 50µs sink"
    );
    println!("   pace per-task Hz so aggregate offered crosses the single-drainer drain capacity.");
    header();
    for aggregate_hz in [50_000.0, 150_000.0, 300_000.0, 0.0] {
        let per_task = if aggregate_hz > 0.0 {
            aggregate_hz / load_tasks as f64
        } else {
            0.0
        };
        let label = if aggregate_hz > 0.0 {
            format!("offered≈{}k/s", (aggregate_hz / 1000.0) as u64)
        } else {
            "flat-out".to_string()
        };
        let run = async_load(
            cores, load_tasks, per_task, cores, network, 1, ring_cap, window,
        );
        row(label, &run);
    }
    println!(
        "  -> below capacity assist%≈0 and p99 is the push; as offered crosses drain capacity"
    );
    println!(
        "     assist% climbs FIRST (dropped stays 0). watch assisted() to know you're under-drained.\n"
    );

    // D. drain provisioning: the lever for C.
    println!("## D. drain provisioning — workers=cores, 512 tasks, network 50µs sink, flat-out");
    header();
    for drainers in [1usize, 2, 4] {
        let run = async_load(cores, 512, 0.0, cores, network, drainers, ring_cap, window);
        row(format!("drainers={drainers}"), &run);
    }
    println!(
        "  -> more drainers (drain_range partitioned) lift sustained rate and cut assist% — the"
    );
    println!("     fix when C shows you're under-drained. all four knobs are config-driven.\n");

    println!("how to read this in prod:");
    println!(
        "  - oversubscription is a WORKER-thread property (sweep B): keep worker_threads≈cores;"
    );
    println!("    async task concurrency (sweep A) is free and is NOT the oversubscription axis.");
    println!("  - 'are we keeping up?' = recorder.assisted() (sweep C): 0 = drain ahead of emit;");
    println!(
        "    climbing = under-drained, add drainers (sweep D) or shed (Drop). dropped() stays 0."
    );
}

fn millions(value: f64) -> f64 {
    value / 1_000_000.0
}

fn fmt_ns(ns: u64) -> String {
    if ns >= 1_000_000 {
        std::format!("{:.1}ms", ns as f64 / 1e6)
    } else if ns >= 1_000 {
        std::format!("{:.1}us", ns as f64 / 1e3)
    } else {
        std::format!("{ns}ns")
    }
}
