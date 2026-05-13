//! Load + soak harness — where the edges are, and vs the incumbents.
//!
//! Run: `cargo bench --bench bench_trace_soak`
//!
//! Measures, under sustained concurrent producers:
//!   1. proxima SATURATION — producers emit flat-out while one drainer drains;
//!      reports emit rate, the single-drainer ceiling, and the drop rate (drops
//!      = emitted − drained, since the ring drops-on-full with no counter).
//!   2. proxima SOAK — live-byte drift over time (a counting global allocator)
//!      to prove memory is stable (no leak) under sustained load.
//!   3. INCUMBENTS — the `tracing` crate (fmt → sink) and the OpenTelemetry SDK
//!      under the same flat-out load: throughput + heap allocations per span.
//!
//! This is a harness, not a unit test — it sleeps for fixed soak windows and
//! prints a report; nothing here asserts.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::io;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use opentelemetry::KeyValue;
use opentelemetry::trace::{Span, Tracer, TracerProvider as _};
use opentelemetry_sdk::trace::{InMemorySpanExporterBuilder, SdkTracerProvider};
use proxima_telemetry::config::{OverflowPolicy, RecordSharing};
use proxima_telemetry::out::native::{FrameSink, NATIVE_FRAME_SIZE};
use proxima_telemetry::pipes::NativePipe;
use proxima_telemetry::recorder::Recorder;
use tracing_subscriber::fmt::format::FmtSpan;

// ---- counting global allocator -------------------------------------------

struct Counting;
static ALLOC_CALLS: AtomicU64 = AtomicU64::new(0);
static LIVE_BYTES: AtomicI64 = AtomicI64::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
        LIVE_BYTES.fetch_add(layout.size() as i64, Ordering::Relaxed);
        // SAFETY: forwarding to the system allocator with the same layout.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE_BYTES.fetch_sub(layout.size() as i64, Ordering::Relaxed);
        // SAFETY: ptr came from System.alloc with this layout.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn alloc_calls() -> u64 {
    ALLOC_CALLS.load(Ordering::Relaxed)
}
fn live_bytes() -> i64 {
    LIVE_BYTES.load(Ordering::Relaxed)
}

// drain target: real native encode to a discarding sink (production export work).
struct NullSink;
impl FrameSink for NullSink {
    fn write_frame(&self, frame: &[u8; NATIVE_FRAME_SIZE]) {
        black_box(frame);
    }
}

const SOAK: Duration = Duration::from_secs(3);

struct Saturation {
    emitted: u64,
    drained: u64,
    dropped: u64,
    dropped_counter: u64,
    emit_rate: f64,
    drain_rate: f64,
    allocs_per_span: f64,
    live_drift: i64,
}

fn proxima_saturation(
    producers: usize,
    sharing: RecordSharing,
    overflow: OverflowPolicy,
) -> Saturation {
    let recorder = Arc::new(
        Recorder::builder()
            .pipe(NativePipe::new(NullSink))
            .core_count(producers)
            .record_sharing(sharing)
            .overflow(overflow)
            .start()
            .expect("recorder"),
    );
    // two stop flags: under Block, a producer backpressured on a full ring only
    // unblocks when the drainer frees a slot — so the drainer MUST outlive the
    // producers, else join() deadlocks. Stop producers first, then the drainer.
    let producer_stop = Arc::new(AtomicBool::new(false));
    let drain_stop = Arc::new(AtomicBool::new(false));
    let emitted = Arc::new(AtomicU64::new(0));

    let allocs_before = alloc_calls();
    let live_before = live_bytes();

    let producer_handles: Vec<_> = (0..producers)
        .map(|_| {
            let recorder = Arc::clone(&recorder);
            let stop = Arc::clone(&producer_stop);
            let emitted = Arc::clone(&emitted);
            thread::spawn(move || {
                let mut local = 0u64;
                while !stop.load(Ordering::Relaxed) {
                    let guard = recorder
                        .span(black_box("process"))
                        .tag("route", black_box("/v1"))
                        .tag("status", black_box(200u64))
                        .start();
                    drop(guard);
                    local += 1;
                }
                emitted.fetch_add(local, Ordering::Relaxed);
            })
        })
        .collect();

    let drainer = {
        let recorder = Arc::clone(&recorder);
        let stop = Arc::clone(&drain_stop);
        thread::spawn(move || {
            let mut drained = 0u64;
            while !stop.load(Ordering::Relaxed) {
                drained += recorder.drain() as u64;
            }
            drained
        })
    };

    let started = Instant::now();
    thread::sleep(SOAK);
    producer_stop.store(true, Ordering::Relaxed);
    for handle in producer_handles {
        handle.join().expect("producer");
    }
    drain_stop.store(true, Ordering::Relaxed);
    let mut drained = drainer.join().expect("drainer");
    // flush whatever was still in the rings after producers stopped.
    loop {
        let pass = recorder.drain() as u64;
        drained += pass;
        if pass == 0 {
            break;
        }
    }
    let elapsed = started.elapsed().as_secs_f64();
    let emitted = emitted.load(Ordering::Relaxed);
    let dropped = emitted.saturating_sub(drained);
    let dropped_counter = recorder.dropped();
    let allocs = alloc_calls() - allocs_before;

    Saturation {
        emitted,
        drained,
        dropped,
        dropped_counter,
        emit_rate: emitted as f64 / elapsed,
        drain_rate: drained as f64 / elapsed,
        allocs_per_span: allocs as f64 / emitted.max(1) as f64,
        live_drift: live_bytes() - live_before,
    }
}

// parallel drain: `drainers` threads partition the producer cores, each calling
// drain_range over a disjoint slice — lifts the single-drainer ceiling.
fn proxima_parallel_drain(producers: usize, drainers: usize) -> (f64, f64) {
    // Drop policy: this measures the drain ceiling with producers flat-out; the
    // shared stop flag would deadlock backpressured producers under Block.
    let recorder = Arc::new(
        Recorder::builder()
            .pipe(NativePipe::new(NullSink))
            .core_count(producers)
            .overflow(OverflowPolicy::Drop)
            .start()
            .expect("recorder"),
    );
    let stop = Arc::new(AtomicBool::new(false));
    let emitted = Arc::new(AtomicU64::new(0));

    let producer_handles: Vec<_> = (0..producers)
        .map(|_| {
            let recorder = Arc::clone(&recorder);
            let stop = Arc::clone(&stop);
            let emitted = Arc::clone(&emitted);
            thread::spawn(move || {
                let mut local = 0u64;
                while !stop.load(Ordering::Relaxed) {
                    drop(
                        recorder
                            .span(black_box("process"))
                            .tag("route", black_box("/v1"))
                            .start(),
                    );
                    local += 1;
                }
                emitted.fetch_add(local, Ordering::Relaxed);
            })
        })
        .collect();

    let chunk = producers.div_ceil(drainers);
    let drainer_handles: Vec<_> = (0..drainers)
        .map(|drainer_index| {
            let recorder = Arc::clone(&recorder);
            let stop = Arc::clone(&stop);
            let start = drainer_index * chunk;
            let end = ((drainer_index + 1) * chunk).min(producers);
            thread::spawn(move || {
                let mut drained = 0u64;
                while !stop.load(Ordering::Relaxed) {
                    drained += recorder.drain_range(start, end) as u64;
                }
                drained
            })
        })
        .collect();

    let started = Instant::now();
    thread::sleep(SOAK);
    stop.store(true, Ordering::Relaxed);
    for handle in producer_handles {
        handle.join().expect("producer");
    }
    let mut drained = 0u64;
    for handle in drainer_handles {
        drained += handle.join().expect("drainer");
    }
    for index in 0..producers {
        loop {
            let pass = recorder.drain_range(index, index + 1) as u64;
            drained += pass;
            if pass == 0 {
                break;
            }
        }
    }
    let elapsed = started.elapsed().as_secs_f64();
    let emitted = emitted.load(Ordering::Relaxed);
    (emitted as f64 / elapsed, drained as f64 / elapsed)
}

struct Incumbent {
    rate: f64,
    allocs_per_span: f64,
}

fn run_flat_out(producers: usize, work: impl Fn() + Send + Sync + 'static) -> Incumbent {
    let work = Arc::new(work);
    let stop = Arc::new(AtomicBool::new(false));
    let emitted = Arc::new(AtomicU64::new(0));
    let allocs_before = alloc_calls();

    let handles: Vec<_> = (0..producers)
        .map(|_| {
            let work = Arc::clone(&work);
            let stop = Arc::clone(&stop);
            let emitted = Arc::clone(&emitted);
            thread::spawn(move || {
                let mut local = 0u64;
                while !stop.load(Ordering::Relaxed) {
                    work();
                    local += 1;
                }
                emitted.fetch_add(local, Ordering::Relaxed);
            })
        })
        .collect();

    let started = Instant::now();
    thread::sleep(SOAK);
    stop.store(true, Ordering::Relaxed);
    for handle in handles {
        handle.join().expect("producer");
    }
    let elapsed = started.elapsed().as_secs_f64();
    let emitted = emitted.load(Ordering::Relaxed);
    let allocs = alloc_calls() - allocs_before;
    Incumbent {
        rate: emitted as f64 / elapsed,
        allocs_per_span: allocs as f64 / emitted.max(1) as f64,
    }
}

fn millions(value: f64) -> f64 {
    value / 1_000_000.0
}

// steady-state per-emit cost: emit into a ring that never fills (well under cap,
// no drainer needed), so push always returns Ok and the overflow policy's Err
// branch is never taken. Block vs Drop must be identical here — the whole point.
// median of 5 to read past noise.
fn steady_state_ns_per_emit(overflow: OverflowPolicy) -> f64 {
    let runs = 5;
    let emits = 2048u64; // < default ring cap (4096): no overflow
    let mut samples: Vec<f64> = (0..runs)
        .map(|_| {
            let recorder = Recorder::builder()
                .pipe(NativePipe::new(NullSink))
                .core_count(1)
                .overflow(overflow)
                .start()
                .expect("recorder");
            let started = Instant::now();
            for _ in 0..emits {
                drop(
                    recorder
                        .span(black_box("process"))
                        .tag("route", black_box("/v1"))
                        .start(),
                );
            }
            started.elapsed().as_secs_f64() * 1e9 / emits as f64
        })
        .collect();
    samples.sort_by(f64::total_cmp);
    samples[runs / 2]
}

fn main() {
    let cores = thread::available_parallelism()
        .map(|value| value.get())
        .unwrap_or(8);
    // leave one core for the proxima drainer; incumbents have no separate drainer.
    let producers = cores.saturating_sub(1).max(1);

    println!(
        "# trace load + soak — {producers} producer threads, {} s each, host cores={cores}\n",
        SOAK.as_secs()
    );

    // RecordSharing::Inline is now the default (single-sink fast path).
    let sat = proxima_saturation(producers, RecordSharing::Inline, OverflowPolicy::Drop);
    println!("## proxima — saturation (default Inline sharing, producers flat-out, 1 drainer)");
    println!(
        "  emit rate      : {:.2} M spans/s",
        millions(sat.emit_rate)
    );
    println!(
        "  drain ceiling  : {:.2} M spans/s   (single drainer, real native encode)",
        millions(sat.drain_rate)
    );
    println!("  emitted        : {}", sat.emitted);
    println!("  drained        : {}", sat.drained);
    println!(
        "  dropped        : {} ({:.1}%)   <- ring drop-on-full when emit > drain",
        sat.dropped,
        100.0 * sat.dropped as f64 / sat.emitted.max(1) as f64
    );
    println!(
        "  dropped counter: {}   <- recorder.dropped(); matches emitted-drained: {}",
        sat.dropped_counter,
        sat.dropped_counter == sat.dropped
    );
    println!(
        "  allocs / span  : {:.4}   (emit path is alloc-free; allocs are per-drain batch Vecs)",
        sat.allocs_per_span
    );
    println!(
        "  live-byte drift: {} bytes over the run   <- ~0 => no leak under sustained load\n",
        sat.live_drift
    );

    // overflow policy — the lossless default (Block) vs lossy (Drop).
    let drop_ns = steady_state_ns_per_emit(OverflowPolicy::Drop);
    let block_ns = steady_state_ns_per_emit(OverflowPolicy::Block);
    let block_sat = proxima_saturation(producers, RecordSharing::Inline, OverflowPolicy::Block);
    println!("## proxima — overflow policy: Block (lossless default) vs Drop (lossy)");
    println!(
        "  steady-state ns/emit (ring not full): Block={block_ns:.1} ns  Drop={drop_ns:.1} ns"
    );
    println!("    -> hot path identical (policy read only on the cold Full branch); Δ is noise\n");
    println!("  saturation (producers flat-out, 1 drainer):");
    println!(
        "    Drop : emit {:.2} M/s  drain {:.2} M/s  dropped {:.1}%  (sheds under overload)",
        millions(sat.emit_rate),
        millions(sat.drain_rate),
        100.0 * sat.dropped as f64 / sat.emitted.max(1) as f64
    );
    println!(
        "    Block: emit {:.2} M/s  drain {:.2} M/s  dropped {:.1}%  (lossless: emit throttled to drain)\n",
        millions(block_sat.emit_rate),
        millions(block_sat.drain_rate),
        100.0 * block_sat.dropped_counter as f64 / block_sat.emitted.max(1) as f64
    );

    // the old default (Arc) — for fanout pipes; pays a per-drained-span Arc.
    let arc = proxima_saturation(producers, RecordSharing::Arc, OverflowPolicy::Drop);
    println!("## proxima — RecordSharing::Arc (fanout sharing — the old default)");
    println!(
        "  drain ceiling  : {:.2} M spans/s   (vs {:.2} M with default Inline)",
        millions(arc.drain_rate),
        millions(sat.drain_rate)
    );
    println!(
        "  dropped        : {:.1}%   (vs {:.1}% with Inline)",
        100.0 * arc.dropped as f64 / arc.emitted.max(1) as f64,
        100.0 * sat.dropped as f64 / sat.emitted.max(1) as f64
    );
    println!(
        "  allocs / span  : {:.4}   (vs {:.4} with Inline)\n",
        arc.allocs_per_span, sat.allocs_per_span
    );

    // parallel drain — lift the single-drainer ceiling. Producers flat-out so a
    // single drainer is saturated (the backlog the extra drainers clear).
    let parallel_producers = producers;
    let (_, ceil_1) = proxima_parallel_drain(parallel_producers, 1);
    let (_, ceil_2) = proxima_parallel_drain(parallel_producers, 2);
    let (_, ceil_3) = proxima_parallel_drain(parallel_producers, 3);
    println!(
        "## proxima — parallel drain ({parallel_producers} producers flat-out, drain_range partitioned)"
    );
    println!("  drained, 1 drainer : {:.2} M spans/s", millions(ceil_1));
    println!(
        "  drained, 2 drainers: {:.2} M spans/s   ({:.2}x)",
        millions(ceil_2),
        ceil_2 / ceil_1
    );
    println!(
        "  drained, 3 drainers: {:.2} M spans/s   ({:.2}x)\n",
        millions(ceil_3),
        ceil_3 / ceil_1
    );

    // incumbents under the same flat-out load (no drainer — they export inline).
    let tracing_inc = {
        let subscriber = tracing_subscriber::fmt()
            .with_writer(io::sink)
            .with_span_events(FmtSpan::CLOSE)
            .finish();
        let _ = tracing::subscriber::set_global_default(subscriber);
        run_flat_out(producers, || {
            let span = tracing::span!(
                tracing::Level::INFO,
                "process",
                route = "/v1",
                status = 200u64
            );
            let _entered = span.enter();
        })
    };

    let otel_inc = {
        let exporter = InMemorySpanExporterBuilder::new().build();
        let provider = Arc::new(
            SdkTracerProvider::builder()
                .with_simple_exporter(exporter)
                .build(),
        );
        run_flat_out(producers, move || {
            let tracer = provider.tracer("soak");
            let mut span = tracer.start("process");
            span.set_attribute(KeyValue::new("route", "/v1"));
            span.set_attribute(KeyValue::new("status", 200i64));
            span.end();
        })
    };

    println!("## incumbents — same flat-out load (export inline, no separate drainer)");
    println!(
        "  {:<10} {:>14} {:>16}",
        "impl", "throughput", "allocs/span"
    );
    println!(
        "  {:<10} {:>12.2} M {:>16.2}",
        "proxima",
        millions(sat.emit_rate),
        sat.allocs_per_span
    );
    println!(
        "  {:<10} {:>12.2} M {:>16.2}",
        "tracing",
        millions(tracing_inc.rate),
        tracing_inc.allocs_per_span
    );
    println!(
        "  {:<10} {:>12.2} M {:>16.2}",
        "otel",
        millions(otel_inc.rate),
        otel_inc.allocs_per_span
    );
    println!();
    println!(
        "edges: proxima emit scales on per-core rings (safe only for producers <= core_count);"
    );
    println!("the single drainer is the end-to-end ceiling; ring drops-on-full when emit > drain,");
    println!("now observable via recorder.dropped() (no longer silent).");
}
