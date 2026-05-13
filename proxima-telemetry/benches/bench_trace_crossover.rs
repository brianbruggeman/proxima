//! Crossover scan — where (if ever) does proxima lose to the incumbents as
//! emit concurrency rises past core_count?
//!
//! Run: `cargo bench --bench bench_trace_crossover`
//!
//! For each concurrency W, all three stacks emit flat-out for a fixed window
//! from W blocking tasks on a tokio runtime (the realistic "telemetry from a
//! tokio app" shape), and we report span-API ingest throughput:
//!   - proxima  — Recorder (core_count = W) + a background drainer thread.
//!   - tokio + tracing — the `tracing` crate, global fmt subscriber → io::sink.
//!   - tokio + otel    — OpenTelemetry SDK with the batch span processor
//!     (bounded queue + background export thread — the production setup).
//!
//! Both proxima (ring) and otel (batch queue) drop on overflow under saturation;
//! tracing formats synchronously (no drop, but pays the format cost inline). The
//! throughput is "spans/sec the app can push through the API" — the number whose
//! crossover answers the question.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::hint::black_box;
use std::io;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use opentelemetry::KeyValue;
use opentelemetry::trace::{Span, Tracer, TracerProvider as _};
use opentelemetry_sdk::trace::{InMemorySpanExporterBuilder, SdkTracerProvider};
use proxima_telemetry::config::OverflowPolicy;
use proxima_telemetry::out::native::{FrameSink, NATIVE_FRAME_SIZE};
use proxima_telemetry::pipes::{NativePipe, NullPipe};
use proxima_telemetry::recorder::Recorder;
use tokio::runtime::Builder;
use tracing_subscriber::fmt::format::FmtSpan;

const WINDOW: Duration = Duration::from_millis(500);
const REPS: usize = 5;
const CONCURRENCY: [usize; 8] = [1, 2, 4, 8, 12, 16, 24, 32];

// median of REPS independent runs — the single-run version was too noisy to read
// the crossover from.
fn median(mut measure: impl FnMut() -> f64) -> f64 {
    let mut samples: Vec<f64> = (0..REPS).map(|_| measure()).collect();
    samples.sort_by(f64::total_cmp);
    samples[REPS / 2]
}

struct NullSink;
impl FrameSink for NullSink {
    fn write_frame(&self, frame: &[u8; NATIVE_FRAME_SIZE]) {
        black_box(frame);
    }
}

fn runtime(workers: usize) -> tokio::runtime::Runtime {
    Builder::new_multi_thread()
        .worker_threads(workers.max(1))
        .enable_time()
        .build()
        .expect("runtime")
}

// run `concurrency` blocking emitters flat-out for WINDOW; return spans/sec
// pushed through `emit` (which is called in a tight loop per task).
fn ingest_rate(concurrency: usize, workers: usize, emit: impl Fn() + Send + Sync + 'static) -> f64 {
    let emit = Arc::new(emit);
    let stop = Arc::new(AtomicBool::new(false));
    let emitted = Arc::new(AtomicU64::new(0));
    let rt = runtime(workers);
    let started = Instant::now();
    rt.block_on(async {
        let mut handles = Vec::new();
        for _ in 0..concurrency {
            let emit = Arc::clone(&emit);
            let stop = Arc::clone(&stop);
            let emitted = Arc::clone(&emitted);
            handles.push(tokio::task::spawn_blocking(move || {
                let mut local = 0u64;
                while !stop.load(Ordering::Relaxed) {
                    emit();
                    local += 1;
                }
                emitted.fetch_add(local, Ordering::Relaxed);
            }));
        }
        tokio::time::sleep(WINDOW).await;
        stop.store(true, Ordering::Relaxed);
        for handle in handles {
            let _ = handle.await;
        }
    });
    emitted.load(Ordering::Relaxed) as f64 / started.elapsed().as_secs_f64()
}

fn proxima_rate(concurrency: usize, drainers: usize, polite: bool) -> f64 {
    let recorder = Arc::new(
        Recorder::builder()
            .pipe(NativePipe::new(NullSink))
            .core_count(concurrency)
            .overflow(OverflowPolicy::Drop)
            .start()
            .expect("recorder"),
    );
    let drain_stop = Arc::new(AtomicBool::new(false));
    let chunk = concurrency.div_ceil(drainers);
    let drainer_handles: Vec<_> = (0..drainers)
        .map(|drainer_index| {
            let recorder = Arc::clone(&recorder);
            let drain_stop = Arc::clone(&drain_stop);
            let start = drainer_index * chunk;
            let end = ((drainer_index + 1) * chunk).min(concurrency);
            thread::spawn(move || {
                while !drain_stop.load(Ordering::Relaxed) {
                    recorder.drain_range(start, end);
                    // polite: yield the core between passes so emitters get it
                    // (tests whether the busy-loop drainer's core-hogging is what
                    // costs the high-W crossover).
                    if polite {
                        thread::yield_now();
                    }
                }
            })
        })
        .collect();
    let emit_recorder = Arc::clone(&recorder);
    let rate = ingest_rate(concurrency, concurrency, move || {
        drop(
            emit_recorder
                .span(black_box("process"))
                .tag("route", black_box("/v1"))
                .start(),
        );
    });
    drain_stop.store(true, Ordering::Relaxed);
    for handle in drainer_handles {
        handle.join().expect("drainer");
    }
    rate
}

// run `concurrency` blocking emitters under a tracing Dispatch (set per-thread,
// so we can compare subscriber configs without the one-shot global default).
fn tracing_rate(
    concurrency: usize,
    make_dispatch: impl Fn() -> tracing::Dispatch + Send + Sync + 'static,
) -> f64 {
    let dispatch = Arc::new(make_dispatch());
    let stop = Arc::new(AtomicBool::new(false));
    let emitted = Arc::new(AtomicU64::new(0));
    let rt = runtime(concurrency);
    let started = Instant::now();
    rt.block_on(async {
        let mut handles = Vec::new();
        for _ in 0..concurrency {
            let dispatch = Arc::clone(&dispatch);
            let stop = Arc::clone(&stop);
            let emitted = Arc::clone(&emitted);
            handles.push(tokio::task::spawn_blocking(move || {
                tracing::dispatcher::with_default(dispatch.as_ref(), || {
                    let mut local = 0u64;
                    while !stop.load(Ordering::Relaxed) {
                        let span = tracing::span!(tracing::Level::INFO, "process", route = "/v1");
                        let _entered = span.enter();
                        local += 1;
                    }
                    emitted.fetch_add(local, Ordering::Relaxed);
                });
            }));
        }
        tokio::time::sleep(WINDOW).await;
        stop.store(true, Ordering::Relaxed);
        for handle in handles {
            let _ = handle.await;
        }
    });
    emitted.load(Ordering::Relaxed) as f64 / started.elapsed().as_secs_f64()
}

// fmt subscriber — formats every span to a sink (the standard fmt::init deploy).
fn tracing_fmt() -> tracing::Dispatch {
    tracing::Dispatch::new(
        tracing_subscriber::fmt()
            .with_writer(io::sink)
            .with_span_events(FmtSpan::CLOSE)
            .finish(),
    )
}

// bare registry — tracks span lifecycle (slab store/free), no formatting, no
// export. the lightest config that still records spans; the fair floor.
fn tracing_registry() -> tracing::Dispatch {
    tracing::Dispatch::new(tracing_subscriber::registry())
}

fn otel_rate(concurrency: usize) -> f64 {
    let exporter = InMemorySpanExporterBuilder::new().build();
    let provider = Arc::new(
        SdkTracerProvider::builder()
            .with_batch_exporter(exporter)
            .build(),
    );
    ingest_rate(concurrency, concurrency, move || {
        let tracer = provider.tracer("crossover");
        let mut span = tracer.start("process");
        span.set_attribute(KeyValue::new("route", "/v1"));
        span.end();
    })
}

fn millions(value: f64) -> f64 {
    value / 1_000_000.0
}

// emit-only (no drainer) — does the EMIT path scale with cores, or cap? a flat
// cap near single-thread throughput == shared-cache-line (Arc refcount) contention.
fn emit_only_rate(concurrency: usize) -> f64 {
    // Drop: emit-only has no drainer, so Block would backpressure-hang on a full ring.
    let recorder = Arc::new(
        Recorder::builder()
            .pipe(NullPipe::new())
            .core_count(concurrency)
            .overflow(OverflowPolicy::Drop)
            .start()
            .expect("recorder"),
    );
    let emit_recorder = Arc::clone(&recorder);
    ingest_rate(concurrency, concurrency, move || {
        drop(
            emit_recorder
                .span(black_box("process"))
                .tag("route", black_box("/v1"))
                .start(),
        );
    })
}

fn main() {
    let cores = thread::available_parallelism()
        .map(|value| value.get())
        .unwrap_or(8);

    println!(
        "# emit-only scaling (no drainer) — does emit scale with cores? M spans/s, median of {REPS}\n"
    );
    print!("  W:");
    for concurrency in [1, 2, 4, 8, 16, 32] {
        print!(
            "  {concurrency:>2}={:>5.2}",
            millions(median(|| emit_only_rate(concurrency)))
        );
    }
    println!("   <- flat near 1-thread => Arc-refcount contention in span()\n");

    // tracing cost spectrum: bare registry (span lifecycle, no fmt — the floor)
    // vs fmt (formats each span — the standard fmt::init deploy). brackets what
    // "tokio + tracing" actually costs, so the crossover isn't a fmt strawman.
    println!("# tokio+tracing cost spectrum — M spans/s, median of {REPS}\n");
    print!("  registry (no fmt):");
    for concurrency in [1, 4, 8, 16, 32] {
        print!(
            "  {concurrency:>2}={:>5.2}",
            millions(median(|| tracing_rate(concurrency, tracing_registry)))
        );
    }
    println!();
    print!("  fmt (formats span):");
    for concurrency in [1, 4, 8, 16, 32] {
        print!(
            "  {concurrency:>2}={:>5.2}",
            millions(median(|| tracing_rate(concurrency, tracing_fmt)))
        );
    }
    println!("\n");

    println!(
        "# crossover scan — span-API ingest throughput (M spans/s), median of {REPS}, host cores={cores}\n"
    );
    println!(
        "  {:>5} {:>9} {:>9} {:>9} {:>9} {:>9}   leader (best proxima)",
        "W", "proxima", "prox/yld", "trc/reg", "trc/fmt", "otel"
    );
    for concurrency in CONCURRENCY {
        let proxima = median(|| proxima_rate(concurrency, 1, false));
        let polite = median(|| proxima_rate(concurrency, 1, true));
        let tracing_reg = median(|| tracing_rate(concurrency, tracing_registry));
        let tracing = median(|| tracing_rate(concurrency, tracing_fmt));
        let otel = median(|| otel_rate(concurrency));
        let best_proxima = proxima.max(polite);
        let best_tracing = tracing_reg.max(tracing);
        let leader = if best_proxima >= best_tracing && best_proxima >= otel {
            "proxima"
        } else if best_tracing >= otel {
            "tracing wins"
        } else {
            "otel wins"
        };
        println!(
            "  {concurrency:>5} {:>9.2} {:>9.2} {:>9.2} {:>9.2} {:>9.2}   {leader}",
            millions(proxima),
            millions(polite),
            millions(tracing_reg),
            millions(tracing),
            millions(otel)
        );
    }
    println!("\nW = emit concurrency (blocking tasks). proxima core_count = W (per-core rings).");
    println!("trc/reg = tracing bare registry (no fmt); trc/fmt = tracing fmt subscriber.");
}
