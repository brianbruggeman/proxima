#![allow(clippy::unwrap_used, clippy::expect_used)]

//! Load throughput under concurrent producers: proxima Recorder vs the two
//! incumbents the Rust world actually reaches for — the `tracing` crate (what
//! tokio + most of the ecosystem use) and the OpenTelemetry SDK.
//!
//! `T` producer threads each emit `SPANS_PER_THREAD` spans; throughput is total
//! spans / wall time. proxima uses `core_count = T` so each producer owns its
//! own per-core ring (the safe SPSC envelope — emitting concurrency must stay ≤
//! core_count). `tracing` dispatches every span to one global subscriber
//! (fmt → io::sink, the real-world setup minus terminal I/O). OTel shares one
//! TracerProvider + InMemorySpanExporter. Each incumbent runs on its own
//! concurrency design point.
//!
//! This measures INGEST under contention. The drain ceiling, drop/saturation
//! edges, and memory-under-soak live in `benches/bench_trace_soak.rs`.

use std::hint::black_box;
use std::io;
use std::sync::Arc;
use std::sync::OnceLock;
use std::thread;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use opentelemetry::KeyValue;
use opentelemetry::trace::{Span, Tracer, TracerProvider as _};
use opentelemetry_sdk::trace::{InMemorySpanExporterBuilder, SdkTracerProvider};
use proxima_telemetry::pipes::InMemoryPipe;
use proxima_telemetry::recorder::Recorder;
use tracing_subscriber::fmt::format::FmtSpan;

const SPANS_PER_THREAD: usize = 50_000;
const THREAD_COUNTS: [usize; 4] = [1, 2, 4, 8];

fn spawn_join<MakeWork, Work>(threads: usize, make: MakeWork)
where
    MakeWork: Fn() -> Work,
    Work: FnOnce() + Send + 'static,
{
    let handles: Vec<_> = (0..threads).map(|_| thread::spawn(make())).collect();
    for handle in handles {
        handle.join().expect("producer panicked");
    }
}

fn bench_proxima(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("trace_load_proxima");
    for threads in THREAD_COUNTS {
        group.throughput(Throughput::Elements((threads * SPANS_PER_THREAD) as u64));
        let recorder = Arc::new(
            Recorder::builder()
                .pipe(InMemoryPipe::new())
                .core_count(threads)
                .start()
                .expect("recorder"),
        );
        group.bench_with_input(
            BenchmarkId::from_parameter(threads),
            &threads,
            |bencher, &threads| {
                bencher.iter(|| {
                    spawn_join(threads, || {
                        let recorder = Arc::clone(&recorder);
                        move || {
                            for _ in 0..SPANS_PER_THREAD {
                                let guard = recorder
                                    .span(black_box("process"))
                                    .tag("route", black_box("/v1"))
                                    .start();
                                black_box(&guard);
                                drop(guard);
                            }
                        }
                    });
                    while recorder.drain() > 0 {}
                });
            },
        );
    }
    group.finish();
}

fn install_tracing() {
    static INIT: OnceLock<()> = OnceLock::new();
    INIT.get_or_init(|| {
        let subscriber = tracing_subscriber::fmt()
            .with_writer(io::sink)
            .with_span_events(FmtSpan::CLOSE)
            .finish();
        let _ = tracing::subscriber::set_global_default(subscriber);
    });
}

fn bench_tracing(criterion: &mut Criterion) {
    install_tracing();
    let mut group = criterion.benchmark_group("trace_load_tracing");
    for threads in THREAD_COUNTS {
        group.throughput(Throughput::Elements((threads * SPANS_PER_THREAD) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(threads),
            &threads,
            |bencher, &threads| {
                bencher.iter(|| {
                    spawn_join(threads, || {
                        move || {
                            for _ in 0..SPANS_PER_THREAD {
                                let span =
                                    tracing::span!(tracing::Level::INFO, "process", route = "/v1");
                                let _entered = span.enter();
                            }
                        }
                    });
                });
            },
        );
    }
    group.finish();
}

fn bench_otel(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("trace_load_otel");
    for threads in THREAD_COUNTS {
        group.throughput(Throughput::Elements((threads * SPANS_PER_THREAD) as u64));
        let exporter = InMemorySpanExporterBuilder::new().build();
        let provider = Arc::new(
            SdkTracerProvider::builder()
                .with_simple_exporter(exporter)
                .build(),
        );
        group.bench_with_input(
            BenchmarkId::from_parameter(threads),
            &threads,
            |bencher, &threads| {
                bencher.iter(|| {
                    spawn_join(threads, || {
                        let provider = Arc::clone(&provider);
                        move || {
                            let tracer = provider.tracer("bench");
                            for _ in 0..SPANS_PER_THREAD {
                                let mut span = tracer.start(black_box("process"));
                                span.set_attribute(KeyValue::new("route", black_box("/v1")));
                                black_box(&span);
                                span.end();
                            }
                        }
                    });
                });
            },
        );
    }
    group.finish();
}

criterion_group!(trace_load, bench_proxima, bench_tracing, bench_otel);
criterion_main!(trace_load);
