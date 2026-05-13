#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

//! P5 bench: TracingLayer adapter throughput vs `tracing_subscriber::fmt`.
//!
//! Three arms, 10 000 events per iteration (matches bench_request_context.rs):
//!
//! 1. `proxima_tracing_bridge` (design-favors: proxima) — installs TracingLayer
//!    backed by a CountingPipe Recorder. Drives `tracing::info!(field=value)` × N.
//!    Drain at the end to flush the ring.
//!
//! 2. `tracing_subscriber_fmt` (design-favors: incumbent / home turf) — installs
//!    `registry().with(EnvFilter).with(fmt::layer().with_writer(io::sink))`. Same
//!    `tracing::info!` × N. No drain needed (fmt layer is synchronous).
//!
//! 3. `proxima_tracing_bridge_nodrain` (design-favors: proxima) — same as arm 1
//!    but measures only the per-event push (no drain) to isolate the Layer → ring
//!    path from the drain pass overhead.
//!
//! Throughput unit: events per second (N_ITEMS / iteration time).

use std::sync::Arc;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use proxima_telemetry::pipes::CountingPipe;
use proxima_telemetry::recorder::Recorder;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::layer::SubscriberExt;

const N_ITEMS: usize = 10_000;

// design-favors: proxima — TracingLayer → per-core ring → drain → CountingPipe.
fn proxima_tracing_bridge(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("proxima_tracing_bridge");
    group.throughput(Throughput::Elements(N_ITEMS as u64));

    let (pipe, _spans, _events, _logs, _metrics, _links) = CountingPipe::new();
    let recorder = Arc::new(
        Recorder::builder()
            .pipe(pipe)
            .core_count(1)
            .start()
            .expect("recorder build"),
    );

    let layer = proxima_telemetry::tracing_bridge::TracingLayer::new(Arc::clone(&recorder));
    let filter = EnvFilter::new("info");
    let subscriber = tracing_subscriber::registry().with(filter).with(layer);

    // register once per bench session
    let _guard = tracing::subscriber::set_default(subscriber);

    group.bench_function(BenchmarkId::from_parameter(N_ITEMS), |bencher| {
        bencher.iter(|| {
            for index in 0..N_ITEMS {
                tracing::info!(index, label = "bench", "bridge event");
            }
            recorder.drain();
        });
    });
    group.finish();
}

// design-favors: incumbent / home turf — tracing_subscriber fmt layer to io::sink.
fn tracing_subscriber_fmt(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("tracing_subscriber_fmt");
    group.throughput(Throughput::Elements(N_ITEMS as u64));

    let filter = EnvFilter::new("info");
    let layer = tracing_subscriber::fmt::layer()
        .with_writer(std::io::sink)
        .with_target(false);
    let subscriber = tracing_subscriber::registry().with(filter).with(layer);
    let _guard = tracing::subscriber::set_default(subscriber);

    group.bench_function(BenchmarkId::from_parameter(N_ITEMS), |bencher| {
        bencher.iter(|| {
            for index in 0..N_ITEMS {
                tracing::info!(index, label = "bench", "fmt event");
            }
        });
    });
    group.finish();
}

// design-favors: proxima — emit path only, no drain (isolates ring-push from drain).
fn proxima_tracing_bridge_nodrain(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("proxima_tracing_bridge_nodrain");
    group.throughput(Throughput::Elements(N_ITEMS as u64));

    let (pipe, _spans, _events, _logs, _metrics, _links) = CountingPipe::new();
    let recorder = Arc::new(
        Recorder::builder()
            .pipe(pipe)
            .core_count(1)
            .start()
            .expect("recorder build"),
    );

    let layer = proxima_telemetry::tracing_bridge::TracingLayer::new(Arc::clone(&recorder));
    let filter = EnvFilter::new("info");
    let subscriber = tracing_subscriber::registry().with(filter).with(layer);
    let _guard = tracing::subscriber::set_default(subscriber);

    group.bench_function(BenchmarkId::from_parameter(N_ITEMS), |bencher| {
        bencher.iter(|| {
            for index in 0..N_ITEMS {
                tracing::info!(index, label = "bench", "nodrain event");
            }
            std::hint::black_box(&recorder);
        });
    });
    group.finish();
}

// design-favors: proxima — span open+close lifecycle through TracingLayer → ring → drain.
// 10_000 tracing::span! + enter + drop per iteration.
fn proxima_tracing_bridge_span_emit(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("proxima_tracing_bridge_span_emit");
    group.throughput(Throughput::Elements(N_ITEMS as u64));

    let (pipe, _spans, _events, _logs, _metrics, _links) = CountingPipe::new();
    let recorder = Arc::new(
        Recorder::builder()
            .pipe(pipe)
            .core_count(1)
            .start()
            .expect("recorder build"),
    );

    let layer = proxima_telemetry::tracing_bridge::TracingLayer::new(Arc::clone(&recorder));
    let filter = EnvFilter::new("info");
    let subscriber = tracing_subscriber::registry().with(filter).with(layer);
    let _guard = tracing::subscriber::set_default(subscriber);

    group.bench_function(BenchmarkId::from_parameter(N_ITEMS), |bencher| {
        bencher.iter(|| {
            for _ in 0..N_ITEMS {
                let span = tracing::span!(tracing::Level::INFO, "bench_span", key = "v");
                let _entered = span.entered();
            }
            recorder.drain();
        });
    });
    group.finish();
}

// design-favors: incumbent / home turf — plain tracing_subscriber registry + fmt layer to sink.
// same span open+close workload, no proxima, measures tracing_subscriber's native span lifecycle.
fn tracing_subscriber_span_emit(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("tracing_subscriber_span_emit");
    group.throughput(Throughput::Elements(N_ITEMS as u64));

    let filter = EnvFilter::new("info");
    let layer = tracing_subscriber::fmt::layer()
        .with_writer(std::io::sink)
        .with_target(false);
    let subscriber = tracing_subscriber::registry().with(filter).with(layer);
    let _guard = tracing::subscriber::set_default(subscriber);

    group.bench_function(BenchmarkId::from_parameter(N_ITEMS), |bencher| {
        bencher.iter(|| {
            for _ in 0..N_ITEMS {
                let span = tracing::span!(tracing::Level::INFO, "bench_span", key = "v");
                let _entered = span.entered();
            }
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    proxima_tracing_bridge,
    tracing_subscriber_fmt,
    proxima_tracing_bridge_nodrain,
    proxima_tracing_bridge_span_emit,
    tracing_subscriber_span_emit,
);
criterion_main!(benches);
