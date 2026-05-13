#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

//! P12 re-bench: proxima Recorder vs OpenTelemetry SDK span emit, matched terminal sinks.
//!
//! Three arms, criterion `--quick` median, N=10_000 spans each:
//!
//! 1. `proxima_recorder_span_emit` (design-favors: proxima) — Recorder with InMemoryPipe.
//!    Loop: recorder.span("process").tag("route", "/v1").start() + drop.
//!    After all N iterations, drain. Both ring push and Mutex+Vec store are measured.
//!    Matched to OTel's InMemorySpanExporter (both sides lock + clone + push).
//!
//! 2. `otel_sdk_span_emit` (design-favors: incumbent — home turf) — full OTel SDK pipeline:
//!    TracerProvider + InMemorySpanExporter + Tracer::start + set_attribute + end.
//!    Engages OTel's actual design point: sampling decisions, attribute set construction,
//!    full exporter pipeline (in-memory, no I/O).
//!
//! 3. `proxima_recorder_span_emit_nodrain` — same as arm 1 but without the drain step,
//!    to isolate ring-push cost from the drain pass.
//!
//! P6 used CountingPipe (atomic only) on the proxima side — mismatched. P12 fixes that
//! by using InMemoryPipe so both arms store records. The honest verdict replaces the P6
//! 2.34× claim.
//!
//! Throughput unit: spans per second (N_ITEMS / iteration time).

use std::hint::black_box;
use std::sync::Arc;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use opentelemetry::KeyValue;
use opentelemetry::trace::{Span, Tracer, TracerProvider as _};
use opentelemetry_sdk::trace::{InMemorySpanExporterBuilder, SdkTracerProvider};
use proxima_telemetry::pipes::InMemoryPipe;
use proxima_telemetry::recorder::Recorder;

const N_ITEMS: usize = 10_000;

fn proxima_recorder_span_emit(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("proxima_recorder_span_emit");
    group.throughput(Throughput::Elements(N_ITEMS as u64));

    let pipe = InMemoryPipe::new();
    let recorder = Arc::new(
        Recorder::builder()
            .pipe(pipe)
            .core_count(1)
            // hold the whole working set: OTel's InMemorySpanExporter buffers all
            // N spans, so a ring smaller than N would make proxima (and ONLY
            // proxima) pay backpressure here — an unfair emit comparison. size the
            // ring to the workload so this measures emit, not overflow.
            .ring_capacity((N_ITEMS * 2).next_power_of_two())
            // realistic config: a background pump overlaps the export with emit,
            // exactly how proxima is run. OTel's SimpleSpanProcessor exports
            // inline per span; proxima decouples emit from export via the ring.
            .managed_drainer(true)
            .start()
            .expect("recorder build"),
    );

    group.bench_function(BenchmarkId::from_parameter(N_ITEMS), |bencher| {
        bencher.iter(|| {
            for _ in 0..N_ITEMS {
                let guard = recorder
                    .span(black_box("process"))
                    .tag("route", black_box("/v1"))
                    .start();
                black_box(&guard);
                drop(guard);
            }
            recorder.drain();
        });
    });
    group.finish();
}

// design-favors: incumbent (home turf) — full OTel TracerProvider + InMemorySpanExporter.
fn otel_sdk_span_emit(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("otel_sdk_span_emit");
    group.throughput(Throughput::Elements(N_ITEMS as u64));

    let exporter = InMemorySpanExporterBuilder::new().build();
    let provider = SdkTracerProvider::builder()
        .with_simple_exporter(exporter)
        .build();
    let tracer = provider.tracer("bench");

    group.bench_function(BenchmarkId::from_parameter(N_ITEMS), |bencher| {
        bencher.iter(|| {
            for _ in 0..N_ITEMS {
                let mut span = tracer.start(black_box("process"));
                span.set_attribute(KeyValue::new("route", black_box("/v1")));
                black_box(&span);
                span.end();
            }
        });
    });
    group.finish();
}

// emit-only, no drain — isolates ring-push cost from the drain pass.
fn proxima_recorder_span_emit_nodrain(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("proxima_recorder_span_emit_nodrain");
    group.throughput(Throughput::Elements(N_ITEMS as u64));

    let pipe = InMemoryPipe::new();
    let recorder = Arc::new(
        Recorder::builder()
            .pipe(pipe)
            .core_count(1)
            .ring_capacity((N_ITEMS * 2).next_power_of_two())
            .start()
            .expect("recorder build"),
    );

    group.bench_function(BenchmarkId::from_parameter(N_ITEMS), |bencher| {
        bencher.iter(|| {
            for _ in 0..N_ITEMS {
                let guard = recorder
                    .span(black_box("process"))
                    .tag("route", black_box("/v1"))
                    .start();
                black_box(&guard);
                drop(guard);
            }
            black_box(&recorder);
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    proxima_recorder_span_emit,
    otel_sdk_span_emit,
    proxima_recorder_span_emit_nodrain,
);
criterion_main!(benches);
