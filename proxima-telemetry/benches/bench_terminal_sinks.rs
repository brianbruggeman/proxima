// P10 bench: matched terminal sinks — InMemoryPipe vs FormatterPipe vs OTel InMemorySpanExporter.
//
// Six arms, N=10_000 spans with 1 attribute each:
//
//  1. proxima_null_pipe          — baseline, no-op (O(1) per call)
//  2. proxima_counting_pipe      — baseline, atomic counters only
//  3. proxima_in_memory_pipe     — stores SpanRecord clones (Mutex<Vec<SpanRecord>>)
//  4. proxima_formatter_human    — formats to Human string → io::sink
//  5. proxima_formatter_json     — formats to JSON string → io::sink
//  6. otel_sdk_in_memory         — OTel TracerProvider + InMemorySpanExporter (home-turf arm)
//
// Arms 3 vs 6 is the apples-to-apples comparison: both store records in memory under a Mutex.
// Arms 4/5 compare proxima's per-record formatting cost against tracing_subscriber::fmt.
//
// Throughput unit: spans per second (N_ITEMS / iteration time).

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

use std::hint::black_box;
use std::io;
use std::sync::Arc;

use bytes::Bytes;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use opentelemetry::KeyValue;
use opentelemetry::trace::{Span, Tracer, TracerProvider as _};
use opentelemetry_sdk::trace::{InMemorySpanExporterBuilder, SdkTracerProvider};
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::request::Response;
use proxima_telemetry::pipes::{
    CountingPipe, FormatterPipe, InMemoryPipe, LogFormat, NullPipe, TelemetryRequest,
};
use proxima_telemetry::recorder::Recorder;

const N_ITEMS: usize = 10_000;

fn make_recorder_with<
    P: SendPipe<In = TelemetryRequest, Out = Response<Bytes>, Err = proxima_primitives::pipe::ProximaError>
        + Send
        + Sync
        + 'static,
>(
    pipe: P,
) -> Arc<Recorder> {
    Arc::new(
        Recorder::builder()
            .pipe(pipe)
            .core_count(1)
            .start()
            .expect("recorder build"),
    )
}

fn emit_n_spans(recorder: &Arc<Recorder>) {
    for _ in 0..N_ITEMS {
        let guard = recorder
            .span(black_box("process"))
            .tag("route", black_box("/v1"))
            .start();
        black_box(&guard);
        drop(guard);
    }
    recorder.drain();
}

fn proxima_null_pipe(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("bench_terminal_sinks");
    group.throughput(Throughput::Elements(N_ITEMS as u64));

    let recorder = make_recorder_with(NullPipe::new());

    group.bench_function(BenchmarkId::new("proxima_null_pipe", N_ITEMS), |bencher| {
        bencher.iter(|| emit_n_spans(&recorder));
    });
    group.finish();
}

fn proxima_counting_pipe(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("bench_terminal_sinks");
    group.throughput(Throughput::Elements(N_ITEMS as u64));

    let (pipe, _spans, _events, _logs, _metrics, _links) = CountingPipe::new();
    let recorder = make_recorder_with(pipe);

    group.bench_function(
        BenchmarkId::new("proxima_counting_pipe", N_ITEMS),
        |bencher| {
            bencher.iter(|| emit_n_spans(&recorder));
        },
    );
    group.finish();
}

fn proxima_in_memory_pipe(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("bench_terminal_sinks");
    group.throughput(Throughput::Elements(N_ITEMS as u64));

    let pipe = InMemoryPipe::new();
    let recorder = make_recorder_with(pipe);

    group.bench_function(
        BenchmarkId::new("proxima_in_memory_pipe", N_ITEMS),
        |bencher| {
            bencher.iter(|| emit_n_spans(&recorder));
        },
    );
    group.finish();
}

fn proxima_formatter_human(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("bench_terminal_sinks");
    group.throughput(Throughput::Elements(N_ITEMS as u64));

    let pipe = FormatterPipe::new(io::sink(), LogFormat::Human);
    let recorder = make_recorder_with(pipe);

    group.bench_function(
        BenchmarkId::new("proxima_formatter_human", N_ITEMS),
        |bencher| {
            bencher.iter(|| emit_n_spans(&recorder));
        },
    );
    group.finish();
}

fn proxima_formatter_json(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("bench_terminal_sinks");
    group.throughput(Throughput::Elements(N_ITEMS as u64));

    let pipe = FormatterPipe::new(io::sink(), LogFormat::Json);
    let recorder = make_recorder_with(pipe);

    group.bench_function(
        BenchmarkId::new("proxima_formatter_json", N_ITEMS),
        |bencher| {
            bencher.iter(|| emit_n_spans(&recorder));
        },
    );
    group.finish();
}

fn otel_sdk_in_memory(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("bench_terminal_sinks");
    group.throughput(Throughput::Elements(N_ITEMS as u64));

    let exporter = InMemorySpanExporterBuilder::new().build();
    let provider = SdkTracerProvider::builder()
        .with_simple_exporter(exporter)
        .build();
    let tracer = provider.tracer("bench");

    group.bench_function(BenchmarkId::new("otel_sdk_in_memory", N_ITEMS), |bencher| {
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

criterion_group!(
    benches,
    proxima_null_pipe,
    proxima_counting_pipe,
    proxima_in_memory_pipe,
    proxima_formatter_human,
    proxima_formatter_json,
    otel_sdk_in_memory,
);
criterion_main!(benches);
