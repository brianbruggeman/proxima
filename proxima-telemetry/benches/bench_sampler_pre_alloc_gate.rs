// P16 bench: at-creation Sampler pre-allocation gate vs OTel TraceIdRatioBased.
//
// Four arms, N=10_000 span attempts per iteration:
//
//  1. proxima_no_sampler_baseline   (design-favors: proxima — no gate)
//     Recorder with no sampler (AlwaysOn default). Emit 10k spans, drain.
//     The throughput floor for fully-through record construction + ring push.
//
//  2. proxima_sampler_alwaysoff     (design-favors: proxima — pure gate cost)
//     Recorder with AlwaysOff sampler. 10k span attempts; all gated before
//     any SpanRecord state is allocated. Measures the cost of the gate itself
//     plus the noop guard drop path. Nothing reaches the ring.
//
//  3. proxima_sampler_ratio_50pct   (design-favors: matched workload)
//     Recorder with TraceIdRatioBased(0.5). 10k spans; ~50% gated before
//     allocation. Matches the same sampling fraction as arm 4. This is the
//     proxima equivalent of OTel's TraceIdRatioBased — apples-to-apples.
//
//  4. otel_sampler_ratio_50pct      (design-favors: incumbent — HOME TURF)
//     OTel TracerProvider with Sampler::TraceIdRatioBased(0.5) +
//     InMemorySpanExporter. 10k tracer.start() + set_attribute + span.end().
//     Same arm as the OTel arm in bench_filter_view_pipes (P12), but now we
//     are comparing apples-to-apples: both are at-creation gates.
//
// Arms 3 vs 4 is the load-bearing verdict: did proxima close the 2.1× gap
// from P12's RandomDropPipe-vs-OTel comparison?

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
use std::sync::Arc;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use opentelemetry::trace::{Span, Tracer, TracerProvider as _};
use opentelemetry_sdk::trace::{InMemorySpanExporterBuilder, Sampler, SdkTracerProvider};

use proxima_telemetry::pipes::NullPipe;
use proxima_telemetry::recorder::Recorder;
use proxima_telemetry::sampler::{AlwaysOff, TraceIdRatioBased};

const N_ITEMS: usize = 10_000;

fn make_recorder_no_sampler() -> Arc<Recorder> {
    Arc::new(
        Recorder::builder()
            .pipe(NullPipe::new())
            .core_count(1)
            .start()
            .expect("recorder build"),
    )
}

fn make_recorder_with_sampler<S: proxima_telemetry::sampler::Sampler + 'static>(
    sampler: S,
) -> Arc<Recorder> {
    Arc::new(
        Recorder::builder()
            .pipe(NullPipe::new())
            .core_count(1)
            .sampler(sampler)
            .start()
            .expect("recorder build"),
    )
}

fn proxima_no_sampler_baseline(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("p16_sampler_gate");
    group.throughput(Throughput::Elements(N_ITEMS as u64));

    let recorder = make_recorder_no_sampler();

    group.bench_function("proxima_no_sampler_baseline", |bencher| {
        bencher.iter(|| {
            for _ in 0..N_ITEMS {
                let guard = recorder.span(black_box("process")).start();
                black_box(&guard);
                drop(guard);
            }
            recorder.drain();
        });
    });
    group.finish();
}

fn proxima_sampler_alwaysoff(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("p16_sampler_gate");
    group.throughput(Throughput::Elements(N_ITEMS as u64));

    let recorder = make_recorder_with_sampler(AlwaysOff);

    group.bench_function("proxima_sampler_alwaysoff", |bencher| {
        bencher.iter(|| {
            for _ in 0..N_ITEMS {
                let guard = recorder.span(black_box("process")).start();
                black_box(&guard);
                drop(guard);
            }
            // drain is still called: measures baseline drain on empty rings
            recorder.drain();
        });
    });
    group.finish();
}

fn proxima_sampler_ratio_50pct(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("p16_sampler_gate");
    group.throughput(Throughput::Elements(N_ITEMS as u64));

    let recorder = make_recorder_with_sampler(TraceIdRatioBased::new(0.5));

    group.bench_function("proxima_sampler_ratio_50pct", |bencher| {
        bencher.iter(|| {
            for _ in 0..N_ITEMS {
                let guard = recorder.span(black_box("process")).start();
                black_box(&guard);
                drop(guard);
            }
            recorder.drain();
        });
    });
    group.finish();
}

fn otel_sampler_ratio_50pct(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("p16_sampler_gate");
    group.throughput(Throughput::Elements(N_ITEMS as u64));

    let exporter = InMemorySpanExporterBuilder::new().build();
    let provider = SdkTracerProvider::builder()
        .with_sampler(Sampler::TraceIdRatioBased(0.5))
        .with_simple_exporter(exporter)
        .build();
    let tracer = provider.tracer("bench");

    group.bench_function("otel_sampler_ratio_50pct", |bencher| {
        bencher.iter(|| {
            for _ in 0..N_ITEMS {
                let mut span = tracer.start(black_box("process"));
                span.set_attribute(opentelemetry::KeyValue::new("route", black_box("/v1")));
                black_box(&span);
                span.end();
            }
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    proxima_no_sampler_baseline,
    proxima_sampler_alwaysoff,
    proxima_sampler_ratio_50pct,
    otel_sampler_ratio_50pct,
);
criterion_main!(benches);
