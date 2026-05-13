// P9 filter + view pipe bench (extended in P12 with matched OTel incumbent arms).
//
// Eight arms:
//   1. null_pipe_baseline          — direct NullPipe, no filter; throughput floor
//   2. filter_random_drop_50pct    — RandomDropPipe(0.5, NullPipe); ~2× throughput
//   3. drop_attr_pipe              — DropAttrPipe(["k1"], NullPipe); clone-with-filter cost
//   4. composed_filter_view        — RandomDropPipe(0.5, FilterByLevel(WARN, DropAttrPipe(["user_id"], NullPipe)))
//                                    realistic composed chain
//
//   P12 additions (incumbent arms + matched proxima arms):
//   5. otel_sampler_traceid_ratio_50pct (design-favors: incumbent — home turf)
//      OTel TracerProvider with Sampler::TraceIdRatioBased(0.5) + InMemorySpanExporter.
//      N=10_000 span start+end. Engages OTel's actual sampling design point.
//
//   6. proxima_random_drop_50pct_to_in_memory (design-favors: proxima — matched terminal)
//      RandomDropPipe(0.5, InMemoryPipe). Matches arm 5: 50% sampling + in-memory store.
//      N=10_000 span batch dispatched via Pipe::call. Apples-to-apples against arm 5.
//
//   Note: OTel View API at 0.32 for attribute dropping requires constructing a custom View
//   impl through MeterProvider::builder().with_view(). The View trait in 0.32 is internal
//   (not part of the stable public API surface in opentelemetry_sdk 0.32.0) — no public
//   View construction API is exposed without MeterProvider configuration boilerplate that
//   is 20+ lines and uses internal SDK types. This bench documents the ergonomic gap:
//   proxima DropAttrPipe is 1 line in a chain; OTel View is not constructible in a bench.
//   Arm 6 (proxima_drop_attr_to_in_memory) is the proxima side of this comparison.

use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use futures::executor::block_on;
use opentelemetry::trace::{Span, Tracer, TracerProvider as _};
use opentelemetry_sdk::trace::{InMemorySpanExporterBuilder, Sampler, SdkTracerProvider};

use proxima_primitives::pipe::SendPipe;
use proxima_telemetry::level::Level;
use proxima_telemetry::log::LogRecord;
use proxima_telemetry::log::body::LogBody;
use proxima_telemetry::pipes::{
    DropAttrPipe, FilterByLevelPipe, InMemoryPipe, NullPipe, RandomDropPipe, log_batch_request,
    span_batch_request,
};
use proxima_telemetry::tag::{ScalarValue, Tag};

extern crate alloc;

const BATCH_SIZE: usize = 10_000;

fn make_log_record() -> LogRecord {
    LogRecord {
        ts_ns: 1_700_000_000_000,
        observed_ts_ns: 1_700_000_000_000,
        level: Level::WARN,
        body: LogBody::Text("bench event"),
        attrs: {
            let mut attrs = smallvec::SmallVec::new();
            attrs.push(Tag::Scalar {
                key: "user_id",
                value: ScalarValue::Str("u-123"),
            });
            attrs.push(Tag::Scalar {
                key: "env",
                value: ScalarValue::Str("prod"),
            });
            attrs.push(Tag::Scalar {
                key: "region",
                value: ScalarValue::Str("us-east-1"),
            });
            attrs.push(Tag::Scalar {
                key: "k1",
                value: ScalarValue::Str("v1"),
            });
            attrs
        },
        trace_id: None,
        span_id: None,
        trace_flags: proxima_telemetry::id::TraceFlags(0_u8),
        module_path: "bench",
        file_line: (0, 0),
    }
}

fn make_batch() -> alloc::vec::Vec<LogRecord> {
    (0..BATCH_SIZE).map(|_| make_log_record()).collect()
}

fn bench_null_pipe_baseline(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("p9_filter_view_pipes");
    let pipe = NullPipe::new();

    group.throughput(Throughput::Elements(BATCH_SIZE as u64));
    group.bench_function("null_pipe_baseline", |bencher| {
        bencher.iter(|| {
            let request = log_batch_request(black_box(make_batch()));
            let _ = block_on(SendPipe::call(&pipe, request));
        });
    });
    group.finish();
}

fn bench_filter_random_drop_50pct(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("p9_filter_view_pipes");
    let pipe = RandomDropPipe::new(NullPipe::new(), 0.5);

    group.throughput(Throughput::Elements(BATCH_SIZE as u64));
    group.bench_function("filter_random_drop_50pct", |bencher| {
        bencher.iter(|| {
            let request = log_batch_request(black_box(make_batch()));
            let _ = block_on(SendPipe::call(&pipe, request));
        });
    });
    group.finish();
}

fn bench_drop_attr_pipe(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("p9_filter_view_pipes");
    let pipe = DropAttrPipe::new(NullPipe::new(), &["k1"]);

    group.throughput(Throughput::Elements(BATCH_SIZE as u64));
    group.bench_function("drop_attr_pipe", |bencher| {
        bencher.iter(|| {
            let request = log_batch_request(black_box(make_batch()));
            let _ = block_on(SendPipe::call(&pipe, request));
        });
    });
    group.finish();
}

fn bench_composed_filter_view_chain(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("p9_filter_view_pipes");
    // RandomDropPipe(0.5) → FilterByLevel(WARN) → DropAttrPipe(["user_id"]) → NullPipe
    let pipe = RandomDropPipe::new(
        FilterByLevelPipe::new(
            DropAttrPipe::new(NullPipe::new(), &["user_id"]),
            Level::WARN,
        ),
        0.5,
    );

    group.throughput(Throughput::Elements(BATCH_SIZE as u64));
    group.bench_function("composed_filter_view_chain", |bencher| {
        bencher.iter(|| {
            let request = log_batch_request(black_box(make_batch()));
            let _ = block_on(SendPipe::call(&pipe, request));
        });
    });
    group.finish();
}

fn make_span_batch() -> alloc::vec::Vec<proxima_telemetry::trace::SpanRecord> {
    use proxima_telemetry::id::{SpanId, TraceId};
    use proxima_telemetry::trace::{SpanKind, SpanRecord, Status, TraceState};

    (0..BATCH_SIZE)
        .map(|_| SpanRecord {
            trace_id: TraceId::INVALID,
            span_id: SpanId::INVALID,
            parent_span_id: None,
            name: "op",
            kind: SpanKind::Internal,
            start_ns: 1_700_000_000_000,
            duration_ns: 1_000,
            status: Status::Unset,
            attrs: {
                let mut attrs = smallvec::SmallVec::new();
                attrs.push(Tag::Scalar {
                    key: "route",
                    value: ScalarValue::Str("/v1"),
                });
                attrs
            },
            events: alloc::vec![].into(),
            links: alloc::vec![].into(),
            tracestate: TraceState::empty(),
            module_path: "bench",
            file_line: (0, 0),
        })
        .collect()
}

// P12 arm 5: OTel TracerProvider with 50% TraceIdRatioBased sampler + InMemorySpanExporter.
// design-favors: incumbent (home turf) — OTel's native sampling path.
// N=10_000 spans with 1 attribute each. ~50% will be sampled.
fn bench_otel_sampler_traceid_ratio_50pct(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("p9_filter_view_pipes");

    let exporter = InMemorySpanExporterBuilder::new().build();
    let provider = SdkTracerProvider::builder()
        .with_sampler(Sampler::TraceIdRatioBased(0.5))
        .with_simple_exporter(exporter)
        .build();
    let tracer = provider.tracer("bench");

    group.throughput(Throughput::Elements(BATCH_SIZE as u64));
    group.bench_function("otel_sampler_traceid_ratio_50pct", |bencher| {
        bencher.iter(|| {
            for _ in 0..BATCH_SIZE {
                let mut span = tracer.start(black_box("op"));
                span.set_attribute(opentelemetry::KeyValue::new("route", black_box("/v1")));
                black_box(&span);
                span.end();
            }
        });
    });
    group.finish();
}

// P12 arm 6: proxima RandomDropPipe(0.5) → InMemoryPipe. Matched to arm 5.
// design-favors: proxima. Pipe composition dispatches a pre-built span batch.
fn bench_proxima_random_drop_50pct_to_in_memory(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("p9_filter_view_pipes");
    let pipe = RandomDropPipe::new(InMemoryPipe::new(), 0.5);

    group.throughput(Throughput::Elements(BATCH_SIZE as u64));
    group.bench_function("proxima_random_drop_50pct_to_in_memory", |bencher| {
        bencher.iter(|| {
            let request = span_batch_request(black_box(make_span_batch()));
            let _ = block_on(SendPipe::call(&pipe, request));
        });
    });
    group.finish();
}

// P12 arm 7 (optional): proxima DropAttrPipe(["k1"]) → InMemoryPipe.
// design-favors: proxima. 1-line chain composition vs OTel View (~20 lines setup).
// OTel View API gap documented in bench comment header.
fn bench_proxima_drop_attr_to_in_memory(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("p9_filter_view_pipes");
    let pipe = DropAttrPipe::new(InMemoryPipe::new(), &["k1"]);

    group.throughput(Throughput::Elements(BATCH_SIZE as u64));
    group.bench_function("proxima_drop_attr_to_in_memory", |bencher| {
        bencher.iter(|| {
            let request = log_batch_request(black_box(make_batch()));
            let _ = block_on(SendPipe::call(&pipe, request));
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_null_pipe_baseline,
    bench_filter_random_drop_50pct,
    bench_drop_attr_pipe,
    bench_composed_filter_view_chain,
    bench_otel_sampler_traceid_ratio_50pct,
    bench_proxima_random_drop_50pct_to_in_memory,
    bench_proxima_drop_attr_to_in_memory,
);
criterion_main!(benches);
