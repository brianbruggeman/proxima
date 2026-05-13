#![allow(clippy::unwrap_used, clippy::expect_used)]
use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};

use proxima_primitives::pipe::header_list::HeaderList;
use proxima_telemetry::id::{SpanId, TraceId, format_traceparent, parse_traceparent};
use proxima_telemetry::propagation::{Propagation, extract, inject};

const REF_TRACEPARENT: &str = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01";
const REF_BAGGAGE: &str = "userId=alice,region=us-east,tier=gold";

fn inbound_headers() -> HeaderList {
    HeaderList::from_pairs([
        ("host", "origin.test"),
        ("traceparent", REF_TRACEPARENT),
        ("baggage", REF_BAGGAGE),
        ("content-type", "application/json"),
        ("accept", "application/json"),
    ])
}

fn bench_format_traceparent(criterion: &mut Criterion) {
    let (trace_id, span_id, flags) = parse_traceparent(REF_TRACEPARENT.as_bytes()).unwrap();
    let mut group = criterion.benchmark_group("propagation");
    group.bench_function("format_traceparent", |bencher| {
        bencher.iter(|| {
            black_box(format_traceparent(
                black_box(&trace_id),
                black_box(&span_id),
                black_box(flags),
            ))
        });
    });
    group.finish();
}

fn bench_generate(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("propagation");
    group.bench_function("generate_trace_and_span", |bencher| {
        bencher.iter(|| black_box((TraceId::generate(), SpanId::generate())));
    });
    group.finish();
}

fn bench_extract(criterion: &mut Criterion) {
    let headers = inbound_headers();
    let mut group = criterion.benchmark_group("propagation");
    group.bench_function("extract", |bencher| {
        bencher.iter(|| black_box(extract(black_box(&headers))));
    });
    group.finish();
}

fn bench_inject(criterion: &mut Criterion) {
    let propagation = extract(&inbound_headers());
    let mut group = criterion.benchmark_group("propagation");
    group.bench_function("inject", |bencher| {
        bencher.iter(|| {
            let mut outbound = HeaderList::new();
            inject(black_box(&propagation), &mut outbound);
            black_box(outbound)
        });
    });
    group.finish();
}

fn bench_round_trip(criterion: &mut Criterion) {
    let headers = inbound_headers();
    let mut group = criterion.benchmark_group("propagation");
    group.bench_function("extract_then_inject", |bencher| {
        bencher.iter(|| {
            let propagation: Propagation = extract(black_box(&headers));
            let mut outbound = HeaderList::new();
            inject(&propagation, &mut outbound);
            black_box(outbound)
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_format_traceparent,
    bench_generate,
    bench_extract,
    bench_inject,
    bench_round_trip,
);
criterion_main!(benches);
