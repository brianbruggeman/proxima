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

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use proxima_telemetry::pipes::NullPipe;
use proxima_telemetry::recorder::Recorder;

fn make_recorder() -> Recorder {
    Recorder::builder()
        .pipe(NullPipe::new())
        .core_count(1)
        .start()
        .expect("recorder build failed in bench")
}

/// Arm 1: typical workload — 3 tags + 2 events per span.
/// All inline with SmallVec<[Tag; 4]> / SmallVec<[EventRecord; 2]>:
/// zero heap allocations after P8 opt-sweep.
fn bench_emit_span_with_3_attrs_2_events(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("p8_record_alloc");
    let recorder = make_recorder();

    group.bench_function("emit_span_with_3_attrs_2_events", |bencher| {
        bencher.iter_batched(
            || recorder.drain(),
            |_| {
                let mut guard = recorder
                    .span(black_box("bench_span"))
                    .tag("http.method", black_box("GET"))
                    .tag("http.status_code", black_box(200u64))
                    .tag("http.route", black_box("/v1/data"))
                    .start();
                if let Some(event) = guard.event("request.received") {
                    event.tag("latency_us", black_box(42u64)).emit();
                }
                if let Some(event) = guard.event("response.sent") {
                    event.tag("bytes", black_box(1024u64)).emit();
                }
                black_box(&guard);
                drop(guard);
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

/// Arm 2: overflow workload — 8 tags + 4 events per span.
/// Overflows the SmallVec inline capacity; falls back to heap.
/// Should be ≈ equal to the pre-opt Vec baseline (no regression).
fn bench_emit_span_with_8_attrs_4_events(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("p8_record_alloc");
    let recorder = make_recorder();

    group.bench_function("emit_span_with_8_attrs_4_events", |bencher| {
        bencher.iter_batched(
            || recorder.drain(),
            |_| {
                let mut guard = recorder
                    .span(black_box("bench_span"))
                    .tag("http.method", black_box("GET"))
                    .tag("http.status_code", black_box(200u64))
                    .tag("http.route", black_box("/v1/data"))
                    .tag("http.host", black_box("example.com"))
                    .tag("http.scheme", black_box("https"))
                    .tag("net.peer.ip", black_box("10.0.0.1"))
                    .tag("net.peer.port", black_box(443u64))
                    .tag("span.kind", black_box("client"))
                    .start();
                if let Some(event) = guard.event("e0") {
                    event.tag("seq", black_box(0u64)).emit();
                }
                if let Some(event) = guard.event("e1") {
                    event.tag("seq", black_box(1u64)).emit();
                }
                if let Some(event) = guard.event("e2") {
                    event.tag("seq", black_box(2u64)).emit();
                }
                if let Some(event) = guard.event("e3") {
                    event.tag("seq", black_box(3u64)).emit();
                }
                black_box(&guard);
                drop(guard);
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_emit_span_with_3_attrs_2_events,
    bench_emit_span_with_8_attrs_4_events,
);
criterion_main!(benches);
