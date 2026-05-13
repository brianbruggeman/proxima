#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

//! `CaptureContext` attach + drain microbench. Measures the per-call
//! sidecar cost a wrapping `RecordUpstream` pays when the chain is
//! being recorded:
//!
//! 1. `attach_single_field` — one (key, value) attach (the common
//!    case: a single trace_id per request).
//! 2. `attach_eight_fields_then_drain` — pathological case for a
//!    chain that fans out into many sub-calls each annotating
//!    metadata.
//! 3. `drain_empty` — a Pipe that recorded nothing pays only the
//!    empty-drain cost.

use std::time::Duration;

use bytes::Bytes;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use proxima::capture_surface::CaptureContext;
use proxima::recording::LiveCaptureContext;

fn attach_single_field(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("capture_attach_single_field");
    group.measurement_time(Duration::from_secs(3));
    group.throughput(Throughput::Elements(1));
    group.bench_function("attach_then_drain", |bencher| {
        bencher.iter(|| {
            let capture = LiveCaptureContext::new();
            capture.attach("trace_id", Bytes::from_static(b"01ARZ"));
            std::hint::black_box(capture.drain());
        });
    });
    group.finish();
}

fn attach_eight_fields_then_drain(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("capture_attach_eight_fields");
    group.measurement_time(Duration::from_secs(3));
    group.throughput(Throughput::Elements(1));
    let fields: [(&str, &[u8]); 8] = [
        ("trace_id", b"01ARZ"),
        ("span_id", b"span-9876"),
        ("pipe", b"echo"),
        ("upstream", b"origin"),
        ("status", b"200"),
        ("region", b"us-east-1"),
        ("tier", b"premium"),
        ("retry", b"0"),
    ];
    group.bench_function("attach_8_drain", |bencher| {
        bencher.iter(|| {
            let capture = LiveCaptureContext::new();
            for (key, value) in fields {
                capture.attach(key, Bytes::copy_from_slice(value));
            }
            std::hint::black_box(capture.drain());
        });
    });
    group.finish();
}

fn drain_empty(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("capture_drain_empty");
    group.measurement_time(Duration::from_secs(3));
    group.throughput(Throughput::Elements(1));
    group.bench_function("drain_no_attach", |bencher| {
        bencher.iter(|| {
            let capture = LiveCaptureContext::new();
            std::hint::black_box(capture.drain());
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    attach_single_field,
    attach_eight_fields_then_drain,
    drain_empty
);
criterion_main!(benches);
