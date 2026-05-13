#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

//! Body envelope construction cost. The request/response body field is now a
//! plain `bytes::Bytes` (the old `Body` enum was yanked), so the envelope cost
//! collapses to `Bytes` construction: empty is a niche-zero, from_static is an
//! Arc-free static view, clone is an Arc bump.

use bytes::Bytes;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

const N_ITEMS: usize = 10_000;

// empty body — niche-zero Bytes, no allocation.
fn body_empty(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("body_empty");
    group.throughput(Throughput::Elements(N_ITEMS as u64));
    group.bench_function(BenchmarkId::from_parameter(N_ITEMS), |bencher| {
        bencher.iter(|| {
            for _ in 0..N_ITEMS {
                let body = Bytes::new();
                std::hint::black_box(body);
            }
        });
    });
    group.finish();
}

// static payload view — no allocation, no refcount.
fn body_from_static(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("body_from_static");
    group.throughput(Throughput::Elements(N_ITEMS as u64));
    group.bench_function(BenchmarkId::from_parameter(N_ITEMS), |bencher| {
        bencher.iter(|| {
            for _ in 0..N_ITEMS {
                let body = Bytes::from_static(b"telemetry-record-payload");
                std::hint::black_box(body);
            }
        });
    });
    group.finish();
}

// shared payload clone — Arc bump per envelope.
fn body_clone_shared(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("body_clone_shared");
    group.throughput(Throughput::Elements(N_ITEMS as u64));
    let payload = Bytes::copy_from_slice(b"telemetry-record-payload");
    group.bench_function(BenchmarkId::from_parameter(N_ITEMS), |bencher| {
        bencher.iter(|| {
            for _ in 0..N_ITEMS {
                let body = payload.clone();
                std::hint::black_box(body);
            }
        });
    });
    group.finish();
}

criterion_group!(benches, body_empty, body_from_static, body_clone_shared);
criterion_main!(benches);
