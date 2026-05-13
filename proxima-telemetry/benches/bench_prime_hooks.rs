#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]
//! C15 bench: prime::spawn span-carry overhead vs baselines.
//!
//! The original arms measured `proxima::runtime::prime`'s `spawn_local`
//! span-carry against tokio+tracing `Instrument`. That runtime lives in the
//! `proxima` umbrella crate, which proxima-telemetry does not depend on, so
//! those arms are dropped here — the spawn machinery they exercised has no
//! equivalent inside this crate. What remains and is still meaningful from a
//! telemetry standpoint is the cost of materialising and carrying the SpanId
//! payload itself, which is the value the runtime hook actually moves across
//! a spawn boundary.

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use proxima_telemetry::id::SpanId;

fn make_span() -> SpanId {
    SpanId::from_bytes([0xde, 0xad, 0xbe, 0xef, 0xca, 0xfe, 0xf0, 0x0d])
}

fn bench_c15(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c15_prime_hooks");

    // span-carry payload: construct the SpanId a spawn hook would clone across
    // the boundary. This is the telemetry-side cost the runtime carry wraps.
    group.bench_function("span_carry_payload_construct", |bencher| {
        bencher.iter(|| {
            let span = make_span();
            black_box(Some(black_box(span)))
        });
    });

    group.finish();
}

criterion_group!(c15_benches, bench_c15);
criterion_main!(c15_benches);
