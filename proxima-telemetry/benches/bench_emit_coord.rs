#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]
//! C1 hot-path proof: the per-record level decision is a `u64` compare + mask.
//! These are the ops `CompiledEmit::decide` (C2) calls per record; they take no
//! heap input and call no allocating API, so they are zero-alloc by construction
//! — the bench measures the latency, the no-alloc property is structural.

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use proxima_telemetry::emit::Coord;
use proxima_telemetry::level::Level;

fn bench_coord(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c1_emit_coord");

    // the keep test against a flat floor: one Coord compare.
    let floor = Coord::from(Level::WARN);
    let record = Coord::parse("17.2.1").unwrap(); // an error-band leaf
    group.bench_function("cmp_vs_floor", |bencher| {
        bencher.iter(|| black_box(black_box(record) >= black_box(floor)));
    });

    // the verbose-subtree test: one mask + one compare.
    let subtree = Coord::parse("1.3").unwrap();
    let in_tree = Coord::parse("1.3.5").unwrap();
    group.bench_function("in_subtree_of", |bencher| {
        bencher.iter(|| black_box(black_box(in_tree).in_subtree_of(black_box(subtree))));
    });

    // flat-Level → Coord bridge (per emit when a log carries a flat Level).
    group.bench_function("from_severity", |bencher| {
        bencher.iter(|| black_box(Coord::from(black_box(Level::INFO))));
    });

    // cold path: parse a dotted coord (config time, not per record).
    group.bench_function("parse_3seg", |bencher| {
        bencher.iter(|| black_box(Coord::parse(black_box("17.2.1"))));
    });

    group.finish();
}

criterion_group!(benches, bench_coord);
criterion_main!(benches);
