//! Per-op sink cost vs the established anchors: source admission ~650 ps
//! (the `CallsiteGate`, elsewhere), sink push ~17 ns (`FanOut` N=1).
//!
//! Measures the new push primitives directly, both expected at/under the 17 ns
//! sink anchor: `SinkFront::emit` + `drain_one` steady pass-through (gate +
//! bounded enqueue + dequeue, the owned sink push); and `DrainSink::accept` +
//! `pop` pass-through (the zero-copy ring sink).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use proxima_primitives::pipe::{DrainSink, RingSink};
use proxima_primitives::pipe::{FailMode, SinkFront};

fn bench(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("sink_perop");

    // owned sink push: emit (gate + bounded enqueue) then drain_one (dequeue).
    let front = SinkFront::<u64>::new(1024, FailMode::DropOldest);
    front.arm();
    group.bench_function("sinkfront_emit_drain", |bencher| {
        bencher.iter(|| {
            let _ = black_box(front.emit(black_box(7u64)));
            black_box(front.drain_one())
        });
    });

    // zero-copy ring sink: accept a borrowed frame then pop it.
    let mut ring: RingSink<1024, 64> = RingSink::new();
    let frame = [0xABu8; 48];
    group.bench_function("drainsink_accept_pop", |bencher| {
        bencher.iter(|| {
            let _ = black_box(ring.accept(black_box(&frame[..])));
            black_box(ring.pop().map(|view| view.len()))
        });
    });

    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
