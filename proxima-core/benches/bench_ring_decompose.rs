// Decomposition of bench_ring's proxima 1M "saturation" (7.5 ns/elem hot ->
// 66 ns/elem at 1M, while crossbeam ArrayQueue stays flat 13.5 -> 14.2). The
// original arm allocates a fresh Ring + an 8 MB `vec![0u64; cap]` out-buffer
// INSIDE the timed loop and bulk-writes the drain into it; the crossbeam arm
// pays none of that (pop-to-register). These arms separate alloc vs push vs
// drain, inline vs hoisted, to attribute the 52 ns/elem gap.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::needless_range_loop,
    clippy::useless_vec
)]
use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use proxima_core::ring::{Drainer, Ring};

// cache sweep: cell is 16 B, so buf bytes = 16 * cap. 4K=64KB(fits L1) crossing
// to 1M=16MB(>L2). The cliff lands between 4K and 16K — i.e. at the L1 boundary.
const SIZES: &[usize] = &[4_096, 16_384, 65_536, 262_144, 1_048_576];

// alloc + init the Vyukov cells only — no push/drain. Isolates Ring::new cost.
fn alloc_ring_only(crit: &mut Criterion) {
    let mut group = crit.benchmark_group("ring_decompose");
    for size in SIZES {
        group.bench_with_input(
            BenchmarkId::new("alloc_ring_only", size),
            size,
            |bench, &cap| {
                bench.iter(|| black_box(Ring::<u64>::new(black_box(cap)).expect("ring")));
            },
        );
    }
    group.finish();
}

// the 8 MB out-buffer alloc + zero that the proxima arm pays and crossbeam does not.
fn alloc_outbuf_only(crit: &mut Criterion) {
    let mut group = crit.benchmark_group("ring_decompose");
    for size in SIZES {
        group.bench_with_input(
            BenchmarkId::new("alloc_outbuf_only", size),
            size,
            |bench, &cap| {
                bench.iter(|| black_box(vec![0u64; black_box(cap)]));
            },
        );
    }
    group.finish();
}

// the ORIGINAL bench_ring shape: alloc ring + out-buffer inside the timed loop.
fn push_drain_inline(crit: &mut Criterion) {
    let mut group = crit.benchmark_group("ring_decompose");
    for size in SIZES {
        group.bench_with_input(
            BenchmarkId::new("push_drain_inline", size),
            size,
            |bench, &cap| {
                bench.iter(|| {
                    let ring = Ring::<u64>::new(cap).expect("ring");
                    let mut drainer = Drainer::new(&ring);
                    let mut out = vec![0u64; cap];
                    for item in 0..cap as u64 {
                        while ring.push(black_box(item)).is_err() {}
                    }
                    let mut total = 0usize;
                    while total < cap {
                        total += drainer.drain_into(&mut out[total..]);
                    }
                    black_box(total)
                });
            },
        );
    }
    group.finish();
}

// ring + out-buffer allocated ONCE outside the timed loop; iter = push+drain only
// (ring returns to empty each pass, so it is reusable). Isolates push/drain
// throughput from per-iteration allocation.
fn push_drain_hoisted(crit: &mut Criterion) {
    let mut group = crit.benchmark_group("ring_decompose");
    for size in SIZES {
        group.bench_with_input(
            BenchmarkId::new("push_drain_hoisted", size),
            size,
            |bench, &cap| {
                let ring = Ring::<u64>::new(cap).expect("ring");
                let mut out = vec![0u64; cap];
                let mut drainer = Drainer::new(&ring);
                bench.iter(|| {
                    for item in 0..cap as u64 {
                        while ring.push(black_box(item)).is_err() {}
                    }
                    let mut total = 0usize;
                    while total < cap {
                        total += drainer.drain_into(&mut out[total..]);
                    }
                    black_box(total)
                });
            },
        );
    }
    group.finish();
}

// push only, ring hoisted, NO drain — isolates the fill half.
fn push_only_hoisted(crit: &mut Criterion) {
    let mut group = crit.benchmark_group("ring_decompose");
    for size in SIZES {
        group.bench_with_input(
            BenchmarkId::new("push_only_hoisted", size),
            size,
            |bench, &cap| {
                let ring = Ring::<u64>::new(cap).expect("ring");
                let mut drainer = Drainer::new(&ring);
                let mut out = vec![0u64; cap];
                bench.iter(|| {
                    for item in 0..cap as u64 {
                        while ring.push(black_box(item)).is_err() {}
                    }
                    // drain to reset for the next iteration, but only the push is the focus.
                    let mut total = 0usize;
                    while total < cap {
                        total += drainer.drain_into(&mut out[total..]);
                    }
                    black_box(ring.len())
                });
            },
        );
    }
    group.finish();
}

// push hoisted, then drain via bare dequeue() to a register (NO out-buffer write).
// Splits "ring cell-access latency" from "the 8 MB out-buffer second write stream":
// if this flattens vs push_drain_hoisted, the out stream was the cost; if it stays
// at ~60 ns/elem the ring's pop is itself latency-bound (the prefetch target).
fn proxima_drain_to_register(crit: &mut Criterion) {
    let mut group = crit.benchmark_group("ring_decompose");
    for size in SIZES {
        group.bench_with_input(
            BenchmarkId::new("proxima_drain_to_register", size),
            size,
            |bench, &cap| {
                let ring = Ring::<u64>::new(cap).expect("ring");
                bench.iter(|| {
                    for item in 0..cap as u64 {
                        while ring.push(black_box(item)).is_err() {}
                    }
                    let mut total = 0usize;
                    while let Some(value) = ring.dequeue() {
                        black_box(value);
                        total += 1;
                    }
                    black_box(total)
                });
            },
        );
    }
    group.finish();
}

// crossbeam push+pop with the queue hoisted out — the fair incumbent comparison
// (pop-to-register, no out-buffer), to confirm crossbeam stays flat hoisted too.
fn crossbeam_hoisted(crit: &mut Criterion) {
    let mut group = crit.benchmark_group("ring_decompose");
    for size in SIZES {
        group.bench_with_input(
            BenchmarkId::new("crossbeam_hoisted", size),
            size,
            |bench, &cap| {
                let queue = crossbeam_queue::ArrayQueue::<u64>::new(cap);
                bench.iter(|| {
                    for item in 0..cap as u64 {
                        let _ = queue.push(black_box(item));
                    }
                    let mut total = 0usize;
                    while let Some(value) = queue.pop() {
                        black_box(value);
                        total += 1;
                    }
                    black_box(total)
                });
            },
        );
    }
    group.finish();
}

// FAIR incumbent comparison: crossbeam draining into the same 8 MB out-buffer that
// proxima's drain_into writes. If crossbeam ALSO cliffs here, the cost is the
// out-buffer materialization (fundamental, both pay it), not the ring.
fn crossbeam_into_buffer(crit: &mut Criterion) {
    let mut group = crit.benchmark_group("ring_decompose");
    for size in SIZES {
        group.bench_with_input(
            BenchmarkId::new("crossbeam_into_buffer", size),
            size,
            |bench, &cap| {
                let queue = crossbeam_queue::ArrayQueue::<u64>::new(cap);
                let mut out = vec![0u64; cap];
                bench.iter(|| {
                    for item in 0..cap as u64 {
                        let _ = queue.push(black_box(item));
                    }
                    let mut count = 0usize;
                    while let Some(value) = queue.pop() {
                        out[count] = value;
                        count += 1;
                    }
                    black_box(&out[..count]);
                    black_box(count)
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    crossbeam_into_buffer,
    alloc_ring_only,
    alloc_outbuf_only,
    push_drain_inline,
    push_drain_hoisted,
    push_only_hoisted,
    proxima_drain_to_register,
    crossbeam_hoisted,
);
criterion_main!(benches);
