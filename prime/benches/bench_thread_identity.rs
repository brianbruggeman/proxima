//! micro-bench for C1 (thread-identity-trait).
//!
//! incumbent (baseline): `std::thread::current().id()` — the canonical stdlib
//!   path any external reviewer would compare against, plus direct
//!   `std::thread_local!` access (the pattern the existing TLS sites in
//!   `prime/src/core/inbox.rs` and `prime/src/core/local_executor.rs` use
//!   today).
//!
//! component-under-test: trait-routed read via `StdThreadIdentity::current()`.
//!
//! design-favors matrix
//! ────────────────────
//! proxima   — engages our impl on its own turf (TLS read, is_owning)
//! incumbent — engages the stdlib design point (std::thread::current().id(),
//!             multi-threaded scheduler routing, cross-thread identity checks)
//! neutral   — apples-to-apples abstraction-tax check (our vs our, same op)
//!
//! expected outcome (current arm): zero delta within noise; inline +
//! monomorphization collapses both to identical assembly.
//!
//! expected outcome (scheduler arm): our TLS-backed u64 should win over
//! `std::thread::current()` because the std path allocates an Arc<Inner>
//! on thread creation and walks struct fields for `.id()`.
//!
//! bench shape: batched per-iter (1024 reads) so per-call cost emerges
//! above criterion's per-iter sampling overhead (~50ns on M1).
//! scheduler arms: 8 worker threads, iter_custom to measure wall-clock
//! across all 8 concurrently.
//!
//! requires-features: runtime-prime-thread-identity

#![cfg(feature = "runtime-prime-thread-identity")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

use core::sync::atomic::{AtomicU64, Ordering};
use std::hint::black_box;
use std::sync::{Arc, Barrier};
use std::time::{Duration, Instant};

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use prime::core::thread_identity::ThreadIdentity;
use prime::os::thread_identity::StdThreadIdentity;

const READS_PER_ITER: usize = 1024;
const SCHEDULER_THREADS: usize = 8;

// direct-access baseline: mirrors the TLS cell in prime::os::thread_identity.
static NEXT_BENCH_ID: AtomicU64 = AtomicU64::new(1);
std::thread_local! {
    static BENCH_THREAD_ID: u64 = NEXT_BENCH_ID.fetch_add(1, Ordering::Relaxed);
}

fn direct_tls_read() -> u64 {
    BENCH_THREAD_ID.with(|id| *id)
}

// ── scheduler routing helpers ────────────────────────────────────────────────

struct SchedulerHarness {
    start: Arc<Barrier>,
    end: Arc<Barrier>,
    stop: Arc<AtomicU64>,
}

impl SchedulerHarness {
    fn spawn_our(threads: usize) -> Self {
        let start = Arc::new(Barrier::new(threads + 1));
        let end = Arc::new(Barrier::new(threads + 1));
        let stop = Arc::new(AtomicU64::new(0));

        let captured = StdThreadIdentity::current();

        for _ in 0..threads {
            let start = start.clone();
            let end = end.clone();
            let stop = stop.clone();
            std::thread::spawn(move || {
                // warm TLS once before entering the measurement loop
                black_box(StdThreadIdentity::current());
                while stop.load(Ordering::Relaxed) == 0 {
                    start.wait();
                    for _ in 0..READS_PER_ITER {
                        black_box(StdThreadIdentity::is_owning(captured));
                    }
                    end.wait();
                }
            });
        }

        Self { start, end, stop }
    }

    fn spawn_std(threads: usize) -> Self {
        let start = Arc::new(Barrier::new(threads + 1));
        let end = Arc::new(Barrier::new(threads + 1));
        let stop = Arc::new(AtomicU64::new(0));

        let captured_id = std::thread::current().id();

        for _ in 0..threads {
            let start = start.clone();
            let end = end.clone();
            let stop = stop.clone();
            std::thread::spawn(move || {
                // warm thread::current() once before measurement
                black_box(std::thread::current().id());
                while stop.load(Ordering::Relaxed) == 0 {
                    start.wait();
                    for _ in 0..READS_PER_ITER {
                        black_box(std::thread::current().id() == captured_id);
                    }
                    end.wait();
                }
            });
        }

        Self { start, end, stop }
    }

    fn run_iters(&self, iters: u64) -> Duration {
        let mut total = Duration::ZERO;
        for _ in 0..iters {
            let start = Instant::now();
            self.start.wait();
            self.end.wait();
            total += start.elapsed();
        }
        total
    }
}

impl Drop for SchedulerHarness {
    fn drop(&mut self) {
        self.stop.store(1, Ordering::Relaxed);
        // unblock any threads waiting on the start barrier so they can exit
        self.start.wait();
        self.end.wait();
    }
}

// ── bench groups ────────────────────────────────────────────────────────────

fn bench_current(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("thread_identity_current");
    group.throughput(Throughput::Elements(READS_PER_ITER as u64));

    // warm TLS init for all paths before measurement starts
    black_box(StdThreadIdentity::current());
    black_box(direct_tls_read());
    black_box(std::thread::current().id());

    // design-favors: proxima — direct std::thread_local!::with read.
    group.bench_function("direct_tls_read", |bencher| {
        bencher.iter(|| {
            for _ in 0..READS_PER_ITER {
                black_box(direct_tls_read());
            }
        });
    });

    // design-favors: proxima — trait-routed read. expectation: identical to
    // direct_tls_read within noise (inline + monomorphization).
    group.bench_function("trait_routed_current", |bencher| {
        bencher.iter(|| {
            for _ in 0..READS_PER_ITER {
                black_box(StdThreadIdentity::current());
            }
        });
    });

    // design-favors: incumbent — canonical stdlib path for "what thread am I?".
    // std::thread::current() returns Arc<Inner>; .id() walks struct fields.
    // expectation: slower than our hand-rolled TLS u64.
    group.bench_function("std_thread_current_id", |bencher| {
        bencher.iter(|| {
            for _ in 0..READS_PER_ITER {
                black_box(std::thread::current().id());
            }
        });
    });

    group.finish();
}

fn bench_is_owning(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("thread_identity_is_owning");
    group.throughput(Throughput::Elements(READS_PER_ITER as u64));

    let captured = StdThreadIdentity::current();

    // design-favors: proxima — direct TLS read + Eq compare.
    group.bench_function("direct_tls_eq", |bencher| {
        bencher.iter(|| {
            for _ in 0..READS_PER_ITER {
                black_box(direct_tls_read() == captured);
            }
        });
    });

    // design-favors: proxima — trait-routed is_owning.
    group.bench_function("trait_routed_is_owning", |bencher| {
        bencher.iter(|| {
            for _ in 0..READS_PER_ITER {
                black_box(StdThreadIdentity::is_owning(captured));
            }
        });
    });

    group.finish();
}

fn bench_cross_thread_identity(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("thread_identity_cross_thread");
    group.throughput(Throughput::Elements(READS_PER_ITER as u64));

    // capture ids on the bench thread (thread A); worker threads (thread B)
    // will check "is this id mine?" — expected false.
    let our_id_from_a = StdThreadIdentity::current();
    let std_id_from_a = std::thread::current().id();

    // design-favors: incumbent — cross-thread is_owning check (expected: false).
    // models a scheduler gate: "is this task owned by me?" when it was captured
    // on a different thread.
    group.bench_function("our_is_owning_cross_thread", |bencher| {
        bencher.iter_custom(|iters| {
            let (tx, rx) = std::sync::mpsc::channel::<Duration>();
            std::thread::spawn(move || {
                // warm our TLS on this thread before measurement
                black_box(StdThreadIdentity::current());
                let start = Instant::now();
                for _ in 0..iters {
                    for _ in 0..READS_PER_ITER {
                        black_box(StdThreadIdentity::is_owning(our_id_from_a));
                    }
                }
                tx.send(start.elapsed()).unwrap();
            });
            rx.recv().unwrap()
        });
    });

    // design-favors: incumbent — stdlib cross-thread id equality check.
    group.bench_function("std_id_eq_cross_thread", |bencher| {
        bencher.iter_custom(|iters| {
            let (tx, rx) = std::sync::mpsc::channel::<Duration>();
            std::thread::spawn(move || {
                // warm std::thread::current() on this thread before measurement
                black_box(std::thread::current().id());
                let start = Instant::now();
                for _ in 0..iters {
                    for _ in 0..READS_PER_ITER {
                        black_box(std::thread::current().id() == std_id_from_a);
                    }
                }
                tx.send(start.elapsed()).unwrap();
            });
            rx.recv().unwrap()
        });
    });

    group.finish();
}

fn bench_scheduler_routing(criterion: &mut Criterion) {
    let elements = (SCHEDULER_THREADS * READS_PER_ITER) as u64;
    let mut group = criterion.benchmark_group("thread_identity_scheduler_routing_8t");
    group.throughput(Throughput::Elements(elements));

    // design-favors: incumbent — the actual design point: multi-threaded
    // scheduler hot path where each thread identifies itself to route work to
    // the owning thread. 8 workers run concurrently; we measure wall-clock
    // across the batch so concurrency is visible in the throughput number.
    //
    // our impl: StdThreadIdentity::is_owning — TLS u64 read + Eq.
    let our_harness = SchedulerHarness::spawn_our(SCHEDULER_THREADS);
    group.bench_function("our_current_eq_captured", |bencher| {
        bencher.iter_custom(|iters| our_harness.run_iters(iters));
    });

    // design-favors: incumbent — stdlib impl of the same routing check.
    // std::thread::current() must re-derive the Arc<Inner> each call;
    // .id() == captured is the idiomatic stdlib pattern.
    let std_harness = SchedulerHarness::spawn_std(SCHEDULER_THREADS);
    group.bench_function("std_current_id_eq_captured", |bencher| {
        bencher.iter_custom(|iters| std_harness.run_iters(iters));
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_current,
    bench_is_owning,
    bench_cross_thread_identity,
    bench_scheduler_routing,
);
criterion_main!(benches);
