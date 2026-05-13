//! `proxima::time` timeout + sleep_until baseline — A4.a of the tokio-parity plan.
//!
//! Two workloads mirroring real proxima usage (process_rpc timeout backstop,
//! http-listener quiesce-window drain). Harness only — numbers come after
//! A4.b ships the `Instant` newtype + `sleep_until` + `timeout_at` impl;
//! see discipline-tokio-parity.md for the log.
//!
//! Arms per workload:
//! - `proxima_time_timeout` / `proxima_time_sleep_until_via_sleep`
//!   — existing proxima::time surface (futures-timer backed)
//! - `tokio_time_timeout` / `tokio_time_sleep_until`
//!   — tokio::time as the parity baseline
//!
//! Run:
//! ```bash
//! cargo bench -p proxima --features sync-wrappers --bench bench_time_instant_parity
//! cargo bench -p proxima --features sync-wrappers --bench bench_time_instant_parity -- timeout
//! ```

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};
use futures::FutureExt;
use tokio::runtime::Builder as TokioBuilder;

// ---------- workload 1 constants (timeout backstop) ----------

// 200 iterations per Criterion sample. Originally 1000 in the plan, but
// the slow-path timer dominates wall time (10 slow × TIMEOUT_DURATION).
// 200 × 1ms timeout × 1% slow = 2ms slow + 200 × 50µs fast = 12ms per sample.
const TIMEOUT_ITERATIONS: u64 = 200;

// 99% of dispatches complete in 50µs; 1% hang until the 1ms timer fires.
// seeded fastrand with a fixed value gives deterministic per-iteration paths.
// Wall-clock ratios preserve the "deadline as backstop" semantic (50µs fast
// path is 5% of the 1ms timeout — slow path is genuinely much slower).
const FAST_SLEEP_US: u64 = 50;
const SLOW_SLEEP_US: u64 = 100_000; // 100ms — intentionally never reached
const TIMEOUT_DURATION: Duration = Duration::from_millis(1);

// probability-of-slow = 1 in this modulus (1/100 = 1%)
const SLOW_MODULUS: u64 = 100;

// ---------- workload 2 constants (quiesce-window sleep_until) ----------

// each full select loop: deadline = now + 100ms; channel fires every 10ms;
// loop exits after 10 channel events.
const QUIESCE_DEADLINE_MS: u64 = 100;
const ACCEPT_INTERVAL_MS: u64 = 10;
const ACCEPT_EVENTS: usize = 10;

fn current_thread_runtime() -> tokio::runtime::Runtime {
    TokioBuilder::new_current_thread()
        .enable_all()
        .build()
        .expect("current thread runtime")
}

// ---------- workload 1: timeout backstop (process_rpc pattern) ----------
//
// process_rpc.rs:93 wraps every dispatch in `tokio::time::timeout(request_timeout_ms, ...)`.
// 99% of calls finish in ~500µs; 1% hang and the backstop fires.
//
// this bench measures per-iteration wall time across 1000 iterations, exposing
// the timeout wrapper overhead on the fast (no-fire) path and correctness on
// the slow (fire) path. fastrand with a fixed seed keeps the path mix
// deterministic across runs.

fn bench_workload1_timeout_backstop(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("time_parity_w1_timeout_backstop");
    group.measurement_time(Duration::from_secs(5));

    group.bench_function("proxima_time_timeout", |bench| {
        let runtime = current_thread_runtime();
        bench.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for outer in 0..iters {
                let elapsed = runtime.block_on(async {
                    let start = std::time::Instant::now();
                    let seed = outer.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let mut rng = fastrand::Rng::with_seed(seed);

                    for counter in 0..TIMEOUT_ITERATIONS {
                        let is_slow = (counter % SLOW_MODULUS) == 0;
                        let sleep_us = if is_slow {
                            SLOW_SLEEP_US
                        } else {
                            FAST_SLEEP_US + (rng.u64(0..200))
                        };
                        let fut = futures_timer::Delay::new(Duration::from_micros(sleep_us));
                        let outcome = proxima::time::timeout(TIMEOUT_DURATION, fut).await;
                        std::hint::black_box(outcome.is_ok());
                    }
                    start.elapsed()
                });
                total += elapsed;
            }
            total
        });
    });

    group.bench_function("tokio_time_timeout", |bench| {
        let runtime = current_thread_runtime();
        bench.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for outer in 0..iters {
                let elapsed = runtime.block_on(async {
                    let start = std::time::Instant::now();
                    let seed = outer.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let mut rng = fastrand::Rng::with_seed(seed);

                    for counter in 0..TIMEOUT_ITERATIONS {
                        let is_slow = (counter % SLOW_MODULUS) == 0;
                        let sleep_us = if is_slow {
                            SLOW_SLEEP_US
                        } else {
                            FAST_SLEEP_US + (rng.u64(0..200))
                        };
                        let fut = tokio::time::sleep(Duration::from_micros(sleep_us));
                        let outcome = tokio::time::timeout(TIMEOUT_DURATION, fut).await;
                        std::hint::black_box(outcome.is_ok());
                    }
                    start.elapsed()
                });
                total += elapsed;
            }
            total
        });
    });

    group.finish();
}

// ---------- workload 2: quiesce-window sleep_until (listener drain pattern) ----------
//
// proxima-http/src/listener/mod.rs:385 and rust/src/listeners/http_uring.rs:165
// both run a select! loop: sleep_until(deadline) races accept().
// accept fires every 10ms; the loop exits after 10 accept events (or the
// deadline fires, whichever comes first).
//
// proxima arm note: proxima::time has no sleep_until yet (A4.b adds it).
// the proxima arm uses `proxima::time::sleep(deadline.saturating_duration_since(now))`
// to simulate what sleep_until will do — same semantics, recomputed each
// iteration of the outer loop. A4.b will replace this with native
// `proxima::time::sleep_until(deadline)` and re-bench; that row is the
// "A4.b impl" changelog entry in the discipline log.
//
// each Criterion iteration = one full select loop cycle (10 accept events).

fn bench_workload2_sleep_until_quiesce(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("time_parity_w2_sleep_until_quiesce");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(20);

    group.bench_function("proxima_time_sleep_until_via_sleep", |bench| {
        let runtime = current_thread_runtime();
        bench.iter(|| {
            runtime.block_on(async {
                let (accept_tx, mut accept_rx) =
                    tokio::sync::mpsc::channel::<()>(ACCEPT_EVENTS + 1);

                // sender task: fires 1 accept event every 10ms
                let sender = tokio::spawn(async move {
                    for _ in 0..ACCEPT_EVENTS {
                        tokio::time::sleep(Duration::from_millis(ACCEPT_INTERVAL_MS)).await;
                        let _ = accept_tx.send(()).await;
                    }
                });

                let deadline =
                    std::time::Instant::now() + Duration::from_millis(QUIESCE_DEADLINE_MS);
                let mut accepted = 0usize;

                loop {
                    // proxima stand-in: recompute remaining each loop iteration.
                    // A4.b replaces this with proxima::time::sleep_until(deadline).
                    let remaining = deadline.saturating_duration_since(std::time::Instant::now());
                    let sleep_fut = proxima::time::sleep(remaining);
                    futures::pin_mut!(sleep_fut);
                    let recv_fut = accept_rx.recv();
                    futures::pin_mut!(recv_fut);

                    futures::select! {
                        _ = sleep_fut.fuse() => break,
                        msg = recv_fut.fuse() => {
                            if msg.is_some() {
                                accepted += 1;
                                if accepted >= ACCEPT_EVENTS {
                                    break;
                                }
                            } else {
                                break;
                            }
                        }
                    }
                }

                std::hint::black_box(accepted);
                let _ = sender.await;
            });
        });
    });

    group.bench_function("tokio_time_sleep_until", |bench| {
        let runtime = current_thread_runtime();
        bench.iter(|| {
            runtime.block_on(async {
                let (accept_tx, mut accept_rx) =
                    tokio::sync::mpsc::channel::<()>(ACCEPT_EVENTS + 1);

                let sender = tokio::spawn(async move {
                    for _ in 0..ACCEPT_EVENTS {
                        tokio::time::sleep(Duration::from_millis(ACCEPT_INTERVAL_MS)).await;
                        let _ = accept_tx.send(()).await;
                    }
                });

                let deadline =
                    tokio::time::Instant::now() + Duration::from_millis(QUIESCE_DEADLINE_MS);
                let mut accepted = 0usize;

                loop {
                    let sleep_fut = tokio::time::sleep_until(deadline);
                    futures::pin_mut!(sleep_fut);
                    let recv_fut = accept_rx.recv();
                    futures::pin_mut!(recv_fut);

                    futures::select! {
                        _ = sleep_fut.fuse() => break,
                        msg = recv_fut.fuse() => {
                            if msg.is_some() {
                                accepted += 1;
                                if accepted >= ACCEPT_EVENTS {
                                    break;
                                }
                            } else {
                                break;
                            }
                        }
                    }
                }

                std::hint::black_box(accepted);
                let _ = sender.await;
            });
        });
    });

    group.finish();
}

criterion_group!(
    bench_time_instant_parity_workloads,
    bench_workload1_timeout_backstop,
    bench_workload2_sleep_until_quiesce,
);
criterion_main!(bench_time_instant_parity_workloads);
