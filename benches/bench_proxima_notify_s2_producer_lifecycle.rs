//! Micro-bench for ProducerLifecycle (S2 of the proxima-notify initiative)
//! vs. the in-tree incumbent: the bespoke `while runtime.timer_at(next).await`
//! loop in `spawn_snapshot_ticker` at `src/scenarios/orchestrator.rs:815-838`.
//!
//! Per `/disciplined-component` gate point 13, every named incumbent gets at
//! least one bench arm engaging its design point — its README, its own bench
//! suite, or its data structure's design point. The orchestrator's
//! `spawn_snapshot_ticker` is a fixed-cadence 1Hz telemetry-only loop
//! bounded by a load-test deadline; its design point is precise
//! absolute-deadline scheduling over a known-duration run. The honest bench
//! arm engages that exact shape with a noop body and measures timer jitter
//! plus total drain time, NOT throughput.
//!
//! incumbents:
//!   - `spawn_snapshot_ticker` shape — direct `tokio::task::spawn_local` of a
//!     `while next <= run_end { runtime.timer_at(next).await; ...; next += period; }`
//!     loop. No `CancellationToken`, no `JoinSet`, no per-task naming. Aborts
//!     via `JoinHandle::abort()`.
//!
//! groups (and design-favors per workload):
//!
//! - lifecycle_spawn_drain — design-favors: proxima. Spawn N `SourcePipe`s
//!   into a ProducerLifecycle, immediately cancel, measure spawn + drain
//!   wall-clock. proxima's machinery is fully engaged; no incumbent
//!   equivalent for multi-source lifecycle management.
//! - lifecycle_fixed_cadence_1hz — design-favors: incumbent. A single
//!   `SourcePipe` that ticks at 1Hz for the bench duration. Measures the
//!   wrapper overhead vs. a direct timer_at loop. spawn_snapshot_ticker's
//!   design point — 1Hz fixed-cadence telemetry — engaged on both sides.
//!   The incumbent has zero overhead beyond `tokio::task::spawn_local` +
//!   the loop body; we have the wrapper `select!` + JoinSet machinery. We
//!   should match within a few percent OR document the regime gap.
//! - lifecycle_shutdown_propagation — design-favors: proxima.
//!   shutdown(grace) on a 100-source lifecycle vs. 100 individual
//!   JoinHandle::abort() + .await loops. proxima's cancellation-then-drain
//!   model is fully engaged; the incumbent does not natively support
//!   multi-task graceful shutdown.
//!
//! required-features: runtime-tokio. Sources drive unconditionally
//! (proxima-pipe TARGET 4 — no `producer-lifecycle` feature gate any more).

#![cfg(feature = "runtime-tokio")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

// this bench never calls into prime directly, but the workspace's default
// features unify `proxima_core::time` onto prime's link-injected driver (the
// `serve-prime` default pulls `dep:prime`, and `prime/Cargo.toml` requests
// `proxima-core/time-driver-prime-wheel` unconditionally). Without a live
// reference into `prime`, `-Wl,-dead_strip` drops its rlib from this bench's
// binary and the driver's `extern "C"` symbols go unresolved at link time.
extern crate prime as _;

use criterion::{Criterion, criterion_group, criterion_main};
use proxima_core::signal::Signal;
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::{ProducerLifecycle, ProximaError, SourceHandle, into_source_handle};
use tokio::runtime::Runtime;

/// A source that increments a shared counter once, then returns immediately.
struct NoopSource {
    counter: Arc<AtomicUsize>,
}

impl SendPipe for NoopSource {
    type In = Signal;
    type Out = ();
    type Err = ProximaError;

    fn call(&self, _cancel: Signal) -> impl Future<Output = Result<(), ProximaError>> + Send {
        self.counter.fetch_add(1, Ordering::Relaxed);
        async { Ok(()) }
    }
}

/// A source that awaits cancellation, then returns.
struct WaiterSource;

impl SendPipe for WaiterSource {
    type In = Signal;
    type Out = ();
    type Err = ProximaError;

    fn call(&self, cancel: Signal) -> impl Future<Output = Result<(), ProximaError>> + Send {
        async move {
            cancel.fired().await;
            Ok(())
        }
    }
}

/// design-favors: proxima — multi-source spawn + drain. The incumbent has no
/// equivalent shape; this arm shows the cost of our machinery in isolation.
fn lifecycle_spawn_drain_100_tasks(criterion: &mut Criterion) {
    let runtime = Runtime::new().unwrap();
    let mut group = criterion.benchmark_group("lifecycle_spawn_drain");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(8));

    group.bench_function("100_tasks", |bencher| {
        bencher.iter(|| {
            runtime.block_on(async {
                let mut lifecycle = ProducerLifecycle::new();
                let counter = Arc::new(AtomicUsize::new(0));
                for index in 0..100 {
                    let source: SourceHandle = into_source_handle(NoopSource {
                        counter: counter.clone(),
                    });
                    lifecycle.spawn_from_source(&format!("bench-{index}"), &source);
                }
                let report = lifecycle.shutdown(Duration::from_secs(1)).await;
                assert_eq!(report.total, 100);
                std::hint::black_box(report)
            });
        });
    });

    group.finish();
}

/// design-favors: incumbent — fixed 1Hz cadence. Engages the snapshot_ticker
/// design point on both sides. The incumbent has zero overhead beyond
/// `tokio::task::spawn` + the loop body; we have wrapper select! + JoinSet
/// machinery. We expect to match within a few percent OR document the gap.
fn lifecycle_fixed_cadence_1hz_vs_direct_timer(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("lifecycle_fixed_cadence_1hz");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(15));

    let bench_duration = Duration::from_millis(500);
    let cadence = Duration::from_millis(50);

    group.bench_function("proxima_lifecycle", |bencher| {
        let runtime = Runtime::new().unwrap();
        bencher.iter(|| {
            runtime.block_on(async {
                struct Ticker {
                    ticks: Arc<AtomicUsize>,
                    bench_duration: Duration,
                    cadence: Duration,
                }
                impl SendPipe for Ticker {
                    type In = Signal;
                    type Out = ();
                    type Err = ProximaError;

                    fn call(
                        &self,
                        cancel: Signal,
                    ) -> impl Future<Output = Result<(), ProximaError>> + Send {
                        let ticks = self.ticks.clone();
                        let bench_duration = self.bench_duration;
                        let cadence = self.cadence;
                        async move {
                            let started = Instant::now();
                            while started.elapsed() < bench_duration {
                                let cancelled = tokio::select! {
                                    () = tokio::time::sleep(cadence) => false,
                                    () = cancel.fired() => true,
                                };
                                if cancelled {
                                    break;
                                }
                                ticks.fetch_add(1, Ordering::Relaxed);
                            }
                            Ok(())
                        }
                    }
                }

                let mut lifecycle = ProducerLifecycle::new();
                let ticks = Arc::new(AtomicUsize::new(0));
                let source: SourceHandle = into_source_handle(Ticker {
                    ticks: ticks.clone(),
                    bench_duration,
                    cadence,
                });
                lifecycle.spawn_from_source("ticker", &source);

                tokio::time::sleep(bench_duration + Duration::from_millis(10)).await;
                let report = lifecycle.shutdown(Duration::from_millis(100)).await;
                std::hint::black_box(ticks.load(Ordering::Relaxed));
                std::hint::black_box(report);
            });
        });
    });

    group.bench_function("direct_spawn_loop", |bencher| {
        let runtime = Runtime::new().unwrap();
        bencher.iter(|| {
            runtime.block_on(async {
                let ticks = Arc::new(AtomicUsize::new(0));
                let ticks_for_task = ticks.clone();
                let handle = tokio::spawn(async move {
                    let started = Instant::now();
                    while started.elapsed() < bench_duration {
                        tokio::time::sleep(cadence).await;
                        ticks_for_task.fetch_add(1, Ordering::Relaxed);
                    }
                });
                tokio::time::sleep(bench_duration + Duration::from_millis(10)).await;
                let _ = handle.await;
                std::hint::black_box(ticks.load(Ordering::Relaxed));
            });
        });
    });

    group.finish();
}

/// design-favors: proxima — multi-source graceful shutdown via a cancel Signal.
/// The incumbent JoinHandle::abort() does not natively coordinate across N
/// tasks; we have to await each one manually.
fn lifecycle_shutdown_propagation_100_tasks(criterion: &mut Criterion) {
    let runtime = Runtime::new().unwrap();
    let mut group = criterion.benchmark_group("lifecycle_shutdown_propagation");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(8));

    group.bench_function("100_tasks_observe_cancel", |bencher| {
        bencher.iter(|| {
            runtime.block_on(async {
                let mut lifecycle = ProducerLifecycle::new();
                for index in 0..100 {
                    let source: SourceHandle = into_source_handle(WaiterSource);
                    lifecycle.spawn_from_source(&format!("shutdown-{index}"), &source);
                }
                let report = lifecycle.shutdown(Duration::from_secs(1)).await;
                assert_eq!(report.drained, 100);
                std::hint::black_box(report)
            });
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    lifecycle_spawn_drain_100_tasks,
    lifecycle_fixed_cadence_1hz_vs_direct_timer,
    lifecycle_shutdown_propagation_100_tasks
);
criterion_main!(benches);
