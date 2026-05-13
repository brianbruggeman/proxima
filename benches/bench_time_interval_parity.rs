//! `proxima::time::Interval` vs `tokio::time::Interval` baseline capture.
//!
//! This is the A5.a bench harness — baselines captured BEFORE the A5.b
//! rewrite that adds `.tick().await`, `MissedTickBehavior`, and `interval_at`.
//! All numbers here reflect the current proxima `Stream<Item=()>` shape so the
//! A5.b post-bench can show the delta honestly.
//!
//! Workloads mirror real proxima usage, not synthetic tight-loops:
//!
//! **Workload 1 — long-period maintenance tick** (rate_limit.rs:267 pattern):
//! 50ms period, consumer busy-spins 5ms (DashMap scan stand-in), 20 ticks.
//! Arms: proxima stream, tokio Skip, tokio Burst (default).
//! Reveals whether proxima's stream-based interval has comparable jitter to
//! tokio under realistic consumer load and whether MissedTickBehavior choice
//! matters when the consumer fits inside the period.
//!
//! **Workload 1b — slow consumer** (same group, extra arm):
//! 50ms period, consumer busy-spins 75ms (exceeds period), 5 ticks.
//! Documents Skip-vs-Burst delivery difference under overload. Proxima's
//! current hardcoded behavior is Skip.
//!
//! **Workload 2 — tight tick baseline** (jitter floor):
//! 10ms period, no-op consumer, 100 ticks.
//! Establishes timer-precision floor for proxima vs tokio.
//!
//! Run:
//! ```bash
//! cargo bench -p proxima --features sync-wrappers --bench bench_time_interval_parity
//! cargo bench -p proxima --features sync-wrappers --bench bench_time_interval_parity -- maintenance
//! cargo bench -p proxima --features sync-wrappers --bench bench_time_interval_parity -- tight
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
use futures::StreamExt;
use tokio::runtime::Builder as TokioBuilder;

const MAINTENANCE_PERIOD_MS: u64 = 50;
const MAINTENANCE_TICKS: usize = 20;
const CONSUMER_FAST_MS: u64 = 5;

const SLOW_PERIOD_MS: u64 = 50;
const SLOW_TICKS: usize = 5;
const CONSUMER_SLOW_MS: u64 = 75;

const TIGHT_PERIOD_MS: u64 = 10;
const TIGHT_TICKS: usize = 100;

fn current_thread_runtime() -> tokio::runtime::Runtime {
    TokioBuilder::new_current_thread()
        .enable_all()
        .build()
        .expect("current thread runtime")
}

/// Busy-spin for `millis` ms without yielding to the executor.
///
/// Simulates a CPU-bound consumer (DashMap scan, token bucket walk) so the
/// timer sees a real wall-clock gap. tokio::time::sleep would park the task
/// and hide the overload from the interval's missed-tick detection.
fn busy_spin(millis: u64) {
    let end = std::time::Instant::now() + Duration::from_millis(millis);
    while std::time::Instant::now() < end {}
}

// ---------- workload 1: maintenance tick, fast consumer ----------

fn bench_maintenance_tick(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("interval_maintenance_50ms_x20");
    group.measurement_time(Duration::from_secs(30));
    group.sample_size(10);

    // proxima arm: Stream<Item=()> via StreamExt::next().await.
    // current hardcoded behavior is Skip — proxima advances next_deadline by
    // period each tick, and if the consumer is slow the saturating_duration_since
    // in poll_next returns 0, so the next tick fires immediately (catching up).
    // under fast-consumer load (5ms < 50ms period) no ticks are missed, so
    // Skip vs Burst is irrelevant here.
    group.bench_function("proxima_time_interval_stream", |bench| {
        let runtime = current_thread_runtime();
        bench.iter(|| {
            runtime.block_on(async {
                let mut interval =
                    proxima::time::interval(Duration::from_millis(MAINTENANCE_PERIOD_MS));
                for _ in 0..MAINTENANCE_TICKS {
                    interval.next().await;
                    busy_spin(CONSUMER_FAST_MS);
                }
            });
        });
    });

    group.bench_function("tokio_time_interval_skip", |bench| {
        let runtime = current_thread_runtime();
        bench.iter(|| {
            runtime.block_on(async {
                let mut interval =
                    tokio::time::interval(Duration::from_millis(MAINTENANCE_PERIOD_MS));
                interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
                for _ in 0..MAINTENANCE_TICKS {
                    interval.tick().await;
                    busy_spin(CONSUMER_FAST_MS);
                }
            });
        });
    });

    // Burst is tokio's default — included so the bench table shows all three
    // variants side-by-side. Under fast-consumer load Burst == Skip in practice.
    group.bench_function("tokio_time_interval_burst", |bench| {
        let runtime = current_thread_runtime();
        bench.iter(|| {
            runtime.block_on(async {
                let mut interval =
                    tokio::time::interval(Duration::from_millis(MAINTENANCE_PERIOD_MS));
                for _ in 0..MAINTENANCE_TICKS {
                    interval.tick().await;
                    busy_spin(CONSUMER_FAST_MS);
                }
            });
        });
    });

    group.finish();
}

// ---------- workload 1b: maintenance tick, slow consumer ----------
//
// Consumer (75ms) exceeds the period (50ms). Every tick is a "missed" tick.
// tokio Skip: next deadline jumps to now+period, so one tick is dropped per
// slow iteration. tokio Burst: delivers all backed-up ticks immediately on
// next poll. Proxima: hardcoded Skip — advances next_deadline by period, but
// saturating_duration_since returns 0, so it fires immediately then resets to
// now+period. Behaviorally similar to Skip for a single-consumer loop.
// The wall-time difference between Burst and Skip becomes visible here.

fn bench_slow_consumer(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("interval_slow_consumer_50ms_x5");
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(10);

    // proxima: hardcoded Skip — see comment above.
    group.bench_function("proxima_time_interval_stream", |bench| {
        let runtime = current_thread_runtime();
        bench.iter(|| {
            runtime.block_on(async {
                let mut interval = proxima::time::interval(Duration::from_millis(SLOW_PERIOD_MS));
                for _ in 0..SLOW_TICKS {
                    interval.next().await;
                    busy_spin(CONSUMER_SLOW_MS);
                }
            });
        });
    });

    group.bench_function("tokio_time_interval_skip", |bench| {
        let runtime = current_thread_runtime();
        bench.iter(|| {
            runtime.block_on(async {
                let mut interval = tokio::time::interval(Duration::from_millis(SLOW_PERIOD_MS));
                interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
                for _ in 0..SLOW_TICKS {
                    interval.tick().await;
                    busy_spin(CONSUMER_SLOW_MS);
                }
            });
        });
    });

    group.bench_function("tokio_time_interval_burst", |bench| {
        let runtime = current_thread_runtime();
        bench.iter(|| {
            runtime.block_on(async {
                let mut interval = tokio::time::interval(Duration::from_millis(SLOW_PERIOD_MS));
                for _ in 0..SLOW_TICKS {
                    interval.tick().await;
                    busy_spin(CONSUMER_SLOW_MS);
                }
            });
        });
    });

    group.finish();
}

// ---------- workload 2: tight tick baseline ----------
//
// 10ms period, no-op consumer, 100 ticks. Establishes the timer-precision
// floor independent of consumer load. Jitter here is purely from the
// underlying timer source (futures-timer helper thread vs tokio time-wheel).

fn bench_tight_tick(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("interval_tight_10ms_x100");
    group.measurement_time(Duration::from_secs(30));
    group.sample_size(10);

    group.bench_function("proxima_time_interval_stream", |bench| {
        let runtime = current_thread_runtime();
        bench.iter(|| {
            runtime.block_on(async {
                let mut interval = proxima::time::interval(Duration::from_millis(TIGHT_PERIOD_MS));
                for _ in 0..TIGHT_TICKS {
                    interval.next().await;
                }
            });
        });
    });

    group.bench_function("tokio_time_interval", |bench| {
        let runtime = current_thread_runtime();
        bench.iter(|| {
            runtime.block_on(async {
                let mut interval = tokio::time::interval(Duration::from_millis(TIGHT_PERIOD_MS));
                for _ in 0..TIGHT_TICKS {
                    interval.tick().await;
                }
            });
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_maintenance_tick,
    bench_slow_consumer,
    bench_tight_tick,
);
criterion_main!(benches);
