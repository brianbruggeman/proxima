//! The background sender: pulls batches off the queue, sends with capped
//! exponential backoff + jitter, reconnects on transport error, and ticks
//! the horizon sweep / drop announcement / self-metrics export. Runs on a
//! dedicated `std::thread` (spawned by `ResilientSink::spawn_with_clock`) so
//! the shared drain never touches it.
//!
//! Supervision: the whole per-iteration body runs inside `catch_unwind`. A
//! panic is counted and logged, never lets the thread exit — an unsupervised
//! panic here would silently and permanently kill export for the rest of the
//! process lifetime, which is worse than any individual send failure.

use alloc::sync::Arc;
use core::sync::atomic::{AtomicU64, Ordering};
use core::time::Duration;

use proxima_primitives::pipe::alloc_tier::SendDynPipe;
use proxima_primitives::pipe::capabilities::Clock;
use proxima_primitives::pipe::resilience::{Backoff, Jitter};

use crate::level::Level;

use super::Shared;
use super::queue::{self, DropReason, PulledBatch, SEVERITY_BUCKETS};

const DROPPED_COUNTER_NAMES: [&str; SEVERITY_BUCKETS] = [
    "proxima.otlp.dropped.trace",
    "proxima.otlp.dropped.debug",
    "proxima.otlp.dropped.info",
    "proxima.otlp.dropped.warn",
    "proxima.otlp.dropped.error",
];
const SURVIVABLE_GAUGE_NAMES: [&str; SEVERITY_BUCKETS] = [
    "proxima.otlp.survivable_seconds.trace",
    "proxima.otlp.survivable_seconds.debug",
    "proxima.otlp.survivable_seconds.info",
    "proxima.otlp.survivable_seconds.warn",
    "proxima.otlp.survivable_seconds.error",
];
const RETRIED_COUNTER_NAME: &str = "proxima.otlp.retried_total";
const RECONNECTED_COUNTER_NAME: &str = "proxima.otlp.reconnected_total";
const SENT_COUNTER_NAME: &str = "proxima.otlp.sent_total";
const PANIC_COUNTER_NAME: &str = "proxima.otlp.worker_panics_total";
const BACKLOG_GAUGE_NAME: &str = "proxima.otlp.backlog_depth";

#[derive(Default)]
pub(super) struct SelfCounters {
    pub(super) sent_lifetime: AtomicU64,
    pub(super) retried_lifetime: AtomicU64,
    pub(super) reconnected_lifetime: AtomicU64,
    pub(super) panics: AtomicU64,
    sent_since_tick: AtomicU64,
    retried_since_tick: AtomicU64,
    reconnected_since_tick: AtomicU64,
}

/// In-flight send state, carried across `step` calls (and across a caught
/// panic — no `unsafe` is involved, so a partially-updated cursor after a
/// panic is still a structurally valid value to resume from).
#[derive(Default)]
pub(super) struct SendCursor {
    pub(super) current: Option<PulledBatch>,
    attempt: u32,
    prev_delay: Duration,
    pub(super) next_attempt_at_nanos: u64,
}

fn backoff(shared: &Shared<impl Clock>) -> Backoff {
    Backoff::Exponential {
        initial: shared.config.backoff_base(),
        factor: 2,
        max: shared.config.backoff_cap(),
    }
}

/// One unit of work: advance the in-flight batch (send or wait out its
/// backoff), pull a new one if idle, and run the periodic tick. Free
/// function (not a method) so tests can drive it directly with a fake clock,
/// with no thread and no real sleep.
pub(super) fn step<Clk: Clock>(shared: &Shared<Clk>, cursor: &mut SendCursor) {
    let now_nanos = shared.clock.now_nanos();

    if cursor.current.is_none() {
        cursor.current = shared
            .queue
            .pull_batch(shared.config.max_batch_items, shared.config.max_batch_bytes);
        cursor.attempt = 0;
        cursor.prev_delay = Duration::ZERO;
        cursor.next_attempt_at_nanos = now_nanos;
    }

    if now_nanos >= cursor.next_attempt_at_nanos
        && let Some(batch) = cursor.current.as_ref()
    {
        let success = attempt_send(shared, batch);
        if success {
            cursor.current = None;
        } else {
            let rand = attempt_rand();
            let delay =
                backoff(shared).delay(cursor.attempt, Jitter::Full, cursor.prev_delay, rand);
            cursor.prev_delay = delay;
            cursor.attempt = cursor.attempt.saturating_add(1);
            cursor.next_attempt_at_nanos =
                now_nanos.saturating_add(delay.as_nanos().min(u128::from(u64::MAX)) as u64);
        }
    }

    maybe_tick(shared, now_nanos);
}

#[cfg(feature = "std")]
fn attempt_rand() -> u64 {
    fastrand::u64(..)
}

/// Send one attempt. Any `Err` is treated as a transport failure — the
/// downstream handle is torn down and rebuilt before the next attempt, so a
/// dead channel (the classic "looks alive, every send fails" gRPC/H1
/// keepalive failure) is never retried forever against itself. An `Ok`
/// response with a 4xx/5xx status is an application-level rejection from a
/// reachable collector: retried, but WITHOUT reconnecting (the channel is
/// fine).
fn attempt_send<Clk: Clock>(shared: &Shared<Clk>, batch: &PulledBatch) -> bool {
    let request = batch.to_request();
    let handle = shared.current.read().clone();
    let outcome = futures::executor::block_on(handle.call_dyn(request));
    let success = matches!(&outcome, Ok(response) if response.status < 400);
    if success {
        let count = batch.len() as u64;
        shared
            .counters
            .sent_lifetime
            .fetch_add(count, Ordering::Relaxed);
        shared
            .counters
            .sent_since_tick
            .fetch_add(count, Ordering::Relaxed);
        return true;
    }
    shared
        .counters
        .retried_lifetime
        .fetch_add(1, Ordering::Relaxed);
    shared
        .counters
        .retried_since_tick
        .fetch_add(1, Ordering::Relaxed);
    if outcome.is_err() {
        let fresh = (shared.factory)();
        *shared.current.write() = fresh;
        shared
            .counters
            .reconnected_lifetime
            .fetch_add(1, Ordering::Relaxed);
        shared
            .counters
            .reconnected_since_tick
            .fetch_add(1, Ordering::Relaxed);
    }
    false
}

/// Horizon sweep + aggregated drop announcement + self-metrics export, rate
/// limited to `drop_announce_interval` — never runs per-record, only ever
/// once per tick, so a drop storm produces a steady trickle of summaries
/// rather than a flood of log lines.
fn maybe_tick<Clk: Clock>(shared: &Shared<Clk>, now_nanos: u64) {
    let last = shared.last_tick_nanos.load(Ordering::Relaxed);
    let interval_nanos = shared
        .config
        .drop_announce_interval()
        .as_nanos()
        .min(u128::from(u64::MAX)) as u64;
    if last != 0 && now_nanos.saturating_sub(last) < interval_nanos {
        return;
    }
    shared.last_tick_nanos.store(now_nanos, Ordering::Relaxed);
    let elapsed = Duration::from_nanos(now_nanos.saturating_sub(last).max(1));

    for event in shared.queue.sweep_horizons(now_nanos) {
        debug_assert!(
            matches!(event.reason, DropReason::Expired)
                || matches!(event.reason, DropReason::Space)
        );
        shared.since_tick_drops[event.bucket].fetch_add(1, Ordering::Relaxed);
    }

    let drop_deltas: [u64; SEVERITY_BUCKETS] =
        core::array::from_fn(|bucket| shared.since_tick_drops[bucket].swap(0, Ordering::Relaxed));
    let ingest_counts = shared.queue.take_ingest_counts();
    let rates = queue::ingest_rates(&ingest_counts, elapsed);
    let lane_lens = shared.queue.lane_lens();

    export_self_metrics(shared, &drop_deltas, &lane_lens, &rates);
    announce_drops(shared, &drop_deltas, now_nanos, elapsed);
}

fn export_self_metrics<Clk: Clock>(
    shared: &Shared<Clk>,
    drop_deltas: &[u64; SEVERITY_BUCKETS],
    lane_lens: &[usize; SEVERITY_BUCKETS],
    rates: &[f64; SEVERITY_BUCKETS],
) {
    let Some(recorder) = shared.self_recorder() else {
        return;
    };
    for bucket in 0..SEVERITY_BUCKETS {
        if drop_deltas[bucket] > 0 {
            recorder
                .counter(DROPPED_COUNTER_NAMES[bucket])
                .add(drop_deltas[bucket], &[]);
        }
        let survivable = queue::survivable_seconds(
            lane_lens,
            rates,
            shared.queue.capacity(),
            &shared.config.horizons,
            bucket,
        );
        recorder
            .gauge(SURVIVABLE_GAUGE_NAMES[bucket])
            .set_f64(survivable, &[]);
    }
    let sent_delta = shared.counters.sent_since_tick.swap(0, Ordering::Relaxed);
    let retried_delta = shared
        .counters
        .retried_since_tick
        .swap(0, Ordering::Relaxed);
    let reconnected_delta = shared
        .counters
        .reconnected_since_tick
        .swap(0, Ordering::Relaxed);
    if sent_delta > 0 {
        recorder.counter(SENT_COUNTER_NAME).add(sent_delta, &[]);
    }
    if retried_delta > 0 {
        recorder
            .counter(RETRIED_COUNTER_NAME)
            .add(retried_delta, &[]);
    }
    if reconnected_delta > 0 {
        recorder
            .counter(RECONNECTED_COUNTER_NAME)
            .add(reconnected_delta, &[]);
    }
    recorder
        .gauge(BACKLOG_GAUGE_NAME)
        .set_u64(shared.queue.total_len() as u64, &[]);
}

fn announce_drops<Clk: Clock>(
    shared: &Shared<Clk>,
    drop_deltas: &[u64; SEVERITY_BUCKETS],
    now_nanos: u64,
    elapsed: Duration,
) {
    let total: u64 = drop_deltas.iter().sum();
    if total == 0 {
        return;
    }
    let Some(recorder) = shared.self_recorder() else {
        return;
    };
    let oldest_ms = shared.queue.oldest_age_nanos(now_nanos).unwrap_or(0) / 1_000_000;
    recorder
        .log()
        .level(Level::WARN)
        .message("otlp sink shedding telemetry under buffer pressure")
        .tag("dropped_trace", drop_deltas[0])
        .tag("dropped_debug", drop_deltas[1])
        .tag("dropped_info", drop_deltas[2])
        .tag("dropped_warn", drop_deltas[3])
        .tag("dropped_error", drop_deltas[4])
        .tag("backlog", shared.queue.total_len() as u64)
        .tag("oldest_ms", oldest_ms)
        .tag("window_secs", elapsed.as_secs())
        .emit();
}

fn wait_duration<Clk: Clock>(
    shared: &Shared<Clk>,
    cursor: &SendCursor,
    now_nanos: u64,
) -> Duration {
    let mut wait = shared.config.idle_poll();
    if cursor.current.is_some() {
        let until_next = cursor.next_attempt_at_nanos.saturating_sub(now_nanos);
        wait = wait.min(Duration::from_nanos(until_next));
    }
    wait
}

/// The supervised loop: run [`step`] under `catch_unwind` forever (until
/// `shutdown`), then wait for either new work (`notify`) or the next
/// actionable deadline (idle poll / backoff), whichever is sooner.
pub(super) fn run<Clk: Clock + Send + Sync + 'static>(shared: Arc<Shared<Clk>>) {
    let mut cursor = SendCursor::default();
    while !shared.shutdown.load(Ordering::Acquire) {
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            step(&shared, &mut cursor);
        }));
        if outcome.is_err() {
            shared.counters.panics.fetch_add(1, Ordering::Relaxed);
            if let Some(recorder) = shared.self_recorder() {
                recorder.counter(PANIC_COUNTER_NAME).add(1, &[]);
                recorder
                    .log()
                    .level(Level::ERROR)
                    .message("otlp resilient worker iteration panicked; recovered and continuing")
                    .emit();
            }
            // the in-flight batch (if any) is left as-is: the next iteration
            // resumes it rather than silently dropping partially-processed
            // work because of an unrelated panic.
        }

        let now_nanos = shared.clock.now_nanos();
        let wait = wait_duration(&shared, &cursor, now_nanos);
        if wait > Duration::ZERO && !shared.shutdown.load(Ordering::Acquire) {
            futures::executor::block_on(async {
                let notified = core::pin::pin!(shared.notify.notified());
                let timeout = core::pin::pin!(shared.clock.delay(wait));
                futures::future::select(notified, timeout).await;
            });
        }
    }
}
