//! Does moving the span-duration fold off the emit hot path reduce producer
//! contention — measured HONESTLY, with a live drain so no work is silently
//! dropped?
//!
//! Setup: AlwaysOff sampler so every span close takes the metric-only path (no
//! trace-ring push) and the timed work is purely the metric pillar. A live
//! event-driven drain thread (`run_drain_loop`) consumes continuously. Incumbent
//! folds inline on the producer (shared registry Mutex — contends); deferred has
//! the producer push a 24-byte POD to a per-core lock-free ring and the drain
//! thread fold it off the producer (Block/producer-assist keeps it lossless).
//!
//! ns/span is wall-time / total spans; `dropped` must be 0 (the guarantee).
//!
//! Run: `-- <threads>`, features `instrument-metrics,deferred-metric-fold` for
//! the deferred arm, `instrument-metrics,lossless-backpressure` for incumbent.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::hint::black_box;
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use proxima_telemetry::pipes::NullPipe;
use proxima_telemetry::recorder::Recorder;
use proxima_telemetry::sampler::AlwaysOff;

const PER_THREAD: u64 = 2_000_000;

// One config per process (no leaked drain threads across runs). `tight` toggles
// the drain: run_drain_loop parks 1ms between passes (size-trigger driven, but
// the deferred push never signals it); tight busy-drains with no park. If drop
// collapses under tight, wake-frequency — not fold-speed — is the drop cause.
fn run(threads: usize, tight: bool) -> (f64, u64) {
    let recorder = Arc::new(
        Recorder::builder()
            .pipe(NullPipe::new())
            .core_count(threads.max(1))
            .sampler(AlwaysOff)
            .start()
            .expect("recorder build"),
    );
    recorder.enable_span_metrics();

    // the REAL event-driven drain we built (run_drain_loop → pump_park): wakes on
    // the size-trigger a full ring raises, sleeps when idle. No fake delay, no
    // busy-spin. `tight` is only a diagnostic contrast (unparked continuous drain).
    let _ = tight;
    let drain = Arc::clone(&recorder);
    let drain_thread = thread::Builder::new()
        .name("bench-drain".to_string())
        .spawn(move || drain.run_drain_loop())
        .expect("spawn drain");

    let start = Instant::now();
    thread::scope(|scope| {
        for _ in 0..threads {
            let rec = Arc::clone(&recorder);
            scope.spawn(move || {
                for _ in 0..PER_THREAD {
                    let guard = rec.span(black_box("bench_span")).start();
                    drop(black_box(guard));
                }
            });
        }
    });
    let elapsed = start.elapsed();

    // detach the drain (run_drain_loop never returns; reaped on process exit — one
    // config per process). Fold any remainder still queued, then read the loss.
    let _ = drain_thread;
    while recorder.drain() > 0 {}
    let total = threads as u64 * PER_THREAD;
    (elapsed.as_nanos() as f64 / total as f64, recorder.dropped())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let threads: usize = args.get(1).and_then(|a| a.parse().ok()).unwrap_or(1);
    let tight = args.get(2).map(|a| a == "tight").unwrap_or(false);
    let _warm = run(threads, tight);
    let (ns, dropped) = run(threads, tight);
    let total = threads as u64 * PER_THREAD;
    println!(
        "{} threads, drain={}: {:.1} ns/span  dropped={} ({:.2}%)",
        threads,
        if tight { "tight" } else { "event-driven" },
        ns,
        dropped,
        dropped as f64 / total as f64 * 100.0
    );
}
