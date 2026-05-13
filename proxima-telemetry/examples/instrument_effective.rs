//! Proof that `#[instrument]` is usable everywhere: one annotation yields a trace
//! span AND a duration metric, the metric records on every thread (not the
//! `count=0` a drain-race hid), it reads back reset-free at the observer tap, and
//! the per-span cost is small enough to carpet a hot path.
//!
//! Validates the `set_default_recorder` fix directly: Section 2 opens the
//! span-metric consumer gate ONLY by installing the recorder as the process
//! default — it never calls `enable_span_metrics`. A nonzero count therefore
//! proves `set_default_recorder` opened the gate. (Ambient `current()` resolution
//! is covered by the `ambient_install_records_instrument_duration` unit test; the
//! macro's `crate::` path can't resolve inside this crate's own examples, so the
//! example passes the recorder explicitly.)
//!
//! Run:
//!   cargo run --release -p proxima-telemetry \
//!       --features instrument-metrics,macros --example instrument_effective

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::hint::black_box;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::thread;
use std::time::Instant;

use proxima_telemetry::capture::capture;
use proxima_telemetry::export::set_default_recorder;
use proxima_telemetry::pipes::NullPipe;
use proxima_telemetry::recorder::Recorder;
use proxima_telemetry::sampler::AlwaysOff;

const THREADS: usize = 4;
const ITERS: u64 = 250_000;

// one annotation → trace span + duration metric, from the single declaration.
#[proxima_telemetry::instrument(name = "load", recorder = rec)]
fn load(rec: &Recorder, id: u64) -> u64 {
    black_box(id).wrapping_mul(2)
}

#[proxima_telemetry::instrument(name = "parse", recorder = rec)]
fn parse(rec: &Recorder, bytes: &[u8]) -> u64 {
    bytes
        .iter()
        .fold(0u64, |acc, byte| acc.wrapping_add(u64::from(*byte)))
}

#[proxima_telemetry::instrument(name = "transform", recorder = rec)]
fn transform(rec: &Recorder, value: u64) -> u64 {
    value.wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(29)
}

// identical bodies, no annotation — the overhead baseline for Section 3.
fn parse_raw(bytes: &[u8]) -> u64 {
    bytes
        .iter()
        .fold(0u64, |acc, byte| acc.wrapping_add(u64::from(*byte)))
}
fn transform_raw(value: u64) -> u64 {
    value.wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(29)
}

fn run_instrumented(recorder: &Recorder, input: &[u8]) {
    thread::scope(|scope| {
        for _ in 0..THREADS {
            scope.spawn(|| {
                let mut acc = 0u64;
                for i in 0..ITERS {
                    let parsed = parse(recorder, black_box(input)).wrapping_add(i);
                    acc = acc.wrapping_add(transform(recorder, black_box(parsed)));
                }
                black_box(acc);
            });
        }
    });
}

fn run_baseline(input: &[u8]) {
    thread::scope(|scope| {
        for _ in 0..THREADS {
            scope.spawn(|| {
                let mut acc = 0u64;
                for i in 0..ITERS {
                    let parsed = parse_raw(black_box(input)).wrapping_add(i);
                    acc = acc.wrapping_add(transform_raw(black_box(parsed)));
                }
                black_box(acc);
            });
        }
    });
}

fn main() {
    println!("proxima #[instrument]: effectiveness proof\n");

    // ── Section 1: three pillars from one annotation ─────────────────────────────
    let captured = capture(|rec| {
        let _ = load(rec, 42);
    });
    let spans = captured.spans().len();
    let metrics = captured.metrics().len();
    println!("[1] one annotation, all pillars (captured in memory):");
    println!("    trace  : {spans} span record (name \"load\")");
    println!("    metric : {metrics} duration histogram");
    assert_eq!(spans, 1, "one span from one annotation");
    assert_eq!(metrics, 1, "one duration metric from the same annotation");

    // ── Section 2: records on every thread, gate opened ONLY by install ──────────
    let recorder = Arc::new(
        Recorder::builder()
            .pipe(NullPipe::new())
            .core_count(THREADS)
            .sampler(AlwaysOff) // trace dropped; metric pillar stays on (metric-only path)
            .start()
            .expect("recorder build"),
    );
    // the ONLY thing that opens the consumer gate — no enable_span_metrics() call.
    set_default_recorder(Arc::clone(&recorder));

    static PARSE_N: AtomicU64 = AtomicU64::new(0);
    static PARSE_NS: AtomicU64 = AtomicU64::new(0);
    static XFORM_N: AtomicU64 = AtomicU64::new(0);
    static XFORM_NS: AtomicU64 = AtomicU64::new(0);
    recorder.set_duration_observer(|name, ns| match name {
        "parse" => {
            PARSE_N.fetch_add(1, Relaxed);
            PARSE_NS.fetch_add(ns, Relaxed);
        }
        "transform" => {
            XFORM_N.fetch_add(1, Relaxed);
            XFORM_NS.fetch_add(ns, Relaxed);
        }
        _ => {}
    });

    let input = [7u8; 64];
    let start = Instant::now();
    run_instrumented(&recorder, &input);
    let instrumented = start.elapsed();

    let expected = THREADS as u64 * ITERS;
    let parse_n = PARSE_N.load(Relaxed);
    let xform_n = XFORM_N.load(Relaxed);
    let recorded_ok = parse_n == expected && xform_n == expected;
    println!(
        "\n[2] recording across {THREADS} threads x {ITERS} iters (observer tap, reset-free):"
    );
    println!(
        "    parse    : count={parse_n:>8}  mean={:>3}ns   (expected {expected})",
        PARSE_NS.load(Relaxed) / parse_n.max(1)
    );
    println!(
        "    transform: count={xform_n:>8}  mean={:>3}ns   (expected {expected})",
        XFORM_NS.load(Relaxed) / xform_n.max(1)
    );
    println!(
        "    -> {}",
        if recorded_ok {
            "every span on every thread recorded, gate opened by install alone — NOT count=0"
        } else {
            "FAIL: counts do not match"
        }
    );

    // ── Section 3: overhead — cheap enough to carpet a hot path ──────────────────
    let start = Instant::now();
    run_baseline(&input);
    let baseline = start.elapsed();

    let spans_total = expected * 2; // parse + transform per iter
    let base_per = baseline.as_nanos() as f64 / spans_total as f64;
    let inst_per = instrumented.as_nanos() as f64 / spans_total as f64;
    println!("\n[3] overhead — one #[instrument] = span + always-on duration metric:");
    println!("    baseline (raw work)   : {base_per:>7.2} ns/call");
    println!("    instrumented (metric) : {inst_per:>7.2} ns/call");
    println!(
        "    added                 : {:>7.2} ns/span",
        inst_per - base_per
    );

    assert!(
        recorded_ok,
        "instrument recording is broken — counts mismatch"
    );
    println!("\nPASS: #[instrument] records on every thread, all pillars, reset-free.");
}
