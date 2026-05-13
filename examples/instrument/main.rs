//! `logs`, `metrics`, and `traces` each taught one pillar. This example is the
//! unification: one `#[proxima::telemetry::instrument]` annotation opens a
//! span (trace) around a function's body AND, when a metric consumer is
//! listening, folds that span's duration into a per-name histogram (metric)
//! -- the exact expansion `proxima-macros/src/span_attr.rs` produces (see
//! `traces.rs`'s `validate_request` for the same macro without the metric
//! feature on). A log emitted from inside the body lands in the SAME
//! `Recorder`, through the SAME pipe -- the third pillar costs nothing extra
//! to wire, because there was never a second telemetry system to wire it into.
//!
//! The honest part: the metric pillar is NOT unconditional. It is a real,
//! deliberate consumer gate (`Recorder::span_metrics_live`, C2 in
//! `docs/unified-instrument/discipline.md`) -- a span's duration is folded
//! into a histogram only when something has subscribed to read it
//! (`enable_span_metrics()`, or the ambient install path, which calls it for
//! you). Section 2 below proves both sides of that gate on the identical
//! annotated function: same span, same log, metric count 0 vs 1.
//!
//! Run: `cargo run --example instrument --features instrument-metrics`
//! (`macros` is already in the default feature set; `instrument-metrics` is
//! the default-off firewall this example exists to demonstrate, so it is not.)

#![allow(clippy::unwrap_used, clippy::expect_used)]

use proxima::telemetry::capture::capture;
use proxima::telemetry::export::Exporter;
use proxima::telemetry::pipes::InMemoryPipe;
use proxima::telemetry::recorder::Recorder;

/// One annotation, wrapping one function body. `recorder = rec` makes the
/// target recorder explicit data (proxima never resolves a span's home via
/// ambient lookup unless the caller asks it to -- see `traces.README.md`),
/// so every section below can prove its point against a recorder it fully
/// controls, without process-wide `set_default_recorder` side effects leaking
/// between sections.
#[proxima::telemetry::instrument(name = "load_user", recorder = rec)]
fn load_user(rec: &Recorder, user_id: u64) -> u64 {
    // the log pillar: not something #[instrument] emits for you, but a plain
    // call against the SAME recorder the span above just opened -- no second
    // telemetry system, no separate sink to wire, no correlation machinery to
    // bolt on. It shows up in whatever this recorder's pipe is, right next to
    // the span and (if the gate is open) the duration metric.
    rec.log()
        .message("loaded user record")
        .tag("user_id", user_id)
        .emit();
    user_id * 2
}

fn main() {
    println!("proxima #[instrument]: one annotation, three pillars\n");

    println!("--- 1. one call, one capture: span + log + metric together ---");
    run_one_annotation_three_pillars();

    println!("\n--- 2. the metric pillar is a consumer gate, not unconditional ---");
    run_consumer_gate_proof();

    println!("\n--- 3. reading it back on stdout (console + in-memory, same recorder) ---");
    run_stdout_and_memory();

    println!("\nPASS: #[instrument] always yields span + log through the shared recorder;");
    println!("      the duration metric is real and correct, but opt-in via the C2");
    println!("      consumer gate (enable_span_metrics / install) -- not automatic.");
}

// ── 1. capture() subscribes to every pillar, so all three show up together ──

fn run_one_annotation_three_pillars() {
    // capture() is the 3-line test helper (proxima_telemetry::capture): a
    // private recorder, EVERY signal buffered in memory, and -- because it IS
    // a span-metric consumer -- the gate this example is about is already open
    // here. See section 2 for what happens when nothing opens it.
    let captured = capture(|rec| {
        let _ = load_user(rec, 42);
    });

    let spans = captured.spans();
    let logs = captured.logs();
    let metrics = captured.metrics();

    println!(
        "  trace  (span)      : {} record(s), name={:?}",
        spans.len(),
        spans.first().map(|span| span.name)
    );
    println!("  log                : {} record(s)", logs.len());
    println!("  metric (histogram) : {} record(s)", metrics.len());

    assert_eq!(
        spans.len(),
        1,
        "one #[instrument] call opens exactly one span"
    );
    assert_eq!(
        logs.len(),
        1,
        "the explicit rec.log() call inside the body lands in the same recorder"
    );
    assert_eq!(
        metrics.len(),
        1,
        "capture() is a span-metric consumer, so the SAME annotation's close also folds a duration histogram"
    );

    println!("  -> ONE #[instrument] annotation produced all three signal kinds from ONE call.");
}

// ── 2. the gap, told precisely: the metric pillar needs a consumer ──────────

fn run_consumer_gate_proof() {
    // closed: a plain recorder, started but never told anyone wants its
    // metrics. Not a stub, not broken -- `Recorder::span_metrics_live` (C2:
    // "the always-on metric is on only when observed") defaults OFF at
    // build-time (`sized::INSTRUMENT_METRICS_DEFAULT`) and nothing here flips it.
    let closed_pipe = InMemoryPipe::new();
    let closed_recorder = Recorder::builder()
        .pipe(closed_pipe.clone())
        .core_count(1)
        .start()
        .expect("closed recorder builds");
    let _ = load_user(&closed_recorder, 7);
    closed_recorder.drain();

    println!(
        "  gate closed : span={} log={} metric={}",
        closed_pipe.spans().len(),
        closed_pipe.logs().len(),
        closed_pipe.metrics().len()
    );
    assert_eq!(
        closed_pipe.spans().len(),
        1,
        "trace pillar is unconditional"
    );
    assert_eq!(closed_pipe.logs().len(), 1, "log pillar is unconditional");
    assert_eq!(
        closed_pipe.metrics().len(),
        0,
        "metric pillar: SAME annotation, SAME call, but no consumer subscribed -> nothing folded"
    );

    // open: the identical function, the identical call, one extra line.
    let open_pipe = InMemoryPipe::new();
    let open_recorder = Recorder::builder()
        .pipe(open_pipe.clone())
        .core_count(1)
        .start()
        .expect("open recorder builds");
    open_recorder.enable_span_metrics();
    let _ = load_user(&open_recorder, 7);
    open_recorder.drain();

    println!(
        "  gate open   : span={} log={} metric={}",
        open_pipe.spans().len(),
        open_pipe.logs().len(),
        open_pipe.metrics().len()
    );
    assert_eq!(open_pipe.spans().len(), 1);
    assert_eq!(open_pipe.logs().len(), 1);
    assert_eq!(
        open_pipe.metrics().len(),
        1,
        "one call -- .enable_span_metrics() -- is the entire difference"
    );

    println!(
        "  -> trace and log pillars are unconditional; the metric pillar is a real, deliberate"
    );
    println!(
        "     opt-in (Recorder::enable_span_metrics / .install()), not a hidden bug and not automatic."
    );
}

// ── 3. a human reads it off stdout; drain's count proves it actually ran ────

fn run_stdout_and_memory() {
    // .install() registers this recorder as the process default via
    // set_default_recorder, which -- unlike a bare .start() -- calls
    // enable_span_metrics() for you. So the console output below carries all
    // three record kinds without a separate enable call.
    let recorder = Recorder::builder()
        .export(Exporter::stdout())
        .expect("stdout exporter composes")
        .core_count(1)
        .install()
        .expect("recorder installs as the process default");

    let _ = load_user(&recorder, 99);
    let drained = recorder.drain();

    println!("  (above: the console lines a human sees for one #[instrument] call)");
    println!("  drained {drained} telemetry records from the installed recorder");
    assert!(
        drained >= 2,
        "at least the log and the span reached the console sink"
    );
}
