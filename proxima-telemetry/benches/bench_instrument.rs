// Perf-seal for the unified-instrument components (C1/C2/C4/C6) under the
// consumer gate. `span_close` is the SAME operation built both ways — run it with
// `--features instrument-metrics` and without; with no metric consumer subscribed
// the delta is just the gate's one relaxed load (the metric must NOT fire when
// nothing receives it). `span_close_consumed` enables a consumer and is the real
// always-on metric cost; the feature-on arms attribute where it goes — the
// metric-only (sampled-out) path and the control-loop observer dispatch.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::default_constructed_unit_structs
)]
use std::hint::black_box;

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use proxima_telemetry::pipes::NullPipe;
use proxima_telemetry::recorder::Recorder;

fn make_recorder() -> Recorder {
    Recorder::builder()
        .pipe(NullPipe::new())
        .core_count(1)
        .start()
        .expect("recorder build failed in bench")
}

// run a span close per timed iteration; drain in (untimed) setup so the ring never
// fills (the timed body is the close, not a drop-on-full).
fn close_loop(group_name: &str, bench_name: &str, recorder: Recorder, criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group(group_name);
    group.bench_function(bench_name, |bencher| {
        bencher.iter_batched(
            || recorder.drain(),
            |_| {
                let guard = recorder.span(black_box("bench_span")).start();
                drop(black_box(guard));
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

// The comparable arm: a span lifecycle with NO metric consumer. Feature-off = pure
// trace. Feature-on = trace + the consumer-gate's one relaxed load, metric skipped.
// The on-minus-off delta is the gate cost — it must be ~zero (no consumer, no fire).
fn span_close(criterion: &mut Criterion) {
    close_loop("instrument", "span_close", make_recorder(), criterion);
}

// A span lifecycle WITH a consumer subscribed (C1/C4): trace + duration histogram
// + exemplar. Minus `span_close` (no consumer) = the always-on metric cost when
// something actually receives it.
#[cfg(feature = "instrument-metrics")]
fn span_close_consumed(criterion: &mut Criterion) {
    let recorder = make_recorder();
    recorder.enable_span_metrics();
    close_loop("instrument", "span_close_consumed", recorder, criterion);
}

// C2 attribution: the metric-only (head-sampled-out) close with a consumer —
// records the duration histogram with no SpanRecord alloc and no ring push. Should
// be CHEAPER than the consumed active close: the always-on metric without the
// trace tax.
#[cfg(feature = "instrument-metrics")]
fn metric_only_close(criterion: &mut Criterion) {
    use proxima_telemetry::sampler::AlwaysOff;

    let recorder = Recorder::builder()
        .pipe(NullPipe::new())
        .core_count(1)
        .sampler(AlwaysOff)
        .start()
        .expect("recorder build failed in bench");
    recorder.enable_span_metrics();
    close_loop("instrument", "metric_only_close", recorder, criterion);
}

// C6 attribution: the consumed active close with a control-loop observer installed
// — the added cost over `span_close_consumed` is the observer dispatch (the
// ArcSwap load + the boxed call) per span.
#[cfg(feature = "instrument-metrics")]
fn observed_close(criterion: &mut Criterion) {
    let recorder = make_recorder();
    recorder.set_duration_observer(|name, duration_ns| {
        black_box((name, duration_ns));
    });
    close_loop("instrument", "observed_close", recorder, criterion);
}

#[cfg(not(feature = "instrument-metrics"))]
fn span_close_consumed(_criterion: &mut Criterion) {}

#[cfg(not(feature = "instrument-metrics"))]
fn metric_only_close(_criterion: &mut Criterion) {}

#[cfg(not(feature = "instrument-metrics"))]
fn observed_close(_criterion: &mut Criterion) {}

criterion_group!(
    benches,
    span_close,
    span_close_consumed,
    metric_only_close,
    observed_close
);
criterion_main!(benches);
