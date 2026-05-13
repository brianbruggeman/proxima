#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Hot-path overhead gate for error-elevation (disciplined-component gate 13):
//! the log macro's below-floor admit branch must cost a `Cell::get` plus, in
//! the verbose arm, one atomic load — never a sampler recompute, never a map
//! lookup. This bench is the proof.
//!
//! two groups:
//! - `elevation_admit_check`: the bare `current::should_admit_below_floor` call,
//!   not-verbose vs verbose. Isolates exactly the branch the macro adds on the
//!   gate-disabled path; nothing else in the emit pipeline runs.
//! - `elevation_macro_emit`: the same delta through a real `trace!()` call into
//!   a recorder built from `TelemetryConfig`, so the number also carries the
//!   record build + ring push the verbose arm pays that the bare check doesn't.
//!
//! `trace!` is gated off at the callsite in both `elevation_macro_emit` arms
//! (the crate's default floor is `error`) — the only difference between the
//! two arms is whether the below-floor admit branch fires.

use std::hint::black_box;

use conflaguration::Validate;
use criterion::{Criterion, criterion_group, criterion_main};
use proxima_telemetry::config::{
    Elevation, ExporterChoice, OverflowPolicy, Retention, TelemetryConfig,
};
use proxima_telemetry::current;
use proxima_telemetry::id::{SpanId, TraceId};
use proxima_telemetry::level::Level;
use proxima_telemetry::recorder::Recorder;

fn bench_admit_check(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("elevation_admit_check");

    // not-verbose: ratio 0, no trace entered -- is_current_verbose() is a
    // single Cell::get reading false, and `&&` short-circuits before the
    // atomic load.
    current::set_verbose_ratio(0.0);
    current::restore(None);
    group.bench_function("not_verbose", |bencher| {
        bencher.iter(|| black_box(current::should_admit_below_floor(black_box(Level::TRACE))));
    });

    // verbose: ratio 1 and a trace entered -- Cell::get reads true, then one
    // atomic load compares against the elevated admit floor.
    current::set_verbose_ratio(1.0);
    current::set_verbose_admit_floor(Level::TRACE);
    let parent = current::enter(TraceId::from_bytes([7; 16]), SpanId::from_bytes([1; 8]));
    group.bench_function("verbose", |bencher| {
        bencher.iter(|| black_box(current::should_admit_below_floor(black_box(Level::TRACE))));
    });
    current::restore(parent);

    group.finish();
}

fn config_none() -> TelemetryConfig {
    TelemetryConfig::builder()
        .core_count(1)
        .ring_logs(4096)
        .overflow(OverflowPolicy::Drop)
        .build()
}

fn config_verbose(sample_ratio: f64) -> TelemetryConfig {
    TelemetryConfig::builder()
        .core_count(1)
        .ring_logs(4096)
        .overflow(OverflowPolicy::Drop)
        .elevation(Elevation {
            floor: Level::INFO,
            elevated: Some(Level::TRACE),
            sample_ratio,
            trigger_level: Level::ERROR,
            exporter: ExporterChoice::Noop,
            retention: Retention::default(),
        })
        .build()
}

fn bench_macro_emit(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("elevation_macro_emit");

    // the None-runtime form: elevation compiled in but not configured. The
    // simple-form guarantee is that this collapses to the pre-elevation cost.
    let none_config = config_none();
    none_config.validate().expect("none config validates");
    let none_recorder = Recorder::from_config(&none_config).start().expect("recorder");
    current::set_verbose_ratio(0.0);
    current::restore(None);
    group.bench_function("none_not_verbose", |bencher| {
        bencher.iter(|| {
            proxima_telemetry::trace!(
                recorder = &none_recorder,
                value = black_box(1u64),
                "elevation hot path"
            );
            none_recorder.drain();
        });
    });

    // building this recorder from config installs the elevation fan (arm A
    // FloorFilter, arm B ElevationSink) and arms the verbose sampler at
    // ratio 1.0 -- every trace admitted below the callsite floor.
    let verbose_config = config_verbose(1.0);
    verbose_config.validate().expect("verbose config validates");
    let verbose_recorder = Recorder::from_config(&verbose_config).start().expect("recorder");
    let parent = current::enter(TraceId::from_bytes([9; 16]), SpanId::from_bytes([1; 8]));
    group.bench_function("verbose_admit_below_floor", |bencher| {
        bencher.iter(|| {
            proxima_telemetry::trace!(
                recorder = &verbose_recorder,
                value = black_box(1u64),
                "elevation hot path"
            );
            verbose_recorder.drain();
        });
    });
    current::restore(parent);

    group.finish();
}

criterion_group!(benches, bench_admit_check, bench_macro_emit);
criterion_main!(benches);
