#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]
use std::hint::black_box;

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use proxima_macros::span;
use proxima_telemetry::pipes::NullPipe;
use proxima_telemetry::recorder::Recorder;
use tracing::instrument;

fn make_recorder() -> Recorder {
    Recorder::builder()
        .pipe(NullPipe::new())
        .core_count(1)
        .start()
        .expect("recorder build failed in bench")
}

// ---- proxima_span_macro_expansion ----
// #[span] on a hot-path function: macro overhead vs. manual guard.

#[span(recorder = recorder)]
fn proxima_spanned(recorder: &Recorder, value: u64) -> u64 {
    black_box(value.wrapping_add(1))
}

fn bench_proxima_span_macro_expansion(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c14_macros");
    let recorder = make_recorder();

    group.bench_function("proxima_span_macro_expansion", |bencher| {
        bencher.iter_batched(
            || recorder.drain(),
            |_| {
                black_box(proxima_spanned(black_box(&recorder), black_box(42u64)));
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

// ---- manual_no_macro ----
// Same fn, same work, manual guard inline. This is the floor.

fn manual_no_macro_fn(recorder: &Recorder, value: u64) -> u64 {
    let _guard = recorder.span("manual_no_macro_fn").start();
    black_box(value.wrapping_add(1))
}

fn bench_manual_no_macro(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c14_macros");
    let recorder = make_recorder();

    group.bench_function("manual_no_macro", |bencher| {
        bencher.iter_batched(
            || recorder.drain(),
            |_| {
                black_box(manual_no_macro_fn(black_box(&recorder), black_box(42u64)));
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

// ---- tracing_instrument_macro ----
// tracing's #[instrument] on an equivalent fn. home-turf arm.

#[instrument]
fn tracing_instrumented(value: u64) -> u64 {
    black_box(value.wrapping_add(1))
}

fn bench_tracing_instrument_macro(criterion: &mut Criterion) {
    let subscriber = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::TRACE)
        .with_writer(std::io::sink)
        .finish();
    let _guard = tracing::subscriber::set_default(subscriber);

    let mut group = criterion.benchmark_group("c14_macros");

    group.bench_function("tracing_instrument_macro", |bencher| {
        bencher.iter(|| {
            black_box(tracing_instrumented(black_box(42u64)));
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_proxima_span_macro_expansion,
    bench_manual_no_macro,
    bench_tracing_instrument_macro,
);
criterion_main!(benches);
