#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]
extern crate alloc;

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use proxima_telemetry::metric::Histogram;

static HIST: Histogram<f64> = Histogram::new("bench_histogram");

fn bench_proxima_histogram_record_no_tags(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c7_histogram");
    group.bench_function("proxima_histogram_record_no_tags", |bencher| {
        bencher.iter(|| {
            black_box(&HIST).record(black_box(1.5f64));
        });
    });
    group.finish();
}

fn bench_proxima_histogram_macro_no_tags(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c7_histogram");
    group.bench_function("proxima_histogram_macro_no_tags", |bencher| {
        bencher.iter(|| {
            proxima_telemetry::histogram!(HIST, black_box(1.5f64));
        });
    });
    group.finish();
}

fn bench_proxima_histogram_macro_8_tags(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c7_histogram");
    group.bench_function("proxima_histogram_macro_8_tags", |bencher| {
        bencher.iter(|| {
            proxima_telemetry::histogram!(
                HIST,
                black_box(1.5f64),
                "k0" = black_box(0i64),
                "k1" = black_box("v1"),
                "k2" = black_box(2u64),
                "k3" = black_box(3.0f64),
                "k4" = black_box(true),
                "k5" = black_box(5i64),
                "k6" = black_box("v6"),
                "k7" = black_box(false),
            );
        });
    });
    group.finish();
}

fn bench_hdrhistogram_record(criterion: &mut Criterion) {
    let mut hist =
        hdrhistogram::Histogram::<u64>::new_with_bounds(1, 60_000_000, 3).expect("valid bounds");
    let mut group = criterion.benchmark_group("c7_histogram");
    group.bench_function("hdrhistogram_record", |bencher| {
        bencher.iter(|| {
            black_box(&mut hist).record(black_box(1_500u64)).ok();
        });
    });
    group.finish();
}

fn bench_prometheus_histogram_observe(criterion: &mut Criterion) {
    let opts = prometheus::HistogramOpts::new("bench_hist", "bench histogram");
    let hist = prometheus::Histogram::with_opts(opts).expect("valid opts");
    let mut group = criterion.benchmark_group("c7_histogram");
    group.bench_function("prometheus_histogram_observe", |bencher| {
        bencher.iter(|| {
            black_box(&hist).observe(black_box(1.5));
        });
    });
    group.finish();
}

fn bench_opentelemetry_histogram_record(criterion: &mut Criterion) {
    use opentelemetry::metrics::{Meter, MeterProvider};

    let provider = opentelemetry::metrics::noop::NoopMeterProvider::new();
    let meter: Meter = provider.meter("bench");
    let hist = meter.f64_histogram("bench_hist").build();

    let mut group = criterion.benchmark_group("c7_histogram");
    group.bench_function("opentelemetry_histogram_record", |bencher| {
        bencher.iter(|| {
            black_box(&hist).record(black_box(1.5), &[]);
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_proxima_histogram_record_no_tags,
    bench_proxima_histogram_macro_no_tags,
    bench_proxima_histogram_macro_8_tags,
    bench_hdrhistogram_record,
    bench_prometheus_histogram_observe,
    bench_opentelemetry_histogram_record,
);
criterion_main!(benches);
