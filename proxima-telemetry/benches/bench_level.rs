use std::hint::black_box;
use std::str::FromStr;

use criterion::{Criterion, criterion_group, criterion_main};
use opentelemetry::logs::Severity;
use proxima_telemetry::level::Level;

fn bench_proxima_level_cmp(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c3_level");
    group.bench_function("proxima_level_cmp", |bencher| {
        bencher.iter(|| {
            let result = black_box(Level::INFO).cmp(&black_box(Level::WARN));
            black_box(result)
        });
    });
    group.finish();
}

fn bench_proxima_level_from_str(criterion: &mut Criterion) {
    let names = ["trace", "debug", "info", "warn", "error", "fatal"];
    let mut group = criterion.benchmark_group("c3_level");
    group.bench_function("proxima_level_from_str", |bencher| {
        bencher.iter(|| {
            for name in black_box(names) {
                black_box(Level::from_str(name).ok());
            }
        });
    });
    group.finish();
}

fn bench_tracing_level_cmp(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c3_level");
    group.bench_function("tracing_level_cmp", |bencher| {
        bencher.iter(|| {
            let info = black_box(&tracing::Level::INFO);
            let warn = black_box(&tracing::Level::WARN);
            let result = info.partial_cmp(warn);
            black_box(result)
        });
    });
    group.finish();
}

fn bench_tracing_level_from_str(criterion: &mut Criterion) {
    let names = ["TRACE", "DEBUG", "INFO", "WARN", "ERROR"];
    let mut group = criterion.benchmark_group("c3_level");
    group.bench_function("tracing_level_from_str", |bencher| {
        bencher.iter(|| {
            for name in black_box(names) {
                black_box(tracing::Level::from_str(name).ok());
            }
        });
    });
    group.finish();
}

fn bench_log_level_cmp(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c3_level");
    group.bench_function("log_level_cmp", |bencher| {
        bencher.iter(|| {
            let result = black_box(log::Level::Info).cmp(&black_box(log::Level::Warn));
            black_box(result)
        });
    });
    group.finish();
}

fn bench_log_level_from_str(criterion: &mut Criterion) {
    let names = ["trace", "debug", "info", "warn", "error"];
    let mut group = criterion.benchmark_group("c3_level");
    group.bench_function("log_level_from_str", |bencher| {
        bencher.iter(|| {
            for name in black_box(names) {
                black_box(log::Level::from_str(name).ok());
            }
        });
    });
    group.finish();
}

fn bench_opentelemetry_severity_cmp(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c3_level");
    group.bench_function("opentelemetry_severity_cmp", |bencher| {
        bencher.iter(|| {
            let result = black_box(Severity::Info).cmp(&black_box(Severity::Warn));
            black_box(result)
        });
    });
    group.finish();
}

fn bench_proxima_custom_level_construct(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c3_level");
    group.bench_function("proxima_custom_level_construct", |bencher| {
        bencher.iter(|| {
            let level = black_box(Level::custom("audit", 18));
            black_box(level)
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_proxima_level_cmp,
    bench_proxima_level_from_str,
    bench_tracing_level_cmp,
    bench_tracing_level_from_str,
    bench_log_level_cmp,
    bench_log_level_from_str,
    bench_opentelemetry_severity_cmp,
    bench_proxima_custom_level_construct,
);
criterion_main!(benches);
