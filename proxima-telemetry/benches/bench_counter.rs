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
use proxima_telemetry::metric::Counter;
use proxima_telemetry::tag::{ScalarValue, Tag};

static COUNTER: Counter = Counter::new("bench_counter");

fn bench_proxima_counter_add_no_tags(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c6_counter");
    group.bench_function("proxima_counter_add_no_tags", |bencher| {
        bencher.iter(|| {
            black_box(&COUNTER).add(black_box(1), &[]);
        });
    });
    group.finish();
}

fn bench_proxima_counter_add_1_tag(criterion: &mut Criterion) {
    let tag = Tag::Scalar {
        key: "route",
        value: ScalarValue::Str("/v1/x"),
    };
    let mut group = criterion.benchmark_group("c6_counter");
    group.bench_function("proxima_counter_add_1_tag", |bencher| {
        bencher.iter(|| {
            black_box(&COUNTER).add(black_box(1), black_box(std::slice::from_ref(&tag)));
        });
    });
    group.finish();
}

fn bench_proxima_counter_add_8_tags(criterion: &mut Criterion) {
    let tags = [
        Tag::Scalar {
            key: "k0",
            value: ScalarValue::I64(0),
        },
        Tag::Scalar {
            key: "k1",
            value: ScalarValue::Str("v1"),
        },
        Tag::Scalar {
            key: "k2",
            value: ScalarValue::U64(2),
        },
        Tag::Scalar {
            key: "k3",
            value: ScalarValue::F64(3.0),
        },
        Tag::Scalar {
            key: "k4",
            value: ScalarValue::Bool(true),
        },
        Tag::Scalar {
            key: "k5",
            value: ScalarValue::I64(5),
        },
        Tag::Scalar {
            key: "k6",
            value: ScalarValue::Str("v6"),
        },
        Tag::Scalar {
            key: "k7",
            value: ScalarValue::Bool(false),
        },
    ];
    let mut group = criterion.benchmark_group("c6_counter");
    group.bench_function("proxima_counter_add_8_tags", |bencher| {
        bencher.iter(|| {
            black_box(&COUNTER).add(black_box(1), black_box(&tags));
        });
    });
    group.finish();
}

fn bench_proxima_counter_macro_no_tags(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c6_counter");
    group.bench_function("proxima_counter_macro_no_tags", |bencher| {
        bencher.iter(|| {
            proxima_telemetry::counter!(COUNTER, black_box(1u64));
        });
    });
    group.finish();
}

fn bench_proxima_counter_macro_8_tags(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c6_counter");
    group.bench_function("proxima_counter_macro_8_tags", |bencher| {
        bencher.iter(|| {
            proxima_telemetry::counter!(
                COUNTER,
                black_box(1u64),
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

fn bench_metrics_counter_increment(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c6_counter");
    group.bench_function("metrics_counter_increment", |bencher| {
        bencher.iter(|| {
            metrics::counter!(black_box("bench.hits")).increment(black_box(1));
        });
    });
    group.finish();
}

fn bench_opentelemetry_counter_add(criterion: &mut Criterion) {
    use opentelemetry::metrics::{Meter, MeterProvider};

    let provider = opentelemetry::metrics::noop::NoopMeterProvider::new();
    let meter: Meter = provider.meter("bench");
    let counter = meter.u64_counter("bench_hits").build();

    let mut group = criterion.benchmark_group("c6_counter");
    group.bench_function("opentelemetry_counter_add", |bencher| {
        bencher.iter(|| {
            black_box(&counter).add(black_box(1), &[]);
        });
    });
    group.finish();
}

fn bench_prometheus_intcounter_inc(criterion: &mut Criterion) {
    let counter = prometheus::IntCounter::new("bench_hits", "bench hits").expect("valid");
    let mut group = criterion.benchmark_group("c6_counter");
    group.bench_function("prometheus_intcounter_inc", |bencher| {
        bencher.iter(|| {
            black_box(&counter).inc();
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_proxima_counter_add_no_tags,
    bench_proxima_counter_add_1_tag,
    bench_proxima_counter_add_8_tags,
    bench_proxima_counter_macro_no_tags,
    bench_proxima_counter_macro_8_tags,
    bench_metrics_counter_increment,
    bench_opentelemetry_counter_add,
    bench_prometheus_intcounter_inc,
);
criterion_main!(benches);
