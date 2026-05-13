use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use opentelemetry::KeyValue;
use proxima_telemetry::tag::{NestedValue, ScalarValue, Tag};

fn bench_proxima_tag_construct_scalar_i64(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c4_attr");
    group.bench_function("proxima_tag_construct_scalar_i64", |bencher| {
        bencher.iter(|| {
            black_box(Tag::Scalar {
                key: black_box("k"),
                value: ScalarValue::I64(black_box(42)),
            })
        });
    });
    group.finish();
}

fn bench_proxima_tag_construct_scalar_str(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c4_attr");
    group.bench_function("proxima_tag_construct_scalar_str", |bencher| {
        bencher.iter(|| {
            black_box(Tag::Scalar {
                key: black_box("k"),
                value: ScalarValue::Str(black_box("v")),
            })
        });
    });
    group.finish();
}

fn bench_proxima_tag_construct_structured_array(criterion: &mut Criterion) {
    static ITEMS: &[NestedValue] = &[
        NestedValue::Scalar(ScalarValue::I64(1)),
        NestedValue::Scalar(ScalarValue::Str("two")),
        NestedValue::Scalar(ScalarValue::Bool(true)),
    ];
    let mut group = criterion.benchmark_group("c4_attr");
    group.bench_function("proxima_tag_construct_structured_array", |bencher| {
        bencher.iter(|| {
            black_box(Tag::Structured {
                key: black_box("dims"),
                value: NestedValue::Array(black_box(ITEMS)),
            })
        });
    });
    group.finish();
}

fn bench_proxima_tag_macro_8_kvs(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c4_attr");
    group.bench_function("proxima_tag_macro_8_kvs", |bencher| {
        bencher.iter(|| {
            let mut sink: Vec<Tag> = Vec::with_capacity(8);
            proxima_telemetry::tag!(
                sink,
                "k0" = black_box(0i64),
                "k1" = black_box("v1"),
                "k2" = black_box(2u64),
                "k3" = black_box(3.0f64),
                "k4" = black_box(true),
                "k5" = black_box(5i64),
                "k6" = black_box("v6"),
                "k7" = black_box(true),
            );
            black_box(sink)
        });
    });
    group.finish();
}

fn bench_opentelemetry_keyvalue_construct(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c4_attr");
    group.bench_function("opentelemetry_keyvalue_construct", |bencher| {
        bencher.iter(|| black_box(KeyValue::new(black_box("k"), black_box(42i64))));
    });
    group.finish();
}

fn bench_tracing_field_value_construct(criterion: &mut Criterion) {
    // tracing uses Box<dyn Value>; this exercises the Box allocation path
    // that the enum-discriminated Tag design is explicitly designed to avoid
    let mut group = criterion.benchmark_group("c4_attr");
    group.bench_function("tracing_field_value_construct", |bencher| {
        bencher.iter(|| {
            let visitor = tracing::field::display(black_box("some-value"));
            black_box(visitor)
        });
    });
    group.finish();
}

fn bench_metrics_label_construct(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c4_attr");
    group.bench_function("metrics_label_construct", |bencher| {
        bencher.iter(|| black_box(metrics::Label::new(black_box("k"), black_box("v"))));
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_proxima_tag_construct_scalar_i64,
    bench_proxima_tag_construct_scalar_str,
    bench_proxima_tag_construct_structured_array,
    bench_proxima_tag_macro_8_kvs,
    bench_opentelemetry_keyvalue_construct,
    bench_tracing_field_value_construct,
    bench_metrics_label_construct,
);
criterion_main!(benches);
