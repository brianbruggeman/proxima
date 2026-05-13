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
use std::sync::Arc;
use std::thread;

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use proxima_telemetry::pipes::NullPipe;
use proxima_telemetry::recorder::Recorder;

fn make_recorder(core_count: usize) -> Recorder {
    Recorder::builder()
        .pipe(NullPipe::new())
        .core_count(core_count)
        .start()
        .expect("recorder build failed in bench")
}

fn bench_proxima_recorder_emit_span(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c9_recorder");
    let recorder = make_recorder(1);

    group.bench_function("proxima_recorder_emit_span", |bencher| {
        bencher.iter_batched(
            || recorder.drain(),
            |_| {
                let guard = recorder.span(black_box("bench_span")).start();
                black_box(&guard);
                drop(guard);
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

fn bench_proxima_recorder_emit_log(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c9_recorder");
    let recorder = make_recorder(1);

    group.bench_function("proxima_recorder_emit_log", |bencher| {
        bencher.iter_batched(
            || recorder.drain(),
            |_| {
                recorder
                    .log()
                    .message(black_box("bench log message"))
                    .emit();
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

fn bench_proxima_recorder_emit_counter_add(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c9_recorder");
    let recorder = make_recorder(1);

    group.bench_function("proxima_recorder_emit_counter_add", |bencher| {
        bencher.iter_batched(
            || recorder.drain(),
            |_| {
                recorder.emit_counter_add(black_box("bench.counter"), black_box(1u64), &[]);
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

#[cfg(feature = "histogram")]
fn bench_proxima_recorder_emit_histogram_record(criterion: &mut Criterion) {
    use proxima_telemetry::metric::Histogram;

    static BOUNDS: &[f64] = &[1.0, 2.0, 4.0, 8.0, 16.0];
    let mut group = criterion.benchmark_group("c9_recorder");
    let recorder = make_recorder(1);
    let hist: Histogram<f64> = Histogram::new("bench_hist");

    group.bench_function("proxima_recorder_emit_histogram_record", |bencher| {
        bencher.iter_batched(
            || recorder.drain(),
            |_| {
                hist.record(black_box(1.5f64));
                let snap = hist.bucket_snapshot();
                let count = hist.count();
                let sum = hist.sum();
                recorder.emit_histogram_record(
                    black_box("bench_hist"),
                    black_box(count),
                    black_box(sum),
                    snap.to_vec(),
                    BOUNDS,
                    &[],
                );
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

fn bench_proxima_recorder_drain_1000_records(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c9_recorder");
    let recorder = make_recorder(1);

    group.bench_function("proxima_recorder_drain_1000_records", |bencher| {
        bencher.iter(|| {
            for _ in 0..1000 {
                let guard = recorder.span(black_box("bulk_span")).start();
                drop(guard);
            }
            black_box(recorder.drain());
        });
    });
    group.finish();
}

fn bench_proxima_recorder_8_cores_emit_8000(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c9_recorder");
    let recorder = Arc::new(make_recorder(8));

    group.bench_function("proxima_recorder_8_cores_emit_8000", |bencher| {
        bencher.iter(|| {
            let handles: Vec<_> = (0..8)
                .map(|_| {
                    let rec = Arc::clone(&recorder);
                    thread::spawn(move || {
                        for _ in 0..1000 {
                            rec.log().message(black_box("thread log")).emit();
                        }
                    })
                })
                .collect();
            for handle in handles {
                handle.join().expect("bench thread panicked");
            }
            black_box(recorder.drain());
        });
    });
    group.finish();
}

fn bench_tracing_subscriber_emit_span(criterion: &mut Criterion) {
    use tracing_subscriber::Registry;
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;

    let subscriber = Registry::default().with(tracing_subscriber::fmt::layer());
    let _guard = subscriber.set_default();

    let mut group = criterion.benchmark_group("c9_recorder");
    group.bench_function("tracing_subscriber_emit_span", |bencher| {
        bencher.iter(|| {
            let span = tracing::info_span!(black_box("bench_op"));
            let guard = span.enter();
            black_box(&guard);
            drop(guard);
        });
    });
    group.finish();
}

fn bench_opentelemetry_sdk_emit_span(criterion: &mut Criterion) {
    use opentelemetry::trace::{Span, Tracer, TracerProvider};
    use opentelemetry_sdk::trace::{InMemorySpanExporterBuilder, SdkTracerProvider};

    let exporter = InMemorySpanExporterBuilder::new().build();
    let provider = SdkTracerProvider::builder()
        .with_simple_exporter(exporter)
        .build();
    let tracer = provider.tracer("bench");

    let mut group = criterion.benchmark_group("c9_recorder");
    group.bench_function("opentelemetry_sdk_emit_span", |bencher| {
        bencher.iter(|| {
            let mut span = tracer.start(black_box("bench_op"));
            black_box(&span);
            span.end();
        });
    });
    group.finish();
}

#[cfg(not(feature = "histogram"))]
fn bench_proxima_recorder_emit_histogram_record(_criterion: &mut Criterion) {}

criterion_group!(
    benches,
    bench_proxima_recorder_emit_span,
    bench_proxima_recorder_emit_log,
    bench_proxima_recorder_emit_counter_add,
    bench_proxima_recorder_emit_histogram_record,
    bench_proxima_recorder_drain_1000_records,
    bench_proxima_recorder_8_cores_emit_8000,
    bench_tracing_subscriber_emit_span,
    bench_opentelemetry_sdk_emit_span,
);
criterion_main!(benches);
