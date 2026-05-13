#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]
use std::cell::RefCell;
use std::hint::black_box;
use std::rc::Rc;

use criterion::{Criterion, criterion_group, criterion_main};
use proxima_telemetry::clock::MonotonicCounter;
use proxima_telemetry::level::Level;
use proxima_telemetry::log::{LogBuilder, LogRecord};

fn make_sink() -> (Rc<RefCell<Vec<LogRecord>>>, impl FnMut(LogRecord)) {
    let collected: Rc<RefCell<Vec<LogRecord>>> = Rc::new(RefCell::new(Vec::new()));
    let inner = Rc::clone(&collected);
    let sink = move |record: LogRecord| {
        black_box(&record);
        inner.borrow_mut().push(record);
    };
    (collected, sink)
}

fn bench_proxima_log_emit_no_attrs(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c8_log");

    group.bench_function("proxima_log_emit_no_attrs", |bencher| {
        let (collected, sink) = make_sink();
        bencher.iter(|| {
            let sink_ref = Rc::clone(&collected);
            LogBuilder::new(
                black_box(Level::INFO),
                move |record: LogRecord| {
                    black_box(&record);
                    sink_ref.borrow_mut().push(record);
                },
                MonotonicCounter::new(0),
            )
            .message(black_box("bench message"))
            .emit();
            collected.borrow_mut().clear();
        });
        drop(sink);
    });
    group.finish();
}

fn bench_proxima_log_emit_4_attrs(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c8_log");

    group.bench_function("proxima_log_emit_4_attrs", |bencher| {
        let (collected, _sink) = make_sink();
        bencher.iter(|| {
            let sink_ref = Rc::clone(&collected);
            let mut builder = LogBuilder::new(
                black_box(Level::INFO),
                move |record: LogRecord| {
                    black_box(&record);
                    sink_ref.borrow_mut().push(record);
                },
                MonotonicCounter::new(0),
            )
            .message(black_box("4 attrs"));
            proxima_telemetry::tag!(
                builder,
                "k0" = black_box(0i64),
                "k1" = black_box("v1"),
                "k2" = black_box(2u64),
                "k3" = black_box(true),
            );
            builder.emit();
            collected.borrow_mut().clear();
        });
    });
    group.finish();
}

fn bench_proxima_log_emit_16_attrs(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c8_log");

    group.bench_function("proxima_log_emit_16_attrs", |bencher| {
        let (collected, _sink) = make_sink();
        bencher.iter(|| {
            let sink_ref = Rc::clone(&collected);
            let mut builder = LogBuilder::new(
                black_box(Level::INFO),
                move |record: LogRecord| {
                    black_box(&record);
                    sink_ref.borrow_mut().push(record);
                },
                MonotonicCounter::new(0),
            )
            .message(black_box("16 attrs overflow path"));
            proxima_telemetry::tag!(
                builder,
                "k0" = black_box(0i64),
                "k1" = black_box(1i64),
                "k2" = black_box(2i64),
                "k3" = black_box(3i64),
                "k4" = black_box(4i64),
                "k5" = black_box(5i64),
                "k6" = black_box(6i64),
                "k7" = black_box(7i64),
                "k8" = black_box(8i64),
                "k9" = black_box(9i64),
                "k10" = black_box(10i64),
                "k11" = black_box(11i64),
                "k12" = black_box(12i64),
                "k13" = black_box(13i64),
                "k14" = black_box(14i64),
                "k15" = black_box(15i64),
            );
            builder.emit();
            collected.borrow_mut().clear();
        });
    });
    group.finish();
}

fn bench_proxima_log_macro_4_attrs(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c8_log");

    group.bench_function("proxima_log_macro_4_attrs", |bencher| {
        let (collected, _sink) = make_sink();
        bencher.iter(|| {
            let sink_ref = Rc::clone(&collected);
            proxima_telemetry::log_record!(
                LogBuilder::new(
                    black_box(Level::INFO),
                    move |record: LogRecord| {
                        black_box(&record);
                        sink_ref.borrow_mut().push(record);
                    },
                    MonotonicCounter::new(0)
                ),
                "msg",
                "k0" = black_box(0i64),
                "k1" = black_box("v1"),
                "k2" = black_box(2u64),
                "k3" = black_box(true),
            );
            collected.borrow_mut().clear();
        });
    });
    group.finish();
}

fn bench_proxima_log_filter_rejected(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c8_log");
    let threshold = Level::ERROR;

    group.bench_function("proxima_log_filter_rejected", |bencher| {
        bencher.iter(|| {
            let level = black_box(Level::DEBUG);
            if level >= threshold {
                let (collected, _sink) = make_sink();
                let sink_ref = Rc::clone(&collected);
                LogBuilder::new(
                    level,
                    move |record: LogRecord| {
                        black_box(&record);
                        sink_ref.borrow_mut().push(record);
                    },
                    MonotonicCounter::new(0),
                )
                .message("filtered")
                .emit();
            }
        });
    });
    group.finish();
}

fn bench_log_record_info(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c8_log");
    group.bench_function("log_record_info", |bencher| {
        bencher.iter(|| {
            log::log!(black_box(log::Level::Info), "bench message");
        });
    });
    group.finish();
}

fn bench_tracing_event_info(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c8_log");
    group.bench_function("tracing_event_info", |bencher| {
        bencher.iter(|| {
            tracing::info!("{}", black_box("bench message"));
        });
    });
    group.finish();
}

fn bench_slog_record_info(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c8_log");
    let drain = slog::Discard;
    let logger = slog::Logger::root(drain, slog::o!());

    group.bench_function("slog_record_info", |bencher| {
        bencher.iter(|| {
            slog::info!(logger, "{}", black_box("bench message"));
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_proxima_log_emit_no_attrs,
    bench_proxima_log_emit_4_attrs,
    bench_proxima_log_emit_16_attrs,
    bench_proxima_log_macro_4_attrs,
    bench_proxima_log_filter_rejected,
    bench_log_record_info,
    bench_tracing_event_info,
    bench_slog_record_info,
);
criterion_main!(benches);
