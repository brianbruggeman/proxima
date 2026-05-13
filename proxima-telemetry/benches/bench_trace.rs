use std::cell::RefCell;
use std::hint::black_box;
use std::rc::Rc;

use criterion::{Criterion, criterion_group, criterion_main};
use proxima_telemetry::id::{SpanId, TraceId};
use proxima_telemetry::trace::{MonotonicCounter, SpanBuilder, SpanRecord};

const TRACE_BYTES: [u8; 16] = [
    0x0a, 0xf7, 0x65, 0x19, 0x16, 0xcd, 0x43, 0xdd, 0x84, 0x48, 0xeb, 0x21, 0x1c, 0x80, 0x31, 0x9c,
];
const SPAN_BYTES: [u8; 8] = [0xb7, 0xad, 0x6b, 0x71, 0x69, 0x20, 0x33, 0x31];

fn make_ids() -> (TraceId, SpanId) {
    (
        TraceId::from_bytes(TRACE_BYTES),
        SpanId::from_bytes(SPAN_BYTES),
    )
}

fn bench_proxima_span_start_close_empty(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c5_trace");
    let clock = MonotonicCounter::new(0);
    let (trace_id, span_id) = make_ids();

    group.bench_function("proxima_span_start_close_empty", |bencher| {
        let collected: Rc<RefCell<Vec<SpanRecord>>> = Rc::new(RefCell::new(Vec::new()));
        bencher.iter(|| {
            let sink_ref = Rc::clone(&collected);
            let guard = SpanBuilder::new(black_box("op"), trace_id, span_id)
                .start(&clock, move |record| {
                    black_box(&record);
                    sink_ref.borrow_mut().push(record);
                })
                .enter(MonotonicCounter::new(1_000_000));
            drop(guard);
            collected.borrow_mut().clear();
        });
    });
    group.finish();
}

fn bench_proxima_span_start_close_1attr(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c5_trace");
    let clock = MonotonicCounter::new(0);
    let (trace_id, span_id) = make_ids();

    group.bench_function("proxima_span_start_close_1attr", |bencher| {
        let collected: Rc<RefCell<Vec<SpanRecord>>> = Rc::new(RefCell::new(Vec::new()));
        bencher.iter(|| {
            let sink_ref = Rc::clone(&collected);
            let guard = SpanBuilder::new(black_box("op"), trace_id, span_id)
                .tag("http.status", black_box(200u64))
                .start(&clock, move |record| {
                    black_box(&record);
                    sink_ref.borrow_mut().push(record);
                })
                .enter(MonotonicCounter::new(1_000_000));
            drop(guard);
            collected.borrow_mut().clear();
        });
    });
    group.finish();
}

fn bench_proxima_span_start_close_8attr(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c5_trace");
    let clock = MonotonicCounter::new(0);
    let (trace_id, span_id) = make_ids();

    group.bench_function("proxima_span_start_close_8attr", |bencher| {
        let collected: Rc<RefCell<Vec<SpanRecord>>> = Rc::new(RefCell::new(Vec::new()));
        bencher.iter(|| {
            let sink_ref = Rc::clone(&collected);
            let guard = SpanBuilder::new(black_box("op"), trace_id, span_id)
                .tag("k0", black_box(0i64))
                .tag("k1", black_box("v1"))
                .tag("k2", black_box(2u64))
                .tag("k3", black_box(3.0f64))
                .tag("k4", black_box(true))
                .tag("k5", black_box(5i64))
                .tag("k6", black_box("v6"))
                .tag("k7", black_box(true))
                .start(&clock, move |record| {
                    black_box(&record);
                    sink_ref.borrow_mut().push(record);
                })
                .enter(MonotonicCounter::new(1_000_000));
            drop(guard);
            collected.borrow_mut().clear();
        });
    });
    group.finish();
}

fn bench_proxima_span_start_close_16attr_overflow(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c5_trace");
    let clock = MonotonicCounter::new(0);
    let (trace_id, span_id) = make_ids();

    group.bench_function("proxima_span_start_close_16attr_overflow", |bencher| {
        let collected: Rc<RefCell<Vec<SpanRecord>>> = Rc::new(RefCell::new(Vec::new()));
        bencher.iter(|| {
            let sink_ref = Rc::clone(&collected);
            let guard = SpanBuilder::new(black_box("op"), trace_id, span_id)
                .tag("k0", black_box(0i64))
                .tag("k1", black_box(1i64))
                .tag("k2", black_box(2i64))
                .tag("k3", black_box(3i64))
                .tag("k4", black_box(4i64))
                .tag("k5", black_box(5i64))
                .tag("k6", black_box(6i64))
                .tag("k7", black_box(7i64))
                .tag("k8", black_box(8i64))
                .tag("k9", black_box(9i64))
                .tag("k10", black_box(10i64))
                .tag("k11", black_box(11i64))
                .tag("k12", black_box(12i64))
                .tag("k13", black_box(13i64))
                .tag("k14", black_box(14i64))
                .tag("k15", black_box(15i64))
                .start(&clock, move |record| {
                    black_box(&record);
                    sink_ref.borrow_mut().push(record);
                })
                .enter(MonotonicCounter::new(1_000_000));
            drop(guard);
            collected.borrow_mut().clear();
        });
    });
    group.finish();
}

fn bench_proxima_span_with_1event(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c5_trace");
    let clock = MonotonicCounter::new(0);
    let (trace_id, span_id) = make_ids();

    group.bench_function("proxima_span_with_1event", |bencher| {
        let collected: Rc<RefCell<Vec<SpanRecord>>> = Rc::new(RefCell::new(Vec::new()));
        bencher.iter(|| {
            let sink_ref = Rc::clone(&collected);
            let mut guard = SpanBuilder::new(black_box("op"), trace_id, span_id)
                .start(&clock, move |record| {
                    black_box(&record);
                    sink_ref.borrow_mut().push(record);
                })
                .enter(MonotonicCounter::new(1_000_000));
            if let Some(evt) = guard.event(black_box("checkpoint")) {
                evt.emit();
            }
            drop(guard);
            collected.borrow_mut().clear();
        });
    });
    group.finish();
}

fn bench_proxima_span_with_1link(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c5_trace");
    let clock = MonotonicCounter::new(0);
    let (trace_id, span_id) = make_ids();

    group.bench_function("proxima_span_with_1link", |bencher| {
        let collected: Rc<RefCell<Vec<SpanRecord>>> = Rc::new(RefCell::new(Vec::new()));
        bencher.iter(|| {
            let sink_ref = Rc::clone(&collected);
            let mut guard = SpanBuilder::new(black_box("op"), trace_id, span_id)
                .start(&clock, move |record| {
                    black_box(&record);
                    sink_ref.borrow_mut().push(record);
                })
                .enter(MonotonicCounter::new(1_000_000));
            guard.link(black_box(trace_id), black_box(span_id));
            drop(guard);
            collected.borrow_mut().clear();
        });
    });
    group.finish();
}

fn bench_tracing_span_start_close_empty(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c5_trace");
    group.bench_function("tracing_span_start_close_empty", |bencher| {
        bencher.iter(|| {
            let span = tracing::span!(tracing::Level::INFO, "op");
            let guard = span.enter();
            black_box(&guard);
            drop(guard);
        });
    });
    group.finish();
}

fn bench_tracing_span_start_close_8attr(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c5_trace");
    group.bench_function("tracing_span_start_close_8attr", |bencher| {
        bencher.iter(|| {
            let span = tracing::span!(
                tracing::Level::INFO,
                "op",
                k0 = black_box(0i64),
                k1 = black_box("v1"),
                k2 = black_box(2u64),
                k3 = black_box(3.0f64),
                k4 = black_box(true),
                k5 = black_box(5i64),
                k6 = black_box("v6"),
                k7 = black_box(true),
            );
            let guard = span.enter();
            black_box(&guard);
            drop(guard);
        });
    });
    group.finish();
}

fn bench_opentelemetry_tracer_start_8attr(criterion: &mut Criterion) {
    use opentelemetry::KeyValue;
    use opentelemetry::trace::{Tracer, TracerProvider};

    let provider = opentelemetry::global::tracer_provider();
    let tracer = provider.tracer("bench");

    let mut group = criterion.benchmark_group("c5_trace");
    group.bench_function("opentelemetry_tracer_start_8attr", |bencher| {
        bencher.iter(|| {
            use opentelemetry::trace::Span;
            let mut span = tracer.start(black_box("op"));
            span.set_attribute(KeyValue::new("k0", black_box(0i64)));
            span.set_attribute(KeyValue::new("k1", black_box("v1")));
            span.set_attribute(KeyValue::new("k2", black_box(2i64)));
            span.set_attribute(KeyValue::new("k3", black_box(3.0f64)));
            span.set_attribute(KeyValue::new("k4", black_box(true)));
            span.set_attribute(KeyValue::new("k5", black_box(5i64)));
            span.set_attribute(KeyValue::new("k6", black_box("v6")));
            span.set_attribute(KeyValue::new("k7", black_box(true)));
            span.end();
            black_box(span)
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_proxima_span_start_close_empty,
    bench_proxima_span_start_close_1attr,
    bench_proxima_span_start_close_8attr,
    bench_proxima_span_start_close_16attr_overflow,
    bench_proxima_span_with_1event,
    bench_proxima_span_with_1link,
    bench_tracing_span_start_close_empty,
    bench_tracing_span_start_close_8attr,
    bench_opentelemetry_tracer_start_8attr,
);
criterion_main!(benches);
