#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]
// P4 pipe dispatch bench.
//
// Home-turf incumbent: the old `Exporter` trait — sync method calls dispatched
// directly to `export_span`, `export_log`, etc. via vtable.
//
// Challenger: `Pipe::call(request).await` via `futures::executor::block_on`
// over a `NullPipe` or `CountingPipe` terminal — one typed-body Arc allocation
// per call plus the async state machine overhead.
//
// Comparison arm: a raw sync function call as the absolute floor — what the
// incumbent's best case is (trait vtable, no alloc, no future).
//
// Design-favors labels:
//  - `proxima_pipe_null_dispatch`   — design-favors: proxima (new path)
//  - `proxima_pipe_counting_dispatch` — design-favors: proxima (new path with atomics)
//  - `raw_sync_fn_floor`            — design-favors: incumbent floor (no alloc, no future)
//  - `old_exporter_vtable_floor`    — design-favors: incumbent (pure vtable, pre-P4 shape)

use std::hint::black_box;
use std::sync::atomic::Ordering;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use futures::executor::block_on;

use proxima_primitives::pipe::SendPipe;
use proxima_telemetry::id::{SpanId, TraceId};
use proxima_telemetry::pipes::{CountingPipe, NullPipe, span_batch_request, span_request};
use proxima_telemetry::trace::Status;
use proxima_telemetry::trace::{SpanKind, SpanRecord, TraceState};

fn make_span_record() -> SpanRecord {
    SpanRecord {
        trace_id: TraceId::INVALID,
        span_id: SpanId::INVALID,
        parent_span_id: None,
        name: "bench_span",
        kind: SpanKind::Internal,
        start_ns: 0,
        duration_ns: 100,
        status: Status::Unset,
        attrs: Default::default(),
        events: Default::default(),
        links: Default::default(),
        tracestate: TraceState::empty(),
        module_path: "bench",
        file_line: (0, 0),
    }
}

// raw sync floor — what a direct function call costs with no indirection.
// this is the absolute lower bound: no vtable, no future, no alloc.
fn raw_sync_noop(_record: &SpanRecord) {}

// simulate the old Exporter trait vtable call shape — sync, per-record ref
trait LegacyExporter: Send + Sync {
    fn export_span(&self, record: &SpanRecord);
}

struct NullLegacyExporter;

impl LegacyExporter for NullLegacyExporter {
    fn export_span(&self, _record: &SpanRecord) {}
}

fn bench_proxima_null_pipe_dispatch(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("p4_pipe_dispatch");
    let pipe = NullPipe::new();

    group.bench_function("proxima_pipe_null_dispatch", |bencher| {
        bencher.iter(|| {
            let request = span_request(black_box(SpanRecord {
                trace_id: TraceId::INVALID,
                span_id: SpanId::INVALID,
                parent_span_id: None,
                name: "bench_span",
                kind: SpanKind::Internal,
                start_ns: 0,
                duration_ns: 100,
                status: Status::Unset,
                attrs: Default::default(),
                events: Default::default(),
                links: Default::default(),
                tracestate: TraceState::empty(),
                module_path: "bench",
                file_line: (0, 0),
            }));
            let _ = block_on(SendPipe::call(&pipe, request));
        });
    });
    group.finish();
}

fn bench_proxima_counting_pipe_dispatch(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("p4_pipe_dispatch");
    let (pipe, spans, _, _, _, _) = CountingPipe::new();

    group.bench_function("proxima_pipe_counting_dispatch", |bencher| {
        bencher.iter(|| {
            let request = span_request(black_box(SpanRecord {
                trace_id: TraceId::INVALID,
                span_id: SpanId::INVALID,
                parent_span_id: None,
                name: "bench_span",
                kind: SpanKind::Internal,
                start_ns: 0,
                duration_ns: 100,
                status: Status::Unset,
                attrs: Default::default(),
                events: Default::default(),
                links: Default::default(),
                tracestate: TraceState::empty(),
                module_path: "bench",
                file_line: (0, 0),
            }));
            let _ = block_on(SendPipe::call(&pipe, request));
            black_box(spans.load(Ordering::Relaxed));
        });
    });
    group.finish();
}

fn bench_raw_sync_floor(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("p4_pipe_dispatch");
    let record = make_span_record();

    group.bench_function("raw_sync_fn_floor", |bencher| {
        bencher.iter(|| {
            raw_sync_noop(black_box(&record));
        });
    });
    group.finish();
}

fn bench_old_exporter_vtable_floor(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("p4_pipe_dispatch");
    let exporter: &dyn LegacyExporter = &NullLegacyExporter;
    let record = make_span_record();

    group.bench_function("old_exporter_vtable_floor", |bencher| {
        bencher.iter(|| {
            exporter.export_span(black_box(&record));
        });
    });
    group.finish();
}

fn make_batch_1000() -> Vec<SpanRecord> {
    (0..1000)
        .map(|_| SpanRecord {
            trace_id: TraceId::INVALID,
            span_id: SpanId::INVALID,
            parent_span_id: None,
            name: "bench_span",
            kind: SpanKind::Internal,
            start_ns: 0,
            duration_ns: 100,
            status: Status::Unset,
            attrs: Default::default(),
            events: Default::default(),
            links: Default::default(),
            tracestate: TraceState::empty(),
            module_path: "bench",
            file_line: (0, 0),
        })
        .collect()
}

fn bench_proxima_null_pipe_batch_1000(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("p4_pipe_dispatch");
    let pipe = NullPipe::new();

    group.throughput(Throughput::Elements(1000));
    group.bench_function("proxima_pipe_null_batch_1000", |bencher| {
        bencher.iter(|| {
            let request = span_batch_request(black_box(make_batch_1000()));
            let _ = block_on(SendPipe::call(&pipe, request));
        });
    });
    group.finish();
}

fn bench_proxima_counting_pipe_batch_1000(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("p4_pipe_dispatch");
    let (pipe, spans, _, _, _, _) = CountingPipe::new();

    group.throughput(Throughput::Elements(1000));
    group.bench_function("proxima_pipe_counting_batch_1000", |bencher| {
        bencher.iter(|| {
            let request = span_batch_request(black_box(make_batch_1000()));
            let _ = block_on(SendPipe::call(&pipe, request));
            let count = black_box(spans.load(Ordering::Relaxed));
            assert!(count > 0);
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_proxima_null_pipe_dispatch,
    bench_proxima_counting_pipe_dispatch,
    bench_raw_sync_floor,
    bench_old_exporter_vtable_floor,
    bench_proxima_null_pipe_batch_1000,
    bench_proxima_counting_pipe_batch_1000,
);
criterion_main!(benches);
