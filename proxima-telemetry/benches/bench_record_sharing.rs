// P17 bench: drain-time Arc-wrap for records + RecordSharing axis.
//
// Measures the cost of N-way fanout for inline (Vec<T>) vs Arc-shared (Vec<Arc<T>>) batches.
//
// Eight fanout arms at K = 1, 3, 5, 8 consumers:
//
//  1. inline_fanout_1c  — inline batch, K=1. Baseline.
//  2. arc_fanout_1c     — Arc batch, K=1.  Arc::new per record + 1 bump at fanout.
//  3. inline_fanout_3c  — inline batch, K=3. Each record cloned 3 times at fanout.
//  4. arc_fanout_3c     — Arc batch, K=3.  3 Arc bumps per Arc at fanout.
//  5. inline_fanout_5c  — inline batch, K=5.
//  6. arc_fanout_5c     — Arc batch, K=5.
//  7. inline_fanout_8c  — inline batch, K=8.
//  8. arc_fanout_8c     — Arc batch, K=8.
//
// "Fanout" is simulated: one batch request dispatched to K independent CountingPipes.
// For inline, each consumer receives a fresh clone of the Vec<T> (simulating memcpy × K).
// For Arc, each consumer receives a clone of Vec<Arc<T>> — Arc bumps only, no record copy.
//
// Throughput unit: span records per second (N_ITEMS × K / iteration time would overcount;
// we use N_ITEMS — the records emitted — and let median time reflect the K-way cost).

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

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use futures::executor::block_on;
use proxima_primitives::pipe::SendPipe;
use proxima_telemetry::id::{SpanId, TraceId};
use proxima_telemetry::pipes::{CountingPipe, span_batch_arc_request, span_batch_request};
use proxima_telemetry::trace::{SpanKind, SpanRecord, Status, TraceState};
use smallvec::SmallVec;

const N_ITEMS: usize = 10_000;

fn make_span() -> SpanRecord {
    SpanRecord {
        trace_id: TraceId::INVALID,
        span_id: SpanId::INVALID,
        parent_span_id: None,
        name: "bench",
        kind: SpanKind::Internal,
        start_ns: 0,
        duration_ns: 100,
        status: Status::Unset,
        attrs: SmallVec::new(),
        events: SmallVec::new(),
        links: SmallVec::new(),
        tracestate: TraceState::empty(),
        module_path: "bench",
        file_line: (0, 0),
    }
}

fn clone_span(r: &SpanRecord) -> SpanRecord {
    SpanRecord {
        trace_id: r.trace_id,
        span_id: r.span_id,
        parent_span_id: r.parent_span_id,
        name: r.name,
        kind: r.kind,
        start_ns: r.start_ns,
        duration_ns: r.duration_ns,
        status: r.status.clone(),
        attrs: r.attrs.clone(),
        events: r.events.clone(),
        links: r.links.clone(),
        tracestate: r.tracestate.clone(),
        module_path: r.module_path,
        file_line: r.file_line,
    }
}

fn build_counting_pipes(k: usize) -> Vec<CountingPipe> {
    (0..k).map(|_| CountingPipe::new().0).collect()
}

// inline fanout: each consumer receives a cloned Vec<SpanRecord> — simulates memcpy × K
fn fanout_inline(records: &[SpanRecord], pipes: &[CountingPipe]) {
    for pipe in pipes {
        let cloned: Vec<SpanRecord> = records.iter().map(clone_span).collect();
        let req = span_batch_request(cloned);
        block_on(SendPipe::call(pipe, req)).expect("pipe call");
    }
}

// arc fanout: each consumer receives a Vec<Arc<SpanRecord>> clone — Arc bump × K
fn fanout_arc(arced: &[Arc<SpanRecord>], pipes: &[CountingPipe]) {
    for pipe in pipes {
        let req = span_batch_arc_request(arced.to_vec());
        block_on(SendPipe::call(pipe, req)).expect("pipe call");
    }
}

fn bench_sharing_pair(criterion: &mut Criterion, k: usize) {
    let mut group = criterion.benchmark_group("bench_record_sharing");
    group.throughput(Throughput::Elements(N_ITEMS as u64));

    let records: Vec<SpanRecord> = (0..N_ITEMS).map(|_| make_span()).collect();
    let arced: Vec<Arc<SpanRecord>> = (0..N_ITEMS).map(|_| Arc::new(make_span())).collect();
    let inline_pipes = build_counting_pipes(k);
    let arc_pipes = build_counting_pipes(k);

    group.bench_function(
        BenchmarkId::new(format!("inline_fanout_{k}c"), N_ITEMS),
        |bencher| bencher.iter(|| fanout_inline(black_box(&records), &inline_pipes)),
    );

    group.bench_function(
        BenchmarkId::new(format!("arc_fanout_{k}c"), N_ITEMS),
        |bencher| bencher.iter(|| fanout_arc(black_box(&arced), &arc_pipes)),
    );

    group.finish();
}

fn fanout_k1(criterion: &mut Criterion) {
    bench_sharing_pair(criterion, 1);
}
fn fanout_k3(criterion: &mut Criterion) {
    bench_sharing_pair(criterion, 3);
}
fn fanout_k5(criterion: &mut Criterion) {
    bench_sharing_pair(criterion, 5);
}
fn fanout_k8(criterion: &mut Criterion) {
    bench_sharing_pair(criterion, 8);
}

criterion_group!(benches, fanout_k1, fanout_k3, fanout_k5, fanout_k8,);
criterion_main!(benches);
