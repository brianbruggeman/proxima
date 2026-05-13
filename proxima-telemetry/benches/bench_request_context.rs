#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

//! `RequestContext::default()` per-call cost bench.
//!
//! Three arms:
//!
//! 1. `proxima_request_context_default` (design-favors: proxima) —
//!    `RequestContext::default()` × N; this is the path Phase 2 needs to
//!    keep cheap because the drainer constructs one per record (or per
//!    batch) on the hot path. After the static-OnceLock opt:
//!    - telemetry handle: 1 Arc clone (atomic increment)
//!    - cancel signal: level-vec clone (Signal is Vec<Arc<Level>>-backed)
//!    - extra_labels: const empty
//!    - path_params: BTreeMap::new() (zero-alloc empty tree)
//!
//!    Total per-call: ~2 atomic increments, zero heap.
//!
//! 2. `tracing_event_dispatch` (design-favors: incumbent) —
//!    Full `tracing_subscriber::registry().with(filter).with(fmt::Layer)`
//!    stack, then drive `tracing::info!(field=value, "msg")` × N. This
//!    is the per-event envelope cost in their model: stack-allocated
//!    Event ref dispatched through a Layer chain.
//!
//! 3. `arc_clone_baseline` (design-favors: neutral) —
//!    Bare `let _ = handle.clone()` on an `Arc<dyn Telemetry>`. The
//!    irreducible atomic-increment cost; proxima's default must be
//!    within ~2× of this.

use std::sync::Arc;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use proxima_primitives::pipe::request::RequestContext;
use proxima_telemetry::{NoopTelemetry, TelemetryHandle};

const N_ITEMS: usize = 10_000;

// design-favors: proxima — post-OnceLock-opt envelope construction.
fn proxima_request_context_default(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("proxima_request_context_default");
    group.throughput(Throughput::Elements(N_ITEMS as u64));
    group.bench_function(BenchmarkId::from_parameter(N_ITEMS), |bencher| {
        bencher.iter(|| {
            for _ in 0..N_ITEMS {
                let context = RequestContext::default();
                std::hint::black_box(context);
            }
        });
    });
    group.finish();
}

// design-favors: incumbent — tracing's per-event dispatch through a
// full installed Subscriber stack. They get to keep the Event on the
// stack; we have an owned envelope.
fn tracing_event_dispatch(criterion: &mut Criterion) {
    use tracing_subscriber::EnvFilter;
    use tracing_subscriber::prelude::*;

    let mut group = criterion.benchmark_group("tracing_event_dispatch");
    group.throughput(Throughput::Elements(N_ITEMS as u64));

    // install once per bench invocation. try_init so the install survives
    // across benches in the same process; subsequent benches re-use it.
    let _ = tracing_subscriber::registry()
        .with(EnvFilter::new("error,proxima=off"))
        .with(
            tracing_subscriber::fmt::layer()
                .with_writer(std::io::sink)
                .with_target(false),
        )
        .try_init();

    group.bench_function(BenchmarkId::from_parameter(N_ITEMS), |bencher| {
        bencher.iter(|| {
            for index in 0..N_ITEMS {
                // emit at error so the filter doesn't cut; writer is sink
                // so we measure dispatch not I/O.
                tracing::error!(item = index, label = "bench", "throughput probe");
            }
        });
    });
    group.finish();
}

// design-favors: neutral — irreducible per-call Arc clone cost. Our
// `RequestContext::default()` is mostly two of these.
fn arc_clone_baseline(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("arc_clone_baseline");
    group.throughput(Throughput::Elements(N_ITEMS as u64));

    let handle: TelemetryHandle = Arc::new(NoopTelemetry);
    group.bench_function(BenchmarkId::from_parameter(N_ITEMS), |bencher| {
        bencher.iter(|| {
            for _ in 0..N_ITEMS {
                let cloned = handle.clone();
                std::hint::black_box(cloned);
            }
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    proxima_request_context_default,
    tracing_event_dispatch,
    arc_clone_baseline,
);
criterion_main!(benches);
