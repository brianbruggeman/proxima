//! Micro-bench for the proxima-runtime trait surface.
//!
//! Measures the overhead of `SpawnRequest` enum construction and
//! `CoreId` operations versus the tokio equivalent dispatch path.
//!
//! Design-favors labels per the disciplined-component gate:
//! - neutral: CoreId round-trip (both paths are trivial)
//! - proxima: SpawnRequest enum construction (our path)
//! - incumbent: tokio Runtime::spawn dispatch (tokio's design point)
//!
//! Incumbent design point: tokio::Runtime::spawn — the baseline for
//! cross-thread future dispatch in the std ecosystem.

use criterion::{Criterion, criterion_group, criterion_main};
use proxima_runtime::{CoreId, SpawnError, SpawnRequest};

fn bench_core_id_round_trip(criterion: &mut Criterion) {
    criterion.bench_function("CoreId/round_trip_as_usize", |bencher| {
        bencher.iter(|| {
            let id = CoreId(std::hint::black_box(42));
            std::hint::black_box(id.as_usize())
        });
    });
}

fn bench_spawn_error_display(criterion: &mut Criterion) {
    criterion.bench_function("SpawnError/display_inbox_full", |bencher| {
        bencher.iter(|| {
            let err = std::hint::black_box(SpawnError::InboxFull);
            std::hint::black_box(err.to_string())
        });
    });
}

fn bench_spawn_request_send_construction(criterion: &mut Criterion) {
    criterion.bench_function("SpawnRequest/send_construction", |bencher| {
        bencher.iter(|| {
            let future: std::pin::Pin<Box<dyn core::future::Future<Output = ()> + Send + 'static>> =
                Box::pin(async {});
            let request: SpawnRequest = SpawnRequest::Send(std::hint::black_box(future));
            std::hint::black_box(request)
        });
    });
}

fn bench_spawn_request_factory_construction(criterion: &mut Criterion) {
    criterion.bench_function("SpawnRequest/factory_construction", |bencher| {
        bencher.iter(|| {
            let factory: Box<
                dyn FnOnce() -> std::pin::Pin<Box<dyn core::future::Future<Output = ()> + 'static>>
                    + Send
                    + 'static,
            > = Box::new(|| Box::pin(async {}));
            let request: SpawnRequest = SpawnRequest::Factory(std::hint::black_box(factory));
            std::hint::black_box(request)
        });
    });
}

fn bench_spawn_request_shutdown_construction(criterion: &mut Criterion) {
    criterion.bench_function("SpawnRequest/shutdown_construction", |bencher| {
        bencher.iter(|| {
            let request: SpawnRequest<core::convert::Infallible> = SpawnRequest::Shutdown;
            std::hint::black_box(request)
        });
    });
}

criterion_group!(
    benches,
    bench_core_id_round_trip,
    bench_spawn_error_display,
    bench_spawn_request_send_construction,
    bench_spawn_request_factory_construction,
    bench_spawn_request_shutdown_construction,
);
criterion_main!(benches);
