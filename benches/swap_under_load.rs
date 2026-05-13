#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

//! `SwappablePipe` swap latency and dispatch overhead under
//! concurrent swaps. Three measurements:
//!
//! 1. `current()` baseline — pure read cost via `ArcSwap::load`,
//!    the hot path the listener takes on every dispatch.
//! 2. `swap()` baseline — write cost when no readers contend.
//! 3. `dispatch_during_swap_storm` — chain dispatch through a
//!    SwappablePipe while a writer task swaps continuously on a
//!    separate thread, to measure read-side regression under
//!    contention.

use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use bytes::Bytes;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use proxima::{PipeHandle, ProximaError, Request, Response, SwappablePipe, into_handle};
use proxima_primitives::pipe::SendPipe;
use tokio::runtime::Runtime;

struct StaticPipe {
    label: &'static str,
}

impl SendPipe for StaticPipe {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        _request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let label = self.label;
        async move { Ok(Response::ok(bytes::Bytes::from_static(label.as_bytes()))) }
    }
}


fn build_runtime() -> Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("tokio runtime")
}

fn current_load_uncontended(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("swap_under_load_current");
    group.measurement_time(Duration::from_secs(3));
    group.throughput(Throughput::Elements(1));
    let swappable = SwappablePipe::new(into_handle(StaticPipe { label: "a" }));
    group.bench_function("uncontended_load", |bencher| {
        bencher.iter(|| {
            let handle = swappable.current();
            std::hint::black_box(handle);
        });
    });
    group.finish();
}

fn swap_uncontended(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("swap_under_load_swap");
    group.measurement_time(Duration::from_secs(3));
    group.throughput(Throughput::Elements(1));
    let swappable = SwappablePipe::new(into_handle(StaticPipe { label: "a" }));
    let new_handle: PipeHandle = into_handle(StaticPipe { label: "b" });
    group.bench_function("uncontended_swap", |bencher| {
        bencher.iter(|| {
            swappable.swap(new_handle.clone());
        });
    });
    group.finish();
}

fn dispatch_during_swap_storm(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("swap_under_load_dispatch_during_storm");
    group.measurement_time(Duration::from_secs(3));
    group.throughput(Throughput::Elements(1));
    let runtime = build_runtime();
    let swappable = Arc::new(SwappablePipe::new(into_handle(StaticPipe { label: "a" })));
    let stop = Arc::new(AtomicBool::new(false));
    let swap_other = into_handle(StaticPipe { label: "b" });
    // Writer task swaps continuously on another worker thread.
    let writer_swappable = swappable.clone();
    let writer_stop = stop.clone();
    let writer = std::thread::spawn(move || {
        let mut toggle = false;
        while !writer_stop.load(Ordering::Relaxed) {
            let next = if toggle {
                into_handle(StaticPipe { label: "a" })
            } else {
                swap_other.clone()
            };
            writer_swappable.swap(next);
            toggle = !toggle;
        }
    });
    group.bench_function("read_then_dispatch", |bencher| {
        bencher.to_async(&runtime).iter(|| {
            let swappable = swappable.clone();
            async move {
                let handle = swappable.current();
                let request = Request::builder()
                    .method("GET")
                    .path("/")
                    .build()
                    .expect("request");
                let _ = SendPipe::call(&handle, request).await.expect("call");
            }
        });
    });
    stop.store(true, Ordering::Relaxed);
    writer.join().ok();
    group.finish();
}

criterion_group!(
    benches,
    current_load_uncontended,
    swap_uncontended,
    dispatch_during_swap_storm
);
criterion_main!(benches);
