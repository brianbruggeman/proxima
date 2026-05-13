#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

// Tier-1 substrate dispatch bench. No I/O, no kernel, no socket.
// Measures Pipe::call overhead through composed middleware chains —
// the ceiling on what proxima can do before networking enters.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use proxima::{
    LoadContext, PipeFactory, PipeHandle, ProximaError, Request, Response, Spec, into_handle,
};
use proxima_primitives::pipe::SendPipe;
use serde_json::{Value, json};
use tokio::runtime::Runtime;

fn build_runtime() -> Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .expect("tokio runtime")
}

struct NoopFactory;

impl PipeFactory for NoopFactory {
    fn name(&self) -> &str {
        "noop"
    }

    fn build(
        &self,
        _spec: &Value,
        inner: Option<PipeHandle>,
    ) -> Pin<Box<dyn Future<Output = Result<PipeHandle, ProximaError>> + Send + '_>> {
        Box::pin(async move {
            let inner =
                inner.ok_or_else(|| ProximaError::Config("noop requires an inner pipe".into()))?;
            Ok(into_handle(Noop { inner }))
        })
    }
}

struct Noop {
    inner: PipeHandle,
}

impl SendPipe for Noop {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let inner = self.inner.clone();
        async move { SendPipe::call(&inner, request).await }
    }
}


fn context_with_noop(runtime: &Runtime) -> LoadContext {
    let context = runtime
        .block_on(async { LoadContext::with_default_registry() })
        .expect("load context");
    context
        .registry
        .register(Arc::new(NoopFactory))
        .expect("register noop");
    context
}

fn build_chain(runtime: &Runtime, context: &LoadContext, depth: usize) -> PipeHandle {
    let middleware: Vec<Value> = (0..depth).map(|_| json!({"type": "noop"})).collect();
    let spec = json!({
        "name": "bench",
        "synth": {"status": 200, "body": ""},
        "middleware": middleware,
    });
    runtime
        .block_on(async { proxima::load(Spec::Inline(spec), context).await })
        .expect("load chain")
}

fn dispatch_overhead_by_depth(criterion: &mut Criterion) {
    let runtime = build_runtime();
    let context = context_with_noop(&runtime);
    let mut group = criterion.benchmark_group("substrate_dispatch");
    group.throughput(Throughput::Elements(1));
    group.measurement_time(Duration::from_secs(3));
    for depth in [0usize, 1, 3, 5, 10] {
        let handle = build_chain(&runtime, &context, depth);
        let label = format!("noop_mw_x{depth}");
        group.bench_function(&label, |bencher| {
            bencher.to_async(&runtime).iter(|| {
                let handle = handle.clone();
                async move {
                    let request = Request::builder()
                        .method("GET")
                        .path("/")
                        .build()
                        .expect("build request");
                    let response = SendPipe::call(&handle, request).await.expect("call");
                    std::hint::black_box(response.status);
                }
            });
        });
    }
    group.finish();
}

fn dispatch_overhead_concurrent(criterion: &mut Criterion) {
    let runtime = build_runtime();
    let context = context_with_noop(&runtime);
    let handle = build_chain(&runtime, &context, 3);
    let mut group = criterion.benchmark_group("substrate_dispatch");
    group.measurement_time(Duration::from_secs(3));
    for concurrency in [1usize, 4, 16, 64] {
        let label = format!("noop_mw_x3_concurrent_{concurrency}");
        group.throughput(Throughput::Elements(concurrency as u64));
        group.bench_function(&label, |bencher| {
            bencher.to_async(&runtime).iter(|| {
                let handle = handle.clone();
                async move {
                    let mut joins = Vec::with_capacity(concurrency);
                    for _ in 0..concurrency {
                        let handle = handle.clone();
                        joins.push(tokio::spawn(async move {
                            let request = Request::builder()
                                .method("GET")
                                .path("/")
                                .build()
                                .expect("build request");
                            SendPipe::call(&handle, request).await.expect("call")
                        }));
                    }
                    for join in joins {
                        let response = join.await.expect("join");
                        std::hint::black_box(response.status);
                    }
                }
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    dispatch_overhead_by_depth,
    dispatch_overhead_concurrent,
);
criterion_main!(benches);
