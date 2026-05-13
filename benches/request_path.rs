#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

use std::sync::Arc;
use std::time::Duration;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use proxima::{LoadContext, Request, Spec};
use proxima_primitives::pipe::SendPipe;
use serde_json::json;
use tokio::runtime::Runtime;

fn build_runtime() -> Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .expect("tokio runtime")
}

fn synth_only_request(criterion: &mut Criterion) {
    let runtime = build_runtime();
    let context = runtime
        .block_on(async { LoadContext::with_default_registry() })
        .expect("load context");
    let handle = runtime
        .block_on(async {
            proxima::load(
                Spec::Inline(json!({"synth": {"status": 200, "body": "hello"}})),
                &context,
            )
            .await
        })
        .expect("load synth");
    let mut group = criterion.benchmark_group("request_path");
    group.throughput(Throughput::Elements(1));
    group.measurement_time(Duration::from_secs(3));
    group.bench_function("synth_only", |bencher| {
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
    group.finish();
}

fn cached_fallthrough_request(criterion: &mut Criterion) {
    let runtime = build_runtime();
    let context = runtime
        .block_on(async { LoadContext::with_default_registry() })
        .expect("load context");
    let spec = json!({
        "name": "cached",
        "upstreams": [
            {"kv": "cache", "max_entries": 1024, "name": "cache"},
            {"synth": {"status": 200, "body": "from-origin"}, "name": "origin"},
        ],
        "select": {"algorithm": "fallthrough", "miss_on": ["no_data"]},
        "write_back": [["origin", "cache"]],
    });
    let handle = runtime
        .block_on(async { proxima::load(Spec::Inline(spec.clone()), &context).await })
        .expect("load cached");
    // prime the cache so the bench measures the steady-state hit path.
    runtime.block_on(async {
        let request = Request::builder()
            .method("GET")
            .path("/v1/items")
            .build()
            .expect("prime");
        let response = SendPipe::call(&handle, request).await.expect("prime call");
        let _ = response.collect_body().await;
    });
    let mut group = criterion.benchmark_group("request_path");
    group.throughput(Throughput::Elements(1));
    group.measurement_time(Duration::from_secs(3));
    group.bench_function("cache_hit", |bencher| {
        bencher.to_async(&runtime).iter(|| {
            let handle = handle.clone();
            async move {
                let request = Request::builder()
                    .method("GET")
                    .path("/v1/items")
                    .build()
                    .expect("build request");
                let response = SendPipe::call(&handle, request).await.expect("call");
                std::hint::black_box(response.status);
            }
        });
    });
    group.finish();
    let _ = Arc::strong_count(&handle);
}

criterion_group!(benches, synth_only_request, cached_fallthrough_request);
criterion_main!(benches);
