#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

// hot-path microbenches: rate_limit key extract, daemon status read,
// record streaming, write_back into kv, codec decode.

use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use proxima::{
    ControlPlane, DaemonControlPlane, JsonCodec, LoadContext, PipeConfig, Request, Spec,
};
use proxima_primitives::pipe::SendPipe;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::runtime::Runtime;

fn build_runtime() -> Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .expect("tokio runtime")
}

fn rate_limit_through_synth(criterion: &mut Criterion) {
    let runtime = build_runtime();
    let context = runtime
        .block_on(async { LoadContext::with_default_registry() })
        .expect("load context");
    let spec = json!({
        "synth": { "status": 200, "body": "ok" },
        "rate_limit": {
            "capacity": 1_000_000,
            "refill_per_sec": 1_000_000,
            "key": "header",
            "header_name": "x-tenant",
        },
    });
    let handle = runtime
        .block_on(async { proxima::load(Spec::Inline(spec), &context).await })
        .expect("load rate_limit");
    let mut group = criterion.benchmark_group("rate_limit");
    group.throughput(Throughput::Elements(1));
    group.measurement_time(Duration::from_secs(3));
    group.bench_function("extract_then_bucket", |bencher| {
        bencher.to_async(&runtime).iter(|| {
            let handle = handle.clone();
            async move {
                let request = Request::builder()
                    .method("GET")
                    .path("/v1/items")
                    .header("x-tenant", "tenant-42")
                    .build()
                    .expect("build request");
                let response = SendPipe::call(&handle, request).await.expect("call");
                std::hint::black_box(response.status);
            }
        });
    });
    group.finish();
}

fn daemon_status_with_writer(criterion: &mut Criterion) {
    let runtime = build_runtime();
    let app = runtime
        .block_on(async {
            proxima::AppBuilder::new()
                .with_defaults()
                .expect("defaults")
                .build()
        })
        .expect("app");
    let configs: Vec<PipeConfig> = (0..32)
        .map(|index| PipeConfig {
            name: format!("svc-{index}"),
            spec: json!({"synth": {"status": 200, "body": "x"}}),
            requires: Vec::new(),
        })
        .collect();
    let plane = Arc::new(DaemonControlPlane::new(app, configs));

    // background writer to surface read-path lock contention
    let writer_plane = plane.clone();
    let writer_runtime = runtime.handle().clone();
    let writer = std::thread::spawn(move || {
        let mut counter: u64 = 0;
        loop {
            counter = counter.wrapping_add(1);
            let plane = writer_plane.clone();
            writer_runtime.block_on(async move {
                let _ = plane.upsert_config(PipeConfig {
                    name: format!("svc-writer-{}", counter % 16),
                    spec: json!({"synth": {"status": 200, "body": "x"}}),
                    requires: Vec::new(),
                });
            });
            if counter > 1_000_000 {
                break;
            }
            std::thread::sleep(Duration::from_micros(500));
        }
    });

    let mut group = criterion.benchmark_group("daemon_control_plane");
    group.throughput(Throughput::Elements(1));
    group.measurement_time(Duration::from_secs(3));
    group.bench_function("status_with_concurrent_writer", |bencher| {
        bencher.to_async(&runtime).iter(|| {
            let plane = plane.clone();
            async move {
                let outcome = plane.status("svc-7").await.expect("status");
                std::hint::black_box(outcome.name);
            }
        });
    });
    group.finish();
    drop(writer);
}

fn record_upstream_64kb(criterion: &mut Criterion) {
    let runtime = build_runtime();
    let context = runtime
        .block_on(async { LoadContext::with_default_registry() })
        .expect("load context");
    let dir = tempfile::tempdir().expect("tempdir");
    let sink_path = dir.path().join("recording.jsonl");
    // 64 KiB synth body, chunked
    let body_bytes: String = "x".repeat(64 * 1024);
    let spec = json!({
        "type": "record",
        "sink": {"type": "jsonl", "path": sink_path.to_string_lossy()},
        "label": "perf",
        "pipe_label": "synth-bench",
        "protocol": "http",
        "inner": {
            "synth": {"status": 200, "body": body_bytes},
        },
    });
    let handle = runtime
        .block_on(async { proxima::load(Spec::Inline(spec), &context).await })
        .expect("load record");
    let mut group = criterion.benchmark_group("record_upstream");
    group.throughput(Throughput::Bytes(64 * 1024));
    group.measurement_time(Duration::from_secs(3));
    group.bench_function("end_to_end_64kb", |bencher| {
        bencher.to_async(&runtime).iter(|| {
            let handle = handle.clone();
            async move {
                let request = Request::builder()
                    .method("GET")
                    .path("/")
                    .build()
                    .expect("build request");
                let response = SendPipe::call(&handle, request).await.expect("call");
                let body = response.collect_body().await.expect("collect");
                std::hint::black_box(body.len());
            }
        });
    });
    group.finish();
}

fn write_back_synth_16kb(criterion: &mut Criterion) {
    let runtime = build_runtime();
    let context = runtime
        .block_on(async { LoadContext::with_default_registry() })
        .expect("load context");
    let body_bytes: String = "y".repeat(16 * 1024);
    let spec = json!({
        "name": "cached",
        "upstreams": [
            {"kv": "cache", "max_entries": 4096, "name": "cache"},
            {"synth": {"status": 200, "body": body_bytes}, "name": "origin"},
        ],
        "select": {"algorithm": "fallthrough", "miss_on": ["no_data"]},
        "write_back": [["origin", "cache"]],
    });
    let handle = runtime
        .block_on(async { proxima::load(Spec::Inline(spec), &context).await })
        .expect("load write_back");
    let counter = std::sync::atomic::AtomicU64::new(0);
    let mut group = criterion.benchmark_group("write_back");
    group.throughput(Throughput::Bytes(16 * 1024));
    group.measurement_time(Duration::from_secs(3));
    group.bench_function("synth_to_kv_cache_16kb", |bencher| {
        bencher.to_async(&runtime).iter(|| {
            let handle = handle.clone();
            // unique path per iter so it's a write, not a hit
            let nonce = counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            async move {
                let path = format!("/v1/items/{nonce}");
                let request = Request::builder()
                    .method("GET")
                    .path(path)
                    .build()
                    .expect("build request");
                let response = SendPipe::call(&handle, request).await.expect("call");
                let body = response.collect_body().await.expect("collect");
                std::hint::black_box(body.len());
            }
        });
    });
    group.finish();
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CodecSample {
    id: String,
    timestamp: u64,
    payload: Vec<CodecField>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CodecField {
    key: String,
    value: String,
}

fn codec_decode_4kb(criterion: &mut Criterion) {
    let sample = CodecSample {
        id: "perf-audit".into(),
        timestamp: 1_741_390_800_123,
        payload: (0..40)
            .map(|index| CodecField {
                key: format!("k{index:04}"),
                value: format!("v{index:08}-some-padding"),
            })
            .collect(),
    };
    let bytes = serde_json::to_vec(&sample).expect("encode fixture");
    let codec: JsonCodec<CodecSample, CodecSample> = JsonCodec::new();
    let payload_len = bytes.len();
    let mut group = criterion.benchmark_group("codec");
    group.throughput(Throughput::Bytes(payload_len as u64));
    group.measurement_time(Duration::from_secs(3));
    group.bench_function("json_decode_input", |bencher| {
        bencher.iter(|| {
            let parsed =
                proxima::codec::MessageCodec::decode_input(&codec, &bytes).expect("decode");
            std::hint::black_box(parsed);
        });
    });
    group.finish();
}

// audit A1, A2, A3, A4, A5, B1, B2, B4, D1: per-request allocation
// cost on the listener -> upstream path. measured indirectly via
// synth_only -> cached -> rate_limit benches above; this bench
// specifically captures the body construction cost (concatenating the
// chunks into a single Bytes + new BTreeMaps for headers/query).
fn request_construction(criterion: &mut Criterion) {
    let chunks = vec![Bytes::from_static(b"hello "), Bytes::from_static(b"world")];
    let mut group = criterion.benchmark_group("request_layout");
    group.throughput(Throughput::Elements(1));
    group.measurement_time(Duration::from_secs(3));
    group.bench_function("builder_with_headers_query_body", |bencher| {
        bencher.iter(|| {
            let request = Request::builder()
                .method("GET")
                .path("/v1/items/42")
                .header("authorization", "bearer abcdef0123")
                .header("content-type", "application/json")
                .header("x-trace-id", "trace-deadbeef")
                .query_param("page", "2")
                .query_param("limit", "20")
                .body(
                    chunks
                        .iter()
                        .fold(bytes::BytesMut::new(), |mut acc, chunk| {
                            acc.extend_from_slice(chunk);
                            acc
                        })
                        .freeze(),
                )
                .build()
                .expect("build");
            std::hint::black_box(request);
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    rate_limit_through_synth,
    daemon_status_with_writer,
    record_upstream_64kb,
    write_back_synth_16kb,
    codec_decode_4kb,
    request_construction,
);
criterion_main!(benches);
