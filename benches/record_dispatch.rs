#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

//! `RecordUpstream` per-call dispatch overhead — the cost a chain pays to be
//! recorded, isolated from the sink's disk write.
//!
//! The recording sink drain is an async tee (`src/upstreams/record.rs`): the
//! request path only does a non-blocking `sender.send()` into an unbounded
//! channel, so the disk write is off the critical path by construction. What
//! is on the path differs by arm:
//!   - unarmed (`record_null_sink`): a fresh `tokio::spawn` + unbounded
//!     channel is paid on every call (legacy per-call drainer).
//!   - armed (`record_armed_spawn_once`): the drainer is spawned once, ever;
//!     `call()` only clones the memoized sender.
//!
//! A null sink (no-op `append`) isolates the dispatch CPU cost from disk I/O.
//!
//! Arms, per response-body size:
//!  - `baseline_synth`          — `synth.call()` + drain (the inner pipe alone)
//!  - `record_armed_spawn_once` — `record(synth).call()` + drain, spigot armed
//!  - `record_null_sink`        — `record(synth).call()` + drain, spigot unarmed
//!
//! `record_null_sink − record_armed_spawn_once` is the per-call spawn+channel
//! tax the spawn-once change removes. The size sweep (64 B vs 16 KiB)
//! separates the fixed per-call cost (spawn/alloc/ULID/clones — the part that
//! dominates for a tiny-message protocol like RESP) from the per-byte
//! body-clone cost.

use std::sync::Arc;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use proxima::pipe::{PipeHandle, into_handle};
use proxima::recording::deferred_runtime;
use proxima::runtime::{Runtime as ProximaRuntime, TokioPerCoreRuntime};
use proxima::{
    DynRecordingSink, LoadContext, RecordUpstream, RecordingAppendFuture, RecordingEvent,
    RecordingSink, Request, SendPipe, Spec, load,
};
use serde_json::json;
use tokio::runtime::Runtime;

/// A sink that discards every event — isolates the dispatch CPU cost from disk
/// I/O. The drainer still runs (so the per-call spawn/channel cost is paid in
/// full on the unarmed arm); only the write is a no-op.
struct NullSink;

impl RecordingSink for NullSink {
    fn append<'lifetime>(
        &'lifetime self,
        _event: RecordingEvent,
    ) -> RecordingAppendFuture<'lifetime> {
        Box::pin(async { Ok(()) })
    }

    fn flush<'lifetime>(&'lifetime self) -> RecordingAppendFuture<'lifetime> {
        Box::pin(async { Ok(()) })
    }
}

fn build_runtime() -> Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .expect("tokio runtime")
}

fn synth_handle(runtime: &Runtime, context: &LoadContext, body_len: usize) -> PipeHandle {
    let body = "x".repeat(body_len);
    runtime
        .block_on(async {
            load(
                Spec::Inline(json!({ "synth": { "status": 200, "body": body } })),
                context,
            )
            .await
        })
        .expect("load synth")
}

/// Unarmed: the drainer spawns fresh (`tokio::spawn` + unbounded channel)
/// on every `call()` — the legacy per-call path this change removes.
fn record_handle(inner: PipeHandle) -> PipeHandle {
    let sink: DynRecordingSink = Arc::new(NullSink);
    into_handle(RecordUpstream::new("record-bench", inner, sink, "synth"))
}

/// Armed with a runtime: the drainer spawns ONCE on that runtime, so `call()`
/// pays no per-call `tokio::spawn` — the fixed path.
fn armed_record_handle(inner: PipeHandle, runtime: Arc<dyn ProximaRuntime>) -> PipeHandle {
    let sink: DynRecordingSink = Arc::new(NullSink);
    let spigot = deferred_runtime();
    let _ = spigot.set(runtime);
    into_handle(
        RecordUpstream::new("record-bench-armed", inner, sink, "synth").with_runtime(spigot),
    )
}

async fn call_and_drain(handle: PipeHandle) {
    let request = Request::builder()
        .method("GET")
        .path("/")
        .build()
        .expect("request");
    let response = SendPipe::call(&handle, request).await.expect("call");
    let body = response.collect_body().await.expect("collect");
    std::hint::black_box(body);
}

fn record_dispatch(criterion: &mut Criterion) {
    let runtime = build_runtime();
    let context = runtime
        .block_on(async { LoadContext::with_default_registry() })
        .expect("ctx");

    let armed_runtime: Arc<dyn ProximaRuntime> =
        Arc::new(TokioPerCoreRuntime::new(1).expect("per-core runtime"));

    let mut group = criterion.benchmark_group("record_dispatch");
    group.measurement_time(Duration::from_secs(5));
    group.throughput(Throughput::Elements(1));

    for &size in &[64_usize, 16 * 1024] {
        let synth = synth_handle(&runtime, &context, size);
        let recorded = record_handle(synth.clone());
        let recorded_armed = armed_record_handle(synth.clone(), armed_runtime.clone());

        group.bench_with_input(
            BenchmarkId::new("baseline_synth", size),
            &synth,
            |bencher, handle| {
                bencher
                    .to_async(&runtime)
                    .iter(|| call_and_drain(handle.clone()));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("record_armed_spawn_once", size),
            &recorded_armed,
            |bencher, handle| {
                bencher
                    .to_async(&runtime)
                    .iter(|| call_and_drain(handle.clone()));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("record_null_sink", size),
            &recorded,
            |bencher, handle| {
                bencher
                    .to_async(&runtime)
                    .iter(|| call_and_drain(handle.clone()));
            },
        );
    }
    group.finish();
}

criterion_group!(benches, record_dispatch);
criterion_main!(benches);
