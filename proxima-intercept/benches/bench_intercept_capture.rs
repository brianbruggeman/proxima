#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

//! C9 — intercept-capture bench. Apples-to-apples vs the incumbent
//! `/tmp` `std::fs::write` text dumps that capture replaced.
//!
//! incumbent: per-interaction `std::fs::write(path, body)` — one fopen + one
//! write per direction. design point: raw blob persistence, grep-replayable
//! by hand, no framing, no parse, no redact.
//!
//! groups (and design-favors per workload):
//!   - capture_typical_post_turn   design-favors: incumbent
//!       (one ~200B request body + one ~1.8KB response body, end-to-end
//!        lifecycle. incumbent's home turf — single blob write per
//!        direction. we expect to lose on raw throughput because we
//!        parse headers, redact secrets, frame zstd, and write 6 events
//!        per interaction. acceptance: stay within an order of magnitude.)
//!   - capture_streaming_chunks    design-favors: neither
//!       (32 small response chunks, mimicking an SSE/WS turn at modest
//!        rate. both sides can append per chunk; primitive op shape.)
//!   - parse_request_header_redact design-favors: proxima
//!       (parse + redact is a feature gap for the incumbent — they can't
//!        do redaction. proxima-favored arm exists as the noise floor;
//!        result is what we get for free vs the incumbent's nothing.)

use std::path::Path;
use std::time::Duration;

use bytes::Bytes;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use proxima_intercept::capture::Capture;

// process-wide armed spigot: one 1-core prime runtime backs every capture's
// off-core blocking I/O (created once; cloning the Arc<OnceLock> is a bump).
fn armed_spigot() -> proxima_recording::pipe::DeferredRuntime {
    static SPIGOT: std::sync::OnceLock<proxima_recording::pipe::DeferredRuntime> =
        std::sync::OnceLock::new();
    SPIGOT
        .get_or_init(|| {
            let spigot = proxima_recording::pipe::deferred_runtime();
            spigot
                .set(
                    std::sync::Arc::new(proxima::runtime::PrimeRuntime::new(1).expect("prime"))
                        as std::sync::Arc<dyn proxima::runtime::Runtime>,
                )
                .ok();
            spigot
        })
        .clone()
}

const REQUEST_WIRE: &[u8] = b"POST /responses HTTP/1.1\r\n\
                              Host: api.individual.example.com\r\n\
                              Content-Type: application/json\r\n\
                              Authorization: Bearer token-value-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\r\n\
                              X-Integration-Id: example-cli\r\n\
                              X-Api-Version: 2026-06-01\r\n\
                              X-Client-Session-Id: bench-session-id-value\r\n\
                              Content-Length: 195\r\n\
                              \r\n";

const REQUEST_BODY: &[u8] = b"{\"model\":\"model-nano\",\"instructions\":\"answer terse\",\"input\":[{\"role\":\"user\",\"content\":[{\"type\":\"input_text\",\"text\":\"What is 7 * 6?\"}],\"type\":\"message\"}],\"tools\":[],\"store\":false}";

const RESPONSE_HEAD: &[u8] = b"HTTP/1.1 200 OK\r\n\
                               Content-Type: application/json\r\n\
                               Content-Length: 1771\r\n\
                               X-Request-Id: 00000-7d41b6d7-5a30-4d80-b386-e689c335331b\r\n\
                               Date: Thu, 28 May 2026 02:24:55 GMT\r\n\
                               \r\n";

fn make_response_body(size: usize) -> Vec<u8> {
    let mut body = Vec::with_capacity(size);
    let chunk = br#"{"created_at":1779980900,"id":"abc","model":"model-nano","output":[{"content":[{"text":"7 * 6 = 42"}]}]}"#;
    while body.len() < size {
        body.extend_from_slice(chunk);
    }
    body.truncate(size);
    body
}

fn write_blob(path: &Path, blob: &[u8]) {
    std::fs::write(path, blob).expect("write blob");
}

fn bench_typical_post_turn(criterion: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().expect("runtime");
    let response_body = make_response_body(1771);
    let total_bytes =
        (REQUEST_WIRE.len() + REQUEST_BODY.len() + RESPONSE_HEAD.len() + response_body.len())
            as u64;

    let mut group = criterion.benchmark_group("capture_typical_post_turn");
    group.measurement_time(Duration::from_secs(3));
    group.throughput(Throughput::Bytes(total_bytes));

    let response_body_arc: std::sync::Arc<Vec<u8>> = std::sync::Arc::new(response_body);

    let response_body_proxima = std::sync::Arc::clone(&response_body_arc);
    group.bench_function("proxima", move |bencher| {
        let response_body_iter = std::sync::Arc::clone(&response_body_proxima);
        bencher.to_async(&runtime).iter_batched(
            || {
                (
                    tempfile::tempdir().expect("tempdir"),
                    std::sync::Arc::clone(&response_body_iter),
                )
            },
            |(temp_dir, response_body)| async move {
                let path = temp_dir.path().join("interaction.bin");
                let capture = Capture::open(&path, armed_spigot()).expect("open");
                let started = std::time::Instant::now();
                let recorder = capture
                    .begin("api.individual.example.com", REQUEST_WIRE, started)
                    .await
                    .expect("begin");
                recorder.push_request(Bytes::from_static(REQUEST_BODY));
                recorder.push_response(Bytes::copy_from_slice(&response_body));
                recorder.finish(RESPONSE_HEAD).await.expect("finish");
                std::hint::black_box(temp_dir);
            },
            criterion::BatchSize::SmallInput,
        );
    });

    let response_body_incumbent = std::sync::Arc::clone(&response_body_arc);
    group.bench_function("incumbent_fs_write", move |bencher| {
        let response_body_iter = std::sync::Arc::clone(&response_body_incumbent);
        bencher.iter_batched(
            || {
                (
                    tempfile::tempdir().expect("tempdir"),
                    std::sync::Arc::clone(&response_body_iter),
                )
            },
            |(temp_dir, response_body)| {
                let req_path = temp_dir.path().join("req.json");
                let resp_path = temp_dir.path().join("resp.txt");
                write_blob(&req_path, REQUEST_BODY);
                let mut resp_blob = RESPONSE_HEAD.to_vec();
                resp_blob.extend_from_slice(&response_body);
                write_blob(&resp_path, &resp_blob);
                std::hint::black_box(temp_dir);
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

fn bench_streaming_chunks(criterion: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().expect("runtime");
    let chunk: Bytes = Bytes::from_static(b"data: {\"delta\":\"hello there from the stream\"}\n\n");
    let chunk_count: usize = 32;
    let total_bytes = (chunk.len() * chunk_count) as u64;

    let mut group = criterion.benchmark_group("capture_streaming_chunks");
    group.measurement_time(Duration::from_secs(3));
    group.throughput(Throughput::Bytes(total_bytes));

    group.bench_function("proxima_chunk_push", |bencher| {
        bencher.to_async(&runtime).iter_batched(
            || tempfile::tempdir().expect("tempdir"),
            |temp_dir| {
                let chunk_loop = chunk.clone();
                async move {
                    let path = temp_dir.path().join("interaction.bin");
                    let capture = Capture::open(&path, armed_spigot()).expect("open");
                    let started = std::time::Instant::now();
                    let recorder = capture
                        .begin("api.example.com", REQUEST_WIRE, started)
                        .await
                        .expect("begin");
                    for _ in 0..chunk_count {
                        recorder.push_response(chunk_loop.clone());
                    }
                    recorder.finish(RESPONSE_HEAD).await.expect("finish");
                    std::hint::black_box(temp_dir);
                }
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.bench_function("incumbent_append_per_chunk", |bencher| {
        bencher.iter_batched(
            || tempfile::tempdir().expect("tempdir"),
            |temp_dir| {
                use std::fs::OpenOptions;
                use std::io::Write;
                let path = temp_dir.path().join("stream.txt");
                let mut file = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&path)
                    .expect("open");
                for _ in 0..chunk_count {
                    file.write_all(&chunk).expect("write");
                }
                file.flush().expect("flush");
                std::hint::black_box(temp_dir);
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

// production shape: the sink is opened ONCE and thousands of interactions append
// to it. the per-iteration `Capture::open` in `capture_streaming_chunks` charges
// proxima two file-creates every turn that a real deployment pays only once — so
// that arm understates us. here the sink (and the incumbent's file) are opened in
// setup and reused, isolating the steady-state per-interaction cost.
fn bench_streaming_open_once(criterion: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().expect("runtime");
    let chunk: Bytes = Bytes::from_static(b"data: {\"delta\":\"hello there from the stream\"}\n\n");
    let chunk_count: usize = 32;
    let total_bytes = (chunk.len() * chunk_count) as u64;

    let mut group = criterion.benchmark_group("capture_streaming_open_once");
    group.measurement_time(Duration::from_secs(3));
    group.throughput(Throughput::Bytes(total_bytes));

    let temp_dir = tempfile::tempdir().expect("tempdir");

    // one arm per ChunkGranularity: PerChunk (faithful, default), Coalesced
    // (one event per direction), Discard (push is a no-op — no store/restore).
    use proxima_intercept::capture::{Capture, ChunkGranularity};
    for (label, granularity) in [
        ("proxima_per_chunk", ChunkGranularity::PerChunk),
        ("proxima_discard", ChunkGranularity::Discard),
    ] {
        let path = temp_dir.path().join(format!("reused_{label}.bin"));
        let capture = Capture::open(&path, armed_spigot())
            .expect("open capture")
            .with_chunk_granularity(granularity);

        group.bench_function(label, |bencher| {
            bencher.to_async(&runtime).iter(|| {
                let capture = capture.clone();
                let chunk = chunk.clone();
                async move {
                    let started = std::time::Instant::now();
                    let recorder = capture
                        .begin("api.individual.example.com", REQUEST_WIRE, started)
                        .await
                        .expect("begin");
                    for _ in 0..chunk_count {
                        recorder.push_response(chunk.clone());
                    }
                    recorder.finish(RESPONSE_HEAD).await.expect("finish");
                }
            });
        });
    }

    let incumbent_path = temp_dir.path().join("reused_incumbent.txt");
    let incumbent_file = std::sync::Mutex::new(
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&incumbent_path)
            .expect("open incumbent"),
    );
    group.bench_function("incumbent_reused_file", |bencher| {
        bencher.iter(|| {
            use std::io::Write;
            let mut file = incumbent_file.lock().expect("lock");
            for _ in 0..chunk_count {
                file.write_all(&chunk).expect("write");
            }
            file.flush().expect("flush");
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_typical_post_turn,
    bench_streaming_chunks,
    bench_streaming_open_once
);
criterion_main!(benches);
