#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]
#![cfg(all(feature = "http3", feature = "runtime-tokio"))]

//! H3 streaming-response throughput. Mirrors `h2_streaming_responses`
//! but proxima-only — hyper/pingora don't ship native h3 servers, so
//! the cross-stack comparison column has no comparable peers.
//!
//! Workload: server returns a response built by emitting 32 chunks
//! of 2 KiB each (64 KiB total) via `Body::from_stream`. Client opens
//! one warm h3 connection, issues GET, drains every DATA frame, then
//! issues the next GET. Sequential request-response cycles for the
//! measurement window.

use std::future::Future;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use bytes::Bytes;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use hdrhistogram::Histogram;
use proxima::ResponseStream;
use proxima::error::ProximaError;
use proxima::pipe::{PipeHandle, into_handle};
use proxima::request::{Request, Response};
use proxima_primitives::pipe::SendPipe;

#[path = "../common/h3_setup.rs"]
mod h3_setup;

const CHUNK_COUNT: usize = 32;
const CHUNK_SIZE: usize = 2 * 1024;
const SAMPLE_WINDOW: Duration = Duration::from_secs(3);

fn fresh_histogram() -> Histogram<u64> {
    Histogram::<u64>::new_with_bounds(1, 60_000_000_000, 3).expect("hdr bounds")
}

static CHUNK_POOL: OnceLock<Vec<Bytes>> = OnceLock::new();

fn chunk_bytes(index: usize) -> Bytes {
    let pool = CHUNK_POOL.get_or_init(|| {
        (0..CHUNK_COUNT)
            .map(|chunk_index| Bytes::from(vec![b'a' + (chunk_index as u8 % 26); CHUNK_SIZE]))
            .collect()
    });
    pool[index].clone()
}

struct StreamingPipe;

impl SendPipe for StreamingPipe {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        _request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        async move {
            let stream = futures::stream::iter(
                (0..CHUNK_COUNT).map(|index| Ok::<Bytes, ProximaError>(chunk_bytes(index))),
            );
            Ok(Response::streamed(ResponseStream::new(stream)))
        }
    }
}


fn proxima_streaming(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("h3_streaming_responses");
    group.measurement_time(SAMPLE_WINDOW);
    group.throughput(Throughput::Bytes((CHUNK_COUNT * CHUNK_SIZE) as u64));

    let runtime = h3_setup::build_runtime();
    let dispatch: PipeHandle = into_handle(StreamingPipe);
    let addr = h3_setup::start_h3_server(&runtime, dispatch);
    let (_client_endpoint, send_request) = runtime.block_on(async {
        let endpoint = h3_setup::make_client_endpoint();
        let send = h3_setup::warm_h3_client(&endpoint, addr).await;
        (endpoint, send)
    });
    let uri = format!("https://localhost:{}/stream", addr.port());

    group.bench_function("32x2KiB_chunks_warm_connection", |bencher| {
        let send_request = send_request.clone();
        let uri = uri.clone();
        bencher.to_async(&runtime).iter_custom(|iters| {
            let uri = uri.clone();
            let mut send_request = send_request.clone();
            async move {
                let mut hist = fresh_histogram();
                for _ in 0..iters {
                    let started = Instant::now();
                    let request = http::Request::builder()
                        .method("GET")
                        .uri(&uri)
                        .body(())
                        .expect("request");
                    let mut stream = send_request.send_request(request).await.expect("send");
                    stream.finish().await.expect("finish");
                    let response = stream.recv_response().await.expect("response");
                    std::hint::black_box(response.status());
                    let mut total = 0usize;
                    while let Some(mut chunk) = stream.recv_data().await.expect("recv_data") {
                        total += bytes::Buf::remaining(&chunk);
                        while bytes::Buf::has_remaining(&chunk) {
                            let slice = bytes::Buf::chunk(&chunk);
                            let advance = slice.len();
                            std::hint::black_box(slice);
                            bytes::Buf::advance(&mut chunk, advance);
                        }
                    }
                    debug_assert_eq!(total, CHUNK_COUNT * CHUNK_SIZE);
                    let elapsed = started.elapsed().as_nanos() as u64;
                    let _ = hist.record(elapsed);
                }
                Duration::from_nanos(hist.value_at_quantile(0.5))
            }
        });
    });
    group.finish();
}

criterion_group!(benches, proxima_streaming);
criterion_main!(benches);
