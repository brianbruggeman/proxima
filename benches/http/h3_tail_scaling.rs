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

//! H3 tail-latency sweep across concurrency levels. Mirrors
//! `h2_tail_scaling`'s shape but proxima-only — hyper/pingora don't
//! ship h3 servers, so the cross-stack column is dropped.
//!
//! Workload: one warm h3 connection. For each iteration, `N` requests
//! issued concurrently via `futures::try_join_all`; per-request
//! latency recorded in an HDR histogram; criterion sees the p50.

use std::future::Future;
use std::time::{Duration, Instant};

use bytes::Bytes;
use criterion::{Criterion, criterion_group, criterion_main};
use hdrhistogram::Histogram;
use proxima::error::ProximaError;
use proxima::pipe::{PipeHandle, into_handle};
use proxima::request::{Request, Response};
use proxima_primitives::pipe::SendPipe;

#[path = "../common/h3_setup.rs"]
mod h3_setup;

const SAMPLE_WINDOW: Duration = Duration::from_secs(3);
const RESPONSE_BODY: &[u8] = b"ok";

fn fresh_histogram() -> Histogram<u64> {
    Histogram::<u64>::new_with_bounds(1, 60_000_000_000, 3).expect("hdr bounds")
}

struct ConstantOk;

impl SendPipe for ConstantOk {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        _request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        async move { Ok(Response::ok(Bytes::from_static(RESPONSE_BODY))) }
    }
}


fn sweep(criterion: &mut Criterion) {
    let runtime = h3_setup::build_runtime();
    let dispatch: PipeHandle = into_handle(ConstantOk);
    let addr = h3_setup::start_h3_server(&runtime, dispatch);
    let (_client_endpoint, send_request) = runtime.block_on(async {
        let endpoint = h3_setup::make_client_endpoint();
        let send = h3_setup::warm_h3_client(&endpoint, addr).await;
        (endpoint, send)
    });
    let uri = format!("https://localhost:{}/", addr.port());

    for &concurrency in &[1usize, 10, 100] {
        let mut group = criterion.benchmark_group(format!("h3_tail_scaling_c{concurrency}"));
        group.measurement_time(SAMPLE_WINDOW);
        group.bench_function("warm_connection_p50_ns", |bencher| {
            let send_request = send_request.clone();
            let uri = uri.clone();
            bencher.to_async(&runtime).iter_custom(|iters| {
                let uri = uri.clone();
                let send_request = send_request.clone();
                async move {
                    let mut hist = fresh_histogram();
                    for _ in 0..iters {
                        let futures = (0..concurrency).map(|_| {
                            let uri = uri.clone();
                            let mut send_request = send_request.clone();
                            async move {
                                let started = Instant::now();
                                let request = http::Request::builder()
                                    .method("GET")
                                    .uri(&uri)
                                    .body(())
                                    .expect("request");
                                let mut stream =
                                    send_request.send_request(request).await.expect("send");
                                stream.finish().await.expect("finish");
                                let response = stream.recv_response().await.expect("response");
                                std::hint::black_box(response.status());
                                while let Some(chunk) = stream.recv_data().await.expect("recv_data")
                                {
                                    std::hint::black_box(chunk);
                                }
                                started.elapsed().as_nanos() as u64
                            }
                        });
                        let elapsed = futures::future::join_all(futures).await;
                        for measurement in elapsed {
                            let _ = hist.record(measurement);
                        }
                    }
                    Duration::from_nanos(hist.value_at_quantile(0.5))
                }
            });
        });
        group.finish();
    }
}

criterion_group!(benches, sweep);
criterion_main!(benches);
