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

//! H3 tail latency across multiple parallel connections. Mirrors
//! `h2_tail_multi_conn`. Each iteration fans out one request per
//! connection in parallel. Tests whether the proxima h3 listener
//! holds latency steady as connection count rises (each connection
//! has its own driver task; per-connection work shouldn't head-of-
//! line block other connections).

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


fn multi_conn(criterion: &mut Criterion) {
    let runtime = h3_setup::build_runtime();
    let dispatch: PipeHandle = into_handle(ConstantOk);
    let addr = h3_setup::start_h3_server(&runtime, dispatch);
    let uri = format!("https://localhost:{}/", addr.port());

    for &connection_count in &[1usize, 4, 16] {
        let warm_clients = runtime.block_on(async {
            let client_endpoint = h3_setup::make_client_endpoint();
            let mut clients = Vec::with_capacity(connection_count);
            for _ in 0..connection_count {
                clients.push(h3_setup::warm_h3_client(&client_endpoint, addr).await);
            }
            // keep the endpoint alive for the entire bench group
            std::mem::forget(client_endpoint);
            clients
        });

        let mut group =
            criterion.benchmark_group(format!("h3_tail_multi_conn_n{connection_count}"));
        group.measurement_time(SAMPLE_WINDOW);
        group.bench_function("one_request_per_connection_p50_ns", |bencher| {
            let warm_clients = warm_clients.clone();
            let uri = uri.clone();
            bencher.to_async(&runtime).iter_custom(|iters| {
                let warm_clients = warm_clients.clone();
                let uri = uri.clone();
                async move {
                    let mut hist = fresh_histogram();
                    for _ in 0..iters {
                        let futures = warm_clients.iter().cloned().map(|mut send_request| {
                            let uri = uri.clone();
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

criterion_group!(benches, multi_conn);
criterion_main!(benches);
