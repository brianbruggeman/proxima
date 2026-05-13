#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]
#![cfg(feature = "http3")]

//! H3 body-size sweep on a warm connection. Mirrors `h2_streaming`:
//! 256 B, 64 KiB, and 16 × 4 KiB request bodies are echoed back through
//! the proxima h3 listener and consumed on the client.

use std::future::Future;
use std::time::Duration;

use bytes::Bytes;
use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use proxima::error::ProximaError;
use proxima::pipe::{PipeHandle, into_handle};
use proxima::request::{Request, Response};
use proxima_primitives::pipe::SendPipe;

#[path = "../common/h3_setup.rs"]
mod h3_setup;

struct EchoBody;

impl SendPipe for EchoBody {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        async move {
            let (_request, bytes) = request.body_bytes().await?;
            Ok(Response::ok(bytes))
        }
    }
}


fn echo_body(criterion: &mut Criterion, label: &str, payload: Bytes) {
    let mut group = criterion.benchmark_group(format!("h3_echo_{label}"));
    group.measurement_time(Duration::from_secs(5));
    group.throughput(Throughput::Bytes(payload.len() as u64));

    let runtime = h3_setup::build_runtime();
    let dispatch: PipeHandle = into_handle(EchoBody);
    let addr = h3_setup::start_h3_server(&runtime, dispatch);
    let (_client_endpoint, send_request) = runtime.block_on(async {
        let endpoint = h3_setup::make_client_endpoint();
        let send = h3_setup::warm_h3_client(&endpoint, addr).await;
        (endpoint, send)
    });
    let uri = format!("https://localhost:{}/echo", addr.port());

    group.bench_function("echo_on_warm_connection", |bencher| {
        let send_request = send_request.clone();
        let uri = uri.clone();
        bencher.to_async(&runtime).iter_batched(
            || payload.clone(),
            |body| {
                let uri = uri.clone();
                let mut send_request = send_request.clone();
                async move {
                    let request = http::Request::builder()
                        .method("POST")
                        .uri(uri)
                        .body(())
                        .expect("request");
                    let mut stream = send_request.send_request(request).await.expect("send");
                    stream.send_data(body).await.expect("send body");
                    stream.finish().await.expect("finish");
                    let response = stream.recv_response().await.expect("response");
                    std::hint::black_box(response.status());
                    while let Some(mut chunk) = stream.recv_data().await.expect("recv_data") {
                        while bytes::Buf::has_remaining(&chunk) {
                            let slice = bytes::Buf::chunk(&chunk);
                            let advance = slice.len();
                            std::hint::black_box(slice);
                            bytes::Buf::advance(&mut chunk, advance);
                        }
                    }
                }
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

fn small_body(criterion: &mut Criterion) {
    echo_body(criterion, "256B", Bytes::from(vec![0xAB_u8; 256]));
}

fn medium_body(criterion: &mut Criterion) {
    echo_body(criterion, "64KiB", Bytes::from(vec![0xCD_u8; 64 * 1024]));
}

fn multi_frame_body(criterion: &mut Criterion) {
    echo_body(
        criterion,
        "16x4KiB",
        Bytes::from(vec![0xEF_u8; 16 * 4 * 1024]),
    );
}

criterion_group!(benches, small_body, medium_body, multi_frame_body);
criterion_main!(benches);
