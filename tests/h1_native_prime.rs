//! Proves `HttpListenProtocol::serve` (the `"http"` listen protocol,
//! reached here via `PrimeServeExt::serve_http`, which is exactly
//! `ListenerSpec::http(bind).attach(pipe).run_with_runtime(..., Some(prime
//! AcceptorFactory), ...)`) drives a real h1 request/response round trip
//! through `serve_via_factory` -> the tokio-free `serve_connection` driver,
//! entirely on the prime runtime. Plain `#[test]`, not `#[tokio::test]` /
//! `#[proxima::test(runtime = "tokio")]` — `serve_http` is synchronous
//! (it blocks on the listener's readiness signal internally) and the
//! client below is a blocking `std::net::TcpStream`, so nothing in this
//! test's own execution touches tokio either.
//!
//! Gated on `http1-native` (NOT `http1`) — the tokio-coupled legacy
//! feature is never enabled to build or run this test. See
//! `examples/h1_native_prime_round_trip.rs` for the runnable-binary form
//! of the same proof plus the `cargo tree -i tokio` command.

#![allow(clippy::unwrap_used, clippy::expect_used)]
#![cfg(all(
    feature = "http1-native",
    feature = "runtime-prime-executor",
    feature = "runtime-prime-inbox-alloc",
    feature = "runtime-prime-reactor",
    feature = "runtime-prime-bgpool"
))]

use std::future::Future;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::Arc;

use bytes::Bytes;
use proxima::SendPipe;
use proxima::error::ProximaError;
use proxima::pipe::{PipeHandle, into_handle};
use proxima::prime::PrimeRuntime;
use proxima::request::{Request, Response};
use proxima::runtime::PrimeServeExt;

const GREETING: &str = "hello from the tokio-free h1 listener";

struct Greeter;

impl SendPipe for Greeter {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        _request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        async move { Ok(Response::ok(GREETING)) }
    }
}

#[test]
fn http_listen_protocol_serves_h1_over_prime_acceptor_factory_with_no_tokio() {
    let runtime = Arc::new(
        PrimeRuntime::builder()
            .cores(1)
            .background_inline()
            .build()
            .expect("build prime runtime"),
    );
    let pipe: PipeHandle = into_handle(Greeter);

    let bind = "127.0.0.1:0".parse().expect("bind addr parses");
    let handle = runtime
        .serve_http(bind, pipe)
        .expect("serve_http should bind through the prime AcceptorFactory");
    let addr = handle.bind_addr().expect("listener reports its bound address");

    let mut stream = TcpStream::connect(addr).expect("connect to the tokio-free listener");
    stream
        .write_all(b"GET /hello HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
        .expect("write request");
    let mut response = Vec::new();
    stream
        .read_to_end(&mut response)
        .expect("read response to EOF");

    let response_text = String::from_utf8_lossy(&response);
    assert!(
        response_text.starts_with("HTTP/1.1 200"),
        "expected a 200 response, got: {response_text}"
    );
    assert!(
        response_text.contains(GREETING),
        "expected the greeting in the response body: {response_text}"
    );

    handle.shutdown();
}
