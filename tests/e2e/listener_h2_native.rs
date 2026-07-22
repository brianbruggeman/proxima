//! End-to-end test for `.h2()` through `Listener::builder()` — the builder
//! axis, not the hand-registered-registry shortcut `tests/e2e/listener_h2.rs`
//! (which drives `serve_h2_connection` directly) or
//! `tests/h2_client_prime_e2e.rs` (which builds a bare `ListenRegistry`) use.
//! Server side is the exact same native, tokio-free `H2ListenProtocol` those
//! reach by hand; here it is reached through `Listener::builder().h2()`.
//! Client is the native `H2ClientUpstream` — no external `h2` crate, no
//! tokio in the request path on either side.

#![allow(clippy::unwrap_used, clippy::expect_used)]
#![cfg(feature = "http2")]

use std::future::Future;
use std::net::{Ipv4Addr, SocketAddr};

use bytes::Bytes;

use proxima::error::ProximaError;
use proxima::h2::H2ClientUpstream;
use proxima::pipe::into_handle;
use proxima::request::{Request, Response};
use proxima::{Listener, ListenerBuilderEntry, PrimeTcpUpstream};
use proxima_primitives::pipe::SendPipe;

struct ConstantOk;

impl SendPipe for ConstantOk {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        _request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        async move { Ok(Response::new(200).with_body(Bytes::from_static(b"ok"))) }
    }
}

/// Bind once to let the OS assign a port, then drop synchronously so the
/// port is free for the real listener bind (`Server` has no `bind_addr()`
/// to discover an ephemeral port after `.serve()` resolves — see
/// `examples/h2_native_server.rs`'s identical helper).
fn free_loopback_addr() -> SocketAddr {
    let probe = std::net::TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).expect("probe bind");
    let addr = probe.local_addr().expect("probe addr");
    drop(probe);
    addr
}

// bare `#[proxima::test]` (no `flavor`/`runtime` override) picks prime,
// the adaptive default — required here: `H2ClientUpstream` rides
// `PrimeTcpUpstream`, which panics if polled off a prime worker
// (`CURRENT_REACTOR` null). The `.h3()` sibling test uses the tokio-flavor
// override because its quinn client genuinely needs tokio; this client
// doesn't.
#[proxima::test]
async fn listener_builder_h2_serves_a_real_native_h2_client() {
    let bind = free_loopback_addr();

    let server = Listener::builder()
        .bind(bind)
        .h2()
        .handle(into_handle(ConstantOk))
        .serve()
        .await
        .expect("Listener::builder().h2() serve");

    // `.serve()` resolves once the listener lane is spawned, not once it is
    // actually accepting (see `ListenerBuilder::serve`'s readiness-race
    // doc) — bounded retry-connect via the client itself closes that gap,
    // same as `examples/h2_native_server.rs`.
    let client = H2ClientUpstream::new(PrimeTcpUpstream::new(bind), format!("{bind}"), false, "h2-native-e2e");
    let mut attempts_left = 200;
    let response = loop {
        let request = Request::builder()
            .method("POST")
            .path("/")
            .body(Bytes::from_static(b"ping"))
            .build()
            .expect("build request");
        match client.call(request).await {
            Ok(response) => break response,
            Err(_) if attempts_left > 0 => {
                attempts_left -= 1;
                proxima::time::sleep(std::time::Duration::from_millis(10)).await;
            }
            Err(error) => panic!("h2 client never connected: {error}"),
        }
    };

    assert_eq!(response.status, 200, "h2 round trip must return 200");
    assert_eq!(
        response.payload.as_ref(),
        b"ok",
        "h2 round trip must read the server's response body"
    );

    server.stop();
}
