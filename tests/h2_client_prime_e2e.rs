//! End-to-end proof of the native HTTP/2 CLIENT (`H2ClientUpstream`) over the
//! PRIME wire, against the native h2 SERVER (`serve_h2_connection`): a unary
//! POST with a request body round-trips to a fixed `200 pong` response. This is
//! the I/O-layer keystone for the client codec role — the sans-IO loopback lives
//! in `proxima_protocols::http2_codec`; this proves the same exchange over a real prime TCP
//! socket, client driver ↔ server driver, no tokio in the path.

#![cfg(all(feature = "http-prime", any(target_os = "linux", target_os = "macos")))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;

use bytes::Bytes;

use proxima::PrimeTcpUpstream;
use proxima::ProximaError;
use proxima::h2::{H2ClientUpstream, serve_h2_connection};
use proxima::pipe::{PipeHandle, into_handle};
use proxima::request::{Request, Response};
use proxima::runtime::prime::os::core_shard::spawn_on_current_core;
use proxima::runtime::prime::os::net::TcpListener;
use proxima_primitives::pipe::SendPipe;

// Fixed server handler: every request gets `200` + `pong`.
struct PongPipe;

impl SendPipe for PongPipe {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        _request: Request<Bytes>,
    ) -> impl core::future::Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        async { Ok(Response::new(200).with_body(Bytes::from_static(b"pong"))) }
    }
}


#[proxima::test]
async fn h2_client_unary_roundtrip_over_prime_wire() {
    let mut listener = TcpListener::bind("127.0.0.1:0".parse().unwrap()).expect("bind");
    let bound = listener.local_addr().expect("local_addr");

    // native h2 server: accept one connection and serve it with PongPipe.
    let dispatch: PipeHandle = into_handle(PongPipe);
    spawn_on_current_core(Box::pin(async move {
        if let Ok((socket, _peer)) = listener.accept().await {
            let admission = proxima_listen::admission::ConnAdmission::unbounded();
            let _ = serve_h2_connection(socket, dispatch, admission, None).await;
        }
    }));

    // native h2 client: one unary POST with a body, over the prime transport.
    let client = H2ClientUpstream::new(
        PrimeTcpUpstream::new(bound),
        format!("{bound}"),
        false,
        "h2-test",
    );
    let request = Request::builder()
        .method("POST")
        .path("/svc.Service/Method")
        .body(Bytes::from_static(b"ping"))
        .build()
        .expect("build request");
    let response = client.call(request).await.expect("h2 client call");

    assert_eq!(
        response.status, 200,
        "native h2 client read the server status"
    );
    assert_eq!(
        response.payload.as_ref(),
        b"pong",
        "native h2 client read the response body"
    );
}
