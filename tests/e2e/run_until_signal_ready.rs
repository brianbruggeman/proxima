//! Proves `App::run_until_signal` does not return until its listener is
//! actually accepting connections. Before the fix, the socket was only
//! resolved — the serve future had not yet been polled on core 0 — so a
//! client dialing the instant `.await` returned got `ECONNREFUSED`. No
//! sleep, no retry loop: a single connect attempt right after `.await`
//! returns must succeed.

#![allow(clippy::unwrap_used, clippy::expect_used)]
#![cfg(feature = "http1")]

use std::future::Future;
use std::net::SocketAddr;

use bytes::Bytes;
use proxima::{App, MountTarget, ProximaError, Request, Response, RunConfig, Spec, into_handle};
use proxima_primitives::pipe::SendPipe;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

struct StaticOk;

impl SendPipe for StaticOk {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        _request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> {
        async move {
            Ok(Response::new(200)
                .with_header("content-length", "2")
                .with_body(Bytes::from_static(b"ok")))
        }
    }
}

async fn pick_free_addr() -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    drop(listener);
    addr
}

#[proxima::test]
async fn run_until_signal_returns_only_once_the_listener_accepts() {
    let mut app = App::new().expect("app");
    app.pipe("ready-test", Spec::Handle(into_handle(StaticOk)))
        .await
        .expect("pipe");
    app.mount("/{*path}", MountTarget::Named("ready-test".into()))
        .expect("mount");

    let listener_addr = pick_free_addr().await;
    let shutdown = app
        .run_until_signal(RunConfig::http(listener_addr))
        .await
        .expect("run");

    // no sleep, no retry: the connect + request must succeed on the
    // very first attempt, immediately after run_until_signal returns.
    let mut stream = TcpStream::connect(listener_addr)
        .await
        .expect("listener must already be accepting when run_until_signal returns");
    stream
        .write_all(b"GET / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
        .await
        .expect("write request");

    let mut response = Vec::new();
    stream
        .read_to_end(&mut response)
        .await
        .expect("read response");
    let response_text = String::from_utf8_lossy(&response);
    assert!(
        response_text.starts_with("HTTP/1.1 200"),
        "expected 200 response, got: {response_text}"
    );

    shutdown.stop();
}
