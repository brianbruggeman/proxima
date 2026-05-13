//! Preface-sniff dispatch: one `HttpListenProtocol` listener
//! accepts both HTTP/1.1 and HTTP/2 prior-knowledge clients on the
//! same socket. No TLS, no ALPN — the first 4-24 bytes tell us
//! which protocol the client is speaking.
//!
//! This covers the UDS path (no TLS available, prior-knowledge h2
//! is the only way to get h2). It also implicitly validates plain-
//! TCP-without-TLS dispatch since the helper is shared.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]
#![cfg(all(feature = "http2", unix))]

use std::future::Future;

use bytes::Bytes;
use proxima::error::ProximaError;
use proxima::pipe::{into_handle};
use proxima::request::{Request, Response};
use proxima::{App, MountTarget, RunConfig};
use proxima_primitives::pipe::SendPipe;
use serde_json::json;
use tempfile::tempdir;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;

struct ConstantOk;

impl SendPipe for ConstantOk {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        _request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        async move { Ok(Response::ok(Bytes::from_static(b"ok"))) }
    }
}


async fn spawn_uds_listener(socket: &std::path::Path) -> proxima::Shutdown {
    let mut app = App::new().expect("app");
    let handle = into_handle(ConstantOk);
    let _ = app
        .pipe("__svc__", proxima::Spec::Handle(handle.clone()))
        .await
        .expect("register");
    app.mount("/{*path}", MountTarget::Handle(handle))
        .expect("mount");
    let mut config = RunConfig::http("127.0.0.1:0".parse().expect("addr"));
    config.spec = json!({"path": socket.to_string_lossy().to_string(), "mode": 0o600});
    app.run_until_signal(config).await.expect("run")
}

#[proxima::test(flavor = "multi_thread", worker_threads = 2)]
async fn h1_client_over_uds_via_preface_sniff_returns_200() {
    let dir = tempdir().expect("tempdir");
    let socket = dir.path().join("proxima.sock");
    let shutdown = spawn_uds_listener(&socket).await;

    // Raw HTTP/1.1 request over the UDS. First 4 bytes are "GET ",
    // which the preface sniff rejects as h2 → routes to h1 driver.
    let mut conn = UnixStream::connect(&socket).await.expect("connect");
    conn.write_all(b"GET / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
        .await
        .expect("write");
    conn.flush().await.expect("flush");

    let mut response = Vec::with_capacity(256);
    conn.read_to_end(&mut response).await.expect("read");
    let response_text = String::from_utf8_lossy(&response);
    assert!(
        response_text.starts_with("HTTP/1.1 200"),
        "expected 200 OK, got: {response_text:?}",
    );
    assert!(
        response_text.contains("ok"),
        "expected body to contain 'ok', got: {response_text:?}",
    );

    shutdown.stop();
}

#[proxima::test(flavor = "multi_thread", worker_threads = 2)]
async fn h2_prior_knowledge_client_over_uds_via_preface_sniff_returns_200() {
    let dir = tempdir().expect("tempdir");
    let socket = dir.path().join("proxima.sock");
    let shutdown = spawn_uds_listener(&socket).await;

    // h2 crate's client::handshake speaks prior-knowledge h2: sends
    // the 24-byte client preface immediately, then expects SETTINGS
    // frames. The preface sniff matches "PRI " in the first 4 bytes
    // → routes to serve_h2_connection.
    let conn = UnixStream::connect(&socket).await.expect("connect");
    let (mut client, h2_conn) = h2::client::handshake(conn).await.expect("h2 handshake");
    let conn_task = tokio::spawn(async move {
        let _ = h2_conn.await;
    });

    let request = http::Request::builder()
        .method("GET")
        .uri("http://localhost/ping")
        .body(())
        .expect("request");
    let (response_future, _) = client.send_request(request, true).expect("send");
    let response = response_future.await.expect("response");
    assert_eq!(response.status(), 200);

    let mut body = response.into_body();
    let mut collected = Vec::new();
    while let Some(chunk) = body.data().await {
        let chunk = chunk.expect("chunk");
        let len = chunk.len();
        body.flow_control()
            .release_capacity(len)
            .expect("flow control");
        collected.extend_from_slice(&chunk);
    }
    assert_eq!(&collected[..], b"ok");

    drop(client);
    drop(conn_task);
    shutdown.stop();
}
