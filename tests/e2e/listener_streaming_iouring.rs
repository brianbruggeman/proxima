//! End-to-end streaming-body coverage for the io_uring listener
//! path. Walks a chunked upload through a real Pipe running
//! inside `tokio_uring::start`, verifying body bytes reach the
//! Pipe via the mpsc + streaming-Body channel and the echo
//! makes it back to the client.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]
#![cfg(all(target_os = "linux", feature = "io-uring"))]

use std::future::Future;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use futures::channel::oneshot;
use proxima::listeners::HttpListenerSpec;
use proxima::{PipeHandle, ProximaError, Request, Response, into_handle};
use serde_json::json;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// Echo Pipe — drains the streaming body and returns it as the
/// response body. Proves the chunks actually flowed through the
/// io_uring pump → mpsc → Pipe.
struct BodyEcho;

impl proxima::SendPipe for BodyEcho {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        async move {
            let (_, body) = request.body_bytes().await?;
            let len = body.len();
            Ok(Response::new(200)
                .with_header("content-length", len.to_string())
                .with_body(body))
        }
    }
}

#[test]
fn iouring_chunked_upload_streams_through_pipe_and_echoes() {
    tokio_uring::start(async move {
        let port = 28465_u16;
        let bind: SocketAddr = format!("127.0.0.1:{port}").parse().expect("addr");
        let dispatch: PipeHandle = into_handle(BodyEcho);
        let spec = Arc::new(HttpListenerSpec {
            max_body_bytes: None,
        });
        let raw_spec = json!({});
        let (shutdown_tx, shutdown_rx) = oneshot::channel();

        let raw_spec_clone = raw_spec.clone();
        let telemetry: proxima::TelemetryHandle = Arc::new(proxima::NoopTelemetry);
        tokio::task::spawn_local(async move {
            if let Err(error) = proxima::listeners::http_uring::serve_uring(
                bind,
                dispatch,
                spec,
                &raw_spec_clone,
                telemetry,
                shutdown_rx,
            )
            .await
            {
                eprintln!("serve_uring error: {error:?}");
            }
        });
        tokio::time::sleep(Duration::from_millis(150)).await;

        // Plain TCP client (no TLS — this test covers plaintext
        // streaming over io_uring). Use tokio::net since we just
        // need a basic AsyncRead/AsyncWrite.
        let mut stream = tokio::net::TcpStream::connect(bind).await.expect("connect");
        stream.set_nodelay(true).expect("nodelay");
        let mut request = Vec::new();
        request.extend_from_slice(
            b"POST /upload HTTP/1.1\r\nHost: test\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n",
        );
        request.extend_from_slice(b"5\r\nhello\r\n");
        request.extend_from_slice(b"6\r\n world\r\n");
        request.extend_from_slice(b"0\r\n\r\n");
        stream.write_all(&request).await.expect("write");
        stream.flush().await.expect("flush");

        let mut response = Vec::new();
        let _ = stream.read_to_end(&mut response).await;
        let text = String::from_utf8_lossy(&response);
        assert!(
            text.starts_with("HTTP/1.1 200"),
            "expected 200; got: {text}"
        );
        assert!(
            text.contains("hello world"),
            "expected body echo; got: {text}"
        );
        let _ = shutdown_tx.send(());
    });
}

#[test]
fn iouring_large_content_length_upload_uses_streaming_path() {
    tokio_uring::start(async move {
        let port = 28466_u16;
        let bind: SocketAddr = format!("127.0.0.1:{port}").parse().expect("addr");
        let dispatch: PipeHandle = into_handle(BodyEcho);
        let spec = Arc::new(HttpListenerSpec {
            max_body_bytes: None,
        });
        let raw_spec = json!({});
        let (shutdown_tx, shutdown_rx) = oneshot::channel();

        let raw_spec_clone = raw_spec.clone();
        let telemetry: proxima::TelemetryHandle = Arc::new(proxima::NoopTelemetry);
        tokio::task::spawn_local(async move {
            if let Err(error) = proxima::listeners::http_uring::serve_uring(
                bind,
                dispatch,
                spec,
                &raw_spec_clone,
                telemetry,
                shutdown_rx,
            )
            .await
            {
                eprintln!("serve_uring error: {error:?}");
            }
        });
        tokio::time::sleep(Duration::from_millis(150)).await;

        // 2 MiB body (over AutoStreamPolicy's default 1 MiB Content-Length
        // threshold) — should route through the streaming dispatch
        // path, not the buffered one.
        let body_size = 2 * 1024 * 1024;
        let body = Bytes::from(vec![b'A'; body_size]);
        let mut stream = tokio::net::TcpStream::connect(bind).await.expect("connect");
        stream.set_nodelay(true).expect("nodelay");
        let head = format!(
            "POST /big HTTP/1.1\r\nHost: test\r\nContent-Length: {body_size}\r\nConnection: close\r\n\r\n"
        );
        stream.write_all(head.as_bytes()).await.expect("write head");
        stream.write_all(&body).await.expect("write body");
        stream.flush().await.expect("flush");

        let mut response = Vec::with_capacity(body_size + 1024);
        let _ = stream.read_to_end(&mut response).await;
        // Parse: status line, headers, then body should be a 2 MiB
        // sequence of 'A'. Find the \r\n\r\n separator.
        let head_end = response
            .windows(4)
            .position(|window| window == b"\r\n\r\n")
            .expect("response head/body separator");
        let response_body = &response[head_end + 4..];
        assert_eq!(response_body.len(), body_size, "echoed body size mismatch");
        assert!(response_body.iter().all(|&byte| byte == b'A'));
        let _ = shutdown_tx.send(());
    });
}
