//! End-to-end coverage for HTTP/1.1 chunked-body trailers (RFC 7230 §4.1.2).
//!
//! Request side: client sends a chunked body with trailing
//! `X-Result: ok` etc. The listener captures them via the body
//! decoder and folds them into `request.headers`. The Pipe reads
//! them and echoes back as response headers.
//!
//! Response side: a Pipe constructs a streamed Response with
//! `ResponseStream::with_trailers(...)`. The listener emits the
//! trailers between the final `0\r\n` chunk-size line and the
//! terminating CRLF.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]
#![cfg(feature = "http1")]

use std::future::Future;
use std::net::SocketAddr;

use bytes::Bytes;
use proxima::{
    App, HeaderList, MountTarget, ProximaError, Request, Response, ResponseStream, RunConfig,
    Spec, into_handle,
};
use proxima_primitives::pipe::SendPipe;
use serde_json::json;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

async fn pick_free_addr() -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    drop(listener);
    addr
}


/// Pipe that reads the request body trailers and echoes them
/// back to the client as the response body so the test can
/// observe them.
struct TrailerEcho;

impl SendPipe for TrailerEcho {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> {
        async move {
            // Request trailers fold into `headers` at chunked-decode end
            // (yank-body). Drain the body first so the listener has folded
            // them, then read the trailer names the test sends.
            let (request, _drained) = request.body_bytes().await?;
            let mut buf: Vec<u8> = Vec::with_capacity(64);
            for name in ["X-Result", "X-Count"] {
                if let Some(value) = request.metadata.get(name) {
                    buf.extend_from_slice(name.as_bytes());
                    buf.extend_from_slice(b"=");
                    buf.extend_from_slice(value);
                    buf.push(b';');
                }
            }
            let body_text = if buf.is_empty() {
                b"no-trailers".to_vec()
            } else {
                buf
            };
            let len = body_text.len();
            Ok(Response::new(200)
                .with_header("content-length", len.to_string())
                .with_body(Bytes::from(body_text)))
        }
    }
}


/// Pipe that returns chunked body + trailers — exercises the
/// listener's trailer emission path.
struct TrailerEmitter;

impl SendPipe for TrailerEmitter {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        _request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> {
        async move {
            let mut trailers = HeaderList::new();
            trailers.insert(Bytes::from_static(b"X-Result"), Bytes::from_static(b"ok"));
            trailers.insert(Bytes::from_static(b"X-Count"), Bytes::from_static(b"42"));
            // No content-length → listener picks chunked framing,
            // which is the only framing that carries trailers.
            let stream =
                ResponseStream::once(Bytes::from_static(b"payload")).with_trailers(trailers);
            Ok(Response::streamed(stream))
        }
    }
}


async fn boot_listener(handle: proxima::PipeHandle) -> (SocketAddr, proxima::Shutdown) {
    let mut app = App::new().expect("app");
    app.pipe("trailers-test", Spec::Handle(handle))
        .await
        .expect("pipe");
    app.mount("/{*path}", MountTarget::Named("trailers-test".into()))
        .expect("mount");
    let listener_addr = pick_free_addr().await;
    let run_config = RunConfig {
        bind: listener_addr,
        protocol: "http".into(),
        spec: json!({}),
    };
    let shutdown = app.run_until_signal(run_config).await.expect("run");
    (listener_addr, shutdown)
}

async fn read_full_response(stream: &mut TcpStream) -> Vec<u8> {
    let mut received: Vec<u8> = Vec::with_capacity(512);
    let mut buf = [0_u8; 1024];
    loop {
        let read = stream.read(&mut buf).await.expect("read");
        if read == 0 {
            return received;
        }
        received.extend_from_slice(&buf[..read]);
    }
}

#[proxima::test]
async fn chunked_request_trailers_are_visible_to_pipe() {
    let (addr, shutdown) = boot_listener(into_handle(TrailerEcho)).await;
    let mut stream = TcpStream::connect(addr).await.expect("connect");
    stream.set_nodelay(true).expect("nodelay");
    let mut request = Vec::new();
    request.extend_from_slice(
        b"POST /upload HTTP/1.1\r\nHost: t\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n",
    );
    request.extend_from_slice(b"5\r\nhello\r\n");
    request.extend_from_slice(b"0\r\nX-Result: ok\r\nX-Count: 42\r\n\r\n");
    stream.write_all(&request).await.expect("write");
    stream.flush().await.expect("flush");

    let raw = read_full_response(&mut stream).await;
    let text = String::from_utf8_lossy(&raw);
    // Body echo: "X-Result=ok;X-Count=42;"
    assert!(text.contains("X-Result=ok"), "response: {text}");
    assert!(text.contains("X-Count=42"), "response: {text}");
    shutdown.stop();
}

#[proxima::test]
async fn chunked_request_without_trailers_yields_none() {
    let (addr, shutdown) = boot_listener(into_handle(TrailerEcho)).await;
    let mut stream = TcpStream::connect(addr).await.expect("connect");
    stream.set_nodelay(true).expect("nodelay");
    let request = b"POST /upload HTTP/1.1\r\nHost: t\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n5\r\nhello\r\n0\r\n\r\n";
    stream.write_all(request).await.expect("write");
    stream.flush().await.expect("flush");

    let raw = read_full_response(&mut stream).await;
    let text = String::from_utf8_lossy(&raw);
    assert!(text.contains("no-trailers"), "response: {text}");
    shutdown.stop();
}

#[proxima::test]
async fn pipe_response_trailers_are_emitted_after_zero_chunk() {
    let (addr, shutdown) = boot_listener(into_handle(TrailerEmitter)).await;
    let mut stream = TcpStream::connect(addr).await.expect("connect");
    stream.set_nodelay(true).expect("nodelay");
    let request = b"GET /any HTTP/1.1\r\nHost: t\r\nConnection: close\r\n\r\n";
    stream.write_all(request).await.expect("write");
    stream.flush().await.expect("flush");

    let raw = read_full_response(&mut stream).await;
    let text = String::from_utf8_lossy(&raw);
    // Body framing is chunked: "7\r\npayload\r\n0\r\nX-Result: ok\r\nX-Count: 42\r\n\r\n"
    assert!(
        text.contains("transfer-encoding: chunked"),
        "response: {text}"
    );
    assert!(text.contains("7\r\npayload\r\n"), "response: {text}");
    assert!(
        text.contains("0\r\nX-Result: ok\r\nX-Count: 42\r\n\r\n"),
        "response: {text}"
    );
    shutdown.stop();
}
