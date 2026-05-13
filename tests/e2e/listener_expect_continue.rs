//! End-to-end coverage for RFC 7231 §5.1.1 `Expect: 100-continue`.
//!
//! Three scenarios:
//!
//! 1. Happy path — client sends `Expect: 100-continue` + small
//!    body, listener writes `100 Continue` before reading the body,
//!    Pipe echoes the body in a 200 response.
//!
//! 2. Rejection by max_body_bytes — client declares a
//!    Content-Length above the listener's limit, listener writes
//!    `413 Payload Too Large` BEFORE the client ever sends the
//!    body, then closes.
//!
//! 3. Streaming-mode interaction — chunked body with
//!    `Expect: 100-continue`. Listener emits 100 before reading any
//!    chunk frames; streaming dispatch proceeds normally afterward.

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
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use bytes::Bytes;
use proxima::{
    App, MountTarget, ProximaError, Request, Response, RunConfig, Spec, into_handle,
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

/// Echo Pipe — reads the body and returns it as the response.
struct BodyEcho {
    seen: Arc<AtomicBool>,
}

impl SendPipe for BodyEcho {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> {
        let seen = self.seen.clone();
        async move {
            seen.store(true, Ordering::Relaxed);
            let (_, body) = request.body_bytes().await?;
            let len = body.len();
            Ok(Response::new(200)
                .with_header("content-length", len.to_string())
                .with_body(body))
        }
    }
}


async fn boot_listener(
    seen: Arc<AtomicBool>,
    max_body_bytes: Option<usize>,
) -> (SocketAddr, proxima::Shutdown) {
    let mut app = App::new().expect("app");
    app.pipe("echo", Spec::Handle(into_handle(BodyEcho { seen })))
        .await
        .expect("pipe");
    app.mount("/{*path}", MountTarget::Named("echo".into()))
        .expect("mount");
    let listener_addr = pick_free_addr().await;
    let spec = match max_body_bytes {
        Some(limit) => json!({"max_body_bytes": limit}),
        None => json!({}),
    };
    let run_config = RunConfig {
        bind: listener_addr,
        protocol: "http".into(),
        spec,
    };
    let shutdown = app.run_until_signal(run_config).await.expect("run");
    (listener_addr, shutdown)
}

/// Read until we've seen a complete status line + headers block
/// terminated by `\r\n\r\n`. Returns the bytes read (including
/// the terminator).
async fn read_one_response_head(stream: &mut TcpStream) -> Vec<u8> {
    // Read ONE byte at a time and stop exactly at the first `\r\n\r\n`.
    // This connection is pipelined: the interim `100 Continue` is followed
    // by the final response, and under load both can land in one TCP segment.
    // A bulk read would pull the final response's bytes into this buffer and
    // drop them on the floor, leaving the next reader to see only EOF
    // ("response head incomplete"). Byte-at-a-time guarantees we leave the
    // final response untouched in the socket for `read_response_with_body`.
    let mut accumulated: Vec<u8> = Vec::with_capacity(64);
    let mut byte = [0_u8; 1];
    loop {
        let read = stream.read(&mut byte).await.expect("read head");
        if read == 0 {
            return accumulated;
        }
        accumulated.push(byte[0]);
        if let Some(end) = find_double_crlf(&accumulated) {
            return accumulated[..end].to_vec();
        }
    }
}

/// Read response head + body when Content-Length is set.
async fn read_response_with_body(stream: &mut TcpStream) -> (u16, Vec<u8>, Vec<u8>) {
    let mut accumulated: Vec<u8> = Vec::with_capacity(512);
    let mut buf = [0_u8; 1024];
    let mut head_end: Option<usize> = None;
    let mut content_length: Option<usize> = None;
    loop {
        let read = stream.read(&mut buf).await.expect("read");
        if read == 0 {
            break;
        }
        accumulated.extend_from_slice(&buf[..read]);
        if head_end.is_none()
            && let Some(end) = find_double_crlf(&accumulated)
        {
            head_end = Some(end);
            let head_text = std::str::from_utf8(&accumulated[..end]).unwrap_or("");
            for line in head_text.lines() {
                let mut parts = line.splitn(2, ':');
                if let (Some(name), Some(value)) = (parts.next(), parts.next())
                    && name.eq_ignore_ascii_case("content-length")
                {
                    content_length = value.trim().parse().ok();
                }
            }
        }
        if let (Some(end), Some(length)) = (head_end, content_length)
            && accumulated.len() >= end + length
        {
            break;
        }
    }
    let end = head_end.expect("response head incomplete");
    let head_text = std::str::from_utf8(&accumulated[..end]).expect("head ascii");
    let status_line = head_text.lines().next().expect("status line");
    let mut parts = status_line.split_whitespace();
    let _ = parts.next();
    let status: u16 = parts.next().expect("status").parse().expect("status num");
    let head = accumulated[..end].to_vec();
    let body = accumulated[end..].to_vec();
    (status, head, body)
}

fn find_double_crlf(buffer: &[u8]) -> Option<usize> {
    buffer
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .map(|index| index + 4)
}

#[proxima::test]
async fn happy_path_writes_100_continue_then_dispatches_body() {
    let seen = Arc::new(AtomicBool::new(false));
    let (addr, shutdown) = boot_listener(seen.clone(), None).await;

    let mut stream = TcpStream::connect(addr).await.expect("connect");
    stream.set_nodelay(true).expect("nodelay");
    // Send head WITH the body in a single write — proves the
    // listener writes 100 Continue independently of body bytes
    // already being available. (A polite client would wait for 100
    // before sending body; this is a strict server-side test.)
    let request =
        b"POST /upload HTTP/1.1\r\nHost: t\r\nContent-Length: 5\r\nExpect: 100-continue\r\nConnection: close\r\n\r\nhello";
    stream.write_all(request).await.expect("write");
    stream.flush().await.expect("flush");

    // First response: 100 Continue.
    let first_head = read_one_response_head(&mut stream).await;
    let first_text = String::from_utf8_lossy(&first_head);
    assert!(
        first_text.starts_with("HTTP/1.1 100 Continue"),
        "expected 100 Continue first, got: {first_text}"
    );

    // Second response: 200 with echoed body.
    let (status, _head, body) = read_response_with_body(&mut stream).await;
    assert_eq!(status, 200);
    assert_eq!(&body[..], b"hello");
    assert!(seen.load(Ordering::Relaxed), "Pipe was reached");
    shutdown.stop();
}

#[proxima::test]
async fn rejection_by_max_body_bytes_returns_413_before_inviting_body() {
    let seen = Arc::new(AtomicBool::new(false));
    // Listener cap = 10 bytes; client declares 9999 bytes.
    let (addr, shutdown) = boot_listener(seen.clone(), Some(10)).await;

    let mut stream = TcpStream::connect(addr).await.expect("connect");
    stream.set_nodelay(true).expect("nodelay");
    // Critical: write head only, no body. If the listener works
    // correctly it will respond 413 before reading body bytes.
    let request_head = b"POST /upload HTTP/1.1\r\nHost: t\r\nContent-Length: 9999\r\nExpect: 100-continue\r\nConnection: close\r\n\r\n";
    stream.write_all(request_head).await.expect("write");
    stream.flush().await.expect("flush");

    // The listener should write 413 and close — read the full
    // response without sending any body.
    let (status, _head, body) = read_response_with_body(&mut stream).await;
    assert_eq!(status, 413);
    assert!(
        std::str::from_utf8(&body)
            .map(|text| text.contains("exceeds limit"))
            .unwrap_or(false),
        "expected limit-exceeded body, got: {:?}",
        String::from_utf8_lossy(&body)
    );
    assert!(
        !seen.load(Ordering::Relaxed),
        "Pipe must NOT be reached for pre-body rejection"
    );
    shutdown.stop();
}

#[proxima::test]
async fn streaming_path_emits_100_before_pumping_chunked_body() {
    let seen = Arc::new(AtomicBool::new(false));
    let (addr, shutdown) = boot_listener(seen.clone(), None).await;

    let mut stream = TcpStream::connect(addr).await.expect("connect");
    stream.set_nodelay(true).expect("nodelay");
    // Chunked transfer triggers the auto-stream policy. Expect must
    // resolve before HeadReady is delivered to the streaming
    // dispatch path.
    let head = b"POST /upload HTTP/1.1\r\nHost: t\r\nTransfer-Encoding: chunked\r\nExpect: 100-continue\r\nConnection: close\r\n\r\n";
    stream.write_all(head).await.expect("write head");
    stream.flush().await.expect("flush head");

    // 100 Continue must come back BEFORE we send any chunks.
    let first_head = read_one_response_head(&mut stream).await;
    let first_text = String::from_utf8_lossy(&first_head);
    assert!(
        first_text.starts_with("HTTP/1.1 100 Continue"),
        "expected 100 Continue first, got: {first_text}"
    );

    // Now send the body chunks.
    stream.write_all(b"5\r\nhello\r\n").await.expect("chunk 1");
    stream.write_all(b"6\r\n world\r\n").await.expect("chunk 2");
    stream.write_all(b"0\r\n\r\n").await.expect("terminator");
    stream.flush().await.expect("flush chunks");

    let (status, _head, body) = read_response_with_body(&mut stream).await;
    assert_eq!(status, 200);
    assert_eq!(&body[..], b"hello world");
    assert!(seen.load(Ordering::Relaxed));
    shutdown.stop();
}

// Keep imports honest.
#[allow(dead_code)]
fn _typecheck() -> (Arc<()>, Bytes) {
    (Arc::new(()), Bytes::new())
}
