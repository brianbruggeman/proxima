//! End-to-end coverage for the listener's streaming dispatch path.
//!
//! Walks a chunked-encoded upload through a real `App` + listener +
//! `Pipe`, verifying the body bytes reach the Pipe via the
//! mpsc body channel and the echoed response makes it back to the
//! client. Also exercises the cancel-token race for client
//! disconnection mid-upload.

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
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use bytes::Bytes;
use proxima::{
    App, MountTarget, ProximaError, Request, Response, RunConfig, Spec, into_handle,
};
use proxima_primitives::pipe::SendPipe;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

async fn pick_free_addr() -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    drop(listener);
    addr
}

/// Body-echo Pipe. Reads the request body to completion and
/// returns it as the response body. Designed to exercise the
/// streaming body channel.
struct BodyEcho {
    /// Set to true the first time `call` is invoked; lets a test
    /// confirm the Pipe was actually dispatched.
    dispatched: Arc<AtomicBool>,
    /// Set to true if `call` observed cancellation (request was
    /// dropped mid-flight). Used by the disconnect test.
    cancelled: Arc<AtomicBool>,
}

impl SendPipe for BodyEcho {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> {
        let dispatched = self.dispatched.clone();
        let cancelled = self.cancelled.clone();
        async move {
            dispatched.store(true, Ordering::Relaxed);
            let cancel = request.context.cancel.clone();
            // Drain the streaming body to completion (or cancel). The
            // listener fires `context.cancel` and drops the request
            // stream on client disconnect, ending the chunk stream.
            let mut chunk_stream = request.into_chunk_stream();
            let mut joined: Vec<u8> = Vec::new();
            let drain: Result<(), ProximaError> = async {
                while let Some(chunk) = futures::StreamExt::next(&mut chunk_stream).await {
                    joined.extend_from_slice(&chunk?);
                }
                Ok(())
            }
            .await;
            if cancel.is_fired() {
                cancelled.store(true, Ordering::Relaxed);
            }
            drain?;
            let len = joined.len();
            let response = Response::new(200)
                .with_header("content-length", len.to_string())
                .with_body(Bytes::from(joined));
            Ok(response)
        }
    }
}


async fn boot_echo_listener() -> (
    SocketAddr,
    proxima::Shutdown,
    Arc<AtomicBool>,
    Arc<AtomicBool>,
) {
    let dispatched = Arc::new(AtomicBool::new(false));
    let cancelled = Arc::new(AtomicBool::new(false));
    let echo = BodyEcho {
        dispatched: dispatched.clone(),
        cancelled: cancelled.clone(),
    };
    let mut app = App::new().expect("app");
    app.pipe("echo", Spec::Handle(into_handle(echo)))
        .await
        .expect("pipe");
    app.mount("/{*path}", MountTarget::Named("echo".into()))
        .expect("mount");
    let listener_addr = pick_free_addr().await;
    let shutdown = app
        .run_until_signal(RunConfig::http(listener_addr))
        .await
        .expect("run");
    (listener_addr, shutdown, dispatched, cancelled)
}

/// Drive a raw TCP request and return (status_code, body_bytes).
/// Reads through to the end of the response body (handles
/// chunked or content-length framing). Returns Err on socket errors.
async fn raw_request(addr: SocketAddr, request_bytes: &[u8]) -> std::io::Result<(u16, Vec<u8>)> {
    let mut stream = TcpStream::connect(addr).await?;
    stream.set_nodelay(true)?;
    stream.write_all(request_bytes).await?;
    stream.flush().await?;
    let mut received: Vec<u8> = Vec::new();
    let mut buf = [0_u8; 8 * 1024];
    loop {
        let read = stream.read(&mut buf).await?;
        if read == 0 {
            break;
        }
        received.extend_from_slice(&buf[..read]);
        if response_complete(&received) {
            break;
        }
    }
    parse_response(&received).ok_or_else(|| {
        std::io::Error::other(format!(
            "response parse failed: {:?}",
            String::from_utf8_lossy(&received)
        ))
    })
}

fn response_complete(buffer: &[u8]) -> bool {
    let Some(headers_end) = find_double_crlf(buffer) else {
        return false;
    };
    let headers_text = std::str::from_utf8(&buffer[..headers_end]).unwrap_or("");
    let mut content_length: Option<usize> = None;
    let mut chunked = false;
    for line in headers_text.lines() {
        let mut split = line.splitn(2, ':');
        match (split.next(), split.next()) {
            (Some(name), Some(value)) if name.eq_ignore_ascii_case("content-length") => {
                content_length = value.trim().parse().ok();
            }
            (Some(name), Some(value))
                if name.eq_ignore_ascii_case("transfer-encoding")
                    && value.trim().eq_ignore_ascii_case("chunked") =>
            {
                chunked = true;
            }
            _ => {}
        }
    }
    if let Some(length) = content_length {
        return buffer.len() >= headers_end + length;
    }
    if chunked {
        // chunked terminator: 0\r\n\r\n somewhere after headers.
        let body = &buffer[headers_end..];
        return body.windows(5).any(|window| window == b"0\r\n\r\n");
    }
    false
}

fn parse_response(buffer: &[u8]) -> Option<(u16, Vec<u8>)> {
    let headers_end = find_double_crlf(buffer)?;
    let status_line_end = buffer.iter().position(|&byte| byte == b'\n')?;
    let status_line = std::str::from_utf8(&buffer[..status_line_end]).ok()?;
    let mut parts = status_line.split_whitespace();
    let _version = parts.next()?;
    let status: u16 = parts.next()?.parse().ok()?;
    let headers_text = std::str::from_utf8(&buffer[..headers_end]).ok()?;
    let mut content_length: Option<usize> = None;
    let mut chunked = false;
    for line in headers_text.lines() {
        let mut split = line.splitn(2, ':');
        match (split.next(), split.next()) {
            (Some(name), Some(value)) if name.eq_ignore_ascii_case("content-length") => {
                content_length = value.trim().parse().ok();
            }
            (Some(name), Some(value))
                if name.eq_ignore_ascii_case("transfer-encoding")
                    && value.trim().eq_ignore_ascii_case("chunked") =>
            {
                chunked = true;
            }
            _ => {}
        }
    }
    let body_bytes = &buffer[headers_end..];
    let body = if let Some(length) = content_length {
        body_bytes.iter().take(length).copied().collect()
    } else if chunked {
        decode_chunked(body_bytes)?
    } else {
        body_bytes.to_vec()
    };
    Some((status, body))
}

fn decode_chunked(buffer: &[u8]) -> Option<Vec<u8>> {
    let mut decoded: Vec<u8> = Vec::new();
    let mut cursor = 0;
    while cursor < buffer.len() {
        let line_end = buffer[cursor..]
            .windows(2)
            .position(|window| window == b"\r\n")?;
        let size_line = std::str::from_utf8(&buffer[cursor..cursor + line_end]).ok()?;
        let size = usize::from_str_radix(size_line.trim(), 16).ok()?;
        cursor += line_end + 2;
        if size == 0 {
            return Some(decoded);
        }
        if cursor + size > buffer.len() {
            return None;
        }
        decoded.extend_from_slice(&buffer[cursor..cursor + size]);
        cursor += size + 2;
    }
    None
}

fn find_double_crlf(buffer: &[u8]) -> Option<usize> {
    buffer
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .map(|index| index + 4)
}

#[proxima::test]
async fn chunked_upload_streams_through_listener_and_echoes() {
    let (addr, shutdown, dispatched, _cancelled) = boot_echo_listener().await;
    // Three chunks: "hello", " streaming", " world". The total body
    // is 21 bytes — under the 1 MiB Content-Length threshold, but
    // Transfer-Encoding: chunked is in `stream_chunked: true`, so
    // the listener takes the streaming path.
    let mut request: Vec<u8> = Vec::new();
    request.extend_from_slice(
        b"POST /upload HTTP/1.1\r\nHost: test\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n",
    );
    request.extend_from_slice(b"5\r\nhello\r\n");
    request.extend_from_slice(b"a\r\n streaming\r\n");
    request.extend_from_slice(b"6\r\n world\r\n");
    request.extend_from_slice(b"0\r\n\r\n");
    let (status, body) = raw_request(addr, &request).await.expect("response");
    assert_eq!(status, 200);
    assert_eq!(&body[..], b"hello streaming world");
    assert!(dispatched.load(Ordering::Relaxed), "echo pipe was reached");
    shutdown.stop();
}

#[proxima::test]
async fn small_content_length_upload_stays_on_buffered_path() {
    // 5-byte body is far under the 1 MiB streaming threshold. The
    // buffered path handles it — verifies the streaming
    // wire-up didn't degrade the small-body case.
    let (addr, shutdown, dispatched, _cancelled) = boot_echo_listener().await;
    let mut request: Vec<u8> = Vec::new();
    request.extend_from_slice(
        b"POST /upload HTTP/1.1\r\nHost: test\r\nContent-Length: 5\r\nConnection: close\r\n\r\n",
    );
    request.extend_from_slice(b"hello");
    let (status, body) = raw_request(addr, &request).await.expect("response");
    assert_eq!(status, 200);
    assert_eq!(&body[..], b"hello");
    assert!(dispatched.load(Ordering::Relaxed));
    shutdown.stop();
}

#[proxima::test]
async fn client_disconnect_mid_chunked_upload_cancels_pipe() {
    let (addr, shutdown, dispatched, cancelled) = boot_echo_listener().await;
    // Write the head + the first chunk frame, then drop the
    // connection BEFORE sending the 0-length terminator. The
    // listener's pump_body_stream sees `Ok(0)` from reader.read and
    // fires the cancel token; BodyEcho observes
    // `request.context.cancel.is_cancelled()` and sets the flag.
    let mut stream = TcpStream::connect(addr).await.expect("connect");
    stream.set_nodelay(true).expect("nodelay");
    let mut request_head: Vec<u8> = Vec::new();
    request_head.extend_from_slice(
        b"POST /upload HTTP/1.1\r\nHost: test\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n",
    );
    request_head.extend_from_slice(b"5\r\nhello\r\n");
    stream.write_all(&request_head).await.expect("write head");
    stream.flush().await.expect("flush");
    // Give the Pipe a beat to start draining the body.
    tokio::time::sleep(Duration::from_millis(50)).await;
    drop(stream);
    // Wait briefly for the cancel to propagate.
    for _ in 0..50 {
        if cancelled.load(Ordering::Relaxed) {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(
        dispatched.load(Ordering::Relaxed),
        "echo pipe must have been reached"
    );
    assert!(
        cancelled.load(Ordering::Relaxed),
        "echo pipe should observe cancel"
    );
    shutdown.stop();
}

// Helps `App::Spec::Handle` accept the unit type — keeps the import
// graph honest by referencing the type explicitly.
#[allow(dead_code)]
fn _typecheck_pin() -> Pin<Box<dyn Future<Output = ()>>> {
    Box::pin(async {})
}
