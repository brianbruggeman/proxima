//! Apples-to-apples isolation diagnostic for the h2-on-prime body
//! delivery hang. Drives a hand-rolled blocking `std::net::TcpStream`
//! h2 client (using proxima's own `h2::frame` primitives — no tokio,
//! no `h2` crate) against TWO server arms:
//!
//! - `prime` — `PrimeRuntime` + `ProximaTcpListener` + `serve_h2_connection`
//! - `tokio` — `TokioPerCoreRuntime` + `tokio::net::TcpListener` + same
//!   `serve_h2_connection`
//!
//! The SAME client drives both. If the prime arm hangs at 5s and the
//! tokio arm passes, the bug is isolated to the prime server side
//! (most likely `ProximaTcpStream::poll_write` lying about `Ok(n)` or
//! `serve_h2_connection`'s `tokio::select!` poll-order interacting
//! badly with the prime executor when the only async actor is the
//! reactor itself).
//!
//! The client does no HPACK encoding of received frames — it skips
//! HEADERS payload bytes entirely and only parses DATA frames on
//! stream 1 until END_STREAM. That's enough to surface "did the server
//! send body bytes within 5s?", which is the question.

#![cfg(all(
    feature = "runtime-tokio",
    all(
        feature = "runtime-prime-executor",
        feature = "runtime-prime-inbox-alloc",
        feature = "runtime-prime-reactor",
        feature = "runtime-prime-bgpool"
    ),
    feature = "http2",
))]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

use std::any::Any;
use std::io::{self, Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use proxima::error::ProximaError;
use proxima::h2::frame::{
    CONNECTION_PREFACE, FRAME_HEADER_LEN, FrameHeader, FramePayload, FrameType, flags,
    parse_payload,
};
use proxima::h2::serve_h2_connection;
use proxima::pipe::{PipeHandle, into_handle};
use proxima::request::{Request, Response};
use proxima::runtime::prime::os::net::TcpListener as ProximaTcpListener;
use proxima::runtime::{CoreId, PrimeRuntime, Runtime, TokioPerCoreRuntime};
use proxima_primitives::pipe::SendPipe;
use tokio::net::TcpListener;
use tokio_util::compat::TokioAsyncReadCompatExt;

const RESPONSE_BODY: &[u8] = b"hello-h2";

struct EightBytePipe;

impl SendPipe for EightBytePipe {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        _request: Request<Bytes>,
    ) -> impl std::future::Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        async move { Ok(Response::ok(Bytes::from_static(RESPONSE_BODY))) }
    }
}


/// Minimal blocking h2 client. Returns the response body bytes for a
/// single `GET /` request on stream 1, or `Err` on timeout / protocol
/// error. The total 5s deadline is enforced via the socket's read
/// timeout — any single `read` that can't advance within 5s fails.
fn blocking_h2_request_body(addr: SocketAddr) -> io::Result<Vec<u8>> {
    let mut socket = TcpStream::connect(addr)?;
    socket.set_nodelay(true)?;
    socket.set_read_timeout(Some(Duration::from_secs(5)))?;
    socket.set_write_timeout(Some(Duration::from_secs(5)))?;

    // 1. Client preface: 24 magic bytes.
    socket.write_all(CONNECTION_PREFACE)?;

    // 2. Empty SETTINGS frame (header only, length=0).
    let settings_header = FrameHeader {
        length: 0,
        frame_type: FrameType::Settings,
        flags: 0,
        stream_id: 0,
    };
    socket.write_all(&settings_header.to_bytes())?;

    // 3. HEADERS for GET /, indexed-only HPACK:
    //   0x82  = :method: GET (static index 2)
    //   0x84  = :path: /     (static index 4)
    //   0x86  = :scheme: http (static index 6)
    //   0x41 0x09 + "localhost" = :authority literal w/ indexed name (idx 1)
    let header_block: [u8; 14] = [
        0x82, 0x84, 0x86, 0x41, 0x09, b'l', b'o', b'c', b'a', b'l', b'h', b'o', b's', b't',
    ];
    let headers_header = FrameHeader {
        length: header_block.len() as u32,
        frame_type: FrameType::Headers,
        flags: flags::END_HEADERS | flags::END_STREAM,
        stream_id: 1,
    };
    socket.write_all(&headers_header.to_bytes())?;
    socket.write_all(&header_block)?;

    // 4. Read frames until DATA(END_STREAM) on stream 1.
    let mut body = Vec::new();
    let mut frame_header_buf = [0u8; FRAME_HEADER_LEN];
    let mut settings_ack_sent = false;
    loop {
        socket
            .read_exact(&mut frame_header_buf)
            .map_err(|error| io::Error::new(error.kind(), format!("read frame header: {error}")))?;
        let header = FrameHeader::parse(&frame_header_buf)
            .ok_or_else(|| io::Error::other("frame header parse"))?;
        let payload_len = header.length as usize;
        let mut payload_buf = vec![0u8; payload_len];
        if payload_len > 0 {
            socket.read_exact(&mut payload_buf).map_err(|error| {
                io::Error::new(error.kind(), format!("read frame payload: {error}"))
            })?;
        }
        match header.frame_type {
            FrameType::Settings if !header.has_flag(flags::ACK) && !settings_ack_sent => {
                // Server's initial SETTINGS. ACK once.
                let ack = FrameHeader {
                    length: 0,
                    frame_type: FrameType::Settings,
                    flags: flags::ACK,
                    stream_id: 0,
                };
                socket.write_all(&ack.to_bytes())?;
                settings_ack_sent = true;
            }
            FrameType::Data if header.stream_id == 1 => {
                let payload = Bytes::from(payload_buf.clone());
                if let Ok(FramePayload::Data { data }) = parse_payload(&header, &payload) {
                    body.extend_from_slice(&data);
                }
                if header.has_flag(flags::END_STREAM) {
                    return Ok(body);
                }
            }
            FrameType::Headers if header.stream_id == 1 && header.has_flag(flags::END_STREAM) => {
                // server returned status-only (no body) and closed the stream.
                return Ok(body);
            }
            FrameType::GoAway => {
                return Err(io::Error::other(format!(
                    "server sent GOAWAY (length={payload_len})"
                )));
            }
            _ => {
                // Ignore everything else (server SETTINGS ACK, WINDOW_UPDATE, etc).
            }
        }
    }
}

fn start_prime_server() -> SocketAddr {
    let runtime: Arc<dyn Runtime> = Arc::new(PrimeRuntime::new(1).expect("prime runtime"));
    let dispatch: PipeHandle = into_handle(EightBytePipe);
    let (addr_tx, addr_rx) = std::sync::mpsc::channel::<SocketAddr>();
    runtime
        .spawn_factory_on_core(
            CoreId(0),
            Box::new(move || {
                let dispatch = dispatch;
                Box::pin(async move {
                    let mut listener =
                        ProximaTcpListener::bind("127.0.0.1:0".parse().expect("parse listen addr"))
                            .expect("bind");
                    let addr = listener.local_addr().expect("local_addr");
                    addr_tx.send(addr).expect("addr send");
                    loop {
                        let (socket, _peer) = match listener.accept().await {
                            Ok(value) => value,
                            Err(_) => break,
                        };
                        let dispatch = dispatch.clone();
                        proxima::runtime::prime::os::core_shard::spawn_on_current_core(Box::pin(
                            async move {
                                let admission =
                                    proxima_listen::admission::ConnAdmission::unbounded();
                                let _ =
                                    serve_h2_connection(socket, dispatch, admission, None)
                                        .await;
                            },
                        ));
                    }
                }) as Pin<Box<dyn std::future::Future<Output = ()> + 'static>>
            }),
        )
        .expect("spawn listener factory");
    let addr = addr_rx.recv().expect("addr");
    std::mem::forget(runtime);
    addr
}

fn start_tokio_server() -> SocketAddr {
    let runtime: Arc<dyn Runtime> = Arc::new(TokioPerCoreRuntime::new(2).expect("tokio per-core"));
    let dispatch: PipeHandle = into_handle(EightBytePipe);
    let (addr_tx, addr_rx) = std::sync::mpsc::channel::<SocketAddr>();
    runtime
        .spawn_factory_on_core(
            CoreId(0),
            Box::new(move || {
                Box::pin(async move {
                    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
                    let addr = listener.local_addr().expect("addr");
                    addr_tx.send(addr).expect("addr send");
                    loop {
                        let (socket, _) = match listener.accept().await {
                            Ok(value) => value,
                            Err(_) => break,
                        };
                        let _ = socket.set_nodelay(true);
                        let dispatch = dispatch.clone();
                        tokio::task::spawn_local(async move {
                            let admission = proxima_listen::admission::ConnAdmission::unbounded();
                            let _ = serve_h2_connection(
                                socket.compat(),
                                dispatch,
                                admission,
                                None,
                            )
                            .await;
                        });
                    }
                }) as Pin<Box<dyn std::future::Future<Output = ()> + 'static>>
            }),
        )
        .expect("spawn listener factory");
    let addr = addr_rx.recv().expect("addr");
    std::mem::forget(runtime);
    addr
}

#[test]
fn tokio_server_responds_to_native_h2_client_within_5s() {
    let addr = start_tokio_server();
    let body = blocking_h2_request_body(addr).expect("tokio server hung or errored");
    assert_eq!(body, RESPONSE_BODY, "tokio server returned unexpected body");
}

#[test]
fn prime_server_responds_to_native_h2_client_within_5s() {
    let addr = start_prime_server();
    let body = blocking_h2_request_body(addr).expect(
        "prime server hung or errored — confirms residual h2-on-prime bug \
         is server-side (same client passes against tokio)",
    );
    assert_eq!(body, RESPONSE_BODY, "prime server returned unexpected body");
}

/// Async variant of `blocking_h2_request_body` using
/// `tokio::net::TcpStream` instead of `std::net::TcpStream`. The wire-
/// level protocol logic is otherwise identical — same SETTINGS, same
/// HEADERS, same frame-reader loop. Used to isolate whether the
/// residual hang is caused by (a) tokio's async TcpStream reading
/// pattern (then this test will hang) or (b) something specific to the
/// `h2` crate's state machine (then this test will pass).
async fn tokio_async_h2_request_body(addr: SocketAddr) -> io::Result<Vec<u8>> {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let mut socket = tokio::net::TcpStream::connect(addr).await?;
    socket.set_nodelay(true)?;

    // 1. preface
    socket.write_all(CONNECTION_PREFACE).await?;
    // 2. empty SETTINGS
    let settings_header = FrameHeader {
        length: 0,
        frame_type: FrameType::Settings,
        flags: 0,
        stream_id: 0,
    };
    socket.write_all(&settings_header.to_bytes()).await?;
    // 3. HEADERS for GET / (same minimal HPACK as the blocking variant).
    let header_block: [u8; 14] = [
        0x82, 0x84, 0x86, 0x41, 0x09, b'l', b'o', b'c', b'a', b'l', b'h', b'o', b's', b't',
    ];
    let headers_header = FrameHeader {
        length: header_block.len() as u32,
        frame_type: FrameType::Headers,
        flags: flags::END_HEADERS | flags::END_STREAM,
        stream_id: 1,
    };
    socket.write_all(&headers_header.to_bytes()).await?;
    socket.write_all(&header_block).await?;

    let mut body = Vec::new();
    let mut frame_header_buf = [0u8; FRAME_HEADER_LEN];
    let mut settings_ack_sent = false;
    loop {
        socket
            .read_exact(&mut frame_header_buf)
            .await
            .map_err(|error| io::Error::new(error.kind(), format!("read frame header: {error}")))?;
        let header = FrameHeader::parse(&frame_header_buf)
            .ok_or_else(|| io::Error::other("frame header parse"))?;
        let payload_len = header.length as usize;
        let mut payload_buf = vec![0u8; payload_len];
        if payload_len > 0 {
            socket.read_exact(&mut payload_buf).await.map_err(|error| {
                io::Error::new(error.kind(), format!("read frame payload: {error}"))
            })?;
        }
        match header.frame_type {
            FrameType::Settings if !header.has_flag(flags::ACK) && !settings_ack_sent => {
                let ack = FrameHeader {
                    length: 0,
                    frame_type: FrameType::Settings,
                    flags: flags::ACK,
                    stream_id: 0,
                };
                socket.write_all(&ack.to_bytes()).await?;
                settings_ack_sent = true;
            }
            FrameType::Data if header.stream_id == 1 => {
                let payload = Bytes::from(payload_buf.clone());
                if let Ok(FramePayload::Data { data }) = parse_payload(&header, &payload) {
                    body.extend_from_slice(&data);
                }
                if header.has_flag(flags::END_STREAM) {
                    return Ok(body);
                }
            }
            FrameType::Headers if header.stream_id == 1 && header.has_flag(flags::END_STREAM) => {
                return Ok(body);
            }
            FrameType::GoAway => {
                return Err(io::Error::other(format!(
                    "server sent GOAWAY (length={payload_len})"
                )));
            }
            _ => {}
        }
    }
}

#[test]
fn prime_server_responds_to_tokio_async_hand_rolled_h2_client_within_5s() {
    let addr = start_prime_server();
    let outer = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("client runtime");
    outer.block_on(async move {
        let result =
            tokio::time::timeout(Duration::from_secs(5), tokio_async_h2_request_body(addr)).await;
        match result {
            Ok(Ok(body)) => assert_eq!(body, RESPONSE_BODY),
            Ok(Err(error)) => panic!("hand-rolled tokio client errored: {error}"),
            Err(_) => panic!(
                "prime + tokio-async + hand-rolled h2 hung past 5s — \
                 isolates the bug to tokio-side async reads, NOT to the h2 crate's state machine"
            ),
        }
    });
}

// Suppress the unused-import warning for `Any` — kept for parity with
// the other repro file in case the test grows BgPool variants later.
#[allow(dead_code)]
fn _any_unused() -> Option<Box<dyn Any + Send>> {
    None
}
