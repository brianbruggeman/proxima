//! End-to-end, tokio-free proof: `.websocket(handler)` (h1-native
//! upgrade, already tokio-free via the `websocket-upgrade` feature)
//! handing the hijacked socket to the new sans-IO `Session`
//! (`websocket-session`) — real bytes over a real loopback socket,
//! entirely on the prime runtime. The client is a plain blocking
//! `std::net::TcpStream` doing its own raw HTTP handshake + RFC 6455
//! framing by hand (via the same `websocket_frame` codec primitives
//! `Session` is built on) — nothing in this test's own execution touches
//! tokio either. Mirrors `tests/h1_native_prime.rs`'s tokio-freedom-proof
//! shape and `tests/e2e/listener_h2_native.rs`'s `Listener::builder()` +
//! bare `#[proxima::test]` (adaptive-picks-prime) pattern.
//!
//! Verify the build has no tokio:
//!
//!   cargo tree -p proxima --no-default-features \
//!     --features "websocket-session,http1-native,serve-prime,macros" \
//!     -e normal -i tokio
//!   # -> "warning: nothing to print" (empty result)
//!
//! Run it:
//!
//!   cargo test --test websocket_session_prime_e2e --no-default-features \
//!     --features "websocket-session,http1-native,serve-prime,macros"

#![allow(clippy::unwrap_used, clippy::expect_used)]
#![cfg(all(
    feature = "websocket-session",
    feature = "http1-native",
    feature = "serve-prime",
    feature = "macros"
))]

use std::future::Future;
use std::io::{Read, Write};
use std::net::{Ipv4Addr, SocketAddr, TcpStream};
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use futures::io::{AsyncReadExt, AsyncWriteExt};

use proxima::error::ProximaError;
use proxima::pipe::into_handle;
use proxima::request::{Request, Response};
use proxima::websocket_frame::session::{Event, Message, Session};
use proxima::websocket_frame::{Opcode, encode_header, parse_frame, unmask_in_place};
use proxima::{HijackedSocket, Listener, ListenerBuilderEntry, ListenerProtocolExt, SendPipe};

/// Bind once to let the OS assign a port, then drop synchronously so the
/// port is free for the real listener bind — same helper
/// `listener_h2_native.rs` uses for the identical reason.
fn free_loopback_addr() -> SocketAddr {
    let probe = std::net::TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).expect("probe bind");
    let addr = probe.local_addr().expect("probe addr");
    drop(probe);
    addr
}

/// `.serve()` resolves once the listener lane is SPAWNED, not once it is
/// actually accepting (`ListenerBuilder::serve`'s own doc) — a bounded
/// connect-retry loop closes that gap, mirroring
/// `listener_h2_native.rs`'s async bounded retry, adapted to a blocking
/// `std::net` client.
fn connect_with_retry(addr: SocketAddr) -> TcpStream {
    let mut attempts_left = 200;
    loop {
        match TcpStream::connect(addr) {
            Ok(stream) => return stream,
            Err(_) if attempts_left > 0 => {
                attempts_left -= 1;
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(error) => panic!("never connected to {addr}: {error}"),
        }
    }
}

/// Dispatch pipe for any request that ISN'T a WebSocket handshake —
/// `.handle(pipe)` is mandatory on `ListenerBuilder` even when every
/// request in this test is expected to upgrade.
struct NotFound;

impl SendPipe for NotFound {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        _request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        async move { Ok(Response::new(404)) }
    }
}

enum Action {
    NeedMoreBytes,
    Reply(Vec<u8>),
    Nothing,
    Done,
}

/// Converts a borrowed `Event<'_>` into an owned `Action` — severs the
/// borrow on `session` immediately so the driving loop below is free to
/// call `&mut session` methods again without a lifetime conflict.
fn classify(event: Event<'_>) -> Action {
    match event {
        Event::Incomplete => Action::NeedMoreBytes,
        Event::Message(Message::Text(text)) => {
            Action::Reply(server_frame(Opcode::Text, text.as_bytes()))
        }
        Event::Message(Message::Binary(data)) => Action::Reply(server_frame(Opcode::Binary, data)),
        Event::Ping(_) | Event::Pong(_) => Action::Nothing,
        Event::Closed { .. } => Action::Done,
    }
}

/// A server frame is never masked (RFC 6455 §5.1) — built with the same
/// `encode_header` primitive `Session` itself uses.
fn server_frame(opcode: Opcode, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    encode_header(true, opcode, payload.len(), None, &mut out);
    out.extend_from_slice(payload);
    out
}

async fn drain_transmit(
    session: &mut Session,
    stream: &mut Box<dyn proxima_primitives::pipe::upgrade::HijackStream>,
    scratch: &mut [u8],
) -> Result<(), ProximaError> {
    while let Some(count) = session.poll_transmit(scratch) {
        if count == 0 {
            break;
        }
        stream.write_all(&scratch[..count]).await.map_err(io_err)?;
    }
    Ok(())
}

fn io_err(error: futures::io::Error) -> ProximaError {
    ProximaError::Io(std::io::Error::other(format!(
        "websocket session io: {error}"
    )))
}

/// The post-101 handler: an echo server driven entirely by `Session` —
/// text/binary messages are echoed back verbatim, PING/CLOSE handling is
/// entirely `Session`'s job (automatic PONG, automatic Close echo). No
/// tokio, no runtime symbol, in this function.
fn echo_handler(
    hijacked: HijackedSocket,
) -> Pin<Box<dyn Future<Output = Result<(), ProximaError>> + Send>> {
    Box::pin(async move {
        let HijackedSocket {
            mut stream,
            leftover,
        } = hijacked;
        let mut session = Session::new();
        session.feed(&leftover);
        let mut read_buf = [0_u8; 4096];
        let mut xmit_buf = [0_u8; 4096];

        loop {
            match classify(session.poll_event()) {
                Action::NeedMoreBytes => {
                    drain_transmit(&mut session, &mut stream, &mut xmit_buf).await?;
                    if session.is_closed() {
                        return Ok(());
                    }
                    let read = stream.read(&mut read_buf).await.map_err(io_err)?;
                    if read == 0 {
                        return Ok(());
                    }
                    session.feed(&read_buf[..read]);
                }
                Action::Reply(bytes) => {
                    drain_transmit(&mut session, &mut stream, &mut xmit_buf).await?;
                    stream.write_all(&bytes).await.map_err(io_err)?;
                }
                Action::Nothing => {
                    drain_transmit(&mut session, &mut stream, &mut xmit_buf).await?;
                }
                Action::Done => {
                    drain_transmit(&mut session, &mut stream, &mut xmit_buf).await?;
                    return Ok(());
                }
            }
        }
    })
}

/// Build a CLIENT (masked) wire frame — the client half of this test
/// mirrors `proxima-protocols`' own `session::tests::client_frame`
/// helper, reusing the same codec primitives.
fn client_frame(fin: bool, opcode: Opcode, payload: &[u8], key: [u8; 4]) -> Vec<u8> {
    let mut buf = Vec::new();
    encode_header(fin, opcode, payload.len(), Some(key), &mut buf);
    let mut masked_payload = payload.to_vec();
    unmask_in_place(&mut masked_payload, key);
    buf.extend_from_slice(&masked_payload);
    buf
}

struct OwnedFrame {
    opcode: Opcode,
    mask: Option<[u8; 4]>,
    payload: Vec<u8>,
}

/// Blocking read of exactly one wire frame off `stream`, growing a local
/// buffer until `parse_frame` succeeds.
fn read_frame(stream: &mut TcpStream) -> OwnedFrame {
    let mut buffer = Vec::new();
    let mut chunk = [0_u8; 512];
    loop {
        if let Ok((frame, _used)) = parse_frame(&buffer) {
            return OwnedFrame {
                opcode: frame.opcode,
                mask: frame.mask,
                payload: frame.payload.to_vec(),
            };
        }
        let read = stream.read(&mut chunk).expect("read frame bytes");
        assert!(read > 0, "connection closed while waiting for a frame");
        buffer.extend_from_slice(&chunk[..read]);
    }
}

fn read_head(stream: &mut TcpStream) -> Vec<u8> {
    let mut buffer = Vec::new();
    let mut chunk = [0_u8; 256];
    loop {
        let read = stream.read(&mut chunk).expect("read response head");
        assert!(
            read > 0,
            "connection closed while reading the response head"
        );
        buffer.extend_from_slice(&chunk[..read]);
        if buffer.windows(4).any(|window| window == b"\r\n\r\n") {
            return buffer;
        }
    }
}

#[proxima::test]
async fn h1_native_websocket_session_echoes_over_a_real_socket_with_no_tokio() {
    let bind = free_loopback_addr();
    let handler: proxima::listener::WebSocketHandler = Arc::new(echo_handler);

    let _server = Listener::builder()
        .bind(bind)
        .websocket(handler)
        .handle(into_handle(NotFound))
        .serve()
        .await
        .expect("Listener::builder().websocket() serve");

    let mut stream = connect_with_retry(bind);
    stream.set_nodelay(true).expect("nodelay");

    // RFC 6455 §1.3's own worked handshake key/accept pair.
    let request = "GET /ws HTTP/1.1\r\nHost: localhost\r\nConnection: Upgrade\r\nUpgrade: websocket\r\n\
                    Sec-WebSocket-Version: 13\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n\r\n";
    stream
        .write_all(request.as_bytes())
        .expect("write handshake");

    let head = read_head(&mut stream);
    let head_text = String::from_utf8_lossy(&head);
    assert!(
        head_text.starts_with("HTTP/1.1 101"),
        "expected 101, got: {head_text}"
    );
    assert!(
        head_text.contains("s3pPLMBiTxaQ9kYGzzhZRbK+xOo="),
        "expected the RFC 6455 §1.3 worked-example accept key, got: {head_text}"
    );

    let key = [0x11, 0x22, 0x33, 0x44];

    // Text message round trip.
    stream
        .write_all(&client_frame(true, Opcode::Text, b"hello prime", key))
        .expect("write text");
    let frame = read_frame(&mut stream);
    assert_eq!(frame.opcode, Opcode::Text);
    assert_eq!(frame.payload, b"hello prime");
    assert!(
        frame.mask.is_none(),
        "server frames must not be masked (RFC 6455 §5.1)"
    );

    // Ping -> automatic Pong.
    stream
        .write_all(&client_frame(true, Opcode::Ping, b"pp", key))
        .expect("write ping");
    let frame = read_frame(&mut stream);
    assert_eq!(frame.opcode, Opcode::Pong);
    assert_eq!(frame.payload, b"pp");

    // Closing handshake.
    let mut close_payload = Vec::new();
    close_payload.extend_from_slice(&1000_u16.to_be_bytes());
    stream
        .write_all(&client_frame(true, Opcode::Close, &close_payload, key))
        .expect("write close");
    let frame = read_frame(&mut stream);
    assert_eq!(frame.opcode, Opcode::Close);
    assert_eq!(&frame.payload[..2], &1000_u16.to_be_bytes());

    // The server drops the transport once the handshake completes.
    let mut trailing = [0_u8; 8];
    let read = stream.read(&mut trailing).expect("final read");
    assert_eq!(
        read, 0,
        "server should close the transport once the WebSocket closing handshake completes"
    );
}
