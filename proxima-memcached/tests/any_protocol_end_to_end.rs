#![allow(clippy::unwrap_used, clippy::expect_used)]
//! End-to-end coverage of `MemcachedAnyProtocol::drive` over a REAL
//! loopback socket — not just `framed_app`'s in-process
//! `MemcachedFramedApp::call` unit tests. Reshaped from the deleted
//! `connection.rs`'s `serve_connection`/`main_loop` test suite (a
//! scripted-socket `drive(wire, config)` helper) plus `pipe.rs`'s
//! CONNECT/upgrade tests — those two files, and the CONNECT/upgrade
//! indirection they existed to bridge, are gone: `MemcachedAnyProtocol`
//! now drives a `proxima_listen::any::FramedAny` directly, so this suite
//! asserts the SAME wire-observable behavior (get/set/noreply/quit/
//! pipelining/protocol-violation/message-too-large/admission-shed)
//! through the new driver instead.

use std::net::{Ipv4Addr, SocketAddr};

use bytes::Bytes;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

use proxima_core::ProximaError;
use proxima_listen::admission::ConnAdmission;
use proxima_listen::any::{AnyProtocol, erase_handler};
use proxima_net::tokio::tokio_stream_listener::TokioTcpConnection;
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::request::Response;
use proxima_primitives::stream::StreamConnection;
use proxima_protocols::memcached::{MemcachedRequest, Reply, StoredValue};

use proxima_memcached::{
    MemcachedAnyProtocol, MemcachedPipeHandle, MemcachedPipeReply, MemcachedPipeRequest,
    MemcachedServerConfig, into_memcached_handle,
};

struct EchoHandler;

impl SendPipe for EchoHandler {
    type In = MemcachedPipeRequest;
    type Out = MemcachedPipeReply;
    type Err = ProximaError;

    async fn call(&self, request: MemcachedPipeRequest) -> Result<MemcachedPipeReply, ProximaError> {
        let reply = match request.payload {
            MemcachedRequest::Get { keys, .. } if keys == Bytes::from_static(b"k") => {
                Reply::Values(vec![StoredValue {
                    key: b"k".to_vec(),
                    flags: 0,
                    data: b"stub-value".to_vec(),
                    cas_unique: None,
                }])
            }
            MemcachedRequest::Get { .. } => Reply::Values(Vec::new()),
            MemcachedRequest::Store { .. } => Reply::Stored,
            MemcachedRequest::Delete { .. } => Reply::Deleted,
            _ => Reply::Error,
        };
        Ok(Response::typed(200, reply))
    }
}

fn handler() -> MemcachedPipeHandle {
    into_memcached_handle(EchoHandler)
}

async fn bind_loopback() -> (TcpListener, SocketAddr) {
    let listener = TcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))
        .await
        .expect("bind");
    let addr = listener.local_addr().expect("local_addr");
    (listener, addr)
}

/// Accepts exactly one connection and drives it to completion through
/// `protocol`, returning once `drive` returns (clean close or error).
async fn accept_and_drive(
    listener: TcpListener,
    protocol: MemcachedAnyProtocol,
    admission: ConnAdmission,
) -> Result<(), ProximaError> {
    let (stream, _peer) = listener.accept().await.expect("accept");
    let connection: Box<dyn StreamConnection> = Box::new(TokioTcpConnection::from_tokio(stream));
    let handler = erase_handler(());
    protocol
        .drive(connection, handler, &serde_json::Value::Null, None, &admission)
        .await
}

#[tokio::test]
async fn get_hit_reaches_the_handler() {
    let (listener, addr) = bind_loopback().await;
    let protocol = MemcachedAnyProtocol::new("memcached", handler());
    let server = tokio::spawn(accept_and_drive(listener, protocol, ConnAdmission::unbounded()));

    let mut stream = tokio::net::TcpStream::connect(addr).await.expect("connect");
    stream.write_all(b"get k\r\nquit\r\n").await.expect("write");
    let mut reply = Vec::new();
    stream.read_to_end(&mut reply).await.expect("read to eof");
    assert_eq!(reply, b"VALUE k 0 10\r\nstub-value\r\nEND\r\n");
    server.await.expect("server task").expect("drive");
}

#[tokio::test]
async fn set_reaches_the_handler_and_replies_stored() {
    let (listener, addr) = bind_loopback().await;
    let protocol = MemcachedAnyProtocol::new("memcached", handler());
    let server = tokio::spawn(accept_and_drive(listener, protocol, ConnAdmission::unbounded()));

    let mut stream = tokio::net::TcpStream::connect(addr).await.expect("connect");
    stream
        .write_all(b"set k 0 0 5\r\nhello\r\nquit\r\n")
        .await
        .expect("write");
    let mut reply = Vec::new();
    stream.read_to_end(&mut reply).await.expect("read to eof");
    assert_eq!(reply, b"STORED\r\n");
    server.await.expect("server task").expect("drive");
}

#[tokio::test]
async fn noreply_set_never_writes_a_reply() {
    let (listener, addr) = bind_loopback().await;
    let protocol = MemcachedAnyProtocol::new("memcached", handler());
    let server = tokio::spawn(accept_and_drive(listener, protocol, ConnAdmission::unbounded()));

    let mut stream = tokio::net::TcpStream::connect(addr).await.expect("connect");
    stream
        .write_all(b"set k 0 0 5 noreply\r\nhello\r\nquit\r\n")
        .await
        .expect("write");
    let mut reply = Vec::new();
    stream.read_to_end(&mut reply).await.expect("read to eof");
    assert_eq!(reply, b"", "noreply must suppress the STORED reply");
    server.await.expect("server task").expect("drive");
}

/// Pipelining: two commands in ONE write, only the second is `noreply`.
/// Proves `AsFrame::as_frame() -> None` skips exactly the silent
/// command's own reply, not the whole batch.
#[tokio::test]
async fn a_noreply_command_pipelined_behind_a_normal_one_only_suppresses_its_own_reply() {
    let (listener, addr) = bind_loopback().await;
    let protocol = MemcachedAnyProtocol::new("memcached", handler());
    let server = tokio::spawn(accept_and_drive(listener, protocol, ConnAdmission::unbounded()));

    let mut stream = tokio::net::TcpStream::connect(addr).await.expect("connect");
    stream
        .write_all(b"set a 0 0 1\r\nx\r\nset b 0 0 1 noreply\r\ny\r\nquit\r\n")
        .await
        .expect("write");
    let mut reply = Vec::new();
    stream.read_to_end(&mut reply).await.expect("read to eof");
    assert_eq!(reply, b"STORED\r\n");
    server.await.expect("server task").expect("drive");
}

/// `quit` closes with no reply at all — the deleted `main_loop`'s
/// `FrameOutcome::Close` behavior, now `MemcachedOutcome::CloseSilent`.
#[tokio::test]
async fn quit_closes_the_connection_with_no_reply() {
    let (listener, addr) = bind_loopback().await;
    let protocol = MemcachedAnyProtocol::new("memcached", handler());
    let server = tokio::spawn(accept_and_drive(listener, protocol, ConnAdmission::unbounded()));

    let mut stream = tokio::net::TcpStream::connect(addr).await.expect("connect");
    stream.write_all(b"quit\r\n").await.expect("write");
    let mut reply = Vec::new();
    stream.read_to_end(&mut reply).await.expect("read to eof");
    assert_eq!(reply, b"");
    server.await.expect("server task").expect("drive");
}

/// An unknown verb answers a bare `ERROR\r\n` and closes the
/// connection — a pipelined command sent in the SAME write right behind
/// it must never be answered (the deleted `Advanced::ProtocolError`'s
/// "no trustworthy boundary to skip past" close-outright behavior, now
/// `MemcachedOutcome::CloseWithReply` + `keep_serving() == false`).
#[tokio::test]
async fn unknown_command_closes_the_connection_with_an_error() {
    let (listener, addr) = bind_loopback().await;
    let protocol = MemcachedAnyProtocol::new("memcached", handler());
    let server = tokio::spawn(accept_and_drive(listener, protocol, ConnAdmission::unbounded()));

    let mut stream = tokio::net::TcpStream::connect(addr).await.expect("connect");
    stream
        .write_all(b"bogus\r\nget k\r\n")
        .await
        .expect("write");
    let mut reply = Vec::new();
    stream.read_to_end(&mut reply).await.expect("read to eof");
    assert_eq!(
        reply, b"ERROR\r\n",
        "the pipelined get must never be answered once a violation closes the connection"
    );
    server.await.expect("server task").expect("drive");
}

/// A still-incomplete `set` whose declared value already exceeds the
/// configured cap closes with a descriptive `SERVER_ERROR`, matching the
/// deleted `Advanced::MessageTooLarge` reply text exactly.
#[tokio::test]
async fn an_oversized_value_closes_with_a_message_too_large_server_error() {
    let (listener, addr) = bind_loopback().await;
    // the command LINE alone ("set k 0 0 1000\r\n") is 16 bytes — a cap of
    // 8 is already exceeded before the declared 1000-byte value even
    // starts arriving, so `parse_frame` folds this into a `Violation`
    // instead of waiting forever for bytes that never come.
    let config = MemcachedServerConfig::builder().max_message_bytes(8).build();
    let protocol = MemcachedAnyProtocol::new("memcached", handler()).with_config(config);
    let server = tokio::spawn(accept_and_drive(listener, protocol, ConnAdmission::unbounded()));

    let mut stream = tokio::net::TcpStream::connect(addr).await.expect("connect");
    stream
        .write_all(b"set k 0 0 1000\r\n")
        .await
        .expect("write");
    let mut reply = Vec::new();
    stream.read_to_end(&mut reply).await.expect("read to eof");
    assert_eq!(reply, b"SERVER_ERROR message exceeds 8 byte limit\r\n");
    server.await.expect("server task").expect("drive");
}

/// The listener's admission policy — not the business handler — decides
/// whether a command reaches the engine while quiescing. `FramedAny`'s
/// generic `AdmittedApp` checks admission on EVERY frame uniformly,
/// `quit` included, but `memcached`'s own `shed_reply` (installed as
/// `FramedAny`'s `Shed`) special-cases `quit` the same way the deleted
/// driver's admission bypass did: a shed `quit` still closes the
/// connection instead of answering a `SERVER_ERROR` and staying open, so
/// a trailing `quit` right behind the shed command now closes cleanly
/// with no reply of its own.
#[tokio::test]
async fn business_command_is_shed_with_a_server_error_reply_while_admission_is_quiescing() {
    let (listener, addr) = bind_loopback().await;
    let protocol = MemcachedAnyProtocol::new("memcached", handler());
    let admission = ConnAdmission::unbounded();
    admission.begin_quiesce();
    let server = tokio::spawn(accept_and_drive(listener, protocol, admission));

    let mut stream = tokio::net::TcpStream::connect(addr).await.expect("connect");
    stream
        .write_all(b"delete k\r\nquit\r\n")
        .await
        .expect("write");
    let mut reply = Vec::new();
    stream.read_to_end(&mut reply).await.expect("read to eof");
    let reply_text = String::from_utf8_lossy(&reply);
    assert!(
        reply_text.starts_with("SERVER_ERROR server is shedding requests"),
        "expected a shed error reply, got: {reply_text:?}"
    );
    assert!(
        reply_text.trim_end().ends_with("retry shortly"),
        "the shed quit must not append its own reply after delete's, got: {reply_text:?}"
    );
    server.await.expect("server task").expect("drive");
}

/// A `noreply`-flagged command that gets admission-shed stays silent —
/// the same silence its own successful dispatch would produce — instead
/// of answering a `SERVER_ERROR` that a real memcached client, having
/// declared `noreply`, would never read.
#[tokio::test]
async fn a_noreply_command_stays_silent_when_admission_shed() {
    let (listener, addr) = bind_loopback().await;
    let protocol = MemcachedAnyProtocol::new("memcached", handler());
    let admission = ConnAdmission::unbounded();
    admission.begin_quiesce();
    let server = tokio::spawn(accept_and_drive(listener, protocol, admission));

    let mut stream = tokio::net::TcpStream::connect(addr).await.expect("connect");
    stream
        .write_all(b"set k 0 0 5 noreply\r\nhello\r\nquit\r\n")
        .await
        .expect("write");
    let mut reply = Vec::new();
    stream.read_to_end(&mut reply).await.expect("read to eof");
    assert_eq!(
        reply, b"",
        "a shed noreply command must stay silent, and the trailing shed quit must not reply either"
    );
    server.await.expect("server task").expect("drive");
}
