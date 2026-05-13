//! End-to-end coverage for the listener's connection-upgrade /
//! hijack path. Two scenarios:
//!
//! 1. CONNECT proxy — a Pipe responds 200 and pipes the client
//!    socket to a backend TCP echo. After the head, the listener
//!    cedes the socket and bytes flow bidirectionally.
//!
//! 2. 101 Switching Protocols — a Pipe responds 101 with an
//!    upgrade handler that runs a tiny line-based protocol over
//!    the hijacked socket. Stand-in for WebSocket / h2c without
//!    pulling in tokio-tungstenite or h2 as a test dep.

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
use std::time::Duration;

use bytes::Bytes;
use futures::AsyncReadExt as FuturesAsyncReadExt;
use futures::AsyncWriteExt as FuturesAsyncWriteExt;
use proxima::{
    App, HijackedSocket, MountTarget, ProximaError, Request, Response, RunConfig, Spec,
    UpgradeHandler, into_handle,
};
use proxima_primitives::pipe::SendPipe;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio_util::compat::TokioAsyncReadCompatExt;

async fn pick_free_addr() -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    drop(listener);
    addr
}

/// Spawn a TCP echo server on an ephemeral port. Each connection
/// reads bytes and writes them back; closes when the client
/// closes. Returns the bound address.
async fn start_tcp_echo() -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind echo");
    let addr = listener.local_addr().expect("echo addr");
    tokio::spawn(async move {
        loop {
            let Ok((mut socket, _)) = listener.accept().await else {
                return;
            };
            tokio::spawn(async move {
                let mut buf = [0_u8; 4096];
                loop {
                    let read = match socket.read(&mut buf).await {
                        Ok(0) => return,
                        Ok(n) => n,
                        Err(_) => return,
                    };
                    if socket.write_all(&buf[..read]).await.is_err() {
                        return;
                    }
                }
            });
        }
    });
    addr
}

/// CONNECT-proxy Pipe. Parses the request target (`host:port`)
/// from method line, dials the target, returns a 200 Response with
/// an `UpgradeHandler` that pipes bytes bidirectionally between the
/// client socket and the target socket.
struct ConnectProxy {
    target_addr: SocketAddr,
}

impl SendPipe for ConnectProxy {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        _request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> {
        let target_addr = self.target_addr;
        async move {
            let handler = UpgradeHandler::new(move |hijacked: HijackedSocket| async move {
                let HijackedSocket { stream, leftover } = hijacked;
                let target = TcpStream::connect(target_addr)
                    .await
                    .map_err(|err| ProximaError::Upstream(format!("connect target: {err}")))?;
                let mut target_compat = target.compat();
                // If the client sent tunnel data eagerly past the
                // CONNECT head, forward it to the target before
                // starting the duplex pump.
                if !leftover.is_empty() {
                    target_compat.write_all(&leftover).await.map_err(|err| {
                        ProximaError::Io(std::io::Error::other(format!("tunnel leftover: {err}")))
                    })?;
                }
                let (mut tr, mut tw) = target_compat.split();
                let (mut cr, mut cw) = stream.split();
                let client_to_target = async {
                    let _ = futures::io::copy(&mut cr, &mut tw).await;
                    let _ = tw.close().await;
                };
                let target_to_client = async {
                    let _ = futures::io::copy(&mut tr, &mut cw).await;
                    let _ = cw.close().await;
                };
                futures::join!(client_to_target, target_to_client);
                Ok(())
            });
            Ok(Response::new(200).with_upgrade(handler))
        }
    }
}


/// 101 Switching Protocols Pipe. Responds with a single 101 +
/// Upgrade: line-protocol, then runs a tiny line-counting protocol
/// over the hijacked socket: for every line of input, write back
/// `pong:<line>\n`. Closes on `quit\n`.
struct LineProtocolUpgrade;

impl SendPipe for LineProtocolUpgrade {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        _request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> {
        async move {
            let handler = UpgradeHandler::new(|hijacked: HijackedSocket| async move {
                let HijackedSocket {
                    mut stream,
                    leftover,
                } = hijacked;
                let mut buffer: Vec<u8> = leftover.to_vec();
                let mut read_buf = [0_u8; 1024];
                loop {
                    while let Some(newline) = buffer.iter().position(|&byte| byte == b'\n') {
                        let line: Vec<u8> = buffer.drain(..=newline).collect();
                        let line_no_lf = &line[..line.len() - 1];
                        if line_no_lf == b"quit" {
                            let _ = FuturesAsyncWriteExt::close(&mut stream).await;
                            return Ok(());
                        }
                        let mut reply: Vec<u8> = Vec::with_capacity(line_no_lf.len() + 6);
                        reply.extend_from_slice(b"pong:");
                        reply.extend_from_slice(line_no_lf);
                        reply.push(b'\n');
                        FuturesAsyncWriteExt::write_all(&mut stream, &reply)
                            .await
                            .map_err(|err| {
                                ProximaError::Io(std::io::Error::other(format!(
                                    "line proto write: {err}"
                                )))
                            })?;
                    }
                    let read = match FuturesAsyncReadExt::read(&mut stream, &mut read_buf).await {
                        Ok(0) => return Ok(()),
                        Ok(n) => n,
                        Err(error) => {
                            return Err(ProximaError::Io(std::io::Error::other(format!(
                                "line proto read: {error}"
                            ))));
                        }
                    };
                    buffer.extend_from_slice(&read_buf[..read]);
                }
            });
            let response = Response::new(101)
                .with_header("connection", "Upgrade")
                .with_header("upgrade", "x-line-proto")
                .with_upgrade(handler);
            Ok(response)
        }
    }
}


async fn boot_listener(handle: proxima::PipeHandle) -> (SocketAddr, proxima::Shutdown) {
    let mut app = App::new().expect("app");
    app.pipe("upgrade-test", Spec::Handle(handle))
        .await
        .expect("pipe");
    app.mount("/{*path}", MountTarget::Named("upgrade-test".into()))
        .expect("mount");
    let listener_addr = pick_free_addr().await;
    let shutdown = app
        .run_until_signal(RunConfig::http(listener_addr))
        .await
        .expect("run");
    (listener_addr, shutdown)
}

/// Read until we've seen `\r\n\r\n`. Returns the header bytes.
async fn read_response_headers(stream: &mut TcpStream) -> Vec<u8> {
    let mut accumulated: Vec<u8> = Vec::with_capacity(256);
    let mut buf = [0_u8; 256];
    loop {
        let read = stream.read(&mut buf).await.expect("read head");
        if read == 0 {
            return accumulated;
        }
        accumulated.extend_from_slice(&buf[..read]);
        if accumulated.windows(4).any(|window| window == b"\r\n\r\n") {
            return accumulated;
        }
    }
}

#[proxima::test]
async fn connect_proxy_pipes_bytes_to_target_after_200() {
    let echo_addr = start_tcp_echo().await;
    let proxy = ConnectProxy {
        target_addr: echo_addr,
    };
    let (listener_addr, shutdown) = boot_listener(into_handle(proxy)).await;

    let mut stream = TcpStream::connect(listener_addr).await.expect("connect");
    stream.set_nodelay(true).expect("nodelay");
    let request = format!("CONNECT {echo_addr} HTTP/1.1\r\nHost: {echo_addr}\r\n\r\n");
    stream.write_all(request.as_bytes()).await.expect("write");
    stream.flush().await.expect("flush");

    let head = read_response_headers(&mut stream).await;
    let head_text = String::from_utf8_lossy(&head);
    assert!(
        head_text.starts_with("HTTP/1.1 200"),
        "expected 200 status, got: {head_text}"
    );
    // Critical: no Transfer-Encoding or Content-Length header — the
    // hijack response must end at the blank line.
    let head_lower = head_text.to_ascii_lowercase();
    assert!(
        !head_lower.contains("transfer-encoding"),
        "200 upgrade response must not carry transfer-encoding: {head_text}"
    );
    assert!(
        !head_lower.contains("content-length"),
        "200 upgrade response must not carry content-length: {head_text}"
    );

    // After 200, the socket is a tunnel. Send "ping" and expect "ping" echoed.
    stream.write_all(b"ping").await.expect("write tunnel");
    stream.flush().await.expect("flush tunnel");
    let mut tunnel_buf = [0_u8; 16];
    let read = stream.read(&mut tunnel_buf).await.expect("read tunnel");
    assert_eq!(&tunnel_buf[..read], b"ping");

    drop(stream);
    shutdown.stop();
}

#[proxima::test]
async fn upgrade_101_runs_handler_protocol_after_switch() {
    let (listener_addr, shutdown) = boot_listener(into_handle(LineProtocolUpgrade)).await;

    let mut stream = TcpStream::connect(listener_addr).await.expect("connect");
    stream.set_nodelay(true).expect("nodelay");
    let request = "GET /upgrade HTTP/1.1\r\nHost: test\r\nConnection: Upgrade\r\nUpgrade: x-line-proto\r\n\r\n";
    stream
        .write_all(request.as_bytes())
        .await
        .expect("write request");
    stream.flush().await.expect("flush");

    let head = read_response_headers(&mut stream).await;
    let head_text = String::from_utf8_lossy(&head);
    assert!(
        head_text.starts_with("HTTP/1.1 101"),
        "expected 101 status, got: {head_text}"
    );
    let head_lower = head_text.to_ascii_lowercase();
    assert!(head_lower.contains("upgrade: x-line-proto"));
    assert!(!head_lower.contains("transfer-encoding"));
    assert!(!head_lower.contains("content-length"));

    // After 101, run the line protocol.
    stream.write_all(b"hello\n").await.expect("write");
    let mut buf = [0_u8; 32];
    let read = stream.read(&mut buf).await.expect("read pong");
    assert_eq!(&buf[..read], b"pong:hello\n");

    stream.write_all(b"world\n").await.expect("write 2");
    let read = stream.read(&mut buf).await.expect("read pong 2");
    assert_eq!(&buf[..read], b"pong:world\n");

    stream.write_all(b"quit\n").await.expect("write quit");
    // Server closes; final read returns 0.
    let read = stream.read(&mut buf).await.expect("final read");
    assert_eq!(read, 0);

    shutdown.stop();
}

#[proxima::test]
async fn upgrade_response_writes_response_head_only_no_chunked_terminator() {
    // Regression: an early bug had the listener inject
    // `transfer-encoding: chunked` + a `0\r\n\r\n` terminator after
    // the 101 head, corrupting the post-upgrade wire. This test
    // confirms the headers end at the blank line and no chunked
    // terminator is emitted.
    let (listener_addr, shutdown) = boot_listener(into_handle(LineProtocolUpgrade)).await;
    let mut stream = TcpStream::connect(listener_addr).await.expect("connect");
    stream.set_nodelay(true).expect("nodelay");
    stream
        .write_all(
            b"GET / HTTP/1.1\r\nHost: t\r\nConnection: Upgrade\r\nUpgrade: x-line-proto\r\n\r\n",
        )
        .await
        .expect("write");
    let head = read_response_headers(&mut stream).await;
    // Immediately after read_response_headers there should be zero
    // post-head bytes buffered server-side — otherwise the next read
    // would return a "0\r\n\r\n" or similar artifact instead of
    // (eventually) being whatever the handler sent.
    let mut peek = [0_u8; 8];
    let read = tokio::time::timeout(Duration::from_millis(100), stream.read(&mut peek)).await;
    // Either zero bytes (no further server output until we send
    // something) or timeout. Both confirm no spurious bytes.
    match read {
        Ok(Ok(0)) => {} // server closed unexpectedly — also acceptable
        Ok(Ok(_)) => panic!("server emitted bytes after upgrade head when none expected: {peek:?}"),
        Ok(Err(_)) => {} // io error fine
        Err(_) => {}     // timeout — expected
    }
    let _ = head;
    let _ = stream.write_all(b"quit\n").await;
    shutdown.stop();
}

// Keeps `Arc` imported even when test feature flags shrink the
// surface — without this the unused-import lint fires.
#[allow(dead_code)]
fn _typecheck_arc() -> Arc<()> {
    Arc::new(())
}

// Keeps `Bytes` imported — used by the leftover field in handler closures.
#[allow(dead_code)]
fn _typecheck_bytes() -> Bytes {
    Bytes::new()
}
