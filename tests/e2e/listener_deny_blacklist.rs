//! Worked example: `.deny(name, literal)` + `.blacklist(config)` on the open
//! universal listener. A connection matching the deny literal is dropped and
//! its peer banned; the SAME peer's next connection — even a legit h1
//! request — is then dropped BEFORE the classifier ever sees it. A legit h1
//! client dials the SAME listener first and gets routed normally, proving
//! the deny signature is reviewed ALONGSIDE the legit candidates rather than
//! shadowing them.

#![allow(clippy::unwrap_used, clippy::expect_used)]
#![cfg(all(
    feature = "http1",
    any(
        feature = "runtime-tokio",
        all(
            feature = "serve-prime",
            feature = "runtime-prime-reactor",
            any(target_os = "linux", target_os = "macos")
        )
    )
))]

use std::future::Future;
use std::net::{SocketAddr, TcpListener as StdTcpListener, TcpStream as StdTcpStream};
use std::time::Duration;

use bytes::Bytes;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

use proxima::error::ProximaError;
use proxima::pipe::into_handle;
use proxima::request::{Request, Response};
use proxima::{Listener, ListenerBuilderEntry, SendPipe, TransportSugar};
use proxima_listen::admission::BlacklistConfig;

/// The scanner literal `.deny("scanner", SCANNER_LITERAL)` registers — an
/// arbitrary malicious-probe-shaped byte string, distinct from any h1
/// method line and from h2's RFC 9113 preface.
const SCANNER_LITERAL: &[u8] = b"XSCANPROBE\r\n";

struct LegitOk;

impl SendPipe for LegitOk {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        _request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        async move { Ok(Response::ok("legit-ok")) }
    }
}

fn free_loopback_addr() -> SocketAddr {
    let probe = StdTcpListener::bind("127.0.0.1:0").expect("probe bind");
    let addr = probe.local_addr().expect("probe addr");
    drop(probe);
    addr
}

fn wait_until_listening(addr: SocketAddr) {
    for _ in 0..200 {
        if StdTcpStream::connect(addr).is_ok() {
            return;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    panic!("listener at {addr} never came up");
}

#[proxima::test(flavor = "multi_thread", worker_threads = 2)]
async fn deny_signature_drops_and_bans_the_peer_pre_classify_without_breaking_legit_traffic() {
    let bind = free_loopback_addr();

    let server = Listener::builder()
        .bind(bind)
        .tcp()
        .handle(into_handle(LegitOk))
        .deny("scanner", SCANNER_LITERAL.to_vec())
        .blacklist(
            BlacklistConfig::layered()
                .with_deny_strike_threshold(1)
                .build(),
        )
        .serve()
        .await
        .expect("listener builder serves");

    wait_until_listening(bind);

    // Phase 1: a legit h1 client, BEFORE any ban is in effect, must route
    // normally — the deny signature is reviewed alongside h1, not instead
    // of it.
    let mut legit_conn = TcpStream::connect(bind).await.expect("legit connect");
    legit_conn
        .write_all(b"GET / HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
        .await
        .expect("legit write");
    legit_conn.flush().await.expect("legit flush");
    let mut legit_response = Vec::with_capacity(256);
    legit_conn
        .read_to_end(&mut legit_response)
        .await
        .expect("legit read");
    let legit_text = String::from_utf8_lossy(&legit_response);
    assert!(
        legit_text.starts_with("HTTP/1.1 200"),
        "legit h1 traffic must still route normally through the same listener; got: {legit_text:?}"
    );
    assert!(
        legit_text.contains("legit-ok"),
        "expected the legit body, got: {legit_text:?}"
    );

    // Phase 2: a connection matching the deny literal must be dropped
    // outright (no HTTP response of any kind) and must ban the peer. The
    // server may close with a clean FIN (empty `read_to_end`) or, since
    // dropping the accepted socket can race the kernel's own buffering,
    // with an RST the client observes as `ConnectionReset` — either is
    // "never got a response"; only a real HTTP status line would mean the
    // deny signature failed to intercept it.
    let mut scanner_conn = TcpStream::connect(bind).await.expect("scanner connect");
    let scanner_response = dial_and_collect(&mut scanner_conn, SCANNER_LITERAL).await;
    let scanner_text = String::from_utf8_lossy(&scanner_response);
    assert!(
        !scanner_text.starts_with("HTTP/"),
        "a deny-signature match must never dispatch to the handler, got: {scanner_text:?}"
    );

    // Phase 3: the SAME peer's next connection — even carrying a legit h1
    // payload — must now be dropped BEFORE the classifier ever inspects
    // it, since the peer is banned.
    let mut banned_conn = TcpStream::connect(bind).await.expect("banned connect");
    let banned_response = dial_and_collect(
        &mut banned_conn,
        b"GET / HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n",
    )
    .await;
    let banned_text = String::from_utf8_lossy(&banned_response);
    assert!(
        !banned_text.starts_with("HTTP/"),
        "a banned peer's next connection must be dropped pre-classify regardless of payload \
         (no HTTP response), got: {banned_text:?}"
    );

    server.stop();
}

/// Write `payload` then read until EOF or error, returning whatever bytes
/// were collected — tolerates the write/read erroring with a reset (the
/// server dropping the socket can race the kernel's buffering into an RST
/// instead of a clean FIN); either way, an empty/error-truncated buffer with
/// no HTTP status line proves the server never dispatched a response.
async fn dial_and_collect(conn: &mut TcpStream, payload: &[u8]) -> Vec<u8> {
    let mut collected = Vec::new();
    if conn.write_all(payload).await.is_ok() {
        let _ = conn.flush().await;
        let _ = conn.read_to_end(&mut collected).await;
    }
    collected
}
