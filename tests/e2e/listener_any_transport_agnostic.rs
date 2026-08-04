//! Headline proof for the transport-agnostic `.any()` open classifier: a
//! STREAM candidate (the built-in h1) and a DATAGRAM candidate (authored
//! right here, exactly like `listener_any_protocol_extension.rs`'s
//! `MiniProtocol` proves third-party extensibility) are registered on the
//! SAME `Listener::builder()`, bound to the SAME port NUMBER — the caller
//! never says "tcp" or "udp" anywhere in this file. A real TCP client and a
//! real UDP client both dial that one port and are classified and driven by
//! the correct candidate.
//!
//! [`LiteralUdpProtocol`] proves the datagram side reuses the IDENTICAL
//! `AnyProtocol`/`ProbeVerdict`/priority-arbitration machinery the stream
//! side already has — it is the exact same trait `MiniProtocol` implements,
//! the only difference is [`proxima::listen::any::AnyProtocol::wants_datagram`]
//! returning `true`. `multiple_datagram_candidates_are_priority_arbitrated_and_ambiguity_errors`
//! additionally proves N datagram candidates share ONE UDP socket under the
//! same priority-ordered-wait rule the classifier's own unit tests already
//! cover, and that two candidates matching at the same priority DROP the
//! datagram (an error, not a silent pick) rather than routing to either one.

#![allow(clippy::unwrap_used, clippy::expect_used)]
#![cfg(all(
    feature = "http1",
    feature = "serve-prime",
    feature = "runtime-prime-executor",
    feature = "runtime-prime-inbox-alloc",
    feature = "runtime-prime-reactor",
    feature = "runtime-prime-bgpool",
    any(target_os = "linux", target_os = "macos")
))]

use std::future::Future;
use std::net::{SocketAddr, TcpListener as StdTcpListener, TcpStream as StdTcpStream, UdpSocket};
use std::pin::Pin;
use std::time::Duration;

use bytes::Bytes;
use serde_json::Value;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

use proxima::SendPipe;
use proxima::error::ProximaError;
use proxima::listen::admission::ConnAdmission;
use proxima::pipe::into_handle;
use proxima::prelude::*;
use proxima::request::{Request, Response};
use proxima::stream::{PeerInfo, StreamConnection};

/// A datagram-wanting `AnyProtocol` candidate: matches a fixed literal
/// prefix on ONE already-received UDP message and replies with a fixed
/// payload — the datagram-side twin of `listener_any_protocol_extension.rs`'s
/// `MiniProtocol`. Parameterized so one type serves every candidate this
/// file registers instead of repeating the same probe/drive boilerplate
/// per literal.
struct LiteralUdpProtocol {
    name: &'static str,
    priority: u16,
    literal: &'static [u8],
    reply: &'static [u8],
}

impl AnyProtocol for LiteralUdpProtocol {
    fn name(&self) -> &str {
        self.name
    }

    fn priority(&self) -> u16 {
        self.priority
    }

    fn max_prefix_bytes(&self) -> usize {
        self.literal.len()
    }

    fn wants_datagram(&self) -> bool {
        true
    }

    fn probe(&self, prefix: &[u8]) -> ProbeVerdict {
        let compare_len = prefix.len().min(self.literal.len());
        if prefix[..compare_len] != self.literal[..compare_len] {
            return ProbeVerdict::No;
        }
        if prefix.len() < self.literal.len() {
            return ProbeVerdict::NeedMore {
                at_least: self.literal.len(),
            };
        }
        ProbeVerdict::Match {
            consumed: self.literal.len(),
        }
    }

    fn drive<'a>(
        &'a self,
        mut stream: Box<dyn StreamConnection>,
        _handler: proxima::listen::any::AnyHandler,
        _spec: &'a Value,
        _peer: Option<PeerInfo>,
        _admission: &'a ConnAdmission,
    ) -> Pin<Box<dyn Future<Output = Result<(), ProximaError>> + Send + 'a>> {
        Box::pin(async move {
            use futures::{AsyncReadExt as _, AsyncWriteExt as _};
            // Drains the one-shot datagram adapter's single already
            // -buffered message (EOF right after) — proves `drive` reads a
            // UDP-sourced connection through the exact same `AsyncRead`
            // contract a TCP-sourced one uses.
            let mut received = Vec::new();
            stream.read_to_end(&mut received).await?;
            stream.write_all(self.reply).await?;
            stream.close().await?;
            Ok(())
        })
    }
}

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

async fn dial_tcp_and_collect(conn: &mut TcpStream, payload: &[u8]) -> Vec<u8> {
    let mut collected = Vec::new();
    if conn.write_all(payload).await.is_ok() {
        let _ = conn.flush().await;
        let _ = conn.read_to_end(&mut collected).await;
    }
    collected
}

/// One send + one bounded-wait recv over a fresh ephemeral UDP socket —
/// `None` if nothing arrives before the timeout (the ambiguous-match case
/// below relies on this to prove a datagram was dropped, not answered).
fn dial_udp_and_collect(bind: SocketAddr, payload: &[u8], timeout: Duration) -> Option<Vec<u8>> {
    let socket = UdpSocket::bind("127.0.0.1:0").expect("client udp bind");
    socket
        .set_read_timeout(Some(timeout))
        .expect("set read timeout");
    socket.send_to(payload, bind).expect("client send_to");
    let mut buf = [0_u8; 2048];
    match socket.recv(&mut buf) {
        Ok(len) => Some(buf[..len].to_vec()),
        Err(_) => None,
    }
}

/// The headline proof: one stream candidate (h1, built in) and one datagram
/// candidate (`LiteralUdpProtocol`, authored in this test) share ONE port
/// number under `.any()`. Neither `.tcp()` nor `.udp()` is ever called —
/// `.any()` alone decides which sockets to bind, from what the registered
/// candidates need.
#[proxima::test(flavor = "multi_thread", worker_threads = 2)]
async fn stream_and_datagram_candidates_share_one_port_under_dot_any() {
    let bind = free_loopback_addr();

    let server = Listener::builder()
        .bind(bind)
        .handle(into_handle(LegitOk))
        .any()
        .protocol(LiteralUdpProtocol {
            name: "udpx",
            priority: 100,
            literal: b"UDPX/1\r\n",
            reply: b"UDPX/1 200 OK\r\nhello-from-datagram-candidate",
        })
        .serve()
        .await
        .expect("listener builder serves both transports on one port");

    wait_until_listening(bind);

    // TCP: the built-in h1 candidate still classifies and drives normally.
    let mut tcp_conn = TcpStream::connect(bind).await.expect("tcp connect");
    let tcp_response = dial_tcp_and_collect(
        &mut tcp_conn,
        b"GET / HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n",
    )
    .await;
    let tcp_text = String::from_utf8_lossy(&tcp_response);
    assert!(
        tcp_text.starts_with("HTTP/1.1 200"),
        "stream traffic on the shared port must still route to h1; got: {tcp_text:?}"
    );
    assert!(tcp_text.contains("legit-ok"), "got: {tcp_text:?}");

    // UDP: the SAME port number, a datagram this time, classified and
    // driven by `LiteralUdpProtocol` — the fan-in proof.
    let udp_response = dial_udp_and_collect(bind, b"UDPX/1\r\nping", Duration::from_secs(2))
        .expect("datagram candidate must reply");
    let udp_text = String::from_utf8_lossy(&udp_response);
    assert!(
        udp_text.starts_with("UDPX/1 200 OK"),
        "the datagram-registered candidate must drive its own reply; got: {udp_text:?}"
    );
    assert!(
        udp_text.contains("hello-from-datagram-candidate"),
        "got: {udp_text:?}"
    );

    server.stop();
}

/// N datagram candidates sharing ONE UDP socket, arbitrated by the exact
/// same priority rule `proxima_listen::any::Classifier`'s own unit tests
/// prove in isolation — this test proves it end-to-end, over a real socket,
/// through `.any()`'s fan-in. `high`/`low` have disjoint literals at
/// different priorities (routes independently regardless of priority
/// order); `tied_a`/`tied_b` share both a literal AND a priority, so a
/// datagram matching it resolves to `ClassifyOutcome::AmbiguousMatch` —
/// dropped, not routed to either candidate (see
/// `classify_and_drive_plaintext`'s explicit `AmbiguousMatch` arm).
#[proxima::test(flavor = "multi_thread", worker_threads = 2)]
async fn multiple_datagram_candidates_are_priority_arbitrated_and_ambiguity_errors() {
    let bind = free_loopback_addr();

    let server = Listener::builder()
        .bind(bind)
        .handle(into_handle(LegitOk))
        .any()
        .protocol(LiteralUdpProtocol {
            name: "hipri",
            priority: 200,
            literal: b"HIPRI/1\r\n",
            reply: b"HIPRI-WINS",
        })
        .protocol(LiteralUdpProtocol {
            name: "lopri",
            priority: 100,
            literal: b"LOPRI/1\r\n",
            reply: b"LOPRI-WINS",
        })
        .protocol(LiteralUdpProtocol {
            name: "tied-a",
            priority: 150,
            literal: b"AMBIG/1\r\n",
            reply: b"TIED-A-WINS",
        })
        .protocol(LiteralUdpProtocol {
            name: "tied-b",
            priority: 150,
            literal: b"AMBIG/1\r\n",
            reply: b"TIED-B-WINS",
        })
        .serve()
        .await
        .expect("listener builder serves 4 datagram candidates on one socket");

    wait_until_listening(bind);

    let hipri = dial_udp_and_collect(bind, b"HIPRI/1\r\nx", Duration::from_secs(2))
        .expect("the high-priority candidate must reply to its own literal");
    assert_eq!(hipri, b"HIPRI-WINS");

    let lopri = dial_udp_and_collect(bind, b"LOPRI/1\r\nx", Duration::from_secs(2))
        .expect("the low-priority candidate must still reply to its own, disjoint literal");
    assert_eq!(lopri, b"LOPRI-WINS");

    // Two SAME-priority candidates both match this literal — the classifier
    // reports `AmbiguousMatch` rather than silently picking one, and the
    // listener drops the connection: no reply from either `tied-a` or
    // `tied-b` within the timeout.
    let ambiguous = dial_udp_and_collect(bind, b"AMBIG/1\r\nx", Duration::from_millis(500));
    assert!(
        ambiguous.is_none(),
        "an ambiguous match at the same priority must be dropped, not routed to either \
         tied candidate; got: {ambiguous:?}"
    );

    server.stop();
}
