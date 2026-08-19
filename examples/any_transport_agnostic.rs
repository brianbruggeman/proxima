#![allow(clippy::unwrap_used, clippy::expect_used)]

//! `.any()` classifying over TWO transports on ONE port — the caller never
//! says "tcp" or "udp" anywhere in this file. `AnyProtocol::probe`/`drive`
//! (`proxima-listen/src/any/probe.rs`) are unchanged from what
//! `extend_protocol.rs` already teaches: pure, sans-IO `&[u8] ->
//! ProbeVerdict`, then one `drive` over a `Box<dyn StreamConnection>`. The
//! ONE new thing is `AnyProtocol::wants_datagram() -> bool` — a candidate
//! that overrides it to `true` gets fed by a UDP datagram instead of a TCP
//! byte stream, through the IDENTICAL classify+drive contract.
//!
//! Two things this file proves:
//! 1. A stream candidate (the built-in h1) and a datagram candidate
//!    (`LiteralUdpProtocol`, authored right here) share ONE bind under
//!    `.any()` — a real TCP client and a real UDP client both dial the SAME
//!    port and are classified and driven by the correct candidate.
//! 2. N datagram candidates are priority-arbitrated by the exact same rule
//!    the classifier already applies to stream candidates — and two
//!    candidates tied at the same priority DROP the datagram rather than
//!    routing it to either one (`ClassifyOutcome::AmbiguousMatch` is never a
//!    silent pick).
//!
//! Grounded in `tests/e2e/listener_any_transport_agnostic.rs` (the same two
//! scenarios, proven end to end as `#[proxima::test]`s) — this file is the
//! `cargo run`-able version.
//!
//! Run: `cargo run --example any_transport_agnostic --features http1-native`

use std::future::Future;
use std::io::{Read, Write};
use std::net::{Ipv4Addr, SocketAddr, TcpStream as StdTcpStream, UdpSocket};
use std::pin::Pin;
use std::time::Duration;

use bytes::Bytes;
use serde_json::Value;

use proxima::listen::admission::ConnAdmission;
use proxima::pipe::into_handle;
use proxima::prelude::*;
use proxima::request::{Request, Response};
use proxima::stream::{PeerInfo, StreamConnection};
use proxima::{ProximaError, SendPipe};

/// A datagram-wanting `AnyProtocol` candidate: matches a fixed literal
/// prefix on ONE already-received UDP message and replies with a fixed
/// payload — the datagram-side twin of `extend_protocol.rs`'s
/// `PingPongProtocol`. Parameterized so one type serves every candidate
/// this file registers instead of repeating the same probe/drive
/// boilerplate per literal.
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

    // The ONE new method: everything else on this impl is the identical
    // probe/drive shape `extend_protocol.rs`'s stream-only `PingPongProtocol`
    // already uses. Defaults to `false` (a stream-only candidate never sees
    // this method exist); overriding it to `true` is the entire opt-in.
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
            // -buffered message (EOF right after) — a UDP-sourced
            // connection is read through the exact same `AsyncRead`
            // contract a TCP-sourced one uses; there is no second API.
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

    async fn call(&self, _request: Request<Bytes>) -> Result<Response<Bytes>, ProximaError> {
        Ok(Response::ok("legit-ok"))
    }
}

fn free_loopback_addr() -> Result<SocketAddr, ProximaError> {
    let probe = std::net::TcpListener::bind((Ipv4Addr::LOCALHOST, 0))?;
    let addr = probe.local_addr()?;
    drop(probe);
    Ok(addr)
}

fn h1_round_trip(addr: SocketAddr) -> Result<String, ProximaError> {
    let mut stream = StdTcpStream::connect(addr)?;
    stream.write_all(b"GET / HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")?;
    let mut response = Vec::new();
    stream.read_to_end(&mut response)?;
    Ok(String::from_utf8_lossy(&response).into_owned())
}

/// One send + one bounded-wait recv over a fresh ephemeral UDP socket —
/// `None` if nothing arrives before the timeout (the ambiguous-match
/// scenario relies on this to prove a datagram was dropped, not answered).
fn udp_round_trip(
    bind: SocketAddr,
    payload: &[u8],
    timeout: Duration,
) -> Result<Option<Vec<u8>>, ProximaError> {
    let socket = UdpSocket::bind("127.0.0.1:0")?;
    socket.set_read_timeout(Some(timeout))?;
    socket.send_to(payload, bind)?;
    let mut buf = [0_u8; 2048];
    match socket.recv(&mut buf) {
        Ok(len) => Ok(Some(buf[..len].to_vec())),
        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => Ok(None),
        Err(error) => Err(ProximaError::Io(error)),
    }
}

/// Scenario 1: one stream candidate (h1, built in) and one datagram
/// candidate (`LiteralUdpProtocol`) share ONE port under `.any()`. Neither
/// `.tcp()` nor `.udp()` is ever called — `.any()` alone decides which
/// sockets to bind, from what the registered candidates need.
async fn stream_and_datagram_share_one_port() -> Result<(), ProximaError> {
    let bind = free_loopback_addr()?;

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
        .await?;

    // TCP: the built-in h1 candidate still classifies and drives normally.
    let h1_text = h1_round_trip(bind)?;
    assert!(
        h1_text.starts_with("HTTP/1.1 200"),
        "stream traffic on the shared port must still route to h1; got: {h1_text:?}"
    );
    assert!(h1_text.contains("legit-ok"), "got: {h1_text:?}");
    println!(".any() on {bind}: a TCP client still gets routed to h1 -> HTTP/1.1 200");

    // UDP: the SAME port number, a datagram this time, classified and
    // driven by `LiteralUdpProtocol` — the fan-in proof.
    let udp_response = udp_round_trip(bind, b"UDPX/1\r\nping", Duration::from_secs(2))?
        .expect("datagram candidate must reply");
    let udp_text = String::from_utf8_lossy(&udp_response);
    assert!(
        udp_text.starts_with("UDPX/1 200 OK"),
        "the datagram-registered candidate must drive its own reply; got: {udp_text:?}"
    );
    println!(
        ".any() on the SAME {bind}: a UDP datagram is classified and driven by LiteralUdpProtocol -> {udp_text:?}"
    );

    server.stop();
    Ok(())
}

/// Scenario 2: N datagram candidates sharing ONE UDP socket, arbitrated by
/// the exact same priority rule `proxima_listen::any::Classifier` already
/// applies to stream candidates. `hipri`/`lopri` have disjoint literals at
/// different priorities (routes independently regardless of priority
/// order); `tied_a`/`tied_b` share both a literal AND a priority, so a
/// datagram matching it resolves to `ClassifyOutcome::AmbiguousMatch` —
/// dropped, not routed to either candidate.
async fn datagram_candidates_are_priority_arbitrated() -> Result<(), ProximaError> {
    let bind = free_loopback_addr()?;

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
        .await?;

    let hipri = udp_round_trip(bind, b"HIPRI/1\r\nx", Duration::from_secs(2))?
        .expect("the high-priority candidate must reply to its own literal");
    assert_eq!(hipri, b"HIPRI-WINS");

    let lopri = udp_round_trip(bind, b"LOPRI/1\r\nx", Duration::from_secs(2))?
        .expect("the low-priority candidate must still reply to its own, disjoint literal");
    assert_eq!(lopri, b"LOPRI-WINS");
    println!(
        "4 datagram candidates, one socket, on {bind}: hipri/lopri each answer their own \
         disjoint literal regardless of priority order"
    );

    // Two SAME-priority candidates both match this literal — the classifier
    // reports `AmbiguousMatch` rather than silently picking one, and the
    // listener drops the connection: no reply from either `tied-a` or
    // `tied-b` within the timeout.
    let ambiguous = udp_round_trip(bind, b"AMBIG/1\r\nx", Duration::from_millis(500))?;
    assert!(
        ambiguous.is_none(),
        "an ambiguous match at the same priority must be dropped, not routed to either tied \
         candidate; got: {ambiguous:?}"
    );
    println!(
        "tied-a/tied-b share both a literal AND a priority: the datagram is DROPPED (no reply \
         within 500ms), never routed to either one"
    );

    server.stop();
    Ok(())
}

#[proxima::main]
async fn main() -> Result<(), ProximaError> {
    stream_and_datagram_share_one_port().await?;
    println!();
    datagram_candidates_are_priority_arbitrated().await?;
    println!("\nany_transport_agnostic: .any() classifies over TCP and UDP through one classifier");
    Ok(())
}
