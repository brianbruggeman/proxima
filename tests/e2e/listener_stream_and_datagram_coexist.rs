//! Regression coverage for a datagram [`ListenProtocol`] served alongside a
//! stream one on the SAME [`App`] — two independent `App::serve()` calls
//! sharing one `App`, each constructing its own `shutdown_tx`/`shutdown_rx`
//! oneshot pair inside `App::run_until_signal` (`src/app.rs`). Both
//! listeners must remain bound and answering traffic for the process
//! lifetime, regardless of registration/serve order — the shutdown wiring
//! is per-`serve()`-call, never shared across a sibling listener on the
//! same `App`.
//!
//! [`EchoDatagramProtocol`] proves liveness the way the acceptance bar
//! demands: a real UDP client sends a real datagram and must get a real
//! reply back, not merely "the task is still scheduled".

#![allow(clippy::unwrap_used, clippy::expect_used)]
#![cfg(feature = "tcp")]

use std::net::SocketAddr;
use std::net::SocketAddr as PeerAddr;
use std::sync::Arc;
use std::time::Duration;

use proxima::listen::stream::{DatagramProtocol, DatagramProtocolListenProtocol};
use proxima::time::Instant;
use proxima::{App, RunConfig};

/// Stateless echo: replies to every datagram with `b"pong:"` + the payload
/// it received. No timer (`next_deadline` -> `None`) — this test only needs
/// the recv arm of `DatagramProtocolListenProtocol`'s race to prove the
/// listener is genuinely alive, not merely bound.
struct EchoDatagramProtocol {
    pending_reply: Option<(Vec<u8>, PeerAddr)>,
}

impl EchoDatagramProtocol {
    fn new() -> Self {
        Self { pending_reply: None }
    }
}

impl DatagramProtocol for EchoDatagramProtocol {
    type Err = std::convert::Infallible;

    fn on_datagram(
        &mut self,
        _now: Instant,
        peer: PeerAddr,
        datagram: &[u8],
    ) -> impl std::future::Future<Output = Result<(), Self::Err>> + Send {
        let mut reply = b"pong:".to_vec();
        reply.extend_from_slice(datagram);
        self.pending_reply = Some((reply, peer));
        async move { Ok(()) }
    }

    fn on_timeout(
        &mut self,
        _now: Instant,
    ) -> impl std::future::Future<Output = Result<(), Self::Err>> + Send {
        async move { Ok(()) }
    }

    fn next_deadline(&self) -> Option<Instant> {
        None
    }

    fn transmit(
        &mut self,
        _now: Instant,
        buf: &mut [u8],
    ) -> impl std::future::Future<Output = Result<Option<(usize, PeerAddr)>, Self::Err>> + Send {
        let outcome = self.pending_reply.take().map(|(reply, peer)| {
            let len = reply.len().min(buf.len());
            buf[..len].copy_from_slice(&reply[..len]);
            (len, peer)
        });
        async move { Ok(outcome) }
    }
}

fn free_udp_addr() -> SocketAddr {
    let socket = std::net::UdpSocket::bind("127.0.0.1:0").expect("bind probe udp");
    let addr = socket.local_addr().expect("addr");
    drop(socket);
    addr
}

fn free_tcp_addr() -> SocketAddr {
    "127.0.0.1:0".parse().expect("addr")
}

/// Real send + bounded-wait real recv over a fresh ephemeral UDP socket —
/// the acceptance bar: "send a real datagram and get a response", not a
/// task-liveness proxy.
fn ping_pong(bind: SocketAddr) -> Option<Vec<u8>> {
    let socket = std::net::UdpSocket::bind("127.0.0.1:0").expect("client udp bind");
    socket
        .set_read_timeout(Some(Duration::from_secs(2)))
        .expect("set read timeout");
    socket.send_to(b"ping", bind).expect("client send_to");
    let mut buf = [0_u8; 64];
    match socket.recv(&mut buf) {
        Ok(len) => Some(buf[..len].to_vec()),
        Err(_) => None,
    }
}

fn build_app_with_raknet() -> App {
    let app = App::new().expect("App::new");
    app.register_listen_protocol(Arc::new(DatagramProtocolListenProtocol::new(
        "raknet-coexist",
        EchoDatagramProtocol::new,
    )))
    .expect("register raknet-coexist");
    app
}

/// Ordering 1: stream served first, then the sibling datagram listener.
/// This is the exact shape a Java-then-Bedrock Minecraft-style dual
/// listener takes on one `App`.
#[proxima::test]
async fn stream_then_datagram_both_answer_traffic() {
    let app = build_app_with_raknet();

    let stream_server = app
        .serve(RunConfig {
            bind: free_tcp_addr(),
            protocol: "stream".into(),
            spec: serde_json::Value::Null,
        })
        .await
        .expect("serve stream");

    let dgram_bind = free_udp_addr();
    let dgram_server = app
        .serve(RunConfig {
            bind: dgram_bind,
            protocol: "raknet-coexist".into(),
            spec: serde_json::Value::Null,
        })
        .await
        .expect("serve raknet-coexist");

    let reply = ping_pong(dgram_bind);
    assert_eq!(
        reply.as_deref(),
        Some(&b"pong:ping"[..]),
        "the datagram listener must still be receiving and replying after a \
         sibling stream listener was served on the same App"
    );

    stream_server.stop();
    dgram_server.stop();
}

/// Ordering 2: datagram served first, then the sibling stream listener —
/// the mirror of the above, proving the result is not order-dependent.
#[proxima::test]
async fn datagram_then_stream_both_answer_traffic() {
    let app = build_app_with_raknet();

    let dgram_bind = free_udp_addr();
    let dgram_server = app
        .serve(RunConfig {
            bind: dgram_bind,
            protocol: "raknet-coexist".into(),
            spec: serde_json::Value::Null,
        })
        .await
        .expect("serve raknet-coexist");

    let stream_server = app
        .serve(RunConfig {
            bind: free_tcp_addr(),
            protocol: "stream".into(),
            spec: serde_json::Value::Null,
        })
        .await
        .expect("serve stream");

    let reply = ping_pong(dgram_bind);
    assert_eq!(
        reply.as_deref(),
        Some(&b"pong:ping"[..]),
        "the datagram listener must still be receiving and replying when served \
         BEFORE a sibling stream listener on the same App"
    );

    stream_server.stop();
    dgram_server.stop();
}

/// A datagram-only `App` (no sibling stream listener at all) must serve
/// correctly on its own — isolates "second serve() breaks the first" from
/// "datagram serve is broken outright".
#[proxima::test]
async fn datagram_only_app_answers_traffic() {
    let app = build_app_with_raknet();

    let dgram_bind = free_udp_addr();
    let dgram_server = app
        .serve(RunConfig {
            bind: dgram_bind,
            protocol: "raknet-coexist".into(),
            spec: serde_json::Value::Null,
        })
        .await
        .expect("serve raknet-coexist");

    let reply = ping_pong(dgram_bind);
    assert_eq!(
        reply.as_deref(),
        Some(&b"pong:ping"[..]),
        "a datagram-only App (no sibling listener) must serve correctly"
    );

    dgram_server.stop();
}
