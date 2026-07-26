//! `DatagramFactory` had exactly one implementation (prime) — a tokio-backed
//! `RuntimeSelection` carried `datagram_factory: None`, so a
//! `DatagramProtocolListenProtocol` was unreachable on tokio. Proves the
//! `TokioDatagramFactory` fix end to end, through the SAME `App`/`serve()`
//! surface [`listener_stream_and_datagram_coexist`] exercises, parameterized
//! over BOTH backends so the two stay symmetric (mirrors
//! `runtime_conformance.rs`'s `runtime_conformance!` macro shape).
//!
//! Each backend's `App` is built with an EXPLICIT `RuntimeSelection` (not
//! `App::new()`'s ambient default), so this is the one place in the suite
//! that proves a tokio-backed `App` can serve a datagram listener at all.

#![allow(clippy::unwrap_used, clippy::expect_used)]
#![cfg(feature = "tcp")]

use std::net::SocketAddr;
use std::time::Duration;

use conflaguration::Validate;
use proxima::listen::stream::{DatagramProtocol, DatagramProtocolListenProtocol};
use proxima::runtime::RuntimeSelection;
use proxima::time::Instant;
use proxima::{App, RunConfig};

/// Stateless echo, identical in shape to
/// `listener_stream_and_datagram_coexist::EchoDatagramProtocol` — kept as
/// its own copy rather than shared so this file stays self-contained and
/// off that (concurrently-changing) module.
struct EchoDatagramProtocol {
    pending_reply: Option<(Vec<u8>, SocketAddr)>,
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
        peer: SocketAddr,
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
    ) -> impl std::future::Future<Output = Result<Option<(usize, SocketAddr)>, Self::Err>> + Send {
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

/// real send + bounded-wait real recv over a fresh ephemeral UDP socket.
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

macro_rules! datagram_served_on {
    ($name:ident, $selection:expr) => {
        #[proxima::test]
        async fn $name() {
            let selection: RuntimeSelection = $selection;
            selection
                .validate()
                .expect("a from_prime/from_tokio selection must always validate");

            let app = App::builder()
                .runtime(selection)
                .build()
                .expect("build app with an explicit runtime selection");
            app.register_listen_protocol(std::sync::Arc::new(DatagramProtocolListenProtocol::new(
                concat!(stringify!($name), "-echo"),
                EchoDatagramProtocol::new,
            )))
            .expect("register echo datagram protocol");

            let bind = free_udp_addr();
            let server = app
                .serve(RunConfig {
                    bind,
                    protocol: concat!(stringify!($name), "-echo").into(),
                    spec: serde_json::Value::Null,
                })
                .await
                .expect("serve datagram protocol on the explicit runtime selection");

            let reply = ping_pong(bind);
            assert_eq!(
                reply.as_deref(),
                Some(&b"pong:ping"[..]),
                concat!(
                    "a datagram listener served on an explicit ",
                    stringify!($name),
                    " RuntimeSelection must answer a real UDP round trip"
                )
            );

            server.stop();
        }
    };
}

#[cfg(feature = "runtime-tokio")]
datagram_served_on!(
    served_on_tokio_answers_real_udp_round_trip,
    RuntimeSelection::tokio(1).expect("build tokio RuntimeSelection")
);

#[cfg(all(
    feature = "runtime-prime-executor",
    feature = "runtime-prime-inbox-alloc",
    feature = "runtime-prime-reactor",
    feature = "runtime-prime-bgpool",
    any(target_os = "linux", target_os = "macos")
))]
datagram_served_on!(
    served_on_prime_answers_real_udp_round_trip,
    RuntimeSelection::prime(1).expect("build prime RuntimeSelection")
);
