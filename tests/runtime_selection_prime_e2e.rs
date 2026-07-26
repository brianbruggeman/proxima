//! C2's payoff, proven end-to-end: `RuntimeSelection::prime()`'s bundled
//! `unix_upstream_factory`/`packet_listener_factory` do real unix-socket and
//! UDP I/O with no tokio anywhere in the graph (`unix`/`udp` name no runtime
//! of their own any more — see their Cargo.toml docs).
//!
//! Both tests run on `#[proxima::test]`'s own prime worker — the same
//! CURRENT_REACTOR context `PrimeUnixListener`/`PrimeUdpListener` require —
//! and mirror the accept/echo shape already proven in
//! `stream_passthrough.rs`'s own prime test (`spawn_on_current_core` for the
//! server half, direct `.await` for the client half). The unix socket lives
//! under a fresh `tempfile::tempdir()` (never a fixed path — a second
//! concurrent test run must not collide). The echo is a fixed-size
//! read_exact/write_all round trip rather than `split()` + `futures::io::copy`
//! — the latter, paired with an early write-half `close()`, was found to
//! trip a reproducible `EINVAL` out of `PrimeUnixListener::poll_accept` on
//! this host that is insensitive to call ordering and only to unrelated
//! code shape (a UB signature, not a logic race); root-causing that belongs
//! to a dedicated prime/socket2 investigation, not this test.

#![allow(clippy::unwrap_used, clippy::expect_used)]
#![cfg(all(feature = "unix", feature = "udp", feature = "serve-prime"))]

use std::net::{Ipv4Addr, SocketAddr};

use bytes::Bytes;
use futures::io::{AsyncReadExt, AsyncWriteExt};

use proxima::runtime::RuntimeSelection;
use proxima_net::packet::{Packet, PacketListenerExt};
use proxima_net::prime::PrimeUnixListener;
use proxima_primitives::stream::{StreamListenerExt, StreamUpstreamExt};

/// Real unix-domain-socket round trip through `RuntimeSelection::prime()`'s
/// bundled `unix_upstream_factory` — the client dials through the factory,
/// the server is a plain `PrimeUnixListener`, both driven on the same prime
/// worker (the accept+echo half via `spawn_on_current_core`).
#[proxima::test]
async fn unix_round_trip_via_prime_unix_upstream_factory() {
    let selection = RuntimeSelection::prime(1).expect("build prime runtime selection");
    let factory = selection
        .unix_upstream_factory
        .clone()
        .expect("prime bundles a UnixUpstreamFactory");

    let temp_dir = tempfile::tempdir().expect("tempdir");
    let socket_path = temp_dir.path().join("runtime-selection-prime.sock");
    let listener = PrimeUnixListener::bind(socket_path.clone()).expect("bind unix listener");

    prime::os::core_shard::spawn_on_current_core(Box::pin(async move {
        let mut conn = listener.accept().await.expect("accept");
        let mut buf = [0_u8; 21];
        conn.read_exact(&mut buf).await.expect("server read");
        conn.write_all(&buf).await.expect("server write");
    }));

    let upstream = factory.connect(socket_path);
    let mut conn = upstream.connect().await.expect("connect");

    conn.write_all(b"hello over prime unix")
        .await
        .expect("write");
    conn.flush().await.expect("flush");

    let mut reply = [0_u8; 21];
    conn.read_exact(&mut reply).await.expect("read echo");

    assert_eq!(&reply, b"hello over prime unix");
}

/// Real UDP round trip through `RuntimeSelection::prime()`'s bundled
/// `packet_listener_factory` — two independently-bound `PacketListener`s
/// exchange a datagram and an ack. `Packet::src` is the send-to target on
/// tx (received-from peer on rx); `Packet::dst` is the local/bind side —
/// see `proxima_net::packet::Packet`'s field docs.
#[proxima::test]
async fn udp_round_trip_via_prime_packet_listener_factory() {
    let selection = RuntimeSelection::prime(1).expect("build prime runtime selection");
    let factory = selection
        .packet_listener_factory
        .clone()
        .expect("prime bundles a PacketListenerFactory");

    let unspecified = SocketAddr::from((Ipv4Addr::LOCALHOST, 0));
    let server = factory.bind(unspecified).expect("bind server socket");
    let client = factory.bind(unspecified).expect("bind client socket");
    let server_addr = server.local_addr().expect("server local_addr");
    let client_addr = client.local_addr().expect("client local_addr");

    let request = Packet {
        src: server_addr,
        dst: client_addr,
        data: Bytes::from_static(b"hello over prime udp"),
    };
    client.send(&request).await.expect("send request");

    let mut server_buf = [0_u8; 128];
    let received = server.recv(&mut server_buf).await.expect("recv request");
    assert_eq!(&received.data[..], b"hello over prime udp");

    let reply = Packet {
        src: received.src,
        dst: server_addr,
        data: Bytes::from_static(b"ack"),
    };
    server.send(&reply).await.expect("send reply");

    let mut client_buf = [0_u8; 128];
    let ack = client.recv(&mut client_buf).await.expect("recv ack");
    assert_eq!(&ack.data[..], b"ack");
}
