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
//! concurrent test run must not collide). The echo below is a fixed-size
//! read_exact/write_all round trip; `unix_split_copy_close_shape_round_trip`
//! (further down) covers the `split()` + `futures::io::copy` + early
//! write-half `close()` shape, which used to trip a reproducible `EINVAL`
//! out of `PrimeUnixListener::poll_accept` — root-caused and fixed (see
//! `unix_accept_after_peer_already_disconnected_does_not_einval` below and
//! `prime::os::net::finish_accepted_unix_socket`).

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

/// Regression test for the `PrimeUnixListener::poll_accept` `EINVAL`:
/// three clients connect, write, and drop (peer gone) *before* the
/// server's spawned task is ever polled to call `accept()` — the exact
/// scheduling that reproduced it. Root cause (proven via
/// `scratchpad/rawtest2` in the investigation, syscall-isolated): on
/// macOS, `socket2::Socket::accept()`'s convenience path additionally
/// applies `SO_NOSIGPIPE` to the freshly-accepted fd, and the kernel
/// rejects that `setsockopt` with `EINVAL` for an AF_UNIX peer that
/// already disconnected — discarding an otherwise perfectly good,
/// already-accepted connection. Fixed in `prime::os::net::UnixListener::
/// poll_accept` by using `accept_raw()` + a tolerant
/// `finish_accepted_unix_socket` helper. The echo shape here
/// (`split()`/`futures::io::copy()`/early write-half `.close()`, the
/// shape originally blamed) was proven NOT to matter — see the sibling
/// `unix_split_copy_close_shape_round_trip` below, and TCP's
/// `tcp_split_copy_close_shape_round_trip`, both of which exercise the
/// exact split/copy/close shape end to end with real data assertions.
#[proxima::test]
async fn unix_accept_after_peer_already_disconnected_does_not_einval() {
    let selection = RuntimeSelection::prime(1).expect("build prime runtime selection");
    let factory = selection
        .unix_upstream_factory
        .clone()
        .expect("prime bundles a UnixUpstreamFactory");

    let temp_dir = tempfile::tempdir().expect("tempdir");
    let socket_path = temp_dir.path().join("accept-after-peer-gone.sock");
    let listener = PrimeUnixListener::bind(socket_path.clone()).expect("bind unix listener");

    prime::os::core_shard::spawn_on_current_core(Box::pin(async move {
        for round in 0..3 {
            let mut conn = listener
                .accept()
                .await
                .unwrap_or_else(|err| panic!("accept failed on round {round}: {err}"));
            let mut buf = [0_u8; 22];
            let _ = conn.read(&mut buf).await;
        }
    }));

    for _round in 0..3 {
        let upstream = factory.connect(socket_path.clone());
        let mut conn = upstream.connect().await.expect("connect");
        conn.write_all(b"hello over prime unix")
            .await
            .expect("write");
        conn.flush().await.expect("flush");
        // peer disconnects before the server ever calls `accept()` — the
        // condition that reproduced the EINVAL.
        drop(conn);
    }
}

/// The originally-blamed shape, proven end to end with a real echoed
/// round trip: `conn.split()` + `futures::io::copy()` on the server, an
/// early write-half `.close()` (half-close, not full teardown) on both
/// sides. Requires two fixes proven necessary while building this test:
/// (1) the `EINVAL` fix above, and (2) `UnixStream::poll_close` using
/// `Shutdown::Write` instead of `Shutdown::Both` — `Shutdown::Both`
/// silently killed the still-live read half a `split()` caller needs
/// after signaling EOF (the same bug already fixed for
/// `TcpStream::poll_close`, undone-for-Unix regression).
#[proxima::test]
async fn unix_split_copy_close_shape_round_trip() {
    const MESSAGE: &[u8] = b"hello over prime unix split copy close";

    let selection = RuntimeSelection::prime(1).expect("build prime runtime selection");
    let factory = selection
        .unix_upstream_factory
        .clone()
        .expect("prime bundles a UnixUpstreamFactory");

    let temp_dir = tempfile::tempdir().expect("tempdir");
    let socket_path = temp_dir.path().join("split-copy-close-round-trip.sock");
    let listener = PrimeUnixListener::bind(socket_path.clone()).expect("bind unix listener");

    prime::os::core_shard::spawn_on_current_core(Box::pin(async move {
        let conn = listener.accept().await.expect("server accept");
        let (mut read_half, mut write_half) = conn.split();
        futures::io::copy(&mut read_half, &mut write_half)
            .await
            .expect("server echo copy");
        write_half.close().await.expect("server write-half close");
    }));

    let upstream = factory.connect(socket_path);
    let conn = upstream.connect().await.expect("client connect");
    let (mut read_half, mut write_half) = conn.split();
    write_half.write_all(MESSAGE).await.expect("client write");
    write_half.close().await.expect("client write-half close");

    let mut reply = Vec::new();
    read_half
        .read_to_end(&mut reply)
        .await
        .expect("client read echo to EOF");
    assert_eq!(reply, MESSAGE);
}

/// TCP sibling of `unix_split_copy_close_shape_round_trip` — proves the
/// split/copy/close contract holds for both transports (TCP never showed
/// the `EINVAL`; `TcpStream::poll_close` already used `Shutdown::Write`).
#[proxima::test]
async fn tcp_split_copy_close_shape_round_trip() {
    const MESSAGE: &[u8] = b"hello over prime tcp split copy close";

    let selection = RuntimeSelection::prime(1).expect("build prime runtime selection");
    let bind_addr = SocketAddr::from((Ipv4Addr::LOCALHOST, 0));
    let mut acceptor = selection
        .acceptor_factory
        .bind(
            bind_addr,
            proxima_primitives::stream::TcpBindOptions::default(),
        )
        .expect("bind tcp acceptor");
    let server_addr = acceptor.local_addr().expect("acceptor local_addr");

    prime::os::core_shard::spawn_on_current_core(Box::pin(async move {
        let conn = futures::future::poll_fn(|cx| acceptor.poll_accept(cx))
            .await
            .expect("server accept");
        let (mut read_half, mut write_half) = futures::io::AsyncReadExt::split(conn);
        futures::io::copy(&mut read_half, &mut write_half)
            .await
            .expect("server echo copy");
        write_half.close().await.expect("server write-half close");
    }));

    let upstream = proxima_net::prime::PrimeTcpUpstream::new(server_addr);
    let conn = upstream.connect().await.expect("client connect");
    let (mut read_half, mut write_half) = conn.split();
    write_half.write_all(MESSAGE).await.expect("client write");
    write_half.close().await.expect("client write-half close");

    let mut reply = Vec::new();
    read_half
        .read_to_end(&mut reply)
        .await
        .expect("client read echo to EOF");
    assert_eq!(reply, MESSAGE);
}
