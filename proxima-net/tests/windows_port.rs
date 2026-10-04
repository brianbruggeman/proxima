#![cfg(all(
    feature = "prime",
    feature = "runtime-prime-inbox-alloc",
    any(target_os = "macos", target_os = "linux", windows),
))]

use std::future::poll_fn;
use std::io::{Read, Write};
use std::net::{
    Ipv4Addr, SocketAddr, TcpListener as StdTcpListener, TcpStream as StdTcpStream,
    UdpSocket as StdUdpSocket,
};
use std::sync::{Arc, mpsc};
use std::task::Poll;
use std::thread;
use std::time::Duration;

use bytes::Bytes;
use futures::channel::oneshot;
use futures::io::{AsyncReadExt, AsyncWriteExt};
use prime::os::core_shard::{self, CoreShardHandle};
use proxima_net::packet::{Packet, PacketListenerExt, PacketListenerFactory};
use proxima_net::prime::{
    PrimeAcceptorFactory, PrimeDatagramFactory, PrimePacketListenerFactory, PrimeTcpUpstream,
};
use proxima_primitives::stream::{
    AcceptorFactory, DatagramFactory, PeerInfo, StreamConnection, StreamUpstream, TcpBindOptions,
};
use proxima_runtime::CoreId;

const GUARD: Duration = Duration::from_secs(10);
const REQUEST: &[u8] = b"GET /windows-port HTTP/1.1\r\nHost: localhost\r\n\r\n";
const RESPONSE: &[u8] = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok";
const DATAGRAM: &[u8] = b"proxima windows udp probe";
const BATCH_FIRST: &[u8] = b"proxima batch first";
const BATCH_SECOND: &[u8] = b"proxima batch second";

fn loopback() -> SocketAddr {
    SocketAddr::from((Ipv4Addr::LOCALHOST, 0))
}

fn worker() -> CoreShardHandle {
    core_shard::launch_with_lanes(CoreId(0), None, 4, 16).expect("launch prime worker")
}

fn configure_tcp_peer(peer: &StdTcpStream) {
    peer.set_read_timeout(Some(GUARD))
        .expect("set peer read guard");
    peer.set_write_timeout(Some(GUARD))
        .expect("set peer write guard");
}

#[test]
fn windows_port_acceptor_factory_payload() {
    let handle = worker();
    let (address_tx, address_rx) = mpsc::channel();
    let (result_tx, result_rx) = mpsc::channel();
    handle
        .dispatch_send_inline(async move {
            let mut acceptor = PrimeAcceptorFactory
                .bind(loopback(), TcpBindOptions::default())
                .expect("bind prime acceptor");
            address_tx
                .send(acceptor.local_addr().expect("acceptor address"))
                .expect("report acceptor address");
            let mut connection = poll_fn(|context| acceptor.poll_accept(context))
                .await
                .expect("accept std peer");
            let mut request = vec![0; REQUEST.len()];
            connection
                .read_exact(&mut request)
                .await
                .expect("read request");
            connection
                .write_all(RESPONSE)
                .await
                .expect("write response");
            let observed_peer = match connection.peer().expect("connection peer") {
                PeerInfo::Tcp(peer) => peer,
                other => panic!("expected TCP peer, got {other:?}"),
            };
            result_tx
                .send((request, observed_peer))
                .expect("report accept result");
        })
        .expect("dispatch acceptor task");
    let address = address_rx
        .recv_timeout(GUARD)
        .expect("receive acceptor address");
    let mut peer =
        StdTcpStream::connect_timeout(&address, GUARD).expect("connect independent std peer");
    configure_tcp_peer(&peer);
    let peer_address = peer.local_addr().expect("read std peer address");
    peer.write_all(REQUEST).expect("send request");
    let mut response = vec![0; RESPONSE.len()];
    peer.read_exact(&mut response).expect("read response");
    assert_eq!(response, RESPONSE);
    let (request, observed_peer) = result_rx
        .recv_timeout(GUARD)
        .expect("receive accept result");
    assert_eq!(request, REQUEST);
    assert_eq!(observed_peer, peer_address);
    handle.shutdown_and_join().expect("join prime worker");
}

#[test]
fn windows_port_tcp_upstream_payload() {
    let listener = StdTcpListener::bind(loopback()).expect("bind independent std listener");
    listener
        .set_nonblocking(false)
        .expect("configure std listener");
    let address = listener.local_addr().expect("read listener address");
    let handle = worker();
    let (result_tx, result_rx) = mpsc::channel();
    let (peer_tx, peer_rx) = mpsc::channel();
    let accept_thread = thread::spawn(move || {
        peer_tx
            .send(listener.accept())
            .expect("report std accept result");
    });
    handle
        .dispatch_send_inline(async move {
            let upstream = PrimeTcpUpstream::new(address);
            let mut connection = upstream.connect().await.expect("prime upstream connects");
            connection.write_all(REQUEST).await.expect("send request");
            let mut response = vec![0; RESPONSE.len()];
            connection
                .read_exact(&mut response)
                .await
                .expect("read response");
            let observed_peer = match connection.peer().expect("upstream peer") {
                PeerInfo::Tcp(peer) => peer,
                other => panic!("expected TCP peer, got {other:?}"),
            };
            result_tx
                .send((response, observed_peer))
                .expect("report upstream result");
        })
        .expect("dispatch upstream task");
    let (mut peer, _) = peer_rx
        .recv_timeout(GUARD)
        .expect("wait for independent std accept")
        .expect("accept prime upstream");
    accept_thread.join().expect("join std accept thread");
    configure_tcp_peer(&peer);
    let mut request = vec![0; REQUEST.len()];
    peer.read_exact(&mut request).expect("read request");
    assert_eq!(request, REQUEST);
    peer.write_all(RESPONSE).expect("write response");
    let (response, observed_peer) = result_rx
        .recv_timeout(GUARD)
        .expect("receive upstream result");
    assert_eq!(response, RESPONSE);
    assert_eq!(observed_peer, address);
    handle.shutdown_and_join().expect("join prime worker");
    hostname_upstream_payload();
    concurrent_upstream_calls(false, false);
    concurrent_upstream_calls(true, false);
    concurrent_upstream_calls(false, true);
    concurrent_upstream_calls(true, true);
}

#[test]
fn windows_port_datagram_factory_payload() {
    let handle = worker();
    let (address_tx, address_rx) = mpsc::channel();
    let (result_tx, result_rx) = mpsc::channel();
    handle
        .dispatch_send_inline(async move {
            let mut datagram = PrimeDatagramFactory
                .bind(loopback())
                .expect("bind prime datagram");
            let local_address = datagram.local_addr().expect("read prime datagram address");
            address_tx
                .send(local_address)
                .expect("report datagram address");
            let mut buffer = [0; 128];
            let (count, source) = poll_fn(|context| datagram.poll_recv_from(context, &mut buffer))
                .await
                .expect("receive datagram");
            assert_eq!(&buffer[..count], DATAGRAM);
            assert_eq!(count, DATAGRAM.len());
            let sent = poll_fn(|context| datagram.poll_send_to(context, DATAGRAM, source))
                .await
                .expect("send datagram response");
            assert_eq!(sent, DATAGRAM.len());

            let mut first_buffer = [0; 64];
            let mut second_buffer = [0; 64];
            let mut buffers: [&mut [u8]; 2] = [&mut first_buffer, &mut second_buffer];
            let mut metadata = [(0, source); 2];
            let mut received_count = 0;
            while received_count < 2 {
                let amount = poll_fn(|context| {
                    datagram.poll_recv_batch(
                        context,
                        &mut buffers[received_count..],
                        &mut metadata[received_count..],
                    )
                })
                .await
                .expect("receive datagram batch");
                assert!(amount > 0, "datagram receive batch must make progress");
                received_count += amount;
            }
            let received = vec![
                (buffers[0][..metadata[0].0].to_vec(), metadata[0].1),
                (buffers[1][..metadata[1].0].to_vec(), metadata[1].1),
            ];

            let packets = [(BATCH_FIRST, source), (BATCH_SECOND, source)];
            let mut sent_count = 0;
            while sent_count < packets.len() {
                let amount =
                    poll_fn(|context| datagram.poll_send_batch(context, &packets[sent_count..]))
                        .await
                        .expect("send datagram batch");
                assert!(amount > 0, "datagram send batch must make progress");
                sent_count += amount;
            }
            result_tx
                .send((
                    local_address,
                    source,
                    count,
                    received_count,
                    received,
                    sent_count,
                ))
                .expect("report datagram results");
        })
        .expect("dispatch datagram task");
    let address = address_rx
        .recv_timeout(GUARD)
        .expect("receive datagram address");
    let peer = StdUdpSocket::bind(loopback()).expect("bind independent std UDP peer");
    peer.set_read_timeout(Some(GUARD))
        .expect("set UDP peer guard");
    let peer_address = peer.local_addr().expect("read std UDP peer address");
    assert_eq!(
        peer.send_to(DATAGRAM, address).expect("send datagram"),
        DATAGRAM.len()
    );
    let mut response = [0; 128];
    let (response_count, response_source) = peer
        .recv_from(&mut response)
        .expect("receive datagram response");
    assert_eq!(&response[..response_count], DATAGRAM);
    assert_eq!(response_source, address);

    peer.send_to(BATCH_FIRST, address)
        .expect("send first batch datagram");
    peer.send_to(BATCH_SECOND, address)
        .expect("send second batch datagram");
    let (first_count, first_source) = peer
        .recv_from(&mut response)
        .expect("receive first batch response");
    assert_eq!(&response[..first_count], BATCH_FIRST);
    assert_eq!(first_source, address);
    let (second_count, second_source) = peer
        .recv_from(&mut response)
        .expect("receive second batch response");
    assert_eq!(&response[..second_count], BATCH_SECOND);
    assert_eq!(second_source, address);
    let (local_address, source, count, received_count, received, sent_count) = result_rx
        .recv_timeout(GUARD)
        .expect("receive datagram results");
    assert_eq!(local_address, address);
    assert_eq!(source, peer_address);
    assert_eq!(count, DATAGRAM.len());
    assert_eq!(received_count, 2);
    assert_eq!(
        received,
        vec![
            (BATCH_FIRST.to_vec(), peer_address),
            (BATCH_SECOND.to_vec(), peer_address)
        ]
    );
    assert_eq!(sent_count, 2);

    let (truncated_tx, truncated_rx) = mpsc::channel();
    let (truncation_address_tx, truncation_address_rx) = mpsc::channel();
    handle
        .dispatch_send_inline(async move {
            let mut datagram = PrimeDatagramFactory
                .bind(loopback())
                .expect("bind truncation datagram");
            truncation_address_tx
                .send(datagram.local_addr().expect("truncation address"))
                .expect("report truncation address");
            let mut prefix = [0; 7];
            let (length, source) = poll_fn(|context| datagram.poll_recv_from(context, &mut prefix))
                .await
                .expect("receive truncated datagram");
            truncated_tx
                .send((prefix, length, source))
                .expect("report truncated datagram");
        })
        .expect("dispatch truncation task");
    let truncation_address = truncation_address_rx
        .recv_timeout(GUARD)
        .expect("receive truncation address");
    peer.send_to(DATAGRAM, truncation_address)
        .expect("send truncation datagram");
    let (prefix, length, source) = truncated_rx
        .recv_timeout(GUARD)
        .expect("receive truncation result");
    assert_eq!(length, 7);
    assert_eq!(&prefix, &DATAGRAM[..7]);
    assert_eq!(source, peer_address);
    handle.shutdown_and_join().expect("join prime worker");
}

#[test]
fn windows_port_packet_listener_factory_payload() {
    let handle = worker();
    let (address_tx, address_rx) = mpsc::channel();
    let (result_tx, result_rx) = mpsc::channel();
    handle
        .dispatch_send_inline(async move {
            let listener = PrimePacketListenerFactory
                .bind(loopback())
                .expect("bind prime packet listener");
            let local_address = listener.local_addr().expect("read packet listener address");
            address_tx
                .send(local_address)
                .expect("report packet listener address");
            let mut buffer = [0; 128];
            let received = listener.recv(&mut buffer).await.expect("receive packet");
            assert_eq!(received.data, Bytes::from_static(DATAGRAM));
            assert_eq!(received.dst, local_address);
            let response_packet = Packet {
                src: received.src,
                dst: received.dst,
                data: received.data.clone(),
            };
            listener
                .send(&response_packet)
                .await
                .expect("reply to packet source");
            result_tx
                .send((received.src, received.dst, received.data.to_vec()))
                .expect("report packet result");
        })
        .expect("dispatch packet task");
    let address = address_rx
        .recv_timeout(GUARD)
        .expect("receive packet listener address");
    let peer = StdUdpSocket::bind(loopback()).expect("bind independent std UDP peer");
    peer.set_read_timeout(Some(GUARD))
        .expect("set UDP peer guard");
    let peer_address = peer.local_addr().expect("read UDP peer address");
    assert_eq!(
        peer.send_to(DATAGRAM, address).expect("send packet probe"),
        DATAGRAM.len()
    );
    let mut response = [0; 128];
    let (count, source) = peer.recv_from(&mut response).expect("receive packet reply");
    assert_eq!(&response[..count], DATAGRAM);
    assert_eq!(source, address);
    let (observed_source, observed_destination, observed_data) = result_rx
        .recv_timeout(GUARD)
        .expect("receive packet result");
    assert_eq!(observed_source, peer_address);
    assert_eq!(observed_destination, address);
    assert_eq!(observed_data, DATAGRAM);
    handle.shutdown_and_join().expect("join prime worker");
}

fn hostname_upstream_payload() {
    let listener = StdTcpListener::bind(loopback()).expect("bind hostname peer");
    let address = listener.local_addr().expect("hostname address");
    let handle = worker();
    let (completed, result) = mpsc::channel();
    handle
        .dispatch_send_inline(async move {
            let upstream = PrimeTcpUpstream::with_host("localhost", address.port());
            let mut connection = upstream
                .connect_future()
                .await
                .expect("resolve hostname and connect");
            connection
                .write_all(REQUEST)
                .await
                .expect("write hostname request");
            let mut bytes = vec![0; RESPONSE.len()];
            connection
                .read_exact(&mut bytes)
                .await
                .expect("read hostname response");
            completed
                .send((bytes, connection.peer()))
                .expect("report hostname response");
        })
        .expect("dispatch hostname dial");
    let (mut peer, _) = listener.accept().expect("accept hostname dial");
    configure_tcp_peer(&peer);
    let mut request = vec![0; REQUEST.len()];
    peer.read_exact(&mut request).expect("hostname request");
    assert_eq!(request, REQUEST);
    peer.write_all(RESPONSE).expect("hostname response");
    let (bytes, peer) = result.recv_timeout(GUARD).expect("hostname completed");
    assert_eq!(bytes, RESPONSE);
    assert!(matches!(peer, Some(PeerInfo::Tcp(peer)) if peer == address));
    handle.shutdown_and_join().expect("join hostname worker");
}

fn concurrent_upstream_calls(cross_worker: bool, cancel_first: bool) {
    let first = worker();
    let second = worker();
    let listener = StdTcpListener::bind(loopback()).expect("bind concurrent upstream peer");
    let address = listener.local_addr().expect("concurrent upstream address");
    let upstream = PrimeTcpUpstream::boxed(address);
    let (announced, initial_polls) = mpsc::channel();
    let (completed, outcomes) = mpsc::channel();
    let mut releases = Vec::new();
    for (index, payload) in [b"left".as_slice(), b"rght".as_slice()]
        .into_iter()
        .enumerate()
    {
        let upstream = Arc::clone(&upstream);
        let announced = announced.clone();
        let completed = completed.clone();
        let (release, released) = oneshot::channel();
        releases.push(release);
        let selected = if cross_worker && index == 1 {
            &second
        } else {
            &first
        };
        selected
            .dispatch_send_inline(async move {
                let mut operation = upstream.connect_future();
                let initial =
                    poll_fn(|context| Poll::Ready(operation.as_mut().poll(context))).await;
                announced
                    .send((index, initial.is_pending()))
                    .expect("report initial independent dial poll");
                released.await.expect("release concurrent dial");
                if cancel_first && index == 0 {
                    drop(initial);
                    drop(operation);
                    completed.send(None).expect("report cancellation");
                    return;
                }
                let mut connection = match initial {
                    Poll::Ready(result) => result,
                    Poll::Pending => operation.await,
                }
                .expect("independent concurrent dial");
                connection
                    .write_all(payload)
                    .await
                    .expect("write distinct request");
                let mut echo = [0; 4];
                connection
                    .read_exact(&mut echo)
                    .await
                    .expect("read distinct response");
                assert_eq!(echo.as_slice(), payload);
                completed
                    .send(Some(echo))
                    .expect("report concurrent response");
            })
            .expect("dispatch independent call");
    }
    let mut pending = 0;
    for _ in 0..2 {
        let (index, is_pending) = initial_polls
            .recv_timeout(GUARD)
            .expect("initial dial polled");
        pending += usize::from(is_pending);
        if cancel_first && index == 0 {
            assert!(is_pending, "cancel an actual pending connection call");
        }
    }
    assert!(pending > 0, "exercise overlapping pending connection state");
    let mut peers = Vec::new();
    for _ in 0..2 {
        let (peer, _) = listener
            .accept()
            .expect("accept both independent connects before release");
        configure_tcp_peer(&peer);
        peers.push(peer);
    }
    for release in releases {
        release.send(()).expect("release connection call");
    }
    let mut received = Vec::new();
    let mut closed = 0;
    for mut peer in peers {
        let mut payload = [0; 4];
        let count = peer
            .read(&mut payload)
            .expect("observe independent request or cancelled EOF");
        if count == 0 {
            closed += 1;
        } else {
            peer.read_exact(&mut payload[count..])
                .expect("complete distinct request");
            peer.write_all(&payload).expect("return distinct payload");
            received.push(payload);
        }
    }
    let mut completed_payloads = Vec::new();
    for _ in 0..2 {
        if let Some(payload) = outcomes.recv_timeout(GUARD).expect("each call terminates") {
            completed_payloads.push(payload);
        }
    }
    let mut expected = if cancel_first {
        vec![*b"rght"]
    } else {
        vec![*b"left", *b"rght"]
    };
    expected.sort();
    received.sort();
    completed_payloads.sort();
    assert_eq!(received, expected);
    assert_eq!(completed_payloads, expected);
    assert_eq!(closed, usize::from(cancel_first));
    first
        .shutdown_and_join()
        .expect("join first concurrent worker");
    second
        .shutdown_and_join()
        .expect("join second concurrent worker");
}
