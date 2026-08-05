//! prime-backed UDP `PacketListener` — the prime sibling of
//! `proxima-net-tokio`'s `TokioUdpListener` (`tokio_packet.rs`). Mirrors it
//! field-for-field (P14): wraps one `prime::os::net::UdpSocket` that both
//! recv and send go through, same as the tokio version wraps one
//! `tokio::net::UdpSocket`.
//!
//! This is DISTINCT from [`super::PrimeDatagramFactory`]/[`super::PrimeDatagram`]
//! even though both ultimately wrap the same `prime::os::net::UdpSocket`
//! primitive: `PrimeDatagramFactory` implements the lower-level
//! `DatagramFactory`/`DatagramSocket` seam (`proxima_primitives::stream`)
//! that `ServeContext::datagram_factory` injects generically — already
//! consumed by both the h3-native QUIC listener AND `proxima-dns`'s
//! non-QUIC `DatagramProtocolListenProtocol` (see
//! `proxima-dns/src/datagram_protocol.rs`), so despite its doc comment
//! mentioning QUIC/h3 it was ALREADY generic and needed no change here.
//! `PrimeUdpListener` instead implements `crate::packet::PacketListener` —
//! the `Packet{src,dst,data}` seam the umbrella's literal `udp` feature
//! drives (`src/listeners/mod.rs`'s `#[cfg(feature = "udp")] pub use
//! proxima_net::tokio::tokio_packet;`). Two different trait shapes over the
//! one OS primitive, exactly mirroring how the tokio side already has both
//! `TokioUdpListener` (`PacketListener`) with no matching
//! `DatagramFactory` impl of its own.

use std::io;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::Mutex;
use std::task::{Context, Poll};

use bytes::Bytes;
use prime::os::net::UdpSocket;

use crate::packet::{Packet, PacketListener};

/// prime-backed UDP listener. Wraps a `prime::os::net::UdpSocket`; both
/// recv and send go through the same socket, the standard UDP pattern —
/// mirrors `TokioUdpListener` exactly.
///
/// WHY the `Mutex`: `PacketListener: Send + Sync`, but prime's
/// `UdpSocket::poll_recv_from`/`poll_send_to` take `Pin<&mut Self>` (an
/// exclusive-access design — unlike tokio's `&self`-safe equivalents).
/// Interior mutability is required to bridge `PacketListener::poll_recv`/
/// `poll_send`'s `&self` receiver to that exclusive access; same
/// uncontested-single-caller reasoning as `PrimeUnixListener`'s `inner`.
pub struct PrimeUdpListener {
    inner: Mutex<UdpSocket>,
    local_addr: Option<SocketAddr>,
}

impl PrimeUdpListener {
    /// bind a prime-backed UDP socket at `addr`. must be called on a
    /// proxima worker thread (CURRENT_REACTOR must be live).
    pub fn bind(addr: SocketAddr) -> io::Result<Self> {
        let inner = UdpSocket::bind(addr)?;
        let local_addr = inner.local_addr().ok();
        Ok(Self {
            inner: Mutex::new(inner),
            local_addr,
        })
    }
}

impl PacketListener for PrimeUdpListener {
    fn poll_recv(&self, cx: &mut Context<'_>, buf: &mut [u8]) -> Poll<io::Result<Packet>> {
        let Ok(mut guard) = self.inner.lock() else {
            return Poll::Ready(Err(io::Error::other("PrimeUdpListener: lock poisoned")));
        };
        match Pin::new(&mut *guard).poll_recv_from(cx, buf) {
            Poll::Ready(Ok((len, src))) => {
                let dst = self.local_addr.unwrap_or_else(|| {
                    SocketAddr::new(std::net::IpAddr::V4(std::net::Ipv4Addr::UNSPECIFIED), 0)
                });
                Poll::Ready(Ok(Packet {
                    src,
                    dst,
                    data: Bytes::copy_from_slice(&buf[..len]),
                }))
            }
            Poll::Ready(Err(err)) => Poll::Ready(Err(err)),
            Poll::Pending => Poll::Pending,
        }
    }

    fn poll_send(&self, cx: &mut Context<'_>, packet: &Packet) -> Poll<io::Result<()>> {
        let Ok(mut guard) = self.inner.lock() else {
            return Poll::Ready(Err(io::Error::other("PrimeUdpListener: lock poisoned")));
        };
        match Pin::new(&mut *guard).poll_send_to(cx, &packet.data, packet.src) {
            Poll::Ready(Ok(_written)) => Poll::Ready(Ok(())),
            Poll::Ready(Err(err)) => Poll::Ready(Err(err)),
            Poll::Pending => Poll::Pending,
        }
    }

    fn local_addr(&self) -> Option<SocketAddr> {
        self.local_addr
    }
}

/// prime-backed [`PacketListenerFactory`] — the runtime-selectable entry
/// point `RuntimeSelection::prime()` bundles. `bind` is synchronous (same
/// contract as `PrimeUdpListener::bind`): must be called from a proxima
/// worker thread with `CURRENT_REACTOR` live.
pub struct PrimePacketListenerFactory;

impl crate::packet::PacketListenerFactory for PrimePacketListenerFactory {
    fn bind(&self, addr: SocketAddr) -> io::Result<std::sync::Arc<dyn PacketListener>> {
        Ok(std::sync::Arc::new(PrimeUdpListener::bind(addr)?))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::packet::PacketListenerExt;
    use prime::os::core_shard;
    use proxima_runtime::CoreId;
    use std::net::Ipv4Addr;
    use std::sync::mpsc;
    use std::time::Duration;

    // hang guard only: the worker signals completion on the channel, so the
    // test blocks on the event itself rather than polling a flag. no sleep.
    const RESULT_TIMEOUT: Duration = Duration::from_secs(5);

    type ReceivedDatagram = (Vec<u8>, SocketAddr);

    /// full round-trip: two prime-backed UDP listeners on the same worker,
    /// client sends, server receives — mirrors `TokioUdpListener`'s
    /// `udp_listener_round_trips_a_datagram` test exactly.
    #[test]
    fn prime_udp_listener_round_trips_a_datagram() {
        let handle = core_shard::launch_with_lanes(CoreId(0), None, 2, 16).expect("launch");
        let (result_tx, result_rx) = mpsc::channel::<ReceivedDatagram>();

        handle
            .dispatch_factory(Box::new(move || {
                Box::pin(async move {
                    let server = PrimeUdpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))
                        .expect("bind server");
                    let server_addr = server.local_addr().expect("server addr");

                    let client = PrimeUdpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))
                        .expect("bind client");
                    let client_addr = client.local_addr().expect("client addr");

                    let outgoing = Packet {
                        src: server_addr,
                        dst: client_addr,
                        data: Bytes::from_static(b"ping"),
                    };
                    client.send(&outgoing).await.expect("send");

                    let mut buf = vec![0_u8; 1500];
                    let received = server.recv(&mut buf).await.expect("recv");
                    let _ = result_tx.send((received.data.to_vec(), received.src));
                })
                    as std::pin::Pin<Box<dyn std::future::Future<Output = ()> + 'static>>
            }))
            .expect("dispatch_factory");

        let (data, _src) = result_rx
            .recv_timeout(RESULT_TIMEOUT)
            .expect("udp round-trip never completed");
        handle.shutdown_and_join().expect("shutdown");

        assert_eq!(&data[..], b"ping");
    }

    /// sad path: a datagram larger than the receiver's buffer must not
    /// panic or corrupt state — the kernel truncates a UDP datagram to the
    /// caller-provided buffer length and reports the truncated length
    /// (POSIX `recvfrom` semantics), so `poll_recv` observes a short read
    /// rather than an error.
    #[test]
    fn prime_udp_listener_oversized_datagram_is_truncated_not_errored() {
        let handle = core_shard::launch_with_lanes(CoreId(0), None, 2, 16).expect("launch");
        let (result_tx, result_rx) = mpsc::channel::<usize>();

        handle
            .dispatch_factory(Box::new(move || {
                Box::pin(async move {
                    let server = PrimeUdpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))
                        .expect("bind server");
                    let server_addr = server.local_addr().expect("server addr");
                    let client = PrimeUdpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))
                        .expect("bind client");
                    let client_addr = client.local_addr().expect("client addr");

                    let payload = vec![7_u8; 512];
                    let outgoing = Packet {
                        src: server_addr,
                        dst: client_addr,
                        data: Bytes::from(payload),
                    };
                    client.send(&outgoing).await.expect("send");

                    // receiver buffer smaller than the datagram.
                    let mut small_buf = vec![0_u8; 64];
                    let received = server.recv(&mut small_buf).await.expect("recv");
                    let _ = result_tx.send(received.data.len());
                })
                    as std::pin::Pin<Box<dyn std::future::Future<Output = ()> + 'static>>
            }))
            .expect("dispatch_factory");

        let received_len = result_rx
            .recv_timeout(RESULT_TIMEOUT)
            .expect("oversized-datagram test never completed");
        handle.shutdown_and_join().expect("shutdown");

        assert_eq!(
            received_len, 64,
            "oversized datagram must be truncated to the receiver's buffer length"
        );
    }

    /// round-trip through the `PacketListenerFactory` seam itself (not
    /// `PrimeUdpListener::bind` directly) — proves the `bind()` method
    /// `RuntimeSelection::prime()` actually wires up works end to end,
    /// which nothing exercised before this test.
    #[test]
    fn prime_packet_listener_factory_binds_and_round_trips_a_datagram() {
        use crate::packet::PacketListenerFactory;

        let handle = core_shard::launch_with_lanes(CoreId(0), None, 2, 16).expect("launch");
        let (result_tx, result_rx) = mpsc::channel::<ReceivedDatagram>();

        handle
            .dispatch_factory(Box::new(move || {
                Box::pin(async move {
                    let factory = PrimePacketListenerFactory;
                    let server = factory
                        .bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))
                        .expect("bind server");
                    let server_addr = server.local_addr().expect("server addr");

                    let client = factory
                        .bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))
                        .expect("bind client");
                    let client_addr = client.local_addr().expect("client addr");

                    let outgoing = Packet {
                        src: server_addr,
                        dst: client_addr,
                        data: Bytes::from_static(b"ping"),
                    };
                    client.send(&outgoing).await.expect("send");

                    let mut buf = vec![0_u8; 1500];
                    let received = server.recv(&mut buf).await.expect("recv");
                    let _ = result_tx.send((received.data.to_vec(), received.src));
                })
                    as std::pin::Pin<Box<dyn std::future::Future<Output = ()> + 'static>>
            }))
            .expect("dispatch_factory");

        let (data, _src) = result_rx
            .recv_timeout(RESULT_TIMEOUT)
            .expect("packet listener factory round-trip never completed");
        handle.shutdown_and_join().expect("shutdown");

        assert_eq!(&data[..], b"ping");
    }
}
