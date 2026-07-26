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
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    type ReceivedDatagram = Arc<std::sync::Mutex<Option<(Vec<u8>, SocketAddr)>>>;

    /// full round-trip: two prime-backed UDP listeners on the same worker,
    /// client sends, server receives — mirrors `TokioUdpListener`'s
    /// `udp_listener_round_trips_a_datagram` test exactly.
    #[test]
    fn prime_udp_listener_round_trips_a_datagram() {
        let handle = core_shard::launch_with_lanes(CoreId(0), None, 2, 16).expect("launch");
        let done = Arc::new(AtomicBool::new(false));
        let done_clone = done.clone();
        let result_chan: ReceivedDatagram = Arc::new(std::sync::Mutex::new(None));
        let result_for_factory = result_chan.clone();

        handle
            .dispatch_factory(Box::new(move || {
                let done = done_clone.clone();
                let result_handle = result_for_factory.clone();
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
                    *result_handle.lock().unwrap() =
                        Some((received.data.to_vec(), received.src));
                    done.store(true, Ordering::Release);
                }) as std::pin::Pin<Box<dyn std::future::Future<Output = ()> + 'static>>
            }))
            .expect("dispatch_factory");

        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while !done.load(Ordering::Acquire) {
            assert!(
                std::time::Instant::now() < deadline,
                "udp round-trip never completed"
            );
            std::thread::sleep(Duration::from_millis(10));
        }

        handle.shutdown_and_join().expect("shutdown");

        let (data, _src) = result_chan.lock().unwrap().clone().expect("result not set");
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
        let done = Arc::new(AtomicBool::new(false));
        let done_clone = done.clone();
        let result_chan: Arc<std::sync::Mutex<Option<usize>>> = Arc::new(std::sync::Mutex::new(None));
        let result_for_factory = result_chan.clone();

        handle
            .dispatch_factory(Box::new(move || {
                let done = done_clone.clone();
                let result_handle = result_for_factory.clone();
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
                    *result_handle.lock().unwrap() = Some(received.data.len());
                    done.store(true, Ordering::Release);
                }) as std::pin::Pin<Box<dyn std::future::Future<Output = ()> + 'static>>
            }))
            .expect("dispatch_factory");

        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while !done.load(Ordering::Acquire) {
            assert!(
                std::time::Instant::now() < deadline,
                "oversized-datagram test never completed"
            );
            std::thread::sleep(Duration::from_millis(10));
        }

        handle.shutdown_and_join().expect("shutdown");

        let received_len = result_chan.lock().unwrap().expect("result not set");
        assert_eq!(
            received_len, 64,
            "oversized datagram must be truncated to the receiver's buffer length"
        );
    }
}
