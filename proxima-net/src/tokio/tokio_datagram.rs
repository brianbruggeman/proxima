//! Tokio-backed [`DatagramFactory`]/[`DatagramSocket`] — the UDP sibling of
//! `TokioAcceptorFactory`, and tokio's counterpart to
//! [`super::super::prime::PrimeDatagramFactory`]/`PrimeDatagram`. Closes the
//! runtime-capability asymmetry where `DatagramFactory` had exactly one
//! implementation (prime): a tokio-backed `RuntimeSelection` previously
//! carried `datagram_factory: None`, so any h3/QUIC or
//! `DatagramListenProtocol` listener was unreachable on a tokio-backed App.
//!
//! Mirrors `TokioAcceptorFactory`'s bridge (`tokio_acceptor.rs`): build the
//! socket via `socket2` so `bind` stays synchronous, then hand the pre-bound
//! std socket to tokio through `from_std`. `bind` must be called from within
//! a future already running on a tokio worker so the reactor is live.
//!
//! `poll_recv_batch`/`poll_send_batch` are NOT overridden here — tokio's
//! `UdpSocket` has no `recvmmsg`/`sendmmsg` equivalent (prime's does, via
//! `prime::os::net::UdpSocket::poll_recv_batch`), so this type inherits
//! `DatagramSocket`'s default loop-of-single-syscalls implementation:
//! correct, just not batched into one kernel entry. A caller that needs the
//! batched syscall path wants the prime backend.

use std::io;
use std::net::SocketAddr;
use std::task::{Context, Poll};

use tokio::net::UdpSocket;

use proxima_primitives::stream::{DatagramFactory, DatagramSocket};

/// tokio-backed [`DatagramSocket`]. Wraps a `tokio::net::UdpSocket`; both
/// recv and send go through the same socket, the standard UDP pattern —
/// mirrors `proxima_net::prime::PrimeDatagram`'s shape.
pub struct TokioDatagram {
    inner: UdpSocket,
}

impl TokioDatagram {
    /// Bind a std socket via `socket2` and register it on the calling
    /// tokio worker's reactor — the same bridge `TokioAcceptorFactory::bind`
    /// uses, so `bind` stays synchronous even though
    /// `tokio::net::UdpSocket::bind` itself is `async`. Must be called from
    /// within a future already running on a tokio worker.
    pub fn bind_sync(addr: SocketAddr) -> io::Result<Self> {
        let inner = UdpSocket::from_std(super::bind_udp_std(addr)?)?;
        Ok(Self { inner })
    }
}

impl DatagramSocket for TokioDatagram {
    fn poll_recv_from(
        &mut self,
        cx: &mut Context<'_>,
        buf: &mut [u8],
    ) -> Poll<io::Result<(usize, SocketAddr)>> {
        let mut read_buf = tokio::io::ReadBuf::new(buf);
        match self.inner.poll_recv_from(cx, &mut read_buf) {
            Poll::Ready(Ok(peer)) => {
                let len = read_buf.filled().len();
                Poll::Ready(Ok((len, peer)))
            }
            Poll::Ready(Err(err)) => Poll::Ready(Err(err)),
            Poll::Pending => Poll::Pending,
        }
    }

    fn poll_send_to(
        &mut self,
        cx: &mut Context<'_>,
        buf: &[u8],
        peer: SocketAddr,
    ) -> Poll<io::Result<usize>> {
        self.inner.poll_send_to(cx, buf, peer)
    }

    fn local_addr(&self) -> io::Result<SocketAddr> {
        self.inner.local_addr()
    }
}

/// tokio-backed [`DatagramFactory`] — the UDP sibling of
/// `TokioAcceptorFactory`. Binds a `tokio::net::UdpSocket` on the calling
/// worker for QUIC/h3 and `DatagramListenProtocol` listeners.
pub struct TokioDatagramFactory;

impl DatagramFactory for TokioDatagramFactory {
    fn bind(&self, addr: SocketAddr) -> io::Result<Box<dyn DatagramSocket>> {
        Ok(Box::new(TokioDatagram::bind_sync(addr)?))
    }

    fn backend_name(&self) -> &'static str {
        "tokio"
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use std::future::poll_fn;
    use std::net::Ipv4Addr;

    #[proxima::test(runtime = "tokio")]
    async fn factory_binds_and_round_trips_a_datagram() {
        let factory = TokioDatagramFactory;
        assert_eq!(factory.backend_name(), "tokio");

        let mut server = factory
            .bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))
            .expect("bind server");
        let server_addr = server.local_addr().expect("server addr");

        let mut client = factory
            .bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))
            .expect("bind client");
        let client_addr = client.local_addr().expect("client addr");

        poll_fn(|cx| client.poll_send_to(cx, b"ping", server_addr))
            .await
            .expect("client send");

        let mut buf = [0_u8; 16];
        let (len, peer) = poll_fn(|cx| server.poll_recv_from(cx, &mut buf))
            .await
            .expect("server recv");
        assert_eq!(peer, client_addr);
        assert_eq!(&buf[..len], b"ping");
    }

    /// sad path: a datagram larger than the receiver's buffer must not
    /// panic or corrupt state — the kernel truncates a UDP datagram to the
    /// caller-provided buffer length, mirroring
    /// `PrimeUdpListener`'s `prime_udp_listener_oversized_datagram_is_truncated_not_errored`.
    #[proxima::test(runtime = "tokio")]
    async fn oversized_datagram_is_truncated_not_errored() {
        let factory = TokioDatagramFactory;
        let mut server = factory
            .bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))
            .expect("bind server");
        let server_addr = server.local_addr().expect("server addr");
        let mut client = factory
            .bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))
            .expect("bind client");

        let payload = vec![7_u8; 512];
        poll_fn(|cx| client.poll_send_to(cx, &payload, server_addr))
            .await
            .expect("client send");

        let mut small_buf = [0_u8; 64];
        let (len, _peer) = poll_fn(|cx| server.poll_recv_from(cx, &mut small_buf))
            .await
            .expect("server recv");
        assert_eq!(
            len, 64,
            "oversized datagram must be truncated to the receiver's buffer length"
        );
    }
}
