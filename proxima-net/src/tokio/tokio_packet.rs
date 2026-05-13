//! Tokio-backed UDP `PacketListener`. The default cross-platform
//! packet backend — handles mesh DHT traffic, DNS-style lookups, and
//! any other datagram protocol that doesn't need kernel-bypass
//! throughput.

use std::io;
use std::net::SocketAddr;
use std::task::{Context, Poll};

use bytes::Bytes;
use tokio::net::UdpSocket;

use crate::packet::{Packet, PacketListener};

/// Tokio-backed UDP listener. Wraps a `tokio::net::UdpSocket`; both
/// recv and send go through the same socket, which is the standard
/// UDP pattern.
pub struct TokioUdpListener {
    inner: UdpSocket,
    local_addr: Option<SocketAddr>,
}

impl TokioUdpListener {
    pub async fn bind(addr: SocketAddr) -> io::Result<Self> {
        let inner = UdpSocket::bind(addr).await?;
        let local_addr = inner.local_addr().ok();
        Ok(Self { inner, local_addr })
    }
}

impl PacketListener for TokioUdpListener {
    fn poll_recv(&self, cx: &mut Context<'_>, buf: &mut [u8]) -> Poll<io::Result<Packet>> {
        let mut read_buf = tokio::io::ReadBuf::new(buf);
        match self.inner.poll_recv_from(cx, &mut read_buf) {
            Poll::Ready(Ok(src)) => {
                let len = read_buf.filled().len();
                let dst = self.local_addr.unwrap_or_else(|| {
                    // best-effort fallback if local_addr was unavailable
                    // at bind time; use the unspecified address.
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
        match self.inner.poll_send_to(cx, &packet.data, packet.src) {
            Poll::Ready(Ok(_n)) => Poll::Ready(Ok(())),
            Poll::Ready(Err(err)) => Poll::Ready(Err(err)),
            Poll::Pending => Poll::Pending,
        }
    }

    fn local_addr(&self) -> Option<SocketAddr> {
        self.local_addr
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::packet::PacketListenerExt;
    use std::net::Ipv4Addr;

    #[proxima::test]
    async fn udp_listener_round_trips_a_datagram() {
        let server = TokioUdpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))
            .await
            .expect("bind server");
        let server_addr = server.local_addr().expect("server addr");

        let client = TokioUdpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))
            .await
            .expect("bind client");
        let client_addr = client.local_addr().expect("client addr");

        // client sends; the `src` field on outgoing Packet is the
        // *destination*, matching the mesh-style usage where each
        // outbound packet says "send this to {peer}".
        let outgoing = Packet {
            src: server_addr,
            dst: client_addr,
            data: Bytes::from_static(b"ping"),
        };
        client.send(&outgoing).await.expect("send");

        let mut buf = vec![0_u8; 1500];
        let received = server.recv(&mut buf).await.expect("recv");
        assert_eq!(&received.data[..], b"ping");
        assert_eq!(received.src, client_addr);
    }
}
