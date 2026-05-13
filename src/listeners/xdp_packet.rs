//! AF_XDP packet-listener skeleton — every async op returns
//! `io::ErrorKind::Unsupported`. Real impl requires Linux + xsk-rs +
//! an eBPF redirect program; the trait shape is locked here so a
//! future implementer can fill it without breaking callers.

use std::io;
use std::net::SocketAddr;
use std::task::{Context, Poll};

use crate::packet::{Packet, PacketListener};

const NOT_IMPLEMENTED: &str = "af_xdp backend skeleton — implementation lives in a future plan; see proxima/rust/src/listeners/xdp_packet.rs module docstring";

/// Skeleton AF_XDP packet listener. Linux-only (`cfg(target_os = \"linux\")`)
/// in production; the skeleton compiles cross-platform so the trait
/// fit and feature flag composition are checked on dev boxes.
pub struct XdpUdpListener {
    _private: (),
}

impl XdpUdpListener {
    pub fn new() -> io::Result<Self> {
        Err(unsupported())
    }
}

impl PacketListener for XdpUdpListener {
    fn poll_recv(&self, _cx: &mut Context<'_>, _buf: &mut [u8]) -> Poll<io::Result<Packet>> {
        Poll::Ready(Err(unsupported()))
    }

    fn poll_send(&self, _cx: &mut Context<'_>, _packet: &Packet) -> Poll<io::Result<()>> {
        Poll::Ready(Err(unsupported()))
    }

    fn local_addr(&self) -> Option<SocketAddr> {
        None
    }
}

fn unsupported() -> io::Error {
    io::Error::new(io::ErrorKind::Unsupported, NOT_IMPLEMENTED)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::packet::PacketListenerExt;

    #[test]
    fn listener_new_returns_unsupported() {
        match XdpUdpListener::new() {
            Err(err) if err.kind() == io::ErrorKind::Unsupported => {
                assert!(
                    err.to_string().contains("af_xdp"),
                    "expected af_xdp-tagged message: {err}"
                );
            }
            Err(err) => panic!("expected Unsupported, got {:?}: {err}", err.kind()),
            Ok(_) => panic!("expected Unsupported, got Ok"),
        }
    }

    #[proxima::test]
    async fn placeholder_recv_yields_unsupported() {
        let listener = XdpUdpListener { _private: () };
        let mut buf = [0_u8; 16];
        let outcome = listener.recv(&mut buf).await;
        assert!(matches!(outcome, Err(err) if err.kind() == io::ErrorKind::Unsupported));
    }

    #[proxima::test]
    async fn placeholder_send_yields_unsupported() {
        let listener = XdpUdpListener { _private: () };
        let packet = Packet {
            src: SocketAddr::from(([127, 0, 0, 1], 9999)),
            dst: SocketAddr::from(([127, 0, 0, 1], 0)),
            data: bytes::Bytes::from_static(b"x"),
        };
        let outcome = listener.send(&packet).await;
        assert!(matches!(outcome, Err(err) if err.kind() == io::ErrorKind::Unsupported));
    }
}
