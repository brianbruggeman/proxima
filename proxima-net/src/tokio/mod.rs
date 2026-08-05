//! Tokio-backed implementations of `proxima-stream` and `proxima-net`
//! trait surfaces. TCP + Unix + UDP listeners and upstreams.

use std::io;
use std::net::SocketAddr;

use socket2::{Domain, Protocol, Socket, Type};

pub mod tokio_acceptor;
pub mod tokio_datagram;
pub mod tokio_packet;
pub mod tokio_stream_listener;
pub mod tokio_stream_upstream;

pub use tokio_acceptor::{TokioAcceptor, TokioAcceptorFactory};
pub use tokio_datagram::{TokioDatagram, TokioDatagramFactory};
pub use tokio_packet::{TokioPacketListenerFactory, TokioUdpListener};
pub use tokio_stream_listener::{
    TokioTcpConnection, TokioTcpListener, TokioUnixConnection, TokioUnixListener,
};
pub use tokio_stream_upstream::{TokioTcpUpstream, TokioUnixUpstream, TokioUnixUpstreamFactory};

pub(crate) fn domain_for(addr: SocketAddr) -> Domain {
    if addr.is_ipv4() {
        Domain::IPV4
    } else {
        Domain::IPV6
    }
}

/// Bind a nonblocking std UDP socket via `socket2`, ready to hand to
/// `tokio::net::UdpSocket::from_std`. Keeps every tokio UDP `bind` synchronous
/// — the same socket2-then-`from_std` bridge `TokioAcceptorFactory::bind`
/// uses for its listen socket.
pub(crate) fn bind_udp_std(addr: SocketAddr) -> io::Result<std::net::UdpSocket> {
    let socket = Socket::new(domain_for(addr), Type::DGRAM, Some(Protocol::UDP))?;
    socket.set_nonblocking(true)?;
    socket.bind(&addr.into())?;
    Ok(socket.into())
}
