//! Bidirectional byte-stream primitive. Trait bound is `futures::io`
//! (not `tokio::io`) so non-tokio backends — AF_XDP, DPDK, glommio,
//! smol — can implement without a tokio dep. Tokio backends use the
//! `tokio-util::compat` shim at the impl boundary.
//!
//! Two-tier by design (principle 3): every trait in this crate is built on
//! `std::io::{Error, Result}` and/or `futures::io`'s `AsyncRead`/`AsyncWrite`,
//! none of which has a `core`/`alloc` analog (`futures-io`'s items only exist
//! under its own `std` feature; see `proxima_core::io`'s doc comment
//! for why a no_std async IO seam needs an associated-`Error` type instead
//! of `std::io::Error`). The surface is therefore gated behind
//! `#[cfg(feature = "std")]` in full; the alloc tier keeps only the
//! tier-agnostic pieces — [`TcpBindOptions`] and the [`BindAddr`]/
//! [`PeerInfo`] re-export.

#[cfg(feature = "std")]
use std::io;

#[cfg(feature = "std")]
use core::future::Future;
#[cfg(feature = "std")]
use core::net::SocketAddr;
#[cfg(feature = "std")]
use core::pin::Pin;
#[cfg(feature = "std")]
use core::task::{Context, Poll};

#[cfg(feature = "std")]
use futures::io::{AsyncRead, AsyncWrite};

// gated on alloc (not left unconditional): `crate::pipe::endpoint` only
// exists under pipe's own alloc feature, so a bare no_std+no-alloc build
// must not reference it.
#[cfg(feature = "alloc")]
pub use crate::pipe::endpoint::{BindAddr, PeerInfo};

#[cfg(feature = "std")]
mod datagram_batch_ext;
#[cfg(feature = "std")]
pub use datagram_batch_ext::DatagramSocketBatchExt;

/// Default `listen()` backlog when the caller does not specify one.
pub const DEFAULT_LISTEN_BACKLOG: u32 = 1024;

#[cfg(feature = "std")]
pub trait StreamConnection: AsyncRead + AsyncWrite + Send + Unpin + 'static {
    fn peer(&self) -> Option<PeerInfo>;
}

// A boxed connection is itself a connection: `futures::io` already impls
// AsyncRead/AsyncWrite for `Box<T: ?Sized + …>`, so a listener that wants to
// accept a caller-supplied wrapper (e.g. a cipher-wrapped socket) can take a
// `Box<dyn StreamConnection>` as its `C` without naming the concrete type.
#[cfg(feature = "std")]
impl StreamConnection for Box<dyn StreamConnection> {
    fn peer(&self) -> Option<PeerInfo> {
        (**self).peer()
    }
}

#[cfg(feature = "std")]
pub trait StreamListener: Send + Sync + 'static {
    type Conn: StreamConnection;

    fn poll_accept(&self, cx: &mut Context<'_>) -> Poll<io::Result<Self::Conn>>;

    fn local_addr(&self) -> Option<BindAddr>;
}

#[cfg(feature = "std")]
pub trait StreamUpstream: Send + Sync + 'static {
    type Conn: StreamConnection;

    fn poll_connect(&self, cx: &mut Context<'_>) -> Poll<io::Result<Self::Conn>>;
}

#[cfg(feature = "std")]
pub trait StreamListenerExt: StreamListener {
    fn accept(&self) -> Accept<'_, Self> {
        Accept { listener: self }
    }
}

#[cfg(feature = "std")]
impl<T: StreamListener + ?Sized> StreamListenerExt for T {}

#[cfg(feature = "std")]
pub trait StreamUpstreamExt: StreamUpstream {
    fn connect(&self) -> Connect<'_, Self> {
        Connect { upstream: self }
    }
}

#[cfg(feature = "std")]
impl<T: StreamUpstream + ?Sized> StreamUpstreamExt for T {}

#[cfg(feature = "std")]
pub struct Accept<'lifetime, L: StreamListener + ?Sized> {
    listener: &'lifetime L,
}

#[cfg(feature = "std")]
impl<L: StreamListener + ?Sized> Future for Accept<'_, L> {
    type Output = io::Result<L::Conn>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.listener.poll_accept(cx)
    }
}

#[cfg(feature = "std")]
pub struct Connect<'lifetime, U: StreamUpstream + ?Sized> {
    upstream: &'lifetime U,
}

#[cfg(feature = "std")]
impl<U: StreamUpstream + ?Sized> Future for Connect<'_, U> {
    type Output = io::Result<U::Conn>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.upstream.poll_connect(cx)
    }
}

/// Listen-socket bind options shared by every runtime backend.
#[derive(Debug, Clone, Copy)]
pub struct TcpBindOptions {
    /// `listen()` queue depth.
    pub backlog: u32,
    /// Request `SO_REUSEPORT` so each core can bind its own lane. Prime
    /// honors a single-core listener today; tokio fans out per core.
    pub reuseport: bool,
    /// Kernel passive-open queue depth for TCP Fast Open (RFC 7413,
    /// Linux). `None` leaves TFO off; non-Linux platforms ignore it.
    pub tcp_fastopen: Option<u32>,
}

impl Default for TcpBindOptions {
    fn default() -> Self {
        Self {
            backlog: DEFAULT_LISTEN_BACKLOG,
            reuseport: false,
            tcp_fastopen: None,
        }
    }
}

/// A bound TCP acceptor pinned to the worker that created it. `Send` (so
/// it can be held across awaits in a `Send` bootstrap future) but NOT
/// `Sync`: prime's reactor-backed listener caches a per-worker reactor
/// pointer that is UB to poll off its origin worker. This is the bound
/// that `StreamListener: Send + Sync` could not satisfy, so the runtime
/// accept surface drops `Sync`.
#[cfg(feature = "std")]
pub trait TcpAcceptor: Send {
    /// Accept the next connection, yielding a boxed `StreamConnection`.
    /// Must be polled on the worker that produced this acceptor.
    fn poll_accept(&mut self, cx: &mut Context<'_>) -> Poll<io::Result<Box<dyn StreamConnection>>>;

    /// The local address the listener bound to.
    fn local_addr(&self) -> io::Result<SocketAddr>;
}

/// Runtime-agnostic factory that binds a [`TcpAcceptor`] on the calling
/// worker. Injected into the serve path so a listener loop can obtain a
/// prime- or tokio-backed acceptor without naming the concrete runtime.
///
/// The factory is `Send + Sync` (it is shared via `Arc` and only builds
/// acceptors); the acceptor it produces is `Send`-but-`!Sync` and lives
/// on one worker. `bind` MUST be called from within a future already
/// running on the target runtime's worker — prime needs `CURRENT_REACTOR`
/// live, tokio needs its reactor entered.
#[cfg(feature = "std")]
pub trait AcceptorFactory: Send + Sync + 'static {
    fn bind(&self, addr: SocketAddr, options: TcpBindOptions) -> io::Result<Box<dyn TcpAcceptor>>;
}

/// A bound datagram (UDP) socket pinned to the worker that created it.
/// `Send`-but-`!Sync` for the same reason as [`TcpAcceptor`]: prime's reactor
/// registration is per-worker. A QUIC/h3 listener demuxes many connections over
/// this one socket, so poll it only on the worker that produced it.
#[cfg(feature = "std")]
pub trait DatagramSocket: Send {
    /// Receive one datagram into `buf`; yields `(len, sender)`.
    fn poll_recv_from(
        &mut self,
        cx: &mut Context<'_>,
        buf: &mut [u8],
    ) -> Poll<io::Result<(usize, SocketAddr)>>;

    /// Send `buf` to `peer`; `Pending` when the kernel send buffer is full.
    fn poll_send_to(
        &mut self,
        cx: &mut Context<'_>,
        buf: &[u8],
        peer: SocketAddr,
    ) -> Poll<io::Result<usize>>;

    /// Receive up to `min(bufs.len(), out_meta.len())` datagrams at once,
    /// writing each datagram's `(len, sender)` into `out_meta` and returning
    /// the count. The default loops [`poll_recv_from`](Self::poll_recv_from);
    /// a backend that can pull many datagrams per syscall (prime's `recvmmsg`)
    /// overrides this to amortize the kernel-entry cost across the burst.
    fn poll_recv_batch(
        &mut self,
        cx: &mut Context<'_>,
        bufs: &mut [&mut [u8]],
        out_meta: &mut [(usize, SocketAddr)],
    ) -> Poll<io::Result<usize>> {
        let want = bufs.len().min(out_meta.len());
        let mut count = 0;
        while count < want {
            // reborrow the slot buffer to a plain &mut [u8] for the single recv.
            let slot: &mut [u8] = &mut *bufs[count];
            match self.poll_recv_from(cx, slot) {
                Poll::Ready(Ok((len, peer))) => {
                    out_meta[count] = (len, peer);
                    count += 1;
                }
                Poll::Ready(Err(_)) if count > 0 => return Poll::Ready(Ok(count)),
                Poll::Ready(Err(err)) => return Poll::Ready(Err(err)),
                Poll::Pending if count > 0 => return Poll::Ready(Ok(count)),
                Poll::Pending => return Poll::Pending,
            }
        }
        Poll::Ready(Ok(count))
    }

    /// Send all `packets` (each `(bytes, peer)`) in as few syscalls as
    /// possible, returning the count accepted. The default loops
    /// [`poll_send_to`](Self::poll_send_to); prime overrides with `sendmmsg`.
    fn poll_send_batch(
        &mut self,
        cx: &mut Context<'_>,
        packets: &[(&[u8], SocketAddr)],
    ) -> Poll<io::Result<usize>> {
        let mut sent = 0;
        while sent < packets.len() {
            let (bytes, peer) = packets[sent];
            match self.poll_send_to(cx, bytes, peer) {
                Poll::Ready(Ok(_)) => sent += 1,
                Poll::Ready(Err(_)) if sent > 0 => return Poll::Ready(Ok(sent)),
                Poll::Ready(Err(err)) => return Poll::Ready(Err(err)),
                Poll::Pending if sent > 0 => return Poll::Ready(Ok(sent)),
                Poll::Pending => return Poll::Pending,
            }
        }
        Poll::Ready(Ok(sent))
    }

    /// The local address the socket bound to.
    fn local_addr(&self) -> io::Result<SocketAddr>;
}

/// Runtime-agnostic factory that binds a [`DatagramSocket`] on the calling
/// worker — the UDP sibling of [`AcceptorFactory`]. Injected into the serve path
/// so a QUIC/h3 listener obtains a prime- or tokio-backed UDP socket without
/// naming the concrete runtime. `bind` MUST be called from within a future
/// already running on the target runtime's worker.
#[cfg(feature = "std")]
pub trait DatagramFactory: Send + Sync + 'static {
    fn bind(&self, addr: SocketAddr) -> io::Result<Box<dyn DatagramSocket>>;
}
