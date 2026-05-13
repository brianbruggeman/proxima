//! prime-backed `StreamUpstream` for TCP. Mirrors `proxima-net-tokio` in
//! shape but uses the prime reactor (`prime::os::net::TcpStream`) instead
//! of tokio — zero tokio dependency.
//!
//! The key types:
//!   - `PrimeTcpConnection` — newtype over `prime::os::net::TcpStream` that
//!     satisfies `StreamConnection` (adds the `peer()` accessor).
//!   - `PrimeTcpUpstream` — `StreamUpstream` that dials a TCP peer via the
//!     prime reactor, returning a `PrimeTcpConnection`.
//!
//! `prime::os::net` is itself gated behind `runtime-prime-inbox-alloc`
//! (mutually exclusive with `runtime-prime-inbox-const`, see
//! `prime/src/core/inbox.rs`), so this whole module goes empty without
//! that feature too — the `prime` feature is an always-on dependency for
//! consumers that turn it on (including via `[dev-dependencies]` for test
//! builds), and forcing alloc on would fight a build that explicitly asked
//! the workspace for const. The `target_os` + `runtime-prime-inbox-alloc`
//! gate lives on this module's declaration in `crate::lib` (see `pub mod
//! prime` there).
//!
//! `PrimeTcpConnection` implements `futures::io::{AsyncRead, AsyncWrite}`
//! ONLY — the industry-standard, std-tier trait `prime::os::net::TcpStream`
//! itself implements (`prime/src/os/net.rs:64,594,638`) and the one
//! `proxima_primitives::stream::StreamConnection` requires. This type is
//! std-only by construction (it wraps a std `Mutex` and prime's std-gated
//! `net` module), so it never needs `proxima_core::io`'s no_std/no-alloc
//! floor form — that form exists ONLY for types that must also compile
//! without std (see `proxima_core::io`'s own module doc). A prior revision
//! of this module carried a second, redundant `proxima_core::io::{AsyncRead,
//! AsyncWrite}` impl here purely as a floor-seam proof; it was removed
//! (`docs/pipe-to-metal/edges.md`, 2026-07-16 concentration entry) because
//! two AsyncRead/AsyncWrite impls on the one real, always-std socket type
//! is exactly the ambiguity principle 1/2 rule out — a reader must be able
//! to find ONE canonical trait per type, not pick between two that do the
//! same thing.

use std::io;
use std::net::{SocketAddr, ToSocketAddrs};
use std::pin::Pin;
use std::sync::Mutex;
use std::task::{Context, Poll};

use futures::io::{AsyncRead, AsyncWrite};
use prime::os::net::{TcpListener, TcpStream, UdpSocket};
use proxima_primitives::stream::{
    AcceptorFactory, DatagramFactory, DatagramSocket, PeerInfo, StreamConnection, StreamUpstream,
    TcpAcceptor, TcpBindOptions,
};

mod connect_tunnel;
pub use connect_tunnel::ConnectTunneledUpstream;

type ConnectFuture =
    Pin<Box<dyn std::future::Future<Output = io::Result<PrimeTcpConnection>> + Send>>;

/// prime-backed TCP connection. wraps `prime::os::net::TcpStream` and
/// carries the peer address so `StreamConnection::peer()` is satisfied.
pub struct PrimeTcpConnection {
    inner: TcpStream,
    peer: Option<SocketAddr>,
}

impl PrimeTcpConnection {
    fn new(stream: TcpStream, peer: SocketAddr) -> Self {
        Self {
            inner: stream,
            peer: Some(peer),
        }
    }

    /// build a connection from an accepted prime stream + its peer addr.
    /// lets the acceptor construct connections without exposing the field.
    pub fn new_connection(stream: TcpStream, peer: SocketAddr) -> Self {
        Self::new(stream, peer)
    }
}

impl AsyncRead for PrimeTcpConnection {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut [u8],
    ) -> Poll<io::Result<usize>> {
        Pin::new(&mut self.get_mut().inner).poll_read(cx, buf)
    }
}

impl AsyncWrite for PrimeTcpConnection {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        Pin::new(&mut self.get_mut().inner).poll_write(cx, buf)
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.get_mut().inner).poll_flush(cx)
    }

    fn poll_close(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.get_mut().inner).poll_close(cx)
    }
}

impl StreamConnection for PrimeTcpConnection {
    fn peer(&self) -> Option<PeerInfo> {
        self.peer.map(PeerInfo::Tcp)
    }
}

/// prime-backed [`AcceptorFactory`]. binds a listening socket on the
/// calling prime worker (CURRENT_REACTOR must be live — the caller's
/// contract) and hands back a [`PrimeAcceptor`].
pub struct PrimeAcceptorFactory;

impl AcceptorFactory for PrimeAcceptorFactory {
    fn bind(&self, addr: SocketAddr, options: TcpBindOptions) -> io::Result<Box<dyn TcpAcceptor>> {
        // prime serve is single-core; reuseport/fastopen per-core fan-out is
        // a follow-on. ignore the flags rather than fake partial support.
        let _ = options.reuseport;
        let _ = options.tcp_fastopen;
        let listener = TcpListener::bind_with_backlog(addr, options.backlog as i32)?;
        Ok(Box::new(PrimeAcceptor { listener }))
    }
}

/// prime-backed [`TcpAcceptor`]. drives `prime::os::net::TcpListener`'s
/// `poll_accept` and wraps each accepted stream as a `PrimeTcpConnection`.
pub struct PrimeAcceptor {
    listener: TcpListener,
}

impl TcpAcceptor for PrimeAcceptor {
    fn poll_accept(&mut self, cx: &mut Context<'_>) -> Poll<io::Result<Box<dyn StreamConnection>>> {
        match Pin::new(&mut self.listener).poll_accept(cx) {
            Poll::Ready(Ok((stream, peer))) => {
                let conn = PrimeTcpConnection::new_connection(stream, peer);
                Poll::Ready(Ok(Box::new(conn) as Box<dyn StreamConnection>))
            }
            Poll::Ready(Err(err)) => Poll::Ready(Err(err)),
            Poll::Pending => Poll::Pending,
        }
    }

    fn local_addr(&self) -> io::Result<SocketAddr> {
        self.listener.local_addr()
    }
}

/// prime-backed [`DatagramFactory`] — the UDP sibling of
/// [`PrimeAcceptorFactory`]. Binds a `prime::os::net::UdpSocket` on the calling
/// prime worker (CURRENT_REACTOR must be live) for QUIC/h3 listeners.
pub struct PrimeDatagramFactory;

impl DatagramFactory for PrimeDatagramFactory {
    fn bind(&self, addr: SocketAddr) -> io::Result<Box<dyn DatagramSocket>> {
        Ok(Box::new(PrimeDatagram {
            socket: UdpSocket::bind(addr)?,
        }))
    }
}

/// prime-backed [`DatagramSocket`] over `prime::os::net::UdpSocket`.
pub struct PrimeDatagram {
    socket: UdpSocket,
}

impl DatagramSocket for PrimeDatagram {
    fn poll_recv_from(
        &mut self,
        cx: &mut Context<'_>,
        buf: &mut [u8],
    ) -> Poll<io::Result<(usize, SocketAddr)>> {
        Pin::new(&mut self.socket).poll_recv_from(cx, buf)
    }

    fn poll_send_to(
        &mut self,
        cx: &mut Context<'_>,
        buf: &[u8],
        peer: SocketAddr,
    ) -> Poll<io::Result<usize>> {
        Pin::new(&mut self.socket).poll_send_to(cx, buf, peer)
    }

    fn poll_recv_batch(
        &mut self,
        cx: &mut Context<'_>,
        bufs: &mut [&mut [u8]],
        out_meta: &mut [(usize, SocketAddr)],
    ) -> Poll<io::Result<usize>> {
        Pin::new(&mut self.socket).poll_recv_batch(cx, bufs, out_meta)
    }

    fn poll_send_batch(
        &mut self,
        cx: &mut Context<'_>,
        packets: &[(&[u8], SocketAddr)],
    ) -> Poll<io::Result<usize>> {
        Pin::new(&mut self.socket).poll_send_batch(cx, packets)
    }

    fn local_addr(&self) -> io::Result<SocketAddr> {
        self.socket.local_addr()
    }
}

/// Dial target. A pre-resolved `SocketAddr` connects immediately; a
/// `Host` defers `getaddrinfo` to connect time so the upstream can be
/// built for a name that is not yet (or never) reachable — the umbrella
/// builds upstream specs with fake hosts that never connect.
enum Target {
    Addr(SocketAddr),
    Host { host: String, port: u16 },
}

/// prime-backed TCP upstream. dials the target via the prime reactor on
/// each `connect()` call. the in-flight future is cached across polls so a
/// pending connect can resume.
///
/// WHY Mutex on `in_flight`: same rationale as `TokioTcpUpstream` —
/// `poll_connect` takes `&self` (trait surface cannot allow `&mut self`)
/// so interior mutability is required to stash the future between polls.
pub struct PrimeTcpUpstream {
    target: Target,
    in_flight: Mutex<Option<ConnectFuture>>,
}

impl PrimeTcpUpstream {
    pub fn new(addr: SocketAddr) -> Self {
        Self {
            target: Target::Addr(addr),
            in_flight: Mutex::new(None),
        }
    }

    /// Build from a host + port, deferring DNS resolution to connect
    /// time. `build()` of an upstream spec must not touch the network or
    /// the resolver — only an actual `connect()` may.
    pub fn with_host(host: impl Into<String>, port: u16) -> Self {
        Self {
            target: Target::Host {
                host: host.into(),
                port,
            },
            in_flight: Mutex::new(None),
        }
    }
}

/// Resolve host + port to the first `SocketAddr` via the system resolver.
fn resolve(host: &str, port: u16) -> io::Result<SocketAddr> {
    (host, port)
        .to_socket_addrs()?
        .next()
        .ok_or_else(|| io::Error::other(format!("no address for {host}:{port}")))
}

impl StreamUpstream for PrimeTcpUpstream {
    type Conn = PrimeTcpConnection;

    fn poll_connect(&self, cx: &mut Context<'_>) -> Poll<io::Result<Self::Conn>> {
        let Ok(mut slot) = self.in_flight.lock() else {
            return Poll::Ready(Err(io::Error::other("PrimeTcpUpstream: lock poisoned")));
        };
        let pre_resolved = match &self.target {
            Target::Addr(addr) => Some(*addr),
            Target::Host { .. } => None,
        };
        let host_port = match &self.target {
            Target::Addr(_) => None,
            Target::Host { host, port } => Some((host.clone(), *port)),
        };
        let future = slot.get_or_insert_with(|| {
            Box::pin(async move {
                let addr = match (pre_resolved, host_port) {
                    (Some(addr), _) => addr,
                    (None, Some((host, port))) => resolve(&host, port)?,
                    (None, None) => return Err(io::Error::other("no dial target")),
                };
                let stream = TcpStream::connect(addr).await?;
                Ok(PrimeTcpConnection::new(stream, addr))
            })
        });
        match future.as_mut().poll(cx) {
            Poll::Ready(result) => {
                *slot = None;
                Poll::Ready(result)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use futures::io::{AsyncReadExt, AsyncWriteExt};
    use prime::os::core_shard;
    use proxima_runtime::CoreId;
    use proxima_primitives::stream::StreamUpstreamExt;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    /// full round-trip: prime listener (server) + PrimeTcpUpstream (client),
    /// both on the same prime worker. client sends 4 bytes, server echoes,
    /// client reads back.
    #[test]
    fn prime_tcp_upstream_connects_and_round_trips_bytes() {
        let handle = core_shard::launch_with_lanes(CoreId(0), None, 2, 16).expect("launch");
        let done = Arc::new(AtomicBool::new(false));
        let done_clone = done.clone();
        let addr_chan: Arc<Mutex<Option<SocketAddr>>> = Arc::new(Mutex::new(None));
        let addr_for_factory = addr_chan.clone();

        handle
            .dispatch_factory(Box::new(move || {
                let done = done_clone.clone();
                let addr_handle = addr_for_factory.clone();
                Box::pin(async move {
                    use prime::os::net::TcpListener;

                    let mut listener =
                        TcpListener::bind("127.0.0.1:0".parse().unwrap()).expect("bind");
                    let bound = listener.local_addr().expect("local_addr");
                    *addr_handle.lock().unwrap() = Some(bound);

                    let server = async move {
                        let (mut stream, _peer) = listener.accept().await.expect("accept");
                        let mut buf = [0u8; 4];
                        stream.read_exact(&mut buf).await.expect("server read");
                        stream.write_all(&buf).await.expect("server write");
                    };

                    let client = async move {
                        // small yield to let the server task start
                        let upstream = PrimeTcpUpstream::new(bound);
                        let mut conn = upstream.connect().await.expect("upstream connect");
                        conn.write_all(b"ping").await.expect("client write");
                        conn.flush().await.expect("client flush");
                        let mut reply = [0u8; 4];
                        conn.read_exact(&mut reply).await.expect("client read");
                        assert_eq!(&reply, b"ping");
                    };

                    futures::future::join(server, client).await;
                    done.store(true, Ordering::Release);
                }) as Pin<Box<dyn std::future::Future<Output = ()> + 'static>>
            }))
            .expect("dispatch_factory");

        // wait for the listener to bind so the client knows the port.
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        loop {
            if addr_chan.lock().unwrap().is_some() {
                break;
            }
            assert!(std::time::Instant::now() < deadline, "listener never bound");
            std::thread::sleep(Duration::from_millis(5));
        }

        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while !done.load(Ordering::Acquire) {
            assert!(
                std::time::Instant::now() < deadline,
                "round-trip never completed"
            );
            std::thread::sleep(Duration::from_millis(10));
        }

        handle.shutdown_and_join().expect("shutdown");
    }

    /// full round-trip through the acceptor abstraction: `PrimeAcceptorFactory`
    /// binds a listener, a server task drives `PrimeAcceptor::poll_accept` to
    /// accept one connection and echoes 4 bytes, and a `PrimeTcpUpstream`
    /// client writes "ping" and reads it back.
    #[test]
    fn prime_acceptor_factory_accepts_and_round_trips_bytes() {
        use core::future::poll_fn;

        let handle = core_shard::launch_with_lanes(CoreId(0), None, 2, 16).expect("launch");
        let done = Arc::new(AtomicBool::new(false));
        let done_clone = done.clone();
        let addr_chan: Arc<Mutex<Option<SocketAddr>>> = Arc::new(Mutex::new(None));
        let addr_for_factory = addr_chan.clone();

        handle
            .dispatch_factory(Box::new(move || {
                let done = done_clone.clone();
                let addr_handle = addr_for_factory.clone();
                Box::pin(async move {
                    let addr = "127.0.0.1:0".parse().expect("parse addr");
                    let mut acceptor = PrimeAcceptorFactory
                        .bind(addr, TcpBindOptions::default())
                        .expect("bind");
                    let bound = acceptor.local_addr().expect("local_addr");
                    *addr_handle.lock().expect("addr lock") = Some(bound);

                    let server = async move {
                        let mut conn = poll_fn(|cx| acceptor.poll_accept(cx))
                            .await
                            .expect("accept");
                        let mut buf = [0u8; 4];
                        conn.read_exact(&mut buf).await.expect("server read");
                        conn.write_all(&buf).await.expect("server write");
                    };

                    let client = async move {
                        let upstream = PrimeTcpUpstream::new(bound);
                        let mut conn = upstream.connect().await.expect("upstream connect");
                        conn.write_all(b"ping").await.expect("client write");
                        conn.flush().await.expect("client flush");
                        let mut reply = [0u8; 4];
                        conn.read_exact(&mut reply).await.expect("client read");
                        assert_eq!(&reply, b"ping");
                    };

                    futures::future::join(server, client).await;
                    done.store(true, Ordering::Release);
                }) as Pin<Box<dyn std::future::Future<Output = ()> + 'static>>
            }))
            .expect("dispatch_factory");

        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        loop {
            if addr_chan.lock().expect("addr lock").is_some() {
                break;
            }
            assert!(std::time::Instant::now() < deadline, "listener never bound");
            std::thread::sleep(Duration::from_millis(5));
        }

        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while !done.load(Ordering::Acquire) {
            assert!(
                std::time::Instant::now() < deadline,
                "acceptor round-trip never completed"
            );
            std::thread::sleep(Duration::from_millis(10));
        }

        handle.shutdown_and_join().expect("shutdown");
    }

    /// `with_host` must not touch the resolver or the network — building
    /// an upstream for a host that never resolves is fine; only `connect()`
    /// may fail. Proves DNS is deferred to connect time.
    #[test]
    fn with_host_defers_resolution_to_connect() {
        let upstream = PrimeTcpUpstream::with_host("definitely-not-a-real-host.invalid", 80);
        match &upstream.target {
            Target::Host { host, port } => {
                assert_eq!(host, "definitely-not-a-real-host.invalid");
                assert_eq!(*port, 80);
            }
            Target::Addr(_) => panic!("with_host should store a Host target, not a resolved addr"),
        }
    }

    /// connect to a port that has no listener — must return an error, not hang.
    #[test]
    fn prime_tcp_upstream_connect_refused_returns_error() {
        let handle = core_shard::launch_with_lanes(CoreId(0), None, 2, 16).expect("launch");
        let done = Arc::new(AtomicBool::new(false));
        let done_clone = done.clone();
        let result_chan: Arc<Mutex<Option<bool>>> = Arc::new(Mutex::new(None));
        let result_for_factory = result_chan.clone();

        // find a free port, then close it so nothing listens on it.
        let closed_port = {
            let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("temp bind");
            listener.local_addr().expect("local_addr").port()
        };
        let refused_addr: SocketAddr = format!("127.0.0.1:{closed_port}").parse().unwrap();

        handle
            .dispatch_factory(Box::new(move || {
                let done = done_clone.clone();
                let result_handle = result_for_factory.clone();
                Box::pin(async move {
                    let upstream = PrimeTcpUpstream::new(refused_addr);
                    let got_error = upstream.connect().await.is_err();
                    *result_handle.lock().unwrap() = Some(got_error);
                    done.store(true, Ordering::Release);
                }) as Pin<Box<dyn std::future::Future<Output = ()> + 'static>>
            }))
            .expect("dispatch_factory");

        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while !done.load(Ordering::Acquire) {
            assert!(
                std::time::Instant::now() < deadline,
                "connect-refused test never completed (possible hang)"
            );
            std::thread::sleep(Duration::from_millis(10));
        }

        handle.shutdown_and_join().expect("shutdown");

        let got_error = result_chan.lock().unwrap().expect("result not set");
        assert!(got_error, "expected an error on refused connect, got Ok");
    }
}
