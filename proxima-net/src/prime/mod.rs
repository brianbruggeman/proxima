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
//! std-only by construction (it wraps prime's std-gated
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
use std::sync::OnceLock;
use std::task::{Context, Poll};

use futures::io::{AsyncRead, AsyncWrite};
use prime::os::background::ProximaBackgroundPool;
use prime::os::net::{TcpListener, TcpStream, UdpSocket};
use proxima_primitives::pipe::ProximaError;
use proxima_primitives::stream::{
    AcceptorFactory, ConnectFuture, DatagramFactory, DatagramSocket, PeerInfo, StreamConnection,
    StreamUpstream, TcpAcceptor, TcpBindOptions,
};

mod connect_tunnel;
pub use connect_tunnel::{ConnectTunnelConnection, ConnectTunneledUpstream};

#[cfg(unix)]
mod unix;
#[cfg(unix)]
pub use unix::{
    PrimeUnixConnection, PrimeUnixListener, PrimeUnixUpstream, PrimeUnixUpstreamFactory,
};

mod packet;
pub use packet::{PrimePacketListenerFactory, PrimeUdpListener};

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

    fn backend_name(&self) -> &'static str {
        "prime"
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

/// each call owns its resolver completion and reactor connection state.
pub struct PrimeTcpUpstream {
    target: Target,
}

impl PrimeTcpUpstream {
    pub fn new(addr: SocketAddr) -> Self {
        Self {
            target: Target::Addr(addr),
        }
    }

    /// construction performs no DNS or socket operations.
    pub fn with_host(host: impl Into<String>, port: u16) -> Self {
        Self {
            target: Target::Host {
                host: host.into(),
                port,
            },
        }
    }

    pub async fn connect(&self) -> io::Result<PrimeTcpConnection> {
        match &self.target {
            Target::Addr(address) => connect_addresses(core::iter::once(*address)).await,
            Target::Host { host, port } => {
                let host = host.clone();
                let port = *port;
                let addresses = resolve_on_pool(move || {
                    (host.as_str(), port)
                        .to_socket_addrs()
                        .map(Iterator::collect)
                })
                .await?;
                connect_addresses(addresses).await
            }
        }
    }

    pub fn boxed(
        addr: SocketAddr,
    ) -> std::sync::Arc<dyn StreamUpstream<Conn = Box<dyn StreamConnection>>> {
        std::sync::Arc::new(BoxedPrimeTcpUpstream(Self::new(addr)))
    }
}

struct BoxedPrimeTcpUpstream(PrimeTcpUpstream);

impl StreamUpstream for BoxedPrimeTcpUpstream {
    type Conn = Box<dyn StreamConnection>;

    fn connect_future(&self) -> ConnectFuture<'_, Self::Conn> {
        Box::pin(async move { Ok(Box::new(self.0.connect().await?) as Box<dyn StreamConnection>) })
    }
}

// pool ownership outlives cancelled calls so dropping a future cannot join getaddrinfo.
static RESOLVER_POOL: OnceLock<Result<ProximaBackgroundPool, String>> = OnceLock::new();

async fn resolve_on_pool(
    work: impl FnOnce() -> io::Result<Vec<SocketAddr>> + Send + 'static,
) -> io::Result<Vec<SocketAddr>> {
    let pool = RESOLVER_POOL
        .get_or_init(|| ProximaBackgroundPool::new().map_err(|error| error.to_string()));
    let pool = pool
        .as_ref()
        .map_err(|message| io::Error::other(message.clone()))?;
    pool.spawn(move || work().map_err(ProximaError::Io))
        .await
        .map_err(|error| match error {
            ProximaError::Io(error) => error,
            other => io::Error::other(other),
        })
}

async fn connect_addresses(
    addresses: impl IntoIterator<Item = SocketAddr>,
) -> io::Result<PrimeTcpConnection> {
    let mut last_error = None;
    for address in addresses {
        match TcpStream::connect(address).await {
            Ok(stream) => return Ok(PrimeTcpConnection::new(stream, address)),
            Err(error) => last_error = Some(error),
        }
    }
    Err(match last_error {
        Some(error) => error,
        None => io::Error::new(
            io::ErrorKind::InvalidInput,
            "resolver returned no addresses",
        ),
    })
}

impl StreamUpstream for PrimeTcpUpstream {
    type Conn = PrimeTcpConnection;

    fn connect_future(&self) -> ConnectFuture<'_, Self::Conn> {
        Box::pin(self.connect())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use futures::io::{AsyncReadExt, AsyncWriteExt};
    use prime::os::core_shard;
    use proxima_runtime::CoreId;
    use std::future::{Future, poll_fn};
    use std::io::{Read, Write};
    use std::net::{TcpListener as StdTcpListener, TcpStream as StdTcpStream};
    use std::pin::pin;
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;

    // hang guard only: the worker signals completion on the channel, so the
    // test blocks on the event itself rather than polling a flag. no sleep.
    const RESULT_TIMEOUT: Duration = Duration::from_secs(5);

    #[test]
    fn upstream_address_fallback_preserves_payload_and_errors() {
        let refused: SocketAddr = "127.0.0.1:0".parse().expect("parse refused address");
        let oracle = StdTcpStream::connect(refused).expect_err("port zero cannot listen");
        let expected_error = (oracle.kind(), oracle.raw_os_error());
        let listener = StdTcpListener::bind("127.0.0.1:0").expect("bind fallback peer");
        let live = listener.local_addr().expect("fallback address");
        let worker =
            core_shard::launch_with_lanes(CoreId(0), None, 2, 16).expect("launch fallback worker");
        let (completed, result) = mpsc::channel();
        worker
            .dispatch_send_inline(async move {
                let mut connection = connect_addresses([refused, live])
                    .await
                    .expect("try live address after refused address");
                assert!(
                    matches!(connection.peer(), Some(PeerInfo::Tcp(address)) if address == live)
                );
                connection
                    .write_all(b"fallback-request")
                    .await
                    .expect("write fallback request");
                let mut response = [0; 17];
                connection
                    .read_exact(&mut response)
                    .await
                    .expect("read fallback response");
                assert_eq!(&response, b"fallback-response");
                let error = connect_addresses([refused, refused])
                    .await
                    .err()
                    .expect("all addresses fail");
                assert_eq!((error.kind(), error.raw_os_error()), expected_error);
                let empty = connect_addresses([])
                    .await
                    .err()
                    .expect("empty address set fails");
                assert_eq!(empty.kind(), io::ErrorKind::InvalidInput);
                completed.send(()).expect("report fallback result");
            })
            .expect("dispatch fallback dial");
        let (mut peer, _) = listener.accept().expect("accept fallback dial");
        peer.set_read_timeout(Some(RESULT_TIMEOUT))
            .expect("set peer read guard");
        peer.set_write_timeout(Some(RESULT_TIMEOUT))
            .expect("set peer write guard");
        let mut request = [0; 16];
        peer.read_exact(&mut request)
            .expect("read fallback request");
        assert_eq!(&request, b"fallback-request");
        peer.write_all(b"fallback-response")
            .expect("send fallback response");
        result
            .recv_timeout(RESULT_TIMEOUT)
            .expect("fallback assertions complete");
        worker.shutdown_and_join().expect("join fallback worker");
    }

    #[test]
    fn upstream_resolver_yields_and_cancels_without_waiting_for_work() {
        for cancel in [false, true] {
            let worker = core_shard::launch_with_lanes(CoreId(0), None, 2, 16)
                .expect("launch resolver worker");
            let (entered, entry) = mpsc::channel();
            let (release, released) = mpsc::channel();
            let (progressed, progress) = mpsc::channel();
            let (completed, result) = mpsc::channel();
            let (finished, finish) = mpsc::channel();
            worker
                .dispatch_send_inline(async move {
                    let worker_thread = thread::current().id();
                    let mut operation = pin!(resolve_on_pool(move || {
                        entered
                            .send(thread::current().id())
                            .expect("report resolver thread");
                        released.recv().expect("hold controlled resolver work");
                        finished
                            .send(())
                            .expect("report resolver closure completed");
                        Ok(vec![
                            "127.0.0.1:12345".parse().expect("parse resolved address"),
                        ])
                    }));
                    let initial =
                        poll_fn(|context| Poll::Ready(operation.as_mut().poll(context))).await;
                    assert!(initial.is_pending(), "resolver remains held by caller");
                    progressed
                        .send(worker_thread)
                        .expect("report reactor progress while resolver held");
                    if cancel {
                        completed
                            .send(None)
                            .expect("report cancelled resolver call");
                    } else {
                        let addresses = operation.await.expect("resume resolver receiver");
                        completed
                            .send(Some(addresses))
                            .expect("report resolved address payload");
                    }
                })
                .expect("dispatch controlled resolver");
            let resolver_thread = entry
                .recv_timeout(RESULT_TIMEOUT)
                .expect("resolver entered");
            let worker_thread = progress
                .recv_timeout(RESULT_TIMEOUT)
                .expect("reactor yields while resolver held");
            assert_ne!(worker_thread, resolver_thread);
            let (other_completed, other_result) = mpsc::channel();
            worker
                .dispatch_send_inline(async move {
                    other_completed
                        .send(thread::current().id())
                        .expect("report unrelated task progress");
                })
                .expect("dispatch unrelated task");
            assert_eq!(
                other_result
                    .recv_timeout(RESULT_TIMEOUT)
                    .expect("unrelated task runs before resolver release"),
                worker_thread
            );
            if cancel {
                assert_eq!(
                    result
                        .recv_timeout(RESULT_TIMEOUT)
                        .expect("cancel without releasing resolver"),
                    None
                );
                // joining the worker proves the receiver's destructor cannot wait for DNS.
                worker
                    .shutdown_and_join()
                    .expect("join worker while resolver still held");
                release.send(()).expect("release cancelled resolver work");
            } else {
                release.send(()).expect("release resolver work");
                assert_eq!(
                    result
                        .recv_timeout(RESULT_TIMEOUT)
                        .expect("resolved payload delivered"),
                    Some(vec!["127.0.0.1:12345".parse().expect("expected address")])
                );
                worker.shutdown_and_join().expect("join resolver worker");
            }
            finish
                .recv_timeout(RESULT_TIMEOUT)
                .expect("resolver closure released");
        }
        let directory = tempfile::tempdir().expect("resolver error oracle directory");
        let missing = directory.path().join("missing");
        let oracle = std::fs::read(&missing).expect_err("missing file gives independent OS error");
        let expected = (oracle.kind(), oracle.raw_os_error());
        let actual = futures::executor::block_on(resolve_on_pool(move || {
            std::fs::read(missing).map(|_| Vec::new())
        }))
        .expect_err("resolver preserves work error");
        assert_eq!((actual.kind(), actual.raw_os_error()), expected);
        let invalid_host = "invalid\0hostname";
        let oracle = (invalid_host, 80)
            .to_socket_addrs()
            .expect_err("embedded NUL rejects DNS input");
        let upstream = PrimeTcpUpstream::with_host(invalid_host, 80);
        let actual = futures::executor::block_on(upstream.connect())
            .err()
            .expect("upstream retains resolver error before socket creation");
        assert_eq!(
            (actual.kind(), actual.raw_os_error()),
            (oracle.kind(), oracle.raw_os_error())
        );
    }

    /// full round-trip: prime listener (server) + PrimeTcpUpstream (client),
    /// both on the same prime worker. client sends 4 bytes, server echoes,
    /// client reads back.
    #[test]
    fn prime_tcp_upstream_connects_and_round_trips_bytes() {
        let handle = core_shard::launch_with_lanes(CoreId(0), None, 2, 16).expect("launch");
        let (done_tx, done_rx) = mpsc::channel::<()>();

        handle
            .dispatch_factory(Box::new(move || {
                Box::pin(async move {
                    use prime::os::net::TcpListener;

                    let mut listener =
                        TcpListener::bind("127.0.0.1:0".parse().unwrap()).expect("bind");
                    let bound = listener.local_addr().expect("local_addr");

                    let server = async move {
                        let (mut stream, _peer) = listener.accept().await.expect("accept");
                        let mut buf = [0u8; 4];
                        stream.read_exact(&mut buf).await.expect("server read");
                        stream.write_all(&buf).await.expect("server write");
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
                    let _ = done_tx.send(());
                }) as Pin<Box<dyn std::future::Future<Output = ()> + 'static>>
            }))
            .expect("dispatch_factory");

        done_rx
            .recv_timeout(RESULT_TIMEOUT)
            .expect("round-trip never completed");
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
        let (done_tx, done_rx) = mpsc::channel::<()>();

        handle
            .dispatch_factory(Box::new(move || {
                Box::pin(async move {
                    let addr = "127.0.0.1:0".parse().expect("parse addr");
                    let mut acceptor = PrimeAcceptorFactory
                        .bind(addr, TcpBindOptions::default())
                        .expect("bind");
                    let bound = acceptor.local_addr().expect("local_addr");

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
                    let _ = done_tx.send(());
                }) as Pin<Box<dyn std::future::Future<Output = ()> + 'static>>
            }))
            .expect("dispatch_factory");

        done_rx
            .recv_timeout(RESULT_TIMEOUT)
            .expect("acceptor round-trip never completed");
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
        let (result_tx, result_rx) = mpsc::channel::<bool>();

        // find a free port, then close it so nothing listens on it.
        let closed_port = {
            let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("temp bind");
            listener.local_addr().expect("local_addr").port()
        };
        let refused_addr: SocketAddr = format!("127.0.0.1:{closed_port}").parse().unwrap();

        handle
            .dispatch_factory(Box::new(move || {
                Box::pin(async move {
                    let upstream = PrimeTcpUpstream::new(refused_addr);
                    let _ = result_tx.send(upstream.connect().await.is_err());
                }) as Pin<Box<dyn std::future::Future<Output = ()> + 'static>>
            }))
            .expect("dispatch_factory");

        let got_error = result_rx
            .recv_timeout(RESULT_TIMEOUT)
            .expect("connect-refused test never completed (possible hang)");
        handle.shutdown_and_join().expect("shutdown");

        assert!(got_error, "expected an error on refused connect, got Ok");
    }

    /// full round-trip through `PrimeDatagramFactory`/`PrimeDatagram`: bind
    /// two sockets via the `DatagramFactory` seam, client sends, server
    /// receives — proves the `ServeContext::datagram_factory` injection
    /// point actually produces a working prime UDP socket, not just that
    /// `prime::os::net::UdpSocket` itself works (already covered by
    /// `PrimeUdpListener`'s tests in `packet.rs`).
    #[test]
    fn prime_datagram_factory_binds_and_round_trips_a_datagram() {
        use core::future::poll_fn;

        let handle = core_shard::launch_with_lanes(CoreId(0), None, 2, 16).expect("launch");
        let (result_tx, result_rx) = mpsc::channel::<Vec<u8>>();

        handle
            .dispatch_factory(Box::new(move || {
                Box::pin(async move {
                    let factory = PrimeDatagramFactory;
                    assert_eq!(factory.backend_name(), "prime");

                    let mut server = factory
                        .bind(SocketAddr::from((std::net::Ipv4Addr::LOCALHOST, 0)))
                        .expect("bind server");
                    let server_addr = server.local_addr().expect("server addr");
                    let mut client = factory
                        .bind(SocketAddr::from((std::net::Ipv4Addr::LOCALHOST, 0)))
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
                    let _ = result_tx.send(buf[..len].to_vec());
                }) as Pin<Box<dyn std::future::Future<Output = ()> + 'static>>
            }))
            .expect("dispatch_factory");

        let received = result_rx
            .recv_timeout(RESULT_TIMEOUT)
            .expect("datagram factory round-trip never completed");
        handle.shutdown_and_join().expect("shutdown");

        assert_eq!(&received[..], b"ping");
    }
}
