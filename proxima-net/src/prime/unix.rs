//! prime-backed `StreamListener`/`StreamUpstream` for Unix-domain sockets.
//! Mirrors `proxima-net-tokio`'s `TokioUnixListener`/`TokioUnixUpstream` in
//! shape and behavior (P14 — the tokio impls are the oracle: same accept
//! semantics, same error mapping, same peer-addr handling, same fresh-bind
//! unlink-on-bind / cleanup-on-drop) but uses the prime reactor
//! (`prime::os::net::{UnixListener, UnixStream}`) instead of tokio — zero
//! tokio dependency.
//!
//! `AcceptorFactory` is TCP-only (`bind(addr: SocketAddr, ..)`), so — same
//! as the tokio side — Unix sockets implement `StreamListener`/
//! `StreamUpstream` directly rather than going through the acceptor-factory
//! seam.

use std::io;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Mutex;
use std::task::{Context, Poll};

use futures::io::{AsyncRead, AsyncWrite};
use prime::os::net::{UnixListener as PrimeUnixListenerInner, UnixStream};
use proxima_primitives::stream::{
    BindAddr, PeerInfo, StreamConnection, StreamListener, StreamUpstream, UnixUpstreamFactory,
};

type ConnectFuture =
    Pin<Box<dyn std::future::Future<Output = io::Result<PrimeUnixConnection>> + Send>>;

/// prime-backed Unix-domain connection. wraps `prime::os::net::UnixStream`
/// and carries the peer path (usually `None` — the common case is an
/// anonymous, unbound client socket) so `StreamConnection::peer()` is
/// satisfied.
pub struct PrimeUnixConnection {
    inner: UnixStream,
    peer: Option<PathBuf>,
}

impl PrimeUnixConnection {
    fn new(stream: UnixStream, peer: Option<PathBuf>) -> Self {
        Self {
            inner: stream,
            peer,
        }
    }
}

impl AsyncRead for PrimeUnixConnection {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut [u8],
    ) -> Poll<io::Result<usize>> {
        Pin::new(&mut self.get_mut().inner).poll_read(cx, buf)
    }
}

impl AsyncWrite for PrimeUnixConnection {
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

impl StreamConnection for PrimeUnixConnection {
    fn peer(&self) -> Option<PeerInfo> {
        Some(PeerInfo::Unix(self.peer.clone()))
    }
}

/// prime-backed Unix-domain `StreamListener`. binds via
/// `prime::os::net::UnixListener::bind`, which unlinks a stale socket file
/// at `path` first — same fresh-bind-on-restart behavior as
/// `TokioUnixListener::bind`.
///
/// WHY the `Mutex`: `StreamListener::poll_accept` takes `&self` (the trait
/// surface cannot allow `&mut self`, same constraint documented on
/// `PrimeTcpUpstream::in_flight`), but the prime OS-layer
/// `UnixListener::poll_accept` takes `Pin<&mut Self>` — interior mutability
/// is required to get there. One accept loop per listener means the lock
/// is uncontested in practice; cost is a single futex-free fast-path lock
/// per accept, dwarfed by the syscall itself.
pub struct PrimeUnixListener {
    inner: Mutex<PrimeUnixListenerInner>,
    bind_path: PathBuf,
}

impl PrimeUnixListener {
    /// bind a prime-backed Unix listener at `path`. must be called on a
    /// proxima worker thread (CURRENT_REACTOR must be live).
    pub fn bind(path: PathBuf) -> io::Result<Self> {
        let inner = PrimeUnixListenerInner::bind(path.clone())?;
        Ok(Self {
            inner: Mutex::new(inner),
            bind_path: path,
        })
    }
}

impl StreamListener for PrimeUnixListener {
    type Conn = PrimeUnixConnection;

    fn poll_accept(&self, cx: &mut Context<'_>) -> Poll<io::Result<Self::Conn>> {
        let Ok(mut guard) = self.inner.lock() else {
            return Poll::Ready(Err(io::Error::other("PrimeUnixListener: lock poisoned")));
        };
        match Pin::new(&mut *guard).poll_accept(cx) {
            Poll::Ready(Ok((stream, peer))) => {
                Poll::Ready(Ok(PrimeUnixConnection::new(stream, peer)))
            }
            Poll::Ready(Err(err)) => Poll::Ready(Err(err)),
            Poll::Pending => Poll::Pending,
        }
    }

    fn local_addr(&self) -> Option<BindAddr> {
        Some(BindAddr::Unix(self.bind_path.clone()))
    }
}

/// prime-backed Unix-domain `StreamUpstream`. dials the target path via the
/// prime reactor on each `connect()` call; the in-flight future is cached
/// across polls so a pending connect can resume — same pattern as
/// `PrimeTcpUpstream`/`TokioUnixUpstream`.
pub struct PrimeUnixUpstream {
    path: PathBuf,
    in_flight: Mutex<Option<ConnectFuture>>,
}

impl PrimeUnixUpstream {
    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            in_flight: Mutex::new(None),
        }
    }
}

impl StreamUpstream for PrimeUnixUpstream {
    type Conn = PrimeUnixConnection;

    fn poll_connect(&self, cx: &mut Context<'_>) -> Poll<io::Result<Self::Conn>> {
        let path = self.path.clone();
        let Ok(mut slot) = self.in_flight.lock() else {
            return Poll::Ready(Err(io::Error::other("PrimeUnixUpstream: lock poisoned")));
        };
        let future = slot.get_or_insert_with(|| {
            Box::pin(async move {
                let stream = UnixStream::connect(&path).await?;
                // On the CLIENT side, `getpeername(2)` on a connected
                // AF_UNIX stream returns the address it dialed (the
                // server's bound path) — unlike the accept side, where the
                // peer is usually an anonymous, unbound client socket.
                // Mirrors `TokioUnixConnection::new`'s `stream.peer_addr()`
                // read on the upstream side, which observes the same path.
                Ok(PrimeUnixConnection::new(stream, Some(path)))
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

/// Type-erases `PrimeUnixUpstream::Conn` to `Box<dyn StreamConnection>` so
/// `RuntimeSelection` can hold either backend's unix-upstream factory behind
/// one field — the same erasure `Box<dyn StreamConnection>`'s own
/// `StreamConnection` impl (`proxima_primitives::stream`) already
/// established, one level up.
struct BoxedPrimeUnixUpstream(PrimeUnixUpstream);

impl StreamUpstream for BoxedPrimeUnixUpstream {
    type Conn = Box<dyn StreamConnection>;

    fn poll_connect(&self, cx: &mut Context<'_>) -> Poll<io::Result<Self::Conn>> {
        match self.0.poll_connect(cx) {
            Poll::Ready(Ok(conn)) => Poll::Ready(Ok(Box::new(conn))),
            Poll::Ready(Err(err)) => Poll::Ready(Err(err)),
            Poll::Pending => Poll::Pending,
        }
    }
}

/// prime-backed [`UnixUpstreamFactory`] — the runtime-selectable entry point
/// `RuntimeSelection::prime()` bundles.
pub struct PrimeUnixUpstreamFactory;

impl UnixUpstreamFactory for PrimeUnixUpstreamFactory {
    fn connect(
        &self,
        path: PathBuf,
    ) -> std::sync::Arc<dyn StreamUpstream<Conn = Box<dyn StreamConnection>>> {
        std::sync::Arc::new(BoxedPrimeUnixUpstream(PrimeUnixUpstream::new(path)))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use futures::io::{AsyncReadExt, AsyncWriteExt};
    use prime::os::core_shard;
    use proxima_primitives::stream::{StreamListenerExt, StreamUpstreamExt};
    use proxima_runtime::CoreId;
    use std::sync::mpsc;
    use std::time::Duration;

    // hang guard only: the worker signals completion on the channel, so the
    // test blocks on the event itself rather than polling a flag. no sleep.
    const RESULT_TIMEOUT: Duration = Duration::from_secs(5);

    /// full round-trip: prime `PrimeUnixListener` (server) +
    /// `PrimeUnixUpstream` (client), both on the same prime worker. client
    /// sends 4 bytes, server echoes, client reads back. Mirrors
    /// `TokioUnixListener`'s `unix_listener_round_trips_a_few_bytes` test —
    /// same shape, prime backend.
    #[test]
    fn prime_unix_upstream_connects_and_round_trips_bytes() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let socket_path = temp_dir.path().join("proxima-net-prime-unix.sock");

        let handle = core_shard::launch_with_lanes(CoreId(0), None, 2, 16).expect("launch");
        let (done_tx, done_rx) = mpsc::channel::<()>();
        let path_for_factory = socket_path.clone();

        handle
            .dispatch_factory(Box::new(move || {
                let path = path_for_factory;
                Box::pin(async move {
                    // bind before the client task exists, so the socket file is
                    // already there when it dials — no wait-for-bind poll.
                    let listener = PrimeUnixListener::bind(path.clone()).expect("bind");

                    let server = async move {
                        let mut conn = listener.accept().await.expect("accept");
                        let mut buf = [0u8; 4];
                        conn.read_exact(&mut buf).await.expect("server read");
                        conn.write_all(&buf).await.expect("server write");
                    };

                    let client = async move {
                        let upstream = PrimeUnixUpstream::new(path.clone());
                        let mut conn = upstream.connect().await.expect("upstream connect");
                        // parity with `TokioUnixUpstream`: the client side's
                        // peer is the dialed server path, not `None`.
                        match conn.peer() {
                            Some(PeerInfo::Unix(Some(peer_path))) => {
                                assert_eq!(peer_path, path);
                            }
                            other => panic!("expected client peer = dialed path, got {other:?}"),
                        }
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

    /// connect to a path with no listener — must return an error, not hang.
    #[test]
    fn prime_unix_upstream_connect_missing_path_returns_error() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let missing_path = temp_dir.path().join("nobody-listening.sock");

        let handle = core_shard::launch_with_lanes(CoreId(0), None, 2, 16).expect("launch");
        let (result_tx, result_rx) = mpsc::channel::<bool>();
        let path_for_factory = missing_path.clone();

        handle
            .dispatch_factory(Box::new(move || {
                let path = path_for_factory;
                Box::pin(async move {
                    let upstream = PrimeUnixUpstream::new(path);
                    let _ = result_tx.send(upstream.connect().await.is_err());
                }) as Pin<Box<dyn std::future::Future<Output = ()> + 'static>>
            }))
            .expect("dispatch_factory");

        let got_error = result_rx
            .recv_timeout(RESULT_TIMEOUT)
            .expect("connect-missing-path test never completed (possible hang)");
        handle.shutdown_and_join().expect("shutdown");

        assert!(
            got_error,
            "expected an error connecting to a missing path, got Ok"
        );
    }

    /// `PrimeUnixListener::local_addr()` reports the bind path — proves the
    /// `BindAddr::Unix` variant carries the path, same as
    /// `TokioUnixListener::local_addr()`.
    #[test]
    fn prime_unix_listener_local_addr_reports_bind_path() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let socket_path = temp_dir.path().join("local-addr.sock");

        let handle = core_shard::launch_with_lanes(CoreId(0), None, 2, 16).expect("launch");
        let (result_tx, result_rx) = mpsc::channel::<PathBuf>();
        let path_for_factory = socket_path.clone();

        handle
            .dispatch_factory(Box::new(move || {
                let path = path_for_factory;
                Box::pin(async move {
                    let listener = PrimeUnixListener::bind(path).expect("bind");
                    let reported = match listener.local_addr() {
                        Some(BindAddr::Unix(reported_path)) => reported_path,
                        other => panic!("expected BindAddr::Unix, got {other:?}"),
                    };
                    let _ = result_tx.send(reported);
                }) as Pin<Box<dyn std::future::Future<Output = ()> + 'static>>
            }))
            .expect("dispatch_factory");

        let reported = result_rx
            .recv_timeout(RESULT_TIMEOUT)
            .expect("local-addr test never completed");
        handle.shutdown_and_join().expect("shutdown");

        assert_eq!(reported, socket_path);
    }
}
