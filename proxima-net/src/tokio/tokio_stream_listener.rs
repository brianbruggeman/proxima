//! Tokio TCP/Unix `StreamListener` impls. `tokio-util::compat`
//! adapts tokio's AsyncRead/AsyncWrite to the futures-io traits.

use std::io;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Mutex;
use std::task::{Context, Poll};

use tokio::net::TcpListener as TokioTcpListenerInner;
use tokio_util::compat::{Compat, TokioAsyncReadCompatExt};

use proxima_primitives::stream::{BindAddr, PeerInfo, StreamConnection, StreamListener};

#[cfg(unix)]
use tokio::net::UnixListener as TokioUnixListenerInner;

/// TCP connection wrapper — tokio's `TcpStream` adapted via the
/// `tokio-util::compat` shim so it satisfies `futures::io::AsyncRead +
/// AsyncWrite + Send + Unpin`.
pub struct TokioTcpConnection {
    inner: Compat<tokio::net::TcpStream>,
    peer: Option<SocketAddr>,
}

impl TokioTcpConnection {
    fn new(stream: tokio::net::TcpStream) -> Self {
        let peer = stream.peer_addr().ok();
        Self {
            inner: stream.compat(),
            peer,
        }
    }
}

/// Crate-private factory used by the upstream module to wrap a freshly
/// connected `TcpStream` without exposing `TokioTcpConnection::new`.
pub fn tcp_connection_from_stream(stream: tokio::net::TcpStream) -> TokioTcpConnection {
    TokioTcpConnection::new(stream)
}

impl TokioTcpConnection {
    // public constructor for consumers that bind via proxima but
    // accept tokio-native (e.g. to set socket options like nodelay
    // before reshaping into futures-io). pairs with
    // `TokioTcpListener::accept_tokio`.
    pub fn from_tokio(stream: tokio::net::TcpStream) -> Self {
        Self::new(stream)
    }
}

impl futures::io::AsyncRead for TokioTcpConnection {
    fn poll_read(
        self: std::pin::Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut [u8],
    ) -> Poll<io::Result<usize>> {
        std::pin::Pin::new(&mut self.get_mut().inner).poll_read(cx, buf)
    }
}

impl futures::io::AsyncWrite for TokioTcpConnection {
    fn poll_write(
        self: std::pin::Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        std::pin::Pin::new(&mut self.get_mut().inner).poll_write(cx, buf)
    }

    fn poll_flush(self: std::pin::Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        std::pin::Pin::new(&mut self.get_mut().inner).poll_flush(cx)
    }

    fn poll_close(self: std::pin::Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        std::pin::Pin::new(&mut self.get_mut().inner).poll_close(cx)
    }
}

impl StreamConnection for TokioTcpConnection {
    fn peer(&self) -> Option<PeerInfo> {
        self.peer.map(PeerInfo::Tcp)
    }
}

/// Tokio-backed TCP `StreamListener`.
pub struct TokioTcpListener {
    inner: TokioTcpListenerInner,
    local_addr: Option<SocketAddr>,
}

impl TokioTcpListener {
    /// Bind a tokio TCP listener at `addr`. Async because the bind
    /// itself is async; once bound the listener is sync-pollable.
    pub async fn bind(addr: SocketAddr) -> io::Result<Self> {
        let inner = TokioTcpListenerInner::bind(addr).await?;
        let local_addr = inner.local_addr().ok();
        Ok(Self { inner, local_addr })
    }

    /// Wrap a pre-bound tokio listener — e.g. a SO_REUSEPORT socket from
    /// `proxima_listen::handle::bind_reuseport_listener`, so each per-core
    /// accept lane owns its own listener on the shared `(addr, port)`.
    #[must_use]
    pub fn from_tokio_listener(inner: TokioTcpListenerInner) -> Self {
        let local_addr = inner.local_addr().ok();
        Self { inner, local_addr }
    }

    // proxima sits under transport-shaped consumers (pgwire, axum, raw
    // tcp) that already expect a tokio TcpStream. expose the raw
    // accept so the substrate owns the bind without forcing every
    // consumer onto the futures-io adapter.
    pub async fn accept_tokio(&self) -> io::Result<(tokio::net::TcpStream, SocketAddr)> {
        self.inner.accept().await
    }

    pub fn into_inner(self) -> TokioTcpListenerInner {
        self.inner
    }

    pub fn as_tokio(&self) -> &TokioTcpListenerInner {
        &self.inner
    }
}

impl StreamListener for TokioTcpListener {
    type Conn = TokioTcpConnection;

    fn poll_accept(&self, cx: &mut Context<'_>) -> Poll<io::Result<Self::Conn>> {
        match self.inner.poll_accept(cx) {
            Poll::Ready(Ok((stream, _peer))) => Poll::Ready(Ok(TokioTcpConnection::new(stream))),
            Poll::Ready(Err(err)) => Poll::Ready(Err(err)),
            Poll::Pending => Poll::Pending,
        }
    }

    fn local_addr(&self) -> Option<BindAddr> {
        self.local_addr.map(BindAddr::Tcp)
    }
}

#[cfg(unix)]
pub struct TokioUnixConnection {
    inner: Compat<tokio::net::UnixStream>,
    peer: Option<PathBuf>,
}

#[cfg(unix)]
impl TokioUnixConnection {
    fn new(stream: tokio::net::UnixStream) -> Self {
        // tokio's UnixStream::peer_addr returns SocketAddr (uds-shaped);
        // converting to PathBuf is best-effort for unnamed sockets.
        let peer = stream
            .peer_addr()
            .ok()
            .and_then(|addr| addr.as_pathname().map(PathBuf::from));
        Self {
            inner: stream.compat(),
            peer,
        }
    }
}

#[cfg(unix)]
pub fn unix_connection_from_stream(stream: tokio::net::UnixStream) -> TokioUnixConnection {
    TokioUnixConnection::new(stream)
}

#[cfg(unix)]
impl futures::io::AsyncRead for TokioUnixConnection {
    fn poll_read(
        self: std::pin::Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut [u8],
    ) -> Poll<io::Result<usize>> {
        std::pin::Pin::new(&mut self.get_mut().inner).poll_read(cx, buf)
    }
}

#[cfg(unix)]
impl futures::io::AsyncWrite for TokioUnixConnection {
    fn poll_write(
        self: std::pin::Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        std::pin::Pin::new(&mut self.get_mut().inner).poll_write(cx, buf)
    }

    fn poll_flush(self: std::pin::Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        std::pin::Pin::new(&mut self.get_mut().inner).poll_flush(cx)
    }

    fn poll_close(self: std::pin::Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        std::pin::Pin::new(&mut self.get_mut().inner).poll_close(cx)
    }
}

#[cfg(unix)]
impl StreamConnection for TokioUnixConnection {
    fn peer(&self) -> Option<PeerInfo> {
        Some(PeerInfo::Unix(self.peer.clone()))
    }
}

#[cfg(unix)]
pub struct TokioUnixListener {
    inner: TokioUnixListenerInner,
    bind_path: PathBuf,
    cleanup_on_drop: Mutex<bool>,
}

#[cfg(unix)]
impl TokioUnixListener {
    /// Bind a tokio Unix listener at `path`. If the socket file
    /// already exists at `path`, remove it first — matches the typical
    /// local-daemon fresh-bind expectation on restart.
    pub async fn bind(path: PathBuf) -> io::Result<Self> {
        if path.exists() {
            std::fs::remove_file(&path)?;
        }
        let inner = TokioUnixListenerInner::bind(&path)?;
        Ok(Self {
            inner,
            bind_path: path,
            cleanup_on_drop: Mutex::new(true),
        })
    }

    pub async fn accept_tokio(
        &self,
    ) -> io::Result<(tokio::net::UnixStream, tokio::net::unix::SocketAddr)> {
        self.inner.accept().await
    }

    pub fn as_tokio(&self) -> &TokioUnixListenerInner {
        &self.inner
    }
}

#[cfg(unix)]
impl Drop for TokioUnixListener {
    fn drop(&mut self) {
        let cleanup = self
            .cleanup_on_drop
            .lock()
            .map(|guard| *guard)
            .unwrap_or(false);
        if cleanup {
            let _ = std::fs::remove_file(&self.bind_path);
        }
    }
}

#[cfg(unix)]
impl StreamListener for TokioUnixListener {
    type Conn = TokioUnixConnection;

    fn poll_accept(&self, cx: &mut Context<'_>) -> Poll<io::Result<Self::Conn>> {
        match self.inner.poll_accept(cx) {
            Poll::Ready(Ok((stream, _peer))) => Poll::Ready(Ok(TokioUnixConnection::new(stream))),
            Poll::Ready(Err(err)) => Poll::Ready(Err(err)),
            Poll::Pending => Poll::Pending,
        }
    }

    fn local_addr(&self) -> Option<BindAddr> {
        Some(BindAddr::Unix(self.bind_path.clone()))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use futures::io::{AsyncReadExt, AsyncWriteExt};
    use proxima_primitives::stream::{StreamListener, StreamListenerExt};
    use std::net::Ipv4Addr;
    use tokio::io::{AsyncReadExt as TokioAsyncReadExt, AsyncWriteExt as TokioAsyncWriteExt};

    #[proxima::test]
    async fn tcp_listener_round_trips_a_few_bytes() {
        let bind = SocketAddr::from((Ipv4Addr::LOCALHOST, 0));
        let listener = TokioTcpListener::bind(bind).await.expect("bind");
        let local = match listener.local_addr().expect("local_addr") {
            BindAddr::Tcp(addr) => addr,
            _ => panic!("expected tcp bind"),
        };

        let server = tokio::spawn(async move {
            let mut conn = listener.accept().await.expect("accept");
            let mut buf = [0_u8; 5];
            conn.read_exact(&mut buf).await.expect("server read");
            conn.write_all(b"world").await.expect("server write");
            conn.flush().await.expect("flush");
            buf
        });

        let mut client = tokio::net::TcpStream::connect(local)
            .await
            .expect("client connect");
        client.write_all(b"hello").await.expect("client write");
        client.flush().await.expect("flush");
        let mut response = [0_u8; 5];
        client.read_exact(&mut response).await.expect("client read");
        assert_eq!(&response, b"world");

        let server_buf = server.await.expect("join");
        assert_eq!(&server_buf, b"hello");
    }

    #[cfg(unix)]
    #[proxima::test]
    async fn unix_listener_round_trips_a_few_bytes() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("proxima-stream.sock");

        let listener = TokioUnixListener::bind(path.clone()).await.expect("bind");
        let server = tokio::spawn(async move {
            let mut conn = listener.accept().await.expect("accept");
            let mut buf = [0_u8; 3];
            conn.read_exact(&mut buf).await.expect("server read");
            conn.write_all(b"pong").await.expect("server write");
            conn.flush().await.expect("flush");
            buf
        });

        // small wait for bind file to appear; mirrors direct.rs test pattern.
        for _ in 0..50 {
            if path.exists() {
                break;
            }
            proxima_core::time::sleep(std::time::Duration::from_millis(20)).await;
        }

        let mut client = tokio::net::UnixStream::connect(&path)
            .await
            .expect("client connect");
        client.write_all(b"foo").await.expect("client write");
        client.flush().await.expect("flush");
        let mut response = [0_u8; 4];
        client.read_exact(&mut response).await.expect("client read");
        assert_eq!(&response, b"pong");

        let server_buf = server.await.expect("join");
        assert_eq!(&server_buf, b"foo");
    }
}
