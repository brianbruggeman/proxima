//! Tokio-backed `StreamUpstream` implementations for TCP and Unix
//! sockets. Symmetric to `listeners/tokio_stream.rs`: each upstream
//! produces a `StreamConnection` once the connect handshake completes.

use std::net::SocketAddr;

use super::tokio_stream_listener::TokioTcpConnection;
use proxima_primitives::stream::{ConnectFuture, StreamUpstream};

#[cfg(unix)]
use super::tokio_stream_listener::TokioUnixConnection;
#[cfg(unix)]
use proxima_primitives::stream::{StreamConnection, UnixUpstreamFactory};
#[cfg(unix)]
use std::path::PathBuf;

/// tokio TCP upstream with an independent connection future per call.
pub struct TokioTcpUpstream {
    addr: SocketAddr,
}

impl TokioTcpUpstream {
    pub fn new(addr: SocketAddr) -> Self {
        Self { addr }
    }
}

impl StreamUpstream for TokioTcpUpstream {
    type Conn = TokioTcpConnection;

    fn connect_future(&self) -> ConnectFuture<'_, Self::Conn> {
        let addr = self.addr;
        Box::pin(async move {
            let stream = tokio::net::TcpStream::connect(addr).await?;
            stream.set_nodelay(true)?;
            Ok(super::tokio_stream_listener::tcp_connection_from_stream(
                stream,
            ))
        })
    }
}

#[cfg(unix)]
pub struct TokioUnixUpstream {
    path: PathBuf,
}

#[cfg(unix)]
impl TokioUnixUpstream {
    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }
}

#[cfg(unix)]
impl StreamUpstream for TokioUnixUpstream {
    type Conn = TokioUnixConnection;

    fn connect_future(&self) -> ConnectFuture<'_, Self::Conn> {
        let path = self.path.clone();
        Box::pin(async move {
            let stream = tokio::net::UnixStream::connect(path).await?;
            Ok(super::tokio_stream_listener::unix_connection_from_stream(
                stream,
            ))
        })
    }
}

/// Type-erases `TokioUnixUpstream::Conn` to `Box<dyn StreamConnection>` —
/// the tokio sibling of `proxima_net::prime::unix::BoxedPrimeUnixUpstream`.
/// Same erasure boundary, so `RuntimeSelection` can hold either backend's
/// unix-upstream factory behind one field.
#[cfg(unix)]
struct BoxedTokioUnixUpstream(TokioUnixUpstream);

#[cfg(unix)]
impl StreamUpstream for BoxedTokioUnixUpstream {
    type Conn = Box<dyn StreamConnection>;

    fn connect_future(&self) -> ConnectFuture<'_, Self::Conn> {
        Box::pin(async move {
            Ok(Box::new(self.0.connect_future().await?) as Box<dyn StreamConnection>)
        })
    }
}

/// tokio-backed [`UnixUpstreamFactory`] — the runtime-selectable entry
/// point `RuntimeSelection::tokio()` bundles.
#[cfg(unix)]
pub struct TokioUnixUpstreamFactory;

#[cfg(unix)]
impl UnixUpstreamFactory for TokioUnixUpstreamFactory {
    fn connect(
        &self,
        path: PathBuf,
    ) -> std::sync::Arc<dyn StreamUpstream<Conn = Box<dyn StreamConnection>>> {
        std::sync::Arc::new(BoxedTokioUnixUpstream(TokioUnixUpstream::new(path)))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::super::tokio_stream_listener::TokioTcpListener;
    use super::*;
    use futures::io::{AsyncReadExt, AsyncWriteExt};
    use proxima_primitives::stream::{StreamListener, StreamListenerExt, StreamUpstreamExt};
    use std::net::Ipv4Addr;

    #[proxima::test(runtime = "tokio")]
    async fn tcp_upstream_connects_to_listener() {
        let listener = TokioTcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))
            .await
            .expect("bind");
        let local = match listener.local_addr().expect("local_addr") {
            proxima_primitives::stream::BindAddr::Tcp(addr) => addr,
            _ => panic!("expected tcp"),
        };

        let server = tokio::spawn(async move {
            let mut conn = listener.accept().await.expect("accept");
            let mut buf = [0_u8; 4];
            conn.read_exact(&mut buf).await.expect("read");
            conn.write_all(b"ack").await.expect("write");
            conn.flush().await.expect("flush");
            buf
        });

        let upstream = TokioTcpUpstream::new(local);
        let mut conn = upstream.connect().await.expect("upstream connect");
        conn.write_all(b"ping").await.expect("write");
        conn.flush().await.expect("flush");
        let mut response = [0_u8; 3];
        conn.read_exact(&mut response).await.expect("read");
        assert_eq!(&response, b"ack");

        let server_buf = server.await.expect("join");
        assert_eq!(&server_buf, b"ping");
    }
}
