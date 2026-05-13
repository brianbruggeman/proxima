//! Tokio-backed `StreamUpstream` implementations for TCP and Unix
//! sockets. Symmetric to `listeners/tokio_stream.rs`: each upstream
//! produces a `StreamConnection` once the connect handshake completes.

use std::io;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Mutex;
use std::task::{Context, Poll};

use super::tokio_stream_listener::TokioTcpConnection;
use proxima_primitives::stream::StreamUpstream;

#[cfg(unix)]
use super::tokio_stream_listener::TokioUnixConnection;

type ConnectFuture<C> = Pin<Box<dyn std::future::Future<Output = io::Result<C>> + Send>>;

/// Tokio-backed TCP upstream. Lazily attempts a connect on each
/// `poll_connect` call; the inner future is cached between polls so a
/// pending connect can resume.
pub struct TokioTcpUpstream {
    addr: SocketAddr,
    // WHY Mutex here:
    //   `poll_connect(&self, ...)` takes `&self` (the trait API
    //   doesn't allow `&mut self`). We need interior mutability to
    //   stash and resume the in-flight connect future across polls.
    //
    // WHY NOT removable:
    //   - `RefCell<Option<Future>>`: not Send, would force the
    //     entire `TokioTcpUpstream` to be !Send. Upstreams cross
    //     thread boundaries (chain dispatch may migrate Send
    //     futures; control plane handles cross-thread).
    //   - Atomic: futures aren't movable through atomic-pointer ops
    //     soundly (fat pointer + lifetime tracking).
    //   - Lock-free queue: doesn't help — there's only one future
    //     at a time, not a queue.
    //   - Restructure to `&mut self`: requires changing the upstream
    //     trait surface, which would break the substrate's "Pipe
    //     and friends are shared-ref-only" model.
    //
    // WHY this is right:
    //   The Mutex is acquired exactly once per poll cycle on this
    //   upstream. Per-connection (not per-request) usage means
    //   contention is bounded to the polling task itself. Lock cost
    //   is ~5ns uncontested, dwarfed by `tokio::net::TcpStream
    //   ::connect` (microseconds for the TCP handshake). No bench
    //   needed — the alternative primitives are structurally ruled
    //   out by the trait surface.
    in_flight: Mutex<Option<ConnectFuture<TokioTcpConnection>>>,
}

impl TokioTcpUpstream {
    pub fn new(addr: SocketAddr) -> Self {
        Self {
            addr,
            in_flight: Mutex::new(None),
        }
    }
}

impl StreamUpstream for TokioTcpUpstream {
    type Conn = TokioTcpConnection;

    fn poll_connect(&self, cx: &mut Context<'_>) -> Poll<io::Result<Self::Conn>> {
        let addr = self.addr;
        let Ok(mut slot) = self.in_flight.lock() else {
            return Poll::Ready(Err(io::Error::other("upstream lock poisoned")));
        };
        let future = slot.get_or_insert_with(|| {
            Box::pin(async move {
                let stream = tokio::net::TcpStream::connect(addr).await?;
                stream.set_nodelay(true).ok();
                Ok(super::tokio_stream_listener::tcp_connection_from_stream(
                    stream,
                ))
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

#[cfg(unix)]
pub struct TokioUnixUpstream {
    path: PathBuf,
    // WHY Mutex here / WHY NOT removable / WHY right: see
    // `TokioTcpUpstream::in_flight` above. Same pattern, same
    // structural constraints. Per-connection upstream, uncontested
    // single-poll-at-a-time lock acquire.
    in_flight: Mutex<Option<ConnectFuture<TokioUnixConnection>>>,
}

#[cfg(unix)]
impl TokioUnixUpstream {
    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            in_flight: Mutex::new(None),
        }
    }
}

#[cfg(unix)]
impl StreamUpstream for TokioUnixUpstream {
    type Conn = TokioUnixConnection;

    fn poll_connect(&self, cx: &mut Context<'_>) -> Poll<io::Result<Self::Conn>> {
        let path = self.path.clone();
        let Ok(mut slot) = self.in_flight.lock() else {
            return Poll::Ready(Err(io::Error::other("upstream lock poisoned")));
        };
        let future = slot.get_or_insert_with(|| {
            Box::pin(async move {
                let stream = tokio::net::UnixStream::connect(&path).await?;
                Ok(super::tokio_stream_listener::unix_connection_from_stream(
                    stream,
                ))
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
    use super::super::tokio_stream_listener::TokioTcpListener;
    use futures::io::{AsyncReadExt, AsyncWriteExt};
    use proxima_primitives::stream::{StreamListener, StreamListenerExt, StreamUpstreamExt};
    use std::net::Ipv4Addr;

    #[proxima::test]
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
