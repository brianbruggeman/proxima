//! Reusable accept-and-invoke-upgrade serve loop for connection `Pipe`s.
//!
//! Some protocols (pgwire, raw tunnels) model a whole connection as a
//! single `Pipe::call`: the listener hands the pipe a CONNECT-style
//! request, the pipe returns a `Response.upgrade`, and the listener
//! cedes the accepted socket to that upgrade handler for the connection's
//! lifetime. The generic stream-accept path doesn't consume
//! `Response.upgrade` (only the h1/io_uring listeners do), so each such
//! protocol grew its own accept loop. This is that loop, extracted once.

use std::io;
use std::net::SocketAddr;
use std::sync::Arc;

use bytes::Bytes;
use futures::FutureExt;
use futures::channel::oneshot;
use futures::future::poll_fn;
use proxima_telemetry::{debug, warn};

use proxima_core::ProximaError;
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::upgrade::{AcceptHandle, HijackStream, HijackedSocket};
use proxima_runtime::Runtime;
use proxima_primitives::stream::{AcceptorFactory, StreamConnection, TcpBindOptions};

/// Drives one accepted socket through the connection pipe: call the
/// `() -> UpgradeHandler` accept hook and invoke the handler against the
/// socket wrapped as a `HijackedSocket` (the accepted connection is
/// already `futures::io`, so it boxes straight to `Box<dyn HijackStream>`).
/// `Out = UpgradeHandler` makes the no-upgrade state unrepresentable —
/// there is no defensive `else` branch to write.
///
/// `pub` — the ONE connection-layer accept-to-upgrade driver. Before the
/// AnyListenProtocol lift, `proxima-pgwire` and `proxima-redis` each carried
/// a byte-identical private copy of this exact function for their own
/// legacy tokio-compat accept loop; both now call this one instead
/// (workspace principle 1: dedup by pointing at the canonical primitive).
pub async fn handle_connection(
    conn: Box<dyn StreamConnection>,
    accept: AcceptHandle,
) -> Result<(), ProximaError> {
    let handler = accept.call(()).await?;
    let stream: Box<dyn HijackStream> = Box::new(conn);
    let hijacked = HijackedSocket::new(stream, Bytes::new());
    handler.invoke(hijacked).await
}

fn spawn_connection(
    conn: Box<dyn StreamConnection>,
    pipe: &AcceptHandle,
    runtime: Option<&Arc<dyn Runtime>>,
    label: &str,
) {
    let pipe = pipe.clone();
    let label_owned = label.to_string();
    let conn_future = async move {
        let peer = conn.peer();
        if let Err(error) = handle_connection(conn, pipe).await {
            debug!(?error, ?peer, label = %label_owned, "upgrade connection ended");
        }
    };
    match runtime {
        Some(runtime) => runtime.spawn_on_current_core(Box::pin(conn_future)),
        #[cfg(feature = "tokio")]
        None => {
            tokio::task::spawn_local(conn_future);
        }
        #[cfg(not(feature = "tokio"))]
        None => {
            warn!(
                label = %label,
                "upgrade connection dropped: no runtime injected and the `tokio` \
                 feature is off, so there is no executor to spawn the ?Send \
                 connection future onto"
            );
            drop(conn_future);
        }
    }
}

/// Binds an acceptor through `factory`, then accepts connections until
/// `shutdown` fires, driving each through `pipe` as a CONNECT →
/// `Response.upgrade` → invoke exchange. The per-connection future runs on
/// `runtime` (`spawn_on_current_core`) when present, else `spawn_local`.
///
/// `label` only colors the bind/accept/connection log lines. `ready_signal`,
/// when present, fires right after the real bind succeeds so
/// `Listener::run_with_runtime`'s per-lane readiness wait observes this lane
/// going live.
///
/// # Errors
/// [`ProximaError::Io`] when the acceptor cannot bind. Per-connection and
/// accept errors are logged, not propagated (the loop keeps serving).
// ready_signal is the readiness-gate wiring shared by every ListenProtocol serve path.
#[allow(clippy::too_many_arguments)]
pub async fn serve_pipe_upgrades(
    factory: Arc<dyn AcceptorFactory>,
    bind: SocketAddr,
    options: TcpBindOptions,
    pipe: AcceptHandle,
    runtime: Option<Arc<dyn Runtime>>,
    mut shutdown: oneshot::Receiver<()>,
    label: &str,
    ready_signal: Option<std::sync::mpsc::Sender<()>>,
) -> Result<(), ProximaError> {
    let mut acceptor = factory.bind(bind, options).map_err(|error| {
        ProximaError::Io(io::Error::other(format!("{label} bind {bind}: {error}")))
    })?;
    if let Some(sender) = ready_signal {
        let _ = sender.send(());
    }
    debug!(label = %label, %bind, "upgrade listener bound");
    loop {
        futures::select_biased! {
            _ = (&mut shutdown).fuse() => return Ok(()),
            accepted = poll_fn(|cx| acceptor.poll_accept(cx)).fuse() => match accepted {
                Ok(conn) => spawn_connection(conn, &pipe, runtime.as_ref(), label),
                Err(error) => warn!(?error, label = %label, "upgrade accept failed"),
            },
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::future::Future;
    use std::pin::Pin;
    use std::task::{Context, Poll};

    use futures::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
    use proxima_primitives::pipe::upgrade::UpgradeHandler;
    use proxima_primitives::stream::{PeerInfo, TcpAcceptor};

    use super::*;

    // a single accepted connection then EOF on poll_accept; the conn is a
    // loopback duplex echo backed by a futures duplex pair the test reads.
    struct OneShotAcceptor {
        conn: Option<Box<dyn StreamConnection>>,
    }

    impl TcpAcceptor for OneShotAcceptor {
        fn poll_accept(
            &mut self,
            _cx: &mut Context<'_>,
        ) -> Poll<io::Result<Box<dyn StreamConnection>>> {
            match self.conn.take() {
                Some(conn) => Poll::Ready(Ok(conn)),
                None => Poll::Pending,
            }
        }

        fn local_addr(&self) -> io::Result<SocketAddr> {
            Ok(SocketAddr::from(([127, 0, 0, 1], 0)))
        }
    }

    struct OneShotFactory {
        conn: std::sync::Mutex<Option<Box<dyn StreamConnection>>>,
    }

    impl AcceptorFactory for OneShotFactory {
        fn bind(
            &self,
            _addr: SocketAddr,
            _options: TcpBindOptions,
        ) -> io::Result<Box<dyn TcpAcceptor>> {
            let conn = self.conn.lock().unwrap().take();
            Ok(Box::new(OneShotAcceptor { conn }))
        }
    }

    // a trivial in-memory bidirectional stream: what we write, we can read.
    struct LoopbackStream {
        buffer: Vec<u8>,
        read_pos: usize,
    }

    impl AsyncRead for LoopbackStream {
        fn poll_read(
            mut self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
            buf: &mut [u8],
        ) -> Poll<io::Result<usize>> {
            let available = &self.buffer[self.read_pos..];
            if available.is_empty() {
                return Poll::Ready(Ok(0));
            }
            let count = available.len().min(buf.len());
            buf[..count].copy_from_slice(&available[..count]);
            self.read_pos += count;
            Poll::Ready(Ok(count))
        }
    }

    impl AsyncWrite for LoopbackStream {
        fn poll_write(
            mut self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
            buf: &[u8],
        ) -> Poll<io::Result<usize>> {
            self.buffer.extend_from_slice(buf);
            Poll::Ready(Ok(buf.len()))
        }

        fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
            Poll::Ready(Ok(()))
        }

        fn poll_close(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
            Poll::Ready(Ok(()))
        }
    }

    impl StreamConnection for LoopbackStream {
        fn peer(&self) -> Option<PeerInfo> {
            None
        }
    }

    struct EchoBytePipe {
        done: Arc<async_channel::Sender<()>>,
    }

    impl proxima_primitives::pipe::SendPipe for EchoBytePipe {
        type In = ();
        type Out = UpgradeHandler;
        type Err = ProximaError;

        fn call(&self, (): ()) -> impl Future<Output = Result<UpgradeHandler, ProximaError>> + Send {
            let done = self.done.clone();
            async move {
                let handler = UpgradeHandler::new(move |mut socket: HijackedSocket| {
                    let done = done.clone();
                    async move {
                        let mut byte = [0_u8; 1];
                        let read = socket.stream.read(&mut byte).await.unwrap_or(0);
                        if read == 1 {
                            let _ = socket.stream.write_all(&byte).await;
                        }
                        let _ = done.send(()).await;
                        Ok(())
                    }
                });
                Ok(handler)
            }
        }
    }

    #[proxima::test(runtime = "tokio")]
    async fn serve_pipe_upgrades_drives_an_upgrade_returning_pipe_to_completion() {
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                let conn = LoopbackStream {
                    buffer: vec![0x42],
                    read_pos: 0,
                };
                let factory = Arc::new(OneShotFactory {
                    conn: std::sync::Mutex::new(Some(Box::new(conn))),
                });
                let (done_tx, done_rx) = async_channel::bounded(1);
                let pipe: AcceptHandle = proxima_primitives::pipe::alloc_tier::into_handle(EchoBytePipe {
                    done: Arc::new(done_tx),
                });
                let (_shutdown_tx, shutdown_rx) = oneshot::channel();

                tokio::task::spawn_local(async move {
                    let _ = serve_pipe_upgrades(
                        factory,
                        SocketAddr::from(([127, 0, 0, 1], 0)),
                        TcpBindOptions::default(),
                        pipe,
                        None,
                        shutdown_rx,
                        "test-upgrade",
                        None,
                    )
                    .await;
                });

                done_rx
                    .recv()
                    .await
                    .expect("upgrade handler must run to completion");
            })
            .await;
    }
}
