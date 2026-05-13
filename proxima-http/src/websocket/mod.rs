//! WebSocket over any `StreamConnection` via `async-tungstenite`.
//! After the handshake, bidirectional bytes — middleware sees a
//! byte-stream, same as any other `StreamListener`.

pub mod upstream;

pub use upstream::WebSocketUpstream;

use std::io;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::Mutex;
use std::task::{Context, Poll};

use async_tungstenite::WebSocketStream;
use async_tungstenite::tungstenite::protocol::Message;
use futures::sink::Sink;
use futures::stream::Stream;

use proxima_net::tokio::tokio_stream_listener::TokioTcpConnection;
use proxima_primitives::stream::{BindAddr, PeerInfo, StreamConnection, StreamListener};

/// WebSocket connection wrapped as a byte stream. Generic over the
/// underlying `StreamConnection`.
pub struct WebSocketConnection<C: StreamConnection> {
    inner: WebSocketStream<C>,
    peer: Option<PeerInfo>,
    // WHY Mutex here:
    //   `AsyncRead::poll_read(self: Pin<&mut Self>, ...)` reads from
    //   the WebSocket message stream which delivers full messages,
    //   not arbitrary-length byte runs. When the caller's `buf` is
    //   smaller than the next message, the remainder must be cached
    //   for the next poll. The cache (`read_buffer`) is mutated via
    //   `&self` even though `Pin<&mut Self>` is available, because
    //   the cache outlives the immediate borrow and Tokio's
    //   read-half splits don't carry &mut Self lifetimes cleanly.
    //
    // WHY NOT removable:
    //   - RefCell: would make WebSocketConnection !Send, breaking
    //     the StreamConnection trait surface (Send + Sync required
    //     for cross-thread dispatch).
    //   - UnsafeCell + unsafe Send/Sync: introduces UB risk for a
    //     primitive accessed only by the poll loop, not justified.
    //   - Atomic: can't atomically swap a Vec.
    //   - Restructure to drop the cache: would require the WebSocket
    //     adapter to discard bytes when buf is too small — wrong.
    //
    // WHY this is right:
    //   Per-connection (one WebSocketConnection per upgraded socket).
    //   Lock held briefly per poll: ~5ns acquire + memcpy. No bench
    //   needed — the trait API + the message-vs-byte mismatch
    //   structurally require interior mutability.
    read_buffer: Mutex<Vec<u8>>,
}

impl<C: StreamConnection> WebSocketConnection<C> {
    fn new(inner: WebSocketStream<C>, peer: Option<PeerInfo>) -> Self {
        Self {
            inner,
            peer,
            read_buffer: Mutex::new(Vec::new()),
        }
    }
}

impl<C: StreamConnection> futures::io::AsyncRead for WebSocketConnection<C> {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut [u8],
    ) -> Poll<io::Result<usize>> {
        let this = self.get_mut();
        // drain whatever's left in the read_buffer first; only poll
        // the underlying ws stream when the buffer is empty.
        {
            let Ok(mut leftover) = this.read_buffer.lock() else {
                return Poll::Ready(Err(io::Error::other("ws read buffer lock poisoned")));
            };
            if !leftover.is_empty() {
                let take = leftover.len().min(buf.len());
                buf[..take].copy_from_slice(&leftover[..take]);
                leftover.drain(..take);
                return Poll::Ready(Ok(take));
            }
        }

        match Stream::poll_next(Pin::new(&mut this.inner), cx) {
            Poll::Ready(Some(Ok(message))) => {
                let payload = match message {
                    Message::Binary(bytes) => bytes.to_vec(),
                    Message::Text(text) => text.as_bytes().to_vec(),
                    // Ping/Pong/Close/Frame don't carry application
                    // bytes; treat as a 0-byte read so the caller can
                    // poll again.
                    _ => return Poll::Ready(Ok(0)),
                };
                let take = payload.len().min(buf.len());
                buf[..take].copy_from_slice(&payload[..take]);
                if take < payload.len() {
                    let Ok(mut leftover) = this.read_buffer.lock() else {
                        return Poll::Ready(Err(io::Error::other("ws read buffer lock poisoned")));
                    };
                    leftover.extend_from_slice(&payload[take..]);
                }
                Poll::Ready(Ok(take))
            }
            Poll::Ready(Some(Err(err))) => {
                Poll::Ready(Err(io::Error::other(format!("ws read: {err}"))))
            }
            Poll::Ready(None) => Poll::Ready(Ok(0)),
            Poll::Pending => Poll::Pending,
        }
    }
}

impl<C: StreamConnection> futures::io::AsyncWrite for WebSocketConnection<C> {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        let this = self.get_mut();
        let message = Message::Binary(buf.to_vec().into());
        match Sink::poll_ready(Pin::new(&mut this.inner), cx) {
            Poll::Ready(Ok(())) => match Sink::start_send(Pin::new(&mut this.inner), message) {
                Ok(()) => Poll::Ready(Ok(buf.len())),
                Err(err) => Poll::Ready(Err(io::Error::other(format!("ws send: {err}")))),
            },
            Poll::Ready(Err(err)) => Poll::Ready(Err(io::Error::other(format!("ws ready: {err}")))),
            Poll::Pending => Poll::Pending,
        }
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        let this = self.get_mut();
        match Sink::poll_flush(Pin::new(&mut this.inner), cx) {
            Poll::Ready(Ok(())) => Poll::Ready(Ok(())),
            Poll::Ready(Err(err)) => Poll::Ready(Err(io::Error::other(format!("ws flush: {err}")))),
            Poll::Pending => Poll::Pending,
        }
    }

    fn poll_close(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        let this = self.get_mut();
        match Sink::poll_close(Pin::new(&mut this.inner), cx) {
            Poll::Ready(Ok(())) => Poll::Ready(Ok(())),
            Poll::Ready(Err(err)) => Poll::Ready(Err(io::Error::other(format!("ws close: {err}")))),
            Poll::Pending => Poll::Pending,
        }
    }
}

impl<C: StreamConnection> StreamConnection for WebSocketConnection<C> {
    fn peer(&self) -> Option<PeerInfo> {
        self.peer.clone()
    }
}

type WsAcceptFut = Pin<
    Box<
        dyn std::future::Future<Output = io::Result<WebSocketConnection<TokioTcpConnection>>>
            + Send,
    >,
>;

/// Tokio TCP-backed WebSocket listener. Accepts a TCP connection,
/// performs the WebSocket handshake via async-tungstenite, yields a
/// `WebSocketConnection<TokioTcpConnection>`. Future DPDK/AF_XDP
/// variants would mirror this with their backend's StreamConnection
/// type substituted.
pub struct WebSocketListener {
    inner: tokio::net::TcpListener,
    local_addr: Option<SocketAddr>,
    // WHY Mutex here / WHY NOT removable / WHY right: same poll-
    // resumable-future pattern as `TokioTcpUpstream::in_flight`
    // (src/upstreams/tokio_stream.rs). Per-listener single-poll-at-
    // a-time slot for the in-flight WS upgrade future.
    in_flight: Mutex<Option<WsAcceptFut>>,
}

impl WebSocketListener {
    pub async fn bind(addr: SocketAddr) -> io::Result<Self> {
        let inner = tokio::net::TcpListener::bind(addr).await?;
        let local_addr = inner.local_addr().ok();
        Ok(Self {
            inner,
            local_addr,
            in_flight: Mutex::new(None),
        })
    }
}

impl StreamListener for WebSocketListener {
    type Conn = WebSocketConnection<TokioTcpConnection>;

    fn poll_accept(&self, cx: &mut Context<'_>) -> Poll<io::Result<Self::Conn>> {
        let Ok(mut slot) = self.in_flight.lock() else {
            return Poll::Ready(Err(io::Error::other("ws in-flight lock poisoned")));
        };
        if slot.is_none() {
            match self.inner.poll_accept(cx) {
                Poll::Ready(Ok((stream, peer))) => {
                    let tokio_conn =
                        proxima_net::tokio::tokio_stream_listener::tcp_connection_from_stream(
                            stream,
                        );
                    *slot = Some(Box::pin(async move {
                        let upgraded = async_tungstenite::accept_async(tokio_conn)
                            .await
                            .map_err(|err| io::Error::other(format!("ws handshake: {err}")))?;
                        Ok(WebSocketConnection::new(
                            upgraded,
                            Some(PeerInfo::Tcp(peer)),
                        ))
                    }));
                }
                Poll::Ready(Err(err)) => return Poll::Ready(Err(err)),
                Poll::Pending => return Poll::Pending,
            }
        }

        let Some(future) = slot.as_mut() else {
            return Poll::Pending;
        };
        match future.as_mut().poll(cx) {
            Poll::Ready(result) => {
                *slot = None;
                Poll::Ready(result)
            }
            Poll::Pending => Poll::Pending,
        }
    }

    fn local_addr(&self) -> Option<BindAddr> {
        self.local_addr.map(BindAddr::Tcp)
    }
}
