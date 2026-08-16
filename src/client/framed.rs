//! `FramedClient` — the client counterpart to
//! [`FramedListenProtocol`](crate::listeners::FramedListenProtocol): dials a
//! length-delimited `[u32 BE len][payload]` listener and holds the
//! connection across many round trips. Frames identically to the listener
//! — same [`proxima_codec::LengthDelimitedCodec`], same 4-byte big-endian
//! header, same one-frame-out/one-frame-back shape — so a `FramedClient`
//! dialing a `FramedListenProtocol` app interoperates on the wire with no
//! adaptation on either side.
//!
//! Before this type, dialing a framed listener meant hand-rolling the
//! length prefix over a raw socket (see this crate's `tests/framed_app.rs`
//! prior to adopting `FramedClient`) — there was no client counterpart to
//! complete the App/Listener/Client trio for framed protocols.

use std::future::Future;
use std::io;

use bytes::{Bytes, BytesMut};
use futures::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

use proxima_codec::{FrameCodec, FrameError, FrameLimits, LengthDelimitedCodec};
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::sync::Mutex;

use crate::error::ProximaError;

const DEFAULT_READ_CHUNK: usize = 64 * 1024;

/// Length-delimited request/reply client. Holds one connection across many
/// round trips — `call` may be invoked repeatedly on the same
/// `FramedClient`, the shape a long-poll caller needs. Generic over the
/// transport (`AsyncRead + AsyncWrite + Unpin`, the same bound
/// `FramedListenProtocol`'s own `C: StreamConnection` resolves to) so any
/// backend's stream — tokio's, a cipher-wrapped one, an in-memory test
/// double — works without this type naming a runtime. Wrap an
/// already-wrapped stream (TLS, a cipher) and pass it to [`Self::new`] to
/// get the same symmetry `FramedListenProtocol::with_conn_transform` gives
/// the server side — no separate transform hook needed on this side; see
/// the module doc on why.
pub struct FramedClient<C> {
    state: Mutex<FramedState<C>>,
}

struct FramedState<C> {
    conn: C,
    codec: LengthDelimitedCodec,
    buf: BytesMut,
    scratch: Vec<u8>,
    out_frame: Vec<u8>,
}

impl<C: AsyncRead + AsyncWrite + Unpin> FramedClient<C> {
    /// Wrap an already-connected stream with the default frame limits
    /// (64 MiB cap, zero-length frames allowed — [`FrameLimits::default`]).
    #[must_use]
    pub fn new(conn: C) -> Self {
        Self::with_limits(conn, FrameLimits::default())
    }

    /// Wrap an already-connected stream with explicit frame limits — pass
    /// the SAME [`FrameLimits`] the peer's `FramedListenProtocol` was
    /// configured with (`max_frame_bytes` / `reject_zero_len` in its
    /// listener spec) so both sides agree on what a valid frame is.
    #[must_use]
    pub fn with_limits(conn: C, limits: FrameLimits) -> Self {
        Self {
            state: Mutex::new(FramedState {
                conn,
                codec: LengthDelimitedCodec::new(limits),
                buf: BytesMut::with_capacity(DEFAULT_READ_CHUNK),
                scratch: vec![0_u8; DEFAULT_READ_CHUNK],
                out_frame: Vec::new(),
            }),
        }
    }

    /// One request frame out, one reply frame back. Safe to call
    /// repeatedly on the same `FramedClient`: each call sends, then waits
    /// for the next full frame, so sequential calls multi-round-trip on
    /// the held connection — exactly what `FramedListenProtocol`'s
    /// per-connection loop expects on the other end. `&self` (not
    /// `&mut self`) so `FramedClient` can implement [`SendPipe`]; the
    /// connection state serializes calls through an async mutex rather
    /// than the caller statically proving exclusivity.
    pub async fn call(&self, payload: Bytes) -> Result<Bytes, ProximaError> {
        self.round_trip(payload).await
    }

    async fn round_trip(&self, payload: Bytes) -> Result<Bytes, ProximaError> {
        let mut state = self.state.lock().await;
        state.send_frame(&payload).await?;
        state.recv_frame().await
    }
}

impl<C: AsyncRead + AsyncWrite + Unpin> FramedState<C> {
    async fn send_frame(&mut self, payload: &[u8]) -> Result<(), ProximaError> {
        self.out_frame.clear();
        self.codec
            .encode_frame(&payload, &mut self.out_frame)
            .map_err(|err| {
                ProximaError::Io(io::Error::new(io::ErrorKind::InvalidData, format!("{err}")))
            })?;
        self.conn
            .write_all(&self.out_frame)
            .await
            .map_err(|err| ProximaError::Io(io::Error::other(format!("frame write: {err}"))))?;
        self.conn
            .flush()
            .await
            .map_err(|err| ProximaError::Io(io::Error::other(format!("frame flush: {err}"))))
    }

    async fn recv_frame(&mut self) -> Result<Bytes, ProximaError> {
        loop {
            match self.codec.parse_frame(&self.buf) {
                Ok((_frame, consumed)) => {
                    // zero-copy: the frame Bytes shares the read buffer's
                    // allocation; slice off the 4-byte length prefix.
                    let whole = self.buf.split_to(consumed).freeze();
                    return Ok(whole.slice(LengthDelimitedCodec::HEADER_BYTES..));
                }
                Err(FrameError::Incomplete) => {
                    let read = self.conn.read(&mut self.scratch).await.map_err(|err| {
                        ProximaError::Io(io::Error::other(format!("frame read: {err}")))
                    })?;
                    if read == 0 {
                        return Err(ProximaError::Io(io::Error::new(
                            io::ErrorKind::UnexpectedEof,
                            "connection closed mid-frame",
                        )));
                    }
                    self.buf.extend_from_slice(&self.scratch[..read]);
                }
                Err(err) => {
                    return Err(ProximaError::Io(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("{err}"),
                    )));
                }
            }
        }
    }
}

/// `FramedClient` is a [`SendPipe`] — `In = Bytes`, `Out = Bytes` — so it
/// composes as a transport stage exactly like [`Client`](crate::Client)
/// does for HTTP, just over the framed wire instead. The connection state
/// lives behind an async mutex (see [`Self::call`]'s doc) so `&self`
/// dispatch is sound without the caller holding `&mut`.
impl<C: AsyncRead + AsyncWrite + Send + Unpin + 'static> SendPipe for FramedClient<C> {
    type In = Bytes;
    type Out = Bytes;
    type Err = ProximaError;

    fn call(&self, payload: Bytes) -> impl Future<Output = Result<Bytes, ProximaError>> + Send {
        self.round_trip(payload)
    }
}

#[cfg(all(feature = "tcp", feature = "tokio"))]
impl FramedClient<crate::listeners::TokioTcpConnection> {
    /// Dial a `FramedListenProtocol` listener over TCP — the one-liner
    /// pairing `FramedClient` with the same tokio-backed transport this
    /// crate's other tokio-gated client/listener surface uses
    /// (`crate::listeners::{TokioTcpConnection, TokioTcpListener}`,
    /// `crate::upstreams::tokio_stream::TokioTcpUpstream`). Gated on the
    /// same `tcp` + `tokio` features `FramedListenProtocol`'s own bind loop
    /// requires, so a caller that can mount the listener can always dial
    /// it. Use [`Self::new`] directly for a non-tokio backend or a
    /// pre-wrapped (TLS/cipher) stream.
    pub async fn connect(addr: std::net::SocketAddr) -> Result<Self, ProximaError> {
        let upstream = crate::upstreams::tokio_stream::TokioTcpUpstream::new(addr);
        let conn = proxima_primitives::stream::StreamUpstreamExt::connect(&upstream)
            .await
            .map_err(ProximaError::Io)?;
        Ok(Self::new(conn))
    }
}

#[cfg(all(test, feature = "tcp", feature = "tokio"))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use proxima_net::tokio::tokio_stream_listener::TokioTcpListener;
    use proxima_primitives::stream::{StreamListener, StreamListenerExt};
    use std::net::{Ipv4Addr, SocketAddr};

    // proves FramedClient's own send/recv loop in isolation from the
    // App/Listener machinery: a raw echo server that speaks the same
    // LengthDelimitedCodec framing, driven with plain tokio::net so this
    // test doesn't depend on FramedClient to also read frames correctly.
    async fn spawn_echo_server(addr: SocketAddr) -> SocketAddr {
        let listener = TokioTcpListener::bind(addr).await.expect("bind");
        let local = match listener.local_addr().expect("local_addr") {
            proxima_primitives::stream::BindAddr::Tcp(addr) => addr,
            other => panic!("expected tcp, got {other:?}"),
        };
        tokio::spawn(async move {
            let mut conn = listener.accept().await.expect("accept");
            let mut buf = [0_u8; 4];
            loop {
                if conn.read_exact(&mut buf).await.is_err() {
                    return;
                }
                let len = u32::from_be_bytes(buf) as usize;
                let mut payload = vec![0_u8; len];
                conn.read_exact(&mut payload).await.expect("read payload");
                conn.write_all(&buf).await.expect("write header");
                conn.write_all(&payload).await.expect("write payload");
                conn.flush().await.expect("flush");
            }
        });
        local
    }

    #[proxima::test]
    async fn framed_client_round_trips_against_a_raw_echo_server() {
        let local = spawn_echo_server(SocketAddr::from((Ipv4Addr::LOCALHOST, 0))).await;
        let client = FramedClient::connect(local).await.expect("connect");

        let reply = client
            .call(Bytes::from_static(b"ping"))
            .await
            .expect("call");
        assert_eq!(reply, Bytes::from_static(b"ping"));
    }

    #[proxima::test]
    async fn framed_client_multi_round_trips_three_sequential_calls_one_connection() {
        let local = spawn_echo_server(SocketAddr::from((Ipv4Addr::LOCALHOST, 0))).await;
        let client = FramedClient::connect(local).await.expect("connect");

        for payload in [&b"one"[..], &b"two"[..], &b"three"[..]] {
            let reply = client
                .call(Bytes::copy_from_slice(payload))
                .await
                .expect("call");
            assert_eq!(reply, Bytes::copy_from_slice(payload));
        }
    }

    #[proxima::test]
    async fn framed_client_wire_bytes_match_length_delimited_codec() {
        // stand up a raw capture server (no framing on its side) and prove
        // FramedClient puts EXACTLY [u32 BE len][payload] on the wire —
        // the same bytes `LengthDelimitedCodec::encode_frame` would produce
        // and `FramedListenProtocol` parses on the other end.
        let listener = TokioTcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))
            .await
            .expect("bind");
        let local = match listener.local_addr().expect("local_addr") {
            proxima_primitives::stream::BindAddr::Tcp(addr) => addr,
            other => panic!("expected tcp, got {other:?}"),
        };

        let capture = tokio::spawn(async move {
            let mut conn = listener.accept().await.expect("accept");
            let mut wire = vec![0_u8; 4 + b"hello wire".len()];
            conn.read_exact(&mut wire).await.expect("read wire bytes");
            wire
        });

        let client = FramedClient::connect(local).await.expect("connect");
        // the call will hang waiting for a reply that never comes; race it
        // against the capture instead of awaiting it.
        tokio::spawn(async move {
            let _ = client.call(Bytes::from_static(b"hello wire")).await;
        });

        let wire = capture.await.expect("capture task");
        let mut expected = Vec::new();
        LengthDelimitedCodec::default()
            .encode_frame(&&b"hello wire"[..], &mut expected)
            .expect("encode");
        assert_eq!(wire, expected);
    }
}
