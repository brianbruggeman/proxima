//! `FramedListenProtocol` — a length-delimited `[u32 BE len][payload]`
//! request/reply `ListenProtocol`. Each inbound frame becomes a
//! `Request` whose body is the frame bytes; the `Pipe`'s `Response`
//! body is written back as one frame, and the loop continues on the
//! same connection (multi-round-trip).
//!
//! Bytes-centric: framing is [`proxima_codec::LengthDelimitedCodec`]
//! (sans-IO, zero-copy parse). Any typing is the `Pipe`'s concern at
//! the edge — this listener moves only bytes. Per-connection handlers
//! are `spawn_local`'d so a Send *or* `!Send` `Pipe::call` stays pinned
//! to the accepting core (no work-stealing); the `Pipe` being Send is
//! the portable case and is all the consumer needs.

use std::future::Future;
use std::io;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::{Bytes, BytesMut};
use futures::channel::oneshot;
use futures::io::{AsyncReadExt, AsyncWriteExt};
use serde_json::Value;
use tracing::{debug, warn};

use proxima_codec::{FrameCodec, FrameError, FrameLimits, LengthDelimitedCodec};
use proxima_core::ProximaError;
use crate::{ListenProtocol, ServeContext};
use proxima_net::tokio::tokio_stream_listener::{TokioTcpConnection, TokioTcpListener};
use proxima_primitives::pipe::Method;
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::header_list::HeaderList;
use proxima_primitives::pipe::handler::PipeHandle;
use proxima_primitives::pipe::request::{Request, RequestContext};
#[cfg(feature = "tokio")]
use proxima_primitives::stream::StreamListenerExt;
use proxima_primitives::stream::StreamConnection;

const DEFAULT_METHOD: &str = "FRAME";
const DEFAULT_PATH: &str = "/";
const DEFAULT_READ_CHUNK: usize = 64 * 1024;

/// Length-delimited request/reply listener. Construct with [`Self::new`],
/// tweak the synthetic request envelope with [`Self::with_method`] /
/// [`Self::with_path`], and register on an `App` via
/// `with_listen_protocol`. Per-serve knobs (`max_frame_bytes`,
/// `reject_zero_len`, `idle_timeout_ms`, `read_chunk_bytes`, `method`,
/// `path`) are read from the listener `spec` so the control plane can
/// tune a deployment without a recompile.
/// Wraps an accepted connection before framing — e.g. a consumer plugging in
/// a cipher so the length prefix itself rides inside an encrypted transport.
/// Erased to `Box<dyn StreamConnection>` so the listener never names the
/// consumer's concrete wrapper type.
pub type ConnTransform = Arc<dyn Fn(TokioTcpConnection) -> Box<dyn StreamConnection> + Send + Sync>;

pub struct FramedListenProtocol {
    label: String,
    method: String,
    path: String,
    conn_transform: Option<ConnTransform>,
}

impl FramedListenProtocol {
    #[must_use]
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            method: DEFAULT_METHOD.into(),
            path: DEFAULT_PATH.into(),
            conn_transform: None,
        }
    }

    /// Install a connection wrapper (e.g. a cipher) applied to every accepted
    /// socket before the frame loop runs. The framed length prefix then rides
    /// inside the wrapped transport — the consumer owns the crypto; the
    /// listener stays generic.
    #[must_use]
    pub fn with_conn_transform(mut self, transform: ConnTransform) -> Self {
        self.conn_transform = Some(transform);
        self
    }

    #[must_use]
    pub fn with_method(mut self, method: impl Into<String>) -> Self {
        self.method = method.into();
        self
    }

    #[must_use]
    pub fn with_path(mut self, path: impl Into<String>) -> Self {
        self.path = path.into();
        self
    }
}

impl ListenProtocol for FramedListenProtocol {
    fn name(&self) -> &str {
        &self.label
    }

    fn serve(
        &self,
        bind: SocketAddr,
        dispatch: PipeHandle,
        spec: &Value,
        context: ServeContext,
        mut shutdown: oneshot::Receiver<()>,
    ) -> Pin<Box<dyn Future<Output = Result<(), ProximaError>> + Send + '_>> {
        let label = self.label.clone();
        let method = Bytes::from(
            spec.get("method")
                .and_then(Value::as_str)
                .unwrap_or(&self.method)
                .to_string(),
        );
        let path = Bytes::from(
            spec.get("path")
                .and_then(Value::as_str)
                .unwrap_or(&self.path)
                .to_string(),
        );
        let max_frame_bytes = spec
            .get("max_frame_bytes")
            .and_then(Value::as_u64)
            .map(|raw| raw as usize)
            .unwrap_or(FrameLimits::DEFAULT_MAX_FRAME_BYTES);
        let reject_zero_len = spec
            .get("reject_zero_len")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let idle_ms = spec
            .get("idle_timeout_ms")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let read_chunk = spec
            .get("read_chunk_bytes")
            .and_then(Value::as_u64)
            .map(|raw| raw.max(1) as usize)
            .unwrap_or(DEFAULT_READ_CHUNK);
        let max_fps = spec
            .get("max_frames_per_second")
            .and_then(Value::as_u64)
            .map(|raw| raw.min(u64::from(u32::MAX)) as u32)
            .unwrap_or(0);
        // per-core fan-out: when App's listener runner marks the spec, each
        // lane binds the same addr with SO_REUSEPORT so the kernel
        // load-balances accepts across cores (mirrors the http listener).
        let use_reuseport = spec
            .get(crate::handle::REUSEPORT_SPEC_KEY)
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let codec = LengthDelimitedCodec::new(FrameLimits::new(max_frame_bytes, reject_zero_len));
        let idle = (idle_ms > 0).then(|| Duration::from_millis(idle_ms));
        let conn_transform = self.conn_transform.clone();
        let ready_signal = context.ready_signal.clone();

        Box::pin(async move {
            let listener = if use_reuseport {
                let tokio_listener = crate::handle::bind_reuseport_listener(bind)
                    .map_err(|err| {
                        ProximaError::Io(io::Error::other(format!(
                            "{label} reuseport bind {bind}: {err}"
                        )))
                    })?;
                TokioTcpListener::from_tokio_listener(tokio_listener)
            } else {
                TokioTcpListener::bind(bind).await.map_err(|err| {
                    ProximaError::Io(io::Error::other(format!("{label} bind {bind}: {err}")))
                })?
            };
            if let Some(sender) = ready_signal {
                let _ = sender.send(());
            }
            debug!(label = %label, %bind, use_reuseport, "framed listener bound");
            loop {
                tokio::select! {
                    outcome = listener.accept() => match outcome {
                        Ok(conn) => match &conn_transform {
                            Some(transform) => spawn_framed_handler(
                                transform(conn),
                                dispatch.clone(), codec, idle,
                                method.clone(), path.clone(), read_chunk, max_fps, label.clone(),
                            ),
                            None => spawn_framed_handler(
                                conn,
                                dispatch.clone(), codec, idle,
                                method.clone(), path.clone(), read_chunk, max_fps, label.clone(),
                            ),
                        },
                        Err(error) => warn!(?error, label = %label, "framed listener accept error"),
                    },
                    _ = &mut shutdown => return Ok(()),
                }
            }
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn spawn_framed_handler<C: StreamConnection>(
    conn: C,
    dispatch: PipeHandle,
    codec: LengthDelimitedCodec,
    idle: Option<Duration>,
    method: Bytes,
    path: Bytes,
    read_chunk: usize,
    max_fps: u32,
    label: String,
) {
    // spawn_local: pins the per-conn future (and any `?Send` Pipe::call) to
    // the accepting core. See default_listener::spawn_handler.
    tokio::task::spawn_local(async move {
        if let Err(error) = handle_framed_connection(
            conn, dispatch, codec, idle, method, path, read_chunk, max_fps,
        )
        .await
        {
            debug!(?error, label = %label, "framed connection ended");
        }
    });
}

#[allow(clippy::too_many_arguments)]
async fn handle_framed_connection<C: StreamConnection>(
    conn: C,
    dispatch: PipeHandle,
    codec: LengthDelimitedCodec,
    idle: Option<Duration>,
    method: Bytes,
    path: Bytes,
    read_chunk: usize,
    max_fps: u32,
) -> Result<(), ProximaError> {
    let (mut read_half, mut write_half) = conn.split();
    let mut buf = BytesMut::with_capacity(read_chunk);
    let mut scratch = vec![0_u8; read_chunk.max(1)];
    let mut out_frame: Vec<u8> = Vec::new();
    let mut rate = FrameRate::new(max_fps);

    loop {
        let payload = loop {
            match codec.parse_frame(&buf) {
                Ok((_frame, consumed)) => {
                    // zero-copy: the frame Bytes shares the read buffer's
                    // allocation; slice off the 4-byte length prefix.
                    let whole = buf.split_to(consumed).freeze();
                    break whole.slice(LengthDelimitedCodec::HEADER_BYTES..);
                }
                Err(FrameError::Incomplete) => {
                    let read = read_chunk_with_idle(&mut read_half, &mut scratch, idle).await?;
                    if read == 0 {
                        if buf.is_empty() {
                            return Ok(());
                        }
                        return Err(ProximaError::Io(io::Error::new(
                            io::ErrorKind::UnexpectedEof,
                            "connection closed mid-frame",
                        )));
                    }
                    buf.extend_from_slice(&scratch[..read]);
                }
                Err(err) => {
                    // zero-length / over-cap → close the connection, matching
                    // a framed RPC server that rejects the frame at the wire.
                    return Err(ProximaError::Io(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("{err}"),
                    )));
                }
            }
        };

        if !rate.admit() {
            warn!(
                limit = max_fps,
                "framed connection closed: frame rate exceeded"
            );
            return Ok(());
        }

        let request = Request {
            method: Method::from_wire(method.clone()),
            path: path.clone(),
            query: HeaderList::new(),
            metadata: HeaderList::new(),
            payload,
            stream: None,
            context: RequestContext::default(),
        };
        let response = SendPipe::call(&dispatch, request).await?;
        let reply = response.collect_body().await?;

        out_frame.clear();
        let reply_slice: &[u8] = reply.as_ref();
        codec
            .encode_frame(&reply_slice, &mut out_frame)
            .map_err(|err| {
                ProximaError::Io(io::Error::new(io::ErrorKind::InvalidData, format!("{err}")))
            })?;
        write_half
            .write_all(&out_frame)
            .await
            .map_err(|err| ProximaError::Io(io::Error::other(format!("frame write: {err}"))))?;
        write_half
            .flush()
            .await
            .map_err(|err| ProximaError::Io(io::Error::other(format!("frame flush: {err}"))))?;
    }
}

async fn read_chunk_with_idle<R>(
    reader: &mut R,
    scratch: &mut [u8],
    idle: Option<Duration>,
) -> Result<usize, ProximaError>
where
    R: AsyncReadExt + Unpin,
{
    let outcome = match idle {
        Some(duration) => proxima_core::time::timeout(duration, reader.read(scratch))
            .await
            .map_err(|_elapsed| {
                ProximaError::Io(io::Error::new(io::ErrorKind::TimedOut, "idle timeout"))
            })?,
        None => reader.read(scratch).await,
    };
    outcome.map_err(|err| ProximaError::Io(io::Error::other(format!("frame read: {err}"))))
}

/// Per-connection frame-rate gate. A zero limit is unlimited. Mirrors the
/// incumbent's `ConnectionFrameRate`: a fixed 1-second window; exceeding the
/// limit closes the connection rather than erroring.
struct FrameRate {
    limit: u32,
    window: Duration,
    window_start: Instant,
    count: u32,
}

impl FrameRate {
    fn new(limit: u32) -> Self {
        Self {
            limit,
            window: Duration::from_secs(1),
            window_start: Instant::now(),
            count: 0,
        }
    }

    fn admit(&mut self) -> bool {
        if self.limit == 0 {
            return true;
        }
        let now = Instant::now();
        if now.duration_since(self.window_start) >= self.window {
            self.window_start = now;
            self.count = 0;
        }
        if self.count >= self.limit {
            return false;
        }
        self.count += 1;
        true
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use proxima_primitives::pipe::handler::into_handle;
    use proxima_primitives::pipe::request::Response as ProximaResponse;
    use proxima_primitives::stream::StreamListener;
    use std::net::Ipv4Addr;
    use tokio::io::{AsyncReadExt as TokioRead, AsyncWriteExt as TokioWrite};

    // echoes the request frame back, uppercased — proves bytes flow in and
    // out per frame and that multiple frames round-trip on one connection.
    struct UppercasePipe;

    impl SendPipe for UppercasePipe {
        type In = Request<Bytes>;
        type Out = ProximaResponse<Bytes>;
        type Err = ProximaError;

        fn call(
            &self,
            request: Request<Bytes>,
        ) -> impl Future<Output = Result<ProximaResponse<Bytes>, ProximaError>> + Send {
            async move {
                let (_, bytes) = request.body_bytes().await?;
                let upper: Vec<u8> = bytes.iter().map(u8::to_ascii_uppercase).collect();
                Ok(ProximaResponse {
                    status: 200,
                    metadata: HeaderList::new(),
                    payload: Bytes::from(upper),
                    stream: None,
                    upgrade: None,
                })
            }
        }
    }


    async fn send_frame(stream: &mut tokio::net::TcpStream, payload: &[u8]) {
        let len = u32::try_from(payload.len()).unwrap();
        stream.write_all(&len.to_be_bytes()).await.unwrap();
        stream.write_all(payload).await.unwrap();
        stream.flush().await.unwrap();
    }

    async fn recv_frame(stream: &mut tokio::net::TcpStream) -> Vec<u8> {
        let mut len_buf = [0_u8; 4];
        stream.read_exact(&mut len_buf).await.unwrap();
        let len = u32::from_be_bytes(len_buf) as usize;
        let mut payload = vec![0_u8; len];
        stream.read_exact(&mut payload).await.unwrap();
        payload
    }

    #[proxima::test]
    async fn framed_round_trips_multiple_frames_on_one_connection() {
        let listener = TokioTcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))
            .await
            .expect("bind");
        let local = match listener.local_addr().expect("local_addr") {
            proxima_primitives::stream::BindAddr::Tcp(addr) => addr,
            other => panic!("expected tcp, got {other:?}"),
        };

        let dispatch = into_handle(UppercasePipe);
        let codec = LengthDelimitedCodec::default();

        let client = tokio::spawn(async move {
            let mut stream = tokio::net::TcpStream::connect(local)
                .await
                .expect("client connect");
            send_frame(&mut stream, b"the quick brown fox").await;
            assert_eq!(recv_frame(&mut stream).await, b"THE QUICK BROWN FOX");
            send_frame(&mut stream, b"second frame").await;
            assert_eq!(recv_frame(&mut stream).await, b"SECOND FRAME");
            stream.shutdown().await.expect("client shutdown");
        });

        let conn = listener.accept().await.expect("accept");
        handle_framed_connection(
            conn,
            dispatch,
            codec,
            None,
            Bytes::from_static(b"FRAME"),
            Bytes::from_static(b"/"),
            32,
            0,
        )
        .await
        .expect("framed handler");

        client.await.expect("client task");
    }

    #[proxima::test]
    async fn framed_rejects_zero_length_frame_when_configured() {
        let listener = TokioTcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))
            .await
            .expect("bind");
        let local = match listener.local_addr().expect("local_addr") {
            proxima_primitives::stream::BindAddr::Tcp(addr) => addr,
            other => panic!("expected tcp, got {other:?}"),
        };

        let dispatch = into_handle(UppercasePipe);
        let codec = LengthDelimitedCodec::new(FrameLimits::new(1024, true));

        let client = tokio::spawn(async move {
            let mut stream = tokio::net::TcpStream::connect(local)
                .await
                .expect("client connect");
            // a zero-length frame: 4 zero bytes, no payload.
            stream.write_all(&0_u32.to_be_bytes()).await.unwrap();
            stream.flush().await.unwrap();
            // server must close; the read returns 0 bytes.
            let mut buf = [0_u8; 1];
            stream.read(&mut buf).await.unwrap_or(0)
        });

        let conn = listener.accept().await.expect("accept");
        let outcome = handle_framed_connection(
            conn,
            dispatch,
            codec,
            None,
            Bytes::new(),
            Bytes::new(),
            32,
            0,
        )
        .await;
        assert!(
            outcome.is_err(),
            "zero-length frame must close the connection"
        );
        let read = client.await.expect("client task");
        assert_eq!(read, 0, "client should observe a closed connection");
    }

    #[test]
    fn frame_rate_admits_to_limit_then_rejects_in_window() {
        let mut rate = FrameRate::new(2);
        assert!(rate.admit());
        assert!(rate.admit());
        assert!(
            !rate.admit(),
            "third frame in the same window must be rejected"
        );
    }

    #[test]
    fn frame_rate_zero_limit_is_unlimited() {
        let mut rate = FrameRate::new(0);
        assert!((0..10_000).all(|_| rate.admit()));
    }

    #[proxima::test]
    async fn framed_closes_connection_when_frame_rate_exceeded() {
        let listener = TokioTcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))
            .await
            .expect("bind");
        let local = match listener.local_addr().expect("local_addr") {
            proxima_primitives::stream::BindAddr::Tcp(addr) => addr,
            other => panic!("expected tcp, got {other:?}"),
        };

        let dispatch = into_handle(UppercasePipe);
        let codec = LengthDelimitedCodec::default();

        let client = tokio::spawn(async move {
            let mut stream = tokio::net::TcpStream::connect(local)
                .await
                .expect("client connect");
            send_frame(&mut stream, b"one").await;
            assert_eq!(recv_frame(&mut stream).await, b"ONE");
            send_frame(&mut stream, b"two").await;
            assert_eq!(recv_frame(&mut stream).await, b"TWO");
            // third frame in the same 1s window exceeds max_fps=2 → server closes.
            send_frame(&mut stream, b"three").await;
            let mut len_buf = [0_u8; 4];
            stream.read(&mut len_buf).await.unwrap_or(0)
        });

        let conn = listener.accept().await.expect("accept");
        handle_framed_connection(
            conn,
            dispatch,
            codec,
            None,
            Bytes::from_static(b"FRAME"),
            Bytes::from_static(b"/"),
            32,
            2,
        )
        .await
        .expect("framed handler");
        let read = client.await.expect("client task");
        assert_eq!(read, 0, "rate-exceeded connection should close");
    }
}
