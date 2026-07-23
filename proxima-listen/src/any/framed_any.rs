//! [`FramedAny<C>`] — a GENERIC, STATELESS [`AnyProtocol`] driver over any
//! `C: OwnFrame` sans-IO frame codec. Wires the pre-existing
//! `proxima_protocols::codec_pipe::{FrameCodecPipe, OnFrame}` pipes (see
//! that module's doc — "ZERO production callers today") into the open
//! universal listener: read bytes, run the composed
//! `AndThen<FrameCodecPipe<C>, OnFrame<App>>` pipe, write the reply frame,
//! loop. No new pipe-algebra type — this is a shell around two pipes that
//! already exist.
//!
//! Any stateless request/reply wire (memcached, DNS-over-TCP, a
//! stateless kafka-lite framing, ...) supplies `(probe, codec, handler)`
//! and gets a `drive` for free, sharing this ONE driver instead of writing
//! a bespoke `serve_connection`.

use std::future::Future;
use std::pin::Pin;

use bytes::{Buf, Bytes, BytesMut};
use futures::io::{AsyncReadExt, AsyncWriteExt};
use serde_json::Value;

use proxima_codec::FrameCodec;
use proxima_core::ProximaError;
use proxima_primitives::pipe::{AndThen, SendPipe};
use proxima_primitives::stream::{PeerInfo, StreamConnection};
use proxima_protocols::codec_pipe::{FrameCodecPipe, Incomplete, OnFrame, OwnFrame};

use crate::admission::{ConnAdmission, RequestAdmit, ShedReason};
use crate::any::probe::{AnyHandler, AnyProtocol, ProbeVerdict};

/// Bytes read from the socket in one `drive` iteration when nothing else
/// is pending — matches `FramedListenProtocol`'s own default.
const DEFAULT_READ_CHUNK: usize = 64 * 1024;

/// Mirrors [`OwnFrame`] in the opposite direction: bridges a handler's
/// OWNED reply value back to a borrowed `C::Frame<'_>` so
/// [`FrameCodec::encode_frame`] can render it onto the wire. Every codec
/// already splits owned vs. borrowed at parse time (`OwnFrame`); this is
/// the same split, read backwards, for encode — the seam every `App`
/// plugged into [`FramedAny`] must supply for its own reply type.
pub trait AsFrame<C: FrameCodec> {
    /// Borrow `self` as the codec's own frame shape for encoding.
    fn as_frame(&self) -> C::Frame<'_>;
}

/// Gates a wrapped `App` on the listener-wide [`ConnAdmission`] handle
/// before every dispatch, rendering `shed_reply(reason)` instead of calling
/// `App` on `Shed` — the same request_admit/request_release/render-shed
/// contract every other `AnyProtocol::drive` in this codebase follows (see
/// [`AnyProtocol::drive`]'s own doc). Composing this INSIDE `OnFrame` (as
/// `OnFrame<AdmittedApp<App, Shed>>`) keeps `FramedAny::drive`'s pipe
/// literally `AndThen<FrameCodecPipe<C>, OnFrame<_>>` — admission is a
/// wrapper around the handler, not a third pipe stage.
struct AdmittedApp<App, Shed> {
    inner: App,
    admission: ConnAdmission,
    shed_reply: Shed,
}

impl<App, Shed> SendPipe for AdmittedApp<App, Shed>
where
    App: SendPipe,
    App::In: Send,
    Shed: Fn(ShedReason) -> App::Out + Send + Sync + 'static,
{
    type In = App::In;
    type Out = App::Out;
    type Err = App::Err;

    fn call(&self, input: Self::In) -> impl Future<Output = Result<Self::Out, Self::Err>> + Send {
        async move {
            match self.admission.request_admit() {
                RequestAdmit::Admit => {
                    let outcome = self.inner.call(input).await;
                    self.admission.request_release();
                    outcome
                }
                RequestAdmit::Shed { reason } => Ok((self.shed_reply)(reason)),
            }
        }
    }
}

/// A stateless request/reply candidate for the open universal listener,
/// generic over any [`FrameCodec`] `C` (via [`OwnFrame`] + [`Incomplete`]).
/// Construct with [`Self::new`]; `probe` classifies the connection prefix,
/// `app` handles one parsed frame at a time, `shed_reply` renders this
/// wire's own admission-shed rejection.
pub struct FramedAny<C, App, Probe, Shed> {
    label: String,
    codec: C,
    app: App,
    probe: Probe,
    shed_reply: Shed,
    max_prefix_bytes: usize,
    priority: u16,
}

impl<C, App, Probe, Shed> FramedAny<C, App, Probe, Shed> {
    /// `max_prefix_bytes` bounds how many leading bytes `probe` ever needs
    /// to reach a verdict (mirrors [`AnyProtocol::max_prefix_bytes`]).
    #[must_use]
    pub fn new(
        label: impl Into<String>,
        codec: C,
        app: App,
        probe: Probe,
        shed_reply: Shed,
        max_prefix_bytes: usize,
    ) -> Self {
        Self {
            label: label.into(),
            codec,
            app,
            probe,
            shed_reply,
            max_prefix_bytes,
            priority: 100,
        }
    }

    /// Overrides the default priority (100) — see [`AnyProtocol::priority`].
    #[must_use]
    pub fn with_priority(mut self, priority: u16) -> Self {
        self.priority = priority;
        self
    }
}

impl<C, App, Probe, Shed> AnyProtocol for FramedAny<C, App, Probe, Shed>
where
    C: OwnFrame + Clone + Send + Sync + 'static,
    C::Error: Incomplete + core::fmt::Display + Send + Sync + 'static,
    C::Owned: Send + 'static,
    App: SendPipe<In = C::Owned> + Clone + Send + Sync + 'static,
    App::Out: AsFrame<C> + Send + 'static,
    App::Err: From<C::Error> + core::fmt::Debug + Send + Sync + 'static,
    Probe: Fn(&[u8]) -> ProbeVerdict + Send + Sync + 'static,
    Shed: Fn(ShedReason) -> App::Out + Clone + Send + Sync + 'static,
{
    fn name(&self) -> &str {
        &self.label
    }

    fn priority(&self) -> u16 {
        self.priority
    }

    fn max_prefix_bytes(&self) -> usize {
        self.max_prefix_bytes
    }

    fn probe(&self, prefix: &[u8]) -> ProbeVerdict {
        (self.probe)(prefix)
    }

    fn drive<'a>(
        &'a self,
        stream: Box<dyn StreamConnection>,
        _handler: AnyHandler,
        _spec: &'a Value,
        _peer: Option<PeerInfo>,
        admission: &'a ConnAdmission,
    ) -> Pin<Box<dyn Future<Output = Result<(), ProximaError>> + Send + 'a>> {
        Box::pin(async move {
            let admitted = AdmittedApp {
                inner: self.app.clone(),
                admission: admission.clone(),
                shed_reply: self.shed_reply.clone(),
            };
            let pipe = AndThen::new(FrameCodecPipe::new(self.codec.clone()), OnFrame::new(admitted));
            let (mut read_half, mut write_half) = stream.split();
            let mut buf = BytesMut::new();
            let mut scratch = vec![0_u8; DEFAULT_READ_CHUNK];
            let mut out_frame: Vec<u8> = Vec::new();

            loop {
                loop {
                    // re-parse from byte zero every attempt, mirroring
                    // `FrameCodecPipe`'s own one-frame-per-call contract
                    // (`codec_pipe.rs`'s doc) — a caller with more than one
                    // buffered frame advances by `consumed` and calls again.
                    let window = Bytes::copy_from_slice(&buf);
                    match SendPipe::call(&pipe, window).await {
                        Ok(Some((reply, consumed))) => {
                            buf.advance(consumed);
                            out_frame.clear();
                            self.codec
                                .encode_frame(&reply.as_frame(), &mut out_frame)
                                .map_err(|error| {
                                    ProximaError::Upstream(format!(
                                        "framed-any '{}' encode: {error}",
                                        self.label
                                    ))
                                })?;
                            write_half
                                .write_all(&out_frame)
                                .await
                                .map_err(ProximaError::Io)?;
                            write_half.flush().await.map_err(ProximaError::Io)?;
                        }
                        Ok(None) => break,
                        Err(error) => {
                            return Err(ProximaError::Upstream(format!(
                                "framed-any '{}': {error:?}",
                                self.label
                            )));
                        }
                    }
                }
                let read = read_half
                    .read(&mut scratch)
                    .await
                    .map_err(ProximaError::Io)?;
                if read == 0 {
                    return Ok(());
                }
                buf.extend_from_slice(&scratch[..read]);
            }
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use std::net::Ipv4Addr;
    use std::net::SocketAddr;
    use std::sync::Arc;
    use std::task::{Context, Poll};

    use crate::any::{Classifier, ClassifyOutcome};
    use proxima_primitives::stream::{StreamListener, StreamListenerExt};
    use tokio::io::{AsyncReadExt as TokioRead, AsyncWriteExt as TokioWrite};

    /// A trivial line codec proving `FramedAny` over a REAL (not
    /// length-delimited) sans-IO wire: `GREET <name>\r\n` in,
    /// `HELLO <name>\r\n` out. `Frame<'a> = &'a str` on both sides, so
    /// [`AsFrame`] is a bare borrow.
    #[derive(Debug, Clone, Copy, Default)]
    struct GreetCodec;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum GreetError {
        Incomplete,
        Malformed,
    }

    impl std::fmt::Display for GreetError {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                GreetError::Incomplete => formatter.write_str("incomplete: no terminating CRLF yet"),
                GreetError::Malformed => formatter.write_str("malformed: missing GREET prefix"),
            }
        }
    }

    impl std::error::Error for GreetError {}

    impl Incomplete for GreetError {
        fn is_incomplete(&self) -> bool {
            matches!(self, GreetError::Incomplete)
        }
    }

    impl FrameCodec for GreetCodec {
        type Frame<'a> = &'a str;
        type Error = GreetError;

        fn parse_frame<'a>(&self, buf: &'a [u8]) -> Result<(&'a str, usize), GreetError> {
            let Some(terminator) = buf.windows(2).position(|window| window == b"\r\n") else {
                return Err(GreetError::Incomplete);
            };
            let line = std::str::from_utf8(&buf[..terminator]).map_err(|_| GreetError::Malformed)?;
            let name = line.strip_prefix("GREET ").ok_or(GreetError::Malformed)?;
            Ok((name, terminator + 2))
        }

        fn encode_frame(&self, frame: &&str, dest: &mut Vec<u8>) -> Result<(), GreetError> {
            dest.extend_from_slice(b"HELLO ");
            dest.extend_from_slice(frame.as_bytes());
            dest.extend_from_slice(b"\r\n");
            Ok(())
        }
    }

    impl OwnFrame for GreetCodec {
        type Owned = String;

        fn own_frame(_source: &Bytes, frame: &&str) -> String {
            (*frame).to_string()
        }
    }

    /// Owned reply value — a thin wrapper (not a bare `String`) so
    /// [`AsFrame`] can be implemented locally without an orphan-rule
    /// conflict on a foreign type.
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct HelloReply(String);

    impl AsFrame<GreetCodec> for HelloReply {
        fn as_frame(&self) -> &str {
            &self.0
        }
    }

    #[derive(Debug)]
    enum GreetAppError {
        Codec(GreetError),
    }

    impl std::fmt::Display for GreetAppError {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                GreetAppError::Codec(error) => write!(formatter, "codec: {error}"),
            }
        }
    }

    impl std::error::Error for GreetAppError {}

    impl From<GreetError> for GreetAppError {
        fn from(error: GreetError) -> Self {
            GreetAppError::Codec(error)
        }
    }

    /// Uppercases the greeted name — proves the App stage actually ran (not
    /// just an echo of the parsed frame).
    #[derive(Clone, Copy)]
    struct GreetApp;

    impl SendPipe for GreetApp {
        type In = String;
        type Out = HelloReply;
        type Err = GreetAppError;

        fn call(
            &self,
            input: String,
        ) -> impl Future<Output = Result<HelloReply, GreetAppError>> + Send {
            async move { Ok(HelloReply(input.to_uppercase())) }
        }
    }

    fn greet_probe(prefix: &[u8]) -> ProbeVerdict {
        const TAG: &[u8] = b"GREET ";
        let compare_len = prefix.len().min(TAG.len());
        if prefix[..compare_len] != TAG[..compare_len] {
            return ProbeVerdict::No;
        }
        if prefix.len() < TAG.len() {
            return ProbeVerdict::NeedMore {
                at_least: TAG.len(),
            };
        }
        ProbeVerdict::Match { consumed: 0 }
    }

    fn greet_candidate() -> Arc<dyn AnyProtocol> {
        Arc::new(FramedAny::new(
            "greet",
            GreetCodec,
            GreetApp,
            greet_probe,
            |_reason: ShedReason| HelloReply("BUSY".to_string()),
            64,
        ))
    }

    /// A second, unrelated candidate standing in for a sibling protocol
    /// sharing the same open listener (this crate has no h1 of its own —
    /// `proxima-http` depends on `proxima-listen`, not the reverse — so this
    /// proves multi-candidate coexistence with a real second `AnyProtocol`
    /// rather than a no-op stub).
    struct EchoAny;

    impl AnyProtocol for EchoAny {
        fn name(&self) -> &str {
            "echo"
        }

        fn max_prefix_bytes(&self) -> usize {
            5
        }

        fn probe(&self, prefix: &[u8]) -> ProbeVerdict {
            const TAG: &[u8] = b"ECHO ";
            let compare_len = prefix.len().min(TAG.len());
            if prefix[..compare_len] != TAG[..compare_len] {
                return ProbeVerdict::No;
            }
            if prefix.len() < TAG.len() {
                return ProbeVerdict::NeedMore {
                    at_least: TAG.len(),
                };
            }
            ProbeVerdict::Match { consumed: 0 }
        }

        fn drive<'a>(
            &'a self,
            stream: Box<dyn StreamConnection>,
            _handler: AnyHandler,
            _spec: &'a Value,
            _peer: Option<PeerInfo>,
            _admission: &'a ConnAdmission,
        ) -> Pin<Box<dyn Future<Output = Result<(), ProximaError>> + Send + 'a>> {
            Box::pin(async move {
                let mut stream = stream;
                let mut buf = [0_u8; 256];
                let read = stream.read(&mut buf).await.map_err(ProximaError::Io)?;
                stream
                    .write_all(&buf[..read])
                    .await
                    .map_err(ProximaError::Io)?;
                stream.flush().await.map_err(ProximaError::Io)?;
                Ok(())
            })
        }
    }

    /// Replays the bytes the classification loop already consumed off the
    /// wire before any candidate's own parser sees them — a trimmed local
    /// stand-in for `proxima-http`'s `any_listener::PrefixedConnection`
    /// (that one lives behind `proxima-core`'s `io-async-compat` feature,
    /// which this crate's test suite has no other reason to pull in).
    struct PrefixedTestConn {
        prefix: Vec<u8>,
        cursor: usize,
        inner: Box<dyn StreamConnection>,
    }

    impl PrefixedTestConn {
        fn new(prefix: Vec<u8>, inner: Box<dyn StreamConnection>) -> Self {
            Self {
                prefix,
                cursor: 0,
                inner,
            }
        }
    }

    impl futures::io::AsyncRead for PrefixedTestConn {
        fn poll_read(
            mut self: Pin<&mut Self>,
            cx: &mut Context<'_>,
            buf: &mut [u8],
        ) -> Poll<std::io::Result<usize>> {
            if self.cursor < self.prefix.len() {
                let start = self.cursor;
                let count = (self.prefix.len() - start).min(buf.len());
                buf[..count].copy_from_slice(&self.prefix[start..start + count]);
                self.cursor += count;
                return Poll::Ready(Ok(count));
            }
            Pin::new(&mut self.inner).poll_read(cx, buf)
        }
    }

    impl futures::io::AsyncWrite for PrefixedTestConn {
        fn poll_write(
            mut self: Pin<&mut Self>,
            cx: &mut Context<'_>,
            buf: &[u8],
        ) -> Poll<std::io::Result<usize>> {
            Pin::new(&mut self.inner).poll_write(cx, buf)
        }

        fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
            Pin::new(&mut self.inner).poll_flush(cx)
        }

        fn poll_close(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
            Pin::new(&mut self.inner).poll_close(cx)
        }
    }

    impl StreamConnection for PrefixedTestConn {
        fn peer(&self) -> Option<PeerInfo> {
            None
        }
    }

    async fn classify_and_drive_once(
        mut conn: Box<dyn StreamConnection>,
        candidates: Arc<[Arc<dyn AnyProtocol>]>,
        admission: &ConnAdmission,
    ) {
        let mut classifier = Classifier::new(candidates, 4096);
        let mut accumulated: Vec<u8> = Vec::new();
        let mut chunk = [0_u8; 256];
        let matched = loop {
            let read = conn.read(&mut chunk).await.expect("classify read");
            assert!(read > 0, "peer closed before a candidate resolved");
            accumulated.extend_from_slice(&chunk[..read]);
            match classifier.advance(&accumulated) {
                ClassifyOutcome::Matched(protocol) => break protocol,
                ClassifyOutcome::NeedMoreBytes { .. } => continue,
                other => panic!("expected a match, got {other:?}"),
            }
        };
        let handler = crate::any::erase_handler(());
        let prefixed: Box<dyn StreamConnection> = Box::new(PrefixedTestConn::new(accumulated, conn));
        matched
            .drive(prefixed, handler, &Value::Null, None, admission)
            .await
            .expect("drive");
    }

    async fn bind_loopback() -> (TokioTcpListener, SocketAddr) {
        let listener = TokioTcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))
            .await
            .expect("bind");
        let addr = match listener.local_addr().expect("local_addr") {
            proxima_primitives::stream::BindAddr::Tcp(addr) => addr,
            other => panic!("expected tcp, got {other:?}"),
        };
        (listener, addr)
    }

    use proxima_net::tokio::tokio_stream_listener::TokioTcpListener;

    #[proxima::test]
    async fn framed_any_serves_a_greet_reply_over_a_real_socket() {
        let (listener, addr) = bind_loopback().await;
        let candidates: Arc<[Arc<dyn AnyProtocol>]> = Arc::from(vec![greet_candidate(), Arc::new(EchoAny) as _]);
        let admission = ConnAdmission::unbounded();

        let client = tokio::spawn(async move {
            let mut stream = tokio::net::TcpStream::connect(addr)
                .await
                .expect("client connect");
            stream
                .write_all(b"GREET alice\r\n")
                .await
                .expect("write greet");
            let mut reply = vec![0_u8; 13];
            stream.read_exact(&mut reply).await.expect("read reply");
            assert_eq!(&reply, b"HELLO ALICE\r\n");
        });

        let conn: Box<dyn StreamConnection> = Box::new(listener.accept().await.expect("accept"));
        classify_and_drive_once(conn, candidates, &admission).await;
        client.await.expect("client task");
    }

    #[proxima::test]
    async fn a_second_any_protocol_candidate_still_routes_on_the_same_classifier() {
        let (listener, addr) = bind_loopback().await;
        let candidates: Arc<[Arc<dyn AnyProtocol>]> = Arc::from(vec![greet_candidate(), Arc::new(EchoAny) as _]);
        let admission = ConnAdmission::unbounded();

        let client = tokio::spawn(async move {
            let mut stream = tokio::net::TcpStream::connect(addr)
                .await
                .expect("client connect");
            stream
                .write_all(b"ECHO ping")
                .await
                .expect("write echo");
            let mut reply = vec![0_u8; 9];
            stream.read_exact(&mut reply).await.expect("read reply");
            assert_eq!(&reply, b"ECHO ping");
        });

        let conn: Box<dyn StreamConnection> = Box::new(listener.accept().await.expect("accept"));
        classify_and_drive_once(conn, candidates, &admission).await;
        client.await.expect("client task");
    }

    #[proxima::test]
    async fn framed_any_renders_the_shed_reply_while_admission_is_draining() {
        let (listener, addr) = bind_loopback().await;
        let candidates: Arc<[Arc<dyn AnyProtocol>]> = Arc::from(vec![greet_candidate()]);
        let admission = ConnAdmission::unbounded();
        admission.begin_drain();

        let client = tokio::spawn(async move {
            let mut stream = tokio::net::TcpStream::connect(addr)
                .await
                .expect("client connect");
            stream
                .write_all(b"GREET bob\r\n")
                .await
                .expect("write greet");
            let mut reply = vec![0_u8; 12];
            stream.read_exact(&mut reply).await.expect("read reply");
            assert_eq!(&reply, b"HELLO BUSY\r\n");
        });

        let conn: Box<dyn StreamConnection> = Box::new(listener.accept().await.expect("accept"));
        classify_and_drive_once(conn, candidates, &admission).await;
        client.await.expect("client task");
    }

    #[test]
    fn greet_probe_matches_and_rejects_correctly() {
        assert_eq!(greet_probe(b"GREET a"), ProbeVerdict::Match { consumed: 0 });
        assert_eq!(greet_probe(b"GRE"), ProbeVerdict::NeedMore { at_least: 6 });
        assert_eq!(greet_probe(b"OTHER"), ProbeVerdict::No);
    }
}
