//! `PgWireConnectionPipe` — the connection layer as a real `Pipe`.
//!
//! A freshly accepted socket arrives as a `CONNECT` request; the pipe
//! answers with `Response.upgrade` (the `proxima_primitives::pipe::upgrade`
//! raw-socket-hijack seam). The upgrade handler runs the startup
//! negotiation and the session loop over the hijacked stream, calling the
//! engine `query` pipe once per protocol operation. This is the RISC
//! payoff: connection handling is a `Pipe`, query handling is a `Pipe`,
//! and every proxima middleware composes onto both.

use std::sync::Arc;

use futures::io::{AsyncRead, AsyncWrite};
// the one-byte direct-TLS peek is the only AsyncReadExt user; with tls off
// there is no peek, so the import would be dead
#[cfg(feature = "tls")]
use futures::io::AsyncReadExt;

use bytes::Bytes;
use proxima_core::ProximaError;
#[cfg(feature = "tls")]
use proxima_core::io::{FromFutures, IntoFutures, Prepend};
use proxima_protocols::pgwire_codec::Session;
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::request::{Request, Response};
use proxima_primitives::pipe::upgrade::{HijackedSocket, UpgradeHandler};

use crate::pipes::PgPipeHandle;

use crate::auth::PgAuth;
use crate::broker::NotifyBroker;
use crate::config::PgServerConfig;
use crate::connection::{CancelRegistry, Negotiation, RuntimeHandle, negotiate, serve_session};
use crate::error::ServeError;

#[cfg(feature = "tls")]
type TlsAcceptor = futures_rustls::TlsAcceptor;
#[cfg(not(feature = "tls"))]
type TlsAcceptor = ();

/// The connection layer as a `Pipe`. Its `call` (for a `CONNECT` request)
/// returns a `Response.upgrade` that negotiates startup and runs the
/// session loop over the hijacked stream, calling `query` per protocol
/// operation. TLS, when configured, is handled inside the upgrade: the
/// pipe answers SSLRequest, wraps the stream, and re-negotiates — so the
/// entire connection lifecycle, plaintext or TLS, is one upgrade.
#[derive(Clone)]
pub struct PgWireConnectionPipe {
    query: PgPipeHandle,
    auth: PgAuth,
    config: Arc<PgServerConfig>,
    registry: Arc<CancelRegistry>,
    broker: Arc<NotifyBroker>,
    tls: Option<TlsAcceptor>,
    runtime: RuntimeHandle,
    label: String,
}

impl PgWireConnectionPipe {
    #[must_use]
    pub fn new(
        label: impl Into<String>,
        query: PgPipeHandle,
        auth: PgAuth,
        config: Arc<PgServerConfig>,
        registry: Arc<CancelRegistry>,
    ) -> Self {
        Self {
            query,
            auth,
            config,
            registry,
            // one broker per pipe, shared by Arc across every connection it
            // upgrades — that shared instance IS the LISTEN/NOTIFY fabric
            broker: Arc::new(NotifyBroker::new()),
            tls: None,
            runtime: None,
            label: label.into(),
        }
    }

    /// Installs the TLS acceptor the upgrade uses to answer SSLRequest.
    /// Without it the pipe is plaintext and refuses SSL.
    #[cfg(feature = "tls")]
    #[must_use]
    pub fn with_tls(mut self, acceptor: Option<TlsAcceptor>) -> Self {
        self.tls = acceptor;
        self
    }

    /// Installs the runtime whose background-blocking pool the SCRAM KDF
    /// offloads onto, keeping the ~0.5-1ms PBKDF2 off the reactor core.
    /// Without it the KDF runs inline.
    #[must_use]
    pub fn with_runtime(mut self, runtime: RuntimeHandle) -> Self {
        self.runtime = runtime;
        self
    }

    /// This pipe's label, set at construction (TARGET 3 — served-Pipe
    /// naming now lives at the mount-site label, not the handle).
    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
    }
}

fn tls_available(tls: &Option<TlsAcceptor>) -> bool {
    tls.is_some()
}

/// First byte of a TLS handshake record (ContentType = handshake). A
/// PostgreSQL 17 `sslnegotiation=direct` client opens with a ClientHello
/// and no SSLRequest, so this byte is the witness; a startup packet begins
/// with the high byte of its length, always `0x00` for any sane packet.
#[cfg(feature = "tls")]
const TLS_HANDSHAKE_CONTENT_TYPE: u8 = 0x16;

/// Classifies the first connection byte as a direct-TLS ClientHello.
#[cfg(feature = "tls")]
#[must_use]
fn is_direct_tls_first_byte(byte: u8) -> bool {
    byte == TLS_HANDSHAKE_CONTENT_TYPE
}

/// Runs the full session lifecycle over one hijacked stream.
///
/// First peeks one byte: a PostgreSQL 17 `sslnegotiation=direct` client
/// opens with a TLS ClientHello (`0x16`) and no SSLRequest, so that byte
/// routes straight to a TLS accept; any other first byte is a startup
/// packet handled by [`negotiate`] (SSLRequest-or-plaintext) as before.
// `mut stream` is consumed by the tls-gated direct-TLS peek; with tls off
// there is no peek and the binding is moved unmutated
#[cfg_attr(not(feature = "tls"), allow(unused_mut))]
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors the connection lifecycle; bundling would invent a type with no other consumer"
)]
async fn drive_session<S>(
    mut stream: S,
    query: PgPipeHandle,
    auth: PgAuth,
    config: Arc<PgServerConfig>,
    registry: Arc<CancelRegistry>,
    broker: Arc<NotifyBroker>,
    tls: Option<TlsAcceptor>,
    runtime: RuntimeHandle,
) -> Result<(), ServeError>
where
    S: AsyncRead + AsyncWrite + Unpin + Send,
{
    #[cfg(feature = "tls")]
    if let Some(byte) = peek_first_byte(&mut stream).await? {
        if is_direct_tls_first_byte(byte) {
            return serve_direct_tls(
                byte, stream, query, auth, config, registry, broker, tls, runtime,
            )
            .await;
        }
        let stream = IntoFutures(Prepend::new(vec![byte], FromFutures(stream)));
        return negotiate_and_serve(stream, query, auth, config, registry, broker, tls, runtime)
            .await;
    }

    negotiate_and_serve(stream, query, auth, config, registry, broker, tls, runtime).await
}

/// Reads exactly one byte off the wire to classify the connection; `None`
/// means the peer closed before sending anything.
#[cfg(feature = "tls")]
async fn peek_first_byte<S>(stream: &mut S) -> Result<Option<u8>, ServeError>
where
    S: AsyncRead + Unpin + Send,
{
    let mut byte = [0_u8; 1];
    let read = stream.read(&mut byte).await?;
    if read == 0 {
        return Ok(None);
    }
    Ok(Some(byte[0]))
}

/// The original lifecycle: negotiate startup (offering TLS when an
/// acceptor is present), then serve.
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors the connection lifecycle; bundling would invent a type with no other consumer"
)]
async fn negotiate_and_serve<S>(
    stream: S,
    query: PgPipeHandle,
    auth: PgAuth,
    config: Arc<PgServerConfig>,
    registry: Arc<CancelRegistry>,
    broker: Arc<NotifyBroker>,
    tls: Option<TlsAcceptor>,
    runtime: RuntimeHandle,
) -> Result<(), ServeError>
where
    S: AsyncRead + AsyncWrite + Unpin + Send,
{
    let mut session = Session::new();
    match negotiate(stream, &mut session, tls_available(&tls)).await? {
        Negotiation::Proceed {
            stream,
            startup,
            leftover,
        } => {
            serve_session(
                stream,
                session,
                startup,
                leftover,
                query,
                &auth,
                &config,
                Some(registry),
                Some(broker),
                runtime,
            )
            .await
        }
        Negotiation::Cancel {
            process_id,
            secret_key,
        } => {
            registry.cancel(process_id, &secret_key);
            Ok(())
        }
        Negotiation::Closed => Ok(()),
        Negotiation::StartTls(stream) => {
            serve_tls(
                stream, session, query, auth, config, registry, broker, tls, runtime,
            )
            .await
        }
    }
}

#[cfg(feature = "tls")]
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors the connection lifecycle; bundling would invent a type with no other consumer"
)]
async fn serve_tls<S>(
    stream: S,
    mut session: Session,
    query: PgPipeHandle,
    auth: PgAuth,
    config: Arc<PgServerConfig>,
    registry: Arc<CancelRegistry>,
    broker: Arc<NotifyBroker>,
    tls: Option<TlsAcceptor>,
    runtime: RuntimeHandle,
) -> Result<(), ServeError>
where
    S: AsyncRead + AsyncWrite + Unpin + Send,
{
    let Some(acceptor) = tls else {
        return Err(ServeError::Config(
            "ssl accepted without a tls acceptor".into(),
        ));
    };
    let tls_stream = acceptor.accept(stream).await?;
    session.tls_established()?;
    match negotiate(tls_stream, &mut session, false).await? {
        Negotiation::Proceed {
            stream,
            startup,
            leftover,
        } => {
            serve_session(
                stream,
                session,
                startup,
                leftover,
                query,
                &auth,
                &config,
                Some(registry),
                Some(broker),
                runtime,
            )
            .await
        }
        Negotiation::Cancel {
            process_id,
            secret_key,
        } => {
            registry.cancel(process_id, &secret_key);
            Ok(())
        }
        Negotiation::Closed => Ok(()),
        Negotiation::StartTls(_) => Err(ServeError::Config(
            "second ssl request inside the tls tunnel".into(),
        )),
    }
}

/// Handles a PostgreSQL 17 direct-TLS connection: the client already sent
/// a ClientHello (no SSLRequest), so the peeked first byte is replayed to
/// the acceptor, then startup runs over the tunnel. A fresh `Session`
/// (no SSL state transition happened on the wire) drives the post-TLS
/// startup, mirroring how the SSLRequest path re-negotiates after accept.
/// Without a configured acceptor a ClientHello cannot proceed, so the
/// connection is dropped.
#[cfg(feature = "tls")]
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors the connection lifecycle; bundling would invent a type with no other consumer"
)]
async fn serve_direct_tls<S>(
    first_byte: u8,
    stream: S,
    query: PgPipeHandle,
    auth: PgAuth,
    config: Arc<PgServerConfig>,
    registry: Arc<CancelRegistry>,
    broker: Arc<NotifyBroker>,
    tls: Option<TlsAcceptor>,
    runtime: RuntimeHandle,
) -> Result<(), ServeError>
where
    S: AsyncRead + AsyncWrite + Unpin + Send,
{
    let Some(acceptor) = tls else {
        return Ok(());
    };
    let prefixed = IntoFutures(Prepend::new(vec![first_byte], FromFutures(stream)));
    let tls_stream = acceptor.accept(prefixed).await?;
    let mut session = Session::new();
    match negotiate(tls_stream, &mut session, false).await? {
        Negotiation::Proceed {
            stream,
            startup,
            leftover,
        } => {
            serve_session(
                stream,
                session,
                startup,
                leftover,
                query,
                &auth,
                &config,
                Some(registry),
                Some(broker),
                runtime,
            )
            .await
        }
        Negotiation::Cancel {
            process_id,
            secret_key,
        } => {
            registry.cancel(process_id, &secret_key);
            Ok(())
        }
        Negotiation::Closed => Ok(()),
        Negotiation::StartTls(_) => Err(ServeError::Config(
            "ssl request inside a direct-tls tunnel".into(),
        )),
    }
}

#[cfg(not(feature = "tls"))]
#[expect(
    clippy::too_many_arguments,
    reason = "signature parity with the tls variant"
)]
async fn serve_tls<S>(
    _stream: S,
    _session: Session,
    _query: PgPipeHandle,
    _auth: PgAuth,
    _config: Arc<PgServerConfig>,
    _registry: Arc<CancelRegistry>,
    _broker: Arc<NotifyBroker>,
    _tls: Option<TlsAcceptor>,
    _runtime: RuntimeHandle,
) -> Result<(), ServeError>
where
    S: AsyncRead + AsyncWrite + Unpin + Send,
{
    Err(ServeError::Config("tls feature disabled".into()))
}

impl SendPipe for PgWireConnectionPipe {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    async fn call(&self, _request: Request<Bytes>) -> Result<Response<Bytes>, ProximaError> {
        let query = self.query.clone();
        let auth = self.auth.clone();
        let config = Arc::clone(&self.config);
        let registry = Arc::clone(&self.registry);
        let broker = Arc::clone(&self.broker);
        // with tls off the acceptor alias is `()`, so `Option<()>` is Copy and
        // clippy flags the clone; it's a real clone of a TlsAcceptor when on
        #[cfg_attr(not(feature = "tls"), allow(clippy::clone_on_copy))]
        let tls = self.tls.clone();
        let runtime = self.runtime.clone();
        let handler = UpgradeHandler::new(move |hijacked: HijackedSocket| async move {
            let HijackedSocket { stream, leftover } = hijacked;
            if !leftover.is_empty() {
                // a raw pgwire socket has no prior protocol head, so the
                // upgrade seam should never hand us pre-buffered bytes;
                // dropping them silently would corrupt the startup phase
                return Err(ProximaError::Upstream(
                    "pgwire upgrade received pre-buffered bytes before startup".into(),
                ));
            }
            drive_session(stream, query, auth, config, registry, broker, tls, runtime)
                .await
                .map_err(|error| ProximaError::Upstream(format!("pgwire session: {error}")))
        });
        Ok(Response::new(200).with_upgrade(handler))
    }
}


#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::pin::Pin;
    use std::sync::Arc;
    use std::task::{Context, Poll};

    use bytes::Bytes;
    use futures::io::{AsyncRead, AsyncWrite};
    use proxima_core::ProximaError;
    use proxima_protocols::pgwire_codec::Oid;
    use proxima_primitives::pipe::request::{Request, RequestContext, Response};
    use proxima_primitives::pipe::upgrade::HijackedSocket;

    use crate::auth::PgAuth;
    use crate::config::PgServerConfig;
    use crate::connection::CancelRegistry;
    use crate::pipe_contract::{ColumnDesc, PgReply, QueryReply, SqlValue, verb};
    use crate::pipes::{PgRequest, PgResponse, into_pg_handle};

    use super::*;

    struct EchoPipe;

    impl SendPipe for EchoPipe {
        type In = PgRequest;
        type Out = PgResponse;
        type Err = ProximaError;

        async fn call(&self, request: PgRequest) -> Result<PgResponse, ProximaError> {
            let reply = match request.method.as_bytes() {
                verb::QUERY => PgReply::Query(QueryReply::rows(
                    vec![ColumnDesc::new("?column?", Oid(23))],
                    vec![vec![SqlValue::Int(1)]],
                )),
                _ => PgReply::Query(QueryReply::tag("OK")),
            };
            Ok(Response::typed(200, reply))
        }
    }

    /// A read-once / write-to-vec fake: the upgrade handler reads the
    /// scripted startup+terminate, then the session ends at EOF.
    struct ScriptedSocket {
        read_data: std::io::Cursor<Vec<u8>>,
        write_data: Vec<u8>,
    }

    impl AsyncRead for ScriptedSocket {
        fn poll_read(
            mut self: Pin<&mut Self>,
            _context: &mut Context<'_>,
            buf: &mut [u8],
        ) -> Poll<std::io::Result<usize>> {
            use std::io::Read;
            Poll::Ready(self.read_data.read(buf))
        }
    }

    impl AsyncWrite for ScriptedSocket {
        fn poll_write(
            mut self: Pin<&mut Self>,
            _context: &mut Context<'_>,
            buf: &[u8],
        ) -> Poll<std::io::Result<usize>> {
            self.write_data.extend_from_slice(buf);
            Poll::Ready(Ok(buf.len()))
        }

        fn poll_flush(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<std::io::Result<()>> {
            Poll::Ready(Ok(()))
        }

        fn poll_close(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<std::io::Result<()>> {
            Poll::Ready(Ok(()))
        }
    }

    fn build_startup_bytes(user: &str) -> Vec<u8> {
        let version_code: i32 = 196608;
        let mut params = Vec::new();
        params.extend_from_slice(b"user\0");
        params.extend_from_slice(user.as_bytes());
        params.push(0);
        params.push(0);
        let total_len = 4 + 4 + params.len();
        let mut buf = Vec::new();
        buf.extend_from_slice(&(total_len as i32).to_be_bytes());
        buf.extend_from_slice(&version_code.to_be_bytes());
        buf.extend_from_slice(&params);
        buf
    }

    fn connect_request() -> Request<Bytes> {
        Request {
            method: proxima_primitives::pipe::method::Method::from_bytes(verb::CONNECT),
            path: Bytes::new(),
            query: proxima_primitives::pipe::header_list::HeaderList::new(),
            metadata: proxima_primitives::pipe::header_list::HeaderList::new(),
            payload: Bytes::new(),
            stream: None,
            context: RequestContext::default(),
        }
    }

    #[proxima::test(runtime = "tokio")]
    async fn connect_request_answers_with_an_upgrade() {
        let pipe = PgWireConnectionPipe::new(
            "pg",
            into_pg_handle(EchoPipe),
            PgAuth::Trust,
            Arc::new(PgServerConfig::builder().parameters(vec![]).build()),
            Arc::new(CancelRegistry::new()),
        );

        let response = pipe
            .call(connect_request())
            .await
            .expect("connect must answer");

        assert_eq!(response.status, 200);
        assert!(
            response.upgrade.is_some(),
            "connect response must carry an upgrade"
        );
    }

    #[proxima::test(runtime = "tokio")]
    async fn upgrade_handler_drives_negotiate_and_serve_to_clean_completion() {
        let pipe = PgWireConnectionPipe::new(
            "pg",
            into_pg_handle(EchoPipe),
            PgAuth::Trust,
            Arc::new(PgServerConfig::builder().parameters(vec![]).build()),
            Arc::new(CancelRegistry::new()),
        );
        let response = pipe
            .call(connect_request())
            .await
            .expect("connect must answer");
        let handler = response.upgrade.expect("upgrade must be present");

        let mut scripted = build_startup_bytes("alice");
        scripted.push(b'X');
        scripted.extend_from_slice(&4_i32.to_be_bytes());
        let socket = ScriptedSocket {
            read_data: std::io::Cursor::new(scripted),
            write_data: Vec::new(),
        };
        let hijacked = HijackedSocket::new(Box::new(socket), Bytes::new());

        handler
            .invoke(hijacked)
            .await
            .expect("session must complete cleanly through the upgrade");
    }

    #[cfg(feature = "tls")]
    #[test]
    fn first_byte_0x16_is_classified_as_direct_tls() {
        assert!(
            super::is_direct_tls_first_byte(0x16),
            "0x16 is the TLS handshake content type"
        );
    }

    #[cfg(feature = "tls")]
    #[test]
    fn first_byte_0x00_is_not_direct_tls() {
        assert!(
            !super::is_direct_tls_first_byte(0x00),
            "a startup packet begins with the high byte of its length (0x00)"
        );
    }

    #[cfg(feature = "tls")]
    #[proxima::test(runtime = "tokio")]
    async fn prefixed_stream_replays_prefix_then_inner() {
        use futures::io::AsyncReadExt;

        let inner = futures::io::Cursor::new(b"world".to_vec());
        let mut prefixed = super::IntoFutures(super::Prepend::new(
            b"hello ".to_vec(),
            super::FromFutures(inner),
        ));
        let mut collected = Vec::new();
        prefixed
            .read_to_end(&mut collected)
            .await
            .expect("read must drain prefix and inner");

        assert_eq!(
            collected, b"hello world",
            "prefix bytes must precede the inner stream bytes"
        );
    }
}
