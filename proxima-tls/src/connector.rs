//! Client-side TLS over the `proxima_primitives::stream::StreamUpstream` interface.
//!
//! Symmetric peer of `build_acceptor_futures_io` (server side): where
//! the acceptor wraps an *accepted* connection in a server-side TLS
//! session, [`TlsStreamUpstream`] wraps a *dialed* connection in a
//! client-side TLS session. The inner backend is any `StreamUpstream`
//! (prime `PrimeTcpUpstream` by default, tokio for tests/benches), so a
//! TLS session runs over whatever byte transport the substrate provides.
//!
//! Lives in proxima-tls because the rustls / futures-rustls / rcgen
//! surface already lives here — keeping client + server TLS adapters in
//! one crate avoids scattering the rustls dep across the net layer.
//! Gated behind the same `futures-io` feature as the server connector.

use std::io;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::Mutex;
use std::task::{Context, Poll};

use bon::Builder;
use conflaguration::{Settings, Validate, ValidationMessage};
use futures::io::{AsyncRead, AsyncWrite};
use futures_rustls::TlsConnector;
use futures_rustls::client::TlsStream;
use proxima_core::ProximaError;
use proxima_primitives::stream::{PeerInfo, StreamConnection, StreamUpstream, StreamUpstreamExt};
use rustls::ClientConfig;
use rustls::RootCertStore;
use rustls::pki_types::ServerName;
use serde::{Deserialize, Serialize};

/// Build a client `ClientConfig` with an EXPLICIT crypto provider, so
/// construction never depends on a process-global `install_default`
/// (which `ClientConfig::builder()` panics without). Mirrors the server
/// side's `get_default` check but picks aws-lc-rs deterministically.
fn build_client_config(
    roots: RootCertStore,
    alpn_protocols: Vec<Vec<u8>>,
) -> Result<ClientConfig, ProximaError> {
    let provider = Arc::new(rustls::crypto::aws_lc_rs::default_provider());
    let mut config = ClientConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()
        .map_err(|err| ProximaError::Config(format!("tls client protocol versions: {err}")))?
        .with_root_certificates(roots)
        .with_no_client_auth();
    config.alpn_protocols = alpn_protocols;
    Ok(config)
}

type ConnectFuture<C> = Pin<Box<dyn std::future::Future<Output = io::Result<TlsConn<C>>> + Send>>;

/// Client-side TLS connection: a `futures_rustls::client::TlsStream`
/// over the inner backend's connection. The inner `peer()` shows
/// through so callers see the underlying transport peer, not a TLS
/// abstraction.
pub struct TlsConn<C> {
    inner: TlsStream<C>,
    peer: Option<PeerInfo>,
}

impl<C: StreamConnection> AsyncRead for TlsConn<C> {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut [u8],
    ) -> Poll<io::Result<usize>> {
        Pin::new(&mut self.get_mut().inner).poll_read(cx, buf)
    }
}

impl<C: StreamConnection> AsyncWrite for TlsConn<C> {
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

impl<C: StreamConnection> StreamConnection for TlsConn<C> {
    fn peer(&self) -> Option<PeerInfo> {
        self.peer.clone()
    }
}

/// A `StreamUpstream` that layers a client-side TLS handshake on top of
/// an inner `StreamUpstream`. Holds the inner backend, the validated
/// `ServerName` to present in SNI / verify against the cert, and the
/// shared `ClientConfig` (root trust + ALPN + protocol versions).
pub struct TlsStreamUpstream<U: StreamUpstream> {
    inner: Arc<U>,
    // parsed eagerly at construction but kept fallible: the trait
    // offers no fallible ctor, so a malformed hostname surfaces as a
    // connect-time io error rather than a panic or a silent sentinel.
    server_name: Result<ServerName<'static>, String>,
    config: Arc<ClientConfig>,
    // WHY Mutex here: `poll_connect(&self, ...)` takes `&self`, so the
    // in-flight TCP-connect + TLS-handshake future needs interior
    // mutability to survive across polls. Same structural constraint
    // and per-connection (not per-request) contention profile as
    // `TokioTcpUpstream::in_flight` / `PrimeTcpUpstream::in_flight`.
    in_flight: Mutex<Option<ConnectFuture<U::Conn>>>,
}

impl<U: StreamUpstream> TlsStreamUpstream<U> {
    /// Build from a caller-supplied `ClientConfig`. `server_name` is
    /// the hostname to present via SNI and verify the server cert
    /// against; an invalid name surfaces lazily on the first
    /// `connect()` as an `io::Error` (the ctor cannot fail).
    pub fn new(inner: U, server_name: impl Into<String>, config: Arc<ClientConfig>) -> Self {
        let raw = server_name.into();
        let server_name = ServerName::try_from(raw.clone())
            .map_err(|err| format!("invalid tls server name `{raw}`: {err}"));
        Self {
            inner: Arc::new(inner),
            server_name,
            config,
            in_flight: Mutex::new(None),
        }
    }

    /// Convenience ctor: default `ClientConfig` trusting the Mozilla
    /// webpki root set, ALPN `http/1.1` (for the prime h1 client).
    /// Use [`Self::new`] with a custom config to trust private CAs or
    /// negotiate other ALPN protocols.
    ///
    /// Fallible: builds the rustls config with an explicit aws-lc-rs
    /// provider (no dependency on a process-global provider), which can
    /// only fail if that provider can't supply the safe protocol set.
    pub fn with_webpki_roots(
        inner: U,
        server_name: impl Into<String>,
    ) -> Result<Self, ProximaError> {
        let mut roots = RootCertStore::empty();
        roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
        let config = build_client_config(roots, vec![b"http/1.1".to_vec()])?;
        Ok(Self::new(inner, server_name, Arc::new(config)))
    }

    /// Build from a [`TlsClientConfig`] (the declarative, serializable
    /// half) plus the live `inner` transport (the runtime half). The
    /// config supplies the SNI hostname + ALPN protocol list and trusts
    /// the Mozilla webpki root set; the `inner` `StreamUpstream` cannot
    /// live in a TOML file so it's injected here — the same config /
    /// runtime split telemetry's `Recorder::from_config` uses. P4
    /// interop: a `TlsClientConfig` loaded from env / file becomes a
    /// live upstream without hand-wiring rustls.
    pub fn from_config(inner: U, config: &TlsClientConfig) -> Result<Self, ProximaError> {
        let mut roots = RootCertStore::empty();
        roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
        let alpn = config
            .alpn_protocols
            .iter()
            .map(|protocol| protocol.clone().into_bytes())
            .collect();
        let client = build_client_config(roots, alpn)?;
        Ok(Self::new(
            inner,
            config.server_name.clone(),
            Arc::new(client),
        ))
    }

    /// Fluent builder for the declarative half. Set `server_name` (and
    /// optionally the ALPN list), then pair it with a live transport via
    /// [`Self::from_config`].
    pub fn config_builder() -> TlsClientConfigBuilder {
        TlsClientConfig::builder()
    }
}

fn default_alpn_protocols() -> Vec<String> {
    vec!["http/1.1".to_string()]
}

/// The declarative, serializable description of a client TLS session —
/// the half of a [`TlsStreamUpstream`] that can live in env / TOML.
///
/// The live `inner` transport (a `StreamUpstream`) is *not* here: it's a
/// runtime object, injected at [`TlsStreamUpstream::from_config`] time,
/// mirroring telemetry's config-vs-runtime split. Trust always uses the
/// Mozilla webpki root set today; private-CA trust stays on the explicit
/// [`TlsStreamUpstream::new`] path with a hand-built `ClientConfig`.
#[derive(Debug, Clone, Builder, Deserialize, Serialize, Settings)]
#[settings(prefix = "TLS_CLIENT")]
#[builder(derive(Clone, Debug))]
pub struct TlsClientConfig {
    /// Hostname presented via SNI and verified against the server cert
    /// (e.g. `"huggingface.co"`). Required — there is no sane default.
    pub server_name: String,
    /// ALPN protocols offered in the handshake, most-preferred first.
    /// Defaults to `["http/1.1"]` for the prime h1 client.
    #[setting(skip)]
    #[serde(default = "default_alpn_protocols")]
    #[builder(default = default_alpn_protocols())]
    pub alpn_protocols: Vec<String>,
}

impl Validate for TlsClientConfig {
    fn validate(&self) -> conflaguration::Result<()> {
        let mut errors = Vec::new();
        if self.server_name.is_empty() {
            errors.push(ValidationMessage::new("server_name", "must be non-empty"));
        }
        if self.alpn_protocols.iter().any(String::is_empty) {
            errors.push(ValidationMessage::new(
                "alpn_protocols",
                "must not contain an empty protocol id",
            ));
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(conflaguration::Error::Validation { errors })
        }
    }
}

impl<U: StreamUpstream> StreamUpstream for TlsStreamUpstream<U> {
    type Conn = TlsConn<U::Conn>;

    fn poll_connect(&self, cx: &mut Context<'_>) -> Poll<io::Result<Self::Conn>> {
        let Ok(mut slot) = self.in_flight.lock() else {
            return Poll::Ready(Err(io::Error::other("TlsStreamUpstream: lock poisoned")));
        };
        let server_name = match &self.server_name {
            Ok(name) => name.clone(),
            Err(message) => return Poll::Ready(Err(io::Error::other(message.clone()))),
        };
        let future = slot.get_or_insert_with(|| {
            let inner = self.inner.clone();
            let connector = TlsConnector::from(self.config.clone());
            Box::pin(async move {
                let conn = inner.connect().await?;
                let peer = conn.peer();
                let tls = connector.connect(server_name, conn).await?;
                Ok(TlsConn { inner: tls, peer })
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
    use crate::{TlsConfig, build_acceptor_futures_io};
    use futures::io::{AsyncReadExt, AsyncWriteExt};
    use proxima_net::tokio::{TokioTcpListener, TokioTcpUpstream};
    use proxima_primitives::stream::{StreamListener, StreamListenerExt, StreamUpstreamExt};
    use std::net::{Ipv4Addr, SocketAddr};

    /// loopback TLS round-trip over the StreamUpstream interface.
    ///
    /// server: tokio TCP listener + futures-rustls acceptor with a real
    /// rcgen self-signed "localhost" cert. client: TlsStreamUpstream
    /// over TokioTcpUpstream, trusting that exact cert via a custom root
    /// store (NOT webpki roots). The bytes round-trip through a real TLS
    /// 1.3 session, proving the client connector works over the
    /// StreamUpstream interface.
    #[proxima::test]
    async fn loopback_tls_round_trips_bytes() {
        rustls::crypto::aws_lc_rs::default_provider()
            .install_default()
            .ok();

        let generated =
            rcgen::generate_simple_self_signed(vec!["localhost".to_string()]).expect("rcgen");
        let cert_pem = generated.cert.pem().into_bytes();
        let key_pem = generated.signing_key.serialize_pem().into_bytes();
        let cert_der = generated.cert.der().clone();

        let server_config = TlsConfig::pem(cert_pem, key_pem);
        let acceptor = build_acceptor_futures_io(&server_config).expect("acceptor");

        let listener = TokioTcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))
            .await
            .expect("bind");
        let local = match listener.local_addr().expect("local_addr") {
            proxima_primitives::stream::BindAddr::Tcp(addr) => addr,
            other => panic!("expected tcp, got {other:?}"),
        };

        let server = tokio::spawn(async move {
            let conn = listener.accept().await.expect("accept");
            let mut tls = acceptor.accept(conn).await.expect("server handshake");
            let mut buf = [0_u8; 5];
            tls.read_exact(&mut buf).await.expect("server read");
            tls.write_all(&buf).await.expect("server echo");
            tls.flush().await.expect("server flush");
        });

        let mut roots = RootCertStore::empty();
        roots.add(cert_der).expect("trust self-signed cert");
        let client_config = ClientConfig::builder()
            .with_root_certificates(roots)
            .with_no_client_auth();

        let upstream = TlsStreamUpstream::new(
            TokioTcpUpstream::new(local),
            "localhost",
            Arc::new(client_config),
        );
        let mut conn = upstream.connect().await.expect("client tls connect");
        conn.write_all(b"hello").await.expect("client write");
        conn.flush().await.expect("client flush");
        let mut reply = [0_u8; 5];
        conn.read_exact(&mut reply).await.expect("client read");
        assert_eq!(&reply, b"hello");

        server.await.expect("join server");
    }

    /// a malformed server name surfaces as a connect-time io error, not
    /// a panic — the ctor cannot fail, so the bad name is carried until
    /// the first connect attempt.
    #[proxima::test]
    async fn invalid_server_name_errors_at_connect() {
        rustls::crypto::aws_lc_rs::default_provider()
            .install_default()
            .ok();
        let mut roots = RootCertStore::empty();
        roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
        let config = ClientConfig::builder()
            .with_root_certificates(roots)
            .with_no_client_auth();
        let upstream = TlsStreamUpstream::new(
            TokioTcpUpstream::new(SocketAddr::from((Ipv4Addr::LOCALHOST, 1))),
            "not a valid hostname",
            Arc::new(config),
        );
        let outcome = upstream.connect().await;
        assert!(outcome.is_err(), "expected invalid-name error, got Ok");
    }

    /// the bon builder and the conflaguration env loader produce the
    /// same config — P4 parity for the lower TLS piece.
    #[test]
    fn tls_client_config_builder_matches_env_loader() {
        let built = TlsClientConfig::builder()
            .server_name("huggingface.co".to_string())
            .build();
        let loaded =
            temp_env::with_vars([("TLS_CLIENT_SERVER_NAME", Some("huggingface.co"))], || {
                TlsClientConfig::from_env().expect("from_env")
            });
        // server_name is the env-loadable scalar; alpn_protocols is
        // `#[setting(skip)]` (conflaguration cannot parse a list from a
        // single env var) so the env loader leaves it at Default — the
        // builder carries the http/1.1 default. Same split telemetry's
        // skipped fields use; parity is on the loadable field.
        assert_eq!(built.server_name, loaded.server_name);
        assert_eq!(built.alpn_protocols, vec!["http/1.1".to_string()]);
    }

    #[test]
    fn tls_client_config_rejects_empty_server_name() {
        let cfg = TlsClientConfig::builder()
            .server_name(String::new())
            .build();
        assert!(cfg.validate().is_err());
    }
}
