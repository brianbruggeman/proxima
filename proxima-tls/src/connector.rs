//! Client-side TLS over the `proxima_primitives::stream::StreamUpstream` interface.
//!
//! Symmetric peer of `build_acceptor_futures_io` (server side): where
//! the acceptor wraps an *accepted* connection in a server-side TLS
//! session, [`TlsStreamUpstream`] wraps a *dialed* connection in a
//! client-side TLS session. The inner backend is any `StreamUpstream`
//! (prime `PrimeTcpUpstream` by default, tokio for tests/benches), so a
//! TLS session runs over whatever byte transport the core provides.
//!
//! Lives in proxima-tls because the rustls / futures-rustls / rcgen
//! surface already lives here — keeping client + server TLS adapters in
//! one crate avoids scattering the rustls dep across the net layer.
//! Gated behind the same `futures-io` feature as the server connector.

use std::io;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use crate::RootSource;
use bon::Builder;
use conflaguration::{Settings, Validate, ValidationMessage};
use futures::io::{AsyncRead, AsyncWrite};
use futures_rustls::TlsConnector;
use futures_rustls::client::TlsStream;
use proxima_core::ProximaError;
use proxima_primitives::stream::{
    ConnectFuture, PeerInfo, StreamConnection, StreamUpstream, StreamUpstreamExt,
};
use rustls::ClientConfig;
use rustls::RootCertStore;
use rustls::pki_types::pem::PemObject;
use rustls::pki_types::{CertificateDer, ServerName};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

#[cfg(test)]
mod generated_defaults {
    include!(concat!(env!("OUT_DIR"), "/tls_client_defaults.rs"));
}

/// Build a client `ClientConfig` with an EXPLICIT crypto provider, so
/// construction never depends on a process-global `install_default`
/// (which `ClientConfig::builder()` panics without). Mirrors the server
/// side's `get_default` check but picks aws-lc-rs deterministically.
fn build_client_config(
    root_source: RootSource,
    custom_roots: Vec<CertificateDer<'static>>,
    alpn_protocols: Vec<Vec<u8>>,
) -> Result<ClientConfig, ProximaError> {
    let provider = Arc::new(rustls::crypto::aws_lc_rs::default_provider());
    let builder = ClientConfig::builder_with_provider(provider.clone())
        .with_safe_default_protocol_versions()
        .map_err(|err| ProximaError::Config(format!("tls client protocol versions: {err}")))?;
    let mut config = match root_source {
        RootSource::Native => {
            let verifier =
                rustls_platform_verifier::Verifier::new_with_extra_roots(custom_roots, provider)
                    .map_err(|err| {
                        ProximaError::Config(format!("tls client native roots: {err}"))
                    })?;
            builder
                .dangerous()
                .with_custom_certificate_verifier(Arc::new(verifier))
                .with_no_client_auth()
        }
        RootSource::Mozilla | RootSource::CustomOnly => {
            let mut roots = RootCertStore::empty();
            if root_source == RootSource::Mozilla {
                roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
            }
            for certificate in custom_roots {
                roots.add(certificate).map_err(|err| {
                    ProximaError::Config(format!("tls client custom root: {err}"))
                })?;
            }
            builder.with_root_certificates(roots).with_no_client_auth()
        }
    };
    config.alpn_protocols = alpn_protocols;
    Ok(config)
}

fn parse_json<T: DeserializeOwned>(raw: &str) -> Result<T, serde_json::Error> {
    serde_json::from_str(raw)
}

fn parse_root_source(raw: &str) -> Result<RootSource, serde_json::Error> {
    serde_json::from_value(serde_json::Value::String(raw.to_owned()))
}

#[cfg(test)]
fn default_root_source() -> RootSource {
    if generated_defaults::TLS_CLIENT_NATIVE_ROOTS_DEFAULT {
        RootSource::Native
    } else {
        RootSource::Mozilla
    }
}

fn default_alpn_protocols() -> Vec<String> {
    vec!["http/1.1".to_string()]
}

fn load_custom_roots(paths: &[PathBuf]) -> Result<Vec<CertificateDer<'static>>, ProximaError> {
    let mut roots = Vec::new();
    for path in paths {
        let certificates = CertificateDer::pem_file_iter(path).map_err(|err| {
            ProximaError::Config(format!(
                "tls client CA bundle `{}` could not be read: {err}",
                path.display()
            ))
        })?;
        let certificates = certificates.collect::<Result<Vec<_>, _>>().map_err(|err| {
            ProximaError::Config(format!(
                "tls client CA bundle `{}` contains invalid PEM: {err}",
                path.display()
            ))
        })?;
        if certificates.is_empty() {
            return Err(ProximaError::Config(format!(
                "tls client CA bundle `{}` contains no certificates",
                path.display()
            )));
        }
        let mut validated = RootCertStore::empty();
        for certificate in certificates {
            validated.add(certificate.clone()).map_err(|err| {
                ProximaError::Config(format!(
                    "tls client CA bundle `{}` contains an invalid root certificate: {err}",
                    path.display()
                ))
            })?;
            roots.push(certificate);
        }
    }
    Ok(roots)
}

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
}

impl<U: StreamUpstream> TlsStreamUpstream<U> {
    /// Build from a caller-supplied `ClientConfig`. `server_name` is
    /// the hostname to present via SNI and verify the server cert
    /// against; an invalid name surfaces lazily on the first
    /// `connect()` as an `io::Error` (the ctor cannot fail).
    #[must_use]
    pub fn new(inner: U, server_name: impl Into<String>, config: Arc<ClientConfig>) -> Self {
        let raw = server_name.into();
        let server_name = ServerName::try_from(raw.clone())
            .map_err(|err| format!("invalid tls server name `{raw}`: {err}"));
        Self {
            inner: Arc::new(inner),
            server_name,
            config,
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
        let config = TlsClientConfig::builder()
            .server_name(server_name.into())
            .root_source(RootSource::Mozilla)
            .build();
        Self::from_config(inner, &config)
    }

    /// Build from a [`TlsClientConfig`] (the declarative, serializable
    /// half) plus the live `inner` transport (the runtime half). The
    /// config supplies the SNI hostname, ALPN list, and trust policy; the
    /// `inner` `StreamUpstream` cannot live in a TOML file so it is injected
    /// here — the same config / runtime split telemetry's
    /// `Recorder::from_config` uses. P4
    /// interop: a `TlsClientConfig` loaded from env / file becomes a
    /// live upstream without hand-wiring rustls. Root files and verifier
    /// construction are resolved before a connection is opened.
    pub fn from_config(inner: U, config: &TlsClientConfig) -> Result<Self, ProximaError> {
        config
            .validate()
            .map_err(|err| ProximaError::Config(format!("tls client config: {err}")))?;
        let client = config.build_rustls_config()?;
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

/// The declarative, serializable description of a client TLS session —
/// the half of a [`TlsStreamUpstream`] that can live in env / TOML.
///
/// The live `inner` transport is injected at [`TlsStreamUpstream::from_config`]
/// time, keeping runtime objects out of serializable configuration.
#[derive(Debug, Clone, PartialEq, Eq, Builder, Deserialize, Serialize, Settings)]
#[settings(prefix = "TLS_CLIENT")]
#[builder(derive(Clone, Debug))]
pub struct TlsClientConfig {
    /// Hostname presented via SNI and verified against the server cert
    /// (e.g. `"huggingface.co"`). Universal URL clients fill this when omitted.
    #[setting(default)]
    #[serde(default)]
    #[builder(default)]
    pub server_name: String,
    /// Selects the platform, Mozilla, or custom-only trust anchors.
    #[setting(resolve_with = "parse_root_source", default)]
    #[serde(default = "RootSource::target_default")]
    #[builder(default = RootSource::target_default())]
    pub root_source: RootSource,
    /// PEM bundles appended to Mozilla or native roots, or used alone.
    #[setting(resolve_with = "parse_json", default_str = "[]")]
    #[serde(default)]
    #[builder(default)]
    pub ca_bundle_paths: Vec<PathBuf>,
    /// ALPN protocols offered in the handshake, most-preferred first.
    /// Defaults to `["http/1.1"]` for the prime h1 client.
    #[setting(resolve_with = "parse_json", default_str = "[\"http/1.1\"]")]
    #[serde(default = "default_alpn_protocols")]
    #[builder(default = default_alpn_protocols())]
    pub alpn_protocols: Vec<String>,
}

impl Default for TlsClientConfig {
    fn default() -> Self {
        Self {
            server_name: String::new(),
            root_source: RootSource::default(),
            ca_bundle_paths: Vec::new(),
            alpn_protocols: default_alpn_protocols(),
        }
    }
}

impl TlsClientConfig {
    /// Start fluent configuration from target defaults, then apply file,
    /// environment, and explicit value layers in call order.
    pub fn layered() -> TlsClientLayerBuilder {
        TlsClientLayerBuilder::new()
    }

    /// Build the rustls client config for a URL-aware TLS connector. The
    /// connector supplies the destination SNI name from its request URI.
    pub fn build_rustls_config(&self) -> Result<ClientConfig, ProximaError> {
        self.validate_trust_settings()
            .map_err(|err| ProximaError::Config(format!("tls client config: {err}")))?;
        let custom_roots = load_custom_roots(&self.ca_bundle_paths)?;
        let alpn = self
            .alpn_protocols
            .iter()
            .map(|protocol| protocol.clone().into_bytes())
            .collect();
        build_client_config(self.root_source, custom_roots, alpn)
    }

    fn validate_trust_settings(&self) -> conflaguration::Result<()> {
        let mut errors = Vec::new();
        if self.alpn_protocols.iter().any(String::is_empty) {
            errors.push(ValidationMessage::new(
                "alpn_protocols",
                "must not contain an empty protocol id",
            ));
        }
        if self
            .ca_bundle_paths
            .iter()
            .any(|path| path.as_os_str().is_empty())
        {
            errors.push(ValidationMessage::new(
                "ca_bundle_paths",
                "must not contain an empty path",
            ));
        }
        if self.root_source == RootSource::CustomOnly && self.ca_bundle_paths.is_empty() {
            errors.push(ValidationMessage::new(
                "ca_bundle_paths",
                "must contain at least one PEM bundle when root_source is custom_only",
            ));
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(conflaguration::Error::Validation { errors })
        }
    }
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
        if self
            .ca_bundle_paths
            .iter()
            .any(|path| path.as_os_str().is_empty())
        {
            errors.push(ValidationMessage::new(
                "ca_bundle_paths",
                "must not contain an empty path",
            ));
        }
        if self.root_source == RootSource::CustomOnly && self.ca_bundle_paths.is_empty() {
            errors.push(ValidationMessage::new(
                "ca_bundle_paths",
                "must contain at least one PEM bundle when root_source is custom_only",
            ));
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(conflaguration::Error::Validation { errors })
        }
    }
}

/// Fluent layered configuration for a TLS client.
#[derive(Debug, Clone)]
pub struct TlsClientLayerBuilder {
    config: TlsClientConfig,
}

impl TlsClientLayerBuilder {
    /// Start from target-specific defaults.
    pub fn new() -> Self {
        Self {
            config: TlsClientConfig::default(),
        }
    }

    /// Load a TOML, JSON, or YAML layer after applying target defaults.
    pub fn from_path(mut self, path: impl AsRef<Path>) -> conflaguration::Result<Self> {
        self.config = conflaguration::builder()
            .value(self.config)
            .file(path)
            .build()?;
        Ok(self)
    }

    /// Override configured values from `TLS_CLIENT_*` environment variables.
    pub fn from_env(mut self) -> conflaguration::Result<Self> {
        self.config = conflaguration::builder().value(self.config).env().build()?;
        Ok(self)
    }

    /// Set the SNI hostname and certificate verification name.
    pub fn with_server_name(mut self, server_name: impl Into<String>) -> Self {
        self.config.server_name = server_name.into();
        self
    }

    /// Set ALPN protocols in preference order.
    pub fn with_alpn_protocols(mut self, protocols: Vec<String>) -> Self {
        self.config.alpn_protocols = protocols;
        self
    }

    /// Set the trust anchor source.
    pub fn with_root_source(mut self, source: RootSource) -> Self {
        self.config.root_source = source;
        self
    }

    /// Set additive or exclusive custom PEM bundle paths.
    pub fn with_ca_bundle_paths(mut self, paths: Vec<PathBuf>) -> Self {
        self.config.ca_bundle_paths = paths;
        self
    }

    /// Validate and return the resolved configuration.
    pub fn build(self) -> conflaguration::Result<TlsClientConfig> {
        self.config.validate()?;
        Ok(self.config)
    }

    /// Validate roots and ALPN for attachment to a destination-aware
    /// universal client. The HTTP/gRPC factory supplies SNI from its URL.
    pub fn build_for_client(self) -> conflaguration::Result<TlsClientConfig> {
        self.config.validate_trust_settings()?;
        Ok(self.config)
    }
}

impl Default for TlsClientLayerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl<U: StreamUpstream> StreamUpstream for TlsStreamUpstream<U> {
    type Conn = TlsConn<U::Conn>;

    fn connect_future(&self) -> ConnectFuture<'_, Self::Conn> {
        Box::pin(async move {
            let server_name = self.server_name.clone().map_err(io::Error::other)?;
            let connection = self.inner.connect().await?;
            let peer = connection.peer();
            let connector = TlsConnector::from(self.config.clone());
            let inner = connector.connect(server_name, connection).await?;
            Ok(TlsConn { inner, peer })
        })
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
    use rcgen::{
        BasicConstraints, CertificateParams, ExtendedKeyUsagePurpose, IsCa, Issuer, KeyPair,
        KeyUsagePurpose,
    };
    #[cfg(windows)]
    use std::ffi::OsStr;
    use std::fs;
    #[cfg(windows)]
    use std::iter;
    use std::net::{Ipv4Addr, SocketAddr};
    #[cfg(windows)]
    use std::ptr;
    #[cfg(windows)]
    use std::slice;

    #[cfg(windows)]
    use std::os::windows::ffi::OsStrExt;
    #[cfg(windows)]
    use windows_sys::Win32::Security::Cryptography::{
        CERT_STORE_ADD_ALWAYS, CERT_STORE_MAXIMUM_ALLOWED_FLAG, CERT_STORE_OPEN_EXISTING_FLAG,
        CERT_STORE_PROV_SYSTEM_W, CERT_SYSTEM_STORE_CURRENT_USER, CertAddEncodedCertificateToStore,
        CertCloseStore, CertDeleteCertificateFromStore, CertEnumCertificatesInStore,
        CertFreeCertificateContext, CertOpenStore, HCERTSTORE, X509_ASN_ENCODING,
    };

    fn generate_ca_and_leaf() -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
        let mut ca_params = CertificateParams::new(Vec::<String>::new()).expect("CA parameters");
        ca_params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
        ca_params.key_usages = vec![KeyUsagePurpose::KeyCertSign, KeyUsagePurpose::CrlSign];
        let ca_key = KeyPair::generate().expect("CA key");
        let ca_certificate = ca_params.self_signed(&ca_key).expect("self-sign CA");
        let issuer = Issuer::new(ca_params, ca_key);

        let mut leaf_params =
            CertificateParams::new(vec!["localhost".to_string()]).expect("leaf parameters");
        leaf_params.key_usages = vec![KeyUsagePurpose::DigitalSignature];
        leaf_params.extended_key_usages = vec![ExtendedKeyUsagePurpose::ServerAuth];
        let leaf_key = KeyPair::generate().expect("leaf key");
        let leaf_certificate = leaf_params
            .signed_by(&leaf_key, &issuer)
            .expect("sign leaf with CA");

        (
            ca_certificate.der().as_ref().to_vec(),
            ca_certificate.pem().into_bytes(),
            leaf_certificate.pem().into_bytes(),
            leaf_key.serialize_pem().into_bytes(),
        )
    }

    /// loopback TLS round-trip over the StreamUpstream interface.
    ///
    /// server: tokio TCP listener + futures-rustls acceptor with a real
    /// rcgen self-signed "localhost" cert. client: TlsStreamUpstream
    /// over TokioTcpUpstream, trusting that exact cert via a custom root
    /// store (NOT webpki roots). The bytes round-trip through a real TLS
    /// 1.3 session, proving the client connector works over the
    /// StreamUpstream interface.
    ///
    /// tokio runtime, not the default prime one: the fixture binds a
    /// `TokioTcpListener` and `tokio::spawn`s the echo half, both of which
    /// need a live tokio reactor on the calling thread.
    #[proxima::test(runtime = "tokio")]
    async fn loopback_tls_round_trips_bytes() {
        assert_tls_trust_round_trip(RootSource::CustomOnly).await;
    }

    #[proxima::test(runtime = "tokio")]
    async fn tls_trust_custom_only_round_trip() {
        assert_tls_trust_round_trip(RootSource::CustomOnly).await;
    }

    #[proxima::test(runtime = "tokio")]
    async fn tls_trust_mozilla_plus_custom_round_trip() {
        assert_tls_trust_round_trip(RootSource::Mozilla).await;
    }

    #[proxima::test(runtime = "tokio")]
    async fn tls_trust_native_plus_custom_round_trip() {
        assert_tls_trust_round_trip(RootSource::Native).await;
    }

    #[proxima::test(runtime = "tokio")]
    async fn tls_trust_missing_malformed_empty_pem_fails() {
        assert_tls_trust_round_trip(RootSource::CustomOnly).await;
        let directory = tempfile::tempdir().expect("temporary certificate directory");
        let missing_path = directory.path().join("missing.pem");
        let malformed_path = directory.path().join("malformed.pem");
        let invalid_der_path = directory.path().join("invalid-der.pem");
        let empty_path = directory.path().join("empty.pem");
        fs::write(&malformed_path, b"not a certificate").expect("write malformed PEM");
        fs::write(
            &invalid_der_path,
            b"-----BEGIN CERTIFICATE-----\nAQID\n-----END CERTIFICATE-----\n",
        )
        .expect("write malformed DER certificate");
        fs::write(&empty_path, b"# empty bundle\n").expect("write empty PEM");

        for path in [
            &missing_path,
            &malformed_path,
            &invalid_der_path,
            &empty_path,
        ] {
            let config = TlsClientConfig::layered()
                .with_server_name("localhost")
                .with_root_source(RootSource::CustomOnly)
                .with_ca_bundle_paths(vec![path.to_path_buf()])
                .build()
                .expect("invalid path still forms structurally valid config");
            let error = TlsStreamUpstream::from_config(
                TokioTcpUpstream::new(SocketAddr::from((Ipv4Addr::LOCALHOST, 1))),
                &config,
            )
            .err()
            .expect("invalid CA bundle must fail before connecting");
            assert!(
                error.to_string().contains(&path.display().to_string()),
                "bundle error must identify {}: {error}",
                path.display()
            );
        }
    }

    #[proxima::test(runtime = "tokio")]
    async fn tls_trust_untrusted_peer_rejected() {
        let (_, _, server_leaf, server_key) = generate_ca_and_leaf();
        let (_, untrusted_ca, _, _) = generate_ca_and_leaf();
        let acceptor =
            build_acceptor_futures_io(&TlsConfig::pem(server_leaf, server_key)).expect("acceptor");
        let listener = TokioTcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))
            .await
            .expect("bind listener");
        let local = match listener.local_addr().expect("local address") {
            proxima_primitives::stream::BindAddr::Tcp(address) => address,
            other => panic!("expected TCP address, got {other:?}"),
        };
        let server = tokio::spawn(async move {
            let connection = listener.accept().await.expect("accept connection");
            acceptor.accept(connection).await.is_err()
        });

        let directory = tempfile::tempdir().expect("temporary certificate directory");
        let ca_path = directory.path().join("untrusted-ca.pem");
        fs::write(&ca_path, untrusted_ca).expect("write unrelated CA");
        let config = TlsClientConfig::layered()
            .with_server_name("localhost")
            .with_root_source(RootSource::CustomOnly)
            .with_ca_bundle_paths(vec![ca_path])
            .build()
            .expect("untrusted custom-root config");
        let upstream = TlsStreamUpstream::from_config(TokioTcpUpstream::new(local), &config)
            .expect("construct TLS upstream before dialing");
        let outcome = upstream.connect().await;
        let error = match outcome {
            Err(error) => error,
            Ok(_) => panic!("untrusted peer completed TLS certificate verification"),
        };
        assert_eq!(
            error.kind(),
            io::ErrorKind::InvalidData,
            "untrusted peer must fail in certificate verification: {error}"
        );
        assert!(server.await.expect("join server"));
    }

    async fn assert_tls_trust_round_trip(root_source: RootSource) {
        rustls::crypto::aws_lc_rs::default_provider()
            .install_default()
            .ok();

        let (cert_der, ca_pem, leaf_pem, key_pem) = generate_ca_and_leaf();
        let directory = tempfile::tempdir().expect("temporary certificate directory");
        let ca_path = directory.path().join("local-ca.pem");
        fs::write(&ca_path, &ca_pem).expect("write local CA bundle");
        let config = TlsClientConfig::layered()
            .with_server_name("localhost")
            .with_root_source(root_source)
            .with_ca_bundle_paths(vec![ca_path])
            .build()
            .expect("custom-root TLS client config");
        assert_tls_trust_config_round_trip(config, cert_der, leaf_pem, key_pem).await;
    }

    async fn assert_tls_trust_config_round_trip(
        config: TlsClientConfig,
        ca_der: Vec<u8>,
        leaf_pem: Vec<u8>,
        key_pem: Vec<u8>,
    ) {
        let server_config = TlsConfig::pem(leaf_pem, key_pem);
        let acceptor = build_acceptor_futures_io(&server_config).expect("acceptor");

        let listener = TokioTcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))
            .await
            .expect("bind");
        let local = match listener.local_addr().expect("local_addr") {
            proxima_primitives::stream::BindAddr::Tcp(addr) => addr,
            other => panic!("expected tcp, got {other:?}"),
        };

        let server = tokio::spawn(async move {
            let first = listener.accept().await.expect("first accept");
            let second = listener.accept().await.expect("second accept");
            let exchange = async |connection| {
                let mut tls = acceptor.accept(connection).await.expect("server handshake");
                let mut payload = [0_u8; 14];
                tls.read_exact(&mut payload).await.expect("server read");
                tls.write_all(&payload).await.expect("server echo");
                tls.flush().await.expect("server flush");
                payload
            };
            let (first, second) = futures::join!(exchange(first), exchange(second));
            assert!(
                first == *b"tls root probe" && second == *b"tls root probe",
                "configured and incumbent requests must deliver the exact probe bytes"
            );
        });

        let configured_upstream =
            TlsStreamUpstream::from_config(TokioTcpUpstream::new(local), &config)
                .expect("configured TLS upstream");

        let mut roots = RootCertStore::empty();
        roots
            .add(CertificateDer::from(ca_der))
            .expect("trust local CA certificate");
        let client_config = ClientConfig::builder()
            .with_root_certificates(roots)
            .with_no_client_auth();

        let incumbent_upstream = TlsStreamUpstream::new(
            TokioTcpUpstream::new(local),
            "localhost",
            Arc::new(client_config),
        );
        let exchange = async |upstream: &TlsStreamUpstream<TokioTcpUpstream>| {
            let mut connection = upstream.connect().await.expect("client tls connect");
            let payload = *b"tls root probe";
            connection.write_all(&payload).await.expect("client write");
            connection.flush().await.expect("client flush");
            let mut reply = [0_u8; 14];
            connection
                .read_exact(&mut reply)
                .await
                .expect("client read");
            assert_eq!(reply, payload);
        };
        futures::join!(
            exchange(&configured_upstream),
            exchange(&incumbent_upstream)
        );
        server.await.expect("join server");
    }

    /// a malformed server name surfaces as a connect-time io error, not
    /// a panic — the ctor cannot fail, so the bad name is carried until
    /// the first connect attempt.
    #[proxima::test(runtime = "tokio")]
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
        // Both sources resolve target defaults and the same explicit hostname.
        assert_eq!(built.server_name, loaded.server_name);
        assert_eq!(built.alpn_protocols, vec!["http/1.1".to_string()]);
        assert_eq!(built.root_source, default_root_source());
        assert_eq!(loaded.root_source, default_root_source());
    }

    #[test]
    fn tls_trust_defaults_follow_target() {
        let expected = if cfg!(windows) {
            RootSource::Native
        } else {
            RootSource::Mozilla
        };
        assert_eq!(RootSource::default(), expected);
        assert_eq!(
            generated_defaults::TLS_CLIENT_NATIVE_ROOTS_DEFAULT,
            cfg!(windows)
        );
    }

    #[proxima::test(runtime = "tokio")]
    async fn tls_trust_builder_env_toml_serde_parity() {
        let directory = tempfile::tempdir().expect("temporary config directory");
        let ca_path = directory.path().join("local-ca.pem");
        let (ca_der, ca_pem, leaf_pem, key_pem) = generate_ca_and_leaf();
        fs::write(&ca_path, ca_pem).expect("write parity fixture CA");
        let expected = TlsClientConfig::builder()
            .server_name("localhost".to_string())
            .root_source(RootSource::CustomOnly)
            .ca_bundle_paths(vec![ca_path.clone()])
            .alpn_protocols(vec!["h2".to_string(), "http/1.1".to_string()])
            .build();

        let serialized = serde_json::to_string(&expected).expect("serialize config");
        let via_serde: TlsClientConfig =
            serde_json::from_str(&serialized).expect("deserialize config");
        let config_path = directory.path().join("tls.toml");
        fs::write(
            &config_path,
            format!(
                "server_name = 'localhost'\nroot_source = 'custom_only'\nca_bundle_paths = ['{}']\nalpn_protocols = ['h2', 'http/1.1']\n",
                ca_path.display()
            ),
        )
        .expect("write TOML config");
        let via_toml = TlsClientConfig::layered()
            .from_path(&config_path)
            .expect("TOML config layer")
            .build()
            .expect("TOML config validation");
        let ca_paths_json = serde_json::to_string(&vec![ca_path.to_string_lossy()])
            .expect("encode Windows-safe JSON CA path");
        let via_env = temp_env::with_vars(
            [
                ("TLS_CLIENT_SERVER_NAME", Some("localhost")),
                ("TLS_CLIENT_ROOT_SOURCE", Some("custom_only")),
                ("TLS_CLIENT_CA_BUNDLE_PATHS", Some(&ca_paths_json)),
                ("TLS_CLIENT_ALPN_PROTOCOLS", Some("[\"h2\",\"http/1.1\"]")),
            ],
            || TlsClientConfig::from_env().expect("environment config"),
        );

        assert_eq!(expected, via_serde);
        assert_eq!(expected, via_toml);
        assert_eq!(expected, via_env);
        assert_tls_trust_config_round_trip(expected, ca_der, leaf_pem, key_pem).await;
    }

    #[test]
    fn tls_client_settings_parse_roots_and_alpn_from_environment() {
        let loaded = temp_env::with_vars(
            [
                ("TLS_CLIENT_SERVER_NAME", Some("service.example")),
                ("TLS_CLIENT_ROOT_SOURCE", Some("custom_only")),
                ("TLS_CLIENT_CA_BUNDLE_PATHS", Some("[\"/tmp/roots.pem\"]")),
                ("TLS_CLIENT_ALPN_PROTOCOLS", Some("[\"h2\",\"http/1.1\"]")),
            ],
            || TlsClientConfig::from_env().expect("settings from environment"),
        );

        assert_eq!(loaded.server_name, "service.example");
        assert_eq!(loaded.root_source, RootSource::CustomOnly);
        assert_eq!(
            loaded.ca_bundle_paths,
            vec![PathBuf::from("/tmp/roots.pem")]
        );
        assert_eq!(loaded.alpn_protocols, vec!["h2", "http/1.1"]);
    }

    #[test]
    fn tls_trust_layer_call_order() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let path = directory.path().join("tls.toml");
        fs::write(
            &path,
            "server_name = 'file.example'\nroot_source = 'custom_only'\nca_bundle_paths = ['/file/roots.pem']\nalpn_protocols = ['h2']\n",
        )
        .expect("write config");

        let config = temp_env::with_vars([("TLS_CLIENT_SERVER_NAME", Some("env.example"))], || {
            TlsClientConfig::layered()
                .from_path(&path)
                .expect("file layer")
                .from_env()
                .expect("environment layer")
                .with_server_name("explicit.example")
                .build()
                .expect("validated config")
        });

        assert_eq!(config.server_name, "explicit.example");
        assert_eq!(config.root_source, RootSource::CustomOnly);
        assert_eq!(
            config.ca_bundle_paths,
            vec![PathBuf::from("/file/roots.pem")]
        );
        assert_eq!(config.alpn_protocols, vec!["h2"]);
    }

    #[test]
    fn tls_trust_validation_rejects_empty_sources_and_paths() {
        let outcome = TlsClientConfig::layered()
            .with_server_name("service.example")
            .with_root_source(RootSource::CustomOnly)
            .build();

        assert!(outcome.is_err());
        let empty_path = TlsClientConfig::layered()
            .with_server_name("service.example")
            .with_ca_bundle_paths(vec![PathBuf::new()])
            .build();
        assert!(empty_path.is_err());
    }

    #[cfg(windows)]
    #[proxima::test(runtime = "tokio")]
    async fn tls_trust_native_store_round_trip() {
        struct UserRootStore {
            handle: HCERTSTORE,
            certificate: Vec<u8>,
            installed: bool,
        }

        impl UserRootStore {
            fn install(certificate: Vec<u8>) -> Self {
                let root_name = OsStr::new("ROOT")
                    .encode_wide()
                    .chain(iter::once(0))
                    .collect::<Vec<_>>();
                // the wide ROOT name remains alive for the entire system-store open call
                let handle = unsafe {
                    CertOpenStore(
                        CERT_STORE_PROV_SYSTEM_W,
                        0,
                        0,
                        CERT_SYSTEM_STORE_CURRENT_USER
                            | CERT_STORE_OPEN_EXISTING_FLAG
                            | CERT_STORE_MAXIMUM_ALLOWED_FLAG,
                        root_name.as_ptr().cast(),
                    )
                };
                assert!(
                    !handle.is_null(),
                    "open current-user ROOT certificate store"
                );
                let mut context = ptr::null_mut();
                // the store and DER slice are valid for the duration of certificate insertion
                let added = unsafe {
                    CertAddEncodedCertificateToStore(
                        handle,
                        X509_ASN_ENCODING,
                        certificate.as_ptr(),
                        certificate.len() as u32,
                        CERT_STORE_ADD_ALWAYS,
                        &mut context,
                    )
                };
                if added == 0 {
                    // the store was opened by this fixture and no guard exists yet
                    unsafe { CertCloseStore(handle, 0) };
                    panic!("install test CA in current-user ROOT store");
                }
                if !context.is_null() {
                    // insertion returned this owned certificate context
                    unsafe { CertFreeCertificateContext(context) };
                }
                Self {
                    handle,
                    certificate,
                    installed: true,
                }
            }

            fn remove(&mut self) -> bool {
                if !self.installed {
                    return true;
                }
                let mut context = ptr::null_mut();
                loop {
                    // enumeration advances the previous context in this open store
                    context = unsafe { CertEnumCertificatesInStore(self.handle, context) };
                    if context.is_null() {
                        break;
                    }
                    // enumeration returned a live context with owned encoded bytes
                    let candidate = unsafe {
                        slice::from_raw_parts(
                            (*context).pbCertEncoded,
                            (*context).cbCertEncoded as usize,
                        )
                    };
                    if candidate == self.certificate {
                        let deleted = unsafe { CertDeleteCertificateFromStore(context) };
                        if deleted != 0 {
                            self.installed = false;
                            return true;
                        }
                        unsafe { CertFreeCertificateContext(context) };
                        return false;
                    }
                }
                false
            }
        }

        impl Drop for UserRootStore {
            fn drop(&mut self) {
                let _ = self.remove();
                // this guard owns the open store handle
                unsafe { CertCloseStore(self.handle, 0) };
            }
        }

        let (certificate, _, leaf_pem, key_pem) = generate_ca_and_leaf();
        let acceptor =
            build_acceptor_futures_io(&TlsConfig::pem(leaf_pem, key_pem)).expect("TLS acceptor");
        let listener = TokioTcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))
            .await
            .expect("bind loopback listener");
        let local = match listener.local_addr().expect("local address") {
            proxima_primitives::stream::BindAddr::Tcp(address) => address,
            other => panic!("expected TCP address, got {other:?}"),
        };
        let server = tokio::spawn(async move {
            for _ in 0..6 {
                let connection = listener.accept().await.expect("accept connection");
                let Ok(mut tls) = acceptor.accept(connection).await else {
                    continue;
                };
                let mut payload = [0_u8; 14];
                if tls.read_exact(&mut payload).await.is_ok() {
                    let _ = tls.write_all(&payload).await;
                    let _ = tls.flush().await;
                }
            }
        });
        let config = TlsClientConfig::builder()
            .server_name("localhost".to_string())
            .root_source(RootSource::Native)
            .build();
        let mut incumbent_roots = RootCertStore::empty();
        incumbent_roots
            .add(CertificateDer::from(certificate.clone()))
            .expect("add local CA to incumbent trust store");
        let incumbent_config = ClientConfig::builder()
            .with_root_certificates(incumbent_roots)
            .with_no_client_auth();
        let incumbent_upstream = TlsStreamUpstream::new(
            TokioTcpUpstream::new(local),
            "localhost",
            Arc::new(incumbent_config),
        );
        let configured_connect = async || -> Result<bool, io::Error> {
            let upstream = TlsStreamUpstream::from_config(TokioTcpUpstream::new(local), &config)
                .expect("native-only TLS upstream");
            let mut connection = upstream.connect().await?;
            let payload = *b"tls root probe";
            connection.write_all(&payload).await?;
            let mut reply = [0_u8; 14];
            connection.read_exact(&mut reply).await?;
            Ok(reply == payload)
        };
        let incumbent_connect = async || -> Result<bool, String> {
            let mut connection = incumbent_upstream
                .connect()
                .await
                .map_err(|error| error.to_string())?;
            let payload = *b"tls root probe";
            connection
                .write_all(&payload)
                .await
                .map_err(|error| error.to_string())?;
            let mut reply = [0_u8; 14];
            connection
                .read_exact(&mut reply)
                .await
                .map_err(|error| error.to_string())?;
            Ok(reply == payload)
        };

        let rejected_before_install = configured_connect().await;
        assert!(
            rejected_before_install
                .as_ref()
                .is_err_and(|error| error.kind() == io::ErrorKind::InvalidData),
            "native-only trust must reject the certificate before install: {rejected_before_install:?}"
        );
        assert!(
            incumbent_connect().await == Ok(true),
            "incumbent trust must pass before install"
        );
        let mut user_roots = UserRootStore::install(certificate);
        assert!(
            configured_connect()
                .await
                .as_ref()
                .is_ok_and(|matched| *matched),
            "native-only trust must accept the installed CA"
        );
        assert!(
            incumbent_connect().await == Ok(true),
            "incumbent trust must pass while installed"
        );
        assert!(
            user_roots.remove(),
            "remove the exact test CA from the user store"
        );
        let rejected_after_removal = configured_connect().await;
        assert!(
            rejected_after_removal
                .as_ref()
                .is_err_and(|error| error.kind() == io::ErrorKind::InvalidData),
            "native-only trust must reject the certificate after removal: {rejected_after_removal:?}"
        );
        assert!(
            incumbent_connect().await == Ok(true),
            "incumbent trust must pass after removal"
        );
        server.await.expect("join TLS server");
    }

    #[test]
    fn tls_client_config_rejects_empty_server_name() {
        let cfg = TlsClientConfig::builder()
            .server_name(String::new())
            .build();
        assert!(cfg.validate().is_err());
    }
}
