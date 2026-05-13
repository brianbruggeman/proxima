//! Std implementation of proxima-tls — gated entirely behind the
//! `std` feature in lib.rs.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use rustls::RootCertStore;
use rustls::ServerConfig;
use rustls::pki_types::{CertificateDer, PrivateKeyDer, ServerName, pem::PemObject};
use rustls::server::{ClientHello, ResolvesServerCert, WebPkiClientVerifier};
use rustls::sign::CertifiedKey;
#[cfg(feature = "tokio")]
use tokio_rustls::TlsAcceptor;

use proxima_core::ProximaError;

/// Listener-side TLS configuration. Construction is type-safe via
/// `TlsMode`; serialization to the spec JSON is symmetric so the
/// config-file path produces the same struct.
#[derive(Debug, Clone)]
pub struct TlsConfig {
    pub mode: TlsMode,
    /// ALPN protocols offered during the handshake, most-preferred
    /// first. `["h2", "http/1.1"]` is the conventional default for
    /// HTTPS listeners that speak both.
    pub alpn: Vec<Vec<u8>>,
    /// Client-cert (mTLS) policy. Default `Disabled` — handshakes
    /// don't request a client cert. `Optional` requests one but lets
    /// the client skip; `Required` rejects clients that don't
    /// present a chain rooted in `trust_anchors`.
    pub client_auth: ClientAuth,
    /// TLS 1.3 session resumption via stateless tickets (RFC 8446
    /// §4.6.1). Default `false` — handshakes are full each time.
    /// Enabling makes resumed connections faster (1-RTT instead of
    /// full handshake) but lets a ticket-key compromise decrypt
    /// resumed sessions; deployments that need perfect forward
    /// secrecy across cert rotations leave this off. The ticketer
    /// is `rustls::Ticketer::new()` — operator key rotation is the
    /// ticketer's responsibility; for finer control build your own
    /// `ProducesTickets` and attach it via `ServerConfig` directly.
    pub session_resumption: bool,
    /// Static OCSP response (DER bytes) to staple onto the
    /// `CertifiedKey`. Set this when you fetch the responder
    /// off-band (cron job, sidecar) and refresh the file before
    /// expiry. Proxima does not fetch / cache OCSP responses
    /// itself — that's the OCSP responder concern, application-layer.
    /// `None` = no stapling.
    pub ocsp_response: Option<Vec<u8>>,
}

impl TlsConfig {
    /// Self-signed cert covering `localhost` + `127.0.0.1`. Dev only —
    /// browsers and standard TLS clients will reject it without a
    /// trust override.
    #[must_use]
    pub fn self_signed() -> Self {
        Self {
            mode: TlsMode::SelfSigned,
            alpn: default_alpn(),
            client_auth: ClientAuth::Disabled,
            session_resumption: false,
            ocsp_response: None,
        }
    }

    /// PEM cert chain + PKCS#8 / PKCS#1 / SEC1 private key as byte
    /// payloads. Bytes-tier ctor — no filesystem; caller supplies the
    /// PEM contents (read from disk, embedded in the binary, fetched
    /// over the wire, whatever).
    #[must_use]
    pub fn pem(cert: Vec<u8>, key: Vec<u8>) -> Self {
        Self {
            mode: TlsMode::Pem { cert, key },
            alpn: default_alpn(),
            client_auth: ClientAuth::Disabled,
            session_resumption: false,
            ocsp_response: None,
        }
    }

    /// Convenience: read `cert` and `key` from the filesystem and
    /// construct a [`Self::pem`] config. Std-only — bytes-tier
    /// consumers call [`Self::pem`] directly.
    pub fn files(cert: PathBuf, key: PathBuf) -> Result<Self, ProximaError> {
        let cert_bytes = std::fs::read(&cert).map_err(|err| {
            ProximaError::Config(format!("tls: read cert {}: {err}", cert.display()))
        })?;
        let key_bytes = std::fs::read(&key).map_err(|err| {
            ProximaError::Config(format!("tls: read key {}: {err}", key.display()))
        })?;
        Ok(Self::pem(cert_bytes, key_bytes))
    }

    #[must_use]
    pub fn with_alpn(mut self, alpn: Vec<Vec<u8>>) -> Self {
        self.alpn = alpn;
        self
    }

    #[must_use]
    pub fn with_client_auth(mut self, client_auth: ClientAuth) -> Self {
        self.client_auth = client_auth;
        self
    }

    #[must_use]
    pub fn with_session_resumption(mut self, enabled: bool) -> Self {
        self.session_resumption = enabled;
        self
    }

    #[must_use]
    pub fn with_ocsp_response(mut self, ocsp_der: Vec<u8>) -> Self {
        self.ocsp_response = Some(ocsp_der);
        self
    }
}

/// mTLS policy. `trust_anchors` is the PEM bytes of CA roots the
/// listener will accept client certs from. `Required` rejects
/// unauthenticated handshakes; `Optional` requests a cert but lets
/// clients omit it (and still validates any chain they do present).
#[derive(Debug, Clone)]
pub enum ClientAuth {
    Disabled,
    Optional { trust_anchors: Vec<u8> },
    Required { trust_anchors: Vec<u8> },
}

/// Cert sources we can build a `ServerConfig` from. Variants we don't
/// implement aren't named — when ACME / mkcert ship, they get added
/// alongside the implementation.
#[derive(Debug, Clone)]
pub enum TlsMode {
    /// Fresh ECDSA-P256 cert generated in memory each time the
    /// listener starts. Covers `localhost` + `127.0.0.1` + `::1`.
    SelfSigned,
    /// PEM cert chain + private key as byte payloads. Caller-supplied;
    /// no filesystem dep.
    Pem { cert: Vec<u8>, key: Vec<u8> },
    /// Multi-tenant TLS: one cert per hostname, selected per
    /// `ClientHello.server_name` (SNI). Map keys are exact hostnames
    /// (case-insensitive); the first entry doubles as the default
    /// for clients that don't send SNI (some legacy stacks).
    MultiSni { hosts: BTreeMap<String, SniCert> },
}

/// One SNI-selected cert chain + key pair.
#[derive(Debug, Clone)]
pub struct SniCert {
    pub cert: Vec<u8>,
    pub key: Vec<u8>,
}

#[allow(clippy::vec_init_then_push, unused_mut)]
fn default_alpn() -> Vec<Vec<u8>> {
    // each push is gated by a feature; the macro form doesn't compose
    // with cfg attributes on individual entries.
    let mut alpn: Vec<Vec<u8>> = Vec::new();
    #[cfg(feature = "http2")]
    alpn.push(b"h2".to_vec());
    #[cfg(feature = "http1")]
    alpn.push(b"http/1.1".to_vec());
    alpn
}

/// Build a `tokio_rustls::TlsAcceptor` ready to wrap accepted TCP
/// connections. Errors are propagated as `ProximaError::Config` so
/// they surface at listener startup, not on the first connection.
///
/// For backends using `futures::io::AsyncRead`/`AsyncWrite` (DPDK,
/// glommio, io_uring direct), use [`build_acceptor_futures_io`]
/// instead (requires the `futures-io` feature).
#[cfg(feature = "tokio")]
pub fn build_acceptor(config: &TlsConfig) -> Result<TlsAcceptor, ProximaError> {
    let server_config = build_server_config(config)?;
    Ok(TlsAcceptor::from(Arc::new(server_config)))
}

/// Build a `futures_rustls::TlsAcceptor` for backends that speak
/// `futures::io::AsyncRead + AsyncWrite` natively (proxima-stream,
/// DPDK, glommio, io_uring direct). Same `rustls::ServerConfig`
/// underneath the hood — different async adapter.
///
/// Requires the `futures-io` feature.
#[cfg(feature = "futures-io")]
pub fn build_acceptor_futures_io(
    config: &TlsConfig,
) -> Result<futures_rustls::TlsAcceptor, ProximaError> {
    let server_config = build_server_config(config)?;
    Ok(futures_rustls::TlsAcceptor::from(Arc::new(server_config)))
}

/// Build the runtime-neutral `rustls::ServerConfig`. Public so callers
/// that need a TLS adapter other than tokio-rustls / futures-rustls
/// can build their own.
pub fn build_server_config(config: &TlsConfig) -> Result<ServerConfig, ProximaError> {
    let builder = ServerConfig::builder();
    let builder = match &config.client_auth {
        ClientAuth::Disabled => builder.with_no_client_auth(),
        ClientAuth::Required { trust_anchors } => {
            let verifier = client_cert_verifier(trust_anchors, false)?;
            builder.with_client_cert_verifier(verifier)
        }
        ClientAuth::Optional { trust_anchors } => {
            let verifier = client_cert_verifier(trust_anchors, true)?;
            builder.with_client_cert_verifier(verifier)
        }
    };

    let mut server_config = match &config.mode {
        TlsMode::SelfSigned => {
            let (cert_chain, key) = generate_self_signed()?;
            build_single_cert_config(builder, cert_chain, key, config.ocsp_response.clone())?
        }
        TlsMode::Pem { cert, key } => {
            let (cert_chain, key) = parse_pem_bytes(cert, key)?;
            build_single_cert_config(builder, cert_chain, key, config.ocsp_response.clone())?
        }
        TlsMode::MultiSni { hosts } => {
            // MultiSni's per-host OCSP would attach to each CertifiedKey;
            // current API only exposes one global OCSP slot, so we
            // ignore it here. Per-host stapling lands as a follow-up
            // when there's demand.
            let resolver = SniResolver::build(hosts)?;
            builder.with_cert_resolver(Arc::new(resolver))
        }
    };
    server_config.alpn_protocols = config.alpn.clone();
    if config.session_resumption {
        // rustls's stateless ticketer encrypts session secrets with a
        // process-local key; the key rotates inside the ticketer. Tied
        // to the active crypto provider — fails if none is installed.
        let ticketer = rustls::crypto::aws_lc_rs::Ticketer::new()
            .map_err(|err| ProximaError::Config(format!("tls: ticketer: {err}")))?;
        server_config.ticketer = ticketer;
    }
    Ok(server_config)
}

fn build_single_cert_config(
    builder: rustls::ConfigBuilder<ServerConfig, rustls::server::WantsServerCert>,
    cert_chain: Vec<CertificateDer<'static>>,
    key: PrivateKeyDer<'static>,
    ocsp_response: Option<Vec<u8>>,
) -> Result<ServerConfig, ProximaError> {
    match ocsp_response {
        Some(ocsp) => builder
            .with_single_cert_with_ocsp(cert_chain, key, ocsp)
            .map_err(|err| ProximaError::Config(format!("tls: build ServerConfig (OCSP): {err}"))),
        None => builder
            .with_single_cert(cert_chain, key)
            .map_err(|err| ProximaError::Config(format!("tls: build ServerConfig: {err}"))),
    }
}

fn client_cert_verifier(
    trust_anchors: &[u8],
    optional: bool,
) -> Result<Arc<dyn rustls::server::danger::ClientCertVerifier>, ProximaError> {
    let mut roots = RootCertStore::empty();
    let anchors: Vec<CertificateDer<'static>> = CertificateDer::pem_slice_iter(trust_anchors)
        .collect::<Result<_, _>>()
        .map_err(|err| ProximaError::Config(format!("tls: parse trust anchors: {err}")))?;
    if anchors.is_empty() {
        return Err(ProximaError::Config(
            "tls: trust anchors contains no certificates".into(),
        ));
    }
    let (added, _ignored) = roots.add_parsable_certificates(anchors);
    if added == 0 {
        return Err(ProximaError::Config(format!(
            "tls: no usable certs in trust anchors {trust_anchors:?}"
        )));
    }
    let mut builder = WebPkiClientVerifier::builder(Arc::new(roots));
    if optional {
        builder = builder.allow_unauthenticated();
    }
    builder
        .build()
        .map_err(|err| ProximaError::Config(format!("tls: build client verifier: {err}")))
}

/// SNI-driven cert resolver. Picks a preloaded `CertifiedKey` per
/// `ClientHello.server_name`. Hosts are lowercased at load; lookups
/// case-insensitive. The first entry doubles as the fallback when
/// the ClientHello carries no SNI (RFC 6066 says SNI is optional).
#[derive(Debug)]
struct SniResolver {
    by_host: BTreeMap<String, Arc<CertifiedKey>>,
    fallback: Arc<CertifiedKey>,
}

impl SniResolver {
    fn build(hosts: &BTreeMap<String, SniCert>) -> Result<Self, ProximaError> {
        if hosts.is_empty() {
            return Err(ProximaError::Config(
                "tls: MultiSni requires at least one host".into(),
            ));
        }
        let provider = rustls::crypto::CryptoProvider::get_default()
            .cloned()
            .ok_or_else(|| {
                ProximaError::Config(
                    "tls: no default rustls crypto provider installed (call install_default)"
                        .into(),
                )
            })?;
        let mut by_host = BTreeMap::new();
        let mut fallback: Option<Arc<CertifiedKey>> = None;
        for (host, sni) in hosts {
            let (cert_chain, key) = parse_pem_bytes(&sni.cert, &sni.key)?;
            let signing_key = provider.key_provider.load_private_key(key).map_err(|err| {
                ProximaError::Config(format!("tls: load sni key for {host}: {err}"))
            })?;
            let certified = Arc::new(CertifiedKey::new(cert_chain, signing_key));
            let lower = host.to_ascii_lowercase();
            if fallback.is_none() {
                fallback = Some(certified.clone());
            }
            by_host.insert(lower, certified);
        }
        let Some(fallback) = fallback else {
            return Err(ProximaError::Config(
                "tls: MultiSni resolver produced no fallback cert".into(),
            ));
        };
        Ok(Self { by_host, fallback })
    }
}

impl ResolvesServerCert for SniResolver {
    fn resolve(&self, client_hello: ClientHello<'_>) -> Option<Arc<CertifiedKey>> {
        match client_hello.server_name() {
            Some(name) => {
                let lower = name.to_ascii_lowercase();
                self.by_host
                    .get(&lower)
                    .cloned()
                    .or_else(|| Some(self.fallback.clone()))
            }
            None => Some(self.fallback.clone()),
        }
    }
}

// `ServerName` is required for the rustls trait surface; bring the
// `_used` so the import isn't dead-code-pruned on builds that don't
// touch SNI paths directly from this module.
const _: fn() = || {
    let _: Option<ServerName<'static>> = None;
};

fn generate_self_signed()
-> Result<(Vec<CertificateDer<'static>>, PrivateKeyDer<'static>), ProximaError> {
    let subject_alt_names = vec![
        "localhost".to_string(),
        "127.0.0.1".to_string(),
        "::1".to_string(),
    ];
    let generated = rcgen::generate_simple_self_signed(subject_alt_names)
        .map_err(|err| ProximaError::Config(format!("tls: generate self-signed cert: {err}")))?;
    let cert_der = CertificateDer::from(generated.cert.der().to_vec());
    // serialize_private_key_der returns PKCS#8 bytes; rustls accepts that
    // directly via PrivateKeyDer::Pkcs8.
    let key_der = PrivateKeyDer::Pkcs8(generated.signing_key.serialize_der().into());
    Ok((vec![cert_der], key_der))
}

fn parse_pem_bytes(
    cert: &[u8],
    key: &[u8],
) -> Result<(Vec<CertificateDer<'static>>, PrivateKeyDer<'static>), ProximaError> {
    let cert_chain: Vec<CertificateDer<'static>> = CertificateDer::pem_slice_iter(cert)
        .collect::<Result<_, _>>()
        .map_err(|err| ProximaError::Config(format!("tls: parse cert PEM: {err}")))?;
    if cert_chain.is_empty() {
        return Err(ProximaError::Config(
            "tls: cert PEM bytes contain no certificates".into(),
        ));
    }
    let key = PrivateKeyDer::from_pem_slice(key)
        .map_err(|err| ProximaError::Config(format!("tls: parse key PEM: {err}")))?;
    Ok((cert_chain, key))
}

/// Wire-format spec keys. The listener serializes a `TlsConfig` into
/// the spec under `__proxima_tls` so the HTTP protocol picks it up
/// without a trait-signature change. Reserved key name — config-file
/// users should use the typed `[listen.tls]` block, which the loader
/// translates to the same JSON.
pub const SPEC_KEY: &str = "__proxima_tls";

/// Serialize a `TlsConfig` into a `serde_json::Value` shape the
/// loader / listener round-trips through the spec.
///
/// Cert / key / trust-anchor PEM payloads serialize as base64-encoded
/// strings (`*_pem_b64` fields). Human-edited TOML config files use a
/// path-based settings layer that loads files into bytes BEFORE
/// constructing the `TlsConfig`; once a `TlsConfig` exists, all
/// material is in-memory bytes that round-trip via base64.
#[must_use]
pub fn config_to_spec_value(config: &TlsConfig) -> serde_json::Value {
    use base64::Engine as _;
    let b64 = |bytes: &[u8]| base64::engine::general_purpose::STANDARD.encode(bytes);
    let mode = match &config.mode {
        TlsMode::SelfSigned => serde_json::json!({"kind": "self_signed"}),
        TlsMode::Pem { cert, key } => serde_json::json!({
            "kind": "pem",
            "cert_pem_b64": b64(cert),
            "key_pem_b64": b64(key),
        }),
        TlsMode::MultiSni { hosts } => {
            let mapping: serde_json::Map<String, serde_json::Value> = hosts
                .iter()
                .map(|(host, sni)| {
                    (
                        host.clone(),
                        serde_json::json!({
                            "cert_pem_b64": b64(&sni.cert),
                            "key_pem_b64": b64(&sni.key),
                        }),
                    )
                })
                .collect();
            serde_json::json!({"kind": "multi_sni", "hosts": mapping})
        }
    };
    let alpn: Vec<String> = config
        .alpn
        .iter()
        .map(|bytes| String::from_utf8_lossy(bytes).into_owned())
        .collect();
    let client_auth = match &config.client_auth {
        ClientAuth::Disabled => serde_json::json!({"kind": "disabled"}),
        ClientAuth::Optional { trust_anchors } => serde_json::json!({
            "kind": "optional",
            "trust_anchors_pem_b64": b64(trust_anchors),
        }),
        ClientAuth::Required { trust_anchors } => serde_json::json!({
            "kind": "required",
            "trust_anchors_pem_b64": b64(trust_anchors),
        }),
    };
    // OCSP response is opaque DER bytes; base64 the wire shape so
    // JSON / TOML round-trip cleanly. `None` omits the key.
    let ocsp_b64 = config.ocsp_response.as_ref().map(|bytes| {
        use base64::Engine as _;
        base64::engine::general_purpose::STANDARD.encode(bytes)
    });
    let mut root = serde_json::json!({
        "mode": mode,
        "alpn": alpn,
        "client_auth": client_auth,
        "session_resumption": config.session_resumption,
    });
    if let (Some(b64), serde_json::Value::Object(table)) = (ocsp_b64, &mut root) {
        table.insert("ocsp_response_b64".into(), serde_json::Value::String(b64));
    }
    root
}

/// Parse the JSON shape `config_to_spec_value` produces back into a
/// `TlsConfig`. Returns `Ok(None)` if the value is missing or null;
/// returns a typed error on shape violations.
pub fn config_from_spec_value(
    value: Option<&serde_json::Value>,
) -> Result<Option<TlsConfig>, ProximaError> {
    let Some(value) = value else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    let table = value.as_object().ok_or_else(|| {
        ProximaError::Config("tls spec must be an object with `mode` and `alpn`".into())
    })?;
    use base64::Engine as _;
    let decode_b64 = |raw: &str, label: &str| -> Result<Vec<u8>, ProximaError> {
        base64::engine::general_purpose::STANDARD
            .decode(raw)
            .map_err(|err| ProximaError::Config(format!("tls.{label}: invalid base64: {err}")))
    };
    let mode_value = table.get("mode").ok_or_else(|| {
        ProximaError::Config("tls spec requires `mode` (self_signed | pem | multi_sni)".into())
    })?;
    let kind = mode_value
        .get("kind")
        .and_then(|raw| raw.as_str())
        .ok_or_else(|| ProximaError::Config("tls.mode.kind must be a string".into()))?;
    let mode = match kind {
        "self_signed" => TlsMode::SelfSigned,
        "pem" => {
            let cert_b64 = mode_value
                .get("cert_pem_b64")
                .and_then(|raw| raw.as_str())
                .ok_or_else(|| {
                    ProximaError::Config("tls.mode.pem requires `cert_pem_b64`".into())
                })?;
            let key_b64 = mode_value
                .get("key_pem_b64")
                .and_then(|raw| raw.as_str())
                .ok_or_else(|| {
                    ProximaError::Config("tls.mode.pem requires `key_pem_b64`".into())
                })?;
            TlsMode::Pem {
                cert: decode_b64(cert_b64, "mode.pem.cert_pem_b64")?,
                key: decode_b64(key_b64, "mode.pem.key_pem_b64")?,
            }
        }
        "multi_sni" => {
            let hosts_value = mode_value
                .get("hosts")
                .and_then(|raw| raw.as_object())
                .ok_or_else(|| {
                    ProximaError::Config(
                        "tls.mode.multi_sni requires `hosts` object {host: {cert_pem_b64, key_pem_b64}}".into(),
                    )
                })?;
            let mut hosts = BTreeMap::new();
            for (host, sni_value) in hosts_value {
                let sni = sni_value.as_object().ok_or_else(|| {
                    ProximaError::Config(format!(
                        "tls.mode.multi_sni.hosts.{host} must be an object with `cert_pem_b64`,`key_pem_b64`"
                    ))
                })?;
                let cert_b64 = sni
                    .get("cert_pem_b64")
                    .and_then(|raw| raw.as_str())
                    .ok_or_else(|| {
                        ProximaError::Config(format!(
                            "tls.mode.multi_sni.hosts.{host}: cert_pem_b64 required"
                        ))
                    })?;
                let key_b64 = sni
                    .get("key_pem_b64")
                    .and_then(|raw| raw.as_str())
                    .ok_or_else(|| {
                        ProximaError::Config(format!(
                            "tls.mode.multi_sni.hosts.{host}: key_pem_b64 required"
                        ))
                    })?;
                hosts.insert(
                    host.clone(),
                    SniCert {
                        cert: decode_b64(cert_b64, "mode.multi_sni.hosts.cert_pem_b64")?,
                        key: decode_b64(key_b64, "mode.multi_sni.hosts.key_pem_b64")?,
                    },
                );
            }
            if hosts.is_empty() {
                return Err(ProximaError::Config(
                    "tls.mode.multi_sni.hosts must contain at least one entry".into(),
                ));
            }
            TlsMode::MultiSni { hosts }
        }
        other => {
            return Err(ProximaError::Config(format!(
                "tls.mode.kind `{other}` unsupported; expected self_signed | pem | multi_sni"
            )));
        }
    };
    let alpn = match table.get("alpn") {
        Some(serde_json::Value::Array(items)) => items
            .iter()
            .map(|entry| match entry.as_str() {
                Some(name) => Ok(name.as_bytes().to_vec()),
                None => Err(ProximaError::Config(
                    "tls.alpn entries must be strings".into(),
                )),
            })
            .collect::<Result<Vec<_>, _>>()?,
        Some(_) => return Err(ProximaError::Config("tls.alpn must be an array".into())),
        None => default_alpn(),
    };
    let client_auth = match table.get("client_auth") {
        Some(serde_json::Value::Null) | None => ClientAuth::Disabled,
        Some(value) => {
            let auth_table = value
                .as_object()
                .ok_or_else(|| ProximaError::Config("tls.client_auth must be an object".into()))?;
            let kind = auth_table
                .get("kind")
                .and_then(|raw| raw.as_str())
                .ok_or_else(|| ProximaError::Config("tls.client_auth.kind required".into()))?;
            match kind {
                "disabled" => ClientAuth::Disabled,
                "optional" => {
                    let trust_b64 = auth_table
                        .get("trust_anchors_pem_b64")
                        .and_then(|raw| raw.as_str())
                        .ok_or_else(|| {
                            ProximaError::Config(
                                "tls.client_auth.optional requires `trust_anchors_pem_b64`".into(),
                            )
                        })?;
                    ClientAuth::Optional {
                        trust_anchors: decode_b64(
                            trust_b64,
                            "client_auth.optional.trust_anchors_pem_b64",
                        )?,
                    }
                }
                "required" => {
                    let trust_b64 = auth_table
                        .get("trust_anchors_pem_b64")
                        .and_then(|raw| raw.as_str())
                        .ok_or_else(|| {
                            ProximaError::Config(
                                "tls.client_auth.required requires `trust_anchors_pem_b64`".into(),
                            )
                        })?;
                    ClientAuth::Required {
                        trust_anchors: decode_b64(
                            trust_b64,
                            "client_auth.required.trust_anchors_pem_b64",
                        )?,
                    }
                }
                other => {
                    return Err(ProximaError::Config(format!(
                        "tls.client_auth.kind `{other}` unsupported; expected disabled | optional | required"
                    )));
                }
            }
        }
    };
    let session_resumption = table
        .get("session_resumption")
        .and_then(|raw| raw.as_bool())
        .unwrap_or(false);
    let ocsp_response = match table.get("ocsp_response_b64") {
        Some(serde_json::Value::String(b64)) => {
            use base64::Engine as _;
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(b64.as_bytes())
                .map_err(|err| {
                    ProximaError::Config(format!("tls.ocsp_response_b64 invalid base64: {err}"))
                })?;
            Some(bytes)
        }
        Some(serde_json::Value::Null) | None => None,
        Some(_) => {
            return Err(ProximaError::Config(
                "tls.ocsp_response_b64 must be a base64 string".into(),
            ));
        }
    };
    Ok(Some(TlsConfig {
        mode,
        alpn,
        client_auth,
        session_resumption,
        ocsp_response,
    }))
}

// gated on `tokio`: every case here exercises `build_acceptor`
// (tokio_rustls-backed). The tokio-free default runs the equivalent
// coverage in `connector.rs` against `build_acceptor_futures_io` instead.
#[cfg(all(test, feature = "tokio"))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn self_signed_builds_a_usable_acceptor() {
        let config = TlsConfig::self_signed();
        let acceptor = build_acceptor(&config).expect("self-signed acceptor builds");
        // No public state to inspect; the fact that build succeeded means
        // ECDSA-P256 keygen, cert assembly, and rustls config wiring all worked.
        let _ = acceptor;
    }

    #[test]
    fn missing_cert_file_returns_typed_config_error() {
        let outcome = TlsConfig::files(
            PathBuf::from("/proxima/does/not/exist/cert.pem"),
            PathBuf::from("/proxima/does/not/exist/key.pem"),
        );
        assert!(matches!(outcome, Err(ProximaError::Config(_))));
    }

    #[test]
    fn invalid_pem_bytes_return_typed_config_error() {
        let config = TlsConfig::pem(b"not a real pem".to_vec(), b"nope".to_vec());
        let outcome = build_acceptor(&config);
        assert!(matches!(outcome, Err(ProximaError::Config(_))));
    }

    #[test]
    fn pem_bytes_from_real_cert_build_a_usable_acceptor() {
        // real rcgen cert -> PEM bytes -> bytes-tier `pem()` -> rustls.
        // proves the bytes path parses what a sysadmin actually has in a
        // pemfile, not just that we round-trip arbitrary bytes through base64.
        let generated =
            rcgen::generate_simple_self_signed(vec!["localhost".to_string()]).expect("rcgen");
        let cert_pem = generated.cert.pem().into_bytes();
        let key_pem = generated.signing_key.serialize_pem().into_bytes();
        let config = TlsConfig::pem(cert_pem, key_pem);
        let _ = build_acceptor(&config).expect("real-pem acceptor builds");
    }

    #[test]
    fn pem_bytes_via_files_round_trip_through_filesystem() {
        // sysadmin path: cert + key on disk -> TlsConfig::files reads them
        // -> bytes-tier `pem()` -> rustls. Proves the std-only files()
        // convenience matches what pem() would have built directly.
        let generated =
            rcgen::generate_simple_self_signed(vec!["localhost".to_string()]).expect("rcgen");
        let tmp = std::env::temp_dir().join(format!(
            "proxima-tls-pem-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or_default()
        ));
        std::fs::create_dir_all(&tmp).expect("mkdtemp");
        let cert_path = tmp.join("cert.pem");
        let key_path = tmp.join("key.pem");
        std::fs::write(&cert_path, generated.cert.pem()).expect("write cert");
        std::fs::write(&key_path, generated.signing_key.serialize_pem()).expect("write key");
        let config = TlsConfig::files(cert_path, key_path).expect("files ctor");
        let _ = build_acceptor(&config).expect("acceptor builds from files-loaded pem");
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn real_trust_anchors_build_a_client_verifier() {
        // mTLS path: real CA PEM bytes -> WebPkiClientVerifier construction
        // succeeds. The existing `invalid_trust_anchors` test proves garbage
        // is rejected; this one proves a valid anchor goes through.
        let ca =
            rcgen::generate_simple_self_signed(vec!["proxima-ca".to_string()]).expect("rcgen ca");
        let ca_pem = ca.cert.pem().into_bytes();
        let leaf =
            rcgen::generate_simple_self_signed(vec!["localhost".to_string()]).expect("rcgen leaf");
        let leaf_cert = leaf.cert.pem().into_bytes();
        let leaf_key = leaf.signing_key.serialize_pem().into_bytes();
        let config = TlsConfig::pem(leaf_cert, leaf_key).with_client_auth(ClientAuth::Required {
            trust_anchors: ca_pem,
        });
        let _ = build_acceptor(&config).expect("acceptor builds with real mTLS anchors");
    }

    #[test]
    fn spec_round_trips_self_signed_with_default_alpn() {
        let original = TlsConfig::self_signed();
        let value = config_to_spec_value(&original);
        let parsed = config_from_spec_value(Some(&value))
            .expect("parse")
            .expect("Some");
        assert!(matches!(parsed.mode, TlsMode::SelfSigned));
        assert_eq!(parsed.alpn, original.alpn);
    }

    #[test]
    fn spec_round_trips_pem_with_custom_alpn() {
        let generated =
            rcgen::generate_simple_self_signed(vec!["localhost".to_string()]).expect("rcgen");
        let cert_bytes = generated.cert.pem().into_bytes();
        let key_bytes = generated.signing_key.serialize_pem().into_bytes();
        let original = TlsConfig::pem(cert_bytes.clone(), key_bytes.clone())
            .with_alpn(vec![b"h3".to_vec(), b"h2".to_vec()]);
        let value = config_to_spec_value(&original);
        let parsed = config_from_spec_value(Some(&value))
            .expect("parse")
            .expect("Some");
        let TlsMode::Pem { cert, key } = &parsed.mode else {
            panic!("expected Pem variant");
        };
        assert_eq!(cert, &cert_bytes);
        assert_eq!(key, &key_bytes);
        assert_eq!(parsed.alpn, vec![b"h3".to_vec(), b"h2".to_vec()]);
        // the bytes round-trip AND parse: rustls accepts what came out of the spec.
        let _ = build_acceptor(&parsed).expect("acceptor builds from spec-round-tripped pem");
    }

    #[test]
    fn spec_none_returns_none_not_error() {
        let parsed = config_from_spec_value(None).expect("ok");
        assert!(parsed.is_none());
    }

    #[test]
    fn spec_unknown_mode_kind_returns_typed_error() {
        let value = serde_json::json!({
            "mode": {"kind": "acme", "email": "admin@example.com"},
            "alpn": ["h2"],
        });
        let outcome = config_from_spec_value(Some(&value));
        assert!(matches!(outcome, Err(ProximaError::Config(_))));
    }

    #[test]
    fn client_auth_round_trips_through_spec() {
        let ca =
            rcgen::generate_simple_self_signed(vec!["proxima-ca".to_string()]).expect("rcgen ca");
        let anchors = ca.cert.pem().into_bytes();
        let original = TlsConfig::self_signed().with_client_auth(ClientAuth::Required {
            trust_anchors: anchors.clone(),
        });
        let value = config_to_spec_value(&original);
        let parsed = config_from_spec_value(Some(&value))
            .expect("parse")
            .expect("Some");
        match &parsed.client_auth {
            ClientAuth::Required { trust_anchors } => {
                assert_eq!(trust_anchors, &anchors);
            }
            other => panic!("expected Required, got {other:?}"),
        }
        // anchors survive the round-trip AND parse: WebPkiClientVerifier accepts them.
        let _ = build_acceptor(&parsed).expect("acceptor builds with round-tripped anchors");
    }

    #[test]
    fn client_auth_defaults_to_disabled() {
        let config = TlsConfig::self_signed();
        assert!(matches!(config.client_auth, ClientAuth::Disabled));
    }

    #[test]
    fn invalid_trust_anchors_returns_typed_config_error() {
        let config = TlsConfig::self_signed().with_client_auth(ClientAuth::Required {
            trust_anchors: b"garbage not pem".to_vec(),
        });
        let outcome = build_acceptor(&config);
        assert!(matches!(outcome, Err(ProximaError::Config(_))));
    }

    #[test]
    fn multi_sni_round_trips_through_spec() {
        let api = rcgen::generate_simple_self_signed(vec!["api.example.com".to_string()])
            .expect("rcgen api");
        let admin = rcgen::generate_simple_self_signed(vec!["admin.example.com".to_string()])
            .expect("rcgen admin");
        let api_cert = api.cert.pem().into_bytes();
        let api_key = api.signing_key.serialize_pem().into_bytes();
        let admin_cert = admin.cert.pem().into_bytes();
        let admin_key = admin.signing_key.serialize_pem().into_bytes();
        let mut hosts = BTreeMap::new();
        hosts.insert(
            "api.example.com".into(),
            SniCert {
                cert: api_cert.clone(),
                key: api_key.clone(),
            },
        );
        hosts.insert(
            "admin.example.com".into(),
            SniCert {
                cert: admin_cert.clone(),
                key: admin_key.clone(),
            },
        );
        let original = TlsConfig {
            mode: TlsMode::MultiSni { hosts },
            alpn: default_alpn(),
            client_auth: ClientAuth::Disabled,
            session_resumption: false,
            ocsp_response: None,
        };
        let value = config_to_spec_value(&original);
        let parsed = config_from_spec_value(Some(&value))
            .expect("parse")
            .expect("Some");
        let TlsMode::MultiSni { hosts } = parsed.mode.clone() else {
            panic!("expected MultiSni");
        };
        assert_eq!(hosts.len(), 2);
        assert_eq!(hosts.get("api.example.com").unwrap().cert, api_cert);
        assert_eq!(hosts.get("admin.example.com").unwrap().key, admin_key);
        // the round-tripped multi-SNI map builds a usable acceptor — proves each
        // host's cert + key are real material rustls will load into a SniResolver.
        // requires the crypto provider, install on demand.
        let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
        let _ = build_acceptor(&parsed).expect("acceptor builds with round-tripped multi-sni");
    }

    #[test]
    fn session_resumption_defaults_off_and_round_trips() {
        let off = TlsConfig::self_signed();
        assert!(!off.session_resumption);
        let on = TlsConfig::self_signed().with_session_resumption(true);
        let value = config_to_spec_value(&on);
        let parsed = config_from_spec_value(Some(&value))
            .expect("parse")
            .expect("Some");
        assert!(parsed.session_resumption);
    }

    #[test]
    fn ocsp_response_round_trips_via_base64() {
        let ocsp_der: Vec<u8> = vec![0x30, 0x82, 0x01, 0xab, 0x00, 0xfe, 0xed, 0xbe];
        let config = TlsConfig::self_signed().with_ocsp_response(ocsp_der.clone());
        let value = config_to_spec_value(&config);
        let parsed = config_from_spec_value(Some(&value))
            .expect("parse")
            .expect("Some");
        assert_eq!(parsed.ocsp_response.as_deref(), Some(ocsp_der.as_slice()));
    }

    #[test]
    fn multi_sni_empty_hosts_returns_typed_error() {
        let value = serde_json::json!({
            "mode": {"kind": "multi_sni", "hosts": {}},
            "alpn": ["h2"],
        });
        let outcome = config_from_spec_value(Some(&value));
        assert!(matches!(outcome, Err(ProximaError::Config(_))));
    }
}
