//! `PipeFactory` for the `h3-native` protocol — a `proxima::Client`
//! transport that speaks HTTP/3 over the NATIVE sans-IO QUIC + H3 stack
//! (no quinn, no h3-quinn). The dual-surface peer of the quinn
//! `Http3Upstream` (P7), selectable side-by-side per the C41 ruling.
//!
//! Reached via the `type` discriminator — `{"type":"h3-native",
//! "addr":"1.2.3.4:443", "server_name":"example.com"}` or
//! `{"type":"h3-native", "url":"https://example.com:443"}` — the same
//! extensible terminal seam pgwire/redis use. Composes
//! [`H3NativeUpstream`](proxima_http::http3::native::H3NativeUpstream) (the prime
//! UDP `Endpoint` + sans-IO H3 client) as a `SendPipe`.
//!
//! `insecure: true` (dev only) installs an accept-any-cert verifier so a
//! self-signed loopback server (the `dev_self_signed` listener) can be
//! dialed in tests/dev — the client mirror of the listener's dev cert.

use std::future::Future;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::Arc;

use bon::Builder;
use conflaguration::Settings;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use proxima_http::http3::native::H3NativeUpstream;
use proxima_primitives::pipe::handler::{PipeHandle, into_handle};
use proxima_primitives::pipe::pipe_factory::PipeFactory;

use crate::error::ProximaError;

/// A [`PipeFactory`] for the `h3-native` key.
#[derive(Debug, Default)]
pub struct H3NativeUpstreamFactory;

impl H3NativeUpstreamFactory {
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl PipeFactory for H3NativeUpstreamFactory {
    fn name(&self) -> &str {
        "h3-native"
    }

    fn build(
        &self,
        spec: &Value,
        _inner: Option<PipeHandle>,
    ) -> Pin<Box<dyn Future<Output = Result<PipeHandle, ProximaError>> + Send + '_>> {
        let spec = spec.clone();
        Box::pin(async move {
            let config: H3NativeConfig = serde_json::from_value(spec)
                .map_err(|err| ProximaError::Config(format!("h3-native config: {err}")))?;
            let dial = config.into_dial()?;
            let mut upstream = if dial.insecure {
                H3NativeUpstream::with_client_config(
                    dial.addr,
                    dial.server_name,
                    insecure_client_config(),
                )
            } else {
                H3NativeUpstream::new(dial.addr, dial.server_name)
            };
            if let Some(ms) = dial.timeout_ms {
                upstream = upstream.with_timeout(std::time::Duration::from_millis(ms));
            }
            Ok(into_handle(upstream))
        })
    }
}

/// Parsed dial target.
#[derive(Debug)]
pub struct Dial {
    addr: SocketAddr,
    server_name: String,
    insecure: bool,
    /// Per-request timeout (the config-surface twin of
    /// `H3NativeUpstream::with_timeout`). `None` keeps the upstream default.
    timeout_ms: Option<u64>,
}

/// Typed config surface for the `h3-native` upstream. The dial target is
/// either an explicit `addr` (+ optional `server_name`) or a `url` whose host
/// is an IP literal (hostname DNS resolution is a follow-on). Lowered to a
/// [`Dial`] by [`H3NativeConfig::into_dial`], which preserves the historical
/// addr-then-url precedence and IP-literal-only constraint.
#[derive(Debug, Clone, Default, Builder, Deserialize, Serialize, Settings)]
#[settings(prefix = "PROXIMA_H3_NATIVE")]
#[builder(derive(Clone, Debug), on(String, into))]
pub struct H3NativeConfig {
    /// Explicit dial address (`ip:port`). Takes precedence over `url`.
    #[setting(default)]
    #[serde(default)]
    pub addr: Option<String>,

    /// TLS server name; defaults to the dial IP when omitted.
    #[setting(default)]
    #[serde(default)]
    pub server_name: Option<String>,

    /// `https://ip:port` dial target (used when `addr` is absent).
    #[setting(default)]
    #[serde(default)]
    pub url: Option<String>,

    /// Accept-any-cert (dev only) for a self-signed loopback server.
    #[setting(default)]
    #[serde(default)]
    #[builder(default)]
    pub insecure: bool,

    /// Per-request timeout in ms; `None` keeps the upstream default.
    #[setting(default)]
    #[serde(default)]
    pub timeout_ms: Option<u64>,
}

impl H3NativeConfig {
    /// Lower the wire config to a parsed [`Dial`] target.
    pub fn into_dial(self) -> Result<Dial, ProximaError> {
        if let Some(addr) = &self.addr {
            let socket_addr: SocketAddr = addr
                .parse()
                .map_err(|err| ProximaError::Config(format!("h3-native addr: {err}")))?;
            let server_name = self
                .server_name
                .clone()
                .unwrap_or_else(|| socket_addr.ip().to_string());
            return Ok(Dial {
                addr: socket_addr,
                server_name,
                insecure: self.insecure,
                timeout_ms: self.timeout_ms,
            });
        }
        if let Some(url) = &self.url {
            let (host, port) = parse_https_authority(url)?;
            let ip = host.parse::<std::net::IpAddr>().map_err(|_| {
                ProximaError::Config(format!(
                    "h3-native url host '{host}' is not an IP literal; use explicit `addr` until hostname resolution lands"
                ))
            })?;
            return Ok(Dial {
                addr: SocketAddr::new(ip, port),
                server_name: host,
                insecure: self.insecure,
                timeout_ms: self.timeout_ms,
            });
        }
        Err(ProximaError::Config(
            "h3-native spec needs `addr` (+ optional `server_name`) or `url`".to_string(),
        ))
    }
}

/// Split `https://host:port` into `(host, port)` — port defaults to 443.
fn parse_https_authority(url: &str) -> Result<(String, u16), ProximaError> {
    let rest = url
        .strip_prefix("https://")
        .or_else(|| url.strip_prefix("http://"))
        .unwrap_or(url);
    let authority = rest.split(['/', '?', '#']).next().unwrap_or(rest);
    match authority.rsplit_once(':') {
        Some((host, port)) => {
            let port: u16 = port
                .parse()
                .map_err(|err| ProximaError::Config(format!("h3-native url port: {err}")))?;
            Ok((host.to_string(), port))
        }
        None => Ok((authority.to_string(), 443)),
    }
}

/// Accept-any-cert rustls client config (TLS 1.3, ALPN `h3`) for dialing
/// a `dev_self_signed` loopback server. DEV ONLY — gated behind the
/// explicit `insecure` spec flag.
fn insecure_client_config() -> rustls::ClientConfig {
    let mut config =
        rustls::ClientConfig::builder_with_protocol_versions(&[&rustls::version::TLS13])
            .dangerous()
            .with_custom_certificate_verifier(Arc::new(AcceptAnyCert))
            .with_no_client_auth();
    config.alpn_protocols = vec![b"h3".to_vec()];
    config
}

#[derive(Debug)]
struct AcceptAnyCert;

impl rustls::client::danger::ServerCertVerifier for AcceptAnyCert {
    fn verify_server_cert(
        &self,
        _end_entity: &rustls::pki_types::CertificateDer<'_>,
        _intermediates: &[rustls::pki_types::CertificateDer<'_>],
        _server_name: &rustls::pki_types::ServerName<'_>,
        _ocsp_response: &[u8],
        _now: rustls::pki_types::UnixTime,
    ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::danger::ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &rustls::pki_types::CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn verify_tls13_signature(
        &self,
        _message: &[u8],
        _cert: &rustls::pki_types::CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        rustls::crypto::aws_lc_rs::default_provider()
            .signature_verification_algorithms
            .supported_schemes()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn dial_from_spec(spec: &Value) -> Result<Dial, ProximaError> {
        let config: H3NativeConfig = serde_json::from_value(spec.clone()).expect("config");
        config.into_dial()
    }

    #[test]
    fn factory_name_is_the_spec_key() {
        assert_eq!(H3NativeUpstreamFactory::new().name(), "h3-native");
    }

    #[test]
    fn dial_from_explicit_addr() {
        let spec = serde_json::json!({
            "type": "h3-native", "addr": "127.0.0.1:4433", "server_name": "localhost"
        });
        let dial = dial_from_spec(&spec).expect("dial");
        assert_eq!(dial.addr, "127.0.0.1:4433".parse().unwrap());
        assert_eq!(dial.server_name, "localhost");
        assert!(!dial.insecure);
    }

    #[test]
    fn dial_from_url_ip_literal() {
        let spec = serde_json::json!({ "type": "h3-native", "url": "https://10.0.0.5:8443", "insecure": true });
        let dial = dial_from_spec(&spec).expect("dial");
        assert_eq!(dial.addr, "10.0.0.5:8443".parse().unwrap());
        assert_eq!(dial.server_name, "10.0.0.5");
        assert!(dial.insecure);
    }

    #[test]
    fn dial_url_hostname_errs_until_resolution_lands() {
        let spec = serde_json::json!({ "type": "h3-native", "url": "https://example.com:443" });
        let err = dial_from_spec(&spec).unwrap_err();
        assert!(format!("{err}").contains("not an IP literal"));
    }

    #[test]
    fn dial_carries_timeout_ms_config_surface() {
        let spec = serde_json::json!({
            "type": "h3-native", "addr": "127.0.0.1:4433", "timeout_ms": 1500
        });
        let dial = dial_from_spec(&spec).expect("dial");
        assert_eq!(dial.timeout_ms, Some(1500));
        // absent -> None (upstream keeps its default)
        let bare = serde_json::json!({ "type": "h3-native", "addr": "127.0.0.1:4433" });
        assert_eq!(dial_from_spec(&bare).expect("dial").timeout_ms, None);
    }

    #[test]
    fn dial_missing_target_errs() {
        let spec = serde_json::json!({ "type": "h3-native" });
        assert!(dial_from_spec(&spec).is_err());
    }

    // principle-4 parity: the fluent builder and the config value must lower to
    // identical Dial state (addr, server_name, insecure, timeout).
    #[test]
    fn parity_fluent_builder_and_config_value_match() {
        let from_value: H3NativeConfig = serde_json::from_value(serde_json::json!({
            "addr": "127.0.0.1:4433",
            "server_name": "localhost",
            "insecure": true,
            "timeout_ms": 1500,
        }))
        .expect("from_value");
        let from_value = from_value.into_dial().expect("dial value");

        let from_builder = H3NativeConfig::builder()
            .addr("127.0.0.1:4433")
            .server_name("localhost")
            .insecure(true)
            .timeout_ms(1500)
            .build()
            .into_dial()
            .expect("dial builder");

        assert_eq!(from_value.addr, from_builder.addr);
        assert_eq!(from_value.server_name, from_builder.server_name);
        assert_eq!(from_value.insecure, from_builder.insecure);
        assert_eq!(from_value.timeout_ms, from_builder.timeout_ms);
    }
}
