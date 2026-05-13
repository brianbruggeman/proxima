//! `PipeFactory` for the `grpc` spec key — a `proxima::Client` transport that
//! speaks gRPC over HTTP/2.
//!
//! Composes the native h2 client ([`H2ClientUpstream`](crate::h2::H2ClientUpstream))
//! over the prime TCP transport ([`PrimeTcpUpstream`](crate::PrimeTcpUpstream)),
//! optionally TLS-wrapped with ALPN `h2` for `grpcs`. Registered like the prime
//! `http` factory, so `Client::builder().grpc(url)` resolves through it and the
//! recorder terminal can POST grpc-framed OTLP bytes over h2 without naming a
//! transport type. The protocol axis (gRPC = grpc-framed OTLP protobuf + the
//! `/svc/Method` path + `content-type: application/grpc`) is the caller's request
//! shape; this factory is purely the transport.

use std::future::Future;
use std::pin::Pin;

use bon::Builder;
use conflaguration::{Settings, Validate, ValidationMessage};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use url::Url;

use proxima_primitives::pipe::handler::{PipeHandle, into_handle};
use proxima_primitives::pipe::pipe_factory::PipeFactory;

use crate::PrimeTcpUpstream;
use crate::ProximaError;
use crate::h2::H2ClientUpstream;

/// Typed config for the `grpc` key. The spec contract mirrors `http`: the
/// umbrella's `canonical_http` folds `{"grpc": "http://host:4317", ...}` into an
/// object carrying a `url` string (plus `name`); this reads `url` + `name`.
#[derive(Debug, Clone, Builder, Deserialize, Serialize, Settings)]
#[settings(prefix = "PROXIMA_GRPC")]
#[builder(derive(Clone, Debug), on(String, into))]
pub struct GrpcConfig {
    /// gRPC upstream url (`http` / `https` / `grpc` / `grpcs`). Required.
    #[setting(default)]
    #[serde(default)]
    #[builder(default)]
    pub url: String,

    /// Pipe label.
    #[setting(default = "grpc")]
    #[serde(default = "default_label")]
    #[builder(default = default_label())]
    pub name: String,
}

fn default_label() -> String {
    "grpc".to_string()
}

impl Validate for GrpcConfig {
    fn validate(&self) -> conflaguration::Result<()> {
        let mut errors = Vec::new();
        if self.url.is_empty() {
            errors.push(ValidationMessage::new(
                "url",
                "grpc upstream requires `url`",
            ));
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(conflaguration::Error::Validation { errors })
        }
    }
}

impl GrpcConfig {
    /// Materialise into the type-erased h2-over-tcp(/tls) transport pipe handle.
    pub fn from_config(self) -> Result<PipeHandle, ProximaError> {
        self.validate()
            .map_err(|err| ProximaError::Config(format!("{err}")))?;
        build_grpc_h2_upstream(&self.url, &self.name)
    }
}

/// A [`PipeFactory`] for the `grpc` key.
#[derive(Debug, Default)]
pub struct GrpcH2PipeFactory;

impl GrpcH2PipeFactory {
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl PipeFactory for GrpcH2PipeFactory {
    fn name(&self) -> &str {
        "grpc"
    }

    fn build(
        &self,
        spec: &Value,
        _inner: Option<PipeHandle>,
    ) -> Pin<Box<dyn Future<Output = Result<PipeHandle, ProximaError>> + Send + '_>> {
        let spec = spec.clone();
        Box::pin(async move {
            let config: GrpcConfig = serde_json::from_value(spec)
                .map_err(|err| ProximaError::Config(format!("grpc config: {err}")))?;
            config.from_config()
        })
    }
}

/// Stack the h2 client over the prime transport for `url`. DNS is resolved lazily
/// at connect time (side-effect-free build, matching the prime `http` factory).
fn build_grpc_h2_upstream(url: &str, label: &str) -> Result<PipeHandle, ProximaError> {
    let parsed = Url::parse(url)
        .map_err(|err| ProximaError::Config(format!("parse grpc url `{url}`: {err}")))?;
    let secure = match parsed.scheme() {
        "https" | "grpcs" => true,
        "http" | "grpc" => false,
        other => {
            return Err(ProximaError::Config(format!(
                "unsupported grpc url scheme `{other}` (http / https / grpc / grpcs)"
            )));
        }
    };
    let host = parsed
        .host_str()
        .ok_or_else(|| ProximaError::Config(format!("grpc url `{url}` has no host")))?
        .to_string();
    let port = parsed
        .port_or_known_default()
        .ok_or_else(|| ProximaError::Config(format!("grpc url `{url}` has no port")))?;
    let authority = authority(&host, port, secure);
    if secure {
        secure_grpc_upstream(host, port, authority, label)
    } else {
        let tcp = PrimeTcpUpstream::with_host(host, port);
        Ok(into_handle(H2ClientUpstream::new(
            tcp,
            authority,
            false,
            label.to_string(),
        )))
    }
}

#[cfg(feature = "tls")]
fn secure_grpc_upstream(
    host: String,
    port: u16,
    authority: String,
    label: &str,
) -> Result<PipeHandle, ProximaError> {
    let tcp = PrimeTcpUpstream::with_host(host.clone(), port);
    // gRPC-over-TLS negotiates ALPN `h2` (not the h1 default `with_webpki_roots`
    // would pick) so the server speaks HTTP/2 on the same socket.
    let tls_config = crate::tls::TlsClientConfig::builder()
        .server_name(host)
        .alpn_protocols(vec!["h2".to_string()])
        .build();
    let tls = crate::tls::TlsStreamUpstream::from_config(tcp, &tls_config)?;
    Ok(into_handle(H2ClientUpstream::new(
        tls,
        authority,
        true,
        label.to_string(),
    )))
}

#[cfg(not(feature = "tls"))]
fn secure_grpc_upstream(
    _host: String,
    _port: u16,
    _authority: String,
    _label: &str,
) -> Result<PipeHandle, ProximaError> {
    Err(ProximaError::Config(
        "grpcs (gRPC over TLS) requires the `tls` feature".into(),
    ))
}

/// `:authority` value: bare host on the scheme's default port, `host:port`
/// otherwise.
fn authority(host: &str, port: u16, secure: bool) -> String {
    let default_port = if secure { 443 } else { 80 };
    if port == default_port {
        host.to_string()
    } else {
        format!("{host}:{port}")
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    // principle-4 parity: the fluent builder and the config value must lower to
    // identical GrpcConfig state (the pipe label), and both build successfully.
    #[test]
    fn parity_fluent_builder_and_config_value_match() {
        let from_value: GrpcConfig = serde_json::from_value(serde_json::json!({
            "url": "http://collector:4317",
            "name": "otlp",
        }))
        .expect("from_value");

        let from_builder = GrpcConfig::builder()
            .url("http://collector:4317")
            .name("otlp")
            .build();

        assert_eq!(from_value.name, from_builder.name);
        assert_eq!(from_value.name, "otlp");

        from_value.clone().from_config().expect("from_config value");
        from_builder.from_config().expect("from_config builder");
    }

    #[test]
    fn missing_url_is_a_config_error() {
        let config = GrpcConfig::builder().build();
        assert!(matches!(config.from_config(), Err(ProximaError::Config(_))));
    }
}
