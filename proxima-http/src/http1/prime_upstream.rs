//! Prime-native `PipeFactory` for the `"http"` spec key — a drop-in
//! replacement for the hyper-backed `HttpPipeFactory` (only present
//! under the `http1` feature) that composes the prime stack (no
//! hyper, no tokio in the request path):
//!
//! - [`PrimeTcpUpstream`] dials the peer on the prime reactor,
//! - [`TlsStreamUpstream`] wraps it for `https`,
//! - [`H1ClientUpstream`] speaks HTTP/1.1 over the byte stream.
//!
//! The spec contract is identical to the hyper factory: the umbrella's
//! `canonical_http` folds `{"http": "https://host", ...}` into an object
//! carrying a `url` string (plus `name` / `timeout` / `method` /
//! `headers`). This factory reads `url` + `name`, resolves the host to a
//! `SocketAddr`, and builds one keep-alive upstream — so swapping it for
//! the hyper factory at registry-build time needs no spec change.

use std::future::Future;
use std::pin::Pin;

use serde_json::Value;
use tracing::debug;
use url::Url;

use proxima_core::ProximaError;
use proxima_net::prime::{ConnectTunneledUpstream, PrimeTcpUpstream};
use proxima_primitives::pipe::handler::{PipeHandle, into_handle};
use proxima_primitives::pipe::pipe_factory::PipeFactory;
use proxima_tls::TlsStreamUpstream;

use crate::http1::client::H1ClientUpstream;
use crate::http1::http_config::HttpConfig;

/// A [`PipeFactory`] for the `"http"` key that builds the prime HTTP/1.1
/// upstream instead of the hyper one. Registered for `"http"` behind the
/// umbrella's `http-prime` feature; the hyper factory is the default.
#[derive(Debug, Default)]
pub struct PrimeHttpPipeFactory;

impl PrimeHttpPipeFactory {
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl PipeFactory for PrimeHttpPipeFactory {
    fn name(&self) -> &str {
        "http"
    }

    fn build(
        &self,
        spec: &Value,
        _inner: Option<PipeHandle>,
    ) -> Pin<Box<dyn Future<Output = Result<PipeHandle, ProximaError>> + Send + '_>> {
        let spec = spec.clone();
        Box::pin(async move {
            // optional egress proxy: when `{"http": "...", "proxy": "http://host:port"}`
            // the connection tunnels to the origin through an HTTP CONNECT proxy.
            // `proxy` is read separately because it is not part of the upstream
            // config the h1 client applies per-request.
            let proxy = spec
                .get("proxy")
                .and_then(Value::as_str)
                .map(str::to_string);
            let config: HttpConfig = serde_json::from_value(spec)
                .map_err(|err| ProximaError::Config(format!("http config: {err}")))?;
            // mirror the hyper factory: the same HttpUpstreamConfig
            // (timeout / method / header forward + inject) is lowered off
            // the spec and applied per-request by the h1 client.
            let runtime = config.into_runtime_config()?;
            build_prime_upstream(
                &config.url,
                &config.name,
                runtime,
                config.response,
                proxy.as_deref(),
            )
        })
    }
}

/// Parse the base url and stack the prime upstream for it. The base url
/// gives scheme/host/port; the per-request path + query ride through the
/// `H1ClientUpstream` from `Request`, so only the authority matters here.
///
/// DNS is NOT resolved here — the host + port are handed to a lazy-resolve
/// `PrimeTcpUpstream` that calls `getaddrinfo` at CONNECT time. This keeps
/// `build()` side-effect-free so an upstream can be configured for a host
/// that is not (yet) reachable, matching the hyper factory which also
/// defers resolution to request time.
fn build_prime_upstream(
    url: &str,
    label: &str,
    config: crate::http1::http_config::HttpUpstreamConfig,
    response: crate::http1::response_config::ResponseHandlingConfig,
    proxy: Option<&str>,
) -> Result<PipeHandle, ProximaError> {
    let parsed =
        Url::parse(url).map_err(|err| ProximaError::Config(format!("parse url `{url}`: {err}")))?;
    let secure = match parsed.scheme() {
        "https" => true,
        "http" => false,
        other => {
            return Err(ProximaError::Config(format!(
                "unsupported url scheme `{other}` (only http / https)"
            )));
        }
    };
    let host = parsed
        .host_str()
        .ok_or_else(|| ProximaError::Config(format!("url `{url}` has no host")))?
        .to_string();
    let port = parsed
        .port_or_known_default()
        .ok_or_else(|| ProximaError::Config(format!("url `{url}` has no port")))?;
    let host_header = authority(&host, port, secure);
    // four combinations of {direct, via-proxy} x {https, http}: the dial layer
    // is either a direct prime tcp upstream or a CONNECT tunnel through the
    // proxy, then TLS wraps it for https. each `into_handle` type-erases a
    // distinct concrete `H1ClientUpstream<..>`, so the branches can't collapse.
    let proxy_dial = proxy.map(parse_proxy).transpose()?;
    let handle: PipeHandle = match (proxy_dial, secure) {
        (Some((proxy_host, proxy_port)), true) => {
            let tunnel = ConnectTunneledUpstream::new(
                PrimeTcpUpstream::with_host(proxy_host, proxy_port),
                host.clone(),
                port,
            );
            let tls = TlsStreamUpstream::with_webpki_roots(tunnel, host.clone())?;
            into_handle(
                H1ClientUpstream::new(tls, host_header, label.to_string())
                    .with_config(config)
                    .with_response_config(response),
            )
        }
        (Some((proxy_host, proxy_port)), false) => {
            let tunnel = ConnectTunneledUpstream::new(
                PrimeTcpUpstream::with_host(proxy_host, proxy_port),
                host.clone(),
                port,
            );
            into_handle(
                H1ClientUpstream::new(tunnel, host_header, label.to_string())
                    .with_config(config)
                    .with_response_config(response),
            )
        }
        (None, true) => {
            let tcp = PrimeTcpUpstream::with_host(host.clone(), port);
            let tls = TlsStreamUpstream::with_webpki_roots(tcp, host.clone())?;
            into_handle(
                H1ClientUpstream::new(tls, host_header, label.to_string())
                    .with_config(config)
                    .with_response_config(response),
            )
        }
        (None, false) => {
            let tcp = PrimeTcpUpstream::with_host(host.clone(), port);
            into_handle(
                H1ClientUpstream::new(tcp, host_header, label.to_string())
                    .with_config(config)
                    .with_response_config(response),
            )
        }
    };
    debug!(host = %host, port, secure, label, proxied = proxy.is_some(), "prime http upstream built");
    Ok(handle)
}

/// Parse a proxy url (`http://host:port`) into the host + port the tunnel
/// dials. Only the authority matters; the CONNECT target is the origin.
fn parse_proxy(proxy_url: &str) -> Result<(String, u16), ProximaError> {
    let parsed = Url::parse(proxy_url)
        .map_err(|err| ProximaError::Config(format!("parse proxy url `{proxy_url}`: {err}")))?;
    let host = parsed
        .host_str()
        .ok_or_else(|| ProximaError::Config(format!("proxy url `{proxy_url}` has no host")))?
        .to_string();
    let port = parsed
        .port_or_known_default()
        .ok_or_else(|| ProximaError::Config(format!("proxy url `{proxy_url}` has no port")))?;
    Ok((host, port))
}

/// Build the `Host` header value: bare host on the scheme's default
/// port, `host:port` otherwise (matching what `requests` / curl send).
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

    #[test]
    fn factory_requires_url_field() {
        let factory = PrimeHttpPipeFactory::new();
        let outcome = futures::executor::block_on(factory.build(&serde_json::json!({}), None));
        assert!(matches!(outcome, Err(ProximaError::Config(_))));
    }

    #[test]
    fn factory_rejects_unsupported_scheme() {
        let factory = PrimeHttpPipeFactory::new();
        let outcome = futures::executor::block_on(
            factory.build(&serde_json::json!({"url": "ftp://example.test"}), None),
        );
        assert!(matches!(outcome, Err(ProximaError::Config(_))));
    }

    #[test]
    fn authority_omits_default_port() {
        assert_eq!(authority("example.test", 80, false), "example.test");
        assert_eq!(authority("example.test", 443, true), "example.test");
        assert_eq!(authority("example.test", 8080, false), "example.test:8080");
    }

    #[test]
    fn factory_name_is_http() {
        assert_eq!(PrimeHttpPipeFactory::new().name(), "http");
    }

    /// The egress-proxy branch: a spec carrying a `proxy` builds the
    /// CONNECT-tunnel stack (DNS is deferred, so this builds without the
    /// network) for both https and http origins. The CONNECT wire protocol
    /// itself is covered by `ConnectTunneledUpstream`'s unit tests; the live
    /// tunnel is proven e2e through the daemon hook path.
    #[test]
    fn build_via_proxy_succeeds_for_https_and_http() {
        let config = crate::http1::http_config::HttpUpstreamConfig::default();
        let response = crate::http1::response_config::ResponseHandlingConfig::default();
        let https = build_prime_upstream(
            "https://api.example.test",
            "proxied-https",
            config.clone(),
            response,
            Some("http://127.0.0.1:8080"),
        );
        assert!(https.is_ok(), "https-via-proxy builds");
        let http = build_prime_upstream(
            "http://api.example.test",
            "proxied-http",
            config,
            response,
            Some("http://127.0.0.1:8080"),
        );
        assert!(http.is_ok(), "http-via-proxy builds");
    }

    #[test]
    fn build_rejects_malformed_proxy_url() {
        let config = crate::http1::http_config::HttpUpstreamConfig::default();
        let response = crate::http1::response_config::ResponseHandlingConfig::default();
        let outcome = build_prime_upstream(
            "https://api.example.test",
            "p",
            config,
            response,
            Some("not a url"),
        );
        assert!(matches!(outcome, Err(ProximaError::Config(_))));
    }

    #[test]
    fn factory_forwards_proxy_spec_key() {
        let factory = PrimeHttpPipeFactory::new();
        let outcome = futures::executor::block_on(factory.build(
            &serde_json::json!({"url": "https://api.example.test", "proxy": "http://127.0.0.1:8080"}),
            None,
        ));
        assert!(outcome.is_ok(), "factory builds with a proxy key");
    }

    /// DNS deferral: building an upstream for a host that does not resolve
    /// must succeed — resolution happens at connect time, not build time.
    /// This is what lets the umbrella's fake-host load tests
    /// (`http://example.test`) build the prime factory without touching
    /// the network.
    #[test]
    fn build_succeeds_for_unresolvable_fake_host() {
        let factory = PrimeHttpPipeFactory::new();
        let outcome = futures::executor::block_on(
            factory.build(&serde_json::json!({"url": "http://example.test"}), None),
        );
        assert!(
            outcome.is_ok(),
            "build must not resolve DNS; got error: {:?}",
            outcome.err()
        );
    }
}
