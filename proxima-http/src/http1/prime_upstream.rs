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

use proxima_telemetry::debug;
use serde_json::Value;
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
            // `.tls()` (`ClientSecurityExt`) — an ASSERTION the wire must be
            // TLS, read separately for the same reason `proxy` is: it names
            // a validation, not an upstream-config field. See
            // `build_prime_upstream`'s `transport == "tls"` + `http://`
            // scheme rejection.
            let transport = spec
                .get("transport")
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
                transport.as_deref(),
            )
        })
    }
}

/// Parse the base url and stack the prime upstream for it. The base url
/// gives scheme/host/port, AND any path (`/v1`, `/llm/openai/v1`, ...),
/// which is stacked onto `config.base_path` and prepended to every
/// request's path by the `H1ClientUpstream`'s `apply_config`. That path is
/// what used to be dropped silently: only the authority was ever read off
/// the parsed url, and the path component was discarded.
///
/// DNS is NOT resolved here — the host + port are handed to a lazy-resolve
/// `PrimeTcpUpstream` that calls `getaddrinfo` at CONNECT time. This keeps
/// `build()` side-effect-free so an upstream can be configured for a host
/// that is not (yet) reachable, matching the hyper factory which also
/// defers resolution to request time.
fn build_prime_upstream(
    url: &str,
    label: &str,
    mut config: crate::http1::http_config::HttpUpstreamConfig,
    response: crate::http1::response_config::ResponseHandlingConfig,
    proxy: Option<&str>,
    transport: Option<&str>,
) -> Result<PipeHandle, ProximaError> {
    // a bare `host:port` with no `http://`/`https://` scheme (e.g. the
    // common Ollama-style `"127.0.0.1:11434"`) is a config error, not a
    // silent normalization to `http://` — `url::Url::parse` already
    // refuses it ("relative URL without a base"), so this is the existing
    // behavior, asserted by `build_rejects_bare_host_without_scheme` rather
    // than left as an accident of the error message.
    let parsed =
        Url::parse(url).map_err(|err| ProximaError::Config(format!("parse url `{url}`: {err}")))?;
    config.base_path = base_path_prefix(parsed.path());
    let secure = match parsed.scheme() {
        "https" => true,
        "http" => false,
        other => {
            return Err(ProximaError::Config(format!(
                "unsupported url scheme `{other}` (only http / https)"
            )));
        }
    };
    // `.tls()` asserts the wire MUST be TLS — an `http://` dial url paired
    // with it is a config error, never a silent plaintext downgrade (the
    // bug `Transport::Tls` used to hide entirely: `canonical_http` never
    // forwarded `transport`, so this check was unreachable).
    if transport == Some("tls") && !secure {
        return Err(ProximaError::Config(format!(
            "url `{url}` is http:// but .tls() asserts the wire must be TLS; use an https:// url"
        )));
    }
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
    debug!(host = %host, port, secure, label = %label, proxied = proxy.is_some(), "prime http upstream built");
    Ok(handle)
}

/// Normalize a parsed url's path into the prefix `apply_config` prepends to
/// every request path: strip the trailing slash, so `"/v1"` and `"/v1/"`
/// compose onto a request path with exactly one slash between them, and a
/// bare host's path (`url::Url` always parses that as `"/"`) normalizes to
/// `""` — no prefix, byte-identical to a base url with no path at all.
fn base_path_prefix(path: &str) -> String {
    path.trim_end_matches('/').to_string()
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
    use rstest::rstest;

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
            None,
        );
        assert!(https.is_ok(), "https-via-proxy builds");
        let http = build_prime_upstream(
            "http://api.example.test",
            "proxied-http",
            config,
            response,
            Some("http://127.0.0.1:8080"),
            None,
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
            None,
        );
        assert!(matches!(outcome, Err(ProximaError::Config(_))));
    }

    /// The bug-fix headline: `.tls()` (`transport == "tls"`) paired with an
    /// `http://` dial url is a config error, never a silent plaintext
    /// downgrade — the exact composition `Transport::Tls` used to hide
    /// entirely (dead: `canonical_http` never forwarded `transport`, so this
    /// check was unreachable before the fix).
    #[test]
    fn tls_transport_with_http_scheme_is_a_config_error() {
        let config = crate::http1::http_config::HttpUpstreamConfig::default();
        let response = crate::http1::response_config::ResponseHandlingConfig::default();
        let outcome = build_prime_upstream(
            "http://api.example.test",
            "p",
            config,
            response,
            None,
            Some("tls"),
        );
        let err = match outcome {
            Ok(_) => panic!(".tls() + http:// must not silently build a plaintext upstream"),
            Err(err) => err,
        };
        assert!(format!("{err}").contains("tls"), "got: {err}");
    }

    /// The matching happy path: `.tls()` with an `https://` url builds fine.
    #[test]
    fn tls_transport_with_https_scheme_builds() {
        let config = crate::http1::http_config::HttpUpstreamConfig::default();
        let response = crate::http1::response_config::ResponseHandlingConfig::default();
        let outcome = build_prime_upstream(
            "https://api.example.test",
            "p",
            config,
            response,
            None,
            Some("tls"),
        );
        assert!(
            outcome.is_ok(),
            "got: {:?}",
            outcome.err().map(|err| err.to_string())
        );
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

    #[rstest]
    #[case::no_path("/", "")]
    #[case::single_segment("/v1", "/v1")]
    #[case::trailing_slash("/v1/", "/v1")]
    #[case::nested("/llm/openai/v1", "/llm/openai/v1")]
    #[case::nested_trailing_slash("/llm/openai/v1/", "/llm/openai/v1")]
    fn base_path_prefix_normalizes_trailing_slash(#[case] path: &str, #[case] expected: &str) {
        assert_eq!(base_path_prefix(path), expected);
    }

    /// A bare `host:port` with no `http://`/`https://` scheme (the common
    /// Ollama-style `"127.0.0.1:11434"`) is a config error: `url::Url`
    /// parses it as a relative-url-without-a-base failure, and this
    /// factory never normalizes it to `http://` on the caller's behalf.
    /// Asserted here so the decision is a test, not an accident of the
    /// error message.
    #[test]
    fn build_rejects_bare_host_without_scheme() {
        let factory = PrimeHttpPipeFactory::new();
        let outcome = futures::executor::block_on(
            factory.build(&serde_json::json!({"url": "127.0.0.1:11434"}), None),
        );
        assert!(matches!(outcome, Err(ProximaError::Config(_))));
    }

    /// The nested-path headline case end to end through the factory: a
    /// base url with a multi-segment path builds without error, proving
    /// `build_prime_upstream` accepts (rather than rejects) a path
    /// component instead of silently discarding it. The per-request byte
    /// assembly is proven in `client.rs`'s
    /// `prefixed_path_produces_the_wire_target` — `PipeHandle` is opaque
    /// past this point, so this test stops at "it builds".
    #[test]
    fn build_succeeds_for_nested_base_path() {
        let factory = PrimeHttpPipeFactory::new();
        let outcome = futures::executor::block_on(factory.build(
            &serde_json::json!({"url": "http://gw.example.test/llm/openai/v1"}),
            None,
        ));
        assert!(outcome.is_ok(), "got: {:?}", outcome.err());
    }
}
