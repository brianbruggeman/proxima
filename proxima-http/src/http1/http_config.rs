//! Wire + runtime config types for the `http` upstream, shared by both
//! the hyper-backed [`super::upstream`] and the tokio-free prime client
//! ([`super::client`] / [`super::prime_upstream`]).
//!
//! Pure data — no hyper, no tokio — so this module compiles under either
//! `http1` (the hyper stack) or `http1-stream-client` (the prime stack)
//! alone. [`HttpConfig::into_upstream`], which materializes the
//! hyper-backed [`super::upstream::HttpUpstream`], stays defined in
//! `upstream.rs` (a second `impl HttpConfig` block) since it needs the
//! hyper-only types.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use bon::Builder;
use conflaguration::{Settings, Validate, ValidationMessage};
use serde::{Deserialize, Serialize};

use crate::http1::response_config::ResponseHandlingConfig;
use proxima_core::ProximaError;

/// The `headers.forward` allow-list: either an explicit array of header names
/// or the string `"all"` (forward everything). Untagged so a JSON array
/// deserialises to [`HeaderForward::List`] and the string `"all"` to
/// [`HeaderForward::All`] — matching the historical hand-parser.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum HeaderForward {
    List(Vec<String>),
    All(String),
}

/// Header injection / forwarding config for an `http` upstream.
#[derive(Debug, Clone, Default, Builder, Serialize, Deserialize)]
#[builder(derive(Clone, Debug))]
pub struct HttpHeadersConfig {
    /// Request-header allow-list: an array of names, or `"all"`. Absent =
    /// forward every header.
    #[serde(default)]
    pub forward: Option<HeaderForward>,

    /// Headers injected onto each outbound request (values support templating).
    #[serde(default)]
    #[builder(default)]
    pub request: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Default)]
pub struct HttpUpstreamConfig {
    pub method_override: Option<String>,
    pub timeout: Option<Duration>,
    /// pre-lowercased allow-list; linear scan beats a HashSet at the
    /// typical < 20 headers. `None` = forward every header.
    pub forward_request_headers: Option<Arc<Vec<bytes::Bytes>>>,
    pub injected_request_headers: BTreeMap<String, String>,
}

/// Typed config surface for the `http` upstream — a proxy to a base URL with
/// optional method override, timeout, and header forwarding/injection. Mirrors
/// the runtime [`HttpUpstreamConfig`] (whose `timeout` is a `Duration` and
/// `forward_request_headers` a pre-lowercased `Arc<Vec<Bytes>>`).
#[derive(Debug, Clone, Builder, Deserialize, Serialize, Settings)]
#[settings(prefix = "PROXIMA_HTTP")]
#[builder(derive(Clone, Debug), on(String, into))]
pub struct HttpConfig {
    /// Base URL the upstream proxies to.
    pub url: String,

    /// Pipe / backend label.
    #[setting(default = "http")]
    #[serde(default = "default_label")]
    #[builder(default = default_label())]
    pub name: String,

    /// Override the request method, e.g. force `POST`.
    #[setting(default)]
    #[serde(default)]
    pub method: Option<String>,

    /// Per-request timeout, e.g. `30s` / `500ms`. `None` waits indefinitely.
    #[setting(default)]
    #[serde(default)]
    pub timeout: Option<String>,

    /// Header forwarding / injection rules.
    #[setting(skip)]
    #[serde(default)]
    #[builder(default)]
    pub headers: HttpHeadersConfig,

    /// Response-handling composition: collect/drain body, all/framing headers.
    /// A load generator sets `drain`+`framing` to consume the response to the
    /// keep-alive boundary without materializing it (the `Discard` preset).
    /// Only honored by the prime factory; the hyper path always collects.
    #[setting(skip)]
    #[serde(default)]
    #[builder(default)]
    pub response: ResponseHandlingConfig,
}

fn default_label() -> String {
    "http".to_string()
}

impl Validate for HttpConfig {
    fn validate(&self) -> conflaguration::Result<()> {
        let mut errors = Vec::new();
        if self.url.is_empty() {
            errors.push(ValidationMessage::new("url", "must not be empty"));
        }
        if let Some(timeout) = &self.timeout
            && parse_duration(timeout).is_err()
        {
            errors.push(ValidationMessage::new(
                "timeout",
                "must be a duration string like '30s' or '500ms'",
            ));
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(conflaguration::Error::Validation { errors })
        }
    }
}

impl HttpConfig {
    /// Lower the wire config to the runtime [`HttpUpstreamConfig`].
    pub fn into_runtime_config(&self) -> Result<HttpUpstreamConfig, ProximaError> {
        self.validate()
            .map_err(|err| ProximaError::Config(format!("{err}")))?;
        let timeout = match &self.timeout {
            Some(raw) => Some(parse_duration(raw)?),
            None => None,
        };
        let forward_request_headers = match &self.headers.forward {
            Some(HeaderForward::List(names)) => Some(Arc::new(
                names
                    .iter()
                    .map(|name| bytes::Bytes::from(name.to_ascii_lowercase()))
                    .collect::<Vec<bytes::Bytes>>(),
            )),
            Some(HeaderForward::All(text)) if text == "all" => None,
            Some(HeaderForward::All(other)) => {
                return Err(ProximaError::Config(format!(
                    "headers.forward must be an array of header names or the string 'all', got '{other}'"
                )));
            }
            None => None,
        };
        Ok(HttpUpstreamConfig {
            method_override: self.method.clone(),
            timeout,
            forward_request_headers,
            injected_request_headers: self.headers.request.clone(),
        })
    }
}

/// Parse a duration string like `"30s"`, `"500ms"`, `"5min"`. Duplicated
/// from `upstreams::kv_cache` so this crate stays leaf — the umbrella's
/// version remains the canonical one for non-HTTP upstream configs.
fn parse_duration(raw: &str) -> Result<Duration, ProximaError> {
    let trimmed = raw.trim();
    let (digits, suffix) = trimmed.split_at(
        trimmed
            .find(|character: char| character.is_alphabetic())
            .unwrap_or(trimmed.len()),
    );
    let amount: u64 = digits
        .parse()
        .map_err(|_| ProximaError::Config(format!("invalid duration value '{raw}'")))?;
    let multiplier_seconds = match suffix {
        "" | "s" | "sec" | "secs" => 1u64,
        "ms" => return Ok(Duration::from_millis(amount)),
        "us" => return Ok(Duration::from_micros(amount)),
        "m" | "min" | "mins" => 60,
        "h" | "hr" | "hrs" => 60 * 60,
        "d" | "day" | "days" => 60 * 60 * 24,
        other => {
            return Err(ProximaError::Config(format!(
                "unknown duration suffix '{other}'"
            )));
        }
    };
    Ok(Duration::from_secs(amount * multiplier_seconds))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn config_reads_timeout() {
        let config: HttpConfig =
            serde_json::from_value(serde_json::json!({"url": "http://x", "timeout": "5s"}))
                .expect("from_value");
        let runtime = config.into_runtime_config().expect("config");
        assert_eq!(runtime.timeout, Some(Duration::from_secs(5)));
    }

    #[test]
    fn config_reads_injected_headers() {
        let config: HttpConfig = serde_json::from_value(serde_json::json!({
            "url": "http://x",
            "headers": {"request": {"x-trace-id": "{{request.id}}"}},
        }))
        .expect("from_value");
        let runtime = config.into_runtime_config().expect("config");
        assert_eq!(
            runtime.injected_request_headers.get("x-trace-id"),
            Some(&"{{request.id}}".into()),
        );
    }
}
