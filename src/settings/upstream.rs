//! Typed upstream configs. Each is a `bon::Builder`-derived struct
//! whose `Into<Spec>` impl produces the `{ "type": "...", ...fields }`
//! shape the existing factory registry already dispatches on (see
//! `load.rs` — the `type` discriminator branch at line 396 was added
//! for exactly this).
//!
//! Today: `HttpUpstream`. Others (Kv, Process, Replay, etc.) layer
//! in as needed.

use std::time::Duration;

use bon::Builder;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::load::Spec;

/// HTTP upstream — proxies requests to an external URL. Same fields
/// the existing `HttpPipeFactory` parses; the typed shape is the
/// fluent surface, serde derive is the TOML round-trip.
///
/// ```ignore
/// let backend = HttpUpstream::builder()
///     .url("https://backend.internal:8443")
///     .timeout(Duration::from_secs(5))
///     .build();
/// app.pipe("backend", backend).await?;
/// ```
#[derive(Debug, Clone, Builder, Deserialize, Serialize)]
#[builder(derive(Clone, Debug), on(String, into))]
pub struct HttpUpstream {
    /// Base URL the upstream forwards to. Required.
    pub url: String,

    /// Optional per-request timeout. Serializes as a duration string
    /// (`"5s"`, `"100ms"`) to round-trip through TOML.
    #[serde(default, with = "duration_serde")]
    pub timeout: Option<Duration>,

    /// Override the upstream HTTP method (rare; default = forward
    /// the inbound method).
    #[serde(default)]
    pub method: Option<String>,
}

impl HttpUpstream {
    /// Shorthand for the common case — just a URL with default
    /// timeout (none) and method (forward).
    pub fn url(url: impl Into<String>) -> Self {
        Self::builder().url(url).build()
    }
}

impl From<HttpUpstream> for Spec {
    fn from(value: HttpUpstream) -> Self {
        let mut map = Map::new();
        map.insert("type".into(), Value::String("http".into()));
        map.insert("url".into(), Value::String(value.url));
        if let Some(timeout) = value.timeout {
            map.insert("timeout".into(), Value::String(humanize_duration(timeout)));
        }
        if let Some(method) = value.method {
            map.insert("method".into(), Value::String(method));
        }
        Spec::Inline(Value::Object(map))
    }
}

fn humanize_duration(duration: Duration) -> String {
    let total_ms = duration.as_millis();
    if total_ms.is_multiple_of(1000) {
        format!("{}s", total_ms / 1000)
    } else {
        format!("{total_ms}ms")
    }
}

/// Serde adapter for `Option<Duration>` — emits `"5s"` / `"100ms"`
/// strings matching what the existing `HttpPipeFactory` parser
/// accepts (via `parse_duration`).
mod duration_serde {
    use std::time::Duration;

    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(
        value: &Option<Duration>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        match value {
            None => serializer.serialize_none(),
            Some(duration) => serializer.serialize_str(&super::humanize_duration(*duration)),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Option<Duration>, D::Error> {
        let raw = Option::<String>::deserialize(deserializer)?;
        match raw {
            None => Ok(None),
            Some(text) => parse_duration(&text)
                .map(Some)
                .map_err(serde::de::Error::custom),
        }
    }

    fn parse_duration(raw: &str) -> Result<Duration, String> {
        // matches the existing parser shape: digits followed by ms / s / m / h.
        let trimmed = raw.trim();
        for (suffix, multiplier) in [("ms", 1u64), ("s", 1_000), ("m", 60_000), ("h", 3_600_000)] {
            if let Some(prefix) = trimmed.strip_suffix(suffix)
                && let Ok(value) = prefix.trim().parse::<u64>()
            {
                return Ok(Duration::from_millis(value * multiplier));
            }
        }
        Err(format!("could not parse duration: {raw}"))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn http_upstream_serializes_with_type_tag() {
        let upstream = HttpUpstream::url("https://example.com");
        let spec: Spec = upstream.into();
        let Spec::Inline(value) = spec else {
            panic!("expected inline")
        };
        assert_eq!(value.get("type").and_then(|v| v.as_str()), Some("http"));
        assert_eq!(
            value.get("url").and_then(|v| v.as_str()),
            Some("https://example.com"),
        );
    }

    #[test]
    fn http_upstream_serializes_timeout_as_duration_string() {
        let upstream = HttpUpstream::builder()
            .url("https://example.com")
            .timeout(Duration::from_secs(5))
            .build();
        let spec: Spec = upstream.into();
        let Spec::Inline(value) = spec else {
            panic!("expected inline")
        };
        assert_eq!(value.get("timeout").and_then(|v| v.as_str()), Some("5s"));
    }

    #[test]
    fn http_upstream_round_trips_through_toml() {
        let original = HttpUpstream::builder()
            .url("https://api.example.com")
            .timeout(Duration::from_millis(750))
            .method("POST")
            .build();
        let toml_text = toml::to_string(&original).expect("encode toml");
        let restored: HttpUpstream = toml::from_str(&toml_text).expect("decode toml");
        assert_eq!(restored.url, original.url);
        assert_eq!(restored.timeout, original.timeout);
        assert_eq!(restored.method, original.method);
    }

    #[test]
    fn humanize_emits_seconds_when_whole() {
        assert_eq!(humanize_duration(Duration::from_secs(5)), "5s");
        assert_eq!(humanize_duration(Duration::from_millis(750)), "750ms");
    }
}
