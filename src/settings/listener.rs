//! Typed listener configs. Each is a `bon::Builder`-derived struct
//! that converts via `Into<RunConfig>` so `App::serve(...)` accepts
//! them transparently. TOML round-trip is via the same serde-derived
//! deserialization the registry-entry path uses.
//!
//! Three concrete shapes today:
//!
//! - [`HttpListener`] — plain TCP HTTP/1.1 (with optional h2 prior-
//!   knowledge dispatch on the listener side — that's a property of
//!   `HttpListenProtocol`, not the spec).
//! - [`HttpsListener`] — TCP + TLS termination + ALPN.
//! - [`HttpUdsListener`] — UDS-bound HTTP/1.1, used by the daemon
//!   control plane and any local-only pipe surface.

use std::net::SocketAddr;
use std::path::PathBuf;

use bon::Builder;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::app::RunConfig;

/// Plain TCP HTTP/1.1 listener. Pass to `App::serve(...)`.
///
/// ```ignore
/// let server = app.serve(
///     HttpListener::builder().addr("0.0.0.0:8080".parse().unwrap()).build()
/// ).await?;
/// ```
#[derive(Debug, Clone, Builder, Deserialize, Serialize)]
#[builder(derive(Clone, Debug))]
pub struct HttpListener {
    pub addr: SocketAddr,

    /// Optional max body bytes per request. `None` = unbounded.
    #[serde(default)]
    pub max_body_bytes: Option<usize>,
}

impl HttpListener {
    /// Shorthand: `HttpListener::http("0.0.0.0:8080".parse()?)`. Builds
    /// with defaults (no body limit).
    #[must_use]
    pub fn http(addr: SocketAddr) -> Self {
        Self::builder().addr(addr).build()
    }
}

impl From<HttpListener> for RunConfig {
    fn from(value: HttpListener) -> Self {
        let mut spec = serde_json::Map::new();
        if let Some(max) = value.max_body_bytes {
            spec.insert("max_body_bytes".into(), Value::from(max));
        }
        Self {
            bind: value.addr,
            protocol: "http".into(),
            spec: if spec.is_empty() {
                Value::Null
            } else {
                Value::Object(spec)
            },
        }
    }
}

/// TLS-terminating HTTP listener. Uses tokio-rustls + aws-lc-rs;
/// ALPN advertises h2 + http/1.1 by default. cert + key are PEM
/// paths.
#[cfg(feature = "tls")]
#[derive(Debug, Clone, Builder, Deserialize, Serialize)]
#[builder(derive(Clone, Debug), on(PathBuf, into))]
pub struct HttpsListener {
    pub addr: SocketAddr,
    pub cert: PathBuf,
    pub key: PathBuf,

    #[serde(default)]
    pub max_body_bytes: Option<usize>,
}

#[cfg(feature = "tls")]
impl HttpsListener {
    /// Shorthand: `HttpsListener::https(addr, cert, key)`.
    #[must_use]
    pub fn https(addr: SocketAddr, cert: PathBuf, key: PathBuf) -> Self {
        Self::builder().addr(addr).cert(cert).key(key).build()
    }
}

#[cfg(feature = "tls")]
impl TryFrom<HttpsListener> for RunConfig {
    type Error = proxima_core::ProximaError;

    fn try_from(value: HttpsListener) -> Result<Self, Self::Error> {
        let tls_config = crate::tls::TlsConfig::files(value.cert, value.key)?;
        let mut spec = serde_json::Map::new();
        spec.insert(
            crate::tls::SPEC_KEY.into(),
            crate::tls::config_to_spec_value(&tls_config),
        );
        if let Some(max) = value.max_body_bytes {
            spec.insert("max_body_bytes".into(), Value::from(max));
        }
        Ok(Self {
            bind: value.addr,
            protocol: "http".into(),
            spec: Value::Object(spec),
        })
    }
}

/// UDS-bound HTTP/1.1 listener. `path` is the socket file (will be
/// removed on bind if stale). `mode` is the optional octal permission
/// applied after bind (`0o600` is typical for local-only daemons).
///
/// `RunConfig::bind` is the loopback ephemeral here — UDS path-bind
/// takes precedence in the listener once spec.path is set, but the
/// SocketAddr field is still required by RunConfig's shape.
#[cfg(unix)]
#[derive(Debug, Clone, Builder, Deserialize, Serialize)]
#[builder(derive(Clone, Debug), on(PathBuf, into))]
pub struct HttpUdsListener {
    pub path: PathBuf,

    /// File mode applied after bind. `None` leaves umask defaults.
    #[serde(default)]
    pub mode: Option<u32>,

    #[serde(default)]
    pub max_body_bytes: Option<usize>,
}

#[cfg(unix)]
impl HttpUdsListener {
    /// Shorthand with mode 0o600 (the common local-only default).
    #[must_use]
    pub fn local(path: PathBuf) -> Self {
        Self::builder().path(path).mode(0o600).build()
    }
}

#[cfg(unix)]
impl From<HttpUdsListener> for RunConfig {
    fn from(value: HttpUdsListener) -> Self {
        let mut spec = serde_json::Map::new();
        spec.insert(
            "path".into(),
            Value::from(value.path.to_string_lossy().to_string()),
        );
        if let Some(mode) = value.mode {
            spec.insert("mode".into(), Value::from(mode));
        }
        if let Some(max) = value.max_body_bytes {
            spec.insert("max_body_bytes".into(), Value::from(max));
        }
        Self {
            // UDS path is what the listener actually binds; this
            // SocketAddr is unused on the UDS path. RunConfig's shape
            // still requires it.
            bind: std::net::SocketAddr::from(([127, 0, 0, 1], 0)),
            protocol: "http".into(),
            spec: Value::Object(spec),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn http_listener_round_trips_via_run_config() {
        let listener = HttpListener::http("0.0.0.0:8080".parse().unwrap());
        let config: RunConfig = listener.into();
        assert_eq!(config.bind, "0.0.0.0:8080".parse().unwrap());
        assert_eq!(config.protocol, "http");
    }

    #[test]
    fn http_listener_with_max_body_bytes_carries_through() {
        let listener = HttpListener::builder()
            .addr("0.0.0.0:8080".parse().unwrap())
            .max_body_bytes(1_048_576)
            .build();
        let config: RunConfig = listener.into();
        let value = config
            .spec
            .get("max_body_bytes")
            .and_then(|v| v.as_u64())
            .expect("max_body_bytes round-trips");
        assert_eq!(value, 1_048_576);
    }

    #[cfg(unix)]
    #[test]
    fn http_uds_listener_carries_path_and_mode() {
        let listener = HttpUdsListener::local(PathBuf::from("/tmp/proxima.test.sock"));
        let config: RunConfig = listener.into();
        assert_eq!(
            config.spec.get("path").and_then(|v| v.as_str()),
            Some("/tmp/proxima.test.sock"),
        );
        assert_eq!(
            config.spec.get("mode").and_then(|v| v.as_u64()),
            Some(0o600),
        );
    }

    #[cfg(feature = "tls")]
    #[test]
    fn https_listener_serializes_tls_block_into_spec() {
        let generated =
            rcgen::generate_simple_self_signed(vec!["localhost".to_string()]).expect("rcgen");
        let tmp = std::env::temp_dir().join(format!(
            "proxima-https-listener-{}-{}",
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
        let listener = HttpsListener::https("0.0.0.0:8443".parse().unwrap(), cert_path, key_path);
        let config: RunConfig = listener
            .try_into()
            .expect("tls config builds from real pem");
        let tls = config
            .spec
            .get(crate::tls::SPEC_KEY)
            .expect("tls block present");
        assert!(tls.is_object());
        std::fs::remove_dir_all(&tmp).ok();
    }
}
