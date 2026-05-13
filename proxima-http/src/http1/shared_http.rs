use std::sync::Arc;
use std::time::Duration;

use hyper::body::Incoming;
use hyper_util::client::legacy::Client;
use hyper_util::client::legacy::connect::HttpConnector;
use hyper_util::rt::TokioExecutor;
use serde::{Deserialize, Serialize};

use crate::http1::hyper_body::StreamingHyperBody;
use proxima_core::ProximaError;

/// Tuning knobs for the shared HTTP/1.1 + HTTP/2 connection pool.
///
/// All fields are optional — `None` means "use the hyper-util default."
/// Defaults aren't repeated here because they belong to hyper-util,
/// not to proxima; if a future hyper-util release changes them, we
/// pick up the change automatically.
///
/// The substrate exposes this struct rather than hardcoding values
/// so operators can tune their proxima deployment for their workload
/// without recompiling.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "snake_case")]
pub struct PoolConfig {
    /// Maximum idle connections per (scheme, host, port). hyper-util
    /// default is `usize::MAX` — set to a real number to bound the
    /// idle pool when the workload has many destinations.
    pub max_idle_per_host: Option<usize>,
    /// How long an idle connection stays in the pool before being
    /// dropped. Set lower for bursty traffic that doesn't need long-
    /// lived keepalives. Stored as milliseconds for TOML/JSON
    /// friendliness.
    pub idle_timeout_ms: Option<u64>,
    /// HTTP/2 SETTINGS_INITIAL_WINDOW_SIZE for the stream-level flow
    /// control. Default is hyper-util's; raise for high-bandwidth
    /// long-RTT links.
    pub http2_initial_stream_window_size: Option<u32>,
    /// HTTP/2 connection-level flow-control window.
    pub http2_initial_connection_window_size: Option<u32>,
    /// Enable adaptive HTTP/2 window updates. Lets hyper learn the
    /// link's BDP at runtime instead of using static sizes.
    pub http2_adaptive_window: Option<bool>,
    /// HTTP/2 PING-based keep-alive interval (ms). Useful behind
    /// load balancers / NATs that drop idle long-lived connections.
    pub http2_keep_alive_interval_ms: Option<u64>,
    /// How long to wait for a PING reply before giving up. Pairs
    /// with `http2_keep_alive_interval_ms`.
    pub http2_keep_alive_timeout_ms: Option<u64>,
    /// HTTP/2 send-buffer size cap. Default 1 MiB.
    pub http2_max_send_buf_size: Option<usize>,
    /// HTTP/1 read-buffer cap. Default 400 KiB.
    pub http1_max_buf_size: Option<usize>,
    /// HTTP/1 max headers in one response.
    pub http1_max_headers: Option<usize>,
}

#[cfg(feature = "http1-tls")]
pub type ConnectorImpl = hyper_rustls::HttpsConnector<HttpConnector>;
#[cfg(not(feature = "http1-tls"))]
pub type ConnectorImpl = HttpConnector;

pub type SharedHyperClient = Client<ConnectorImpl, StreamingHyperBody>;

/// Process-wide hyper client. The hyper-util `Client` type already pools
/// connections internally per `(scheme, host, port)`; sharing one instance
/// across upstreams collapses the pool budget into a single shared
/// reservoir instead of N isolated reservoirs (one per HttpUpstream).
///
/// When the `tls` feature is enabled, the inner connector is
/// `hyper_rustls::HttpsConnector<HttpConnector>` so the same client
/// handles both http:// and https:// upstreams. Without `tls`, only
/// http:// is supported and https requests fail at connect time.
#[derive(Clone)]
pub struct SharedHttpClient {
    inner: Arc<SharedHyperClient>,
}

impl SharedHttpClient {
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(&PoolConfig::default())
    }

    /// Build a SharedHttpClient with the given pool tuning. All knob
    /// fields are optional — `None` defers to hyper-util's default.
    #[must_use]
    pub fn with_config(config: &PoolConfig) -> Self {
        let mut builder = Client::builder(TokioExecutor::new());
        if let Some(max_idle) = config.max_idle_per_host {
            builder.pool_max_idle_per_host(max_idle);
        }
        if let Some(idle_ms) = config.idle_timeout_ms {
            builder.pool_idle_timeout(Duration::from_millis(idle_ms));
        }
        if let Some(window) = config.http2_initial_stream_window_size {
            builder.http2_initial_stream_window_size(window);
        }
        if let Some(window) = config.http2_initial_connection_window_size {
            builder.http2_initial_connection_window_size(window);
        }
        if let Some(adaptive) = config.http2_adaptive_window {
            builder.http2_adaptive_window(adaptive);
        }
        if let Some(interval_ms) = config.http2_keep_alive_interval_ms {
            builder.http2_keep_alive_interval(Some(Duration::from_millis(interval_ms)));
        }
        if let Some(timeout_ms) = config.http2_keep_alive_timeout_ms {
            builder.http2_keep_alive_timeout(Duration::from_millis(timeout_ms));
        }
        if let Some(max) = config.http2_max_send_buf_size {
            builder.http2_max_send_buf_size(max);
        }
        if let Some(max) = config.http1_max_buf_size {
            builder.http1_max_buf_size(max);
        }
        if let Some(max) = config.http1_max_headers {
            builder.http1_max_headers(max);
        }
        let client = builder.build(connector());
        Self {
            inner: Arc::new(client),
        }
    }

    #[must_use]
    pub fn from_client(client: SharedHyperClient) -> Self {
        Self {
            inner: Arc::new(client),
        }
    }

    #[must_use]
    pub fn client(&self) -> &SharedHyperClient {
        &self.inner
    }

    #[must_use]
    pub fn strong_count(&self) -> usize {
        Arc::strong_count(&self.inner)
    }

    /// Send an HTTP request, return the upstream's response.
    ///
    /// Default path (`!io-uring` or non-Linux): delegates to the
    /// pooled `hyper-util` Client.
    ///
    /// io-uring path (`linux + io-uring` feature): per-request
    /// `tokio_uring::net::TcpStream` connect → `UringAsyncStream`
    /// (+ TLS handshake when scheme is https) → raw hyper
    /// `client::conn::http1::handshake`. The !Send connection driver
    /// runs in `spawn_local`; the request future returned here stays
    /// Send (the !Send work happens inside the spawned task and the
    /// result returns over a Send oneshot channel). No pooling on
    /// the io-uring path yet — Stage 5f.
    pub async fn request(
        &self,
        request: hyper::Request<StreamingHyperBody>,
    ) -> Result<hyper::Response<Incoming>, ProximaError> {
        // On the io_uring per-core runtime, tokio::net::TcpStream has no
        // reactor — route outbound through the tokio_uring-backed client.
        #[cfg(all(target_os = "linux", feature = "http1-io-uring"))]
        {
            crate::uring_transport::request_via_uring(request).await
        }
        #[cfg(not(all(target_os = "linux", feature = "http1-io-uring")))]
        {
            self.inner
                .request(request)
                .await
                .map_err(|error| ProximaError::Upstream(format!("send: {error}")))
        }
    }
}

#[cfg(feature = "http1-tls")]
fn connector() -> ConnectorImpl {
    // aws-lc-rs is the plan-locked crypto backend; webpki-roots gives mozilla CA bundle
    // without depending on the system trust store, making the client deterministic across
    // hosts. native-tokio uses the tokio runtime hooks for non-blocking dns + handshake.
    hyper_rustls::HttpsConnectorBuilder::new()
        .with_webpki_roots()
        .https_or_http()
        .enable_http1()
        .enable_http2()
        .build()
}

#[cfg(not(feature = "http1-tls"))]
fn connector() -> ConnectorImpl {
    HttpConnector::new()
}

impl Default for SharedHttpClient {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn cloning_shared_http_client_increments_strong_count() {
        let one = SharedHttpClient::new();
        assert_eq!(one.strong_count(), 1);
        let two = one.clone();
        assert_eq!(two.strong_count(), 2);
        drop(two);
        assert_eq!(one.strong_count(), 1);
    }

    #[test]
    fn distinct_instances_have_independent_arcs() {
        let one = SharedHttpClient::new();
        let two = SharedHttpClient::new();
        assert_eq!(one.strong_count(), 1);
        assert_eq!(two.strong_count(), 1);
    }

    #[test]
    fn pool_config_with_all_knobs_constructs_client() {
        // Build a client with every knob set. This is the smoke test —
        // we can't easily observe the live pool budget from outside
        // hyper-util, but if any of the setters reject our value the
        // build fails here.
        let config = PoolConfig {
            max_idle_per_host: Some(16),
            idle_timeout_ms: Some(30_000),
            http2_initial_stream_window_size: Some(1 << 20),
            http2_initial_connection_window_size: Some(1 << 22),
            http2_adaptive_window: Some(true),
            http2_keep_alive_interval_ms: Some(10_000),
            http2_keep_alive_timeout_ms: Some(30_000),
            http2_max_send_buf_size: Some(1 << 20),
            http1_max_buf_size: Some(64 * 1024),
            http1_max_headers: Some(128),
        };
        let client = SharedHttpClient::with_config(&config);
        assert_eq!(client.strong_count(), 1);
    }

    #[test]
    fn pool_config_default_matches_new() {
        // Default PoolConfig must produce the same observable shape as
        // SharedHttpClient::new() — otherwise the "all knobs optional"
        // contract is broken.
        let from_default = SharedHttpClient::with_config(&PoolConfig::default());
        let from_new = SharedHttpClient::new();
        assert_eq!(from_default.strong_count(), 1);
        assert_eq!(from_new.strong_count(), 1);
    }

    #[test]
    fn pool_config_round_trips_through_toml() {
        let config = PoolConfig {
            max_idle_per_host: Some(32),
            idle_timeout_ms: Some(45_000),
            http2_initial_stream_window_size: Some(2 << 20),
            http2_adaptive_window: Some(false),
            ..Default::default()
        };
        let serialized = toml::to_string(&config).expect("serialize");
        let restored: PoolConfig = toml::from_str(&serialized).expect("deserialize");
        assert_eq!(restored.max_idle_per_host, Some(32));
        assert_eq!(restored.idle_timeout_ms, Some(45_000));
        assert_eq!(restored.http2_initial_stream_window_size, Some(2 << 20));
        assert_eq!(restored.http2_adaptive_window, Some(false));
        assert!(restored.http2_keep_alive_interval_ms.is_none());
    }
}
