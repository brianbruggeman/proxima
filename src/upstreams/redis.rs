//! `PipeFactory` for the `redis` protocol — a `proxima::Client` transport that
//! speaks the Redis/Valkey RESP wire protocol.
//!
//! Reached via the `type` discriminator (`{"type":"redis", "dsn": "..."}` or
//! `{"type":"redis", "host":..., "port":..., ...}`), so it needs no edit to the
//! spec precedence chain — the extensible terminal seam. Composes the sans-IO
//! redis client ([`RedisClientUpstream`](proxima_redis::RedisClientUpstream))
//! over the prime TCP transport ([`PrimeTcpUpstream`](crate::PrimeTcpUpstream)),
//! exactly like the prime `pgwire`/`grpc` factories. Valkey is the same RESP
//! wire protocol, so `.valkey(dsn)` aliases this same factory — one codec, one
//! client, one factory cover both. The RESP-over-Pipe request shape (verb in
//! `Request.method`, args in body / the `RedisRequest` carry) is the caller's;
//! this factory is purely the transport.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde_json::Value;

use proxima_primitives::pipe::handler::{PipeHandle, into_handle};
use proxima_primitives::pipe::pipe_factory::PipeFactory;
use proxima_redis::{RedisClientConfig, RedisClientUpstream};

use crate::PrimeTcpUpstream;
use crate::client::handle::ClientProtocol;
use crate::error::ProximaError;

/// A [`PipeFactory`] for the `redis` key (Valkey shares it). Builds a client
/// `Pipe` from a [`RedisClientConfig`] parsed out of the spec.
#[derive(Debug, Default)]
pub struct RedisPipeFactory;

impl RedisPipeFactory {
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl PipeFactory for RedisPipeFactory {
    fn name(&self) -> &str {
        "redis"
    }

    fn build(
        &self,
        spec: &Value,
        _inner: Option<PipeHandle>,
    ) -> Pin<Box<dyn Future<Output = Result<PipeHandle, ProximaError>> + Send + '_>> {
        let spec = spec.clone();
        Box::pin(async move {
            let config = config_from_spec(&spec)?;
            // DNS is resolved lazily by the prime upstream on connect, so `build`
            // stays side-effect-free (mirrors the prime pgwire factory).
            let upstream = PrimeTcpUpstream::with_host(config.host.clone(), config.port);
            Ok(into_handle(RedisClientUpstream::new(upstream, config)))
        })
    }
}

/// Parse a [`RedisClientConfig`] from the spec: prefer a `dsn` string, else
/// deserialize the field form (serde ignores the `type` discriminator).
fn config_from_spec(spec: &Value) -> Result<RedisClientConfig, ProximaError> {
    if let Some(dsn) = spec.get("dsn").and_then(Value::as_str) {
        return RedisClientConfig::from_dsn(dsn)
            .map_err(|err| ProximaError::Config(format!("redis dsn: {err}")));
    }
    serde_json::from_value(spec.clone())
        .map_err(|err| ProximaError::Config(format!("redis config: {err}")))
}

/// The out-of-crate [`ClientProtocol`] a `.redis(dsn)` / `.valkey(dsn)`
/// builder call merges — migrated OFF the old bespoke inherent
/// `ClientBuilder::{redis,valkey}` onto the same `.protocol()` mechanism
/// every other protocol terminal uses, wrapping this SAME
/// [`RedisPipeFactory`] (net-zero runtime change; see Section E of the
/// builder-sugar design).
pub struct RedisClientProtocol {
    dsn: String,
}

impl RedisClientProtocol {
    /// Point at a Redis/Valkey server by DSN (`redis://[user:pass@]host[:port][/db]`).
    #[must_use]
    pub fn dsn(dsn: impl Into<String>) -> Self {
        Self { dsn: dsn.into() }
    }
}

impl ClientProtocol for RedisClientProtocol {
    fn spec(&self) -> Value {
        serde_json::json!({"type": "redis", "dsn": self.dsn})
    }

    fn factory(&self) -> Arc<dyn PipeFactory> {
        Arc::new(RedisPipeFactory::new())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::client::protocol::ClientProtocolExt;

    #[test]
    fn config_from_dsn_spec() {
        let spec = serde_json::json!({ "type": "redis", "dsn": "redis://u:p@h:6380/2" });
        let config = config_from_spec(&spec).expect("config");
        assert_eq!(
            (config.host.as_str(), config.port, config.db),
            ("h", 6380, 2)
        );
        assert_eq!(
            (config.username.as_str(), config.password.as_str()),
            ("u", "p")
        );
    }

    #[test]
    fn config_from_fields_ignores_type_discriminator() {
        let spec = serde_json::json!({ "type": "redis", "host": "cache", "port": 6390, "db": 1 });
        let config = config_from_spec(&spec).expect("config");
        assert_eq!(
            (config.host.as_str(), config.port, config.db),
            ("cache", 6390, 1)
        );
        assert!(config.resp3, "unspecified field falls back to default");
    }

    #[test]
    fn factory_name_is_the_spec_key() {
        assert_eq!(RedisPipeFactory::new().name(), "redis");
    }

    #[test]
    fn client_protocol_lowers_to_the_type_and_dsn_spec() {
        let protocol = RedisClientProtocol::dsn("redis://h:6379");
        let spec = protocol.spec();
        assert_eq!(spec["type"], "redis");
        assert_eq!(spec["dsn"], "redis://h:6379");
        assert_eq!(protocol.factory().name(), "redis");
    }

    /// The headline: redis reached through `proxima::Client` like any other
    /// protocol. `.redis(dsn)` / `.valkey(dsn)` lower to the `type` terminal,
    /// `load()` resolves this factory. Args are NUL-delimited in the body;
    /// the reply is RESP wire bytes in the response payload.
    /// Off-worker: `Client` auto-dispatches onto the shared prime runtime.
    /// Env-gated on a reachable server.
    #[cfg(feature = "runtime-prime")]
    #[test]
    fn redis_through_client_round_trips_real_server() {
        let host = match std::env::var("REDIS_REAL_HOST") {
            Ok(host) if !host.is_empty() => host,
            _ => {
                eprintln!("skipping redis_through_client: REDIS_REAL_HOST unset (no server)");
                return;
            }
        };
        let port = std::env::var("REDIS_REAL_PORT").unwrap_or_else(|_| "6379".to_string());
        let dsn = format!("redis://{host}:{port}");

        let ok = futures::executor::block_on(async move {
            let client = crate::Client::builder()
                .redis(&dsn)
                .build()
                .expect("build client");
            // SET proxima:e2e ok — args are NUL-delimited in the body
            let set_bytes = client
                .call("SET", "/")
                .body(b"proxima:e2e\0ok".as_slice())
                .send()
                .await
                .expect("set")
                .bytes()
                .await
                .expect("set body");
            let (set_frame, _) = proxima_redis::parse(&set_bytes).expect("parse set reply");
            let set_value = proxima_redis::RespValue::from_frame(&set_frame);
            assert_eq!(set_value.as_str(), Some("OK"));
            let got_bytes = client
                .call("GET", "/")
                .body("proxima:e2e")
                .send()
                .await
                .expect("get")
                .bytes()
                .await
                .expect("get body");
            let (got_frame, _) = proxima_redis::parse(&got_bytes).expect("parse get reply");
            proxima_redis::RespValue::from_frame(&got_frame)
                .as_str()
                .map(str::to_string)
        });
        assert_eq!(ok.as_deref(), Some("ok"));
    }
}
