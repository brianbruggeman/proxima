//! `PipeFactory` + [`ClientProtocol`] for the `memcached` protocol terminal —
//! a `proxima::Client` transport that speaks the memcached ASCII protocol.
//!
//! Reached via the `type` discriminator (`{"type":"memcached", "dsn":
//! "memcached://cache:11211"}` or the field form) or the `.memcached(dsn)`
//! builder sugar (`ClientProtocolExt`), which lowers to
//! `.protocol(MemcachedClientProtocol::dsn(dsn))`. Composes the sans-IO
//! memcached client
//! ([`MemcachedClientUpstream`](proxima_memcached::MemcachedClientUpstream))
//! over the prime TCP transport ([`PrimeTcpUpstream`](crate::PrimeTcpUpstream)),
//! exactly like the prime `kafka`/`pgwire`/`redis` factories.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde_json::Value;

use proxima_memcached::{MemcachedClientConfig, MemcachedClientUpstream};
use proxima_primitives::pipe::handler::{PipeHandle, into_handle};
use proxima_primitives::pipe::pipe_factory::PipeFactory;

use crate::PrimeTcpUpstream;
use crate::client::handle::ClientProtocol;
use crate::error::ProximaError;

/// A [`PipeFactory`] for the `memcached` key. Builds a client `Pipe` from a
/// [`MemcachedClientConfig`] parsed out of the spec.
#[derive(Debug, Default)]
pub struct MemcachedPipeFactory;

impl MemcachedPipeFactory {
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl PipeFactory for MemcachedPipeFactory {
    fn name(&self) -> &str {
        "memcached"
    }

    fn build(
        &self,
        spec: &Value,
        _inner: Option<PipeHandle>,
    ) -> Pin<Box<dyn Future<Output = Result<PipeHandle, ProximaError>> + Send + '_>> {
        let spec = spec.clone();
        Box::pin(async move {
            let config = config_from_spec(&spec)?;
            let upstream = PrimeTcpUpstream::with_host(config.host.clone(), config.port);
            Ok(into_handle(MemcachedClientUpstream::new(upstream, config)))
        })
    }
}

/// Parse a [`MemcachedClientConfig`] from the spec: prefer a `dsn` string,
/// else deserialize the field form (serde ignores the `type` discriminator).
fn config_from_spec(spec: &Value) -> Result<MemcachedClientConfig, ProximaError> {
    if let Some(dsn) = spec.get("dsn").and_then(Value::as_str) {
        return MemcachedClientConfig::from_dsn(dsn)
            .map_err(|err| ProximaError::Config(format!("memcached dsn: {err}")));
    }
    serde_json::from_value(spec.clone())
        .map_err(|err| ProximaError::Config(format!("memcached config: {err}")))
}

/// The out-of-crate [`ClientProtocol`] a `.memcached(dsn)` builder call merges.
pub struct MemcachedClientProtocol {
    dsn: String,
}

impl MemcachedClientProtocol {
    /// Point at a memcached server by DSN (`memcached://host[:port]`).
    #[must_use]
    pub fn dsn(dsn: impl Into<String>) -> Self {
        Self { dsn: dsn.into() }
    }
}

impl ClientProtocol for MemcachedClientProtocol {
    fn spec(&self) -> Value {
        serde_json::json!({"type": "memcached", "dsn": self.dsn})
    }

    fn factory(&self) -> Arc<dyn PipeFactory> {
        Arc::new(MemcachedPipeFactory::new())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn config_from_dsn_spec() {
        let spec = serde_json::json!({ "type": "memcached", "dsn": "memcached://cache:11212" });
        let config = config_from_spec(&spec).expect("config");
        assert_eq!((config.host.as_str(), config.port), ("cache", 11212));
    }

    #[test]
    fn factory_name_is_the_spec_key() {
        assert_eq!(MemcachedPipeFactory::new().name(), "memcached");
    }

    #[test]
    fn client_protocol_lowers_to_the_type_and_dsn_spec() {
        let protocol = MemcachedClientProtocol::dsn("memcached://cache:11211");
        let spec = protocol.spec();
        assert_eq!(spec["type"], "memcached");
        assert_eq!(spec["dsn"], "memcached://cache:11211");
        assert_eq!(protocol.factory().name(), "memcached");
    }
}
