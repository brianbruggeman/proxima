//! `PipeFactory` + [`ClientProtocol`] for the `kafka` protocol terminal — a
//! `proxima::Client` transport that speaks the Kafka wire protocol.
//!
//! Reached via the `type` discriminator (`{"type":"kafka", "dsn":
//! "kafka://broker:9092"}` or the field form) or the `.kafka(dsn)` builder
//! sugar (`ClientProtocolExt`), which lowers to `.protocol(KafkaClientProtocol::dsn(dsn))`.
//! Composes the sans-IO Kafka client
//! ([`KafkaClientUpstream`](proxima_kafka::KafkaClientUpstream)) over the
//! prime TCP transport ([`PrimeTcpUpstream`](crate::PrimeTcpUpstream)),
//! exactly like the prime `pgwire`/`redis` factories — `KafkaClientUpstream`
//! already speaks `Request<Bytes>`/`Response<Bytes>` directly (its own
//! PRODUCE/FETCH/METADATA method convention), so no translation wrapper is
//! needed here, unlike the DNS terminal.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde_json::Value;

use proxima_kafka::{KafkaClientConfig, KafkaClientUpstream};
use proxima_primitives::pipe::handler::{PipeHandle, into_handle};
use proxima_primitives::pipe::pipe_factory::PipeFactory;

use crate::PrimeTcpUpstream;
use crate::client::handle::ClientProtocol;
use crate::error::ProximaError;

/// A [`PipeFactory`] for the `kafka` key. Builds a client `Pipe` from a
/// [`KafkaClientConfig`] parsed out of the spec.
#[derive(Debug, Default)]
pub struct KafkaPipeFactory;

impl KafkaPipeFactory {
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl PipeFactory for KafkaPipeFactory {
    fn name(&self) -> &str {
        "kafka"
    }

    fn build(
        &self,
        spec: &Value,
        _inner: Option<PipeHandle>,
    ) -> Pin<Box<dyn Future<Output = Result<PipeHandle, ProximaError>> + Send + '_>> {
        let spec = spec.clone();
        Box::pin(async move {
            let config = config_from_spec(&spec)?;
            // DNS is resolved lazily by the prime upstream on connect, so
            // `build` stays side-effect-free (mirrors the prime pgwire/redis
            // factories).
            let upstream = PrimeTcpUpstream::with_host(config.host.clone(), config.port);
            Ok(into_handle(KafkaClientUpstream::new(upstream, config)))
        })
    }
}

/// Parse a [`KafkaClientConfig`] from the spec: prefer a `dsn` string, else
/// deserialize the field form (serde ignores the `type` discriminator).
fn config_from_spec(spec: &Value) -> Result<KafkaClientConfig, ProximaError> {
    if let Some(dsn) = spec.get("dsn").and_then(Value::as_str) {
        return KafkaClientConfig::from_dsn(dsn)
            .map_err(|err| ProximaError::Config(format!("kafka dsn: {err}")));
    }
    serde_json::from_value(spec.clone())
        .map_err(|err| ProximaError::Config(format!("kafka config: {err}")))
}

/// The out-of-crate [`ClientProtocol`] a `.kafka(dsn)` builder call merges —
/// a thin DSN carrier, since [`KafkaPipeFactory`] itself reads the spec.
pub struct KafkaClientProtocol {
    dsn: String,
}

impl KafkaClientProtocol {
    /// Point at a Kafka broker by DSN (`kafka://broker[:port]`).
    #[must_use]
    pub fn dsn(dsn: impl Into<String>) -> Self {
        Self { dsn: dsn.into() }
    }
}

impl ClientProtocol for KafkaClientProtocol {
    fn spec(&self) -> Value {
        serde_json::json!({"type": "kafka", "dsn": self.dsn})
    }

    fn factory(&self) -> Arc<dyn PipeFactory> {
        Arc::new(KafkaPipeFactory::new())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn config_from_dsn_spec() {
        let spec = serde_json::json!({ "type": "kafka", "dsn": "kafka://broker:9093" });
        let config = config_from_spec(&spec).expect("config");
        assert_eq!((config.host.as_str(), config.port), ("broker", 9093));
    }

    #[test]
    fn factory_name_is_the_spec_key() {
        assert_eq!(KafkaPipeFactory::new().name(), "kafka");
    }

    #[test]
    fn client_protocol_lowers_to_the_type_and_dsn_spec() {
        let protocol = KafkaClientProtocol::dsn("kafka://broker:9092");
        let spec = protocol.spec();
        assert_eq!(spec["type"], "kafka");
        assert_eq!(spec["dsn"], "kafka://broker:9092");
        assert_eq!(protocol.factory().name(), "kafka");
    }
}
