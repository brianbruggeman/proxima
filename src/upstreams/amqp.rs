//! `PipeFactory` + [`ClientProtocol`] for the `amqp` protocol terminal — a
//! `proxima::Client` transport that speaks AMQP 0-9-1.
//!
//! Reached via the `type` discriminator (`{"type":"amqp", "dsn":
//! "amqp://user:pass@broker:5672/vhost"}` or the field form) or the
//! `.amqp(dsn)` builder sugar (`ClientProtocolExt`), which lowers to
//! `.protocol(AmqpClientProtocol::dsn(dsn))`. Composes the sans-IO AMQP
//! client ([`AmqpClientUpstream`](proxima_amqp::AmqpClientUpstream)) over
//! the prime TCP transport ([`PrimeTcpUpstream`](crate::PrimeTcpUpstream)),
//! exactly like the prime `kafka`/`pgwire`/`redis` factories.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde_json::Value;

use proxima_amqp::{AmqpClientConfig, AmqpClientUpstream};
use proxima_primitives::pipe::handler::{PipeHandle, into_handle};
use proxima_primitives::pipe::pipe_factory::PipeFactory;

use crate::PrimeTcpUpstream;
use crate::client::handle::ClientProtocol;
use crate::error::ProximaError;

/// A [`PipeFactory`] for the `amqp` key. Builds a client `Pipe` from an
/// [`AmqpClientConfig`] parsed out of the spec.
#[derive(Debug, Default)]
pub struct AmqpPipeFactory;

impl AmqpPipeFactory {
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl PipeFactory for AmqpPipeFactory {
    fn name(&self) -> &str {
        "amqp"
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
            Ok(into_handle(AmqpClientUpstream::new(upstream, config)))
        })
    }
}

/// Parse an [`AmqpClientConfig`] from the spec: prefer a `dsn` string, else
/// deserialize the field form (serde ignores the `type` discriminator).
fn config_from_spec(spec: &Value) -> Result<AmqpClientConfig, ProximaError> {
    if let Some(dsn) = spec.get("dsn").and_then(Value::as_str) {
        return AmqpClientConfig::from_dsn(dsn)
            .map_err(|err| ProximaError::Config(format!("amqp dsn: {err}")));
    }
    serde_json::from_value(spec.clone())
        .map_err(|err| ProximaError::Config(format!("amqp config: {err}")))
}

/// The out-of-crate [`ClientProtocol`] a `.amqp(dsn)` builder call merges.
pub struct AmqpClientProtocol {
    dsn: String,
}

impl AmqpClientProtocol {
    /// Point at an AMQP broker by DSN (`amqp://[user:pass@]broker[:port][/vhost]`).
    #[must_use]
    pub fn dsn(dsn: impl Into<String>) -> Self {
        Self { dsn: dsn.into() }
    }
}

impl ClientProtocol for AmqpClientProtocol {
    fn spec(&self) -> Value {
        serde_json::json!({"type": "amqp", "dsn": self.dsn})
    }

    fn factory(&self) -> Arc<dyn PipeFactory> {
        Arc::new(AmqpPipeFactory::new())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn config_from_dsn_spec() {
        let spec = serde_json::json!({ "type": "amqp", "dsn": "amqp://broker:5673" });
        let config = config_from_spec(&spec).expect("config");
        assert_eq!((config.host.as_str(), config.port), ("broker", 5673));
    }

    #[test]
    fn factory_name_is_the_spec_key() {
        assert_eq!(AmqpPipeFactory::new().name(), "amqp");
    }

    #[test]
    fn client_protocol_lowers_to_the_type_and_dsn_spec() {
        let protocol = AmqpClientProtocol::dsn("amqp://broker:5672");
        let spec = protocol.spec();
        assert_eq!(spec["type"], "amqp");
        assert_eq!(spec["dsn"], "amqp://broker:5672");
        assert_eq!(protocol.factory().name(), "amqp");
    }
}
