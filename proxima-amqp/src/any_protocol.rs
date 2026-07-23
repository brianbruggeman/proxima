//! `AmqpAnyProtocol` — AMQP 0-9-1 as an [`AnyProtocol`] candidate for the
//! open universal listener (`Listener::builder().accept("amqp")` /
//! `AnyListenProtocol`), mirroring `proxima_redis::any_protocol::RedisAnyProtocol`
//! 1:1: no standalone `AmqpListenProtocol` bind+accept loop — AMQP's
//! listen-side surface is an `AnyProtocol` candidate from the start.
//!
//! Positive-match probe: every real AMQP 0-9-1 client sends the literal
//! 8-byte protocol header `"AMQP\0\0\x09\x01"` before anything else
//! (§4.2.2) — an exact, fixed-length prefix, the same shape h2's RFC 9113
//! preface probe uses.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde_json::Value;

use proxima_core::ProximaError;
use proxima_listen::admission::ConnAdmission;
use proxima_listen::any::{AnyHandler, AnyProtocol, ProbeVerdict};
use proxima_primitives::pipe::handler::into_handle;
use proxima_primitives::stream::{PeerInfo, StreamConnection};

use crate::config::AmqpServerConfig;
use crate::fsm::PROTOCOL_HEADER;
use crate::pipe::AmqpConnectionPipe;
use crate::pipes::AmqpPipeHandle;

/// AMQP 0-9-1 wire candidate for the open universal listener.
pub struct AmqpAnyProtocol {
    label: String,
    handler: AmqpPipeHandle,
    config: AmqpServerConfig,
    /// Built ONCE here, not per connection — `drive` installs this SAME
    /// `Arc` onto every fresh per-connection `AmqpConnectionPipe` it
    /// builds (see `AmqpConnectionPipe::with_broker`'s doc).
    broker: Arc<crate::broker::AmqpBroker>,
}

impl AmqpAnyProtocol {
    #[must_use]
    pub fn new(label: impl Into<String>, handler: AmqpPipeHandle) -> Self {
        Self {
            label: label.into(),
            handler,
            config: AmqpServerConfig::default(),
            broker: Arc::new(crate::broker::AmqpBroker::new()),
        }
    }

    /// Replaces the default [`AmqpServerConfig`]; an `amqp` object in the
    /// listener spec still wins at drive time.
    #[must_use]
    pub fn with_config(mut self, config: AmqpServerConfig) -> Self {
        self.config = config;
        self
    }
}

fn resolve_config(base: &AmqpServerConfig, spec: &Value) -> Result<AmqpServerConfig, ProximaError> {
    match spec.get("amqp") {
        None => Ok(base.clone()),
        Some(overrides) => serde_json::from_value(overrides.clone())
            .map_err(|error| ProximaError::Config(format!("amqp spec: {error}"))),
    }
}

impl AnyProtocol for AmqpAnyProtocol {
    fn name(&self) -> &str {
        &self.label
    }

    fn max_prefix_bytes(&self) -> usize {
        PROTOCOL_HEADER.len()
    }

    fn probe(&self, prefix: &[u8]) -> ProbeVerdict {
        if prefix.len() < PROTOCOL_HEADER.len() {
            if PROTOCOL_HEADER.starts_with(prefix) {
                return ProbeVerdict::NeedMore {
                    at_least: PROTOCOL_HEADER.len(),
                };
            }
            return ProbeVerdict::No;
        }
        if prefix[..PROTOCOL_HEADER.len()] == PROTOCOL_HEADER {
            ProbeVerdict::Match {
                consumed: PROTOCOL_HEADER.len(),
            }
        } else {
            ProbeVerdict::No
        }
    }

    fn drive<'a>(
        &'a self,
        stream: Box<dyn StreamConnection>,
        _handler: AnyHandler,
        spec: &'a Value,
        _peer: Option<PeerInfo>,
        admission: &'a ConnAdmission,
    ) -> Pin<Box<dyn Future<Output = Result<(), ProximaError>> + Send + 'a>> {
        Box::pin(async move {
            let config = resolve_config(&self.config, spec)?;
            let connection_pipe =
                AmqpConnectionPipe::new(self.label.clone(), self.handler.clone(), Arc::new(config))
                    .with_broker(Arc::clone(&self.broker))
                    .with_admission(admission.clone());
            let pipe = into_handle(connection_pipe);
            proxima_listen::serve_pipe::handle_connection(stream, pipe).await
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use proxima_primitives::pipe::request::Response;

    struct EchoPipe;

    impl proxima_primitives::pipe::SendPipe for EchoPipe {
        type In = crate::pipes::AmqpPipeRequest;
        type Out = crate::pipes::AmqpPipeReply;
        type Err = ProximaError;

        async fn call(&self, _request: Self::In) -> Result<Self::Out, ProximaError> {
            Ok(Response::typed(200, ()))
        }
    }

    fn handler() -> AmqpPipeHandle {
        crate::pipes::into_amqp_handle(EchoPipe)
    }

    #[test]
    fn probe_matches_the_full_protocol_header_and_rejects_anything_else() {
        let protocol = AmqpAnyProtocol::new("amqp", handler());
        assert_eq!(
            protocol.probe(b"AMQP\0\0\x09\x01"),
            ProbeVerdict::Match { consumed: 8 }
        );
        assert_eq!(protocol.probe(b""), ProbeVerdict::NeedMore { at_least: 8 });
        assert_eq!(
            protocol.probe(b"AMQP\0\0"),
            ProbeVerdict::NeedMore { at_least: 8 }
        );
        assert_eq!(protocol.probe(b"GET / HTTP/1.1\r\n"), ProbeVerdict::No);
        assert_eq!(protocol.probe(b"AMQP\0\0\x09\x02"), ProbeVerdict::No);
    }
}
