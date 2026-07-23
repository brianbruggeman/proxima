//! `RedisAnyProtocol` — redis/RESP as an [`AnyProtocol`] candidate for the
//! open universal listener (`Listener::builder().accept("redis")` /
//! `AnyListenProtocol`). Authored directly against `AnyProtocol` — there is
//! no standalone `RedisListenProtocol` bind+accept loop preceding this one;
//! redis's listen-side surface has always been an `AnyProtocol` candidate.
//!
//! Positive-match probe: every RESP2/RESP3 command a real redis client
//! sends is a multi-bulk array, whose first byte is the `*` sigil (a bare
//! RESP inline command with no `*` prefix is out of scope for the shared
//! open classifier — mount via `.accept("redis")`, a single-candidate
//! registration, exactly like pgwire's own positive-match reasoning).
//!
//! `drive` carries its own engine (`handler`, `config`) as a struct field —
//! the same `AnyHandler`-unused asymmetry [`crate::pipe::RedisConnectionPipe`]
//! docs. Redis's handler is NOT bespoke: [`crate::pipes::RedisPipeHandle`]
//! is `SendPipe<RedisRequest, RespValue>` (no `Request`/`Response`
//! envelope), the same de-enveloped typed-handle shape pgwire's
//! [`crate::pipes::RedisPipeHandle`] sibling (`proxima_pgwire::PgPipeHandle`)
//! uses. Each accepted connection builds a
//! FRESH [`RedisConnectionPipe`] carrying THIS connection's [`ConnAdmission`]
//! clone, erases it, and hands it to
//! [`proxima_listen::serve_pipe::handle_connection`] — the ONE
//! CONNECT-request/upgrade-handler driver pgwire and redis now share.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde_json::Value;

use proxima_core::ProximaError;
use proxima_listen::admission::ConnAdmission;
use proxima_listen::any::{AnyHandler, AnyProtocol, ProbeVerdict};
use proxima_primitives::pipe::handler::into_handle;
use proxima_primitives::stream::{PeerInfo, StreamConnection};

use crate::config::RedisServerConfig;
use crate::pipe::RedisConnectionPipe;
use crate::pipes::RedisPipeHandle;

/// RESP multi-bulk array sigil — the first byte of every real redis
/// command a client sends.
const RESP_ARRAY_SIGIL: u8 = b'*';

/// Redis/Valkey wire candidate for the open universal listener.
pub struct RedisAnyProtocol {
    label: String,
    handler: RedisPipeHandle,
    config: RedisServerConfig,
    /// Built ONCE here, not per connection — `drive` installs this SAME
    /// `Arc` onto every fresh per-connection `RedisConnectionPipe` it
    /// builds (see `RedisConnectionPipe::with_broker`'s doc for why a
    /// fresh broker per connection would silently break PUBLISH/SUBSCRIBE
    /// across connections).
    broker: Arc<crate::broker::RedisBroker>,
}

impl RedisAnyProtocol {
    #[must_use]
    pub fn new(label: impl Into<String>, handler: RedisPipeHandle) -> Self {
        Self {
            label: label.into(),
            handler,
            config: RedisServerConfig::default(),
            broker: Arc::new(crate::broker::RedisBroker::new()),
        }
    }

    /// Replaces the default [`RedisServerConfig`]; a `redis` object in the
    /// listener spec still wins at drive time.
    #[must_use]
    pub fn with_config(mut self, config: RedisServerConfig) -> Self {
        self.config = config;
        self
    }
}

fn resolve_config(
    base: &RedisServerConfig,
    spec: &Value,
) -> Result<RedisServerConfig, ProximaError> {
    match spec.get("redis") {
        None => Ok(base.clone()),
        Some(overrides) => serde_json::from_value(overrides.clone())
            .map_err(|error| ProximaError::Config(format!("redis spec: {error}"))),
    }
}

impl AnyProtocol for RedisAnyProtocol {
    fn name(&self) -> &str {
        &self.label
    }

    /// One byte suffices: the `*` sigil either is or isn't there.
    fn max_prefix_bytes(&self) -> usize {
        1
    }

    fn probe(&self, prefix: &[u8]) -> ProbeVerdict {
        match prefix.first() {
            None => ProbeVerdict::NeedMore { at_least: 1 },
            Some(&RESP_ARRAY_SIGIL) => ProbeVerdict::Match { consumed: 0 },
            Some(_) => ProbeVerdict::No,
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
            let connection_pipe = RedisConnectionPipe::new(
                self.label.clone(),
                self.handler.clone(),
                Arc::new(config),
            )
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

    struct EchoPipe;

    impl proxima_primitives::pipe::SendPipe for EchoPipe {
        type In = proxima_protocols::redis::RedisRequest;
        type Out = proxima_protocols::redis::RespValue;
        type Err = ProximaError;

        async fn call(&self, _request: Self::In) -> Result<Self::Out, ProximaError> {
            Ok(proxima_protocols::redis::RespValue::SimpleString(
                "OK".to_string(),
            ))
        }
    }

    fn handler() -> RedisPipeHandle {
        crate::pipes::into_redis_handle(EchoPipe)
    }

    #[test]
    fn probe_matches_the_array_sigil_and_rejects_anything_else() {
        let protocol = RedisAnyProtocol::new("redis", handler());
        assert_eq!(protocol.probe(b"*1\r\n"), ProbeVerdict::Match { consumed: 0 });
        assert_eq!(protocol.probe(b""), ProbeVerdict::NeedMore { at_least: 1 });
        assert_eq!(protocol.probe(b"GET foo"), ProbeVerdict::No);
    }
}
