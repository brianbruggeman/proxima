//! `KafkaAnyProtocol` — Kafka as an [`AnyProtocol`] candidate for the open
//! universal listener (`Listener::builder().accept("kafka")` /
//! `AnyListenProtocol`).
//!
//! `drive` builds and delegates to a
//! [`proxima_listen::any::FramedAny<KafkaCodec, KafkaFramedApp, _, _>`] —
//! the generic stateless `AnyProtocol` driver, proving Kafka's own
//! request/reply wire drops onto the pipe-centered driver rather than
//! hand-rolling `serve_connection`/a CONNECT-and-upgrade indirection per
//! protocol (see git history: `connection.rs`, `pipe.rs`, both deleted).
//! Kafka's `Fetch` long-poll wait lives entirely INSIDE the handler
//! pipe's own `call` (`crate::broker::KafkaBroker::fetch_partition`
//! races a wake ping against `max_wait_ms`) — `FramedAny` just awaits
//! that call like any other, so Kafka is a stateless `FramedAny`
//! candidate, not a stateful/server-push one. `KafkaAnyProtocol` itself
//! stays a thin, named constructor: it resolves the per-connection
//! `KafkaServerConfig` from the listener spec and BUILDS a fresh
//! `FramedAny` per accepted connection, rather than hand-rolling
//! `impl AnyProtocol` end to end.
//!
//! Positive-match probe: a real Kafka request opens with an 8-byte prefix —
//! a 4-byte big-endian frame length, then the request header's first two
//! fields, `api_key` (`i16`) and `api_version` (`i16`). A length below the
//! smallest possible v0 header (10 bytes: `api_key` + `api_version` +
//! `correlation_id` + a 2-byte `client_id` length) or an `api_key` outside
//! [`wire::SUPPORTED_API_VERSIONS`] is definitively not this facade's wire
//! (mirrors pgwire's own length+code 8-byte positive match).

use std::future::Future;
use std::pin::Pin;

use serde_json::Value;

use proxima_core::ProximaError;
use proxima_listen::admission::{ConnAdmission, ShedReason};
use proxima_listen::any::{AnyHandler, AnyProtocol, FramedAny, ProbeVerdict};
use proxima_primitives::stream::{PeerInfo, StreamConnection};

use crate::config::KafkaServerConfig;
use crate::frame_codec::{KafkaCodec, KafkaOwnedFrame};
use crate::framed_app::{KafkaFramedApp, KafkaOutcome, shed_reply};
use crate::pipes::KafkaPipeHandle;
use crate::wire;

/// 4-byte frame length + 2-byte `api_key` + 2-byte `api_version`.
const PROBE_PREFIX_BYTES: usize = 8;
/// The smallest possible v0 request body length past the frame-length
/// prefix: `api_key`(2) + `api_version`(2) + `correlation_id`(4) +
/// `client_id` nullable-string length(2).
const MIN_V0_HEADER_BYTES: i32 = 10;

/// The concrete [`FramedAny`] instantiation Kafka drives — `Probe`/`Shed`
/// are plain `fn` items (no captured state), so `KafkaAnyProtocol` needs
/// no generic parameters of its own to name this type.
type KafkaFramedAny =
    FramedAny<KafkaCodec, KafkaFramedApp, fn(&[u8]) -> ProbeVerdict, fn(ShedReason, &KafkaOwnedFrame) -> KafkaOutcome>;

fn probe_kafka(prefix: &[u8]) -> ProbeVerdict {
    if prefix.len() < PROBE_PREFIX_BYTES {
        return ProbeVerdict::NeedMore {
            at_least: PROBE_PREFIX_BYTES,
        };
    }
    let length = i32::from_be_bytes([prefix[0], prefix[1], prefix[2], prefix[3]]);
    if length < MIN_V0_HEADER_BYTES {
        return ProbeVerdict::No;
    }
    let api_key = i16::from_be_bytes([prefix[4], prefix[5]]);
    if wire::SUPPORTED_API_VERSIONS.iter().any(|&(key, _, _)| key == api_key) {
        ProbeVerdict::Match { consumed: 0 }
    } else {
        ProbeVerdict::No
    }
}

fn resolve_config(base: &KafkaServerConfig, spec: &Value) -> Result<KafkaServerConfig, ProximaError> {
    match spec.get("kafka") {
        None => Ok(base.clone()),
        Some(overrides) => {
            serde_json::from_value(overrides.clone()).map_err(|error| ProximaError::Config(format!("kafka spec: {error}")))
        }
    }
}

/// Kafka wire candidate for the open universal listener.
pub struct KafkaAnyProtocol {
    label: String,
    handler: KafkaPipeHandle,
    config: KafkaServerConfig,
}

impl KafkaAnyProtocol {
    #[must_use]
    pub fn new(label: impl Into<String>, handler: KafkaPipeHandle) -> Self {
        Self {
            label: label.into(),
            handler,
            config: KafkaServerConfig::default(),
        }
    }

    /// Replaces the default [`KafkaServerConfig`]; a `kafka` object in the
    /// listener spec still wins at drive time.
    #[must_use]
    pub fn with_config(mut self, config: KafkaServerConfig) -> Self {
        self.config = config;
        self
    }

    /// Builds the [`FramedAny`] this connection drives, from `config`
    /// (already resolved against the listener spec).
    fn build(&self, config: &KafkaServerConfig) -> KafkaFramedAny {
        FramedAny::new(
            self.label.clone(),
            KafkaCodec::new(config.max_message_bytes),
            KafkaFramedApp::new(self.handler.clone()),
            probe_kafka as fn(&[u8]) -> ProbeVerdict,
            shed_reply as fn(ShedReason, &KafkaOwnedFrame) -> KafkaOutcome,
            PROBE_PREFIX_BYTES,
        )
    }
}

impl AnyProtocol for KafkaAnyProtocol {
    fn name(&self) -> &str {
        &self.label
    }

    fn max_prefix_bytes(&self) -> usize {
        PROBE_PREFIX_BYTES
    }

    fn probe(&self, prefix: &[u8]) -> ProbeVerdict {
        probe_kafka(prefix)
    }

    fn drive<'a>(
        &'a self,
        stream: Box<dyn StreamConnection>,
        handler: AnyHandler,
        spec: &'a Value,
        peer: Option<PeerInfo>,
        admission: &'a ConnAdmission,
    ) -> Pin<Box<dyn Future<Output = Result<(), ProximaError>> + Send + 'a>> {
        Box::pin(async move {
            let config = resolve_config(&self.config, spec)?;
            let framed = self.build(&config);
            framed.drive(stream, handler, spec, peer, admission).await
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
        type In = crate::pipes::KafkaPipeRequest;
        type Out = crate::pipes::KafkaPipeReply;
        type Err = ProximaError;

        async fn call(&self, _request: Self::In) -> Result<Self::Out, ProximaError> {
            Ok(Response::typed(
                200,
                crate::wire::ResponseBody::ApiVersions(crate::wire::ApiVersionsResponse::supported()),
            ))
        }
    }

    fn handler() -> KafkaPipeHandle {
        crate::pipes::into_kafka_handle(EchoPipe)
    }

    fn api_versions_prefix(correlation_id: i32) -> Vec<u8> {
        let mut payload = Vec::new();
        payload.extend_from_slice(&18_i16.to_be_bytes());
        payload.extend_from_slice(&0_i16.to_be_bytes());
        payload.extend_from_slice(&correlation_id.to_be_bytes());
        payload.extend_from_slice(&(-1_i16).to_be_bytes());
        let mut wire = Vec::new();
        wire.extend_from_slice(&(payload.len() as i32).to_be_bytes());
        wire.extend_from_slice(&payload);
        wire
    }

    #[test]
    fn probe_matches_a_real_api_versions_request_prefix() {
        let protocol = KafkaAnyProtocol::new("kafka", handler());
        assert_eq!(protocol.probe(&api_versions_prefix(1)), ProbeVerdict::Match { consumed: 0 });
    }

    #[test]
    fn probe_needs_more_bytes_below_the_eight_byte_prefix() {
        let protocol = KafkaAnyProtocol::new("kafka", handler());
        assert_eq!(protocol.probe(b"\x00\x00\x00"), ProbeVerdict::NeedMore { at_least: 8 });
    }

    #[test]
    fn probe_rejects_an_unknown_api_key() {
        let protocol = KafkaAnyProtocol::new("kafka", handler());
        let mut prefix = api_versions_prefix(1);
        // overwrite api_key (bytes 4..6) with an api this facade never
        // recognizes.
        prefix[4..6].copy_from_slice(&999_i16.to_be_bytes());
        assert_eq!(protocol.probe(&prefix), ProbeVerdict::No);
    }

    #[test]
    fn probe_rejects_a_declared_length_below_the_smallest_valid_header() {
        let protocol = KafkaAnyProtocol::new("kafka", handler());
        let mut prefix = api_versions_prefix(1);
        prefix[0..4].copy_from_slice(&2_i32.to_be_bytes());
        assert_eq!(protocol.probe(&prefix), ProbeVerdict::No);
    }

    #[test]
    fn probe_rejects_redis_wire_bytes() {
        let protocol = KafkaAnyProtocol::new("kafka", handler());
        // RESP's `*1\r\n$4\r\nPING\r\n` — no valid Kafka frame length lives
        // at byte 0 of this.
        assert_eq!(protocol.probe(b"*1\r\n$4\r\nPING\r\n"), ProbeVerdict::No);
    }

    #[test]
    fn resolve_config_overrides_max_message_bytes_from_the_spec() {
        let base = KafkaServerConfig::default();
        let spec = serde_json::json!({ "kafka": { "max_message_bytes": 4096 } });
        let resolved = resolve_config(&base, &spec).expect("spec resolves");
        assert_eq!(resolved.max_message_bytes, 4096);
    }

    #[test]
    fn resolve_config_falls_back_to_the_base_config_with_no_spec_override() {
        let base = KafkaServerConfig::default();
        let resolved = resolve_config(&base, &Value::Null).expect("no override resolves");
        assert_eq!(resolved, base);
    }
}
