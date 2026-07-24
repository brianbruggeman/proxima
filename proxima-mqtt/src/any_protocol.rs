//! `MqttAnyProtocol` — MQTT as an [`AnyProtocol`] candidate for the open
//! universal listener (`Listener::builder().accept("mqtt")` /
//! `AnyListenProtocol`). Authored directly against `AnyProtocol` — there is
//! no standalone `MqttListenProtocol` bind+accept loop preceding this one,
//! mirroring `proxima_redis::any_protocol::RedisAnyProtocol`'s identical
//! reasoning.
//!
//! Positive-match probe: every MQTT connection opens with a `CONNECT`
//! packet, whose fixed header is the single byte `0x10` (packet type 1,
//! and the low-nibble flags are RFC-reserved to be exactly `0`). That
//! alone is a plausible false-positive magic byte, so the probe goes one
//! step further once enough bytes have arrived — it decodes the
//! "remaining length" varint with [`decode_remaining_length`] (the SAME
//! sans-IO primitive [`parse_packet`] itself uses, not a duplicate) to find
//! the protocol-name field, then checks it reads `MQTT` (v3.1.1/v5) or
//! `MQIsdp` (v3.1). [`AnyProtocol::max_prefix_bytes`] bounds that walk at
//! 13 — `1` (fixed header) + `4` (worst-case remaining-length varint) + `2`
//! (protocol-name length prefix) + `6` (`MQIsdp`, the longest legal name).
//!
//! `drive` carries its own engine (`handler`, `config`) as a struct field —
//! the same `AnyHandler`-unused asymmetry [`crate::pipe::MqttConnectionPipe`]
//! docs describe for redis. Each accepted connection builds a FRESH
//! [`MqttConnectionPipe`] carrying THIS connection's [`ConnAdmission`]
//! clone, erases it, and hands it to
//! [`proxima_listen::serve_pipe::handle_connection`] — the ONE
//! CONNECT-request/upgrade-handler driver pgwire, redis, and mqtt now
//! share.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde_json::Value;

use proxima_core::ProximaError;
use proxima_listen::admission::ConnAdmission;
use proxima_listen::any::{AnyHandler, AnyProtocol, ProbeVerdict};
use proxima_primitives::pipe::alloc_tier;
use proxima_primitives::stream::{PeerInfo, StreamConnection};

use proxima_protocols::mqtt::{ParseError, decode_remaining_length};

use crate::config::MqttServerConfig;
use crate::pipe::MqttConnectionPipe;
use crate::pipes::MqttPipeHandle;

/// `CONNECT`'s fixed header: packet type 1 in the high nibble, the
/// RFC-reserved-zero flags nibble.
const MQTT_CONNECT_FIXED_HEADER: u8 = 0x10;

/// `1` (fixed header) + `4` (worst-case remaining-length varint) + `2`
/// (protocol-name length prefix) + `6` (`MQIsdp`, the longest legal
/// protocol name).
const MAX_PROBE_BYTES: usize = 13;

/// MQTT wire candidate for the open universal listener.
///
/// ```
/// use proxima_listen::any::AnyProtocol;
/// use proxima_mqtt::{MqttAnyProtocol, MqttPipeRequest, MqttPipeReply, into_mqtt_handle};
/// use proxima_core::ProximaError;
/// use proxima_primitives::pipe::SendPipe;
///
/// struct Unimplemented; // no client dials in this doctest
/// impl SendPipe for Unimplemented {
///     type In = MqttPipeRequest;
///     type Out = MqttPipeReply;
///     type Err = ProximaError;
///     async fn call(&self, _request: MqttPipeRequest) -> Result<MqttPipeReply, ProximaError> {
///         unreachable!()
///     }
/// }
///
/// let candidate = MqttAnyProtocol::new("mqtt", into_mqtt_handle(Unimplemented));
/// assert_eq!(candidate.name(), "mqtt");
/// ```
pub struct MqttAnyProtocol {
    label: String,
    handler: MqttPipeHandle,
    config: MqttServerConfig,
    /// Built ONCE here, not per connection — `drive` installs this SAME
    /// `Arc` onto every fresh per-connection `MqttConnectionPipe` it
    /// builds (see `MqttConnectionPipe::with_broker`'s doc for why a
    /// fresh broker per connection would silently break PUBLISH/SUBSCRIBE
    /// across connections).
    broker: Arc<crate::broker::MqttBroker>,
}

impl MqttAnyProtocol {
    #[must_use]
    pub fn new(label: impl Into<String>, handler: MqttPipeHandle) -> Self {
        Self {
            label: label.into(),
            handler,
            config: MqttServerConfig::default(),
            broker: Arc::new(crate::broker::MqttBroker::new()),
        }
    }

    /// Replaces the default [`MqttServerConfig`]; an `mqtt` object in the
    /// listener spec still wins at drive time.
    #[must_use]
    pub fn with_config(mut self, config: MqttServerConfig) -> Self {
        self.config = config;
        self
    }
}

fn resolve_config(base: &MqttServerConfig, spec: &Value) -> Result<MqttServerConfig, ProximaError> {
    match spec.get("mqtt") {
        None => Ok(base.clone()),
        Some(overrides) => serde_json::from_value(overrides.clone())
            .map_err(|error| ProximaError::Config(format!("mqtt spec: {error}"))),
    }
}

impl AnyProtocol for MqttAnyProtocol {
    fn name(&self) -> &str {
        &self.label
    }

    fn max_prefix_bytes(&self) -> usize {
        MAX_PROBE_BYTES
    }

    fn probe(&self, prefix: &[u8]) -> ProbeVerdict {
        match prefix.first() {
            None => return ProbeVerdict::NeedMore { at_least: 1 },
            Some(&MQTT_CONNECT_FIXED_HEADER) => {}
            Some(_) => return ProbeVerdict::No,
        }

        let rem_len_bytes = match decode_remaining_length(&prefix[1..]) {
            Ok((_, used)) => used,
            Err(ParseError::RemainingLengthOverflow) => return ProbeVerdict::No,
            Err(_) => return ProbeVerdict::NeedMore { at_least: prefix.len() + 1 },
        };
        let header_len = 1 + rem_len_bytes;
        let name_field_end = header_len + 2;
        if prefix.len() < name_field_end {
            return ProbeVerdict::NeedMore { at_least: name_field_end };
        }
        let name_len = u16::from_be_bytes([prefix[header_len], prefix[header_len + 1]]) as usize;
        let name_end = name_field_end + name_len;
        if prefix.len() < name_end {
            return ProbeVerdict::NeedMore { at_least: name_end };
        }
        match &prefix[name_field_end..name_end] {
            b"MQTT" | b"MQIsdp" => ProbeVerdict::Match { consumed: 0 },
            _ => ProbeVerdict::No,
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
            let connection_pipe = MqttConnectionPipe::new(
                self.label.clone(),
                self.handler.clone(),
                Arc::new(config),
            )
            .with_broker(Arc::clone(&self.broker))
            .with_admission(admission.clone());
            let pipe = alloc_tier::into_handle(connection_pipe);
            proxima_listen::serve_pipe::handle_connection(stream, pipe).await
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use proxima_primitives::pipe::request::Response;
    use proxima_protocols::mqtt::MqttReply;
    use proxima_protocols::mqtt::encode::encode_connect;

    struct AcceptAllPipe;

    impl proxima_primitives::pipe::SendPipe for AcceptAllPipe {
        type In = crate::pipes::MqttPipeRequest;
        type Out = crate::pipes::MqttPipeReply;
        type Err = ProximaError;

        async fn call(&self, _request: Self::In) -> Result<Self::Out, ProximaError> {
            Ok(Response::typed(
                200,
                MqttReply::ConnAck { session_present: false, return_code: 0 },
            ))
        }
    }

    fn handler() -> MqttPipeHandle {
        crate::pipes::into_mqtt_handle(AcceptAllPipe)
    }

    #[test]
    fn probe_matches_a_real_connect_packet() {
        let protocol = MqttAnyProtocol::new("mqtt", handler());
        let mut wire = Vec::new();
        encode_connect(b"client-1", true, 30, None, None, &mut wire);
        assert_eq!(protocol.probe(&wire), ProbeVerdict::Match { consumed: 0 });
    }

    #[test]
    fn probe_rejects_a_foreign_first_byte() {
        let protocol = MqttAnyProtocol::new("mqtt", handler());
        assert_eq!(protocol.probe(b"*1\r\n"), ProbeVerdict::No);
    }

    #[test]
    fn probe_rejects_a_connect_shaped_header_with_a_foreign_protocol_name() {
        let protocol = MqttAnyProtocol::new("mqtt", handler());
        // fixed header (CONNECT) + rem-len byte + a 4-byte name field that
        // is not "MQTT".
        let wire = [0x10, 0x06, 0x00, 0x04, b'N', b'O', b'P', b'E'];
        assert_eq!(protocol.probe(&wire), ProbeVerdict::No);
    }

    #[test]
    fn probe_asks_for_more_bytes_on_a_short_prefix() {
        let protocol = MqttAnyProtocol::new("mqtt", handler());
        assert_eq!(protocol.probe(b""), ProbeVerdict::NeedMore { at_least: 1 });
        assert_eq!(
            protocol.probe(&[0x10]),
            ProbeVerdict::NeedMore { at_least: 2 }
        );
    }

    #[test]
    fn probe_matches_the_legacy_v3_1_protocol_name() {
        let protocol = MqttAnyProtocol::new("mqtt", handler());
        // fixed header + rem-len + "MQIsdp" (6 bytes, v3.1's protocol name).
        let wire = [0x10, 0x08, 0x00, 0x06, b'M', b'Q', b'I', b's', b'd', b'p'];
        assert_eq!(protocol.probe(&wire), ProbeVerdict::Match { consumed: 0 });
    }
}
