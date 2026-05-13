//! Cassette provenance: a single `ProtocolEvent::Custom` event stamped at the
//! head of a recording. Old readers ignore the unknown kind, so cassettes with
//! and without the stamp stay interchangeable at the wire level; policy layers
//! (staleness gates) treat an absent stamp as "age unknown".

use alloc::string::{String, ToString};

use serde_json::Value;

use proxima_core::ProximaError;

pub const CASSETTE_META_KIND: &str = "cassette-meta";

/// Provenance of one recorded cassette.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CassetteMeta {
    /// Wall-clock unix ms when the recording was made. The only
    /// intentionally nondeterministic value in a cassette — every replayed
    /// event keeps zeroed timestamps.
    pub recorded_at_ms: u64,
    /// Human-readable recorder identity, e.g. `proxima 0.1.0`.
    pub recorder: String,
    /// Whether request bodies were captured as `RequestChunk` events.
    /// Cassettes recorded before body capture existed cannot serve
    /// body-keyed replay and must be re-recorded for it.
    pub request_bodies: bool,
}

impl CassetteMeta {
    #[must_use]
    pub fn to_payload(&self) -> Value {
        serde_json::json!({
            "recorded_at_ms": self.recorded_at_ms,
            "recorder": self.recorder,
            "request_bodies": self.request_bodies,
        })
    }

    /// Parse the stamp back out of a `Custom` payload.
    ///
    /// # Errors
    /// Returns `ProximaError::Record` when a field is missing or mistyped —
    /// a malformed stamp means the cassette was hand-edited, which is an
    /// integrity failure, not an absent stamp.
    pub fn from_payload(payload: &Value) -> Result<Self, ProximaError> {
        let recorded_at_ms = payload
            .get("recorded_at_ms")
            .and_then(Value::as_u64)
            .ok_or_else(|| malformed("recorded_at_ms"))?;
        let recorder = payload
            .get("recorder")
            .and_then(Value::as_str)
            .ok_or_else(|| malformed("recorder"))?
            .to_string();
        let request_bodies = payload
            .get("request_bodies")
            .and_then(Value::as_bool)
            .ok_or_else(|| malformed("request_bodies"))?;
        Ok(Self {
            recorded_at_ms,
            recorder,
            request_bodies,
        })
    }
}

fn malformed(field: &str) -> ProximaError {
    ProximaError::Record(alloc::format!(
        "malformed cassette-meta payload: missing or mistyped `{field}`"
    ))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn sample() -> CassetteMeta {
        CassetteMeta {
            recorded_at_ms: 1_750_000_000_000,
            recorder: "proxima 0.1.0".to_string(),
            request_bodies: true,
        }
    }

    #[test]
    fn payload_round_trips() {
        let original = sample();
        let restored = CassetteMeta::from_payload(&original.to_payload()).expect("round trip");
        assert_eq!(restored, original);
    }

    #[test]
    fn missing_field_is_a_typed_error() {
        let mut payload = sample().to_payload();
        payload
            .as_object_mut()
            .expect("object payload")
            .remove("recorder");
        let outcome = CassetteMeta::from_payload(&payload);
        assert!(matches!(outcome, Err(ProximaError::Record(_))));
    }

    #[test]
    fn mistyped_field_is_a_typed_error() {
        let mut payload = sample().to_payload();
        payload
            .as_object_mut()
            .expect("object payload")
            .insert("recorded_at_ms".to_string(), Value::from("yesterday"));
        let outcome = CassetteMeta::from_payload(&payload);
        assert!(matches!(outcome, Err(ProximaError::Record(_))));
    }
}
