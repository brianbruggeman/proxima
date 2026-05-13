//! Round-trip tests for proxima-notify-proto: encode → decode → equality.
//!
//! Per principle 9 (real-world data in tests), AlertEvent fixtures are
//! built from realistic field values — `kind = "heartbeat"`, real label
//! shapes, ms-precision timestamps — not `b"AAAA"` stubs.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

use proxima_patterns::alert::event::{
    AgentId, AlertEvent, AlertId, AnswerString, ContextBytes, GuidanceAnswer, GuidanceQuestion,
    GuidanceRequestId, KindString, LabelKey, LabelMap, LabelValue, Payload, QuestionString,
    ResponderString, Severity, decode_alert, decode_guidance_answer, decode_guidance_question,
    encode_alert, encode_guidance_answer, encode_guidance_question,
};

fn sample_alert_event() -> AlertEvent {
    let mut labels = LabelMap::new();
    labels
        .insert(
            LabelKey::try_from("host").expect("host fits"),
            LabelValue::try_from("proxima-node-1").expect("value fits"),
        )
        .expect("insert host");
    labels
        .insert(
            LabelKey::try_from("source").expect("source fits"),
            LabelValue::try_from("scheduled_trigger").expect("value fits"),
        )
        .expect("insert source");

    AlertEvent {
        id: AlertId::from_bytes([
            0x01, 0x92, 0x3B, 0x5C, 0x7E, 0x8F, 0x4A, 0x6D, 0x9E, 0x0F, 0x01, 0x23, 0x45, 0x67,
            0x89, 0xAB,
        ]),
        severity: Severity::Warn,
        kind: KindString::try_from("heartbeat").expect("kind fits"),
        labels,
        payload: Payload::from_slice(&[
            0x04, 0x00, 0x02, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ])
        .expect("payload fits"),
        fired_at_micros: 1_748_284_800_000_000,
    }
}

fn sample_guidance_question() -> GuidanceQuestion {
    GuidanceQuestion {
        id: GuidanceRequestId::from_bytes([0xAB; 16]),
        agent_id: AgentId::from_bytes([0xCD; 16]),
        parent_id: Some(AgentId::from_bytes([0xEF; 16])),
        question: QuestionString::try_from(
            "Should I push the refactor or land the parser change first?",
        )
        .expect("question fits"),
        context: ContextBytes::from_slice(&[0xDE, 0xAD, 0xBE, 0xEF]).expect("context fits"),
        asked_at_micros: 1_748_284_900_000_000,
        timeout_micros: 300_000_000,
    }
}

fn sample_guidance_answer() -> GuidanceAnswer {
    GuidanceAnswer {
        request_id: GuidanceRequestId::from_bytes([0xAB; 16]),
        content: AnswerString::try_from("Parser change first, then refactor.")
            .expect("answer fits"),
        responder: ResponderString::try_from("stdin").expect("responder fits"),
        responded_at_micros: 1_748_284_950_000_000,
    }
}

#[test]
fn alert_event_postcard_roundtrip_is_byte_exact() {
    let event = sample_alert_event();
    let mut buffer = [0_u8; 8192];
    let bytes_written = encode_alert(&event, &mut buffer).expect("encode");
    assert!(bytes_written > 0, "should write at least one byte");
    let decoded = decode_alert(&buffer[..bytes_written]).expect("decode");
    assert_eq!(event, decoded);
}

#[test]
fn alert_event_encoded_size_is_below_cap_for_typical_inputs() {
    // Sanity check: a "typical" event with a few labels and a small payload
    // fits in under 1 KB after postcard encoding. The cap (event-size +
    // label-size + payload-size sums) is much higher; this assertion
    // documents the on-the-wire footprint a producer should plan for.
    let event = sample_alert_event();
    let mut buffer = [0_u8; 8192];
    let bytes_written = encode_alert(&event, &mut buffer).expect("encode");
    assert!(
        bytes_written < 256,
        "typical alert event should encode in <256 bytes; got {bytes_written}"
    );
}

#[test]
fn guidance_question_postcard_roundtrip_preserves_all_fields() {
    let question = sample_guidance_question();
    let mut buffer = [0_u8; 16_384];
    let bytes_written = encode_guidance_question(&question, &mut buffer).expect("encode");
    let decoded = decode_guidance_question(&buffer[..bytes_written]).expect("decode");
    assert_eq!(question, decoded);
}

#[test]
fn guidance_answer_postcard_roundtrip_preserves_all_fields() {
    let answer = sample_guidance_answer();
    let mut buffer = [0_u8; 16_384];
    let bytes_written = encode_guidance_answer(&answer, &mut buffer).expect("encode");
    let decoded = decode_guidance_answer(&buffer[..bytes_written]).expect("decode");
    assert_eq!(answer, decoded);
}

#[test]
fn encode_into_undersized_buffer_returns_encode_error_not_panic() {
    let event = sample_alert_event();
    let mut tiny = [0_u8; 4];
    let result = encode_alert(&event, &mut tiny);
    assert!(result.is_err(), "encoding into a 4-byte buffer must fail");
}

#[test]
fn decode_from_corrupted_bytes_returns_decode_error_not_panic() {
    let result = decode_alert(b"\x00\x01\x02\xFF\xFF\xFF");
    assert!(
        result.is_err(),
        "decoding garbage bytes must return Err, not panic"
    );
}

#[test]
fn severity_numeric_values_match_proxima_telemetry_level_convention() {
    assert_eq!(Severity::Trace.as_u8(), 1);
    assert_eq!(Severity::Debug.as_u8(), 5);
    assert_eq!(Severity::Info.as_u8(), 9);
    assert_eq!(Severity::Warn.as_u8(), 13);
    assert_eq!(Severity::Error.as_u8(), 17);
    assert_eq!(Severity::Fatal.as_u8(), 21);
}

#[test]
fn severity_lowercase_names_match_documented_json_schema() {
    assert_eq!(Severity::Trace.as_str(), "trace");
    assert_eq!(Severity::Debug.as_str(), "debug");
    assert_eq!(Severity::Info.as_str(), "info");
    assert_eq!(Severity::Warn.as_str(), "warn");
    assert_eq!(Severity::Error.as_str(), "error");
    assert_eq!(Severity::Fatal.as_str(), "fatal");
}

#[cfg(feature = "json-shape")]
#[test]
fn alert_event_as_json_shape_matches_documented_schema_fixture() {
    use proxima_patterns::alert::event::json_shape::alert_event_to_json;
    use serde_json::json;

    let event = sample_alert_event();
    let actual = alert_event_to_json(&event);
    // ULID rendering of bytes [0x01, 0x92, 0x3B, 0x5C, 0x7E, 0x8F, 0x4A,
    // 0x6D, 0x9E, 0x0F, 0x01, 0x23, 0x45, 0x67, 0x89, 0xAB] per the
    // `ulid` crate. The bit ordering is u128 big-endian; the rendering
    // is Crockford base32 of the 128-bit value. Per principle 14 the
    // incumbent (`ulid` crate) is the source of truth; the schema doc
    // is the projection.
    let expected = json!({
        "id": "01J8XNRZMF99PSW3R14D2PF2DB",
        "severity": "warn",
        "kind": "heartbeat",
        "labels": {
            "host": "proxima-node-1",
            "source": "scheduled_trigger"
        },
        "payload_bytes_base64": "BAACAAEAAAAAAAAAAA",
        "fired_at_micros": 1_748_284_800_000_000_i64,
    });
    assert_eq!(
        actual, expected,
        "AlertEvent JSON shape must match docs/proxima-notify/ALERT_EVENT_SCHEMA.md fixture"
    );
}
