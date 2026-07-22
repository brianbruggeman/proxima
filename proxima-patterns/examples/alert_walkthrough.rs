#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]
//! Worked example per principle 11 (sans-IO state-machine walkthrough)
//! and `/algorithm-development` (paper-first algorithm development).
//!
//! Constructs an [`AlertEvent`] by hand, encodes it via postcard, prints
//! the byte breakdown, decodes back, and asserts byte-exact round-trip.
//!
//! Also exercises the documented `as_json_shape()` mapping helper. Run with:
//!
//! ```text
//! cargo run --example alert_walkthrough -p proxima-notify-proto \
//!     --features std,json-shape
//! ```

use proxima_patterns::alert::event::{
    AlertEvent, AlertId, KindString, LabelKey, LabelMap, LabelValue, Payload, Severity,
    decode_alert, encode_alert, sized,
};

fn main() {
    println!("proxima-notify-proto alert walkthrough\n");

    // Step 1: construct an AlertEvent with known field values.
    let mut labels = LabelMap::new();
    let _ = labels.insert(
        LabelKey::try_from("host").unwrap(),
        LabelValue::try_from("proxima-node-1").unwrap(),
    );
    let _ = labels.insert(
        LabelKey::try_from("source").unwrap(),
        LabelValue::try_from("scheduled_trigger").unwrap(),
    );

    let event = AlertEvent {
        id: AlertId::from_bytes([
            0x01, 0x92, 0x3B, 0x5C, 0x7E, 0x8F, 0x4A, 0x6D, 0x9E, 0x0F, 0x01, 0x23, 0x45, 0x67,
            0x89, 0xAB,
        ]),
        severity: Severity::Warn,
        kind: KindString::try_from("heartbeat").unwrap(),
        labels,
        payload: Payload::from_slice(&[
            0x04, 0x00, 0x02, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ])
        .unwrap(),
        fired_at_micros: 1_748_284_800_000_000,
    };

    println!("Constructed AlertEvent:");
    println!("  id              = {} (ULID)", event.id.0);
    println!(
        "  severity        = {:?} ({})",
        event.severity,
        event.severity.as_str()
    );
    println!("  kind            = {:?}", event.kind.as_str());
    println!("  labels          = {} entries", event.labels.len());
    for (key, value) in event.labels.iter() {
        println!("    {} = {}", key.as_str(), value.as_str());
    }
    println!("  payload         = {} bytes", event.payload.len());
    println!("  fired_at_micros = {}", event.fired_at_micros);
    println!();
    println!("Cap budget (from build.rs, principle 12):");
    println!("  ALERT_LABEL_KEY_MAX = {}", sized::ALERT_LABEL_KEY_MAX);
    println!("  ALERT_LABEL_VAL_MAX = {}", sized::ALERT_LABEL_VAL_MAX);
    println!("  ALERT_LABELS_MAX    = {}", sized::ALERT_LABELS_MAX);
    println!("  ALERT_KIND_MAX      = {}", sized::ALERT_KIND_MAX);
    println!("  ALERT_PAYLOAD_MAX   = {}", sized::ALERT_PAYLOAD_MAX);
    println!();

    // Step 2: encode via postcard into a fixed-size stack buffer.
    // Total bytes ≤ sum of caps; the buffer is way overshot but cheap.
    let mut buffer = [0_u8; 8192];
    let bytes_written = encode_alert(&event, &mut buffer).expect("encode");
    println!("postcard encoded {bytes_written} bytes:");
    let hex: String = buffer[..bytes_written]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<Vec<_>>()
        .join(" ");
    println!("  {hex}");
    println!();

    // Step 3: decode back and assert byte-exact equality.
    let decoded = decode_alert(&buffer[..bytes_written]).expect("decode");
    assert_eq!(event, decoded, "AlertEvent round-trip must be byte-exact");
    println!("round-trip OK (decoded == original)");
    println!();

    // Step 4: render the documented JSON shape (parity-test bridge).
    #[cfg(feature = "json-shape")]
    {
        use proxima_patterns::alert::event::json_shape::alert_event_to_json;
        let json_shape = alert_event_to_json(&event);
        println!("Documented JSON shape:");
        println!(
            "{}",
            serde_json::to_string_pretty(&json_shape).expect("json pretty")
        );
    }
}
