#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

use proxima_protocols::http3_codec::settings::{Settings, SettingsError};
use proxima_protocols::quic::varint;

const NATIVE_LISTENER_INITIAL_UNI_CREDIT: usize = 65_536;
const UNIQUE_UNKNOWN_SETTINGS: u64 = 16_384;

fn mounted_path_adversarial_settings_payload() -> Vec<u8> {
    let mut payload = Vec::new();
    let mut scratch = [0_u8; 8];
    for offset in 0..UNIQUE_UNKNOWN_SETTINGS {
        let identifier = 128 + offset;
        let identifier_length =
            varint::encode(identifier, &mut scratch).expect("encode unknown setting identifier");
        payload.extend_from_slice(&scratch[..identifier_length]);
        let value_length = varint::encode(0, &mut scratch).expect("encode setting value");
        payload.extend_from_slice(&scratch[..value_length]);
    }
    payload
}

#[test]
fn mounted_path_accepts_unique_unknown_settings_within_initial_uni_credit() {
    let payload = mounted_path_adversarial_settings_payload();

    assert_eq!(payload.len(), 49_408, "fixture size must remain explicit");
    assert!(
        payload.len() < NATIVE_LISTENER_INITIAL_UNI_CREDIT,
        "fixture must fit the mounted native listener control stream credit"
    );

    Settings::default()
        .apply_payload(&payload)
        .expect("unique unknown SETTINGS identifiers are permitted");
}

#[test]
fn mounted_path_rejects_duplicate_unknown_setting_within_initial_uni_credit() {
    let mut payload = mounted_path_adversarial_settings_payload();
    let mut scratch = [0_u8; 8];
    let identifier_length =
        varint::encode(128, &mut scratch).expect("encode duplicate setting identifier");
    payload.extend_from_slice(&scratch[..identifier_length]);
    let value_length = varint::encode(0, &mut scratch).expect("encode duplicate setting value");
    payload.extend_from_slice(&scratch[..value_length]);

    assert!(payload.len() < NATIVE_LISTENER_INITIAL_UNI_CREDIT);

    let error = Settings::default()
        .apply_payload(&payload)
        .expect_err("duplicate SETTINGS identifier must be rejected");
    assert!(matches!(error, SettingsError::DuplicateId { id: 128 }));
}
