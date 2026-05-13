#![allow(clippy::expect_used)]
//! Config / API parity (workspace principle 4): the env-loaded config,
//! the fluent builder, and the serde wire form must reach identical
//! state.

use conflaguration::{Settings, Validate};
use proxima_pgwire::{AuthConfig, PgServerConfig};

#[test]
fn default_equals_builder_with_no_setters() {
    let built = PgServerConfig::builder().build();
    assert_eq!(PgServerConfig::default(), built);
}

#[test]
fn env_loader_and_builder_reach_identical_state() {
    let from_env = temp_env::with_vars(
        [
            ("PGWIRE_READ_BUFFER_BYTES", Some("16384")),
            ("PGWIRE_WRITE_HIGH_WATER_BYTES", Some("131072")),
            ("PGWIRE_MAX_MESSAGE_BYTES", Some("33554432")),
            ("PGWIRE_MAX_STATEMENTS", Some("512")),
            ("PGWIRE_MAX_PORTALS", Some("128")),
        ],
        || PgServerConfig::from_env().expect("env loader must accept valid overrides"),
    );
    let built = PgServerConfig::builder()
        .read_buffer_bytes(16384)
        .write_high_water_bytes(131_072)
        .max_message_bytes(33_554_432)
        .max_statements(512)
        .max_portals(128)
        .build();
    assert_eq!(from_env, built);
}

#[test]
fn serde_json_round_trip_preserves_state() {
    let original = PgServerConfig::builder()
        .max_statements(7)
        .auth(AuthConfig::Cleartext {
            username: "svc".into(),
            password: "hunter2".into(),
        })
        .build();
    let wire = serde_json::to_value(&original).expect("config must serialize");
    let parsed: PgServerConfig =
        serde_json::from_value(wire).expect("config must deserialize from its own wire form");
    assert_eq!(original, parsed);
}

#[test]
fn partial_spec_object_fills_remaining_fields_with_defaults() {
    let parsed: PgServerConfig = serde_json::from_value(serde_json::json!({ "max_portals": 8 }))
        .expect("partial config must deserialize");
    assert_eq!(parsed.max_portals, 8);
    assert_eq!(
        parsed.read_buffer_bytes,
        PgServerConfig::default().read_buffer_bytes
    );
}

#[test]
fn validate_rejects_undersized_read_buffer() {
    let config = PgServerConfig::builder().read_buffer_bytes(64).build();
    assert!(config.validate().is_err());
}

#[test]
fn validate_rejects_max_message_below_read_buffer() {
    let config = PgServerConfig::builder()
        .read_buffer_bytes(8192)
        .max_message_bytes(4096)
        .build();
    assert!(config.validate().is_err());
}

#[test]
fn validate_rejects_nul_in_reported_parameters() {
    let config = PgServerConfig::builder()
        .parameters(vec![("server_version".into(), "16.0\0evil".into())])
        .build();
    assert!(config.validate().is_err());
}

#[test]
fn validate_accepts_defaults() {
    PgServerConfig::default()
        .validate()
        .expect("defaults must validate");
}

#[test]
fn build_auth_trust_by_default_and_cleartext_requires_username() {
    let trust = PgServerConfig::default();
    assert!(matches!(
        trust.build_auth().expect("trust auth must build"),
        proxima_pgwire::PgAuth::Trust
    ));
    let broken = PgServerConfig::builder()
        .auth(AuthConfig::Cleartext {
            username: String::new(),
            password: "x".into(),
        })
        .build();
    assert!(broken.build_auth().is_err());
}
