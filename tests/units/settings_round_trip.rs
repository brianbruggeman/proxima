//! TOML ⇄ ProximaSettings round-trip identity. The load-bearing
//! invariant for the fluent ⇄ Settings story — if these diverge, the
//! whole Stage B + E premise is broken.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

use std::collections::BTreeMap;

use proxima::settings::{HttpTuning, ProximaSettings, RegistryEntry, ZstdTuning};
use serde_json::json;

fn fixture_settings() -> ProximaSettings {
    let mut listeners = BTreeMap::new();
    listeners.insert(
        "public".into(),
        RegistryEntry {
            r#type: "http".into(),
            spec: json!({ "addr": "0.0.0.0:8443", "tls": { "cert": "cert.pem", "key": "key.pem" } }),
        },
    );
    listeners.insert(
        "admin".into(),
        RegistryEntry {
            r#type: "http".into(),
            spec: json!({ "path": "/var/run/proxima.sock", "mode": 0o600 }),
        },
    );

    let mut upstreams = BTreeMap::new();
    upstreams.insert(
        "backend".into(),
        RegistryEntry {
            r#type: "http".into(),
            spec: json!({ "url": "https://backend.internal:8443", "timeout_ms": 5000 }),
        },
    );

    let mut middlewares = BTreeMap::new();
    middlewares.insert(
        "auth".into(),
        RegistryEntry {
            r#type: "bearer".into(),
            spec: json!({ "allow": ["t-1", "t-2"] }),
        },
    );
    middlewares.insert(
        "rl".into(),
        RegistryEntry {
            r#type: "token_bucket".into(),
            spec: json!({ "capacity": 100, "refill_per_sec": 50 }),
        },
    );

    let mut pipes = BTreeMap::new();
    pipes.insert(
        "api".into(),
        RegistryEntry {
            r#type: "chain".into(),
            spec: json!({
                "mount": "/api/{*path}",
                "methods": ["GET", "POST"],
                "chain": ["auth", "rl", "backend"],
            }),
        },
    );

    ProximaSettings::builder()
        .listeners(listeners)
        .upstreams(upstreams)
        .middlewares(middlewares)
        .pipes(pipes)
        .http(HttpTuning::builder().response_buffer_bytes(32_768).build())
        .zstd(ZstdTuning::builder().compression_level(9).build())
        .build()
}

#[test]
fn settings_round_trip_via_toml_is_identity() {
    let original = fixture_settings();
    let toml_text = toml::to_string(&original).expect("encode toml");
    let restored: ProximaSettings = toml::from_str(&toml_text).expect("decode toml");

    // We can't compare ProximaSettings via PartialEq (none derived);
    // compare the JSON projection instead — that's what the loader
    // sees anyway, and any meaningful divergence shows up there.
    let original_json = serde_json::to_value(&original).expect("encode json");
    let restored_json = serde_json::to_value(&restored).expect("encode json");
    assert_eq!(original_json, restored_json, "TOML round-trip diverged");
}

#[test]
fn settings_round_trip_via_json_is_identity() {
    let original = fixture_settings();
    let json_text = serde_json::to_string(&original).expect("encode json");
    let restored: ProximaSettings = serde_json::from_str(&json_text).expect("decode json");
    let original_json = serde_json::to_value(&original).expect("encode original");
    let restored_json = serde_json::to_value(&restored).expect("encode restored");
    assert_eq!(original_json, restored_json, "JSON round-trip diverged");
}

#[test]
fn default_settings_serialize_and_round_trip() {
    let original = ProximaSettings::builder().build();
    let toml_text = toml::to_string(&original).expect("encode toml");
    let restored: ProximaSettings = toml::from_str(&toml_text).expect("decode toml from defaults");
    let restored_json = serde_json::to_value(&restored).expect("encode");
    let original_json = serde_json::to_value(&original).expect("encode");
    assert_eq!(original_json, restored_json);
}

#[test]
fn tuning_env_overrides_apply_when_set() {
    temp_env::with_vars(
        [
            ("PROXIMA_HTTP_RESPONSE_BUFFER_BYTES", Some("65536")),
            ("PROXIMA_ZSTD_COMPRESSION_LEVEL", Some("19")),
        ],
        || {
            use conflaguration::Settings as _;
            let settings: ProximaSettings = ProximaSettings::from_env().expect("env load");
            assert_eq!(settings.http.response_buffer_bytes, 65_536);
            assert_eq!(settings.zstd.compression_level, 19);
        },
    );
}
