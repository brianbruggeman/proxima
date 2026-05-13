//! Phase 5: end-to-end TOML -> Settings -> App materialization.
//!
//! Writes a TOML fixture; loads it via `ProximaSettings::from_path`;
//! materializes upstreams + composed pipes into a fresh App;
//! verifies the pipes are registered with the expected names.
//!
//! Listeners are not materialized here (deferred — needs richer
//! RunConfig). Middlewares are referenced by name from pipe
//! chains and registered as composed pipes, not standalone.

// `flavor = "multi_thread"` / `worker_threads` are tokio-only #[proxima::test]
// knobs (see proxima-macros/src/main_attr.rs) — needs `tokio`.
#![cfg(feature = "tokio")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

use proxima::App;
use proxima::settings::ProximaSettings;
use std::io::Write;
use tempfile::NamedTempFile;

const FIXTURE_TOML: &str = r#"
[upstreams.echo]
type = "synth"
status = 200
body = "echo-v1"

[upstreams.health]
type = "synth"
status = 200
body = "ok"

[middlewares.auth]
type = "auth"
allow = ["t-1", "t-2"]

[pipes.public]
type = "synth"
mount = "/echo/{*path}"
chain = ["auth", "echo"]

[pipes.healthz]
type = "synth"
mount = "/healthz"
chain = ["health"]
"#;

fn write_fixture() -> NamedTempFile {
    let mut file = NamedTempFile::with_suffix(".toml").expect("tempfile");
    file.write_all(FIXTURE_TOML.as_bytes()).expect("write");
    file.flush().expect("flush");
    file
}

#[proxima::test(flavor = "multi_thread", worker_threads = 2)]
async fn settings_load_from_toml_then_apply_to_app() {
    let file = write_fixture();
    let settings = ProximaSettings::from_path(file.path()).expect("load settings");

    // The map-keyed registries deserialized.
    assert!(settings.upstreams.contains_key("echo"));
    assert!(settings.upstreams.contains_key("health"));
    assert!(settings.middlewares.contains_key("auth"));
    assert!(settings.pipes.contains_key("public"));
    assert!(settings.pipes.contains_key("healthz"));

    // Materialize into a fresh App. The upstreams register as named
    // Pipes; the pipes register as composed Pipes with the
    // chained middlewares wrapping the leaf upstream.
    let mut app = App::new().expect("app");
    app.apply_settings(&settings).await.expect("apply");

    // Every named entry should be a registered Pipe.
    assert!(app.pipes().contains_key("echo"));
    assert!(app.pipes().contains_key("health"));
    assert!(app.pipes().contains_key("public"));
    assert!(app.pipes().contains_key("healthz"));
}

#[proxima::test(flavor = "multi_thread", worker_threads = 2)]
async fn settings_round_trip_toml_via_load_from_path() {
    // The fixture written above round-trips to Settings; serializing
    // those Settings back to TOML and reloading produces identical
    // JSON projection. This complements the in-memory round-trip
    // tests in settings_round_trip.rs with a real filesystem path.
    let file = write_fixture();
    let original = ProximaSettings::from_path(file.path()).expect("load 1");

    // Re-emit + re-load: another tempfile, write the re-serialized
    // TOML, load it back.
    let toml_text = toml::to_string(&original).expect("encode");
    let mut file2 = NamedTempFile::with_suffix(".toml").expect("tempfile 2");
    file2.write_all(toml_text.as_bytes()).expect("write 2");
    file2.flush().expect("flush 2");
    let reloaded = ProximaSettings::from_path(file2.path()).expect("load 2");

    let original_json = serde_json::to_value(&original).expect("json 1");
    let reloaded_json = serde_json::to_value(&reloaded).expect("json 2");
    assert_eq!(original_json, reloaded_json, "round-trip diverged");
}
