#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]
//! Integration test for S3: producer-graph schema in ProximaSettings.
//!
//! Tests that:
//!   1. TOML with a `[producers.<name>]` section parses into ProximaSettings.
//!   2. App::apply_settings resolves producer-tagged entries through
//!      `LoadContext::source_registry` (a `SourceFactory` registry, the
//!      producer-shaped sibling of the pipe factory registry) into
//!      `App::source(...)` entries.
//!   3. Registered sources are picked up unconditionally by S2's
//!      `ProducerLifecycle` driver (proxima-pipe TARGET 4 — no feature gate).
//!      The end-to-end path: TOML → ProximaSettings → App → SourceFactory →
//!      `ProducerLifecycle` → spawned source.

#![cfg(feature = "producer-graph-config")]

use proxima::settings::{ProximaSettings, RegistryEntry};
use std::collections::BTreeMap;

fn parse_toml(text: &str) -> ProximaSettings {
    let value: serde_json::Value = toml::from_str::<toml::Value>(text)
        .expect("toml parse")
        .try_into()
        .expect("toml to json");
    serde_json::from_value(value).expect("ProximaSettings from value")
}

#[test]
fn toml_with_producers_section_parses_into_proxima_settings() {
    let text = r#"
[producers.heartbeat]
type = "synth"
status = 200
body = "tick"

[producers.daily_report]
type = "synth"
status = 200
body = "report"
"#;

    let settings = parse_toml(text);
    assert_eq!(settings.producers.len(), 2);
    assert!(settings.producers.contains_key("heartbeat"));
    assert!(settings.producers.contains_key("daily_report"));

    let heartbeat = settings
        .producers
        .get("heartbeat")
        .expect("heartbeat present");
    assert_eq!(heartbeat.r#type, "synth");
    assert_eq!(
        heartbeat.spec.get("body").and_then(|v| v.as_str()),
        Some("tick")
    );
}

#[test]
fn empty_producers_field_defaults_to_empty_btreemap() {
    let text = r#"
# no producers; existing config shape stays valid
[pipes.api]
type = "synth"
status = 200
body = "ok"
"#;
    let settings = parse_toml(text);
    assert!(
        settings.producers.is_empty(),
        "producers should default to empty when section absent"
    );
    assert_eq!(settings.pipes.len(), 1);
}

#[test]
fn producers_btreemap_preserves_ordering_for_apply_settings() {
    let text = r#"
[producers.b_second]
type = "synth"
status = 200
body = "b"

[producers.a_first]
type = "synth"
status = 200
body = "a"

[producers.c_third]
type = "synth"
status = 200
body = "c"
"#;
    let settings = parse_toml(text);
    let keys: Vec<&str> = settings.producers.keys().map(String::as_str).collect();
    // BTreeMap orders by key
    assert_eq!(keys, vec!["a_first", "b_second", "c_third"]);
}

#[test]
fn fluent_builder_constructs_settings_with_producers() {
    let mut producers = BTreeMap::new();
    producers.insert(
        "tick".to_string(),
        RegistryEntry {
            r#type: "synth".to_string(),
            spec: serde_json::json!({
                "status": 200,
                "body": "tick"
            }),
        },
    );

    let settings = ProximaSettings::builder().producers(producers).build();
    assert_eq!(settings.producers.len(), 1);
    assert_eq!(settings.producers.get("tick").unwrap().r#type, "synth");
}

/// Test-only `SourceFactory`: builds a `TickSource` that increments a
/// shared counter once per `call` and returns `Ok(())` immediately —
/// enough to prove the factory dispatch + `ProducerLifecycle` spawn wiring
/// without a real timer loop.
struct TickSourceFactory {
    name: &'static str,
    ticks: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

impl proxima_core::factory::Named for TickSourceFactory {
    fn name(&self) -> &str {
        self.name
    }
}

impl proxima_primitives::pipe::SourceFactory for TickSourceFactory {
    fn build(
        &self,
        _spec: &serde_json::Value,
    ) -> Result<proxima_primitives::pipe::SourceHandle, proxima_primitives::pipe::ProximaError> {
        struct TickSource(std::sync::Arc<std::sync::atomic::AtomicUsize>);
        impl proxima_primitives::pipe::SendPipe for TickSource {
            type In = proxima_core::signal::Signal;
            type Out = ();
            type Err = proxima_primitives::pipe::ProximaError;

            fn call(
                &self,
                _cancel: proxima_core::signal::Signal,
            ) -> impl std::future::Future<Output = Result<(), proxima_primitives::pipe::ProximaError>> + Send
            {
                self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                async { Ok(()) }
            }
        }
        Ok(proxima_primitives::pipe::into_source_handle(TickSource(
            self.ticks.clone(),
        )))
    }
}

#[proxima::test]
async fn app_apply_settings_resolves_producers_into_source_map() {
    use proxima::app::App;

    let text = r#"
[producers.tick_producer]
type = "test_tick"
"#;
    let settings = parse_toml(text);

    let mut app = App::new().expect("App::new");
    let ticks = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    app.load_context()
        .source_registry
        .register(std::sync::Arc::new(TickSourceFactory {
            name: "test_tick",
            ticks,
        }))
        .expect("register test source factory");

    app.apply_settings(&settings).await.expect("apply_settings");

    // After apply_settings, the producer should be registered as a source
    // (TARGET 4) — not a pipe; producers never resolve through the pipe
    // factory registry any more.
    assert!(
        app.sources().any(|name| name == "tick_producer"),
        "producer 'tick_producer' should be registered as a source; got sources: {:?}",
        app.sources().collect::<Vec<_>>()
    );
    assert!(!app.pipes().contains_key("tick_producer"));
}

// E2E: a producer resolved through S3's schema is picked up unconditionally
// by S2's ProducerLifecycle driver (proxima-pipe TARGET 4 — no feature gate;
// this is the substrate composition test).
#[proxima::test]
async fn producer_source_picked_up_by_lifecycle_driver() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use proxima::app::App;

    let text = r#"
[producers.e2e_producer]
type = "test_tick"
"#;
    let settings = parse_toml(text);

    let ticks = Arc::new(AtomicUsize::new(0));
    let mut app = App::new().expect("App::new");
    app.load_context()
        .source_registry
        .register(Arc::new(TickSourceFactory {
            name: "test_tick",
            ticks: ticks.clone(),
        }))
        .expect("register test source factory");
    app.apply_settings(&settings).await.expect("apply_settings");

    let source = app
        .lookup_source("e2e_producer")
        .expect("e2e_producer registered as a source");

    let mut lifecycle = proxima_primitives::pipe::ProducerLifecycle::new();
    lifecycle.spawn_from_source("e2e_producer", &source);
    assert_eq!(
        lifecycle.task_count(),
        1,
        "S2 lifecycle should pick up the producer"
    );

    let report = lifecycle.shutdown(Duration::from_secs(1)).await;
    assert_eq!(report.total, 1);
    assert_eq!(report.drained, 1);
    assert_eq!(
        ticks.load(Ordering::SeqCst),
        1,
        "the source body should have run"
    );
}
