//! Round-trip CI gate against drift between
//! [`crate::scenarios::spec`] serde structs and the schema entries in
//! [`crate::schema::scenario`]. If anyone adds or removes a field on a
//! scenario type without touching the schema, one of the validations
//! below fails.
//!
//! Validators rely on the proxima-native `Schema::validate`. The
//! `proxima describe --format json-schema` JSON-Schema emission rides
//! the same IR; the IR is the source of truth.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

use std::path::PathBuf;

use proxima::scenarios::{DurationSpec, Expectation, ProfileStep};
use proxima::{LoadContext, Scenario, WorkloadSpec};

fn workspace_root() -> PathBuf {
    // post-Phase-A: umbrella package IS at workspace root.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn registered_scenario_schema() -> (proxima::Schema, std::sync::Arc<proxima::SchemaRegistry>) {
    let context = LoadContext::with_default_registry().expect("load context");
    let schema = context
        .schemas
        .get("scenario")
        .expect("LoadContext::with_default_registry registers `scenario`");
    (schema, context.schemas.clone())
}

#[test]
fn smoke_scenario_toml_round_trips_through_scenario_schema() {
    let scenario_path = workspace_root()
        .join("scenarios")
        .join("open-loop-smoke")
        .join("scenario.toml");
    let scenario = Scenario::from_toml_file(&scenario_path).expect("parse smoke scenario");
    let as_value = serde_json::to_value(&scenario).expect("serialize scenario");

    let (schema, resolver) = registered_scenario_schema();
    schema
        .validate(&as_value, resolver.as_ref())
        .unwrap_or_else(|err| {
            panic!(
                "open-loop-smoke/scenario.toml fails registered `scenario` schema: \
                 drift between scenarios::spec and schema::scenario? {err}\n\
                 value={as_value}"
            )
        });
}

#[test]
fn programmatic_open_loop_scenario_validates_against_schema() {
    let scenario = Scenario::new_programmatic(
        WorkloadSpec::new_open_loop("echo", 100, DurationSpec::from_secs(5)).with_concurrency(32),
    )
    .with_pipe("echo", serde_json::json!({ "synth": { "status": 200 } }))
    .with_expectation(Expectation::SuccessRateGe { ratio: 0.99 });
    let as_value = serde_json::to_value(&scenario).expect("serialize scenario");

    let (schema, resolver) = registered_scenario_schema();
    schema
        .validate(&as_value, resolver.as_ref())
        .expect("programmatic open-loop scenario must validate");
}

#[test]
fn closed_loop_scenario_validates_against_schema() {
    let scenario = Scenario::new_programmatic(WorkloadSpec::new("echo", 25).with_concurrency(4))
        .with_pipe("echo", serde_json::json!({ "synth": { "status": 200 } }));
    let as_value = serde_json::to_value(&scenario).expect("serialize scenario");

    let (schema, resolver) = registered_scenario_schema();
    schema
        .validate(&as_value, resolver.as_ref())
        .expect("closed-loop scenario must validate");
}

#[test]
fn profile_step_validates_against_schema() {
    let step = ProfileStep {
        rate: 100,
        duration: DurationSpec::from_secs(30),
    };
    let as_value = serde_json::to_value(&step).expect("serialize step");

    let context = LoadContext::with_default_registry().expect("load context");
    let schema = context
        .schemas
        .get("profile_step")
        .expect("profile_step schema registered");
    schema
        .validate(&as_value, context.schemas.as_ref())
        .expect("profile step must validate");
}

#[test]
fn proxima_describe_format_json_schema_emits_scenario_schema() {
    // The CLI `proxima describe --format json-schema` rides the same
    // SchemaRegistry. Verify the named scenario schema emits to a
    // non-trivial JSON Schema (smoke check that downstream binding
    // generators have something to consume).
    let context = LoadContext::with_default_registry().expect("load context");
    let schema = context.schemas.get("scenario").expect("scenario schema");
    let refs = context.schemas.snapshot();
    let emitted = proxima::schema::emit::emit_json_schema(&schema, &refs);
    // smoke: must declare a top-level "type" and reference at least
    // one of the registered child schemas.
    assert!(
        emitted.get("type").is_some() || emitted.get("$ref").is_some(),
        "emitted scenario schema must carry a `type` or `$ref` field at the root: {emitted}"
    );
}
