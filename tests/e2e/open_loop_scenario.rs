//! Integration tests for the open-loop scenario driver (`proxima load`
//! programmatic path). Covers:
//!
//! - `proxima/scenarios/open-loop-smoke/scenario.toml` parses, runs,
//!   passes its expectations, and emits per-second windows.
//! - The programmatic builder API (`WorkloadSpec::new_open_loop`)
//!   produces an equivalent run when invoked with the same shape.
//! - `ScenarioReport.windows` is empty for closed-loop scenarios
//!   (regression guard on the dispatch wiring).

// the open-loop driver is tokio-coupled by construction: it runs inside a
// `tokio::task::LocalSet` with `spawn_local` and awaits `Runtime::timer_at`
// on the calling thread, which PrimeRuntime (the default) cannot serve. these
// integration tests therefore require the `runtime-tokio` opt-out. run with
// `--features runtime-tokio`. (see scenarios/orchestrator.rs open_loop_runtime)
#![cfg(feature = "runtime-tokio")]
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

use proxima::scenarios::{DurationSpec, WorkloadMode};
use proxima::{Expectation, LoadContext, Scenario, WorkloadSpec, run_scenario};

fn workspace_root() -> PathBuf {
    // post-Phase-A: umbrella package IS at workspace root.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

#[proxima::test(runtime = "tokio")]
async fn open_loop_smoke_scenario_file_passes_via_run_scenario() {
    let scenario_path = workspace_root()
        .join("scenarios")
        .join("open-loop-smoke")
        .join("scenario.toml");
    let scenario =
        Scenario::from_toml_file(&scenario_path).expect("parse open-loop-smoke/scenario.toml");
    assert_eq!(
        scenario.workload.mode().expect("mode"),
        WorkloadMode::OpenLoop,
        "smoke scenario must classify as open-loop"
    );

    let context = LoadContext::with_default_registry().expect("load context");
    let report = run_scenario(&scenario, &context).await.expect("run smoke");

    assert!(
        report.passed(),
        "smoke scenario must pass; failed_expectations={:?}",
        report.failed_expectations
    );
    assert!(
        report.completed >= 50,
        "expected ~100 completed (50 rps x 2s), got {}",
        report.completed
    );
    assert_eq!(report.failures, 0, "synth pipe should not fail");
    assert!(
        !report.windows.is_empty(),
        "open-loop run must populate ScenarioReport.windows"
    );
}

#[proxima::test(runtime = "tokio")]
async fn programmatic_open_loop_matches_scenario_file_shape() {
    let scenario = Scenario::new_programmatic(
        WorkloadSpec::new_open_loop("echo", 50, DurationSpec::from_secs(2)).with_concurrency(16),
    )
    .with_pipe(
        "echo",
        serde_json::json!({ "synth": { "status": 200, "body": "ok" } }),
    )
    .with_expectation(Expectation::SuccessRateGe { ratio: 0.95 });

    let context = LoadContext::with_default_registry().expect("load context");
    let report = run_scenario(&scenario, &context).await.expect("run");

    assert!(
        report.passed(),
        "programmatic build must pass like the toml"
    );
    assert!(report.completed >= 50);
    assert!(!report.windows.is_empty());
}

#[proxima::test(runtime = "tokio")]
async fn closed_loop_scenario_leaves_windows_empty() {
    let scenario = Scenario::new_programmatic(WorkloadSpec::new("echo", 25).with_concurrency(4))
        .with_pipe(
            "echo",
            serde_json::json!({ "synth": { "status": 200, "body": "ok" } }),
        );

    let context = LoadContext::with_default_registry().expect("load context");
    let report = run_scenario(&scenario, &context).await.expect("run");

    assert_eq!(report.completed, 25);
    assert!(
        report.windows.is_empty(),
        "closed-loop dispatch must leave windows empty"
    );
}
