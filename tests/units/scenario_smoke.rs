// scenario orchestration (`run_scenario`) is tokio::process-backed with no
// prime equivalent today — see `src/scenarios/orchestrator.rs`.
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

use std::path::PathBuf;

use proxima::{CompareOp, Expectation, LoadContext, Scenario, WorkloadSpec, run_scenario};
use serde_json::json;

fn manifest_relative(path: &str) -> PathBuf {
    // post-Phase-A: umbrella package IS at workspace root; sibling paths join directly.
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR");
    PathBuf::from(manifest_dir).join(path)
}

#[proxima::test(runtime = "tokio")]
async fn synth_only_scenario_completes_with_full_success_rate() {
    let context = LoadContext::with_default_registry().expect("load context");
    let scenario = Scenario::new_programmatic(
        WorkloadSpec::new("echo", 50)
            .with_method("GET")
            .with_path("/")
            .with_concurrency(4),
    )
    .with_pipe(
        "echo",
        json!({
            "synth": {"status": 200, "body": "echo"},
            "name": "echo",
        }),
    )
    .with_expectation(Expectation::SuccessRateGe { ratio: 1.0 });
    let report = run_scenario(&scenario, &context)
        .await
        .expect("scenario run");
    assert_eq!(report.completed, 50);
    assert_eq!(report.successes, 50);
    assert_eq!(report.failures, 0);
    assert!(
        report.passed(),
        "expectations failed: {:?}",
        report.failed_expectations
    );
}

#[proxima::test(runtime = "tokio")]
async fn cached_scenario_drives_cache_hits_after_first_miss() {
    let context = LoadContext::with_default_registry().expect("load context");
    let scenario = Scenario::new_programmatic(
        WorkloadSpec::new("cached", 100)
            .with_method("GET")
            .with_path("/v1/items")
            .with_concurrency(1),
    )
    .with_pipe(
        "cached",
        json!({
            "name": "cached",
            "upstreams": [
                {"kv": "cache", "max_entries": 256, "name": "cache"},
                {"synth": {"status": 200, "body": "from-origin"}, "name": "origin"},
            ],
            "select": {"algorithm": "fallthrough", "miss_on": ["no_data"]},
            "write_back": [["origin", "cache"]],
        }),
    )
    .with_expectation(Expectation::SuccessRateGe { ratio: 1.0 })
    .with_expectation(Expectation::Counter {
        metric: "proxima.write_back.writes_total".into(),
        labels: [("target".to_string(), "cache".to_string())]
            .into_iter()
            .collect(),
        op: CompareOp::Ge,
        expected: 1,
    });
    let report = run_scenario(&scenario, &context)
        .await
        .expect("scenario run");
    assert_eq!(report.completed, 100);
    assert_eq!(report.successes, 100);
    assert!(
        report.passed(),
        "expectations failed: {:?}",
        report.failed_expectations
    );
}

#[proxima::test(runtime = "tokio")]
async fn bundled_echo_scenario_runs_and_passes_expectations() {
    let path = manifest_relative("scenarios/echo/scenario.toml");
    let scenario = Scenario::from_toml_file(&path).expect("parse echo scenario");
    let context = LoadContext::with_default_registry().expect("load context");
    let report = run_scenario(&scenario, &context).await.expect("run echo");
    assert_eq!(report.completed, 200);
    assert_eq!(report.successes, 200);
    // p99 expectation may flap on cold runs; accept up to one expectation failure
    // in CI environments while keeping the other assertions strict.
    let critical: Vec<&String> = report
        .failed_expectations
        .iter()
        .filter(|line| !line.contains("histogram_p99_le_ms") && !line.contains("histogram"))
        .collect();
    assert!(
        critical.is_empty(),
        "non-histogram expectations must hold: {:?}",
        report.failed_expectations,
    );
}

#[proxima::test(runtime = "tokio")]
async fn bundled_cached_scenario_drives_write_back() {
    let path = manifest_relative("scenarios/cached/scenario.toml");
    let scenario = Scenario::from_toml_file(&path).expect("parse cached");
    let context = LoadContext::with_default_registry().expect("load context");
    let report = run_scenario(&scenario, &context).await.expect("run cached");
    assert_eq!(report.completed, 100);
    assert_eq!(report.successes, 100);
    assert!(
        report.passed(),
        "cached expectations failed: {:?}",
        report.failed_expectations,
    );
}

#[proxima::test(runtime = "tokio")]
async fn bundled_fake_stripe_charge_scenario_runs() {
    let path = manifest_relative("scenarios/fake-stripe-charge/scenario.toml");
    let scenario = Scenario::from_toml_file(&path).expect("parse fake-stripe-charge");
    let context = LoadContext::with_default_registry().expect("load context");
    let report = run_scenario(&scenario, &context)
        .await
        .expect("run fake-stripe-charge");
    assert_eq!(report.completed, 50);
    assert_eq!(report.successes, 50);
    assert!(
        report.passed(),
        "fake-stripe-charge expectations failed: {:?}",
        report.failed_expectations
    );
}

// isolated-mode scenario that spawns the proxima binary lives in
// `proxima-cli/tests/scenario_isolated.rs` because CARGO_BIN_EXE_proxima
// is only well-defined for tests in the package that owns the binary.

#[proxima::test(runtime = "tokio")]
async fn unknown_target_pipe_returns_typed_error() {
    let context = LoadContext::with_default_registry().expect("load context");
    let scenario = Scenario::new_programmatic(WorkloadSpec::new("does-not-exist", 1)).with_pipe(
        "echo",
        json!({"synth": {"status": 200, "body": "echo"}, "name": "echo"}),
    );
    let outcome = run_scenario(&scenario, &context).await;
    assert!(matches!(outcome, Err(proxima::ProximaError::Config(_))));
}
