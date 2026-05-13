#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

use proxima::{Expectation, LoadContext, OrchestrationMode, Scenario, WorkloadSpec, run_scenario};
use serde_json::json;

#[proxima::test(runtime = "tokio")]
async fn isolated_mode_spawns_proxima_serve_child_and_drives_traffic() {
    // PROXIMA_CLI must point at the freshly-built `proxima` binary; cargo wires
    // CARGO_BIN_EXE_proxima for tests inside the proxima-cli crate.
    unsafe {
        std::env::set_var("PROXIMA_CLI", env!("CARGO_BIN_EXE_proxima"));
    }
    let scenario = Scenario::new_programmatic(
        WorkloadSpec::new("echo_isolated", 5)
            .with_method("GET")
            .with_path("/")
            .with_concurrency(1),
    )
    .with_mode(OrchestrationMode::Isolated)
    .with_pipe(
        "echo_isolated",
        json!({"synth": {"status": 200, "body": "from-child"}, "name": "echo_isolated"}),
    )
    .with_expectation(Expectation::SuccessRateGe { ratio: 1.0 });
    let context = LoadContext::with_default_registry().expect("load context");
    let report = run_scenario(&scenario, &context)
        .await
        .expect("isolated scenario");
    assert_eq!(report.completed, 5);
    assert_eq!(report.successes, 5);
    assert!(
        report.passed(),
        "isolated scenario expectations failed: {:?}",
        report.failed_expectations,
    );
}
