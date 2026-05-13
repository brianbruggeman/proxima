//! `proxima load` end-to-end CLI smoke tests. Exercises the binary
//! against the committed `scenarios/open-loop-smoke/scenario.toml`
//! fixture and against tempdir fixtures for the error paths.

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
use std::process::Stdio;

use tempfile::tempdir;
use tokio::process::Command;

fn proxima_bin() -> &'static str {
    env!("CARGO_BIN_EXE_proxima")
}

fn smoke_scenario_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("manifest dir has parent")
        .parent()
        .expect("tools dir has parent")
        .join("scenarios")
        .join("open-loop-smoke")
}

#[proxima::test]
async fn load_at_smoke_dir_discovers_scenario_and_passes() {
    let output = Command::new(proxima_bin())
        .arg("load")
        .current_dir(smoke_scenario_dir())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
        .expect("spawn proxima load");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "expected exit 0; stdout={stdout}\nstderr={stderr}"
    );
    assert!(
        stdout.contains("PASS"),
        "summary should report PASS: {stdout}"
    );
    assert!(
        stderr.contains("p50_ms"),
        "stderr should include window header: {stderr}"
    );
}

#[proxima::test]
async fn load_with_json_emits_structured_report() {
    let output = Command::new(proxima_bin())
        .arg("load")
        .arg("--json")
        .current_dir(smoke_scenario_dir())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
        .expect("spawn proxima load --json");

    assert!(output.status.success(), "expected exit 0 on --json run");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let report: serde_json::Value =
        serde_json::from_str(stdout.trim()).expect("--json output must parse as JSON");
    assert_eq!(report["passed"], serde_json::Value::Bool(true));
    assert!(
        report["completed"].as_u64().unwrap_or(0) >= 50,
        "expected at least 50 completed: {report}"
    );
    assert!(report["windows"].is_array(), "windows must be a JSON array");
}

#[proxima::test]
async fn load_with_rps_and_duration_overrides_applies_them() {
    let output = Command::new(proxima_bin())
        .arg("load")
        .arg("--json")
        .arg("--rps")
        .arg("100")
        .arg("--duration")
        .arg("1s")
        .current_dir(smoke_scenario_dir())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
        .expect("spawn proxima load with overrides");

    assert!(output.status.success(), "expected exit 0");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let report: serde_json::Value = serde_json::from_str(stdout.trim()).expect("parse JSON");
    let completed = report["completed"].as_u64().unwrap_or(0);
    assert!(
        (60..=130).contains(&completed),
        "100 rps * 1s should yield ~100 completed (60-130 tolerance), got {completed}"
    );
}

#[proxima::test]
async fn load_with_no_scenario_in_cwd_exits_1() {
    let empty = tempdir().expect("tempdir");
    let output = Command::new(proxima_bin())
        .arg("load")
        .current_dir(empty.path())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
        .expect("spawn proxima load");

    let code = output.status.code().unwrap_or(-1);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(
        code, 1,
        "expected exit 1 for missing scenario; stderr={stderr}"
    );
    assert!(
        stderr.contains("no scenario found"),
        "stderr should describe discovery search: {stderr}"
    );
}

#[proxima::test]
async fn load_with_failing_expectation_exits_1() {
    let dir = tempdir().expect("tempdir");
    let scenario_path = dir.path().join("scenario.toml");
    tokio::fs::write(
        &scenario_path,
        r#"
[workload]
target_pipe = "echo"
target_rps  = 25
duration    = "1s"
concurrency = 4

[[pipe]]
name = "echo"
[pipe.synth]
status = 200

[[expect]]
kind     = "counter"
metric   = "proxima.fake.never_emitted"
op       = "ge"
expected = 999999
"#,
    )
    .await
    .expect("write scenario");

    let output = Command::new(proxima_bin())
        .arg("load")
        .current_dir(dir.path())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
        .expect("spawn proxima load");

    let code = output.status.code().unwrap_or(-1);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(
        code, 1,
        "expected exit 1 on expectation fail; stdout={stdout}"
    );
    assert!(
        stdout.contains("FAIL"),
        "summary should mark FAIL: {stdout}"
    );
}

#[proxima::test]
async fn load_remote_stage2_placeholder_exits_2() {
    let output = Command::new(proxima_bin())
        .arg("load")
        .arg("--remote")
        .arg("127.0.0.1:9999")
        .current_dir(smoke_scenario_dir())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
        .expect("spawn proxima load --remote");

    let code = output.status.code().unwrap_or(-1);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(
        code, 2,
        "--remote is a Stage 2 placeholder; expected exit 2; stderr={stderr}"
    );
}

#[proxima::test]
async fn load_with_both_open_and_closed_loop_fields_exits_2() {
    let dir = tempdir().expect("tempdir");
    let scenario_path = dir.path().join("scenario.toml");
    tokio::fs::write(
        &scenario_path,
        r#"
[workload]
target_pipe = "echo"
requests    = 10
target_rps  = 50
duration    = "1s"

[[pipe]]
name = "echo"
[pipe.synth]
status = 200
"#,
    )
    .await
    .expect("write scenario");

    let output = Command::new(proxima_bin())
        .arg("load")
        .current_dir(dir.path())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
        .expect("spawn proxima load");

    let code = output.status.code().unwrap_or(-1);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(
        code, 2,
        "mode collision should exit 2 (invocation error); stderr={stderr}"
    );
}
