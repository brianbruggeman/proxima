#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

use std::process::Stdio;
use std::sync::Arc;

use proxima::{
    App, ControlPlane, ControlPlanePipe, DaemonControlPlane, MountTarget, PipeConfig, PipeState,
    PipeStatus, RunConfig, StaticControlPlane,
};
use serde_json::json;
use tempfile::tempdir;
use tokio::process::Command;

async fn spawn_daemon(socket: &std::path::Path) -> proxima::Shutdown {
    let plane: Arc<dyn ControlPlane> = Arc::new(StaticControlPlane::new(vec![PipeStatus {
        name: "cart_api".into(),
        state: PipeState::Running,
        uptime_ms: Some(1234),
        restart_count: 0,
        last_message: None,
    }]));
    spawn_daemon_listener(socket, plane).await
}

async fn spawn_daemon_listener(
    socket: &std::path::Path,
    plane: Arc<dyn ControlPlane>,
) -> proxima::Shutdown {
    let mut app = App::new().expect("app");
    let handle = proxima::pipe::into_handle(ControlPlanePipe::new(plane));
    let _ = app
        .pipe("__cp__", proxima::Spec::Handle(handle.clone()))
        .await
        .expect("register");
    app.mount("/{*path}", MountTarget::Handle(handle))
        .expect("mount");
    // RISC: the same HttpListenProtocol that runs the data plane
    // also runs the control plane. UDS dispatch is triggered by
    // `spec.path`; the SocketAddr is unused on the UDS path but
    // RunConfig::http requires one — pass the loopback ephemeral.
    let mut config = RunConfig::http("127.0.0.1:0".parse().expect("addr"));
    config.spec = json!({"path": socket.to_string_lossy().to_string(), "mode": 0o600});
    app.run_until_signal(config).await.expect("run")
}

#[proxima::test]
async fn daemon_list_prints_known_pipes() {
    let dir = tempdir().expect("tempdir");
    let socket = dir.path().join("proxima.sock");
    let shutdown = spawn_daemon(&socket).await;

    let output = Command::new(env!("CARGO_BIN_EXE_proxima"))
        .arg("daemon")
        .arg("--socket")
        .arg(&socket)
        .arg("list")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
        .expect("spawn cli");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "exit: {} stderr: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(stdout.contains("cart_api"), "stdout: {stdout}");
    assert!(stdout.contains("running"), "stdout: {stdout}");

    shutdown.stop();
}

#[proxima::test]
async fn daemon_status_pretty_prints_one_pipe() {
    let dir = tempdir().expect("tempdir");
    let socket = dir.path().join("proxima.sock");
    let shutdown = spawn_daemon(&socket).await;

    let output = Command::new(env!("CARGO_BIN_EXE_proxima"))
        .arg("daemon")
        .arg("--socket")
        .arg(&socket)
        .arg("status")
        .arg("cart_api")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
        .expect("spawn cli");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success());
    assert!(stdout.contains("\"name\""));
    assert!(stdout.contains("cart_api"));
    assert!(stdout.contains("\"state\""));

    shutdown.stop();
}

#[proxima::test]
async fn daemon_status_unknown_name_writes_404_to_stderr() {
    let dir = tempdir().expect("tempdir");
    let socket = dir.path().join("proxima.sock");
    let shutdown = spawn_daemon(&socket).await;

    let output = Command::new(env!("CARGO_BIN_EXE_proxima"))
        .arg("daemon")
        .arg("--socket")
        .arg(&socket)
        .arg("status")
        .arg("does-not-exist")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
        .expect("spawn cli");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("404") || stderr.contains("status: 4"),
        "stderr: {stderr}"
    );

    shutdown.stop();
}

/// Spin a daemon backed by a real DaemonControlPlane so the apply
/// route actually swaps the registered pipe. Returns the
/// underlying plane so the test can inspect post-apply state.
async fn spawn_daemon_with_real_plane(
    socket: &std::path::Path,
    initial_spec: serde_json::Value,
) -> (Arc<DaemonControlPlane>, proxima::Shutdown) {
    let inner_app = App::new().expect("inner app");
    let plane = Arc::new(DaemonControlPlane::new(
        inner_app,
        vec![PipeConfig {
            name: "echo".into(),
            spec: initial_spec,
            requires: vec![],
        }],
    ));
    let shutdown = spawn_daemon_listener(socket, plane.clone()).await;
    (plane, shutdown)
}

#[proxima::test]
async fn daemon_apply_updates_pipe_spec_and_exits_zero() {
    let dir = tempdir().expect("tempdir");
    let socket = dir.path().join("proxima.sock");
    let v1 = json!({"synth": {"status": 200, "body": "v1"}});
    let (plane, shutdown) = spawn_daemon_with_real_plane(&socket, v1).await;

    // apply swaps an already-registered pipe; bring "echo" online first
    // so the inner App holds a handle that update_pipe can replace.
    plane.start("echo").await.expect("start echo");

    // Write the v2 spec to a temp file so the CLI reads it via --spec.
    let v2 = json!({"synth": {"status": 200, "body": "v2"}});
    let spec_path = dir.path().join("echo-v2.json");
    std::fs::write(&spec_path, serde_json::to_vec(&v2).expect("encode v2")).expect("write v2 spec");

    let output = Command::new(env!("CARGO_BIN_EXE_proxima"))
        .arg("daemon")
        .arg("--socket")
        .arg(&socket)
        .arg("apply")
        .arg("echo")
        .arg("--spec")
        .arg(&spec_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
        .expect("spawn cli");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "exit: {} stdout: {stdout} stderr: {stderr}",
        output.status,
    );
    assert!(
        stdout.contains("\"name\"") && stdout.contains("echo"),
        "expected PipeStatus JSON for echo. stdout: {stdout}, stderr: {stderr}",
    );

    // Apply persisted to the daemon's known-pipes state.
    let status = plane.status("echo").await.expect("status");
    assert_eq!(status.name, "echo");

    shutdown.stop();
}

#[proxima::test]
async fn daemon_apply_unknown_pipe_returns_404() {
    let dir = tempdir().expect("tempdir");
    let socket = dir.path().join("proxima.sock");
    let initial = json!({"synth": {"status": 200, "body": "v1"}});
    let (_plane, shutdown) = spawn_daemon_with_real_plane(&socket, initial).await;

    let spec_path = dir.path().join("ignored.json");
    std::fs::write(&spec_path, b"{\"synth\":{\"body\":\"x\"}}").expect("write spec");

    let output = Command::new(env!("CARGO_BIN_EXE_proxima"))
        .arg("daemon")
        .arg("--socket")
        .arg(&socket)
        .arg("apply")
        .arg("does-not-exist")
        .arg("--spec")
        .arg(&spec_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
        .expect("spawn cli");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("404") || stderr.contains("not found"),
        "stderr: {stderr}",
    );

    shutdown.stop();
}

#[proxima::test]
async fn daemon_metrics_pretty_prints_envelope() {
    let dir = tempdir().expect("tempdir");
    let socket = dir.path().join("proxima.sock");
    let shutdown = spawn_daemon(&socket).await;

    let output = Command::new(env!("CARGO_BIN_EXE_proxima"))
        .arg("daemon")
        .arg("--socket")
        .arg(&socket)
        .arg("metrics")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
        .expect("spawn cli");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success());
    assert!(stdout.contains("counters"));
    assert!(stdout.contains("histograms"));

    shutdown.stop();
}
