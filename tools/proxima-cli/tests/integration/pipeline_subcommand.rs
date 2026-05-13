//! End-to-end smoke for `proxima pipeline …` over a local proximad UDS.
//!
//! Spawns the proximad binary, waits for READY, then invokes the
//! proxima CLI with `--socket <path>` to submit a pipeline spec and
//! verify the list/resolve/inspect flow. Skips the SSH-stdio leg —
//! that needs a remote host. Local UDS is the symmetric path; if it
//! works here, SSH-stdio works once a remote daemon is up.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

use std::io;
use std::process::Stdio;
use std::time::Duration;

use tempfile::tempdir;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;

/// Discover the proximad binary at runtime. Cargo's
/// `CARGO_BIN_FILE_PROXIMAD` env var only works with nightly artifact
/// deps; stable workspaces need a runtime lookup. Try
/// `$CARGO_TARGET_DIR/debug/proximad` first (proxima's `.envrc` sets
/// CARGO_TARGET_DIR=/tmp/cargo_target), then fall back to
/// `<workspace>/target/debug/proximad`.
fn proximad_bin() -> std::path::PathBuf {
    if let Ok(target_dir) = std::env::var("CARGO_TARGET_DIR") {
        return std::path::PathBuf::from(target_dir)
            .join("debug")
            .join("proximad");
    }
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    std::path::Path::new(manifest_dir)
        .parent()
        .expect("tools dir")
        .parent()
        .expect("workspace root")
        .join("target")
        .join("debug")
        .join("proximad")
}

fn cli_bin() -> &'static str {
    env!("CARGO_BIN_EXE_proxima")
}

/// Kills the wrapped proximad child on drop, including on test panic —
/// otherwise an assertion failure mid-test leaves the daemon (and its
/// piped stdout) running, which leaks a process and can wedge anything
/// downstream still reading that pipe.
struct DaemonGuard(tokio::process::Child);

impl Drop for DaemonGuard {
    fn drop(&mut self) {
        let _ = self.0.start_kill();
    }
}

async fn spawn_proximad(
    socket: &std::path::Path,
    state_dir: &std::path::Path,
) -> io::Result<DaemonGuard> {
    let bin = proximad_bin();
    assert!(
        bin.exists(),
        "proximad binary not found at {bin:?}; run `cargo build -p proximad` first"
    );
    let mut child = Command::new(&bin)
        .arg("serve")
        .arg("--unix")
        .arg(socket)
        .arg("--state-dir")
        .arg(state_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()?;
    let stdout = child.stdout.take().expect("stdout");
    let mut reader = BufReader::new(stdout);
    let mut ready_line = String::new();
    let deadline = tokio::time::sleep(Duration::from_secs(10));
    tokio::pin!(deadline);
    tokio::select! {
        outcome = reader.read_line(&mut ready_line) => {
            outcome.expect("read READY");
        }
        _ = &mut deadline => {
            let _ = child.start_kill();
            panic!("proximad never printed READY");
        }
    }
    assert!(
        ready_line.starts_with("READY "),
        "first line must be READY, got: {ready_line:?}"
    );
    Ok(DaemonGuard(child))
}

async fn run_cli(args: &[&str]) -> (String, String, i32) {
    let output = Command::new(cli_bin())
        .args(args)
        .output()
        .await
        .expect("spawn proxima cli");
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    let code = output.status.code().unwrap_or(-1);
    (stdout, stderr, code)
}

#[proxima::test(flavor = "multi_thread")]
async fn pipeline_submit_list_resolve_inspect_round_trip() -> io::Result<()> {
    let state = tempdir()?;
    let sock_parent = tempdir()?;
    let sock_path = sock_parent.path().join("proximad.sock");

    let mut daemon = spawn_proximad(&sock_path, state.path()).await?;
    let socket_arg = sock_path.to_string_lossy().into_owned();

    // write a tiny pipeline spec
    let spec_dir = tempdir()?;
    let spec_path = spec_dir.path().join("pipe.toml");
    tokio::fs::write(
        &spec_path,
        r#"name = "cli-roundtrip"

[[stages]]
name = "only"
command = "/bin/sh"
args = ["-c", "exit 0"]
"#,
    )
    .await?;
    let spec_arg = spec_path.to_string_lossy().into_owned();

    // submit
    let (stdout, stderr, code) =
        run_cli(&["pipeline", "--socket", &socket_arg, "submit", &spec_arg]).await;
    assert_eq!(code, 0, "submit must exit 0; stderr={stderr}");
    assert!(
        stdout.contains("pipeline_id"),
        "submit response must include pipeline_id: {stdout}"
    );

    // give the daemon a moment to publish the Pipeline::Started event
    tokio::time::sleep(Duration::from_millis(50)).await;

    // list
    let (stdout, stderr, code) = run_cli(&["pipeline", "--socket", &socket_arg, "list"]).await;
    assert_eq!(code, 0, "list must exit 0; stderr={stderr}");
    assert!(
        stdout.contains("cli-roundtrip"),
        "list must surface the name: {stdout}"
    );

    // resolve by name
    let (stdout, stderr, code) = run_cli(&[
        "pipeline",
        "--socket",
        &socket_arg,
        "resolve",
        "cli-roundtrip",
    ])
    .await;
    assert_eq!(code, 0, "resolve must exit 0; stderr={stderr}");
    assert!(
        stdout.contains("pipeline_id"),
        "resolve response must include pipeline_id: {stdout}"
    );

    // inspect by name
    let (stdout, stderr, code) = run_cli(&[
        "pipeline",
        "--socket",
        &socket_arg,
        "inspect",
        "cli-roundtrip",
    ])
    .await;
    assert_eq!(code, 0, "inspect must exit 0; stderr={stderr}");
    assert!(
        stdout.contains("cli-roundtrip"),
        "inspect response must surface the name: {stdout}"
    );

    // tail — opens a chunked stream and streams events for the
    // already-terminal pipeline. Server replays historicals then ends
    // the chunked stream, so the CLI should exit cleanly with a few
    // recorded events on stdout.
    let (stdout, stderr, code) =
        run_cli(&["pipeline", "--socket", &socket_arg, "tail", "cli-roundtrip"]).await;
    assert_eq!(code, 0, "tail must exit 0; stderr={stderr}");
    let lines: Vec<&str> = stdout.lines().filter(|line| !line.is_empty()).collect();
    assert!(
        !lines.is_empty(),
        "tail must surface at least one event for a terminal pipeline"
    );
    // first event for a pipeline is always Pipeline::Started — its
    // JSON shape carries proto=pipeline,phase=started.
    assert!(
        lines
            .iter()
            .any(|line| line.contains("\"proto\":\"pipeline\"")
                && line.contains("\"phase\":\"started\"")),
        "tail must include the Pipeline::Started event: {lines:?}"
    );

    // tear down (DaemonGuard also kills on drop, so panics upstream are covered)
    let _ = daemon.0.start_kill();
    let _ = daemon.0.wait().await;
    Ok(())
}
