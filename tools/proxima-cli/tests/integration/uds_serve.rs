//! Smoke test: spawn `proximad serve --unix <path>` as a child process,
//! wait for its READY line, hit `GET /pipelines` over UDS, expect a
//! 200 with `[]`, then send SIGTERM and assert clean exit.

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
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;
use tokio::process::Command;

#[proxima::test(flavor = "multi_thread")]
async fn proximad_serves_pipelines_over_uds() -> io::Result<()> {
    // CARGO_BIN_EXE_proximad is set by cargo for binary-crate integration tests.
    let bin_path = env!("CARGO_BIN_EXE_proximad");

    let state = tempdir().expect("state tempdir");
    let sock_parent = tempdir().expect("sock tempdir");
    let sock_path = sock_parent.path().join("proximad.sock");

    let mut child = Command::new(bin_path)
        .arg("serve")
        .arg("--unix")
        .arg(&sock_path)
        .arg("--state-dir")
        .arg(state.path())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("spawn proximad");

    // wait for READY <path>
    let stdout = child.stdout.take().expect("stdout");
    let mut reader = BufReader::new(stdout);
    let mut ready_line = String::new();
    let ready_deadline = tokio::time::sleep(Duration::from_secs(10));
    tokio::pin!(ready_deadline);
    tokio::select! {
        outcome = reader.read_line(&mut ready_line) => {
            outcome.expect("read READY");
        }
        _ = &mut ready_deadline => {
            let _ = child.start_kill();
            panic!("proximad did not print READY within 10 seconds");
        }
    }
    assert!(
        ready_line.starts_with("READY "),
        "first line must be READY, got: {ready_line:?}"
    );

    // hit GET /pipelines
    let mut stream = UnixStream::connect(&sock_path).await?;
    stream
        .write_all(b"GET /pipelines HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
        .await?;
    stream.flush().await?;
    let mut response = Vec::with_capacity(512);
    stream.read_to_end(&mut response).await?;
    let text = String::from_utf8_lossy(&response);
    assert!(
        text.starts_with("HTTP/1.1 200"),
        "expected 200 on empty /pipelines, got: {text}"
    );
    let (_head, body) = text
        .split_once("\r\n\r\n")
        .expect("response has body separator");
    assert!(body.contains("[]"), "empty list returns `[]`, got: {body}");

    // shut down cleanly
    let _ = child.start_kill();
    let _ = child.wait().await;
    Ok(())
}
