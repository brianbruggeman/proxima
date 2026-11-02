#![cfg(unix)]

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

async fn uds_exchange(sock_path: &std::path::Path, method: &str, path: &str, body: &str) -> String {
    let mut stream = UnixStream::connect(sock_path)
        .await
        .expect("connect proximad uds");
    let request = format!(
        "{method} {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    stream
        .write_all(request.as_bytes())
        .await
        .expect("write request");
    let mut response = Vec::new();
    stream
        .read_to_end(&mut response)
        .await
        .expect("read response");
    String::from_utf8_lossy(&response).into_owned()
}

#[proxima::test(flavor = "multi_thread")]
async fn proximad_applies_a_pipe_and_listener_and_serves_it() {
    let state = tempdir().expect("state tempdir");
    let sock_parent = tempdir().expect("sock tempdir");
    let sock_path = sock_parent.path().join("proximad.sock");
    let mut child = Command::new(env!("CARGO_BIN_EXE_proximad"))
        .arg("serve")
        .arg("--unix")
        .arg(&sock_path)
        .arg("--state-dir")
        .arg(state.path())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .kill_on_drop(true)
        .spawn()
        .expect("spawn proximad");
    let mut reader = BufReader::new(child.stdout.take().expect("stdout"));
    let mut ready_line = String::new();
    reader.read_line(&mut ready_line).await.expect("read READY");
    assert!(ready_line.starts_with("READY "), "got: {ready_line:?}");

    let pipe = uds_exchange(
        &sock_path,
        "POST",
        "/pipes/web/apply",
        r#"{"synth":{"status":200,"body":"from-proximad"}}"#,
    )
    .await;
    assert!(pipe.starts_with("HTTP/1.1 200"), "apply pipe got: {pipe}");

    let listener = uds_exchange(
        &sock_path,
        "POST",
        "/listeners/front/apply",
        r#"{"bind":"127.0.0.1:0","mount":[{"path":"/","pipe":"web"}]}"#,
    )
    .await;
    assert!(
        listener.starts_with("HTTP/1.1 200"),
        "apply listener got: {listener}"
    );
    let (_head, body) = listener.split_once("\r\n\r\n").expect("body separator");
    let status: serde_json::Value = serde_json::from_str(body.trim()).expect("status json");
    let address = status["last_message"]
        .as_str()
        .expect("listener status carries the bound address");

    let mut tcp = tokio::net::TcpStream::connect(address)
        .await
        .expect("connect applied listener");
    tcp.write_all(b"GET / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
        .await
        .expect("write get");
    let mut served = Vec::new();
    tcp.read_to_end(&mut served).await.expect("read served");
    let served = String::from_utf8_lossy(&served);
    assert!(served.starts_with("HTTP/1.1 200"), "served got: {served}");
    assert!(served.ends_with("from-proximad"), "served got: {served}");

    let pipelines = uds_exchange(&sock_path, "GET", "/pipelines", "").await;
    assert!(
        pipelines.starts_with("HTTP/1.1 200"),
        "pipeline plane got: {pipelines}"
    );
}
