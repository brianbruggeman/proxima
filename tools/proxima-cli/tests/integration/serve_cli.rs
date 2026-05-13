#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

use std::net::TcpListener;
use std::process::Stdio;

use tempfile::tempdir;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;
use tokio::process::Command;

#[proxima::test]
async fn serve_default_ephemeral_addr_announces_reachable_listener() {
    let dir = tempdir().expect("tempdir");
    let config_path = dir.path().join("hello.toml");
    tokio::fs::write(
        &config_path,
        r#"
name = "hello"

[[upstreams]]
synth = { status = 200, body = "hello from default port\n" }
"#,
    )
    .await
    .expect("write config");

    let mut child = Command::new(env!("CARGO_BIN_EXE_proxima"))
        .arg("serve")
        .arg("--config")
        .arg(&config_path)
        .arg("--mount")
        .arg("/{*path}")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn proxima serve");

    let stdout = child.stdout.take().expect("stdout pipe");
    let mut lines = BufReader::new(stdout).lines();
    let ready_line = tokio::time::timeout(std::time::Duration::from_secs(5), lines.next_line())
        .await
        .expect("ready timeout")
        .expect("read ready line")
        .expect("ready line");
    let addr = ready_line
        .strip_prefix("READY ")
        .expect("ready prefix")
        .to_string();
    assert!(!addr.ends_with(":0"), "ready addr must be concrete: {addr}");

    let mut stream = TcpStream::connect(&addr).await.expect("connect ready addr");
    stream
        .write_all(b"GET /anything HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
        .await
        .expect("write request");
    let mut response = String::new();
    stream
        .read_to_string(&mut response)
        .await
        .expect("read response");
    assert!(response.contains("200 OK"), "response: {response}");
    assert!(
        response.contains("hello from default port"),
        "response: {response}",
    );

    child.kill().await.expect("kill child");
}

#[proxima::test]
async fn serve_fails_before_ready_when_requested_addr_is_busy() {
    let dir = tempdir().expect("tempdir");
    let config_path = dir.path().join("hello.toml");
    tokio::fs::write(
        &config_path,
        r#"
name = "hello"

[[upstreams]]
synth = { status = 200, body = "hello\n" }
"#,
    )
    .await
    .expect("write config");
    let listener = TcpListener::bind("127.0.0.1:0").expect("reserve addr");
    let addr = listener.local_addr().expect("local addr");

    let output = Command::new(env!("CARGO_BIN_EXE_proxima"))
        .arg("serve")
        .arg("--config")
        .arg(&config_path)
        .arg("--addr")
        .arg(addr.to_string())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
        .expect("spawn proxima serve");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !output.status.success(),
        "stdout: {stdout} stderr: {stderr}"
    );
    assert!(!stdout.contains("READY"), "stdout: {stdout}");
}

#[proxima::test]
async fn serve_full_config_fails_before_ready_when_listener_addr_is_busy() {
    let dir = tempdir().expect("tempdir");
    let config_path = dir.path().join("full.toml");
    let listener = TcpListener::bind("127.0.0.1:0").expect("reserve addr");
    let addr = listener.local_addr().expect("local addr");
    tokio::fs::write(
        &config_path,
        format!(
            r#"
[[pipe]]
name = "hello"
upstreams = [
  {{ synth = {{ status = 200, body = "hello\n" }} }},
]

[[listen]]
type = "http"
bind = "{addr}"
[[listen.mount]]
path = "/{{*path}}"
pipe = "hello"
"#,
        ),
    )
    .await
    .expect("write config");

    let output = Command::new(env!("CARGO_BIN_EXE_proxima"))
        .arg("serve")
        .arg("--config")
        .arg(&config_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
        .expect("spawn proxima serve");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !output.status.success(),
        "stdout: {stdout} stderr: {stderr}"
    );
    assert!(!stdout.contains("READY"), "stdout: {stdout}");
}
