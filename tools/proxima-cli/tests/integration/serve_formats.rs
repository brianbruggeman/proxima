#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

use std::path::Path;
use std::process::Stdio;
use std::time::Duration;

use tempfile::tempdir;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;
use tokio::process::{Child, Command};

async fn assert_serve_responds_to(child: &mut Child, expected_body: &str) {
    let stdout = child.stdout.take().expect("stdout pipe");
    let mut lines = BufReader::new(stdout).lines();
    let ready_line = tokio::time::timeout(Duration::from_secs(5), lines.next_line())
        .await
        .expect("ready timeout")
        .expect("read ready line")
        .expect("ready line");
    let addr = ready_line
        .strip_prefix("READY ")
        .expect("ready prefix")
        .to_string();

    let mut stream = TcpStream::connect(&addr).await.expect("connect ready addr");
    stream
        .write_all(b"GET /any HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
        .await
        .expect("write request");
    let mut response = String::new();
    stream
        .read_to_string(&mut response)
        .await
        .expect("read response");
    assert!(response.contains("200 OK"), "response: {response}");
    assert!(
        response.contains(expected_body),
        "expected `{expected_body}` in response: {response}",
    );
}

async fn spawn_with_config(config_path: &Path) -> Child {
    Command::new(env!("CARGO_BIN_EXE_proxima"))
        .arg("serve")
        .arg("--config")
        .arg(config_path)
        .arg("--mount")
        .arg("/{*path}")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn proxima serve")
}

#[proxima::test]
async fn serve_yaml_config_responds_200() {
    let dir = tempdir().expect("tempdir");
    let config_path = dir.path().join("hello.yaml");
    tokio::fs::write(
        &config_path,
        "name: hello\nsynth:\n  status: 200\n  body: \"yaml ok\\n\"\n",
    )
    .await
    .expect("write yaml config");
    let mut child = spawn_with_config(&config_path).await;
    assert_serve_responds_to(&mut child, "yaml ok").await;
    child.kill().await.expect("kill child");
}

#[proxima::test]
async fn serve_json_config_responds_200() {
    let dir = tempdir().expect("tempdir");
    let config_path = dir.path().join("hello.json");
    tokio::fs::write(
        &config_path,
        r#"{"name":"hello","synth":{"status":200,"body":"json ok\n"}}"#,
    )
    .await
    .expect("write json config");
    let mut child = spawn_with_config(&config_path).await;
    assert_serve_responds_to(&mut child, "json ok").await;
    child.kill().await.expect("kill child");
}

#[proxima::test]
async fn serve_json5_config_responds_200() {
    let dir = tempdir().expect("tempdir");
    let config_path = dir.path().join("hello.json5");
    tokio::fs::write(
        &config_path,
        r#"{
            // json5 supports comments + trailing commas
            "name": "hello",
            "synth": {
                "status": 200,
                "body": "json5 ok\n",
            },
        }"#,
    )
    .await
    .expect("write json5 config");
    let mut child = spawn_with_config(&config_path).await;
    assert_serve_responds_to(&mut child, "json5 ok").await;
    child.kill().await.expect("kill child");
}

#[proxima::test]
async fn serve_ron_config_responds_200() {
    let dir = tempdir().expect("tempdir");
    let config_path = dir.path().join("hello.ron");
    tokio::fs::write(
        &config_path,
        r#"{"name": "hello", "synth": {"status": 200, "body": "ron ok\n"}}"#,
    )
    .await
    .expect("write ron config");
    let mut child = spawn_with_config(&config_path).await;
    assert_serve_responds_to(&mut child, "ron ok").await;
    child.kill().await.expect("kill child");
}

#[proxima::test]
async fn serve_inline_json_positional_responds_200() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_proxima"))
        .arg("serve")
        .arg(r#"{"name":"hello","synth":{"status":200,"body":"inline ok\n"}}"#)
        .arg("--mount")
        .arg("/{*path}")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn proxima serve");
    assert_serve_responds_to(&mut child, "inline ok").await;
    child.kill().await.expect("kill child");
}
