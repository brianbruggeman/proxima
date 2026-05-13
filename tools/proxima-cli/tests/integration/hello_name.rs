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
use std::time::Duration;

use tempfile::tempdir;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;
use tokio::process::{Child, Command};

const HELLO_NAME_CONFIG: &str = r#"{
    "name": "hello-name",
    "middleware": [{
        "type": "validate",
        "schema": {
            "type": "struct",
            "value": {
                "name": "User",
                "fields": [{
                    "name": "name",
                    "schema": {"type": "string", "value": {"min_len": 1}},
                    "flags": {}
                }]
            }
        }
    }],
    "synth": {
        "status": 200,
        "headers": {"content-type": "text/plain"},
        "body_template": "hello, {{body.name}}\n"
    }
}"#;

async fn read_ready_addr(child: &mut Child) -> String {
    let stdout = child.stdout.take().expect("stdout pipe");
    let mut lines = BufReader::new(stdout).lines();
    let ready_line = tokio::time::timeout(Duration::from_secs(5), lines.next_line())
        .await
        .expect("ready timeout")
        .expect("read ready line")
        .expect("ready line");
    ready_line
        .strip_prefix("READY ")
        .expect("ready prefix")
        .to_string()
}

async fn post_json(addr: &str, payload: &[u8]) -> String {
    let mut stream = TcpStream::connect(addr).await.expect("connect");
    let request = format!(
        "POST / HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        payload.len()
    );
    stream
        .write_all(request.as_bytes())
        .await
        .expect("write headers");
    stream.write_all(payload).await.expect("write body");
    stream.flush().await.expect("flush");
    let mut response = String::new();
    stream
        .read_to_string(&mut response)
        .await
        .expect("read response");
    response
}

#[proxima::test]
async fn post_with_valid_name_returns_templated_hello() {
    let dir = tempdir().expect("tempdir");
    let config_path = dir.path().join("hello-name.json");
    tokio::fs::write(&config_path, HELLO_NAME_CONFIG)
        .await
        .expect("write config");

    let mut child = Command::new(env!("CARGO_BIN_EXE_proxima"))
        .arg("serve")
        .arg("--config")
        .arg(&config_path)
        .arg("--mount")
        .arg("/")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn");

    let addr = read_ready_addr(&mut child).await;
    let response = post_json(&addr, br#"{"name":"brian"}"#).await;
    assert!(response.contains("200 OK"), "response: {response}");
    assert!(
        response.contains("hello, brian"),
        "expected templated response: {response}"
    );
    child.kill().await.expect("kill");
}

#[proxima::test]
async fn post_with_missing_name_is_rejected_by_validate() {
    let dir = tempdir().expect("tempdir");
    let config_path = dir.path().join("hello-name.json");
    tokio::fs::write(&config_path, HELLO_NAME_CONFIG)
        .await
        .expect("write config");

    let mut child = Command::new(env!("CARGO_BIN_EXE_proxima"))
        .arg("serve")
        .arg("--config")
        .arg(&config_path)
        .arg("--mount")
        .arg("/")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn");

    let addr = read_ready_addr(&mut child).await;
    let response = post_json(&addr, br#"{}"#).await;
    assert!(response.contains("400"), "expected 400: {response}");
    assert!(
        response.contains("validation_failed"),
        "expected validation error envelope: {response}"
    );
    child.kill().await.expect("kill");
}

#[proxima::test]
async fn post_with_empty_name_is_rejected_for_min_len() {
    let dir = tempdir().expect("tempdir");
    let config_path = dir.path().join("hello-name.json");
    tokio::fs::write(&config_path, HELLO_NAME_CONFIG)
        .await
        .expect("write config");

    let mut child = Command::new(env!("CARGO_BIN_EXE_proxima"))
        .arg("serve")
        .arg("--config")
        .arg(&config_path)
        .arg("--mount")
        .arg("/")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn");

    let addr = read_ready_addr(&mut child).await;
    let response = post_json(&addr, br#"{"name":""}"#).await;
    assert!(response.contains("400"), "expected 400: {response}");
    assert!(
        response.contains("min_len"),
        "expected min_len violation: {response}"
    );
    child.kill().await.expect("kill");
}
