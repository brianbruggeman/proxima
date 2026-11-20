//! End to end: the real `proxima serve <config>` answers initialize, tools/list
//! and a configured `review_ticket` tools/call over stdio, and nothing else.

#![cfg(unix)]
#![allow(clippy::expect_used)]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{ExitStatus, Stdio};
use std::time::Duration;

use proxima::default_config_format_registry;
use serde_json::{Value, json};
use tempfile::tempdir;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{ChildStdout, Command};

const TRACKER: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/fixtures/mcp_stub/tracker.sh");
const INCUMBENT_TOOLS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/fixtures/mcp_stub/incumbent_tools.json");
const FIXTURE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/mcp_config_tool/review_ticket.json");
const TWIN: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/mcp_config_tool/review_ticket.toml");
const TOOL_TEXT_OK: &str = r#"{"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"text","text":"{\"content\":[{\"text\":\"review of PROXIMA-24: adds StageBody\",\"type\":\"text\"}]}"}]}}"#;
const TOOL_ERROR_MESSAGE: &str = "tool `review_ticket` returned 502: stage `pr` failed: stage `pr` placeholder `body.ticket.result.structuredContent.repo` did not resolve";
const BOUND: Duration = Duration::from_secs(30);

fn read_json(path: &str) -> Value {
    let text = fs::read_to_string(path).expect("read json fixture");
    serde_json::from_str(&text).expect("parse json fixture")
}

fn incumbent_tools() -> Vec<Value> {
    read_json(INCUMBENT_TOOLS).as_array().expect("incumbent tools array").clone()
}

fn configured_tool() -> Value {
    read_json(FIXTURE)["listen"][0]["tools"][0].clone()
}

fn check_tool_list(tools: &Value, configured: &[Value]) -> Result<(), String> {
    let listed = tools.as_array().ok_or("tools is not an array")?;
    let incumbent = incumbent_tools();
    if listed.len() != incumbent.len() + configured.len() {
        return Err(format!("expected {} tools, got {}", incumbent.len() + configured.len(), listed.len()));
    }
    let (head, tail) = listed.split_at(incumbent.len());
    if head != incumbent.as_slice() {
        return Err("incumbent tools differ".to_string());
    }
    if tail != configured {
        return Err("configured tools differ".to_string());
    }
    Ok(())
}

fn check_tool_text(line: &str, expected: &str) -> Result<(), String> {
    let actual: Value = serde_json::from_str(line).map_err(|error| format!("actual: {error}"))?;
    let wanted: Value = serde_json::from_str(expected).map_err(|error| format!("expected: {error}"))?;
    if actual == wanted { Ok(()) } else { Err(format!("{actual} != {wanted}")) }
}

fn write_config(dir: &Path, fixture: &str, strip_init: bool) -> PathBuf {
    fs::copy(TRACKER, dir.join("tracker.sh")).expect("copy tracker.sh");
    let text = fs::read_to_string(fixture).expect("read fixture").replace("<dir>", &dir.display().to_string());
    let name = Path::new(fixture).file_name().expect("fixture file name");
    let path = dir.join(name);
    let content = if strip_init {
        let mut config: Value = serde_json::from_str(&text).expect("parse fixture");
        config["pipe"][0]["pipeline"]["pipes"]["tracker"]["process_rpc"]
            .as_object_mut()
            .expect("process_rpc object")
            .remove("init");
        config.to_string()
    } else {
        text
    };
    fs::write(&path, content).expect("write config");
    path
}

fn session_requests() -> String {
    let frames = [
        json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"0"}}}),
        json!({"jsonrpc":"2.0","method":"notifications/initialized"}),
        json!({"jsonrpc":"2.0","id":2,"method":"tools/list"}),
        json!({"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"review_ticket","arguments":{"ticket":"PROXIMA-24"}}}),
    ];
    frames.iter().map(|frame| format!("{frame}\n")).collect()
}

async fn read_lines(reader: &mut BufReader<ChildStdout>, count: usize) -> Vec<String> {
    let mut lines = Vec::new();
    for _ in 0..count {
        let mut line = String::new();
        reader.read_line(&mut line).await.expect("read response line");
        lines.push(line.trim().to_string());
    }
    lines
}

async fn serve_session(config: &Path) -> (Vec<String>, String, ExitStatus) {
    let mut child = Command::new(env!("CARGO_BIN_EXE_proxima"))
        .arg("serve").arg(config)
        .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::inherit())
        .spawn().expect("spawn proxima serve");
    let pid = child.id().expect("child pid");
    let mut stdin = child.stdin.take().expect("stdin");
    let mut reader = BufReader::new(child.stdout.take().expect("stdout"));
    stdin.write_all(session_requests().as_bytes()).await.expect("write requests");
    stdin.flush().await.expect("flush requests");
    let lines = read_lines(&mut reader, 3).await;
    drop(stdin);
    std::process::Command::new("kill").args(["-TERM", &pid.to_string()]).status().expect("send SIGTERM");
    let mut rest = String::new();
    reader.read_to_string(&mut rest).await.expect("read stdout to eof");
    let status = child.wait().await.expect("wait for exit");
    (lines, rest, status)
}

async fn bounded_session(fixture: &str, strip_init: bool) -> (Vec<String>, String, ExitStatus) {
    let dir = tempdir().expect("tempdir");
    let config = write_config(dir.path(), fixture, strip_init);
    tokio::time::timeout(BOUND, serve_session(&config)).await.expect("session within 30s")
}

fn parse(line: &str) -> Value {
    serde_json::from_str(line).expect("response line is json")
}

#[proxima::test(flavor = "multi_thread")]
async fn mcp_config_tool_review_ticket_served_over_stdio() {
    let (lines, rest, status) = bounded_session(FIXTURE, false).await;
    assert_eq!(parse(&lines[0])["result"]["protocolVersion"], "2024-11-05");
    check_tool_list(&parse(&lines[1])["result"]["tools"], &[configured_tool()]).expect("tool list");
    check_tool_text(&lines[2], TOOL_TEXT_OK).expect("tool call text");
    assert_eq!(rest, "", "stdout carries exactly the three response lines");
    assert!(status.success(), "serve must exit 0 after SIGTERM, got {status}");
    for line in &lines {
        assert_eq!(parse(line)["jsonrpc"], "2.0");
    }
}

#[proxima::test(flavor = "multi_thread")]
async fn ctl_serve_without_init_returns_the_502_error() {
    let (lines, _, _) = bounded_session(FIXTURE, true).await;
    let response = parse(&lines[2]);
    assert_eq!(response["error"]["code"], -32603);
    assert_eq!(response["error"]["message"], TOOL_ERROR_MESSAGE);
    assert!(check_tool_text(&lines[2], TOOL_TEXT_OK).is_err());
}

#[proxima::test(flavor = "multi_thread")]
async fn ctl_serve_tool_text_accepts_the_expected_line() {
    assert!(check_tool_text(TOOL_TEXT_OK, TOOL_TEXT_OK).is_ok());
}

#[proxima::test(flavor = "multi_thread")]
async fn ctl_serve_tool_list_accepts_incumbent_plus_configured() {
    let mut tools = incumbent_tools();
    tools.push(configured_tool());
    assert!(check_tool_list(&Value::Array(tools), &[configured_tool()]).is_ok());
}

#[proxima::test(flavor = "multi_thread")]
async fn ctl_serve_tool_list_rejects_missing_configured_tool() {
    assert!(check_tool_list(&Value::Array(incumbent_tools()), &[configured_tool()]).is_err());
}

fn json_text_leaves(value: &Value) -> usize {
    match value {
        Value::String(text) => usize::from(matches!(serde_json::from_str::<Value>(text), Ok(Value::Object(_) | Value::Array(_)))),
        Value::Array(items) => items.iter().map(json_text_leaves).sum(),
        Value::Object(map) => map.values().map(json_text_leaves).sum(),
        _ => 0,
    }
}

#[proxima::test(flavor = "multi_thread")]
async fn mcp_config_payload_review_ticket_twin_served_over_stdio() {
    let (lines, rest, status) = bounded_session(TWIN, false).await;
    assert_eq!(parse(&lines[0])["result"]["protocolVersion"], "2024-11-05");
    check_tool_list(&parse(&lines[1])["result"]["tools"], &[configured_tool()]).expect("tool list");
    check_tool_text(&lines[2], TOOL_TEXT_OK).expect("tool call text");
    assert_eq!(rest, "", "stdout carries exactly the three response lines");
    assert!(status.success(), "serve must exit 0 after SIGTERM, got {status}");
    for line in &lines {
        assert_eq!(parse(line)["jsonrpc"], "2.0");
    }
}

#[proxima::test(flavor = "multi_thread")]
async fn mcp_config_payload_twin_has_no_json_text_leaf() {
    let text = fs::read_to_string(TWIN).expect("read twin");
    let registry = default_config_format_registry().expect("format registry");
    let twin: Value = registry.parse_with_hint(&text, Some("toml")).expect("parse twin as toml");
    assert_eq!(json_text_leaves(&twin), 0);
}

#[proxima::test(flavor = "multi_thread")]
async fn ctl_payload_review_ticket_json_has_six_json_text_leaves() {
    assert_eq!(json_text_leaves(&read_json(FIXTURE)), 6);
}
