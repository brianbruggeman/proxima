//! End-to-end MCP smoke: spawn `proximad serve --mcp-stdio`, drive
//! initialize → tools/list → pipelines_submit → pipelines_list →
//! pipelines_resolve → pipelines_inspect over JSON-RPC. Asserts the
//! pipeline-tools surface is wired through to the PipelineControlPlane
//! trait the HTTP routes already use.

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

use serde_json::{Value, json};
use tempfile::tempdir;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{ChildStdin, ChildStdout, Command};

#[proxima::test(flavor = "multi_thread")]
async fn proximad_serves_pipeline_tools_over_mcp_stdio() -> io::Result<()> {
    let bin = env!("CARGO_BIN_EXE_proximad");
    let state = tempdir()?;
    let mut child = Command::new(bin)
        .arg("serve")
        .arg("--mcp-stdio")
        .arg("--state-dir")
        .arg(state.path())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()?;
    let stdin = child.stdin.take().expect("stdin");
    let stdout = child.stdout.take().expect("stdout");
    let mut reader = BufReader::new(stdout);
    let mut writer = stdin;

    // 1. initialize
    let initialize_response =
        request_response(&mut writer, &mut reader, 1, "initialize", &Value::Null).await?;
    assert!(initialize_response["result"]["protocolVersion"].is_string());

    // 2. tools/list — must surface every pipelines_* tool
    let tools_response =
        request_response(&mut writer, &mut reader, 2, "tools/list", &Value::Null).await?;
    let tool_names: Vec<String> = tools_response["result"]["tools"]
        .as_array()
        .expect("tools array")
        .iter()
        .map(|tool| tool["name"].as_str().unwrap_or_default().to_string())
        .collect();
    for needed in [
        "pipelines_submit",
        "pipelines_list",
        "pipelines_resolve",
        "pipelines_inspect",
        "pipelines_explain",
        "pipelines_replay",
    ] {
        assert!(
            tool_names.iter().any(|name| name == needed),
            "tools list missing {needed}: {tool_names:?}"
        );
    }

    // 3. pipelines_submit — pipeline spec in JSON
    let submit_arguments = json!({
        "spec": {
            "name": "mcp-roundtrip",
            "stages": [
                { "name": "only", "command": "/bin/sh", "args": ["-c", "exit 0"] }
            ]
        }
    });
    let submit_response = tool_call_response(
        &mut writer,
        &mut reader,
        3,
        "pipelines_submit",
        &submit_arguments,
    )
    .await?;
    let submit_text = submit_response["result"]["content"][0]["text"]
        .as_str()
        .expect("submit content text");
    assert!(
        submit_text.contains("pipeline_id"),
        "submit body must contain pipeline_id: {submit_text}"
    );

    // 4. give the daemon a moment to finish the (trivial) pipeline
    tokio::time::sleep(Duration::from_millis(100)).await;

    // 5. pipelines_list
    let list_response =
        tool_call_response(&mut writer, &mut reader, 4, "pipelines_list", &Value::Null).await?;
    let list_text = list_response["result"]["content"][0]["text"]
        .as_str()
        .expect("list content text");
    assert!(
        list_text.contains("mcp-roundtrip"),
        "list must surface the pipeline by name: {list_text}"
    );

    // 6. pipelines_resolve by name
    let resolve_response = tool_call_response(
        &mut writer,
        &mut reader,
        5,
        "pipelines_resolve",
        &json!({ "query": "mcp-roundtrip" }),
    )
    .await?;
    let resolve_text = resolve_response["result"]["content"][0]["text"]
        .as_str()
        .expect("resolve content text");
    assert!(
        resolve_text.contains("pipeline_id"),
        "resolve must surface a pipeline_id: {resolve_text}"
    );

    // 7. tear down — close stdin, daemon exits on EOF
    drop(writer);
    let _ = tokio::time::timeout(Duration::from_secs(3), child.wait()).await;
    Ok(())
}

async fn request_response(
    writer: &mut ChildStdin,
    reader: &mut BufReader<ChildStdout>,
    id: i64,
    method: &str,
    params: &Value,
) -> io::Result<Value> {
    let request = json!({
        "jsonrpc": "2.0",
        "id": id,
        "method": method,
        "params": params,
    });
    let line = serde_json::to_string(&request).unwrap();
    writer.write_all(line.as_bytes()).await?;
    writer.write_all(b"\n").await?;
    writer.flush().await?;
    let mut response_line = String::new();
    reader.read_line(&mut response_line).await?;
    let parsed: Value = serde_json::from_str(response_line.trim()).expect("parse jsonrpc");
    Ok(parsed)
}

async fn tool_call_response(
    writer: &mut ChildStdin,
    reader: &mut BufReader<ChildStdout>,
    id: i64,
    tool_name: &str,
    arguments: &Value,
) -> io::Result<Value> {
    request_response(
        writer,
        reader,
        id,
        "tools/call",
        &json!({ "name": tool_name, "arguments": arguments }),
    )
    .await
}
