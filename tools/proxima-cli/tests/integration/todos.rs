#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

use std::net::{SocketAddr, TcpListener as StdTcpListener};
use std::path::Path;
use std::process::Stdio;
use std::time::Duration;

use tempfile::tempdir;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;
use tokio::process::{Child, Command};

fn reserve_port() -> SocketAddr {
    let listener = StdTcpListener::bind("127.0.0.1:0").expect("reserve port");
    let addr = listener.local_addr().expect("local addr");
    drop(listener);
    addr
}

fn todos_config(kv_path: &Path, bind: SocketAddr) -> String {
    format!(
        r#"{{
            "schema": [{{
                "name": "Todo",
                "schema": {{
                    "type": "struct",
                    "value": {{
                        "name": "Todo",
                        "fields": [
                            {{"name": "title", "schema": {{"type": "string", "value": {{"min_len": 1}}}}, "flags": {{}}}},
                            {{"name": "done", "schema": {{"type": "bool"}}, "flags": {{"optional": true}}}}
                        ]
                    }}
                }}
            }}],
            "pipe": [{{
                "name": "todos",
                "kv": "file",
                "path": "{}",
                "max_entries": 1000,
                "list_mode": true,
                "middleware": [{{"type": "validate", "schema": "Todo"}}]
            }}],
            "listen": [{{
                "type": "http",
                "bind": "{}",
                "mount": [
                    {{"path": "/todos/{{id}}", "pipe": "todos"}},
                    {{"path": "/todos", "pipe": "todos"}}
                ]
            }}]
        }}"#,
        kv_path.display(),
        bind
    )
}

async fn read_first_ready(child: &mut Child) -> String {
    let stdout = child.stdout.take().expect("stdout");
    let mut lines = BufReader::new(stdout).lines();
    let ready = tokio::time::timeout(Duration::from_secs(5), lines.next_line())
        .await
        .expect("ready timeout")
        .expect("read ready")
        .expect("ready line");
    ready
        .strip_prefix("READY ")
        .expect("ready prefix")
        .to_string()
}

async fn http_call(addr: &str, method: &str, path: &str, body: Option<&[u8]>) -> String {
    let mut stream = TcpStream::connect(addr).await.expect("connect");
    let body_bytes = body.unwrap_or_default();
    let mut head = format!("{method} {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n");
    if !body_bytes.is_empty() {
        head.push_str(&format!(
            "Content-Type: application/json\r\nContent-Length: {}\r\n",
            body_bytes.len()
        ));
    }
    head.push_str("\r\n");
    stream.write_all(head.as_bytes()).await.expect("write head");
    if !body_bytes.is_empty() {
        stream.write_all(body_bytes).await.expect("write body");
    }
    stream.flush().await.expect("flush");
    let mut response = String::new();
    stream
        .read_to_string(&mut response)
        .await
        .expect("read response");
    response
}

fn split_response_body(response: &str) -> &str {
    response
        .split_once("\r\n\r\n")
        .map(|(_, body)| body)
        .unwrap_or("")
}

async fn spawn_serve(config_path: &Path) -> Child {
    Command::new(env!("CARGO_BIN_EXE_proxima"))
        .arg("serve")
        .arg("--config")
        .arg(config_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn proxima")
}

#[proxima::test]
async fn todos_full_crud_and_list_round_trip() {
    let dir = tempdir().expect("tempdir");
    let kv_path = dir.path().join("todos");
    tokio::fs::create_dir_all(&kv_path).await.expect("kv dir");
    let config_path = dir.path().join("todos.json");
    let bind = reserve_port();
    tokio::fs::write(&config_path, todos_config(&kv_path, bind))
        .await
        .expect("write config");

    let mut child = spawn_serve(&config_path).await;
    let addr = read_first_ready(&mut child).await;

    // CREATE
    let post = http_call(
        &addr,
        "POST",
        "/todos/abc",
        Some(br#"{"title":"buy milk"}"#),
    )
    .await;
    assert!(post.contains("200 OK"), "post status: {post}");
    let body = split_response_body(&post);
    assert!(body.contains("buy milk"), "post body: {body}");

    // READ
    let get = http_call(&addr, "GET", "/todos/abc", None).await;
    assert!(get.contains("200 OK"), "get status: {get}");
    let body = split_response_body(&get);
    assert!(body.contains("buy milk"), "get body: {body}");

    // UPDATE
    let put = http_call(
        &addr,
        "PUT",
        "/todos/abc",
        Some(br#"{"title":"buy oat milk","done":false}"#),
    )
    .await;
    assert!(put.contains("200 OK"), "put status: {put}");
    assert!(
        split_response_body(&put).contains("oat milk"),
        "put body: {put}"
    );

    // CREATE second todo for list-mode
    let _ = http_call(
        &addr,
        "POST",
        "/todos/xyz",
        Some(br#"{"title":"walk dog"}"#),
    )
    .await;

    // LIST
    let list = http_call(&addr, "GET", "/todos", None).await;
    assert!(list.contains("200 OK"), "list status: {list}");
    let body = split_response_body(&list);
    assert!(
        body.contains('['),
        "list body should contain a JSON array: {body}"
    );
    assert!(
        body.contains("oat milk"),
        "list body must include first todo: {body}"
    );
    assert!(
        body.contains("walk dog"),
        "list body must include second todo: {body}"
    );

    // DELETE
    let del = http_call(&addr, "DELETE", "/todos/abc", None).await;
    assert!(del.contains("204"), "delete status: {del}");

    // GET-after-DELETE → 404 (NoData maps to Not Found at the HTTP layer)
    let after = http_call(&addr, "GET", "/todos/abc", None).await;
    assert!(after.contains("404"), "after-delete status: {after}");

    child.kill().await.expect("kill");
}

#[proxima::test]
async fn todos_post_with_missing_field_is_rejected() {
    let dir = tempdir().expect("tempdir");
    let kv_path = dir.path().join("todos");
    tokio::fs::create_dir_all(&kv_path).await.expect("kv dir");
    let config_path = dir.path().join("todos.json");
    let bind = reserve_port();
    tokio::fs::write(&config_path, todos_config(&kv_path, bind))
        .await
        .expect("write config");

    let mut child = spawn_serve(&config_path).await;
    let addr = read_first_ready(&mut child).await;

    let response = http_call(&addr, "POST", "/todos/abc", Some(br#"{}"#)).await;
    assert!(response.contains("400"), "expected 400: {response}");
    assert!(
        split_response_body(&response).contains("validation_failed"),
        "expected validation envelope: {response}"
    );

    child.kill().await.expect("kill");
}
