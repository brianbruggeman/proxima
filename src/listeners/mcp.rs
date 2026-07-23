use std::future::Future;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::pin::Pin;

use bytes::Bytes;
use futures::channel::oneshot;
use futures::{FutureExt, select};
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
use tracing::{debug, warn};

use proxima_primitives::pipe::Method;
use proxima_primitives::pipe::SendPipe;
use proxima_protocols::jsonrpc::{JsonRpcError, JsonRpcRequest, JsonRpcResponse};

use crate::error::ProximaError;
use crate::pipe::PipeHandle;
use crate::request::{Request, RequestContext};
use proxima_listen::{ListenProtocol, ServeContext};

const PROTOCOL_VERSION: &str = "2024-11-05";
const SERVER_NAME: &str = "proxima";
const SERVER_VERSION: &str = env!("CARGO_PKG_VERSION");

/// MCP listener — JSON-RPC 2.0 over stdio or Unix domain socket.
/// Translates `tools/call` into a substrate `Request`; tool list is
/// fixed to the control-plane routes (`list_pipes`,
/// `pipe_status`, `metrics_snapshot`).
pub struct McpListenProtocol {
    label: String,
}

impl McpListenProtocol {
    #[must_use]
    pub fn new() -> Self {
        Self {
            label: "mcp".into(),
        }
    }
}

impl Default for McpListenProtocol {
    fn default() -> Self {
        Self::new()
    }
}

impl ListenProtocol for McpListenProtocol {
    fn name(&self) -> &str {
        &self.label
    }

    fn serve(
        &self,
        _bind: SocketAddr,
        dispatch: PipeHandle,
        spec: &Value,
        context: ServeContext,
        shutdown: oneshot::Receiver<()>,
    ) -> Pin<Box<dyn Future<Output = Result<(), ProximaError>> + Send + '_>> {
        let transport = spec
            .get("transport")
            .and_then(Value::as_str)
            .unwrap_or("stdio")
            .to_string();
        let path: Option<PathBuf> = spec.get("path").and_then(Value::as_str).map(PathBuf::from);
        let ready_signal = context.ready_signal.clone();
        Box::pin(async move {
            match transport.as_str() {
                "stdio" => serve_stdio(dispatch, shutdown, ready_signal).await,
                "unix" | "uds" => {
                    let path = path.ok_or_else(|| {
                        ProximaError::Config(
                            "mcp transport `unix` requires `path = \"/...\"`".into(),
                        )
                    })?;
                    serve_unix(path, dispatch, shutdown, ready_signal).await
                }
                other => Err(ProximaError::Config(format!(
                    "unknown mcp transport `{other}` — expected stdio | unix"
                ))),
            }
        })
    }
}

async fn serve_stdio(
    dispatch: PipeHandle,
    mut shutdown: oneshot::Receiver<()>,
    ready_signal: Option<std::sync::mpsc::Sender<()>>,
) -> Result<(), ProximaError> {
    debug!("mcp listener bound (stdio)");
    // stdio has no bind step — it is ready the instant the handles exist.
    if let Some(sender) = ready_signal {
        let _ = sender.send(());
    }
    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();
    let reader = BufReader::new(stdin);
    let writer = stdout;
    let writer_mutex = std::sync::Arc::new(tokio::sync::Mutex::new(writer));
    let serve = serve_jsonrpc_loop(reader, writer_mutex, dispatch).fuse();
    futures::pin_mut!(serve);
    select! {
        outcome = serve => outcome,
        _ = (&mut shutdown).fuse() => Ok(()),
    }
}

async fn serve_unix(
    path: PathBuf,
    dispatch: PipeHandle,
    mut shutdown: oneshot::Receiver<()>,
    ready_signal: Option<std::sync::mpsc::Sender<()>>,
) -> Result<(), ProximaError> {
    if path.exists() {
        std::fs::remove_file(&path).map_err(|err| {
            ProximaError::Io(std::io::Error::other(format!(
                "remove stale mcp socket: {err}"
            )))
        })?;
    }
    let listener = UnixListener::bind(&path)
        .map_err(|err| ProximaError::Io(std::io::Error::other(format!("mcp uds bind: {err}"))))?;
    if let Some(sender) = ready_signal {
        let _ = sender.send(());
    }
    debug!(?path, "mcp listener bound (uds)");
    loop {
        select! {
            outcome = listener.accept().fuse() => match outcome {
                Ok((stream, _peer)) => spawn_mcp_unix_handler(stream, dispatch.clone()),
                Err(error) => warn!(?error, "mcp uds accept error"),
            },
            _ = (&mut shutdown).fuse() => {
                let _ = std::fs::remove_file(&path);
                return Ok(());
            }
        }
    }
}

fn spawn_mcp_unix_handler(stream: UnixStream, dispatch: PipeHandle) {
    // spawn_local: surrounding listener serve runs on a TokioPerCoreRuntime
    // worker (current-thread runtime + LocalSet). `?Send` per-request futures
    // (Pipe::call) stay on this core for life.
    tokio::task::spawn_local(async move {
        let (read_half, write_half) = stream.into_split();
        let writer_mutex = std::sync::Arc::new(tokio::sync::Mutex::new(write_half));
        if let Err(error) =
            serve_jsonrpc_loop(BufReader::new(read_half), writer_mutex, dispatch).await
        {
            warn!(?error, "mcp uds connection error");
        }
    });
}

async fn serve_jsonrpc_loop<R, W>(
    mut reader: BufReader<R>,
    writer: std::sync::Arc<tokio::sync::Mutex<W>>,
    dispatch: PipeHandle,
) -> Result<(), ProximaError>
where
    R: tokio::io::AsyncRead + Unpin,
    W: tokio::io::AsyncWrite + Unpin + Send + 'static,
{
    let mut line = String::new();
    loop {
        line.clear();
        let bytes_read = reader.read_line(&mut line).await.map_err(|err| {
            ProximaError::Io(std::io::Error::other(format!("mcp read line: {err}")))
        })?;
        if bytes_read == 0 {
            return Ok(());
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let response = handle_jsonrpc(trimmed, &dispatch).await;
        if let Some(response_text) = response {
            let mut guard = writer.lock().await;
            guard
                .write_all(response_text.as_bytes())
                .await
                .map_err(|err| {
                    ProximaError::Io(std::io::Error::other(format!("mcp write: {err}")))
                })?;
            guard.write_all(b"\n").await.map_err(|err| {
                ProximaError::Io(std::io::Error::other(format!("mcp write nl: {err}")))
            })?;
            guard.flush().await.map_err(|err| {
                ProximaError::Io(std::io::Error::other(format!("mcp flush: {err}")))
            })?;
        }
    }
}

async fn handle_jsonrpc(line: &str, dispatch: &PipeHandle) -> Option<String> {
    let request: JsonRpcRequest = match serde_json::from_str(line) {
        Ok(parsed) => parsed,
        Err(err) => {
            let response = JsonRpcResponse::failure(
                None,
                JsonRpcError::parse_error(format!("parse error: {err}")),
            );
            return Some(to_wire(&response));
        }
    };
    let Some(id) = request.id.clone() else {
        // notification — dispatch but don't respond.
        let _ = dispatch_method(&request, dispatch).await;
        return None;
    };
    let response = match dispatch_method(&request, dispatch).await {
        Ok(result) => JsonRpcResponse::success(id, result),
        Err((code, message)) => JsonRpcResponse::failure(
            Some(id),
            JsonRpcError {
                code: i64::from(code),
                message,
                data: None,
            },
        ),
    };
    Some(to_wire(&response))
}

fn to_wire(response: &JsonRpcResponse) -> String {
    serde_json::to_string(response).unwrap_or_else(|_| String::new())
}

async fn dispatch_method(
    request: &JsonRpcRequest,
    dispatch: &PipeHandle,
) -> Result<Value, (i32, String)> {
    match request.method.as_str() {
        "initialize" => Ok(initialize_response()),
        "initialized" | "notifications/initialized" => Ok(Value::Null),
        "tools/list" => Ok(tools_list_response()),
        "tools/call" => tool_call(request.params.as_ref().unwrap_or(&Value::Null), dispatch).await,
        other => Err((-32601, format!("method `{other}` not found"))),
    }
}

fn initialize_response() -> Value {
    serde_json::json!({
        "protocolVersion": PROTOCOL_VERSION,
        "capabilities": { "tools": {} },
        "serverInfo": { "name": SERVER_NAME, "version": SERVER_VERSION },
    })
}

fn tools_list_response() -> Value {
    serde_json::json!({
        "tools": [
            {
                "name": "list_pipes",
                "description": "List every pipe the proxima daemon knows with its current state.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": false,
                },
            },
            {
                "name": "pipe_status",
                "description": "Return the status of one named pipe.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "name": { "type": "string", "description": "pipe name" },
                    },
                    "required": ["name"],
                    "additionalProperties": false,
                },
            },
            {
                "name": "metrics_snapshot",
                "description": "Snapshot the daemon's counters / gauges / histograms.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": false,
                },
            },
            {
                "name": "pipe_logs",
                "description": "Tail the most recent stdout/stderr lines from one supervised pipe.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "name": { "type": "string", "description": "pipe name" },
                        "max_lines": { "type": "integer", "minimum": 1, "default": 100 },
                    },
                    "required": ["name"],
                    "additionalProperties": false,
                },
            },
            {
                "name": "start_pipe",
                "description": "Start a pipe. Walks the configured dep graph and starts every required pipe in topological order.",
                "inputSchema": {
                    "type": "object",
                    "properties": { "name": { "type": "string" } },
                    "required": ["name"],
                    "additionalProperties": false,
                },
            },
            {
                "name": "stop_pipe",
                "description": "Stop a pipe. Does not cascade to dependents — caller manages dep order on the way down.",
                "inputSchema": {
                    "type": "object",
                    "properties": { "name": { "type": "string" } },
                    "required": ["name"],
                    "additionalProperties": false,
                },
            },
            {
                "name": "restart_pipe",
                "description": "Restart a pipe (stop + start, bumps restart count).",
                "inputSchema": {
                    "type": "object",
                    "properties": { "name": { "type": "string" } },
                    "required": ["name"],
                    "additionalProperties": false,
                },
            },
            {
                "name": "daemon_reload",
                "description": "Re-read the daemon's config from disk and apply any new pipe definitions.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": false,
                },
            },
            {
                "name": "pipelines_submit",
                "description": "Submit a new pipeline (DAG of stages) to the daemon. Spec is JSON shaped like proxima::PipelineSpec.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "spec": { "type": "object", "description": "PipelineSpec body (JSON form)" },
                    },
                    "required": ["spec"],
                    "additionalProperties": false,
                },
            },
            {
                "name": "pipelines_list",
                "description": "List submitted pipelines, newest first. Optional filters narrow by name or spec_hash_hex.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "name": { "type": "string" },
                        "spec_hash_hex": { "type": "string" },
                    },
                    "additionalProperties": false,
                },
            },
            {
                "name": "pipelines_resolve",
                "description": "Resolve a query string (id / name / prefix) to a canonical pipeline id.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": { "type": "string", "description": "id, name, or unique id/name prefix" },
                    },
                    "required": ["query"],
                    "additionalProperties": false,
                },
            },
            {
                "name": "pipelines_inspect",
                "description": "Inspect a pipeline by canonical id (use pipelines_resolve first).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "id": { "type": "string" },
                    },
                    "required": ["id"],
                    "additionalProperties": false,
                },
            },
            {
                "name": "pipelines_explain",
                "description": "Walk a stage's depends_on chain. Returns ordered ancestor list (depth 0 = queried stage).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "id": { "type": "string" },
                        "stage": { "type": "string" },
                    },
                    "required": ["id", "stage"],
                    "additionalProperties": false,
                },
            },
            {
                "name": "pipelines_replay",
                "description": "Replay a recorded pipeline under a new id. Optional substitutes map replaces selected stages with fresh StageSpecs that run live; non-substituted stages re-emit their recorded events.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "id": { "type": "string" },
                        "substitutes": { "type": "object", "description": "stage_name → StageSpec map; omit to replay verbatim" },
                    },
                    "required": ["id"],
                    "additionalProperties": false,
                },
            },
            {
                "name": "verify_static",
                "description": "Walk a parsed proxima spec and report policy violations (cycle detection, timeout coverage, custom rules). Pass either spec_path or spec_inline; same for policy. Returns a text+json report.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "spec_path": { "type": "string" },
                        "spec_inline": { "type": "object" },
                        "policy_path": { "type": "string" },
                        "policy_inline": { "type": "object" },
                        "strict": { "type": "boolean", "default": false },
                    },
                    "additionalProperties": false,
                },
            },
            {
                "name": "verify_replay",
                "description": "Stream a .bin recording and run replay-policy assertions. recording_path is required (no inline body — recordings are binary). Returns a text+json report.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "recording_path": { "type": "string" },
                        "policy_path": { "type": "string" },
                        "policy_inline": { "type": "object" },
                        "strict": { "type": "boolean", "default": false },
                    },
                    "required": ["recording_path"],
                    "additionalProperties": false,
                },
            },
            {
                "name": "scenario_run",
                "description": "Run a proxima load scenario by file path or inline value. Drives the target Pipe via the closed-loop or open-loop branch (chosen by workload.mode()), evaluates expectations, returns a structured ScenarioReport with passed/completed/successes/failures/failed_expectations/windows/metrics_snapshot.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "scenario_path": { "type": "string", "description": "path to a scenario.toml file" },
                        "scenario_inline": { "type": "object", "description": "Scenario value inline (alternative to scenario_path)" },
                    },
                    "additionalProperties": false,
                },
            },
        ],
    })
}

async fn tool_call(params: &Value, dispatch: &PipeHandle) -> Result<Value, (i32, String)> {
    let tool_name = params
        .get("name")
        .and_then(Value::as_str)
        .ok_or((-32602, "tools/call requires `name`".to_string()))?;
    let arguments = params.get("arguments").cloned().unwrap_or(Value::Null);

    // Verify tools call the library directly — they do not go through
    // the daemon control plane. Same library entry points the CLI uses.
    match tool_name {
        "verify_static" => return verify_static_tool(&arguments).await,
        "verify_replay" => return verify_replay_tool(&arguments).await,
        "scenario_run" => return scenario_run_tool(&arguments).await,
        _ => {}
    }

    // body-carrying tools route to (method, path, body) inline; otherwise the
    // (method, path) variant from `route_for_tool` with empty body.
    let (method, path, body) = match tool_name {
        "pipelines_submit" => {
            let spec = arguments
                .get("spec")
                .cloned()
                .ok_or((-32602, "pipelines_submit requires `spec` argument".into()))?;
            let body = serde_json::to_vec(&spec)
                .map_err(|err| (-32603, format!("encode pipelines_submit body: {err}")))?;
            ("POST".into(), "/pipelines/submit".to_string(), Some(body))
        }
        "pipelines_replay" => {
            let id = arguments
                .get("id")
                .and_then(Value::as_str)
                .ok_or((-32602, "pipelines_replay requires `id`".into()))?
                .to_string();
            let substitutes = arguments
                .get("substitutes")
                .cloned()
                .unwrap_or(Value::Object(Default::default()));
            let body = serde_json::to_vec(&substitutes).map_err(|err| {
                (
                    -32603,
                    format!("encode pipelines_replay substitutes: {err}"),
                )
            })?;
            ("POST".into(), format!("/pipelines/{id}/replay"), Some(body))
        }
        _ => {
            let (method, path) =
                route_for_tool(tool_name, &arguments).map_err(|err| (-32602, err))?;
            (method, path, None)
        }
    };
    let mut request =
        build_request(&method, &path).map_err(|err| (-32603, format!("build request: {err}")))?;
    if let Some(body_bytes) = body {
        request
            .metadata
            .insert("content-type".to_string(), "application/json".to_string());
        request.payload = Bytes::from(body_bytes);
    }
    let response = SendPipe::call(dispatch, request)
        .await
        .map_err(|err| (-32603, format!("dispatch: {err}")))?;
    let status = response.status;
    let body = response
        .collect_body()
        .await
        .map_err(|err| (-32603, format!("read body: {err}")))?;
    let text = String::from_utf8_lossy(&body).to_string();
    if !(200..400).contains(&status) {
        return Err((
            -32603,
            format!("tool `{tool_name}` returned {status}: {text}"),
        ));
    }
    Ok(serde_json::json!({
        "content": [{ "type": "text", "text": text }],
    }))
}

async fn verify_static_tool(arguments: &Value) -> Result<Value, (i32, String)> {
    let spec = resolve_spec_value(arguments)?;
    let policy = resolve_policy(arguments)?;
    let report = crate::verify::verify_static(&spec, &policy);
    Ok(report_to_mcp_content(&report))
}

async fn verify_replay_tool(arguments: &Value) -> Result<Value, (i32, String)> {
    let recording_path = arguments
        .get("recording_path")
        .and_then(Value::as_str)
        .ok_or((-32602, "verify_replay requires `recording_path`".into()))?;
    let policy = resolve_policy(arguments)?;
    // Spec is optional for replay — needed for `idempotence_contract`
    // and `byte_drift`. Try to resolve from explicit args or skip.
    let spec_value = resolve_spec_value(arguments).ok();
    // recording sources offload their blocking file reads onto this runtime's
    // background pool (the read-side seam off tokio::fs).
    let runtime = crate::offline_runtime()
        .map_err(|err| (-32603, format!("verify_replay: build runtime: {err}")))?;
    let report = crate::verify::verify_replay_with_spec(
        std::path::Path::new(recording_path),
        &policy,
        spec_value.as_ref(),
        &runtime,
    )
    .await
    .map_err(|err| (-32603, format!("verify_replay: {err}")))?;
    Ok(report_to_mcp_content(&report))
}

async fn scenario_run_tool(arguments: &Value) -> Result<Value, (i32, String)> {
    let scenario = resolve_scenario_value(arguments)?;
    // run_scenario is ?Send — its closed-loop driver runs `Pipe::call` inline and
    // build_pipes constructs per-core pipes. MCP's dispatcher requires Send, so
    // drive the whole run on a dedicated current-thread tokio runtime via
    // spawn_blocking; the Send wall ends at the JoinHandle. The scenario's own
    // runtime (prime or tokio, chosen internally by offline_runtime) is built and
    // driven separately — its per-request spawns land on ITS worker, not here, so
    // no LocalSet is needed on this outer driver.
    let report = tokio::task::spawn_blocking(move || {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|err| format!("build scenario_run runtime: {err}"))?;
        runtime.block_on(async move {
            let context = crate::load::LoadContext::with_default_registry()
                .map_err(|err| format!("build load context: {err}"))?;
            crate::scenarios::run_scenario(&scenario, &context)
                .await
                .map_err(|err| format!("run_scenario: {err}"))
        })
    })
    .await
    .map_err(|err| (-32603, format!("spawn_blocking: {err}")))?
    .map_err(|err| (-32603, err))?;
    let envelope = serde_json::json!({
        "passed": report.passed(),
        "completed": report.completed,
        "successes": report.successes,
        "failures": report.failures,
        "failed_expectations": report.failed_expectations,
        "windows": report.windows,
        "metrics_snapshot": report.metrics_snapshot,
    });
    let text = serde_json::to_string(&envelope)
        .map_err(|err| (-32603, format!("serialize ScenarioReport: {err}")))?;
    Ok(serde_json::json!({
        "content": [{ "type": "text", "text": text }],
    }))
}

fn resolve_scenario_value(arguments: &Value) -> Result<crate::scenarios::Scenario, (i32, String)> {
    if let Some(inline) = arguments.get("scenario_inline")
        && !inline.is_null()
    {
        return serde_json::from_value(inline.clone())
            .map_err(|err| (-32603, format!("parse scenario_inline: {err}")));
    }
    let path = arguments
        .get("scenario_path")
        .and_then(Value::as_str)
        .ok_or((
            -32602,
            "scenario_run requires `scenario_path` or `scenario_inline`".into(),
        ))?;
    crate::scenarios::Scenario::from_toml_file(path)
        .map_err(|err| (-32603, format!("load scenario {path}: {err}")))
}

fn resolve_spec_value(arguments: &Value) -> Result<Value, (i32, String)> {
    if let Some(inline) = arguments.get("spec_inline")
        && !inline.is_null()
    {
        return Ok(inline.clone());
    }
    let path = arguments.get("spec_path").and_then(Value::as_str).ok_or((
        -32602,
        "verify_static requires `spec_path` or `spec_inline`".into(),
    ))?;
    let text = std::fs::read_to_string(path)
        .map_err(|err| (-32603, format!("read spec {path}: {err}")))?;
    let registry = crate::default_config_format_registry()
        .map_err(|err| (-32603, format!("config formats: {err}")))?;
    let hint = std::path::Path::new(path)
        .extension()
        .and_then(|ext| ext.to_str());
    registry
        .parse_with_hint(&text, hint)
        .map_err(|err| (-32603, format!("parse spec {path}: {err}")))
}

fn resolve_policy(arguments: &Value) -> Result<crate::verify::Policy, (i32, String)> {
    if let Some(inline) = arguments.get("policy_inline")
        && !inline.is_null()
    {
        return serde_json::from_value(inline.clone())
            .map_err(|err| (-32603, format!("parse policy_inline: {err}")));
    }
    if let Some(path) = arguments.get("policy_path").and_then(Value::as_str) {
        return crate::verify::Policy::from_path(path)
            .map_err(|err| (-32603, format!("load policy {path}: {err}")));
    }
    Ok(crate::verify::Policy::default())
}

fn report_to_mcp_content(report: &crate::verify::Report) -> Value {
    let json_text = report
        .emit_json()
        .unwrap_or_else(|err| format!("{{\"error\": \"emit_json failed: {err}\"}}"));
    serde_json::json!({
        "content": [
            { "type": "text", "text": report.emit_text() },
            { "type": "text", "text": json_text },
        ],
    })
}

fn route_for_tool(tool_name: &str, arguments: &Value) -> Result<(String, String), String> {
    match tool_name {
        "list_pipes" => Ok(("GET".into(), "/pipes".into())),
        "pipe_status" => {
            let name = arguments
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| format!("tool `{tool_name}` requires `name` argument"))?;
            Ok(("GET".into(), format!("/pipes/{name}")))
        }
        "metrics_snapshot" => Ok(("GET".into(), "/metrics".into())),
        "pipe_logs" => {
            let name = arguments
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| format!("tool `{tool_name}` requires `name` argument"))?;
            let max_lines = arguments
                .get("max_lines")
                .and_then(Value::as_u64)
                .unwrap_or(100);
            Ok((
                "GET".into(),
                format!("/pipes/{name}/logs?max_lines={max_lines}"),
            ))
        }
        "start_pipe" => {
            let name = arguments
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| format!("tool `{tool_name}` requires `name` argument"))?;
            Ok(("POST".into(), format!("/pipes/{name}/start")))
        }
        "stop_pipe" => {
            let name = arguments
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| format!("tool `{tool_name}` requires `name` argument"))?;
            Ok(("POST".into(), format!("/pipes/{name}/stop")))
        }
        "restart_pipe" => {
            let name = arguments
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| format!("tool `{tool_name}` requires `name` argument"))?;
            Ok(("POST".into(), format!("/pipes/{name}/restart")))
        }
        "daemon_reload" => Ok(("POST".into(), "/reload".into())),
        "pipelines_list" => {
            let mut params: Vec<String> = Vec::new();
            if let Some(name) = arguments.get("name").and_then(Value::as_str) {
                params.push(format!("name={}", url_encode(name)));
            }
            if let Some(hex) = arguments.get("spec_hash_hex").and_then(Value::as_str) {
                params.push(format!("spec_hash_hex={}", url_encode(hex)));
            }
            let path = if params.is_empty() {
                "/pipelines".into()
            } else {
                format!("/pipelines?{}", params.join("&"))
            };
            Ok(("GET".into(), path))
        }
        "pipelines_resolve" => {
            let query = arguments
                .get("query")
                .and_then(Value::as_str)
                .ok_or_else(|| format!("tool `{tool_name}` requires `query` argument"))?;
            Ok((
                "GET".into(),
                format!("/pipelines/resolve?q={}", url_encode(query)),
            ))
        }
        "pipelines_inspect" => {
            let id = arguments
                .get("id")
                .and_then(Value::as_str)
                .ok_or_else(|| format!("tool `{tool_name}` requires `id` argument"))?;
            Ok(("GET".into(), format!("/pipelines/{id}")))
        }
        "pipelines_explain" => {
            let id = arguments
                .get("id")
                .and_then(Value::as_str)
                .ok_or_else(|| format!("tool `{tool_name}` requires `id` argument"))?;
            let stage = arguments
                .get("stage")
                .and_then(Value::as_str)
                .ok_or_else(|| format!("tool `{tool_name}` requires `stage` argument"))?;
            Ok((
                "GET".into(),
                format!("/pipelines/{id}/explain?stage={}", url_encode(stage)),
            ))
        }
        // pipelines_submit and pipelines_replay carry a body — they're handled
        // by tool_call directly (see `body_for_tool`) instead of returning a
        // method/path pair here, since route_for_tool's signature can't carry
        // a body.
        "pipelines_submit" | "pipelines_replay" => Err(format!(
            "{tool_name} handled inline in tool_call (carries a JSON body)"
        )),
        other => Err(format!("unknown tool `{other}`")),
    }
}

fn url_encode(input: &str) -> String {
    let mut output = String::with_capacity(input.len());
    for byte in input.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                output.push(byte as char);
            }
            _ => output.push_str(&format!("%{byte:02X}")),
        }
    }
    output
}

fn build_request(method: &str, path_with_query: &str) -> Result<Request<Bytes>, ProximaError> {
    let context = RequestContext::default();
    let (path, query) = match path_with_query.split_once('?') {
        Some((path, raw)) => {
            let mut list = crate::header_list::HeaderList::new();
            for pair in raw.split('&') {
                if let Some((name, value)) = pair.split_once('=') {
                    list.insert(name.to_string(), value.to_string());
                }
            }
            (path.to_string(), list)
        }
        None => (
            path_with_query.to_string(),
            crate::header_list::HeaderList::new(),
        ),
    };
    Ok(Request {
        method: Method::from(method),
        path: Bytes::from(path),
        query,
        metadata: crate::header_list::HeaderList::new(),
        payload: Bytes::new(),
        stream: None,
        context,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::control_plane::{ControlPlanePipe, PipeState, PipeStatus, StaticControlPlane};
    use crate::pipe::into_handle;
    use serde_json::json;
    use std::sync::Arc;

    fn fixture_handle() -> PipeHandle {
        let plane = Arc::new(StaticControlPlane::new(vec![PipeStatus {
            name: "cart_api".into(),
            state: PipeState::Running,
            uptime_ms: Some(1_234),
            restart_count: 0,
            last_message: None,
        }]));
        into_handle(ControlPlanePipe::new(plane))
    }

    #[proxima::test]
    async fn initialize_returns_protocol_version_and_server_info() {
        let dispatch = fixture_handle();
        let line = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
        })
        .to_string();
        let response = handle_jsonrpc(&line, &dispatch).await.expect("response");
        let parsed: Value = serde_json::from_str(&response).expect("parse");
        assert_eq!(parsed["jsonrpc"], "2.0");
        assert_eq!(parsed["id"], 1);
        assert_eq!(parsed["result"]["protocolVersion"], PROTOCOL_VERSION);
        assert_eq!(parsed["result"]["serverInfo"]["name"], "proxima");
    }

    #[proxima::test]
    async fn tools_list_advertises_control_plane_tools() {
        let dispatch = fixture_handle();
        let line = json!({"jsonrpc":"2.0","id":2,"method":"tools/list"}).to_string();
        let response = handle_jsonrpc(&line, &dispatch).await.expect("response");
        let parsed: Value = serde_json::from_str(&response).expect("parse");
        let tools = parsed["result"]["tools"].as_array().expect("tools array");
        let names: Vec<&str> = tools
            .iter()
            .map(|t| t["name"].as_str().unwrap_or(""))
            .collect();
        assert!(names.contains(&"list_pipes"));
        assert!(names.contains(&"pipe_status"));
        assert!(names.contains(&"metrics_snapshot"));
        assert!(names.contains(&"pipe_logs"));
        assert!(names.contains(&"verify_static"));
        assert!(names.contains(&"verify_replay"));
        assert!(names.contains(&"scenario_run"));
    }

    // exercises the open-loop scenario driver end-to-end through MCP. that
    // driver is tokio-coupled (LocalSet + spawn_local + tokio-thread timer_at),
    // so it requires the tokio runtime. run with `--features runtime-tokio`.
    #[cfg(feature = "runtime-tokio")]
    #[proxima::test]
    async fn scenario_run_inline_open_loop_returns_passed_report() {
        let dispatch = fixture_handle();
        let scenario_inline = json!({
            "workload": {
                "target_pipe": "echo",
                "target_rps": 25,
                "duration": { "secs": 1 },
                "concurrency": 8,
            },
            "pipe": [
                {
                    "name": "echo",
                    "synth": { "status": 200, "body": "ok" }
                }
            ],
            "expect": [
                { "kind": "success_rate_ge", "ratio": 0.95 }
            ]
        });
        let line = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "scenario_run",
                "arguments": { "scenario_inline": scenario_inline }
            }
        })
        .to_string();
        let response = handle_jsonrpc(&line, &dispatch).await.expect("response");
        let parsed: Value = serde_json::from_str(&response).expect("parse");
        let text = parsed["result"]["content"][0]["text"]
            .as_str()
            .expect("content text");
        let report: Value = serde_json::from_str(text).expect("inner report json");
        assert_eq!(report["passed"], Value::Bool(true));
        assert!(
            report["completed"].as_u64().unwrap_or(0) >= 5,
            "expected some completed requests; report={report}"
        );
    }

    #[proxima::test]
    async fn scenario_run_requires_path_or_inline() {
        let dispatch = fixture_handle();
        let line = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": { "name": "scenario_run", "arguments": {} }
        })
        .to_string();
        let response = handle_jsonrpc(&line, &dispatch).await.expect("response");
        let parsed: Value = serde_json::from_str(&response).expect("parse");
        assert!(
            parsed["error"]["message"]
                .as_str()
                .unwrap_or("")
                .contains("scenario_path"),
            "expected error message naming scenario_path; got {parsed}"
        );
    }

    #[proxima::test]
    async fn verify_static_inline_spec_runs_built_in_invariants() {
        let dispatch = fixture_handle();
        let line = json!({
            "jsonrpc": "2.0",
            "id": 100,
            "method": "tools/call",
            "params": {
                "name": "verify_static",
                "arguments": {
                    "spec_inline": {
                        "upstreams": {
                            "origin": {
                                "type": "http",
                                "url": "https://api.example.com",
                                "timeout": "5s"
                            }
                        }
                    }
                }
            }
        })
        .to_string();
        let response = handle_jsonrpc(&line, &dispatch).await.expect("response");
        let parsed: Value = serde_json::from_str(&response).expect("parse");
        let text = parsed["result"]["content"][0]["text"]
            .as_str()
            .expect("text content");
        assert!(text.contains("PASS no_cycles"), "got: {text}");
        assert!(
            text.contains("PASS all_upstreams_have_timeouts"),
            "got: {text}"
        );
    }

    #[proxima::test]
    async fn verify_static_cycle_reports_fail_via_mcp() {
        let dispatch = fixture_handle();
        let line = json!({
            "jsonrpc": "2.0",
            "id": 101,
            "method": "tools/call",
            "params": {
                "name": "verify_static",
                "arguments": {
                    "spec_inline": {
                        "pipes": {
                            "a": { "chain": ["b"] },
                            "b": { "chain": ["a"] }
                        }
                    }
                }
            }
        })
        .to_string();
        let response = handle_jsonrpc(&line, &dispatch).await.expect("response");
        let parsed: Value = serde_json::from_str(&response).expect("parse");
        let text = parsed["result"]["content"][0]["text"]
            .as_str()
            .expect("text content");
        assert!(text.contains("FAIL no_cycles"), "got: {text}");
    }

    #[proxima::test]
    async fn verify_static_requires_spec_path_or_inline() {
        let dispatch = fixture_handle();
        let line = json!({
            "jsonrpc": "2.0",
            "id": 102,
            "method": "tools/call",
            "params": { "name": "verify_static", "arguments": {} }
        })
        .to_string();
        let response = handle_jsonrpc(&line, &dispatch).await.expect("response");
        let parsed: Value = serde_json::from_str(&response).expect("parse");
        assert!(parsed.get("error").is_some(), "expected error: {parsed}");
    }

    #[proxima::test]
    async fn tools_call_list_pipes_returns_running_pipe() {
        let dispatch = fixture_handle();
        let line = json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": { "name": "list_pipes", "arguments": {} },
        })
        .to_string();
        let response = handle_jsonrpc(&line, &dispatch).await.expect("response");
        let parsed: Value = serde_json::from_str(&response).expect("parse");
        let text = parsed["result"]["content"][0]["text"]
            .as_str()
            .expect("text");
        assert!(text.contains("cart_api"), "text: {text}");
        assert!(text.contains("running"), "text: {text}");
    }

    #[proxima::test]
    async fn tools_call_pipe_status_with_name_arg_returns_status_json() {
        let dispatch = fixture_handle();
        let line = json!({
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": { "name": "pipe_status", "arguments": { "name": "cart_api" } },
        })
        .to_string();
        let response = handle_jsonrpc(&line, &dispatch).await.expect("response");
        let parsed: Value = serde_json::from_str(&response).expect("parse");
        let text = parsed["result"]["content"][0]["text"]
            .as_str()
            .expect("text");
        assert!(text.contains("\"name\""));
        assert!(text.contains("cart_api"));
    }

    #[proxima::test]
    async fn tools_call_unknown_tool_returns_jsonrpc_error() {
        let dispatch = fixture_handle();
        let line = json!({
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": { "name": "definitely_not_a_tool", "arguments": {} },
        })
        .to_string();
        let response = handle_jsonrpc(&line, &dispatch).await.expect("response");
        let parsed: Value = serde_json::from_str(&response).expect("parse");
        assert!(parsed.get("error").is_some(), "expected error: {parsed}");
        let code = parsed["error"]["code"].as_i64().expect("code");
        assert_eq!(code, -32602);
    }

    #[proxima::test]
    async fn unknown_method_returns_method_not_found() {
        let dispatch = fixture_handle();
        let line = json!({"jsonrpc":"2.0","id":6,"method":"prompts/list"}).to_string();
        let response = handle_jsonrpc(&line, &dispatch).await.expect("response");
        let parsed: Value = serde_json::from_str(&response).expect("parse");
        assert_eq!(parsed["error"]["code"], -32601);
    }

    #[proxima::test]
    async fn malformed_json_returns_parse_error_with_null_id() {
        let dispatch = fixture_handle();
        let response = handle_jsonrpc("{not json", &dispatch)
            .await
            .expect("response");
        let parsed: Value = serde_json::from_str(&response).expect("parse");
        assert_eq!(parsed["error"]["code"], -32700);
        assert_eq!(parsed["id"], Value::Null);
    }

    #[proxima::test]
    async fn notification_without_id_returns_no_response() {
        let dispatch = fixture_handle();
        let line = json!({"jsonrpc":"2.0","method":"notifications/initialized"}).to_string();
        let response = handle_jsonrpc(&line, &dispatch).await;
        assert!(response.is_none());
    }
}
