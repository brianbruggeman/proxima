use std::collections::BTreeMap;
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

#[cfg(feature = "sync-wrappers")]
use crate::sync::{Mutex, MutexGuard};
use bon::Builder;
use bytes::Bytes;
use conflaguration::{Settings, Validate, ValidationMessage};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
#[cfg(not(feature = "sync-wrappers"))]
use tokio::sync::{Mutex, MutexGuard};

use proxima_primitives::pipe::SendPipe;

use crate::error::ProximaError;
use crate::pipe::{PipeHandle, into_handle};
use crate::pipe_factory::PipeFactory;
use crate::request::{Request, Response};

/// Newline-framed stdin/stdout pipe to a subprocess. One request per
/// `\n` write + one line read; protocol agnostic (JSON-RPC, MCP, etc.).
/// Serialized through a tokio Mutex — one in-flight call per upstream
/// because stdin/stdout are shared. On death the next call respawns
/// (when `restart = true`); Drop kills + reaps.
pub struct ProcessRpcUpstream {
    label: String,
    state: Arc<RpcState>,
}

#[derive(Debug, Clone)]
pub struct ProcessRpcSpec {
    pub command: String,
    pub args: Vec<String>,
    pub working_dir: Option<PathBuf>,
    pub env: BTreeMap<String, String>,
    pub restart: bool,
    pub request_timeout_ms: u64,
}

struct RpcState {
    spec: ProcessRpcSpec,
    label: String,
    child: Mutex<Option<RpcChild>>,
}

struct RpcChild {
    process: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl ProcessRpcUpstream {
    pub fn new(label: impl Into<String>, spec: ProcessRpcSpec) -> Self {
        Self {
            label: label.into(),
            state: Arc::new(RpcState {
                spec,
                label: String::new(),
                child: Mutex::new(None),
            }),
        }
    }
}

impl Drop for ProcessRpcUpstream {
    fn drop(&mut self) {
        if let Ok(mut guard) = self.state.child.try_lock()
            && let Some(rpc_child) = guard.as_mut()
        {
            let _ = rpc_child.process.start_kill();
        }
    }
}

impl SendPipe for ProcessRpcUpstream {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> {
        let state = self.state.clone();
        let label = self.label.clone();
        async move {
            let (_, body) = request.body_bytes().await?;
            let mut guard = state.child.lock().await;
            ensure_alive(&state, &mut guard).await?;
            let rpc_child = guard.as_mut().ok_or_else(|| {
                ProximaError::Upstream(format!(
                    "process_rpc `{label}`: subprocess unexpectedly absent after spawn"
                ))
            })?;
            let timeout = Duration::from_millis(state.spec.request_timeout_ms);
            let outcome = tokio::time::timeout(timeout, dispatch_one(rpc_child, &body)).await;
            match outcome {
                Ok(Ok(line)) => Ok(Response::ok(line)),
                Ok(Err(error)) => {
                    *guard = None;
                    Err(error)
                }
                Err(_elapsed) => {
                    let _ = rpc_child.process.start_kill();
                    *guard = None;
                    Err(ProximaError::Timeout(timeout))
                }
            }
        }
    }
}


async fn ensure_alive(
    state: &Arc<RpcState>,
    guard: &mut MutexGuard<'_, Option<RpcChild>>,
) -> Result<(), ProximaError> {
    if let Some(rpc_child) = guard.as_mut() {
        match rpc_child.process.try_wait() {
            Ok(Some(_status)) => {
                if !state.spec.restart {
                    return Err(ProximaError::Upstream(format!(
                        "process_rpc `{}`: subprocess exited and restart=false",
                        state.label,
                    )));
                }
                **guard = None;
            }
            Ok(None) => return Ok(()),
            Err(error) => {
                **guard = None;
                return Err(ProximaError::Upstream(format!(
                    "process_rpc `{}`: try_wait: {error}",
                    state.label,
                )));
            }
        }
    }
    let child = spawn_child(&state.spec, &state.label)?;
    **guard = Some(child);
    Ok(())
}

fn spawn_child(spec: &ProcessRpcSpec, label: &str) -> Result<RpcChild, ProximaError> {
    let mut command = Command::new(&spec.command);
    command.args(&spec.args);
    if let Some(dir) = &spec.working_dir {
        command.current_dir(dir);
    }
    for (key, value) in &spec.env {
        command.env(key, value);
    }
    command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .kill_on_drop(true);
    let mut child = command.spawn().map_err(|err| {
        ProximaError::Upstream(format!(
            "process_rpc `{label}` spawn ({}): {err}",
            spec.command
        ))
    })?;
    let stdin = child.stdin.take().ok_or_else(|| {
        ProximaError::Upstream(format!("process_rpc `{label}` has no stdin pipe"))
    })?;
    let stdout = child.stdout.take().ok_or_else(|| {
        ProximaError::Upstream(format!("process_rpc `{label}` has no stdout pipe"))
    })?;
    Ok(RpcChild {
        process: child,
        stdin,
        stdout: BufReader::new(stdout),
    })
}

async fn dispatch_one(rpc_child: &mut RpcChild, body: &[u8]) -> Result<Bytes, ProximaError> {
    rpc_child
        .stdin
        .write_all(body)
        .await
        .map_err(|err| ProximaError::Upstream(format!("write rpc body: {err}")))?;
    if !body.ends_with(b"\n") {
        rpc_child
            .stdin
            .write_all(b"\n")
            .await
            .map_err(|err| ProximaError::Upstream(format!("write rpc newline: {err}")))?;
    }
    rpc_child
        .stdin
        .flush()
        .await
        .map_err(|err| ProximaError::Upstream(format!("flush rpc stdin: {err}")))?;
    let mut line = String::new();
    let bytes_read = rpc_child
        .stdout
        .read_line(&mut line)
        .await
        .map_err(|err| ProximaError::Upstream(format!("read rpc reply: {err}")))?;
    if bytes_read == 0 {
        return Err(ProximaError::Upstream(
            "process_rpc subprocess closed stdout".into(),
        ));
    }
    if line.ends_with('\n') {
        line.pop();
        if line.ends_with('\r') {
            line.pop();
        }
    }
    Ok(Bytes::from(line.into_bytes()))
}

pub struct ProcessRpcPipeFactory;

impl PipeFactory for ProcessRpcPipeFactory {
    fn name(&self) -> &str {
        "process_rpc"
    }

    fn build(
        &self,
        spec: &Value,
        _inner: Option<PipeHandle>,
    ) -> Pin<Box<dyn Future<Output = Result<PipeHandle, ProximaError>> + Send + '_>> {
        let spec = spec.clone();
        Box::pin(async move {
            let config: ProcessRpcConfig = serde_json::from_value(spec)
                .map_err(|err| ProximaError::Config(format!("process_rpc config: {err}")))?;
            let label = config.name.clone();
            let parsed = config.into_spec()?;
            let upstream = ProcessRpcUpstream::new(label, parsed);
            Ok(into_handle(upstream))
        })
    }
}

fn default_rpc_label() -> String {
    "process_rpc".to_string()
}

fn default_restart() -> bool {
    true
}

fn default_request_timeout_ms() -> u64 {
    30_000
}

/// Typed config surface for the `process_rpc` upstream — a newline-framed
/// stdin/stdout RPC to a subprocess. Mirrors [`ProcessRpcSpec`] (whose
/// `working_dir` is a `PathBuf`); the `name` label is carried here too.
#[derive(Debug, Clone, Builder, Deserialize, Serialize, Settings)]
#[settings(prefix = "PROXIMA_PROCESS_RPC")]
#[builder(derive(Clone, Debug), on(String, into))]
pub struct ProcessRpcConfig {
    /// Executable to launch.
    pub command: String,

    /// Handler / upstream label.
    #[setting(default = "process_rpc")]
    #[serde(default = "default_rpc_label")]
    #[builder(default = default_rpc_label())]
    pub name: String,

    /// Arguments passed to `command`.
    #[setting(skip)]
    #[serde(default)]
    #[builder(default)]
    pub args: Vec<String>,

    /// Working directory; inherited from the proxy when absent.
    #[setting(default)]
    #[serde(default)]
    pub working_dir: Option<String>,

    /// Extra environment variables for the child.
    #[setting(skip)]
    #[serde(default)]
    #[builder(default)]
    pub env: BTreeMap<String, String>,

    /// Respawn the subprocess on death before the next call. Defaults to true.
    #[setting(default = true)]
    #[serde(default = "default_restart")]
    #[builder(default = default_restart())]
    pub restart: bool,

    /// Per-request timeout in ms. Defaults to 30000.
    #[setting(default = 30000)]
    #[serde(default = "default_request_timeout_ms")]
    #[builder(default = default_request_timeout_ms())]
    pub request_timeout_ms: u64,
}

impl Validate for ProcessRpcConfig {
    fn validate(&self) -> conflaguration::Result<()> {
        let mut errors = Vec::new();
        if self.command.is_empty() {
            errors.push(ValidationMessage::new("command", "must not be empty"));
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(conflaguration::Error::Validation { errors })
        }
    }
}

impl ProcessRpcConfig {
    /// Lower the wire config to the runtime [`ProcessRpcSpec`].
    pub fn into_spec(self) -> Result<ProcessRpcSpec, ProximaError> {
        self.validate()
            .map_err(|err| ProximaError::Config(format!("{err}")))?;
        Ok(ProcessRpcSpec {
            command: self.command,
            args: self.args,
            working_dir: self.working_dir.map(PathBuf::from),
            env: self.env,
            restart: self.restart,
            request_timeout_ms: self.request_timeout_ms,
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use serde_json::json;

    fn shell() -> &'static str {
        if cfg!(windows) { "cmd" } else { "sh" }
    }

    fn shell_arg() -> &'static str {
        if cfg!(windows) { "/c" } else { "-c" }
    }

    async fn build(spec: Value) -> PipeHandle {
        let factory = ProcessRpcPipeFactory;
        factory.build(&spec, None).await.expect("build")
    }

    // principle-4 parity: the fluent builder and the config value must lower to
    // identical ProcessRpcSpec state (command/args/dir/env/restart/timeout).
    #[test]
    fn parity_fluent_builder_and_config_value_match() {
        let from_value: ProcessRpcConfig = serde_json::from_value(json!({
            "name": "rpc",
            "command": "server",
            "args": ["--stdio"],
            "working_dir": "/srv",
            "env": {"MODE": "rpc"},
            "restart": false,
            "request_timeout_ms": 5000,
        }))
        .expect("from_value");
        let from_value = from_value.into_spec().expect("into_spec value");

        let mut env = BTreeMap::new();
        env.insert("MODE".to_string(), "rpc".to_string());
        let from_builder = ProcessRpcConfig::builder()
            .name("rpc")
            .command("server")
            .args(vec!["--stdio".to_string()])
            .working_dir("/srv".to_string())
            .env(env)
            .restart(false)
            .request_timeout_ms(5000)
            .build()
            .into_spec()
            .expect("into_spec builder");

        assert_eq!(from_value.command, from_builder.command);
        assert_eq!(from_value.args, from_builder.args);
        assert_eq!(from_value.working_dir, from_builder.working_dir);
        assert_eq!(from_value.env, from_builder.env);
        assert_eq!(from_value.restart, from_builder.restart);
        assert_eq!(
            from_value.request_timeout_ms,
            from_builder.request_timeout_ms
        );
    }

    #[proxima::test]
    async fn echoes_request_body_through_subprocess_stdio() {
        let handle = build(json!({
            "name": "echo",
            "command": shell(),
            "args": [shell_arg(), "while read line; do echo $line; done"],
        }))
        .await;
        let request = Request::builder()
            .method("POST")
            .path("/")
            .body("hello")
            .build()
            .expect("request");
        let response = SendPipe::call(&handle, request).await.expect("call");
        assert_eq!(response.status, 200);
        let body = response.collect_body().await.expect("collect");
        assert_eq!(&body[..], b"hello");
    }

    #[proxima::test]
    async fn round_trips_jsonrpc_envelope() {
        // a tiny shell-script "mcp server" that echoes the request envelope back
        // wrapped in {"jsonrpc":"2.0","id":N,"result":<body>}.
        let handle = build(json!({
            "name": "fake_mcp",
            "command": shell(),
            "args": [
                shell_arg(),
                "while read line; do echo \"{\\\"jsonrpc\\\":\\\"2.0\\\",\\\"id\\\":1,\\\"result\\\":$line}\"; done",
            ],
        }))
        .await;
        let payload = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
        });
        let request = Request::builder()
            .method("POST")
            .path("/")
            .body(serde_json::to_vec(&payload).expect("encode"))
            .build()
            .expect("request");
        let response = SendPipe::call(&handle, request).await.expect("call");
        let body = response.collect_body().await.expect("collect");
        let parsed: Value = serde_json::from_slice(&body).expect("parse reply");
        assert_eq!(parsed["jsonrpc"], "2.0");
        assert_eq!(parsed["id"], 1);
        assert_eq!(parsed["result"]["method"], "tools/list");
    }

    #[proxima::test]
    async fn keeps_subprocess_alive_across_multiple_calls() {
        let handle = build(json!({
            "name": "counter",
            "command": shell(),
            "args": [
                shell_arg(),
                "n=0; while read line; do n=$((n+1)); echo \"n=$n line=$line\"; done",
            ],
        }))
        .await;
        for index in 0..3 {
            let request = Request::builder()
                .method("POST")
                .path("/")
                .body(format!("call-{index}"))
                .build()
                .expect("request");
            let response = SendPipe::call(&handle, request).await.expect("call");
            let body = response.collect_body().await.expect("collect");
            let text = String::from_utf8_lossy(&body);
            assert!(
                text.starts_with(&format!("n={}", index + 1)),
                "text: {text}"
            );
            assert!(
                text.ends_with(&format!("line=call-{index}")),
                "text: {text}"
            );
        }
    }

    #[proxima::test]
    async fn timeout_returns_typed_error_and_respawns_subprocess() {
        let handle = build(json!({
            "name": "slow",
            "command": shell(),
            "args": [shell_arg(), "sleep 30"],
            "request_timeout_ms": 100,
        }))
        .await;
        let request = Request::builder()
            .method("POST")
            .path("/")
            .body("ping")
            .build()
            .expect("request");
        let outcome = SendPipe::call(&handle, request).await;
        assert!(matches!(outcome, Err(ProximaError::Timeout(_))));
    }

    #[proxima::test]
    async fn missing_command_returns_config_error() {
        let factory = ProcessRpcPipeFactory;
        let outcome = factory.build(&json!({"name": "no_cmd"}), None).await;
        assert!(matches!(outcome, Err(ProximaError::Config(_))));
    }
}
