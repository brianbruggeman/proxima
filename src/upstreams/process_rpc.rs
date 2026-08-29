use std::collections::BTreeMap;
use std::future::Future;
use std::io;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use bon::Builder;
use bytes::Bytes;
use conflaguration::{Settings, Validate, ValidationMessage};
use proxima_codec::{DelimiterCodec, DelimiterFraming};
use proxima_core::time;
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::sync::{AsyncMutex, AsyncMutexGuard, oneshot};
use proxima_process::{Child, Command, Stdio};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::ProximaError;
use crate::pipe::{PipeHandle, into_handle};
use crate::pipe_factory::PipeFactory;
use crate::request::{Request, Response};

/// Newline-framed stdin/stdout pipe to a subprocess. One request per
/// `\n` write + one line read; protocol agnostic (JSON-RPC, MCP, etc.).
/// Serialized through an `AsyncMutex` — one in-flight call per upstream
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
    child: AsyncMutex<Option<RpcChild>>,
}

struct RpcChild {
    // `Option` so `Drop` can move the handle onto a background reap
    // thread; `None` only in the instant between that move and RpcChild's
    // own destruction.
    child: Option<Child>,
    stdin: OwnedFd,
    stdout: OwnedFd,
    framing: DelimiterFraming,
}

impl RpcChild {
    fn child_mut(&mut self) -> Result<&mut Child, ProximaError> {
        self.child.as_mut().ok_or_else(|| {
            ProximaError::Upstream("process_rpc: subprocess handle already taken".into())
        })
    }
}

impl Drop for RpcChild {
    fn drop(&mut self) {
        let Some(mut child) = self.child.take() else {
            return;
        };
        let _ = child.kill();
        // reap on a background thread — waitpid(2) is a blocking syscall
        // and must not run inline in Drop. Child::wait caches the exit
        // code once reaped, so if this pid was already reaped elsewhere
        // (ensure_alive's try_wait, the timeout path) this returns
        // instantly instead of risking a waitpid on a recycled pid.
        thread::spawn(move || {
            let _ = child.wait();
        });
    }
}

impl ProcessRpcUpstream {
    pub fn new(label: impl Into<String>, spec: ProcessRpcSpec) -> Self {
        Self {
            label: label.into(),
            state: Arc::new(RpcState {
                spec,
                label: String::new(),
                child: AsyncMutex::new(None),
            }),
        }
    }
}

impl Drop for ProcessRpcUpstream {
    fn drop(&mut self) {
        // take() removes the RpcChild from the guard and drops it here,
        // which runs RpcChild::drop (kill + background reap) exactly once.
        if let Some(mut guard) = self.state.child.try_lock() {
            guard.take();
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
            let outcome = time::timeout(timeout, dispatch_one(rpc_child, body)).await;
            match outcome {
                Ok(Ok(line)) => Ok(Response::ok(line)),
                Ok(Err(error)) => {
                    // dropping the old RpcChild here runs its Drop impl:
                    // kill + background reap.
                    *guard = None;
                    Err(error)
                }
                Err(_elapsed) => {
                    *guard = None;
                    Err(ProximaError::Timeout(timeout))
                }
            }
        }
    }
}

async fn ensure_alive(
    state: &Arc<RpcState>,
    guard: &mut AsyncMutexGuard<'_, Option<RpcChild>>,
) -> Result<(), ProximaError> {
    if let Some(rpc_child) = guard.as_mut() {
        match rpc_child.child_mut()?.try_wait() {
            Ok(Some(_code)) => {
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
        .stderr(Stdio::inherit());
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
        child: Some(child),
        stdin,
        stdout,
        framing: DelimiterFraming::new(),
    })
}

async fn dispatch_one(rpc_child: &mut RpcChild, body: Bytes) -> Result<Bytes, ProximaError> {
    let stdin_fd = rpc_child.stdin.as_raw_fd();
    let stdout_fd = rpc_child.stdout.as_raw_fd();
    let needs_newline = !body.ends_with(b"\n");
    let framing = std::mem::replace(&mut rpc_child.framing, DelimiterFraming::new());

    // duped so the worker owns independent descriptors: `time::timeout`
    // cancels this future, not the OS thread, and a timed-out `call` drops
    // RpcChild (closing its fds) while the orphaned worker may still be
    // blocked in read/write on the same fd numbers.
    let stdin_owned = dup_fd(stdin_fd, "stdin")?;
    let stdout_owned = dup_fd(stdout_fd, "stdout")?;

    let (sender, receiver) = oneshot::channel();
    thread::spawn(move || {
        let stdin_fd = stdin_owned.as_raw_fd();
        let stdout_fd = stdout_owned.as_raw_fd();
        let outcome = blocking_round_trip(stdin_fd, stdout_fd, &body, needs_newline, framing);
        let _ = sender.send(outcome);
    });
    let outcome = receiver.await.unwrap_or_else(|_| {
        Err(ProximaError::Upstream(
            "process_rpc round-trip worker thread dropped".into(),
        ))
    });

    match outcome {
        Ok((line, framing)) => {
            rpc_child.framing = framing;
            Ok(line)
        }
        Err(error) => Err(error),
    }
}

// one background thread owns the whole write-then-read round trip for a
// single call — mirrors fd_pipe.rs's "spawn once per call, not per syscall"
// discipline. Safe because the AsyncMutex already serializes calls per
// child, so nothing else touches these fds while this thread runs.
fn blocking_round_trip(
    stdin_fd: RawFd,
    stdout_fd: RawFd,
    body: &[u8],
    needs_newline: bool,
    mut framing: DelimiterFraming,
) -> Result<(Bytes, DelimiterFraming), ProximaError> {
    write_all_raw(stdin_fd, body)
        .map_err(|err| ProximaError::Upstream(format!("write rpc body: {err}")))?;
    if needs_newline {
        write_all_raw(stdin_fd, b"\n")
            .map_err(|err| ProximaError::Upstream(format!("write rpc newline: {err}")))?;
    }

    let codec = DelimiterCodec::unbounded(b"\n");
    loop {
        let (next_state, frame) = framing
            .next_frame(codec)
            .map_err(|err| ProximaError::Upstream(format!("read rpc reply: {err}")))?;
        framing = next_state;
        if let Some(line) = frame {
            return Ok((strip_trailing_cr(line), framing));
        }
        let chunk = read_chunk_raw(stdout_fd)
            .map_err(|err| ProximaError::Upstream(format!("read rpc reply: {err}")))?;
        if chunk.is_empty() {
            return Err(ProximaError::Upstream(
                "process_rpc subprocess closed stdout".into(),
            ));
        }
        framing = framing.push(&chunk);
    }
}

fn dup_fd(raw_fd: RawFd, purpose: &str) -> Result<OwnedFd, ProximaError> {
    // SAFETY: raw_fd is an open descriptor owned by the caller for the
    // duration of this call; dup(2) either fails or returns a fresh fd.
    let duplicated = unsafe { libc::dup(raw_fd) };
    if duplicated < 0 {
        let error = io::Error::last_os_error();
        return Err(ProximaError::Upstream(format!(
            "process_rpc dup {purpose}: {error}"
        )));
    }
    // SAFETY: duplicated was just returned by dup(2) above and is not owned
    // by anything else.
    Ok(unsafe { OwnedFd::from_raw_fd(duplicated) })
}

fn strip_trailing_cr(mut line: Bytes) -> Bytes {
    if line.last() == Some(&b'\r') {
        line.truncate(line.len() - 1);
    }
    line
}

fn write_all_raw(raw_fd: RawFd, buffer: &[u8]) -> io::Result<()> {
    let mut written_total = 0;
    while written_total < buffer.len() {
        // SAFETY: raw_fd is a pipe fd owned by the caller's RpcChild for the
        // duration of this call; the slice outlives the syscall.
        let written = unsafe {
            libc::write(
                raw_fd,
                buffer[written_total..].as_ptr().cast(),
                buffer.len() - written_total,
            )
        };
        if written < 0 {
            let error = io::Error::last_os_error();
            if error.kind() == io::ErrorKind::Interrupted {
                continue;
            }
            return Err(error);
        }
        written_total += written as usize;
    }
    Ok(())
}

fn read_chunk_raw(raw_fd: RawFd) -> io::Result<Vec<u8>> {
    let mut buffer = [0_u8; 4096];
    loop {
        // SAFETY: raw_fd is a pipe fd owned by the caller's RpcChild; buffer
        // is stack-local and sized for the requested read length.
        let bytes_read = unsafe { libc::read(raw_fd, buffer.as_mut_ptr().cast(), buffer.len()) };
        if bytes_read < 0 {
            let error = io::Error::last_os_error();
            if error.kind() == io::ErrorKind::Interrupted {
                continue;
            }
            return Err(error);
        }
        return Ok(buffer[..bytes_read as usize].to_vec());
    }
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
    use std::sync::mpsc;

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

    // reproduces the use-after-close race directly: without dup_fd, the
    // worker would still hold the *number* `original_read` after the parent
    // closes it, and a subsequent open() reusing that number would hand the
    // worker's next syscall someone else's file. This drives that exact
    // sequence with channels (never a sleep) so it is deterministic under
    // nextest's process-per-test isolation.
    #[test]
    fn duped_fd_outlives_original_and_ignores_recycled_number() {
        let mut original = [0_i32; 2];
        let pipe_result = unsafe { libc::pipe(original.as_mut_ptr()) };
        assert_eq!(pipe_result, 0, "pipe: {}", io::Error::last_os_error());
        let original_read = original[0];
        let original_write = original[1];

        let duped_read = dup_fd(original_read, "stdout").expect("dup_fd");

        let (worker_ready_tx, worker_ready_rx) = mpsc::channel::<()>();
        let (proceed_tx, proceed_rx) = mpsc::channel::<()>();
        let (result_tx, result_rx) = mpsc::channel::<Vec<u8>>();

        let worker = thread::spawn(move || {
            worker_ready_tx.send(()).expect("signal ready");
            proceed_rx.recv().expect("wait for proceed");
            let mut buffer = [0_u8; 32];
            // SAFETY: duped_read is a live fd owned by this thread.
            let bytes_read = unsafe {
                libc::read(
                    duped_read.as_raw_fd(),
                    buffer.as_mut_ptr().cast(),
                    buffer.len(),
                )
            };
            assert!(bytes_read >= 0, "read: {}", io::Error::last_os_error());
            result_tx
                .send(buffer[..bytes_read as usize].to_vec())
                .expect("send result");
        });

        worker_ready_rx.recv().expect("worker ready");

        // mirrors `*guard = None` dropping RpcChild's OwnedFd on the timeout
        // path, while the worker above is still holding its own duped copy.
        let closed = unsafe { libc::close(original_read) };
        assert_eq!(closed, 0, "close: {}", io::Error::last_os_error());

        let mut recycled = [0_i32; 2];
        let recycled_result = unsafe { libc::pipe(recycled.as_mut_ptr()) };
        assert_eq!(recycled_result, 0, "pipe: {}", io::Error::last_os_error());
        assert_eq!(
            recycled[0], original_read,
            "test assumption violated: kernel did not recycle the freed fd number"
        );
        let sentinel = b"wrong pipe";
        write_all_raw(recycled[1], sentinel).expect("write sentinel into recycled fd");

        let genuine = b"right pipe";
        write_all_raw(original_write, genuine).expect("write to the real pipe");

        proceed_tx.send(()).expect("release worker");
        let observed = result_rx.recv().expect("worker result");
        worker.join().expect("worker thread panicked");

        assert_eq!(
            &observed[..],
            genuine,
            "worker must read the genuine pipe through its duped fd, never the sentinel \
             written into the recycled fd number"
        );

        unsafe {
            libc::close(original_write);
            libc::close(recycled[0]);
            libc::close(recycled[1]);
        }
    }
}
