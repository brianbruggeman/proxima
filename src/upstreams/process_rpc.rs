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
    pub init: Vec<ProcessRpcInitFrame>,
}

/// One line written to a freshly spawned child before the call's own
/// request; `reply` says whether exactly one reply line is read and
/// discarded. Runs through the same framing as [`ProcessRpcUpstream`] calls.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct ProcessRpcInitFrame {
    /// The line to write, without its trailing newline.
    pub send: String,

    /// Read and discard one reply line after writing.
    #[serde(default = "default_reply")]
    pub reply: bool,
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
        let label = label.into();
        Self {
            state: Arc::new(RpcState {
                spec,
                label: label.clone(),
                child: AsyncMutex::new(None),
            }),
            label,
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
            let outcome = time::timeout(timeout, dispatch_one(rpc_child, body, true)).await;
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
    if child_is_live(state, guard)? {
        return Ok(());
    }
    let mut child = spawn_child(&state.spec, &state.label)?;
    run_init(state, &mut child).await?;
    **guard = Some(child);
    Ok(())
}

fn child_is_live(state: &RpcState, slot: &mut Option<RpcChild>) -> Result<bool, ProximaError> {
    let Some(rpc_child) = slot.as_mut() else {
        return Ok(false);
    };
    match rpc_child.child_mut()?.try_wait() {
        Ok(None) => Ok(true),
        Ok(Some(_code)) if !state.spec.restart => {
            Err(state_error(state, "subprocess exited and restart=false"))
        }
        Ok(Some(_code)) => {
            *slot = None;
            Ok(false)
        }
        Err(error) => {
            *slot = None;
            Err(state_error(state, &format!("try_wait: {error}")))
        }
    }
}

fn state_error(state: &RpcState, detail: &str) -> ProximaError {
    ProximaError::Upstream(format!("process_rpc `{}`: {detail}", state.label))
}

async fn run_init(state: &RpcState, rpc_child: &mut RpcChild) -> Result<(), ProximaError> {
    let timeout = Duration::from_millis(state.spec.request_timeout_ms);
    for (index, frame) in state.spec.init.iter().enumerate() {
        let body = Bytes::from(frame.send.clone());
        let outcome = time::timeout(timeout, dispatch_one(rpc_child, body, frame.reply)).await;
        let detail = match outcome {
            Ok(Ok(_discarded)) => continue,
            Ok(Err(ProximaError::Upstream(text))) => text,
            Ok(Err(other)) => other.to_string(),
            Err(_elapsed) => format!("timed out after {timeout:?}"),
        };
        return Err(ProximaError::Upstream(format!(
            "process_rpc `{}` init frame {index}: {detail}",
            state.label
        )));
    }
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

type RoundTrip = Result<(Bytes, DelimiterFraming), ProximaError>;

async fn dispatch_one(
    rpc_child: &mut RpcChild,
    body: Bytes,
    expect_reply: bool,
) -> Result<Bytes, ProximaError> {
    let receiver = spawn_round_trip(rpc_child, body, expect_reply)?;
    let outcome = receiver.await.unwrap_or_else(|_| {
        Err(ProximaError::Upstream(
            "process_rpc round-trip worker thread dropped".into(),
        ))
    });
    let (line, framing) = outcome?;
    rpc_child.framing = framing;
    Ok(line)
}

// one background thread owns the whole write-then-read round trip for a
// single call — mirrors fd_pipe.rs's "spawn once per call, not per syscall"
// discipline. Safe because the AsyncMutex already serializes calls per
// child, so nothing else touches these fds while this thread runs.
// the fds are duped so the worker owns independent descriptors: `time::timeout`
// cancels this future, not the OS thread, and a timed-out `call` drops
// RpcChild (closing its fds) while the orphaned worker may still be
// blocked in read/write on the same fd numbers.
fn spawn_round_trip(
    rpc_child: &mut RpcChild,
    body: Bytes,
    expect_reply: bool,
) -> Result<oneshot::Receiver<RoundTrip>, ProximaError> {
    let needs_newline = !body.ends_with(b"\n");
    let framing = std::mem::replace(&mut rpc_child.framing, DelimiterFraming::new());
    let stdin_owned = dup_fd(rpc_child.stdin.as_raw_fd(), "stdin")?;
    let stdout_owned = dup_fd(rpc_child.stdout.as_raw_fd(), "stdout")?;
    let (sender, receiver) = oneshot::channel();
    thread::spawn(move || {
        let sent = write_request(stdin_owned.as_raw_fd(), &body, needs_newline);
        let outcome = sent.and_then(|()| match expect_reply {
            true => read_reply(stdout_owned.as_raw_fd(), framing),
            false => Ok((Bytes::new(), framing)),
        });
        let _ = sender.send(outcome);
    });
    Ok(receiver)
}

fn write_request(stdin_fd: RawFd, body: &[u8], needs_newline: bool) -> Result<(), ProximaError> {
    write_all_raw(stdin_fd, body)
        .map_err(|err| ProximaError::Upstream(format!("write rpc body: {err}")))?;
    if needs_newline {
        write_all_raw(stdin_fd, b"\n")
            .map_err(|err| ProximaError::Upstream(format!("write rpc newline: {err}")))?;
    }
    Ok(())
}

fn read_reply(stdout_fd: RawFd, mut framing: DelimiterFraming) -> RoundTrip {
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

fn default_reply() -> bool {
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

    /// Frames written to every freshly spawned child before the call's own request.
    #[setting(skip)]
    #[serde(default)]
    #[builder(default)]
    pub init: Vec<ProcessRpcInitFrame>,
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
            init: self.init,
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
    // sequence with channels (never a sleep), pinning the recycled number with
    // dup2 so it is deterministic however many sibling threads hold fds.
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
        // dup2 swaps the number in one syscall: it is never free, so a sibling
        // test thread cannot claim it the way it can after a close-then-pipe.
        let mut recycled = [0_i32; 2];
        let recycled_result = unsafe { libc::pipe(recycled.as_mut_ptr()) };
        assert_eq!(recycled_result, 0, "pipe: {}", io::Error::last_os_error());
        let replaced = unsafe { libc::dup2(recycled[0], original_read) };
        assert_eq!(
            replaced,
            original_read,
            "dup2: {}",
            io::Error::last_os_error()
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
            libc::close(original_read);
            libc::close(original_write);
            libc::close(recycled[0]);
            libc::close(recycled[1]);
        }
    }

    const STUB: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/mcp_stub/tracker.sh"
    );
    const INITIALIZE: &str = r#"{"jsonrpc":"2.0","id":0,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"proxima","version":"0"}}}"#;
    const INITIALIZED: &str = r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#;
    const TOOLS_CALL: &str = r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"get_ticket","arguments":{"id":"PROXIMA-24"}}}"#;
    const REPLY_READY: &str = r#"{"jsonrpc":"2.0","id":1,"result":{"content":[{"type":"text","text":"inits=1"}],"structuredContent":{"ticket":"PROXIMA-24","repo":"brianbruggeman/proxima","pr":42}}}"#;
    const REPLY_NOT_READY: &str =
        r#"{"jsonrpc":"2.0","id":1,"error":{"code":-32002,"message":"server not initialized"}}"#;
    const EXIT: &str = r#"{"jsonrpc":"2.0","id":9,"method":"exit"}"#;
    const HANG: &str = r#"{"jsonrpc":"2.0","id":5,"method":"hang"}"#;
    const INIT_ERROR: &str =
        "process_rpc `tracker` init frame 0: process_rpc subprocess closed stdout";
    const INIT_ERROR_NO_LABEL: &str =
        "process_rpc `` init frame 0: process_rpc subprocess closed stdout";

    fn check_reply(body: &[u8], expected: &str) -> Result<(), String> {
        if body == expected.as_bytes() {
            return Ok(());
        }
        Err(format!(
            "reply {:?} is not {expected:?}",
            String::from_utf8_lossy(body)
        ))
    }

    fn check_init_error(error: &str, label: &str, index: usize) -> Result<(), String> {
        let named = format!("process_rpc `{label}`");
        let frame = format!("init frame {index}");
        if error.contains(&named) && error.contains(&frame) {
            return Ok(());
        }
        Err(format!("error {error:?} lacks {named:?} or {frame:?}"))
    }

    fn handshake() -> Value {
        json!([{"send": INITIALIZE}, {"send": INITIALIZED, "reply": false}])
    }

    async fn load_rpc(fields: Value) -> PipeHandle {
        let context = crate::load::LoadContext::with_default_registry().expect("default context");
        let spec = json!({"process_rpc": fields});
        crate::load::load(spec, &context)
            .await
            .expect("load process_rpc spec")
    }

    async fn call_line(handle: &PipeHandle, line: &str) -> Result<Bytes, ProximaError> {
        let request = Request::builder()
            .method("POST")
            .path("/")
            .body(line.to_string())
            .build()
            .expect("request");
        SendPipe::call(handle, request).await?.collect_body().await
    }

    fn stub_spec(init: Value) -> Value {
        json!({"name": "tracker", "command": "/bin/sh", "args": [STUB], "init": init})
    }

    #[proxima::test]
    async fn process_rpc_init_runs_handshake_before_first_call() {
        let handle = load_rpc(stub_spec(handshake())).await;
        let body = call_line(&handle, TOOLS_CALL).await.expect("call");
        check_reply(&body, REPLY_READY).expect("initialized reply");
    }

    #[proxima::test]
    async fn process_rpc_init_runs_once_per_live_child() {
        let handle = load_rpc(stub_spec(handshake())).await;
        for _ in 0..2 {
            let body = call_line(&handle, TOOLS_CALL).await.expect("call");
            check_reply(&body, REPLY_READY).expect("inits=1 on both calls");
        }
    }

    #[proxima::test]
    async fn process_rpc_init_runs_again_on_respawned_child() {
        let handle = load_rpc(stub_spec(handshake())).await;
        call_line(&handle, EXIT)
            .await
            .expect_err("exit closes stdout");
        let body = call_line(&handle, TOOLS_CALL)
            .await
            .expect("call after respawn");
        check_reply(&body, REPLY_READY).expect("respawned child initialized");
    }

    #[proxima::test]
    async fn process_rpc_init_fails_when_child_exits_before_reply() {
        let handle = load_rpc(json!({
            "name": "tracker", "command": "/bin/sh", "args": ["-c", "exit 0"],
            "init": [{"send": INITIALIZE}],
        }))
        .await;
        let error = call_line(&handle, TOOLS_CALL)
            .await
            .expect_err("init must fail");
        check_init_error(&error.to_string(), "tracker", 0).expect("frame 0 named");
    }

    #[proxima::test]
    async fn process_rpc_init_fails_when_frame_times_out() {
        let mut fields = stub_spec(json!([{"send": INITIALIZE}, {"send": HANG}]));
        fields["request_timeout_ms"] = json!(200);
        let handle = load_rpc(fields).await;
        let error = call_line(&handle, TOOLS_CALL)
            .await
            .expect_err("init must time out");
        check_init_error(&error.to_string(), "tracker", 1).expect("frame 1 named");
    }

    #[proxima::test]
    async fn process_rpc_init_fails_then_respawned_child_initializes() {
        let dir = tempfile::tempdir().expect("tempdir");
        let marker = dir.path().join("marker");
        let script = "if [ -e \"$0\" ]; then exec /bin/sh \"$1\"; fi; : > \"$0\"";
        let handle = load_rpc(json!({
            "name": "tracker", "command": "/bin/sh",
            "args": ["-c", script, marker.to_str().expect("utf8 path"), STUB],
            "init": handshake(),
        }))
        .await;
        let error = call_line(&handle, TOOLS_CALL)
            .await
            .expect_err("first child exits");
        check_init_error(&error.to_string(), "tracker", 0).expect("frame 0 named");
        let body = call_line(&handle, TOOLS_CALL).await.expect("second call");
        check_reply(&body, REPLY_READY).expect("respawned child initialized");
    }

    #[proxima::test]
    async fn ctl_rpc_without_init_stub_refuses_tools_call() {
        let handle = load_rpc(stub_spec(json!([]))).await;
        let body = call_line(&handle, TOOLS_CALL).await.expect("call");
        check_reply(&body, REPLY_NOT_READY).expect("stub refuses before initialize");
        check_reply(&body, REPLY_READY).expect_err("helper rejects the refusal");
    }

    #[test]
    fn ctl_rpc_check_reply_accepts_equal_bytes() {
        check_reply(REPLY_READY.as_bytes(), REPLY_READY).expect("equal bytes");
    }

    #[test]
    fn ctl_rpc_check_init_error_accepts_label_and_index() {
        check_init_error(INIT_ERROR, "tracker", 0).expect("label and index present");
    }

    #[test]
    fn ctl_rpc_check_init_error_rejects_wrong_index() {
        check_init_error(INIT_ERROR, "tracker", 1).expect_err("index is checked");
    }

    #[test]
    fn ctl_rpc_check_init_error_rejects_empty_label() {
        check_init_error(INIT_ERROR_NO_LABEL, "tracker", 0).expect_err("empty label caught");
    }
}
