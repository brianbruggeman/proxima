use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::process::{ExitStatus, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::time::Duration;

#[cfg(feature = "sync-wrappers")]
use crate::sync::Mutex;
use bon::Builder;
use bytes::Bytes;
use conflaguration::{Settings, Validate, ValidationMessage};
use futures::{FutureExt, select};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, ChildStderr, ChildStdout, Command};
#[cfg(not(feature = "sync-wrappers"))]
use tokio::sync::Mutex;
use tokio::sync::{mpsc, watch};
use tokio::task::JoinHandle;
use tracing::{debug, error, warn};

use proxima_primitives::pipe::SendPipe;

use crate::error::ProximaError;
use crate::log_buffer::{DEFAULT_LOG_BUFFER_CAPACITY, LogBuffer, LogBufferRegistry};
use crate::pipe::{PipeHandle, into_handle};
use crate::pipe_factory::PipeFactory;
use crate::request::{Request, Response};

#[derive(Debug, Clone)]
pub struct ProcessSpec {
    pub command: String,
    pub args: Vec<String>,
    pub working_dir: Option<std::path::PathBuf>,
    pub env: BTreeMap<String, String>,
    pub restart: RestartPolicy,
    pub restart_delay_ms: u64,
    pub max_restart_attempts: u32,
    pub ready_probe: ReadyProbe,
    pub shutdown_signal: ShutdownSignal,
    pub shutdown_timeout_ms: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestartPolicy {
    Always,
    Never,
    OnFailure,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShutdownSignal {
    Term,
    Int,
    Kill,
}

#[derive(Debug, Clone)]
pub enum ReadyProbe {
    None,
    StdoutLine {
        pattern: String,
        timeout_ms: u64,
    },
    Tcp {
        addr: std::net::SocketAddr,
        timeout_ms: u64,
    },
}

pub struct ProcessUpstream {
    label: String,
    state: Arc<ProcessState>,
    shutdown_tx: mpsc::Sender<()>,
    log_buffer_registry: Option<Arc<LogBufferRegistry>>,
}

struct ProcessState {
    spec: ProcessSpec,
    child: Mutex<Option<Child>>,
    shutting_down: AtomicBool,
    restart_attempts: AtomicU32,
    label: String,
    log_buffer: Arc<LogBuffer>,
    stderr_buffer: Arc<LogBuffer>,
    exit_tx: watch::Sender<Option<ExitStatus>>,
    /// The current generation's stdout/stderr drain tasks (see
    /// `spawn_stdout_drain`/`spawn_stderr_drain`). `run_supervisor` joins
    /// these right after observing the child exit, before firing
    /// `exit_tx` — otherwise a caller can see the exit before trailing
    /// output the child wrote right before its pipes closed has made it
    /// into `log_buffer`/`stderr_buffer` (the two are independently
    /// scheduled tasks with no ordering guarantee against `Child::wait`).
    stdout_task: Mutex<Option<JoinHandle<()>>>,
    stderr_task: Mutex<Option<JoinHandle<()>>>,
}

impl ProcessUpstream {
    pub async fn spawn(label: impl Into<String>, spec: ProcessSpec) -> Result<Self, ProximaError> {
        Self::spawn_with_log_buffers(label, spec, None).await
    }

    pub async fn spawn_with_log_buffers(
        label: impl Into<String>,
        spec: ProcessSpec,
        log_buffer_registry: Option<Arc<LogBufferRegistry>>,
    ) -> Result<Self, ProximaError> {
        Self::spawn_with_buffers(label, spec, None, None, log_buffer_registry).await
    }

    /// Spawn with caller-supplied log buffers. Use this when a consumer
    /// (e.g. `ProcessEventBridge`) needs to subscribe to the buffers
    /// before any process I/O lands — passing a pre-built buffer that the
    /// consumer already subscribed to closes the subscribe-vs-drain race.
    pub async fn spawn_with_buffers(
        label: impl Into<String>,
        spec: ProcessSpec,
        stdout_buffer: Option<Arc<LogBuffer>>,
        stderr_buffer: Option<Arc<LogBuffer>>,
        log_buffer_registry: Option<Arc<LogBufferRegistry>>,
    ) -> Result<Self, ProximaError> {
        let label = label.into();
        let log_buffer =
            stdout_buffer.unwrap_or_else(|| Arc::new(LogBuffer::new(DEFAULT_LOG_BUFFER_CAPACITY)));
        let stderr_buffer =
            stderr_buffer.unwrap_or_else(|| Arc::new(LogBuffer::new(DEFAULT_LOG_BUFFER_CAPACITY)));
        if let Some(registry) = log_buffer_registry.as_ref() {
            registry.register(label.clone(), log_buffer.clone());
        }
        let (mut child, captured_stdout, captured_stderr) = launch(&spec, &label).await?;
        let (stdout_task, stderr_task) = await_ready_with_buffer(
            &spec.ready_probe,
            &mut child,
            captured_stdout,
            captured_stderr,
            &label,
            log_buffer.clone(),
            stderr_buffer.clone(),
        )
        .await?;
        let (exit_tx, _exit_rx) = watch::channel(None);
        let state = Arc::new(ProcessState {
            spec,
            child: Mutex::new(Some(child)),
            shutting_down: AtomicBool::new(false),
            restart_attempts: AtomicU32::new(0),
            label: label.clone(),
            log_buffer: log_buffer.clone(),
            stderr_buffer: stderr_buffer.clone(),
            exit_tx,
            stdout_task: Mutex::new(Some(stdout_task)),
            stderr_task: Mutex::new(Some(stderr_task)),
        });
        let (shutdown_tx, shutdown_rx) = mpsc::channel::<()>(1);
        let supervisor_state = state.clone();
        tokio::spawn(async move { run_supervisor(supervisor_state, shutdown_rx).await });
        Ok(Self {
            label,
            state,
            shutdown_tx,
            log_buffer_registry,
        })
    }

    #[must_use]
    pub fn log_buffer(&self) -> Arc<LogBuffer> {
        self.state.log_buffer.clone()
    }

    #[must_use]
    pub fn stderr_buffer(&self) -> Arc<LogBuffer> {
        self.state.stderr_buffer.clone()
    }

    #[must_use]
    pub fn spec(&self) -> &ProcessSpec {
        &self.state.spec
    }

    /// Subscribe to terminal exit notification. The watch channel fires
    /// `Some(status)` when the supervisor decides not to restart and the
    /// child has exited; for `RestartPolicy::Never` that's the first
    /// exit. Sender lives on the state — receivers stay valid for the
    /// lifetime of the upstream.
    #[must_use]
    pub fn subscribe_exit(&self) -> watch::Receiver<Option<ExitStatus>> {
        self.state.exit_tx.subscribe()
    }

    /// Convenience: await one terminal exit. Returns `None` if the
    /// supervisor terminated without observing an exit (shutdown
    /// before launch / relaunch failure).
    pub async fn wait_for_exit(&self) -> Option<ExitStatus> {
        let mut rx = self.subscribe_exit();
        loop {
            if let Some(status) = *rx.borrow() {
                return Some(status);
            }
            if rx.changed().await.is_err() {
                return *rx.borrow();
            }
        }
    }
}

impl Drop for ProcessUpstream {
    fn drop(&mut self) {
        self.state.shutting_down.store(true, Ordering::SeqCst);
        let _ = self.shutdown_tx.try_send(());
        if let Some(registry) = self.log_buffer_registry.as_ref() {
            let _ = registry.deregister(&self.label);
        }
        // best-effort kill — supervisor task takes the rest. holding a Mutex
        // across drop is fine because Drop is sync; the lock is contention-free
        // by the time we get here.
        if let Ok(mut guard) = self.state.child.try_lock()
            && let Some(child) = guard.as_mut()
        {
            let _ = child.start_kill();
        }
    }
}

impl SendPipe for ProcessUpstream {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        _request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> {
        let label = self.label.clone();
        async move {
            // a process upstream is supervisor-only by default. fronting it with HTTP
            // is composed by stacking an `http` upstream pointing at the process's listen
            // addr. that keeps this factory single-purpose: own the lifecycle, expose
            // liveness, let other factories handle protocol.
            Ok(Response::ok(format!("process `{label}` is alive")))
        }
    }
}


async fn launch(
    spec: &ProcessSpec,
    label: &str,
) -> Result<(Child, ChildStdout, ChildStderr), ProximaError> {
    let mut command = Command::new(&spec.command);
    command.args(&spec.args);
    if let Some(dir) = &spec.working_dir {
        command.current_dir(dir);
    }
    for (key, value) in &spec.env {
        command.env(key, value);
    }
    command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    let mut child = command.spawn().map_err(|err| {
        ProximaError::Upstream(format!("spawn `{label}` ({}): {err}", spec.command))
    })?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| ProximaError::Upstream(format!("process `{label}` has no stdout pipe")))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| ProximaError::Upstream(format!("process `{label}` has no stderr pipe")))?;
    debug!(label, command = %spec.command, "process spawned");
    Ok((child, stdout, stderr))
}

async fn await_ready_with_buffer(
    probe: &ReadyProbe,
    child: &mut Child,
    stdout: ChildStdout,
    stderr: ChildStderr,
    label: &str,
    log_buffer: Arc<LogBuffer>,
    stderr_buffer: Arc<LogBuffer>,
) -> Result<(JoinHandle<()>, JoinHandle<()>), ProximaError> {
    // stderr always drains in the background; the ready probe is a stdout-only signal today.
    let stderr_task = spawn_stderr_drain(stderr, stderr_buffer);
    let stdout_task = match probe {
        ReadyProbe::None => spawn_stdout_drain(stdout, log_buffer),
        ReadyProbe::StdoutLine {
            pattern,
            timeout_ms,
        } => {
            let pattern = pattern.clone();
            let label_owned = label.to_string();
            let log_buffer_for_probe = log_buffer.clone();
            let probe_future = async move {
                let mut reader = BufReader::new(stdout).lines();
                while let Some(line) = reader
                    .next_line()
                    .await
                    .map_err(|err| ProximaError::Upstream(format!("read stdout: {err}")))?
                {
                    log_buffer_for_probe.push(line.clone());
                    if line.contains(&pattern) {
                        debug!(label = %label_owned, %pattern, "ready probe matched");
                        // continue draining stdout in the background, still pushing into the buffer.
                        let buffer_for_drain = log_buffer_for_probe.clone();
                        let continuation = tokio::spawn(async move {
                            while let Ok(Some(line)) = reader.next_line().await {
                                buffer_for_drain.push(line);
                            }
                        });
                        return Ok(continuation);
                    }
                }
                Err(ProximaError::Upstream(format!(
                    "process `{label_owned}` exited before stdout matched `{pattern}`"
                )))
            };
            match tokio::time::timeout(Duration::from_millis(*timeout_ms), probe_future).await {
                Ok(result) => result?,
                Err(_) => {
                    let _ = child.start_kill();
                    return Err(ProximaError::Timeout(Duration::from_millis(*timeout_ms)));
                }
            }
        }
        ReadyProbe::Tcp { addr, timeout_ms } => {
            let stdout_task = spawn_stdout_drain(stdout, log_buffer);
            let deadline = std::time::Instant::now() + Duration::from_millis(*timeout_ms);
            loop {
                if tokio::net::TcpStream::connect(*addr).await.is_ok() {
                    break stdout_task;
                }
                if std::time::Instant::now() >= deadline {
                    let _ = child.start_kill();
                    return Err(ProximaError::Timeout(Duration::from_millis(*timeout_ms)));
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        }
    };
    Ok((stdout_task, stderr_task))
}

fn spawn_stdout_drain(stdout: ChildStdout, log_buffer: Arc<LogBuffer>) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut reader = BufReader::new(stdout).lines();
        while let Ok(Some(line)) = reader.next_line().await {
            log_buffer.push(line);
        }
    })
}

fn spawn_stderr_drain(stderr: ChildStderr, log_buffer: Arc<LogBuffer>) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut reader = BufReader::new(stderr).lines();
        while let Ok(Some(line)) = reader.next_line().await {
            log_buffer.push(line);
        }
    })
}

async fn run_supervisor(state: Arc<ProcessState>, mut shutdown_rx: mpsc::Receiver<()>) {
    let mut last_exit: Option<ExitStatus> = None;
    loop {
        let exit = wait_or_shutdown(&state, &mut shutdown_rx).await;
        if let Ok(status) = &exit {
            last_exit = Some(*status);
        }
        // the child's pipes close at/around exit, independently of this
        // `wait()` resolving — join the current drain generation now so
        // any exit_tx fired below (or a restart's fresh spawn) can't race
        // ahead of trailing stdout/stderr still being pushed to the buffers.
        if let Some(task) = state.stdout_task.lock().await.take() {
            let _ = task.await;
        }
        if let Some(task) = state.stderr_task.lock().await.take() {
            let _ = task.await;
        }
        if state.shutting_down.load(Ordering::SeqCst) {
            let _ = state.exit_tx.send(last_exit);
            return;
        }
        let should_restart = match (state.spec.restart, &exit) {
            (RestartPolicy::Always, _) => true,
            (RestartPolicy::Never, _) => false,
            (RestartPolicy::OnFailure, Ok(status)) => !status.success(),
            (RestartPolicy::OnFailure, Err(_)) => true,
        };
        if !should_restart {
            debug!(label = %state.label, "process exited; restart policy declines");
            let _ = state.exit_tx.send(last_exit);
            return;
        }
        let attempts = state.restart_attempts.fetch_add(1, Ordering::SeqCst) + 1;
        if state.spec.max_restart_attempts > 0 && attempts > state.spec.max_restart_attempts {
            warn!(
                label = %state.label,
                attempts,
                "process exceeded max_restart_attempts; supervisor stopping"
            );
            let _ = state.exit_tx.send(last_exit);
            return;
        }
        if state.spec.restart_delay_ms > 0 {
            tokio::time::sleep(Duration::from_millis(state.spec.restart_delay_ms)).await;
        }
        match launch(&state.spec, &state.label).await {
            Ok((child, stdout, stderr)) => {
                let stdout_task = spawn_stdout_drain(stdout, state.log_buffer.clone());
                let stderr_task = spawn_stderr_drain(stderr, state.stderr_buffer.clone());
                *state.stdout_task.lock().await = Some(stdout_task);
                *state.stderr_task.lock().await = Some(stderr_task);
                let mut guard = state.child.lock().await;
                *guard = Some(child);
            }
            Err(error) => {
                error!(label = %state.label, ?error, "supervisor relaunch failed");
                let _ = state.exit_tx.send(last_exit);
                return;
            }
        }
    }
}

async fn wait_or_shutdown(
    state: &ProcessState,
    shutdown_rx: &mut mpsc::Receiver<()>,
) -> Result<std::process::ExitStatus, std::io::Error> {
    let mut guard = state.child.lock().await;
    let Some(child) = guard.as_mut() else {
        return Err(std::io::Error::other("no child to wait on"));
    };
    select! {
        outcome = child.wait().fuse() => outcome,
        _ = shutdown_rx.recv().fuse() => {
            let _ = child.start_kill();
            child.wait().await
        }
    }
}

/// Serialisable restart policy — the config mirror of [`RestartPolicy`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RestartChoice {
    Always,
    #[default]
    Never,
    OnFailure,
}

impl From<RestartChoice> for RestartPolicy {
    fn from(choice: RestartChoice) -> Self {
        match choice {
            RestartChoice::Always => RestartPolicy::Always,
            RestartChoice::Never => RestartPolicy::Never,
            RestartChoice::OnFailure => RestartPolicy::OnFailure,
        }
    }
}

/// Serialisable shutdown signal — the config mirror of [`ShutdownSignal`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShutdownSignalChoice {
    #[default]
    Term,
    Int,
    Kill,
}

impl From<ShutdownSignalChoice> for ShutdownSignal {
    fn from(choice: ShutdownSignalChoice) -> Self {
        match choice {
            ShutdownSignalChoice::Term => ShutdownSignal::Term,
            ShutdownSignalChoice::Int => ShutdownSignal::Int,
            ShutdownSignalChoice::Kill => ShutdownSignal::Kill,
        }
    }
}

fn default_ready_timeout_ms() -> u64 {
    5_000
}

/// Serialisable readiness probe — the config mirror of [`ReadyProbe`]. Tagged
/// by `type`; the `tcp` variant parses its `addr` to a [`SocketAddr`] at
/// [`ProcessConfig::into_spec`] time so the wire form stays a plain string.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ProbeConfig {
    #[default]
    None,
    StdoutLine {
        pattern: String,
        #[serde(default = "default_ready_timeout_ms")]
        timeout_ms: u64,
    },
    Tcp {
        addr: String,
        #[serde(default = "default_ready_timeout_ms")]
        timeout_ms: u64,
    },
}

impl ProbeConfig {
    fn into_probe(self) -> Result<ReadyProbe, ProximaError> {
        match self {
            ProbeConfig::None => Ok(ReadyProbe::None),
            ProbeConfig::StdoutLine {
                pattern,
                timeout_ms,
            } => Ok(ReadyProbe::StdoutLine {
                pattern,
                timeout_ms,
            }),
            ProbeConfig::Tcp { addr, timeout_ms } => {
                let parsed = addr.parse::<std::net::SocketAddr>().map_err(|err| {
                    ProximaError::Config(format!("ready_probe tcp invalid addr `{addr}`: {err}"))
                })?;
                Ok(ReadyProbe::Tcp {
                    addr: parsed,
                    timeout_ms,
                })
            }
        }
    }
}

fn default_label() -> String {
    "process".to_string()
}

fn default_restart_delay_ms() -> u64 {
    1_000
}

fn default_shutdown_timeout_ms() -> u64 {
    5_000
}

/// Typed config surface for the `process` upstream — a supervised child
/// process with a readiness probe and restart policy. Mirrors [`ProcessSpec`]
/// (whose `working_dir` is a `PathBuf` and `ready_probe`/`restart` are runtime
/// enums); the `name` label is carried here too (the factory read it off the
/// same spec historically).
#[derive(Debug, Clone, Builder, Deserialize, Serialize, Settings)]
#[settings(prefix = "PROXIMA_PROCESS")]
#[builder(derive(Clone, Debug), on(String, into))]
pub struct ProcessConfig {
    /// Executable to launch.
    pub command: String,

    /// Pipe / supervisor label.
    #[setting(default = "process")]
    #[serde(default = "default_label")]
    #[builder(default = default_label())]
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

    /// Restart policy. Defaults to `never`.
    #[setting(skip)]
    #[serde(default)]
    #[builder(default)]
    pub restart: RestartChoice,

    /// Delay before a restart, in ms. Defaults to 1000.
    #[setting(default = 1000)]
    #[serde(default = "default_restart_delay_ms")]
    #[builder(default = default_restart_delay_ms())]
    pub restart_delay_ms: u64,

    /// Restart attempt ceiling (0 = unlimited).
    #[setting(default)]
    #[serde(default)]
    #[builder(default)]
    pub max_restart_attempts: u32,

    /// Readiness probe. Defaults to none (ready immediately).
    #[setting(skip)]
    #[serde(default)]
    #[builder(default)]
    pub ready_probe: ProbeConfig,

    /// Signal sent on shutdown. Defaults to `term`.
    #[setting(skip)]
    #[serde(default)]
    #[builder(default)]
    pub shutdown_signal: ShutdownSignalChoice,

    /// Grace period after the shutdown signal before kill, in ms.
    #[setting(default = 5000)]
    #[serde(default = "default_shutdown_timeout_ms")]
    #[builder(default = default_shutdown_timeout_ms())]
    pub shutdown_timeout_ms: u64,
}

impl Validate for ProcessConfig {
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

impl ProcessConfig {
    /// Lower the wire config to the runtime [`ProcessSpec`].
    pub fn into_spec(self) -> Result<ProcessSpec, ProximaError> {
        self.validate()
            .map_err(|err| ProximaError::Config(format!("{err}")))?;
        Ok(ProcessSpec {
            command: self.command,
            args: self.args,
            working_dir: self.working_dir.map(std::path::PathBuf::from),
            env: self.env,
            restart: self.restart.into(),
            restart_delay_ms: self.restart_delay_ms,
            max_restart_attempts: self.max_restart_attempts,
            ready_probe: self.ready_probe.into_probe()?,
            shutdown_signal: self.shutdown_signal.into(),
            shutdown_timeout_ms: self.shutdown_timeout_ms,
        })
    }
}

pub struct ProcessPipeFactory {
    log_buffers: Option<Arc<LogBufferRegistry>>,
}

impl ProcessPipeFactory {
    #[must_use]
    pub fn new() -> Self {
        Self { log_buffers: None }
    }

    #[must_use]
    pub fn with_log_buffer_registry(log_buffers: Arc<LogBufferRegistry>) -> Self {
        Self {
            log_buffers: Some(log_buffers),
        }
    }
}

impl Default for ProcessPipeFactory {
    fn default() -> Self {
        Self::new()
    }
}

impl PipeFactory for ProcessPipeFactory {
    fn name(&self) -> &str {
        "process"
    }

    fn build(
        &self,
        spec: &Value,
        _inner: Option<PipeHandle>,
    ) -> Pin<Box<dyn Future<Output = Result<PipeHandle, ProximaError>> + Send + '_>> {
        let spec = spec.clone();
        let log_buffers = self.log_buffers.clone();
        Box::pin(async move {
            let config: ProcessConfig = serde_json::from_value(spec)
                .map_err(|err| ProximaError::Config(format!("process config: {err}")))?;
            let label = config.name.clone();
            let parsed = config.into_spec()?;
            let upstream =
                ProcessUpstream::spawn_with_log_buffers(label, parsed, log_buffers).await?;
            Ok(into_handle(upstream))
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

    // principle-4 parity: the fluent builder and the config value must lower to
    // identical ProcessSpec state (command/args/dirs/policy/probe/signals).
    #[test]
    fn parity_fluent_builder_and_config_value_match() {
        let from_value: ProcessConfig = serde_json::from_value(json!({
            "name": "svc",
            "command": "server",
            "args": ["--port", "8080"],
            "working_dir": "/srv",
            "env": {"LOG": "info"},
            "restart": "on_failure",
            "restart_delay_ms": 250,
            "max_restart_attempts": 3,
            "ready_probe": {"type": "tcp", "addr": "127.0.0.1:8080", "timeout_ms": 1500},
            "shutdown_signal": "int",
            "shutdown_timeout_ms": 2000,
        }))
        .expect("from_value");
        let from_value = from_value.into_spec().expect("into_spec value");

        let mut env = BTreeMap::new();
        env.insert("LOG".to_string(), "info".to_string());
        let from_builder = ProcessConfig::builder()
            .name("svc")
            .command("server")
            .args(vec!["--port".to_string(), "8080".to_string()])
            .working_dir("/srv".to_string())
            .env(env)
            .restart(RestartChoice::OnFailure)
            .restart_delay_ms(250)
            .max_restart_attempts(3)
            .ready_probe(ProbeConfig::Tcp {
                addr: "127.0.0.1:8080".to_string(),
                timeout_ms: 1500,
            })
            .shutdown_signal(ShutdownSignalChoice::Int)
            .shutdown_timeout_ms(2000)
            .build()
            .into_spec()
            .expect("into_spec builder");

        assert_eq!(from_value.command, from_builder.command);
        assert_eq!(from_value.args, from_builder.args);
        assert_eq!(from_value.working_dir, from_builder.working_dir);
        assert_eq!(from_value.env, from_builder.env);
        assert_eq!(from_value.restart, from_builder.restart);
        assert_eq!(from_value.restart_delay_ms, from_builder.restart_delay_ms);
        assert_eq!(
            from_value.max_restart_attempts,
            from_builder.max_restart_attempts
        );
        assert_eq!(from_value.shutdown_signal, from_builder.shutdown_signal);
        assert_eq!(
            from_value.shutdown_timeout_ms,
            from_builder.shutdown_timeout_ms
        );
        match (&from_value.ready_probe, &from_builder.ready_probe) {
            (
                ReadyProbe::Tcp {
                    addr: left_addr,
                    timeout_ms: left_ms,
                },
                ReadyProbe::Tcp {
                    addr: right_addr,
                    timeout_ms: right_ms,
                },
            ) => {
                assert_eq!(left_addr, right_addr);
                assert_eq!(left_ms, right_ms);
            }
            other => panic!("expected matching tcp probes, got {other:?}"),
        }
    }

    #[proxima::test]
    async fn spawns_then_ready_probe_matches_stdout_line() {
        let factory = ProcessPipeFactory::new();
        let handle = factory
            .build(
                &json!({
                    "name": "echo_ready",
                    "command": shell(),
                    "args": [shell_arg(), "echo READY; sleep 5"],
                    "ready_probe": {"type": "stdout_line", "pattern": "READY", "timeout_ms": 2000},
                    "restart": "never",
                }),
                None,
            )
            .await
            .expect("spawn");
        let response = SendPipe::call(
            &handle,
            Request::builder()
                .method("GET")
                .path("/")
                .build()
                .expect("request"),
        )
        .await
        .expect("call");
        assert_eq!(response.status, 200);
    }

    #[proxima::test]
    async fn missing_ready_pattern_within_timeout_returns_typed_error() {
        let factory = ProcessPipeFactory::new();
        let outcome = factory
            .build(
                &json!({
                    "name": "never_ready",
                    "command": shell(),
                    "args": [shell_arg(), "sleep 10"],
                    "ready_probe": {"type": "stdout_line", "pattern": "READY", "timeout_ms": 200},
                    "restart": "never",
                }),
                None,
            )
            .await;
        assert!(matches!(
            outcome,
            Err(ProximaError::Timeout(_)) | Err(ProximaError::Upstream(_))
        ));
    }

    #[proxima::test]
    async fn missing_command_returns_config_error() {
        let factory = ProcessPipeFactory::new();
        let outcome = factory.build(&json!({"name": "no_cmd"}), None).await;
        assert!(matches!(outcome, Err(ProximaError::Config(_))));
    }

    #[proxima::test]
    async fn ready_probe_none_returns_immediately() {
        let factory = ProcessPipeFactory::new();
        let handle = factory
            .build(
                &json!({
                    "name": "no_probe",
                    "command": shell(),
                    "args": [shell_arg(), "sleep 5"],
                    "ready_probe": {"type": "none"},
                    "restart": "never",
                }),
                None,
            )
            .await
            .expect("spawn");
        // `name` config threading now only surfaces at the mount site
        // (App::mount's MountTarget::Named), not on the raw handle — the
        // handle-level name/name_dyn() surface was deleted (TARGET 3).
        // This test's remaining assertion is that a `ready_probe: none`
        // config spawns immediately without error.
        let _ = handle;
    }

    #[proxima::test]
    async fn unknown_restart_policy_returns_config_error() {
        let factory = ProcessPipeFactory::new();
        let outcome = factory
            .build(
                &json!({
                    "name": "bad_policy",
                    "command": shell(),
                    "args": [shell_arg(), "true"],
                    "restart": "lolwhat",
                }),
                None,
            )
            .await;
        assert!(matches!(outcome, Err(ProximaError::Config(_))));
    }

    #[proxima::test]
    async fn unknown_ready_probe_type_returns_config_error() {
        let factory = ProcessPipeFactory::new();
        let outcome = factory
            .build(
                &json!({
                    "name": "bad_probe",
                    "command": shell(),
                    "args": [shell_arg(), "true"],
                    "ready_probe": {"type": "smoke_signal"},
                }),
                None,
            )
            .await;
        assert!(matches!(outcome, Err(ProximaError::Config(_))));
    }
}
