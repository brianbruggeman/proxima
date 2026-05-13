//! Std-tier [`Command`] — drop-in mirror of
//! [`std::process::Command`].
//!
//! Goal: `use proxima_process::Command;` is interchangeable with
//! `use std::process::Command;` for any code using the fluent API.
//! Same setter signatures (`AsRef<OsStr>`, `AsRef<Path>`,
//! `Into<Stdio>`); same terminal methods (`spawn`, `output`,
//! `status`). Tokio and async-std mirror std on the setter side
//! too, so code written for any of the three compiles unmodified
//! against this `Command`.
//!
//! # Additive surface (not in std)
//!
//! These extras don't break drop-in — std users never call them:
//!
//! - [`Command::dispatch`] — attach a sandbox chain
//!   (`SendPipe<In = ChildRequest, Out = ChildResponse>`).
//!   Default is no chain (vanilla spawn).
//! - [`Command::libc_shim`] — opt into the co-located libc-
//!   interpose `.dylib`/`.so`. Default off.
//! - [`Command::controlling_tty`] / [`Command::umask`] —
//!   Unix-specific knobs std handles via `pre_exec` closures.
//! - [`Command::inherit_current_env`] — explicit env-snapshot
//!   opt-in (std snapshots implicitly unless you call
//!   `env_clear`; we make it explicit to match the alloc-tier
//!   [`CommandDescriptor`] discipline).
//! - `impl Pipe for Command` — call as a `proxima_primitives::pipe::Pipe`:
//!   request body → child stdin, response body ← child stdout.
//!
//! # Storage / lowering
//!
//! Fields use [`OsString`] / [`PathBuf`] for drop-in compat.
//! Lowering to the alloc-tier [`CommandDescriptor`] happens at
//! spawn time via [`Command::to_descriptor`]; NUL bytes in any
//! string produce `Err(ProximaError)` — matches std's behaviour
//! of deferring conversion errors to `spawn()`.
//!
//! # G1–G9 contract
//!
//! `Command` MUST NOT carry any of the
//! [`WithoutFilesystem`/`Network`/`Spawn`/`Time`/`Random`]
//! markers, must NOT impl `Deterministic`/`IsPure`/`NoStd`/
//! `AllocFree`. Spawning a subprocess is the wholly unconstrained
//! ground: filesystem, network, spawn, time, random, allocation,
//! std — all on the table. Asserting absence here would break
//! every chain that depends on `WithoutSpawn`. The regression test
//! at `tests/type_system_guarantees.rs` will fail to compile if a
//! future change accidentally adds one of those bounds.
use bytes::Bytes;

use std::ffi::{OsStr, OsString};
use std::future::Future;
use std::os::fd::RawFd;
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::thread;

use alloc::ffi::CString;
use bon::Builder;
use futures::stream::{self, StreamExt};
use proxima_primitives::pipe::alloc_tier;
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::{ProximaError, Request, Response, ResponseStream};

use super::descriptor::{CommandDescriptor, Stdio};
use super::dispatched::{DispatchedChild, spawn_and_dispatch_with_options};
use super::env::Env;
use super::fd_pipe::FdPairPipe;
use super::libc_shim;
use super::protocol::{ChildRequest, ChildResponse};
use super::spawn::{Child, SpawnOptions, spawn};

/// Drop-in mirror of [`std::process::Command`].
///
/// ```ignore
/// use proxima_process::{Command, Stdio};
///
/// let mut cmd = Command::new("/bin/ls");
/// cmd.arg("-la")
///    .current_dir("/tmp")
///    .stdout(Stdio::piped());
/// let child = cmd.spawn()?;
/// ```
///
/// Additive proxima-specific methods compose on the same builder:
///
/// ```ignore
/// use proxima_process::{Command, grounds::Deny};
///
/// let mut cmd = Command::new("/usr/bin/agent");
/// cmd.arg("--non-interactive")
///    .dispatch(Deny::new(libc::EACCES))
///    .libc_shim();
/// // Command impls Pipe; use it directly through proxima_primitives::pipe.
/// ```
#[derive(Builder)]
#[builder(builder_type = CommandBuilder, finish_fn = build)]
pub struct Command {
    /// `argv[0]`. Builder setter takes `impl Into<OsString>`.
    #[builder(into)]
    program: OsString,

    /// `argv[1..]`. Builder setter takes the whole `Vec`; for
    /// item-by-item append on the std-shape Command itself, use
    /// `.arg(...)`.
    #[builder(default)]
    args: Vec<OsString>,

    /// Working directory. Builder setter takes `impl Into<PathBuf>`.
    #[builder(into)]
    current_dir: Option<PathBuf>,
    // (bon auto-defaults Option<_> to None — no #[builder(default)] needed)
    /// Explicit env entries. Builder setter takes the whole
    /// `Vec`; for higher-level construction, build an [`Env`]
    /// and call `Command::env_from(env)` after `build()`.
    #[builder(default)]
    envs: Vec<(OsString, OsString)>,

    /// Whether to snapshot the parent's env into the child.
    /// Default `false` — explicit (matches the alloc-tier
    /// `CommandDescriptor` discipline).
    #[builder(default)]
    inherit_parent_env: bool,

    /// stdin wiring (kernel fd 0).
    #[builder(default)]
    stdin: Stdio,

    /// stdout wiring (kernel fd 1).
    #[builder(default)]
    stdout: Stdio,

    /// stderr wiring (kernel fd 2).
    #[builder(default)]
    stderr: Stdio,

    /// Additive sandbox dispatch chain.
    chain: Option<alloc_tier::PipeHandle<ChildRequest, ChildResponse>>,

    /// Additive opt-in for the libc-interpose shim.
    #[builder(default)]
    libc_shim: bool,

    /// Additive controlling-tty acquisition (PTY path).
    #[builder(default)]
    controlling_tty: bool,

    /// Additive umask override.
    umask: Option<u32>,
}

impl core::fmt::Debug for Command {
    /// Mirrors [`std::process::Command`]'s Debug output: prints
    /// the program + args in a way that's mostly cut-and-paste
    /// into a shell. The additive surface (chain, libc_shim,
    /// controlling_tty, umask) appears as a trailing tag.
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(formatter, "{:?}", self.program)?;
        for arg in &self.args {
            write!(formatter, " {:?}", arg)?;
        }
        if self.chain.is_some() || self.libc_shim || self.controlling_tty {
            formatter.write_str(" [+")?;
            if self.chain.is_some() {
                formatter.write_str("dispatch ")?;
            }
            if self.libc_shim {
                formatter.write_str("libc_shim ")?;
            }
            if self.controlling_tty {
                formatter.write_str("controlling_tty ")?;
            }
            formatter.write_str("]")?;
        }
        Ok(())
    }
}

impl Command {
    /// Mirrors [`std::process::Command::new`].
    #[must_use]
    pub fn new<S: AsRef<OsStr>>(program: S) -> Self {
        Self {
            program: program.as_ref().to_os_string(),
            args: Vec::new(),
            current_dir: None,
            envs: Vec::new(),
            inherit_parent_env: false,
            stdin: Stdio::Inherit,
            stdout: Stdio::Inherit,
            stderr: Stdio::Inherit,
            chain: None,
            libc_shim: false,
            controlling_tty: false,
            umask: None,
        }
    }

    /// Mirrors [`std::process::Command::arg`].
    pub fn arg<S: AsRef<OsStr>>(&mut self, arg: S) -> &mut Self {
        self.args.push(arg.as_ref().to_os_string());
        self
    }

    /// Mirrors [`std::process::Command::args`].
    pub fn args<I, S>(&mut self, args: I) -> &mut Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
    {
        for arg in args {
            self.arg(arg);
        }
        self
    }

    /// Mirrors [`std::process::Command::env`].
    pub fn env<K, V>(&mut self, key: K, value: V) -> &mut Self
    where
        K: AsRef<OsStr>,
        V: AsRef<OsStr>,
    {
        let key = key.as_ref().to_os_string();
        let value = value.as_ref().to_os_string();
        if let Some(slot) = self.envs.iter_mut().find(|(existing, _)| *existing == key) {
            slot.1 = value;
        } else {
            self.envs.push((key, value));
        }
        self
    }

    /// Mirrors [`std::process::Command::envs`].
    pub fn envs<I, K, V>(&mut self, vars: I) -> &mut Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: AsRef<OsStr>,
        V: AsRef<OsStr>,
    {
        for (key, value) in vars {
            self.env(key, value);
        }
        self
    }

    /// Mirrors [`std::process::Command::env_remove`].
    pub fn env_remove<K: AsRef<OsStr>>(&mut self, key: K) -> &mut Self {
        let key = key.as_ref();
        self.envs
            .retain(|(existing, _)| existing.as_os_str() != key);
        self
    }

    /// Mirrors [`std::process::Command::env_clear`]. Also clears
    /// the additive `inherit_parent_env` flag (matches std's
    /// expectation that `env_clear` followed by no further calls
    /// gives the child an empty env).
    pub fn env_clear(&mut self) -> &mut Self {
        self.envs.clear();
        self.inherit_parent_env = false;
        self
    }

    /// Mirrors [`std::process::Command::current_dir`].
    pub fn current_dir<P: AsRef<Path>>(&mut self, dir: P) -> &mut Self {
        self.current_dir = Some(dir.as_ref().to_path_buf());
        self
    }

    /// Mirrors [`std::process::Command::stdin`].
    pub fn stdin<T: Into<Stdio>>(&mut self, cfg: T) -> &mut Self {
        self.stdin = cfg.into();
        self
    }

    /// Mirrors [`std::process::Command::stdout`].
    pub fn stdout<T: Into<Stdio>>(&mut self, cfg: T) -> &mut Self {
        self.stdout = cfg.into();
        self
    }

    /// Mirrors [`std::process::Command::stderr`].
    pub fn stderr<T: Into<Stdio>>(&mut self, cfg: T) -> &mut Self {
        self.stderr = cfg.into();
        self
    }

    /// Mirrors [`std::process::Command::get_program`].
    #[must_use]
    pub fn get_program(&self) -> &OsStr {
        &self.program
    }

    /// Mirrors [`std::process::Command::get_current_dir`].
    #[must_use]
    pub fn get_current_dir(&self) -> Option<&Path> {
        self.current_dir.as_deref()
    }

    /// Mirrors [`std::process::Command::get_args`]. Borrows the
    /// arg list as a streaming iterator of `&OsStr`.
    pub fn get_args(&self) -> impl ExactSizeIterator<Item = &OsStr> {
        self.args.iter().map(OsString::as_os_str)
    }

    /// Mirrors [`std::process::Command::get_envs`]. Iterates
    /// `(&OsStr, Option<&OsStr>)` pairs — std uses `Option` to
    /// distinguish set-to-value vs unset. We never carry the
    /// "unset" form internally (`env_remove` drops the entry
    /// outright), so the second slot is always `Some` here.
    pub fn get_envs(&self) -> impl Iterator<Item = (&OsStr, Option<&OsStr>)> {
        self.envs
            .iter()
            .map(|(key, value)| (key.as_os_str(), Some(value.as_os_str())))
    }

    /// Introspect the stdio slots.
    #[must_use]
    pub fn get_stdin(&self) -> Stdio {
        self.stdin
    }
    #[must_use]
    pub fn get_stdout(&self) -> Stdio {
        self.stdout
    }
    #[must_use]
    pub fn get_stderr(&self) -> Stdio {
        self.stderr
    }

    /// Whether a dispatch chain is attached. The chain itself is
    /// type-erased (`Arc<dyn ...>`) so we don't return it; this
    /// is the introspection answer to "would `Pipe::call` route
    /// through the dispatch path?".
    #[must_use]
    pub fn has_dispatch_chain(&self) -> bool {
        self.chain.is_some()
    }

    /// Whether the libc-interpose shim opt-in is set.
    #[must_use]
    pub fn is_libc_shim(&self) -> bool {
        self.libc_shim
    }

    /// Whether the controlling-tty flag is set (PTY path).
    #[must_use]
    pub fn is_controlling_tty(&self) -> bool {
        self.controlling_tty
    }

    /// The umask override, if set.
    #[must_use]
    pub fn get_umask(&self) -> Option<u32> {
        self.umask
    }

    /// Whether the parent's env will be inherited at spawn time.
    #[must_use]
    pub fn inherits_parent_env(&self) -> bool {
        self.inherit_parent_env
    }

    /// Replace the entire env with the contents of an [`Env`]
    /// value. Existing entries are dropped. Use when you've
    /// composed an env elsewhere (e.g. loaded from
    /// [`CommandConfig`](super::command_config::CommandConfig))
    /// and want to apply it as a unit.
    pub fn env_from(&mut self, env: Env) -> &mut Self {
        self.envs.clear();
        for (key, value) in env.iter() {
            self.envs.push((OsString::from(key), OsString::from(value)));
        }
        self
    }

    /// Snapshot the current env as an [`Env`] value. Lossy for
    /// non-UTF-8 keys/values (they're skipped); for full fidelity
    /// iterate `get_envs()` directly.
    #[must_use]
    pub fn env_snapshot(&self) -> Env {
        self.envs
            .iter()
            .filter_map(|(key, value)| {
                let key = key.to_str()?;
                let value = value.to_str()?;
                Some((key.to_string(), value.to_string()))
            })
            .collect()
    }

    /// Additive: snapshot the parent's env into the descriptor.
    /// Std's `Command` inherits implicitly; ours is empty unless
    /// this is called (matches the explicit alloc-tier
    /// [`CommandDescriptor`] discipline + makes the descriptor
    /// serialisable without ambient state).
    pub fn inherit_current_env(&mut self) -> &mut Self {
        self.inherit_parent_env = true;
        self
    }

    /// Additive: attach a sandbox dispatch chain. Default is no
    /// chain — spawn behaves like std. Setting a chain switches
    /// the spawn path to [`spawn_and_dispatch_with_options`]
    /// which wires `extra_fd[7]` + `PROXIMA_DISPATCH_FD` on the
    /// child.
    pub fn dispatch<C>(&mut self, chain: C) -> &mut Self
    where
        C: SendPipe<In = ChildRequest, Out = ChildResponse, Err = ProximaError>
            + Send
            + Sync
            + 'static,
    {
        self.chain = Some(alloc_tier::into_handle(chain));
        self
    }

    /// Additive: opt the spawned child into the libc-interpose
    /// shim (`DYLD_INSERT_LIBRARIES` / `LD_PRELOAD`).
    pub fn libc_shim(&mut self) -> &mut Self {
        self.libc_shim = true;
        self
    }

    /// Additive: make the child the controlling-tty session
    /// leader. PTY wrapper code sets this; vanilla callers don't
    /// touch it.
    pub fn controlling_tty(&mut self, value: bool) -> &mut Self {
        self.controlling_tty = value;
        self
    }

    /// Additive: set the child's file-creation umask.
    pub fn umask(&mut self, mask: u32) -> &mut Self {
        self.umask = Some(mask);
        self
    }

    /// Lower to the alloc-tier [`CommandDescriptor`]. Validates
    /// NUL-freedom on every string; matches std's behaviour of
    /// deferring conversion errors to spawn time.
    pub fn to_descriptor(&self) -> Result<CommandDescriptor, ProximaError> {
        let program = osstr_to_cstring(&self.program, "program")?;
        let mut descriptor = CommandDescriptor::new(program);
        if self.inherit_parent_env {
            descriptor.inherit_current_env();
        }
        for arg in &self.args {
            descriptor.arg(osstr_to_cstring(arg, "arg")?);
        }
        if let Some(dir) = &self.current_dir {
            descriptor.current_dir(path_to_cstring(dir)?);
        }
        for (key, value) in &self.envs {
            descriptor.env(
                osstr_to_cstring(key, "env key")?,
                osstr_to_cstring(value, "env value")?,
            );
        }
        descriptor.stdin(self.stdin);
        descriptor.stdout(self.stdout);
        descriptor.stderr(self.stderr);
        if self.libc_shim {
            let key = CString::new(libc_shim::PRELOAD_ENV_VAR).map_err(|err| {
                ProximaError::Body(format!("libc-shim preload env var contains NUL: {err}"))
            })?;
            let value = CString::new(libc_shim::PATH)
                .map_err(|err| ProximaError::Body(format!("libc-shim path contains NUL: {err}")))?;
            descriptor.env(key, value);
        }
        Ok(descriptor)
    }

    /// Mirrors [`std::process::Command::spawn`] in shape. Lowers
    /// to the descriptor, applies [`SpawnOptions`], fork+execs.
    /// Returns a [`Child`] with the parent-side fds for any
    /// `Stdio::Piped` slots — no chain/dispatch wiring here
    /// (use [`Pipe::call`] for that — it handles the dispatch
    /// thread lifecycle alongside the byte shuttle).
    pub fn spawn(&mut self) -> Result<Child, ProximaError> {
        let descriptor = self.to_descriptor()?;
        spawn(&descriptor, self.spawn_options(None))
    }

    /// Mirrors [`std::process::Command::output`] in shape. Spawns
    /// the child with `stdout` + `stderr` forced to
    /// [`Stdio::Piped`], collects both, waits for exit, returns
    /// status + buffered output. `stdin` stays whatever the
    /// caller configured (default `Inherit`).
    ///
    /// Drop-in alignment with std uses `io::Result<Output>`; we
    /// return `Result<Output, ProximaError>` because the rest of
    /// the crate uses `ProximaError` — same shape, different
    /// error type.
    pub fn output(&mut self) -> Result<Output, ProximaError> {
        let original_stdout = self.stdout;
        let original_stderr = self.stderr;
        self.stdout = Stdio::Piped;
        self.stderr = Stdio::Piped;
        let result = self.spawn_and_collect();
        self.stdout = original_stdout;
        self.stderr = original_stderr;
        result
    }

    /// Mirrors [`std::process::Command::status`] in shape. Spawns
    /// the child with stdio as configured (no forced piping),
    /// waits for exit, returns the raw `wait(2)` status code.
    pub fn status(&mut self) -> Result<i32, ProximaError> {
        let mut child = self.spawn()?;
        wait_child(&mut child)
    }

    fn spawn_and_collect(&mut self) -> Result<Output, ProximaError> {
        let mut child = self.spawn()?;
        let stdout_fd = child
            .stdout
            .take()
            .ok_or_else(|| ProximaError::Body("output requires piped stdout".into()))?;
        let stderr_fd = child
            .stderr
            .take()
            .ok_or_else(|| ProximaError::Body("output requires piped stderr".into()))?;
        let stdout_bytes = drain_fd(stdout_fd);
        let stderr_bytes = drain_fd(stderr_fd);
        let status = wait_child(&mut child)?;
        Ok(Output {
            status,
            stdout: stdout_bytes,
            stderr: stderr_bytes,
        })
    }

    fn spawn_options(&self, dispatch_fd: Option<RawFd>) -> SpawnOptions {
        SpawnOptions {
            dispatch_fd,
            controlling_tty: self.controlling_tty,
            umask: self.umask,
        }
    }
}

/// Mirrors [`std::process::Output`] in shape — exit status +
/// buffered `stdout` / `stderr` bytes from a [`Command::output`]
/// call. The `status` field is the raw `wait(2)` int (std wraps
/// it in `ExitStatus`; we keep it raw for cross-tier
/// portability).
#[derive(Debug, Clone)]
pub struct Output {
    /// Raw `wait(2)` status code. Inspect via `libc::WIFEXITED`
    /// / `libc::WEXITSTATUS` etc.
    pub status: i32,
    /// Buffered child stdout.
    pub stdout: Vec<u8>,
    /// Buffered child stderr.
    pub stderr: Vec<u8>,
}

fn drain_fd(fd: std::os::fd::OwnedFd) -> Vec<u8> {
    use std::io::Read;
    use std::os::fd::{FromRawFd, IntoRawFd};
    let raw = fd.into_raw_fd();
    // SAFETY: we own raw; converting through File for ergonomic
    // Read; File's Drop closes raw exactly once.
    let mut file = unsafe { std::fs::File::from_raw_fd(raw) };
    let mut buffer = Vec::new();
    let _ = file.read_to_end(&mut buffer);
    buffer
}

fn wait_child(child: &mut Child) -> Result<i32, ProximaError> {
    let pid = child.pid;
    let mut status: libc::c_int = 0;
    // SAFETY: waitpid kernel call; no Rust invariants at risk.
    let waited = unsafe { libc::waitpid(pid, &mut status, 0) };
    if waited != pid {
        return Err(ProximaError::Body(format!(
            "waitpid returned {waited}, expected {pid}"
        )));
    }
    Ok(status)
}

// TryFrom — shell-parse a command line into program + args.
//
// `Command::new("/bin/ls")` matches std (program-only).
// `Command::try_from("git checkout main")` shell-parses (program
// + args from one string). Not in std; additive. Powered by the
// `shell-words` crate (POSIX shell quoting + escaping).

impl TryFrom<&str> for Command {
    type Error = super::descriptor::CommandParseError;
    fn try_from(input: &str) -> Result<Self, Self::Error> {
        parse_command_line(input)
    }
}

impl TryFrom<String> for Command {
    type Error = super::descriptor::CommandParseError;
    fn try_from(input: String) -> Result<Self, Self::Error> {
        parse_command_line(&input)
    }
}

impl TryFrom<&[u8]> for Command {
    type Error = super::descriptor::CommandParseError;
    fn try_from(bytes: &[u8]) -> Result<Self, Self::Error> {
        parse_command_line(std::str::from_utf8(bytes)?)
    }
}

impl<const N: usize> TryFrom<&[u8; N]> for Command {
    type Error = super::descriptor::CommandParseError;
    fn try_from(bytes: &[u8; N]) -> Result<Self, Self::Error> {
        Self::try_from(bytes.as_slice())
    }
}

impl TryFrom<Vec<u8>> for Command {
    type Error = super::descriptor::CommandParseError;
    fn try_from(bytes: Vec<u8>) -> Result<Self, Self::Error> {
        Self::try_from(bytes.as_slice())
    }
}

fn parse_command_line(input: &str) -> Result<Command, super::descriptor::CommandParseError> {
    use super::descriptor::CommandParseError;
    let mut tokens = shell_words::split(input)?.into_iter();
    let program = tokens.next().ok_or(CommandParseError::Empty)?;
    let mut command = Command::new(program);
    for token in tokens {
        command.arg(token);
    }
    Ok(command)
}

impl SendPipe for Command {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let descriptor_result = self.to_descriptor();
        let chain = self.chain.clone();
        let controlling_tty = self.controlling_tty;
        let umask = self.umask;
        async move {
            let mut descriptor = descriptor_result?;
            descriptor.stdin(Stdio::Piped).stdout(Stdio::Piped);
            run_pipe_call(descriptor, chain, controlling_tty, umask, request).await
        }
    }
}


async fn run_pipe_call(
    descriptor: CommandDescriptor,
    chain: Option<alloc_tier::PipeHandle<ChildRequest, ChildResponse>>,
    controlling_tty: bool,
    umask: Option<u32>,
    request: Request<Bytes>,
) -> Result<Response<Bytes>, ProximaError> {
    let (stdin_fd, stdout_fd, pid, after) = match chain {
        Some(chain) => {
            let mut dispatched = spawn_and_dispatch_with_options(
                &descriptor,
                chain,
                SpawnOptions {
                    dispatch_fd: None,
                    controlling_tty,
                    umask,
                },
            )?;
            let pid = dispatched.child.pid;
            let stdin = take_piped(&mut dispatched.child.stdin, "stdin")?;
            let stdout = take_piped(&mut dispatched.child.stdout, "stdout")?;
            (stdin, stdout, pid, AfterPipe::Dispatched(dispatched))
        }
        None => {
            let mut child = spawn(
                &descriptor,
                SpawnOptions {
                    dispatch_fd: None,
                    controlling_tty,
                    umask,
                },
            )?;
            let pid = child.pid;
            let stdin = take_piped(&mut child.stdin, "stdin")?;
            let stdout = take_piped(&mut child.stdout, "stdout")?;
            (stdin, stdout, pid, AfterPipe::Vanilla)
        }
    };

    // child_pid=0: skip FdPairPipe's auto-reap; AfterPipe drives
    // the single reap path (joins dispatch thread + waitpids).
    let inner_response = FdPairPipe::with_child(stdin_fd, stdout_fd, 0)
        .call(request)
        .await?;

    Ok(Response::streamed(wrap_body(inner_response, after, pid)))
}

fn take_piped(
    slot: &mut Option<std::os::fd::OwnedFd>,
    label: &'static str,
) -> Result<std::os::fd::OwnedFd, ProximaError> {
    slot.take()
        .ok_or_else(|| ProximaError::Body(format!("spawn returned no Piped {label} fd")))
}

enum AfterPipe {
    Vanilla,
    Dispatched(DispatchedChild),
}

fn wrap_body(response: Response<Bytes>, after: AfterPipe, pid: libc::pid_t) -> ResponseStream {
    let stream = response.into_chunk_stream();
    ResponseStream::new(stream::unfold(
        (stream, Some(after), pid),
        move |(mut stream, mut after, pid)| async move {
            match stream.next().await {
                Some(item) => Some((item, (stream, after, pid))),
                None => {
                    finish(after.take(), pid);
                    None
                }
            }
        },
    ))
}

fn finish(after: Option<AfterPipe>, pid: libc::pid_t) {
    match after {
        Some(AfterPipe::Vanilla) => {
            thread::spawn(move || {
                let mut status: libc::c_int = 0;
                unsafe { libc::waitpid(pid, &mut status, 0) };
            });
        }
        Some(AfterPipe::Dispatched(dispatched)) => {
            thread::spawn(move || {
                let _ = dispatched.wait();
            });
        }
        None => {}
    }
}

fn osstr_to_cstring(value: &OsStr, field: &'static str) -> Result<CString, ProximaError> {
    CString::new(value.as_bytes())
        .map_err(|err| ProximaError::Body(format!("{field} contains NUL: {err}")))
}

fn path_to_cstring(value: &Path) -> Result<CString, ProximaError> {
    CString::new(value.as_os_str().as_bytes())
        .map_err(|err| ProximaError::Body(format!("current_dir contains NUL: {err}")))
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::field_reassign_with_default,
        clippy::type_complexity,
        clippy::useless_vec,
        clippy::needless_range_loop,
        clippy::default_constructed_unit_structs
    )]

    use super::*;
    use bytes::Bytes;
    use futures::executor::block_on;

    #[test]
    fn drop_in_compat_compiles_against_std_shape_call_sites() {
        // Mirrors typical std::process::Command code paths. If any
        // signature drifts (e.g. arg loses AsRef<OsStr>), this
        // stops compiling.
        let mut cmd = Command::new("/bin/ls");
        cmd.arg("-la")
            .args(["/etc", "/tmp"])
            .env("LANG", "C")
            .envs([("A", "1"), ("B", "2")])
            .env_remove("LANG")
            .current_dir("/tmp")
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::inherit());
        assert_eq!(cmd.get_program(), OsStr::new("/bin/ls"));
        assert_eq!(cmd.get_current_dir(), Some(Path::new("/tmp")));
    }

    #[test]
    fn to_descriptor_lowers_to_cstring_form() {
        let mut cmd = Command::new("/bin/echo");
        cmd.arg("hi");
        let descriptor = cmd.to_descriptor().expect("lower");
        assert_eq!(descriptor.program.to_str().expect("utf8"), "/bin/echo");
        assert_eq!(descriptor.args.len(), 1);
    }

    #[test]
    fn nul_in_arg_rejected_at_lowering() {
        let mut cmd = Command::new("/bin/echo");
        cmd.arg("has\0nul");
        let err = cmd.to_descriptor().expect_err("nul rejected");
        assert!(format!("{err}").contains("NUL"));
    }

    #[test]
    fn libc_shim_flag_stamps_preload_env_at_lowering() {
        let mut cmd = Command::new("/bin/true");
        cmd.libc_shim();
        let descriptor = cmd.to_descriptor().expect("lower");
        let preload_key = std::ffi::CString::new(libc_shim::PRELOAD_ENV_VAR).expect("ascii");
        assert!(descriptor.env.iter().any(|entry| entry.key == preload_key));
    }

    #[test]
    fn bon_builder_constructs_immutable_command() {
        let cmd = Command::builder()
            .program("/bin/ls")
            .args(vec![OsString::from("-la"), OsString::from("/tmp")])
            .current_dir(std::path::PathBuf::from("/var"))
            .stdin(Stdio::piped())
            .libc_shim(true)
            .build();
        assert_eq!(cmd.get_program(), OsStr::new("/bin/ls"));
        assert_eq!(cmd.get_args().count(), 2);
        assert_eq!(cmd.get_current_dir(), Some(std::path::Path::new("/var")));
        assert_eq!(cmd.get_stdin(), Stdio::Piped);
        assert!(cmd.is_libc_shim());
    }

    #[test]
    fn introspection_getters_round_trip_what_setters_did() {
        let mut cmd = Command::new("/bin/echo");
        cmd.arg("hi")
            .env("LANG", "C")
            .env("PATH", "/usr/bin")
            .stdin(Stdio::piped())
            .controlling_tty(true)
            .umask(0o022);
        let args: Vec<&OsStr> = cmd.get_args().collect();
        assert_eq!(args, vec![OsStr::new("hi")]);
        let envs: Vec<(&OsStr, Option<&OsStr>)> = cmd.get_envs().collect();
        assert_eq!(envs.len(), 2);
        assert!(envs.iter().any(|(key, _)| *key == OsStr::new("LANG")));
        assert!(cmd.is_controlling_tty());
        assert_eq!(cmd.get_umask(), Some(0o022));
        assert!(!cmd.has_dispatch_chain());
    }

    #[test]
    fn env_from_replaces_existing_envs() {
        let mut cmd = Command::new("/bin/env");
        cmd.env("STALE", "value");
        let env: Env = [("LANG", "C"), ("PATH", "/usr/bin")].into_iter().collect();
        cmd.env_from(env);
        let envs: Vec<(&OsStr, Option<&OsStr>)> = cmd.get_envs().collect();
        assert_eq!(envs.len(), 2);
        assert!(!envs.iter().any(|(key, _)| *key == OsStr::new("STALE")));
    }

    #[test]
    fn env_snapshot_captures_current_state() {
        let mut cmd = Command::new("/bin/env");
        cmd.env("LANG", "C").env("PATH", "/usr/bin");
        let snapshot = cmd.env_snapshot();
        assert_eq!(snapshot.get("LANG"), Some("C"));
        assert_eq!(snapshot.get("PATH"), Some("/usr/bin"));
    }

    #[test]
    fn try_from_str_shell_parses_program_and_args() {
        let cmd: Command = "git checkout main".try_into().expect("shell parse");
        assert_eq!(cmd.get_program(), OsStr::new("git"));
        assert_eq!(cmd.args.len(), 2);
        assert_eq!(cmd.args[0], OsStr::new("checkout"));
        assert_eq!(cmd.args[1], OsStr::new("main"));
    }

    #[test]
    fn try_from_str_handles_quoted_args() {
        let cmd: Command = r#"git commit -m "the message""#.try_into().expect("parse");
        assert_eq!(cmd.get_program(), OsStr::new("git"));
        assert_eq!(cmd.args[2], OsStr::new("the message"));
    }

    #[test]
    fn try_from_byte_slice_decodes_then_parses() {
        let cmd: Command = b"/bin/ls -la".as_slice().try_into().expect("parse");
        assert_eq!(cmd.get_program(), OsStr::new("/bin/ls"));
        assert_eq!(cmd.args[0], OsStr::new("-la"));
    }

    #[test]
    fn try_from_byte_array_uses_array_overload() {
        let cmd: Command = b"/bin/echo hi".try_into().expect("parse");
        assert_eq!(cmd.get_program(), OsStr::new("/bin/echo"));
    }

    #[test]
    fn try_from_empty_string_returns_error() {
        let err = Command::try_from("").expect_err("empty must fail");
        assert!(matches!(
            err,
            super::super::descriptor::CommandParseError::Empty
        ));
    }

    #[test]
    fn output_collects_stdout_and_returns_exit_status() {
        let mut cmd = Command::new("/bin/echo");
        cmd.arg("hi from output");
        let out = cmd.output().expect("output");
        assert!(libc::WIFEXITED(out.status));
        assert_eq!(libc::WEXITSTATUS(out.status), 0);
        assert_eq!(out.stdout, b"hi from output\n");
        assert!(out.stderr.is_empty());
    }

    #[test]
    fn output_restores_caller_configured_stdio_after_collect() {
        let mut cmd = Command::new("/bin/echo");
        cmd.arg("hi").stdin(Stdio::null());
        let _ = cmd.output().expect("output");
        // Caller's stdin choice survives the output() call.
        assert_eq!(cmd.get_stdin(), Stdio::Null);
        // stdout/stderr restore to Inherit (the default the
        // caller hadn't touched).
        assert_eq!(cmd.get_stdout(), Stdio::Inherit);
        assert_eq!(cmd.get_stderr(), Stdio::Inherit);
    }

    #[test]
    fn status_waits_for_child_and_returns_exit_code() {
        let mut cmd = Command::new("/bin/sh");
        cmd.arg("-c").arg("exit 42");
        let status = cmd.status().expect("status");
        assert!(libc::WIFEXITED(status));
        assert_eq!(libc::WEXITSTATUS(status), 42);
    }

    #[test]
    fn spawn_pipe_round_trips_echo_with_no_chain() {
        let mut cmd = Command::new("/bin/echo");
        cmd.arg("hello via Command Pipe");
        let request = Request::builder()
            .method("POST")
            .path("/")
            .body(Bytes::new())
            .build()
            .expect("request builds");
        let response = block_on(<Command as SendPipe>::call(&cmd, request)).expect("call");
        let mut stream = response.into_chunk_stream();
        let mut output = Vec::new();
        block_on(async {
            while let Some(chunk) = stream.next().await {
                output.extend_from_slice(&chunk.expect("chunk"));
            }
        });
        assert_eq!(output, b"hello via Command Pipe\n");
    }
}
