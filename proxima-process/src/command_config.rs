//! `CommandConfig` + `Command::to_config` / `CommandConfig::into_command`
//! bridge, plus a layered builder for call-order precedence.
//!
//! Mirrors `proxima-telemetry`'s `TelemetryConfig` pattern: a single
//! data shape carries `bon::Builder`, `serde` (file I/O), and
//! `conflaguration::{Settings, Validate}` (env loading + runtime
//! invariants). `CommandConfig::layered()` exposes
//! `.from_path` / `.from_env` / `.with_*` ordering so operator
//! config and code-level overrides compose deterministically:
//!
//! - **Operator config wins**: put `.with_*` BEFORE `.from_path` /
//!   `.from_env`.
//! - **Code overrides win**: put `.with_*` AFTER `.from_path` /
//!   `.from_env`.
//!
//! # Lossy round-trip
//!
//! `CommandDescriptor::to_config` is best-effort: `Stdio::Fd(_)` slots downgrade
//! to `Stdio::Inherit` and `extra_fds` are dropped, since neither is
//! serialisable (raw fds are process-local handles, not data).
//! `CommandConfig::into_command` is the inverse and produces a
//! `CommandDescriptor` ready to hand to the typestate
//! [`CommandPipe`](super::command_pipe::CommandPipe) builder.

extern crate alloc;

use alloc::ffi::CString;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use std::collections::BTreeSet;

use bon::Builder;
use conflaguration::{Settings, Validate, ValidationMessage};
use serde::Deserialize;
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::{Map, Value};

use proxima_primitives::pipe::ProximaError;
use proxima_primitives::pipe::alloc_tier;

use super::command_pipe::{CommandPipe, CommandPipeBuilder, Set};
use super::descriptor::{CommandDescriptor, Stdio};
use super::env::Env;
use super::grounds::{Deny, Empty};
use super::protocol::{ChildRequest, ChildResponse};

/// Type-erased dispatch chain handle — the config-driven materialisation of
/// a [`DispatchChoice`] variant.
type DispatchChainHandle = alloc_tier::PipeHandle<ChildRequest, ChildResponse>;

/// Serialisable view of a [`CommandDescriptor`] suitable for env / TOML /
/// JSON loading. Mirrors `CommandDescriptor`'s shape using `String`s in
/// place of `CString`s; the bridge methods validate NUL-freedom
/// on conversion.
#[derive(Debug, Clone, Default, Builder, Deserialize, Serialize, Settings)]
#[settings(prefix = "PROXIMA_PROCESS")]
#[builder(derive(Clone, Debug))]
pub struct CommandConfig {
    /// `argv[0]` — program to exec. Bare names resolve via `PATH`
    /// when the underlying spawner does PATH lookup; the tier-1
    /// libc spawner requires an absolute / cwd-relative path.
    #[setting(default)]
    #[serde(default)]
    #[builder(default)]
    pub program: String,

    /// `argv[1..]` — additional arguments. Does NOT include
    /// `argv[0]`.
    #[setting(default)]
    #[serde(default)]
    #[builder(default)]
    pub args: Vec<String>,

    /// Working directory for the spawned process. `None` inherits
    /// the parent's cwd. Name matches `std::process::Command::current_dir`.
    #[setting(default)]
    #[serde(default)]
    pub current_dir: Option<String>,

    /// Explicit env entries as a first-class [`Env`] value. Build
    /// it elsewhere, hand it to the config, hand the config to a
    /// [`super::command::Command`] or
    /// [`CommandDescriptor`] — no incremental update API at this
    /// layer. `#[setting(skip)]` because nested env-of-envs maps
    /// don't flatten into env-var keys; load via TOML / JSON
    /// instead.
    #[setting(skip)]
    #[serde(default)]
    #[builder(default)]
    pub env: Env,

    /// If `true`, the parent's `std::env::vars_os()` snapshot is
    /// applied first, then [`Self::env`] entries override on top.
    /// If `false`, the child sees ONLY [`Self::env`] (empty by
    /// default).
    #[setting(default = true)]
    #[serde(default = "default_inherit_parent_env")]
    #[builder(default = true)]
    pub inherit_parent_env: bool,

    /// Wiring for the child's stdin (kernel fd 0). Name matches
    /// `std::process::Command::stdin`.
    /// `#[setting(skip)]` because enum variants don't map to a
    /// flat env-var value; set via TOML/JSON or the layered
    /// builder's `with_stdin`.
    #[setting(skip)]
    #[serde(default)]
    #[builder(default)]
    pub stdin: StdioChoice,

    /// Wiring for the child's stdout (kernel fd 1). Name matches
    /// `std::process::Command::stdout`.
    #[setting(skip)]
    #[serde(default)]
    #[builder(default)]
    pub stdout: StdioChoice,

    /// Wiring for the child's stderr (kernel fd 2). Name matches
    /// `std::process::Command::stderr`.
    #[setting(skip)]
    #[serde(default)]
    #[builder(default)]
    pub stderr: StdioChoice,

    /// File-creation umask applied in the child before exec.
    /// `None` inherits. Maximum effective value `0o777`.
    #[setting(default)]
    #[serde(default)]
    pub umask: Option<u32>,

    /// If `true`, child calls `setsid()` + `ioctl(TIOCSCTTY)` to
    /// become its session leader. Required for PTY-attached
    /// children.
    #[setting(default)]
    #[serde(default)]
    #[builder(default)]
    pub controlling_tty: bool,

    /// Dispatch chain the spawned child's `extra_fd[7]` traffic
    /// (and any libc-interpose hits) is routed through. Default
    /// is [`DispatchChoice::Empty`] — synthesises empty responses
    /// for any request, harmless when no shim is loaded.
    /// `#[setting(skip)]` because tagged-union variants don't
    /// flatten into env-var keys; set via TOML / JSON or the
    /// layered builder.
    #[setting(skip)]
    #[serde(default)]
    #[builder(default)]
    pub dispatch: DispatchChoice,

    /// If `true`, [`Self::into_pipe_builder`] calls
    /// `.libc_shim()` on the underlying builder — adds the
    /// platform-correct preload env var pointing at
    /// [`super::libc_shim::PATH`] to the child's env. Default off.
    #[setting(default)]
    #[serde(default)]
    #[builder(default)]
    pub libc_shim: bool,
}

/// Serialisable view of the dispatch chain attached to the
/// spawned child. Mirrors the `ExporterChoice` pattern from
/// `proxima-telemetry`: each variant maps to one concrete chain
/// type at materialisation time, and the chain is wrapped in a
/// [`alloc_tier::PipeHandle`] so it slots into the typestate
/// [`CommandPipe`] builder.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DispatchChoice {
    /// Synthesise empty responses for every request — read returns
    /// EOF, write reports success-with-zero, stat returns zeros.
    #[default]
    Empty,
    /// Reject every request with the given errno.
    Deny { errno: i32 },
}

impl DispatchChoice {
    /// Materialise the choice into the type-erased
    /// [`alloc_tier::PipeHandle`] that
    /// [`CommandPipe::from_config_dispatch`] feeds to the
    /// typestate builder.
    #[must_use]
    pub fn into_dyn_chain(self) -> DispatchChainHandle {
        match self {
            DispatchChoice::Empty => alloc_tier::into_handle(Empty),
            DispatchChoice::Deny { errno } => alloc_tier::into_handle(Deny::new(errno)),
        }
    }
}

/// Serialisable subset of [`Stdio`]. Drops the `Stdio::Fd(_)` variant
/// (raw fds aren't data) — `CommandDescriptor::to_config` downgrades `Fd`
/// to [`StdioChoice::Inherit`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StdioChoice {
    #[default]
    Inherit,
    Null,
    Piped,
}

impl From<StdioChoice> for Stdio {
    fn from(choice: StdioChoice) -> Self {
        match choice {
            StdioChoice::Inherit => Stdio::Inherit,
            StdioChoice::Null => Stdio::Null,
            StdioChoice::Piped => Stdio::Piped,
        }
    }
}

impl From<Stdio> for StdioChoice {
    fn from(io: Stdio) -> Self {
        match io {
            Stdio::Inherit | Stdio::Fd(_) => StdioChoice::Inherit,
            Stdio::Null => StdioChoice::Null,
            Stdio::Piped => StdioChoice::Piped,
        }
    }
}

fn default_inherit_parent_env() -> bool {
    true
}

impl Validate for CommandConfig {
    fn validate(&self) -> conflaguration::Result<()> {
        let mut errors = Vec::new();
        if self.program.is_empty() {
            errors.push(ValidationMessage::new("program", "must be non-empty"));
        }
        if self.program.contains('\0') {
            errors.push(ValidationMessage::new("program", "must not contain NUL"));
        }
        if self.args.iter().any(|arg| arg.contains('\0')) {
            errors.push(ValidationMessage::new(
                "args",
                "entries must not contain NUL",
            ));
        }
        if let Some(cwd) = &self.current_dir
            && cwd.contains('\0')
        {
            errors.push(ValidationMessage::new("cwd", "must not contain NUL"));
        }
        if self
            .env
            .iter()
            .any(|(key, value)| key.contains('\0') || value.contains('\0'))
        {
            errors.push(ValidationMessage::new(
                "env",
                "keys and values must not contain NUL",
            ));
        }
        if let Some(umask) = self.umask
            && umask > 0o777
        {
            errors.push(ValidationMessage::new("umask", "must be <= 0o777"));
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(conflaguration::Error::Validation { errors })
        }
    }
}

impl CommandConfig {
    /// Materialise a runtime [`CommandDescriptor`] from this config.
    /// Validates first; returns `Err` on any invariant break.
    ///
    /// `umask` and `controlling_tty` ride alongside in
    /// [`Self::spawn_options`] — they're not on the descriptor
    /// (they're runtime-spawn knobs, not data) per the post-RISC
    /// split.
    pub fn into_command(self) -> Result<CommandDescriptor, ProximaError> {
        self.validate()
            .map_err(|err| ProximaError::Body(format!("{err}")))?;
        let program = cstring(self.program, "program")?;
        let mut command = CommandDescriptor::new(program);
        if self.inherit_parent_env {
            command.inherit_current_env();
        }
        for arg in self.args {
            command.arg(cstring(arg, "arg")?);
        }
        if let Some(cwd) = self.current_dir {
            command.current_dir(cstring(cwd, "current_dir")?);
        }
        for (key, value) in self.env.iter() {
            let key = cstring(key.clone(), "env key")?;
            let value = cstring(value.clone(), "env value")?;
            command.env(key, value);
        }
        command
            .stdin(self.stdin.into())
            .stdout(self.stdout.into())
            .stderr(self.stderr.into());
        Ok(command)
    }

    /// `umask` + `controlling_tty` materialised into the runtime
    /// [`SpawnOptions`] shape. `dispatch_fd` stays `None` here —
    /// the dispatch socket is allocated by `spawn_and_dispatch`,
    /// not by config.
    #[must_use]
    pub fn spawn_options(&self) -> super::spawn::SpawnOptions {
        super::spawn::SpawnOptions {
            dispatch_fd: None,
            controlling_tty: self.controlling_tty,
            umask: self.umask,
        }
    }

    /// Materialise the entire config — program, args, env,
    /// stdio slots, current_dir, dispatch chain, libc_shim flag,
    /// controlling_tty, umask — into a fully-configured
    /// std-tier [`super::command::Command`]. `into_command()` is
    /// the alloc-tier bridge to [`CommandDescriptor`]; this is
    /// the drop-in-friendly bridge for code that wants
    /// `Command::spawn` / `Command::output` / `Command as Pipe`
    /// straight out of config.
    pub fn into_std_command(self) -> Result<super::command::Command, ProximaError> {
        self.validate()
            .map_err(|err| ProximaError::Body(format!("{err}")))?;
        let mut command = super::command::Command::new(self.program);
        if self.inherit_parent_env {
            command.inherit_current_env();
        }
        command.args(self.args);
        if let Some(dir) = self.current_dir {
            command.current_dir(dir);
        }
        command.env_from(self.env);
        command
            .stdin(Stdio::from(self.stdin))
            .stdout(Stdio::from(self.stdout))
            .stderr(Stdio::from(self.stderr));
        if self.controlling_tty {
            command.controlling_tty(true);
        }
        if let Some(mask) = self.umask {
            command.umask(mask);
        }
        if self.libc_shim {
            command.libc_shim();
        }
        // Materialise the chain only if the variant is non-trivial;
        // default `DispatchChoice::Empty` matches the Command
        // default (no chain attached, pure std spawn path).
        if !matches!(self.dispatch, DispatchChoice::Empty) {
            command.dispatch(self.dispatch.into_dyn_chain());
        }
        Ok(command)
    }

    /// Entry to the call-order-precedence layered builder.
    #[must_use]
    pub fn layered() -> CommandConfigLayered {
        CommandConfigLayered {
            inner: CommandConfig::default(),
            touched: BTreeSet::new(),
        }
    }
}

impl CommandDescriptor {
    /// Best-effort serialisable view. `Stdio::Fd(_)` slots downgrade
    /// to `StdioChoice::Inherit`; `extra_fds` are dropped; env entries
    /// with non-UTF-8 keys or values are skipped. Sets
    /// `inherit_parent_env = false` since the carried env is the
    /// materialised set (no further inheritance to apply).
    #[must_use]
    pub fn to_config(&self) -> CommandConfig {
        let program = self.program.to_str().unwrap_or_default().to_string();
        let args = self
            .args
            .iter()
            .filter_map(|arg| arg.to_str().ok().map(ToString::to_string))
            .collect();
        let cwd = self
            .current_dir
            .as_ref()
            .and_then(|cwd| cwd.to_str().ok())
            .map(ToString::to_string);
        let env: Env = self
            .env
            .iter()
            .filter_map(|entry| {
                let key = entry.key.to_str().ok()?;
                let value = entry.value.to_str().ok()?;
                Some((key.to_string(), value.to_string()))
            })
            .collect();
        CommandConfig {
            program,
            args,
            current_dir: cwd,
            env,
            inherit_parent_env: false,
            stdin: self.stdin.into(),
            stdout: self.stdout.into(),
            stderr: self.stderr.into(),
            // umask + controlling_tty live on SpawnOptions now,
            // not on CommandDescriptor — round-trip starts at
            // defaults; the caller carries SpawnOptions separately
            // if they care.
            umask: None,
            controlling_tty: false,
            dispatch: DispatchChoice::default(),
            libc_shim: false,
        }
    }
}

impl super::command::Command {
    /// Snapshot the std-tier [`Command`](super::command::Command)
    /// into a serialisable [`CommandConfig`] — the **Built →
    /// Config** half of the principle-4 interop contract. The
    /// snapshot lowers via [`Command::to_descriptor`], then maps
    /// the descriptor to a `CommandConfig`, finally lifts the
    /// additive flags (`controlling_tty`, `umask`, `libc_shim`)
    /// onto the result.
    ///
    /// # Lossy
    /// - Non-UTF-8 env keys / values are skipped (the config
    ///   surface is `String`-based for serde friendliness).
    /// - The dispatch chain (type-erased `Arc<dyn ...>`)
    ///   collapses to [`DispatchChoice::Empty`] — the variant
    ///   tag isn't recoverable from the runtime form.
    /// - NUL bytes in any field surface as `ProximaError`
    ///   (matches std's deferred-error semantics).
    pub fn to_config(&self) -> Result<CommandConfig, ProximaError> {
        let descriptor = self.to_descriptor()?;
        let mut config = descriptor.to_config();
        config.controlling_tty = self.is_controlling_tty();
        config.umask = self.get_umask();
        config.libc_shim = self.is_libc_shim();
        config.inherit_parent_env = self.inherits_parent_env();
        Ok(config)
    }

    /// Construct a [`Command`](super::command::Command) from a
    /// [`CommandConfig`] — the **Config → Built** half of the
    /// principle-4 interop contract. Symmetric alias for
    /// [`CommandConfig::into_std_command`].
    pub fn from_config(config: CommandConfig) -> Result<Self, ProximaError> {
        config.into_std_command()
    }
}

impl CommandConfig {
    /// Bridge the entire config — command, dispatch chain, and
    /// `libc_shim` flag — into a ready-to-build typestate
    /// builder. The chain is type-erased through a
    /// [`alloc_tier::PipeHandle`]; the caller's only remaining step is
    /// `.build()`.
    pub fn into_pipe_builder(
        self,
    ) -> Result<CommandPipeBuilder<Set<CommandDescriptor>, Set<DispatchChainHandle>>, ProximaError>
    {
        let libc_shim = self.libc_shim;
        let dispatch = self.dispatch.clone().into_dyn_chain();
        let command = self.into_command()?;
        let mut builder = CommandPipe::builder().command(command).dispatch(dispatch);
        if libc_shim {
            builder = builder.libc_shim();
        }
        Ok(builder)
    }
}

/// Layered builder for [`CommandConfig`] — matches the shape of
/// `proxima-telemetry::TelemetryLayerBuilder` so operators learn one pattern
/// and apply it everywhere. Every source (`.from_path`, `.from_env`,
/// `.underlay_path`, `.underlay_env`, `.with_*`) contributes only the fields
/// it actually specifies, merged onto the accumulated config — a field a
/// source doesn't touch falls through to whatever prior layers set.
/// `.from_path`/`.from_env` override (last writer wins per field);
/// `.underlay_path`/`.underlay_env` fill only fields still unset; `.with_*`
/// always acts as an override at its call position.
pub struct CommandConfigLayered {
    inner: CommandConfig,
    touched: BTreeSet<String>,
}

impl CommandConfigLayered {
    /// Merge a config file's fields onto the accumulated config (TOML,
    /// JSON, etc. per `conflaguration`'s format registry); the file wins
    /// for every field it specifies.
    ///
    /// # Errors
    /// Returns `conflaguration::Error` on parse or schema failure.
    pub fn from_path<P: AsRef<std::path::Path>>(
        mut self,
        path: P,
    ) -> Result<Self, conflaguration::Error> {
        let incoming: Value = conflaguration::from_file(path.as_ref())?;
        apply_layer(
            &mut self.inner,
            &mut self.touched,
            incoming,
            MergeMode::Override,
        )?;
        Ok(self)
    }

    /// Fill any still-unset fields from a config file; already-set fields
    /// are left untouched.
    ///
    /// # Errors
    /// Returns `conflaguration::Error` on parse or schema failure.
    pub fn underlay_path<P: AsRef<std::path::Path>>(
        mut self,
        path: P,
    ) -> Result<Self, conflaguration::Error> {
        let incoming: Value = conflaguration::from_file(path.as_ref())?;
        apply_layer(
            &mut self.inner,
            &mut self.touched,
            incoming,
            MergeMode::Underlay,
        )?;
        Ok(self)
    }

    /// Merge env-set fields (the `PROXIMA_PROCESS_` prefix) onto the
    /// accumulated config; env wins for every field it sets.
    ///
    /// # Errors
    /// Returns `conflaguration::Error` on parse failure.
    pub fn from_env(mut self) -> Result<Self, conflaguration::Error> {
        let incoming = command_env_partial()?;
        apply_layer(
            &mut self.inner,
            &mut self.touched,
            incoming,
            MergeMode::Override,
        )?;
        Ok(self)
    }

    /// Fill any still-unset fields from env vars; already-set fields are
    /// left untouched even if the matching env var is set.
    ///
    /// # Errors
    /// Returns `conflaguration::Error` on parse failure.
    pub fn underlay_env(mut self) -> Result<Self, conflaguration::Error> {
        let incoming = command_env_partial()?;
        apply_layer(
            &mut self.inner,
            &mut self.touched,
            incoming,
            MergeMode::Underlay,
        )?;
        Ok(self)
    }

    #[must_use]
    pub fn with_program(mut self, program: impl Into<String>) -> Self {
        self.inner.program = program.into();
        self.touched.insert("program".to_string());
        self
    }

    #[must_use]
    pub fn with_args<I, S>(mut self, args: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.inner.args = args.into_iter().map(Into::into).collect();
        self.touched.insert("args".to_string());
        self
    }

    #[must_use]
    pub fn with_current_dir(mut self, current_dir: impl Into<String>) -> Self {
        self.inner.current_dir = Some(current_dir.into());
        self.touched.insert("current_dir".to_string());
        self
    }

    #[must_use]
    pub fn with_env_entry(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.inner.env.insert(key, value);
        self.touched.insert("env".to_string());
        self
    }

    #[must_use]
    pub fn with_inherit_parent_env(mut self, value: bool) -> Self {
        self.inner.inherit_parent_env = value;
        self.touched.insert("inherit_parent_env".to_string());
        self
    }

    #[must_use]
    pub fn with_stdin(mut self, choice: StdioChoice) -> Self {
        self.inner.stdin = choice;
        self.touched.insert("stdin".to_string());
        self
    }

    #[must_use]
    pub fn with_stdout(mut self, choice: StdioChoice) -> Self {
        self.inner.stdout = choice;
        self.touched.insert("stdout".to_string());
        self
    }

    #[must_use]
    pub fn with_stderr(mut self, choice: StdioChoice) -> Self {
        self.inner.stderr = choice;
        self.touched.insert("stderr".to_string());
        self
    }

    #[must_use]
    pub fn with_umask(mut self, umask: u32) -> Self {
        self.inner.umask = Some(umask);
        self.touched.insert("umask".to_string());
        self
    }

    #[must_use]
    pub fn with_controlling_tty(mut self, value: bool) -> Self {
        self.inner.controlling_tty = value;
        self.touched.insert("controlling_tty".to_string());
        self
    }

    #[must_use]
    pub fn build(self) -> CommandConfig {
        self.inner
    }
}

/// Whether an incoming layer's fields win over an already-touched field
/// (`Override`) or only fill a field nothing has set yet (`Underlay`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MergeMode {
    Override,
    Underlay,
}

/// Merge `incoming`'s present fields onto `inner`, tracking which top-level
/// fields have been touched so `Underlay` layers never clobber an
/// already-set value. Every field on [`CommandConfig`] is a scalar or a
/// Vec/Map data collection (never a `#[setting(nested)]` sub-config), so a
/// flat, one-level merge is correct: a collection (`args`, `env`) is
/// replaced wholesale when a source provides it, never element-merged.
fn apply_layer<T>(
    inner: &mut T,
    touched: &mut BTreeSet<String>,
    incoming: Value,
    mode: MergeMode,
) -> Result<(), conflaguration::Error>
where
    T: Serialize + DeserializeOwned,
{
    let Value::Object(incoming_map) = incoming else {
        return Ok(());
    };
    let mut base = to_value(inner)?;
    let Value::Object(base_map) = &mut base else {
        return Ok(());
    };
    for (key, value) in incoming_map {
        apply_leaf(base_map, &key, value, mode, &key, touched);
    }
    *inner = from_value(base)?;
    Ok(())
}

fn apply_leaf(
    map: &mut Map<String, Value>,
    key: &str,
    value: Value,
    mode: MergeMode,
    touched_path: &str,
    touched: &mut BTreeSet<String>,
) {
    let should_apply = match mode {
        MergeMode::Override => true,
        MergeMode::Underlay => !touched.contains(touched_path),
    };
    if should_apply {
        map.insert(key.to_string(), value);
        touched.insert(touched_path.to_string());
    }
}

/// The env-set subset of [`CommandConfig`]'s fields, as a partial JSON
/// object containing only the fields whose env var is actually present —
/// never the ones `Settings::from_env` filled with a default.
fn command_env_partial() -> Result<Value, conflaguration::Error> {
    let resolved = CommandConfig::from_env()?;
    let mut partial = Map::new();
    insert_if_env_set(
        &mut partial,
        "program",
        &["PROXIMA_PROCESS_PROGRAM"],
        &resolved.program,
    )?;
    insert_if_env_set(
        &mut partial,
        "args",
        &["PROXIMA_PROCESS_ARGS"],
        &resolved.args,
    )?;
    insert_if_env_set(
        &mut partial,
        "current_dir",
        &["PROXIMA_PROCESS_CURRENT_DIR"],
        &resolved.current_dir,
    )?;
    insert_if_env_set(
        &mut partial,
        "inherit_parent_env",
        &["PROXIMA_PROCESS_INHERIT_PARENT_ENV"],
        &resolved.inherit_parent_env,
    )?;
    insert_if_env_set(
        &mut partial,
        "umask",
        &["PROXIMA_PROCESS_UMASK"],
        &resolved.umask,
    )?;
    insert_if_env_set(
        &mut partial,
        "controlling_tty",
        &["PROXIMA_PROCESS_CONTROLLING_TTY"],
        &resolved.controlling_tty,
    )?;
    insert_if_env_set(
        &mut partial,
        "libc_shim",
        &["PROXIMA_PROCESS_LIBC_SHIM"],
        &resolved.libc_shim,
    )?;
    Ok(Value::Object(partial))
}

fn insert_if_env_set<T: Serialize>(
    partial: &mut Map<String, Value>,
    field: &str,
    env_names: &[&str],
    value: &T,
) -> Result<(), conflaguration::Error> {
    if env_names.iter().any(|name| std::env::var(name).is_ok()) {
        partial.insert(field.to_string(), to_value(value)?);
    }
    Ok(())
}

fn to_value<T: Serialize>(value: &T) -> Result<Value, conflaguration::Error> {
    serde_json::to_value(value).map_err(|error| conflaguration::Error::Validation {
        errors: alloc::vec![ValidationMessage::new(
            "layered",
            format!("serialize failed: {error}"),
        )],
    })
}

fn from_value<T: DeserializeOwned>(value: Value) -> Result<T, conflaguration::Error> {
    serde_json::from_value(value).map_err(|error| conflaguration::Error::Validation {
        errors: alloc::vec![ValidationMessage::new(
            "layered",
            format!("deserialize failed: {error}"),
        )],
    })
}

fn cstring(value: String, field: &'static str) -> Result<CString, ProximaError> {
    CString::new(value).map_err(|err| ProximaError::Body(format!("{field} contains NUL: {err}")))
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

    #[test]
    fn default_config_fails_validation_without_program() {
        let config = CommandConfig::default();
        assert!(config.validate().is_err());
    }

    #[test]
    fn nonempty_program_validates() {
        let config = CommandConfig::builder()
            .program("/bin/true".to_string())
            .build();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn nul_in_program_rejected() {
        let config = CommandConfig::builder()
            .program("/bin/with\0nul".to_string())
            .build();
        let err = config.validate().expect_err("nul must be rejected");
        let message = format!("{err:?}");
        assert!(message.contains("program"), "got: {message}");
    }

    #[test]
    fn umask_above_octal_777_rejected() {
        let mut config = CommandConfig::builder()
            .program("/bin/true".to_string())
            .build();
        config.umask = Some(0o1000);
        assert!(config.validate().is_err());
    }

    #[test]
    fn into_command_materialises_args_and_env() {
        let config = CommandConfig::builder()
            .program("/bin/echo".to_string())
            .args(alloc::vec!["hi".to_string(), "there".to_string()])
            .env(Env::from_iter([("FOO", "bar")]))
            .inherit_parent_env(false)
            .build();
        let command = config.into_command().expect("into_command");
        assert_eq!(command.program.to_str().expect("utf8"), "/bin/echo");
        assert_eq!(command.args.len(), 2);
        assert_eq!(command.env.len(), 1);
        assert_eq!(command.env[0].key.to_str().expect("utf8"), "FOO");
    }

    #[test]
    fn to_config_round_trip_preserves_visible_fields() {
        let original = CommandConfig::builder()
            .program("/bin/echo".to_string())
            .args(alloc::vec!["a".to_string()])
            .env(Env::from_iter([("K", "V")]))
            .inherit_parent_env(false)
            .stdin(StdioChoice::Piped)
            .stdout(StdioChoice::Piped)
            .build();
        let command = original.clone().into_command().expect("into_command");
        let restored = command.to_config();
        // Round-trip carries everything that lives on the
        // descriptor. controlling_tty + umask live on SpawnOptions
        // post-RISC split, so they DON'T round-trip through
        // CommandDescriptor — those fields drop to defaults.
        assert_eq!(restored.program, original.program);
        assert_eq!(restored.args, original.args);
        assert_eq!(restored.env, original.env);
        assert_eq!(restored.stdin, original.stdin);
        assert_eq!(restored.stdout, original.stdout);
    }

    #[test]
    fn layered_with_then_path_lets_path_win() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let path = dir.path().join("command.toml");
        std::fs::write(
            &path,
            r#"program = "/bin/from-file"
"#,
        )
        .expect("write toml");
        let config = CommandConfig::layered()
            .with_program("/bin/from-code")
            .from_path(&path)
            .expect("from_path")
            .build();
        assert_eq!(config.program, "/bin/from-file");
    }

    #[test]
    fn layered_path_then_with_lets_code_win() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let path = dir.path().join("command.toml");
        std::fs::write(
            &path,
            r#"program = "/bin/from-file"
"#,
        )
        .expect("write toml");
        let config = CommandConfig::layered()
            .from_path(&path)
            .expect("from_path")
            .with_program("/bin/from-code")
            .build();
        assert_eq!(config.program, "/bin/from-code");
    }

    #[test]
    fn layered_from_env_reads_prefixed_vars() {
        temp_env::with_vars([("PROXIMA_PROCESS_PROGRAM", Some("/bin/from-env"))], || {
            let config = CommandConfig::layered()
                .from_env()
                .expect("from_env")
                .build();
            assert_eq!(config.program, "/bin/from-env");
        });
    }

    // the exact seam-#3 case: a file sets TWO fields, env sets only ONE —
    // the file's other field must survive `.from_path().from_env()`.
    #[test]
    fn seam_3_from_path_then_from_env_preserves_files_untouched_field() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let path = dir.path().join("command.toml");
        std::fs::write(
            &path,
            "program = \"/bin/from-file\"\ncurrent_dir = \"/var/tmp\"\n",
        )
        .expect("write toml");
        temp_env::with_vars([("PROXIMA_PROCESS_PROGRAM", Some("/bin/from-env"))], || {
            let config = CommandConfig::layered()
                .from_path(&path)
                .expect("from_path")
                .from_env()
                .expect("from_env")
                .build();
            assert_eq!(
                config.program, "/bin/from-env",
                "env wins the field it sets"
            );
            assert_eq!(
                config.current_dir.as_deref(),
                Some("/var/tmp"),
                "the file's field must survive"
            );
        });
    }

    // order-independence: the same two sources, built both orders.
    #[test]
    fn order_independence_file_then_env_vs_env_then_file() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let path = dir.path().join("command.toml");
        std::fs::write(&path, "current_dir = \"/var/tmp\"\n").expect("write toml");
        temp_env::with_vars([("PROXIMA_PROCESS_PROGRAM", Some("/bin/from-env"))], || {
            let file_then_env = CommandConfig::layered()
                .from_path(&path)
                .expect("from_path")
                .from_env()
                .expect("from_env")
                .build();
            assert_eq!(
                file_then_env.current_dir.as_deref(),
                Some("/var/tmp"),
                "file survives"
            );
            assert_eq!(file_then_env.program, "/bin/from-env", "env applies");

            let env_then_file = CommandConfig::layered()
                .from_env()
                .expect("from_env")
                .from_path(&path)
                .expect("from_path")
                .build();
            assert_eq!(env_then_file.program, "/bin/from-env", "env survives");
            assert_eq!(
                env_then_file.current_dir.as_deref(),
                Some("/var/tmp"),
                "file applies"
            );
        });
    }

    // full stack: defaults < file < env < with_*.
    #[test]
    fn full_stack_defaults_file_env_with_override_each_field() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let path = dir.path().join("command.toml");
        std::fs::write(
            &path,
            "current_dir = \"/var/tmp\"\ninherit_parent_env = false\n",
        )
        .expect("write toml");
        temp_env::with_vars([("PROXIMA_PROCESS_PROGRAM", Some("/bin/from-env"))], || {
            let config = CommandConfig::layered()
                .from_path(&path)
                .expect("from_path")
                .from_env()
                .expect("from_env")
                .with_controlling_tty(true)
                .build();
            assert_eq!(config.program, "/bin/from-env", "env layer");
            assert_eq!(
                config.current_dir.as_deref(),
                Some("/var/tmp"),
                "file layer"
            );
            assert!(!config.inherit_parent_env, "file layer");
            assert!(config.controlling_tty, "with_* layer");
            assert!(
                config.umask.is_none(),
                "untouched — falls through to the default"
            );
        });
    }

    // underlay never clobbers an already-set field; it DOES fill an unset one.
    #[test]
    fn underlay_path_fills_only_unset_fields() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let path = dir.path().join("command.toml");
        std::fs::write(
            &path,
            "program = \"/bin/from-file\"\ncurrent_dir = \"/var/tmp\"\n",
        )
        .expect("write toml");
        let config = CommandConfig::layered()
            .with_program("/bin/from-code")
            .underlay_path(&path)
            .expect("underlay_path")
            .build();
        assert_eq!(
            config.program, "/bin/from-code",
            "with_* already set it; file is dropped"
        );
        assert_eq!(
            config.current_dir.as_deref(),
            Some("/var/tmp"),
            "unset; underlay fills it"
        );
    }

    #[test]
    fn underlay_env_fills_only_unset_fields() {
        temp_env::with_vars(
            [
                ("PROXIMA_PROCESS_PROGRAM", Some("/bin/from-env")),
                ("PROXIMA_PROCESS_CURRENT_DIR", Some("/var/tmp")),
            ],
            || {
                let config = CommandConfig::layered()
                    .with_program("/bin/from-code")
                    .underlay_env()
                    .expect("underlay_env")
                    .build();
                assert_eq!(
                    config.program, "/bin/from-code",
                    "already set; env's value dropped"
                );
                assert_eq!(
                    config.current_dir.as_deref(),
                    Some("/var/tmp"),
                    "unset; env fills it"
                );
            },
        );
    }

    // order-independence for underlay: the first-applied source wins for a
    // field both specify.
    #[test]
    fn order_independence_underlay_flavor_first_setter_wins_either_direction() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let path = dir.path().join("command.toml");
        std::fs::write(&path, "program = \"/bin/from-file\"\n").expect("write toml");
        temp_env::with_vars([("PROXIMA_PROCESS_PROGRAM", Some("/bin/from-env"))], || {
            let file_first = CommandConfig::layered()
                .underlay_path(&path)
                .expect("underlay_path")
                .underlay_env()
                .expect("underlay_env")
                .build();
            assert_eq!(
                file_first.program, "/bin/from-file",
                "file applied first, wins"
            );

            let env_first = CommandConfig::layered()
                .underlay_env()
                .expect("underlay_env")
                .underlay_path(&path)
                .expect("underlay_path")
                .build();
            assert_eq!(
                env_first.program, "/bin/from-env",
                "env applied first, wins"
            );
        });
    }

    // combined: defaults -> underlay(file) -> override(env) -> override(with_*).
    #[test]
    fn combined_underlay_file_then_override_env_then_with() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let path = dir.path().join("command.toml");
        std::fs::write(
            &path,
            "program = \"/bin/from-file\"\ncurrent_dir = \"/var/tmp\"\n",
        )
        .expect("write toml");
        temp_env::with_vars([("PROXIMA_PROCESS_PROGRAM", Some("/bin/from-env"))], || {
            let config = CommandConfig::layered()
                .underlay_path(&path)
                .expect("underlay_path")
                .from_env()
                .expect("from_env")
                .with_current_dir("/opt/override")
                .build();
            assert_eq!(
                config.program, "/bin/from-env",
                "override(env) wins over underlay(file)"
            );
            assert_eq!(
                config.current_dir.as_deref(),
                Some("/opt/override"),
                "the later with_* overrides underlay(file)"
            );
        });
    }

    // collection (Vec) replace-if-present: a second layer providing `args`
    // replaces it wholesale, never appends/unions.
    #[test]
    fn args_collection_replaces_wholesale_not_union() {
        let config = CommandConfig::layered()
            .with_args(["a", "b"])
            .with_args(["c"])
            .build();
        assert_eq!(
            config.args,
            vec!["c".to_string()],
            "replaced wholesale, not unioned"
        );
    }

    // collection underlay: an already-set collection is never touched, even
    // by a file layer that specifies a different one.
    #[test]
    fn args_collection_underlay_never_element_merges() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let path = dir.path().join("command.toml");
        std::fs::write(&path, "args = [\"from-file-1\", \"from-file-2\"]\n").expect("write toml");
        let config = CommandConfig::layered()
            .with_args(["explicit"])
            .underlay_path(&path)
            .expect("underlay_path")
            .build();
        assert_eq!(
            config.args,
            vec!["explicit".to_string()],
            "already-set collection untouched"
        );
    }

    #[test]
    fn into_pipe_builder_returns_fully_set_typestate_builder() {
        let config = CommandConfig::builder()
            .program("/bin/true".to_string())
            .build();
        let pipe = config
            .into_pipe_builder()
            .expect("into_pipe_builder")
            .build();
        // TARGET 3 — served-Pipe `.name()` is gone; a successful
        // build() is the behavioral proof.
        let _ = pipe;
    }

    #[test]
    fn dispatch_choice_round_trips_through_toml() {
        let toml_text = r#"
program = "/bin/true"

[dispatch]
kind = "deny"
errno = 13
"#;
        let config: CommandConfig = toml::from_str(toml_text).expect("toml parse");
        assert!(matches!(
            config.dispatch,
            DispatchChoice::Deny { errno: 13 }
        ));
        let pipe = config
            .into_pipe_builder()
            .expect("into_pipe_builder")
            .build();
        // TARGET 3 — served-Pipe `.name()` is gone; a successful
        // build() is the behavioral proof.
        let _ = pipe;
    }

    #[test]
    fn dispatch_choice_default_is_empty() {
        assert!(matches!(DispatchChoice::default(), DispatchChoice::Empty));
    }

    // Principle-4 parity tests: same component built via the two
    // first-class paths must have equivalent internal state.

    #[test]
    fn parity_config_loader_and_command_builder_produce_equivalent_state() {
        // Path A: config-driven (the conflaguration / serde surface).
        let config = CommandConfig::builder()
            .program("/bin/echo".to_string())
            .args(alloc::vec!["hello".to_string()])
            .env(Env::from_iter([("LANG", "C")]))
            .inherit_parent_env(false)
            .libc_shim(true)
            .build();
        let from_config = config.clone().into_std_command().expect("config → command");

        // Path B: bon-derived fluent builder (the .builder().build()
        // surface).
        let from_builder = super::super::command::Command::builder()
            .program("/bin/echo")
            .args(alloc::vec![std::ffi::OsString::from("hello")])
            .libc_shim(true)
            .build();
        let mut from_builder = from_builder;
        from_builder.env_from(Env::from_iter([("LANG", "C")]));

        assert_eq!(from_config.get_program(), from_builder.get_program());
        assert_eq!(
            from_config.get_args().collect::<alloc::vec::Vec<_>>(),
            from_builder.get_args().collect::<alloc::vec::Vec<_>>(),
        );
        assert_eq!(from_config.env_snapshot(), from_builder.env_snapshot());
        assert_eq!(from_config.is_libc_shim(), from_builder.is_libc_shim());
    }

    #[test]
    fn parity_command_to_config_round_trip_preserves_additive_flags() {
        let mut cmd = super::super::command::Command::new("/bin/true");
        cmd.controlling_tty(true).umask(0o022).libc_shim();
        let restored = cmd.to_config().expect("to_config");
        assert!(restored.controlling_tty);
        assert_eq!(restored.umask, Some(0o022));
        assert!(restored.libc_shim);
    }

    #[test]
    fn parity_command_from_config_alias_matches_into_std_command() {
        let config = CommandConfig::builder()
            .program("/bin/echo".to_string())
            .args(alloc::vec!["hi".to_string()])
            .build();
        let via_into = config.clone().into_std_command().expect("into_std_command");
        let via_from = super::super::command::Command::from_config(config).expect("from_config");
        assert_eq!(via_into.get_program(), via_from.get_program());
        assert_eq!(
            via_into.get_args().collect::<alloc::vec::Vec<_>>(),
            via_from.get_args().collect::<alloc::vec::Vec<_>>()
        );
    }
}
