//! Alloc-tier descriptor of a process to spawn — the CString-based
//! raw form that [`super::spawn::spawn`] consumes.
//!
//! [`CommandDescriptor`] is the no_std+alloc primitive: pure data,
//! `CString` fields, no behaviour. It carries everything `execvp`
//! needs to launch a child. Std users do NOT touch this type —
//! they reach for [`super::command::Command`] (std-tier mirror of
//! [`std::process::Command`]) which lowers to a
//! `CommandDescriptor` at spawn time.
//!
//! # Two-tier split (per guiding principle 3 + drop-in compat)
//!
//! - `CommandDescriptor` (this type) — alloc-tier, CString
//!   storage, always compiles. Mechanical primitive.
//! - [`super::command::Command`] — std-tier, OsString/PathBuf
//!   storage, mirrors `std::process::Command` exactly so
//!   `use proxima_process::Command` is a drop-in for
//!   `use std::process::Command`. Lowers to `CommandDescriptor`
//!   at the conversion boundary.
//!
//! See `pty-tester/docs/proxima-pty/guiding-principles.md` for the
//! rationale.
//!
//! # Environment model
//!
//! `CommandDescriptor.env` is the **complete environment** the child will see
//! — a flat list of (key, value) pairs. It is NOT a list of edits on
//! top of the parent's environ. Callers who want the parent's env to
//! reach the child call [`CommandDescriptor::inherit_current_env`] to snapshot
//! `std::env::vars_os()` into the descriptor; from there, fluent
//! `env(key, value)` / `env_remove(key)` calls layer overrides onto
//! the snapshot. The default is **empty env** — explicit, no surprise
//! inheritance through fork.
//!
//! # Things NOT on CommandDescriptor
//!
//! Per std's lead, runtime-spawn knobs (umask, controlling-tty,
//! dispatch fd) do NOT live on the descriptor. They live in
//! [`super::spawn::SpawnOptions`], which the spawn primitive takes
//! alongside the command. Std uses `pre_exec` closures for the same
//! split; we use a typed options struct so the same shape can be
//! threaded through `CommandConfig` and proximad-over-the-wire
//! without needing serialisable closures.

extern crate alloc;

use alloc::ffi::CString;
use alloc::vec::Vec;
use core::ffi::c_int;

/// Wiring for a single byte stream between parent and child.
/// Mirrors [`std::process::Stdio`] — constructors are
/// [`Stdio::inherit`] / [`Stdio::null`] / [`Stdio::piped`], with
/// [`Stdio::Fd`] kept additive (std uses `From<OwnedFd>` for the
/// same role; we keep the enum tagged because `CommandConfig`
/// round-trips through serde).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Stdio {
    /// Inherit the parent process's fd. Standard default.
    #[default]
    Inherit,
    /// Open `/dev/null` and use it. Read returns EOF immediately;
    /// writes are discarded.
    Null,
    /// Use a caller-provided file descriptor. The spawner duplicates
    /// the fd into the child during pre-exec; the caller retains
    /// ownership of the original.
    ///
    /// PTY use-case: pass the slave-side fd here for all three
    /// streams so the child runs against the pseudo-terminal.
    Fd(c_int),
    /// Spawn allocates a `pipe(2)`. The child gets one end (read
    /// for `stdin`, write for `stdout`/`stderr`); the caller
    /// receives the other end in the [`Child`](super::spawn::Child)
    /// result and uses it to feed bytes or drain bytes.
    Piped,
}

impl Stdio {
    /// `Stdio::Inherit` — mirrors [`std::process::Stdio::inherit`].
    #[must_use]
    pub const fn inherit() -> Self {
        Self::Inherit
    }

    /// `Stdio::Null` — mirrors [`std::process::Stdio::null`].
    #[must_use]
    pub const fn null() -> Self {
        Self::Null
    }

    /// `Stdio::Piped` — mirrors [`std::process::Stdio::piped`].
    #[must_use]
    pub const fn piped() -> Self {
        Self::Piped
    }
}

impl From<c_int> for Stdio {
    /// Additive constructor — std exposes the same role through
    /// `From<OwnedFd>`/`From<ChildStdin>` rather than a raw int.
    /// We keep the raw form for the PTY slave-fd case.
    fn from(raw: c_int) -> Self {
        Self::Fd(raw)
    }
}

#[cfg(feature = "std")]
impl From<std::os::fd::OwnedFd> for Stdio {
    /// Mirrors [`std::process::Stdio: From<OwnedFd>`]. Ownership
    /// of the fd transfers in — we extract the raw fd via
    /// `into_raw_fd()` and the caller is on the hook for closing
    /// it after spawn (typically by retaining the original
    /// `OwnedFd` until the child has dup2'd, then dropping).
    fn from(fd: std::os::fd::OwnedFd) -> Self {
        use std::os::fd::IntoRawFd;
        Self::Fd(fd.into_raw_fd())
    }
}

#[cfg(feature = "std")]
impl From<std::fs::File> for Stdio {
    /// Mirrors [`std::process::Stdio: From<File>`]. The file's
    /// underlying fd becomes the child's stdio. Ownership
    /// transfers — the File is consumed; its fd lives until
    /// explicit close (or process exit).
    fn from(file: std::fs::File) -> Self {
        use std::os::fd::{IntoRawFd, OwnedFd};
        Self::Fd(OwnedFd::from(file).into_raw_fd())
    }
}

/// One entry in the child's environment: `key=value`.
///
/// Keys and values are `CString` to match `execve`'s
/// `*const *const c_char` envp expectation without any
/// allocation-in-the-child shenanigans. NUL bytes inside either
/// half are rejected at construction time by `CString::new`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnvVar {
    /// Variable name. Example: `CString::new("LANG").unwrap()`.
    pub key: CString,
    /// Variable value. Example: `CString::new("C").unwrap()`.
    pub value: CString,
}

/// Descriptor of a process to spawn — drop-in mirror of
/// [`std::process::Command`].
///
/// Construct via [`CommandDescriptor::new`] then chain fluent methods to add
/// args, env overrides, current_dir, and stdio overrides. The
/// descriptor itself is data; behaviour lives in the spawn
/// primitive (see [`super::spawn::spawn`]).
///
/// # Example
///
/// ```ignore
/// use alloc::ffi::CString;
/// use proxima_process::{CommandDescriptor, Stdio};
///
/// let mut cmd = CommandDescriptor::new(CString::new("/bin/ls").unwrap());
/// cmd.inherit_current_env()
///    .arg(CString::new("-la").unwrap())
///    .env(CString::new("LANG").unwrap(), CString::new("C").unwrap())
///    .current_dir(CString::new("/tmp").unwrap())
///    .stderr(Stdio::null());
/// ```
///
/// # Fields
///
/// All fields are `pub` so callers can construct the descriptor
/// directly (e.g. from a deserialized
/// [`super::command_config::CommandConfig`]) instead of going
/// through the fluent builder.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CommandDescriptor {
    /// `argv[0]` — the program to exec. May be a bare name
    /// (resolved via `PATH` at spawn time by the `std`-backed
    /// spawner) or an absolute / cwd-relative path (required by the
    /// tier-1 libc spawner, which performs no `PATH` search).
    pub program: CString,
    /// `argv[1..]` — additional arguments. Does NOT include
    /// `argv[0]`.
    pub args: Vec<CString>,
    /// Working directory for the spawned process. `None` inherits
    /// the parent's cwd. Name matches `std::process::Command::current_dir`.
    pub current_dir: Option<CString>,
    /// **Complete** environment for the child. Each `EnvVar` is a
    /// `KEY=VALUE` entry that lands in the child's `environ`. NOT
    /// a list of edits on top of the parent's env — this *is* the
    /// env. Defaults to empty; call [`Self::inherit_current_env`]
    /// to populate from the parent process.
    pub env: Vec<EnvVar>,
    /// Wiring for the child's stdin (kernel fd 0).
    /// Defaults to [`Stdio::Inherit`].
    pub stdin: Stdio,
    /// Wiring for the child's stdout (kernel fd 1).
    /// Defaults to [`Stdio::Inherit`].
    pub stdout: Stdio,
    /// Wiring for the child's stderr (kernel fd 2).
    /// Defaults to [`Stdio::Inherit`].
    pub stderr: Stdio,
}

impl CommandDescriptor {
    /// Construct a new `CommandDescriptor` invoking `program` with no args,
    /// **empty environment**, inherited current_dir, and inherited
    /// stdio. Call [`Self::inherit_current_env`] if you want the
    /// parent's env to reach the child. Mirrors
    /// [`std::process::Command::new`].
    #[must_use]
    pub fn new(program: CString) -> Self {
        Self {
            program,
            args: Vec::new(),
            current_dir: None,
            env: Vec::new(),
            stdin: Stdio::Inherit,
            stdout: Stdio::Inherit,
            stderr: Stdio::Inherit,
        }
    }

    /// Append a single argument to `argv`.
    pub fn arg(&mut self, argument: CString) -> &mut Self {
        self.args.push(argument);
        self
    }

    /// Append multiple arguments.
    pub fn args<I: IntoIterator<Item = CString>>(&mut self, arguments: I) -> &mut Self {
        self.args.extend(arguments);
        self
    }

    /// Set the working directory. Mirrors
    /// [`std::process::Command::current_dir`].
    pub fn current_dir(&mut self, directory: CString) -> &mut Self {
        self.current_dir = Some(directory);
        self
    }

    /// Set or replace `key=value` in the env list. If `key`
    /// already exists, its value is overwritten; otherwise the
    /// entry is appended.
    pub fn env(&mut self, key: CString, value: CString) -> &mut Self {
        if let Some(entry) = self.env.iter_mut().find(|entry| entry.key == key) {
            entry.value = value;
        } else {
            self.env.push(EnvVar { key, value });
        }
        self
    }

    /// Replace the env with the supplied iterator of pairs. Matches
    /// [`std::process::Command::envs`]'s spirit (additive variant
    /// available — chain `env(...)` calls).
    pub fn envs<I: IntoIterator<Item = (CString, CString)>>(&mut self, entries: I) -> &mut Self {
        for (key, value) in entries {
            self.env(key, value);
        }
        self
    }

    /// Remove `key` from the env list if present.
    pub fn env_remove(&mut self, key: &core::ffi::CStr) -> &mut Self {
        self.env.retain(|entry| entry.key.as_c_str() != key);
        self
    }

    /// Clear the entire env list. The child will see an empty
    /// environment unless subsequent [`Self::env`] /
    /// [`Self::inherit_current_env`] calls repopulate.
    pub fn env_clear(&mut self) -> &mut Self {
        self.env.clear();
        self
    }

    /// Snapshot the current process's environment into `self.env`.
    ///
    /// Each (key, value) from `std::env::vars_os()` becomes (or
    /// replaces) an `EnvVar` in the list. UTF-8-invalid keys/values
    /// or those containing NUL bytes are silently skipped — `execve`
    /// can't carry them anyway. Additive — std has no parallel
    /// (its CommandDescriptor snapshots automatically unless you call
    /// `env_clear`).
    #[cfg(feature = "std")]
    pub fn inherit_current_env(&mut self) -> &mut Self {
        for (raw_key, raw_value) in std::env::vars_os() {
            let Some(key_str) = raw_key.to_str() else {
                continue;
            };
            let Some(value_str) = raw_value.to_str() else {
                continue;
            };
            let Ok(key) = CString::new(key_str) else {
                continue;
            };
            let Ok(value) = CString::new(value_str) else {
                continue;
            };
            self.env(key, value);
        }
        self
    }

    /// Configure the child's stdin (kernel fd 0). Mirrors
    /// [`std::process::Command::stdin`].
    pub fn stdin(&mut self, kind: Stdio) -> &mut Self {
        self.stdin = kind;
        self
    }

    /// Configure the child's stdout (kernel fd 1). Mirrors
    /// [`std::process::Command::stdout`].
    pub fn stdout(&mut self, kind: Stdio) -> &mut Self {
        self.stdout = kind;
        self
    }

    /// Configure the child's stderr (kernel fd 2). Mirrors
    /// [`std::process::Command::stderr`].
    pub fn stderr(&mut self, kind: Stdio) -> &mut Self {
        self.stderr = kind;
        self
    }
}

/// Error returned by [`CommandDescriptor`]'s `TryFrom`
/// conversions when the input can't be split into a valid
/// `program + args` shape.
#[derive(Debug)]
pub enum CommandParseError {
    /// Input was empty / whitespace-only — no program token.
    Empty,
    /// shell-words parser rejected the input (unbalanced quotes,
    /// trailing backslash, etc.).
    Shell(shell_words::ParseError),
    /// One of the parsed tokens contained a NUL byte —
    /// `CString::new` rejected it.
    Nul(alloc::ffi::NulError),
    /// Input bytes were not valid UTF-8 (only relevant for the
    /// byte-slice `TryFrom`s).
    Utf8(core::str::Utf8Error),
}

impl core::fmt::Display for CommandParseError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Empty => formatter.write_str("command string is empty"),
            Self::Shell(err) => write!(formatter, "shell parse: {err}"),
            Self::Nul(err) => write!(formatter, "token contains NUL: {err}"),
            Self::Utf8(err) => write!(formatter, "bytes are not utf-8: {err}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CommandParseError {}

impl From<shell_words::ParseError> for CommandParseError {
    fn from(err: shell_words::ParseError) -> Self {
        Self::Shell(err)
    }
}

impl From<alloc::ffi::NulError> for CommandParseError {
    fn from(err: alloc::ffi::NulError) -> Self {
        Self::Nul(err)
    }
}

impl From<core::str::Utf8Error> for CommandParseError {
    fn from(err: core::str::Utf8Error) -> Self {
        Self::Utf8(err)
    }
}

fn parse_command_line(input: &str) -> Result<CommandDescriptor, CommandParseError> {
    let mut tokens = shell_words::split(input)?.into_iter();
    let program = tokens.next().ok_or(CommandParseError::Empty)?;
    let mut descriptor = CommandDescriptor::new(CString::new(program)?);
    for token in tokens {
        descriptor.arg(CString::new(token)?);
    }
    Ok(descriptor)
}

impl TryFrom<&str> for CommandDescriptor {
    type Error = CommandParseError;
    /// Shell-parse the input into `program + args`. Quoting and
    /// escapes follow POSIX shell rules via the `shell-words`
    /// crate. `"git checkout main"` → `program="git",
    /// args=["checkout", "main"]`. NULs in any token return
    /// [`CommandParseError::Nul`].
    fn try_from(input: &str) -> Result<Self, Self::Error> {
        parse_command_line(input)
    }
}

impl TryFrom<alloc::string::String> for CommandDescriptor {
    type Error = CommandParseError;
    fn try_from(input: alloc::string::String) -> Result<Self, Self::Error> {
        parse_command_line(&input)
    }
}

impl TryFrom<&[u8]> for CommandDescriptor {
    type Error = CommandParseError;
    /// Decode as UTF-8, then shell-parse. Useful for config bytes
    /// from a file / socket where you haven't enforced UTF-8
    /// up-stream.
    fn try_from(bytes: &[u8]) -> Result<Self, Self::Error> {
        parse_command_line(core::str::from_utf8(bytes)?)
    }
}

impl<const N: usize> TryFrom<&[u8; N]> for CommandDescriptor {
    type Error = CommandParseError;
    fn try_from(bytes: &[u8; N]) -> Result<Self, Self::Error> {
        Self::try_from(bytes.as_slice())
    }
}

impl TryFrom<alloc::vec::Vec<u8>> for CommandDescriptor {
    type Error = CommandParseError;
    fn try_from(bytes: alloc::vec::Vec<u8>) -> Result<Self, Self::Error> {
        Self::try_from(bytes.as_slice())
    }
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
    use alloc::vec;

    fn cstr(text: &str) -> CString {
        CString::new(text).expect("test literal contains no interior NUL")
    }

    #[test]
    fn new_command_starts_with_empty_env_and_inherited_stdio() {
        let command = CommandDescriptor::new(cstr("/bin/true"));

        assert_eq!(command.program, cstr("/bin/true"));
        assert!(command.args.is_empty());
        assert_eq!(command.current_dir, None);
        assert!(command.env.is_empty());
        assert_eq!(command.stdin, Stdio::Inherit);
        assert_eq!(command.stdout, Stdio::Inherit);
        assert_eq!(command.stderr, Stdio::Inherit);
    }

    #[test]
    fn fluent_builder_chains_in_order() {
        let mut command = CommandDescriptor::new(cstr("/bin/ls"));
        command
            .arg(cstr("-la"))
            .args([cstr("/etc"), cstr("/tmp")])
            .current_dir(cstr("/var"))
            .env(cstr("LANG"), cstr("C"))
            .stderr(Stdio::null());

        assert_eq!(command.args, vec![cstr("-la"), cstr("/etc"), cstr("/tmp")]);
        assert_eq!(command.current_dir, Some(cstr("/var")));
        assert_eq!(command.env.len(), 1);
        assert_eq!(command.env[0].key, cstr("LANG"));
        assert_eq!(command.env[0].value, cstr("C"));
        assert_eq!(command.stderr, Stdio::Null);
    }

    #[test]
    fn env_set_overwrites_existing_key() {
        let mut command = CommandDescriptor::new(cstr("/bin/env"));
        command
            .env(cstr("LANG"), cstr("C"))
            .env(cstr("LANG"), cstr("en_US.UTF-8"));

        assert_eq!(command.env.len(), 1);
        assert_eq!(command.env[0].key, cstr("LANG"));
        assert_eq!(command.env[0].value, cstr("en_US.UTF-8"));
    }

    #[test]
    fn env_remove_drops_existing_entry() {
        let mut command = CommandDescriptor::new(cstr("/bin/env"));
        command
            .env(cstr("LANG"), cstr("C"))
            .env(cstr("PATH"), cstr("/usr/bin"))
            .env_remove(cstr("LANG").as_c_str());

        assert_eq!(command.env.len(), 1);
        assert_eq!(command.env[0].key, cstr("PATH"));
    }

    #[test]
    fn env_clear_empties_the_list() {
        let mut command = CommandDescriptor::new(cstr("/bin/env"));
        command
            .env(cstr("LANG"), cstr("C"))
            .env(cstr("PATH"), cstr("/usr/bin"))
            .env_clear()
            .env(cstr("ONLY"), cstr("survivor"));

        assert_eq!(command.env.len(), 1);
        assert_eq!(command.env[0].key, cstr("ONLY"));
    }

    #[test]
    fn envs_inserts_multiple_pairs() {
        let mut command = CommandDescriptor::new(cstr("/bin/env"));
        command.envs([(cstr("A"), cstr("1")), (cstr("B"), cstr("2"))]);
        assert_eq!(command.env.len(), 2);
        assert_eq!(command.env[0].key, cstr("A"));
        assert_eq!(command.env[1].key, cstr("B"));
    }

    #[cfg(feature = "std")]
    #[test]
    fn inherit_current_env_populates_from_process() {
        let key = "PTY_TESTER_INHERIT_MARKER";
        unsafe { std::env::set_var(key, "captured") };

        let mut command = CommandDescriptor::new(cstr("/bin/true"));
        command.inherit_current_env();
        let captured = command
            .env
            .iter()
            .find(|entry| entry.key.as_bytes() == key.as_bytes())
            .expect("marker var captured");
        assert_eq!(captured.value, cstr("captured"));

        unsafe { std::env::remove_var(key) };
    }

    #[test]
    fn stdio_fd_carries_caller_owned_descriptor() {
        let pty_slave: c_int = 7;
        let mut command = CommandDescriptor::new(cstr("/bin/sh"));
        command
            .stdin(Stdio::Fd(pty_slave))
            .stdout(Stdio::Fd(pty_slave))
            .stderr(Stdio::Fd(pty_slave));

        assert_eq!(command.stdin, Stdio::Fd(7));
        assert_eq!(command.stdout, Stdio::Fd(7));
        assert_eq!(command.stderr, Stdio::Fd(7));
    }

    #[test]
    fn stdio_piped_signals_pipe_allocation() {
        let mut command = CommandDescriptor::new(cstr("/bin/cat"));
        command.stdin(Stdio::piped()).stdout(Stdio::piped());
        assert_eq!(command.stdin, Stdio::Piped);
        assert_eq!(command.stdout, Stdio::Piped);
        assert_eq!(command.stderr, Stdio::Inherit);
    }

    #[test]
    fn try_from_str_shell_parses_program_and_args() {
        let cmd: CommandDescriptor = "git checkout main".try_into().expect("shell parse");
        assert_eq!(cmd.program.to_str().expect("utf8"), "git");
        assert_eq!(cmd.args.len(), 2);
        assert_eq!(cmd.args[0].to_str().expect("utf8"), "checkout");
        assert_eq!(cmd.args[1].to_str().expect("utf8"), "main");
    }

    #[test]
    fn try_from_byte_array_decodes_then_parses() {
        let cmd: CommandDescriptor = b"/bin/ls -la".try_into().expect("parse");
        assert_eq!(cmd.program.to_str().expect("utf8"), "/bin/ls");
    }

    #[test]
    fn try_from_nul_in_token_rejected() {
        let err = CommandDescriptor::try_from("/bin/echo \"has\0nul\"").expect_err("nul must fail");
        assert!(matches!(err, CommandParseError::Nul(_)));
    }

    #[test]
    fn try_from_empty_returns_error() {
        let err = CommandDescriptor::try_from("   ").expect_err("empty must fail");
        assert!(matches!(err, CommandParseError::Empty));
    }

    #[test]
    fn try_from_invalid_utf8_returns_error() {
        let bytes: &[u8] = &[0xff, 0xfe, 0xfd];
        let err = CommandDescriptor::try_from(bytes).expect_err("invalid utf8 fails");
        assert!(matches!(err, CommandParseError::Utf8(_)));
    }

    #[test]
    fn stdio_from_int_uses_fd_variant() {
        let stdio: Stdio = 5.into();
        assert_eq!(stdio, Stdio::Fd(5));
    }
}
