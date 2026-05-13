//! libc-based process spawn primitive.
//!
//! Consumes a tier-1 [`CommandDescriptor`] descriptor and produces a
//! [`Child`] containing the child's pid plus parent-side
//! file descriptors for every [`Stdio::Piped`] slot.
//!
//! # What it honors
//!
//! Every field of [`CommandDescriptor`]:
//! - `program` + `args` → `execvp` argv
//! - `cwd` → `chdir(2)` in the child before exec
//! - `env` → built into a `*const *const c_char` envp **in the
//!   parent** (allocation-safe context), then handed to the child
//!   via a single `environ` pointer swap. The child performs **no**
//!   `setenv` / `unsetenv` / `clearenv` calls — pure data pass-
//!   through. `execvp` uses the swapped `environ` for both the
//!   child's env AND its `PATH` search.
//! - `umask` → `umask(2)` in the child before exec
//! - `controlling_tty` → `setsid()` + `ioctl(TIOCSCTTY, 0)` on fd 0
//! - per-slot [`Stdio`] wiring:
//!   - [`Stdio::Inherit`] — child inherits parent's fd at that index
//!   - [`Stdio::Null`] — open `/dev/null` in the child, dup2 onto the slot
//!   - [`Stdio::Fd(n)`] — child dup2s `n` onto the slot (caller retains
//!     ownership of `n`)
//!   - [`Stdio::Piped`] — `pipe(2)` allocated in the parent before fork;
//!     child dup2s its end, parent keeps the other in [`Child`]
//!
//! # Hazard
//!
//! `fork(2)` from a multi-threaded parent is unsafe (held mutexes
//! survive into the child with no thread to release them). Direct
//! callers of `spawn` should route through
//! [`ForkServer`](super::fork_server::ForkServer) when the parent
//! is multi-threaded. The fork-server is single-threaded by
//! construction so its `spawn` is safe; multi-threaded direct
//! callers risk deadlocks.

use std::ffi::CString;
use std::os::fd::{AsRawFd, OwnedFd, RawFd};
use std::ptr;

use proxima_primitives::pipe::ProximaError;

use super::descriptor::{CommandDescriptor, EnvVar, Stdio};
use super::fd_pipe::make_pipe;

/// Result of a successful spawn: child pid + parent-side fds for
/// every [`Stdio::Piped`] slot.
#[derive(Debug)]
pub struct Child {
    /// Child process id.
    pub pid: libc::pid_t,
    /// Parent's write end of the input pipe when
    /// `CommandDescriptor.input == Stdio::Piped`; `None` otherwise.
    pub stdin: Option<OwnedFd>,
    /// Parent's read end of the output pipe when
    /// `CommandDescriptor.output == Stdio::Piped`; `None` otherwise.
    pub stdout: Option<OwnedFd>,
    /// Parent's read end of the error pipe when
    /// `CommandDescriptor.error == Stdio::Piped`; `None` otherwise.
    pub stderr: Option<OwnedFd>,
}

/// Slot direction: where in the child the spawned fd ends up
/// (kernel fd 0 / 1 / 2) and whether the child reads or writes it.
#[derive(Debug, Clone, Copy)]
enum Slot {
    Input,
    Output,
    Error,
}

impl Slot {
    const fn child_fd(self) -> libc::c_int {
        match self {
            Self::Input => 0,
            Self::Output => 1,
            Self::Error => 2,
        }
    }

    const fn child_reads(self) -> bool {
        matches!(self, Self::Input)
    }
}

/// Allocated state for one [`Stdio::Piped`] slot before fork.
struct PipedPair {
    slot: Slot,
    /// fd the child dup2s onto its slot (then closes the original).
    child_end: OwnedFd,
    /// fd the parent retains in the returned [`Child`].
    parent_end: OwnedFd,
}

/// Pre-built envp: owned `KEY=VALUE\0` strings + a null-terminated
/// pointer array for handing to `execvp` via `environ`.
struct Envp {
    /// Backing storage so the C strings stay alive across fork.
    _entries: Vec<CString>,
    /// `*const *const c_char` array, null-terminated. Cast to
    /// `*mut *mut c_char` only at the `environ` assignment site —
    /// `execvp` reads, it does not write.
    pointers: Vec<*mut libc::c_char>,
}

/// Target fd in the child for the dispatch socket. Matches
/// [`super::dispatched::DISPATCH_FD`] — the one well-known
/// side-channel fd convention this crate carries.
pub const DISPATCH_FD_TARGET: libc::c_int = 7;

/// Runtime knobs honoured by [`spawn`] that do NOT belong on the
/// pure-data [`CommandDescriptor`] descriptor. Std uses
/// `std::os::unix::process::CommandExt::pre_exec` closures for
/// the same concerns; we use a typed struct so the same shape
/// rides over the wire through
/// [`super::command_config::CommandConfig`] without needing
/// serialisable closures.
///
/// `Default` zeros every field — no dispatch socket, no
/// controlling-tty acquisition, no umask change. Most call sites
/// should construct via the struct literal so the intent is
/// explicit at the call.
#[derive(Debug, Clone, Copy, Default)]
pub struct SpawnOptions {
    /// If `Some(parent_socket_fd)`, pre-exec dup2s the fd onto
    /// [`DISPATCH_FD_TARGET`] so the child sees the dispatch
    /// socket at the canonical fd number.
    pub dispatch_fd: Option<RawFd>,
    /// If `true`, child calls `setsid()` + `ioctl(TIOCSCTTY, 0)`
    /// on fd 0 so it becomes the controlling-tty session leader.
    /// PTY-wrapper code sets this; vanilla `CommandPipe` leaves
    /// it `false`.
    pub controlling_tty: bool,
    /// File-creation umask to apply in the child before exec.
    /// `None` inherits the parent's umask. Maximum 0o777.
    pub umask: Option<u32>,
}

/// Spawn `command` as a child process honouring `options`
/// (dispatch-fd wiring, controlling-tty setup, umask). `CommandDescriptor`
/// itself stays pure data; anything that varies per spawn site
/// rides in [`SpawnOptions`].
pub fn spawn(command: &CommandDescriptor, options: SpawnOptions) -> Result<Child, ProximaError> {
    let mut piped_pairs: Vec<PipedPair> = Vec::new();
    for slot in [Slot::Input, Slot::Output, Slot::Error] {
        if matches!(slot_io(command, slot), Stdio::Piped) {
            piped_pairs.push(allocate_piped_pair(slot)?);
        }
    }

    let mut argv_storage: Vec<&std::ffi::CStr> = Vec::with_capacity(command.args.len() + 1);
    argv_storage.push(&command.program);
    for argument in &command.args {
        argv_storage.push(argument);
    }
    let argv_pointers: Vec<*const libc::c_char> = argv_storage
        .iter()
        .map(|cstr| cstr.as_ptr())
        .chain(std::iter::once(ptr::null()))
        .collect();

    let envp = build_envp(&command.env);
    let program_pointer = command.program.as_ptr();

    let pid = unsafe { libc::fork() };
    if pid < 0 {
        return Err(ProximaError::Body(format!(
            "libc::fork failed: {}",
            std::io::Error::last_os_error()
        )));
    }

    if pid == 0 {
        run_child(
            command,
            &piped_pairs,
            &envp,
            program_pointer,
            argv_pointers.as_ptr(),
            options,
        );
    }

    Ok(parent_after_fork(pid, piped_pairs))
}

fn slot_io(command: &CommandDescriptor, slot: Slot) -> Stdio {
    match slot {
        Slot::Input => command.stdin,
        Slot::Output => command.stdout,
        Slot::Error => command.stderr,
    }
}

fn allocate_piped_pair(slot: Slot) -> Result<PipedPair, ProximaError> {
    let (read_end, write_end) = make_pipe()?;
    let (child_end, parent_end) = if slot.child_reads() {
        (read_end, write_end)
    } else {
        (write_end, read_end)
    };
    Ok(PipedPair {
        slot,
        child_end,
        parent_end,
    })
}

fn parent_after_fork(pid: libc::pid_t, piped_pairs: Vec<PipedPair>) -> Child {
    let mut child = Child {
        pid,
        stdin: None,
        stdout: None,
        stderr: None,
    };
    for pair in piped_pairs {
        let PipedPair {
            slot,
            child_end,
            parent_end,
        } = pair;
        drop(child_end);
        match slot {
            Slot::Input => child.stdin = Some(parent_end),
            Slot::Output => child.stdout = Some(parent_end),
            Slot::Error => child.stderr = Some(parent_end),
        }
    }
    child
}

fn build_envp(env: &[EnvVar]) -> Envp {
    let entries: Vec<CString> = env
        .iter()
        .filter_map(|entry| {
            let mut combined =
                Vec::with_capacity(entry.key.as_bytes().len() + 1 + entry.value.as_bytes().len());
            combined.extend_from_slice(entry.key.as_bytes());
            combined.push(b'=');
            combined.extend_from_slice(entry.value.as_bytes());
            CString::new(combined).ok()
        })
        .collect();
    let mut pointers: Vec<*mut libc::c_char> = entries
        .iter()
        .map(|entry| entry.as_ptr() as *mut _)
        .collect();
    pointers.push(ptr::null_mut());
    Envp {
        _entries: entries,
        pointers,
    }
}

fn run_child(
    command: &CommandDescriptor,
    piped_pairs: &[PipedPair],
    envp: &Envp,
    program: *const libc::c_char,
    argv: *const *const libc::c_char,
    options: SpawnOptions,
) -> ! {
    for slot in [Slot::Input, Slot::Output, Slot::Error] {
        if wire_slot_in_child(command, slot, piped_pairs).is_err() {
            unsafe { libc::_exit(127) };
        }
    }

    // Close every parent-side piped fd that the child inherited.
    // MUST happen BEFORE the dispatch-fd dup2 — otherwise a
    // parent_end whose fd number happens to collide with
    // DISPATCH_FD_TARGET (or with the dispatch socket's source
    // fd) would be closed AFTER the dup2, wiping the dispatch
    // socket. Order: stdio wired, then parent_ends closed, then
    // dispatch dup'd onto its canonical slot.
    for pair in piped_pairs {
        unsafe {
            libc::close(pair.parent_end.as_raw_fd());
        }
    }

    // Wire the dispatch socket onto its canonical fd target if
    // the caller asked for it. Single optional fd instead of a
    // generic Vec — proxima-process only needs the one side
    // channel.
    if let Some(source) = options.dispatch_fd
        && source != DISPATCH_FD_TARGET
        && unsafe { libc::dup2(source, DISPATCH_FD_TARGET) } < 0
    {
        unsafe { libc::_exit(127) };
    }

    if let Some(cwd) = command.current_dir.as_ref()
        && unsafe { libc::chdir(cwd.as_ptr()) } != 0
    {
        unsafe { libc::_exit(127) };
    }

    if let Some(mask) = options.umask {
        unsafe { libc::umask(mask as libc::mode_t) };
    }

    // Swap `environ` to point at the envp we built in the parent.
    // execvp will read this for the child's environment AND for
    // `PATH` resolution. One pointer assignment — no per-var
    // setenv/unsetenv calls, no async-signal-unsafe libc churn
    // between fork and exec.
    unsafe {
        #[allow(static_mut_refs)]
        {
            environ = envp.pointers.as_ptr() as *mut _;
        }
    }

    if options.controlling_tty {
        unsafe {
            if libc::setsid() == -1 {
                libc::_exit(127);
            }
            if libc::ioctl(0, libc::TIOCSCTTY as _, 0) == -1 {
                libc::_exit(127);
            }
        }
    }

    unsafe {
        libc::execvp(program, argv);
        libc::_exit(127);
    }
}

unsafe extern "C" {
    static mut environ: *mut *mut libc::c_char;
}

fn wire_slot_in_child(
    command: &CommandDescriptor,
    slot: Slot,
    piped_pairs: &[PipedPair],
) -> Result<(), libc::c_int> {
    let slot_fd = slot.child_fd();
    match slot_io(command, slot) {
        Stdio::Inherit => Ok(()),
        Stdio::Null => {
            let null_fd = unsafe { libc::open(c"/dev/null".as_ptr(), libc::O_RDWR) };
            if null_fd < 0 {
                return Err(127);
            }
            let result = unsafe { libc::dup2(null_fd, slot_fd) };
            unsafe { libc::close(null_fd) };
            if result < 0 { Err(127) } else { Ok(()) }
        }
        Stdio::Fd(source_fd) => {
            if unsafe { libc::dup2(source_fd, slot_fd) } < 0 {
                Err(127)
            } else {
                Ok(())
            }
        }
        Stdio::Piped => {
            let pair = piped_pairs
                .iter()
                .find(|pair| pair.slot.child_fd() == slot_fd)
                .ok_or(127)?;
            let source = pair.child_end.as_raw_fd();
            if unsafe { libc::dup2(source, slot_fd) } < 0 {
                Err(127)
            } else {
                Ok(())
            }
        }
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
    use std::io::{Read, Write};
    use std::os::fd::{FromRawFd, IntoRawFd};

    fn cstr(text: &str) -> CString {
        CString::new(text).expect("test literal contains no interior NUL")
    }

    fn drain_to_string(fd: OwnedFd) -> String {
        let raw_fd = fd.into_raw_fd();
        let mut file = unsafe { std::fs::File::from_raw_fd(raw_fd) };
        let mut buffer = String::new();
        file.read_to_string(&mut buffer).expect("read child output");
        buffer
    }

    fn wait_child(pid: libc::pid_t) -> libc::c_int {
        let mut status: libc::c_int = 0;
        let waited = unsafe { libc::waitpid(pid, &mut status, 0) };
        assert_eq!(waited, pid, "waitpid returned unexpected pid");
        status
    }

    #[test]
    fn spawn_echo_with_piped_output() {
        let mut command = CommandDescriptor::new(cstr("/bin/echo"));
        command.arg(cstr("hello via spawn")).stdout(Stdio::Piped);

        let mut spawned = spawn(&command, SpawnOptions::default()).expect("spawn");
        assert!(spawned.stdin.is_none());
        assert!(spawned.stderr.is_none());
        let output_fd = spawned.stdout.take().expect("piped output fd");
        let captured = drain_to_string(output_fd);
        assert_eq!(captured, "hello via spawn\n");
        wait_child(spawned.pid);
    }

    #[test]
    fn spawn_tr_with_piped_input_and_output() {
        let mut command = CommandDescriptor::new(cstr("/usr/bin/tr"));
        command
            .arg(cstr("a-z"))
            .arg(cstr("A-Z"))
            .stdin(Stdio::Piped)
            .stdout(Stdio::Piped);

        let mut spawned = spawn(&command, SpawnOptions::default()).expect("spawn");
        let input_fd = spawned.stdin.take().expect("piped input fd");
        let output_fd = spawned.stdout.take().expect("piped output fd");

        {
            let raw = input_fd.into_raw_fd();
            let mut writer = unsafe { std::fs::File::from_raw_fd(raw) };
            writer
                .write_all(b"hello uppercase")
                .expect("write to tr stdin");
        }

        let captured = drain_to_string(output_fd);
        assert_eq!(captured, "HELLO UPPERCASE");
        wait_child(spawned.pid);
    }

    #[test]
    fn spawn_with_cwd_changes_directory() {
        let mut command = CommandDescriptor::new(cstr("/bin/pwd"));
        command.current_dir(cstr("/tmp")).stdout(Stdio::Piped);

        let mut spawned = spawn(&command, SpawnOptions::default()).expect("spawn");
        let output_fd = spawned.stdout.take().expect("piped output");
        let captured = drain_to_string(output_fd);
        assert!(
            captured.trim() == "/tmp" || captured.trim() == "/private/tmp",
            "pwd output was {:?}",
            captured.trim()
        );
        wait_child(spawned.pid);
    }

    #[test]
    fn spawn_passes_env_through_verbatim() {
        // /usr/bin/env (absolute) is the program — no PATH search
        // needed; the env we hand to the child IS exactly what we
        // built in the parent.
        let mut command = CommandDescriptor::new(cstr("/usr/bin/env"));
        command
            .env(cstr("FORCE_VALUE"), cstr("set-by-spawn"))
            .env(cstr("LANG"), cstr("C"))
            .stdout(Stdio::Piped);

        let mut spawned = spawn(&command, SpawnOptions::default()).expect("spawn");
        let captured = drain_to_string(spawned.stdout.take().expect("piped output"));
        assert!(
            captured.contains("FORCE_VALUE=set-by-spawn"),
            "env output: {captured:?}"
        );
        assert!(captured.contains("LANG=C"), "env output: {captured:?}");
        // FORCE_VALUE is not set in our test environment normally;
        // confirms we passed through our env, not the parent's.
        assert!(
            !captured.contains("HOME=") || captured.matches('\n').count() <= 3,
            "expected ONLY our explicit env, got: {captured:?}"
        );
        wait_child(spawned.pid);
    }

    #[test]
    fn spawn_inherit_current_env_lets_path_resolve_bare_names() {
        // "true" without a leading slash → execvp needs PATH to
        // resolve. inherit_current_env carries the parent's PATH
        // into the child's environ via our envp swap.
        let mut command = CommandDescriptor::new(cstr("true"));
        command.inherit_current_env();
        let spawned = spawn(&command, SpawnOptions::default()).expect("spawn");
        assert!(spawned.stdin.is_none());
        assert!(spawned.stdout.is_none());
        assert!(spawned.stderr.is_none());
        let status = wait_child(spawned.pid);
        assert!(libc::WIFEXITED(status) && libc::WEXITSTATUS(status) == 0);
    }

    #[test]
    fn spawn_dispatch_fd_makes_socket_visible_to_child() {
        // Parent allocates a pipe; child gets the write end on the
        // canonical dispatch fd target (DISPATCH_FD_TARGET = 7) and
        // writes via `echo hi >&7`; parent reads from the read end
        // and asserts.
        let (read_end, write_end) = make_pipe().expect("pipe");
        let source_fd = write_end.as_raw_fd();

        let mut command = CommandDescriptor::new(cstr("/bin/sh"));
        command
            .inherit_current_env()
            .arg(cstr("-c"))
            .arg(cstr("echo hi >&7"));

        let spawned = spawn(
            &command,
            SpawnOptions {
                dispatch_fd: Some(source_fd),
                ..SpawnOptions::default()
            },
        )
        .expect("spawn /bin/sh with dispatch fd");

        drop(write_end);

        let captured = drain_to_string(read_end);
        assert_eq!(captured, "hi\n");

        let status = wait_child(spawned.pid);
        assert!(libc::WIFEXITED(status) && libc::WEXITSTATUS(status) == 0);
    }

    #[test]
    fn spawn_inherit_default_leaves_stdio_alone() {
        let spawned = spawn(
            CommandDescriptor::new(cstr("/bin/true")).inherit_current_env(),
            SpawnOptions::default(),
        )
        .expect("spawn /bin/true");
        assert!(spawned.stdin.is_none());
        assert!(spawned.stdout.is_none());
        assert!(spawned.stderr.is_none());
        wait_child(spawned.pid);
    }
}
