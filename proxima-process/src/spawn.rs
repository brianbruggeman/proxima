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
//! # Spawn path
//!
//! [`spawn_external`] uses Linux `posix_spawnp(3)`, so ordinary external
//! commands are safe when callers resolve several children from a
//! multi-threaded parent. [`spawn`] remains the full pre-exec `fork(2)` path
//! for dispatch sockets, controlling terminals, umasks, and legacy callers;
//! those callers still need the single-threaded fork-server boundary.

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
    /// Child process id. Private — once reaped, the OS is free to
    /// recycle this number for an unrelated process, so it must
    /// only ever reach a syscall through [`Child`]'s own methods,
    /// which know whether it has been reaped.
    pid: libc::pid_t,
    /// Cached exit code once the child has been reaped via
    /// [`Child::wait`] or [`Child::try_wait`]. `None` means not yet
    /// reaped — the pid is still safe to signal or wait on.
    exit: Option<i32>,
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

impl Child {
    /// The child's process id.
    ///
    /// Prefer [`Child::try_wait`] / [`Child::kill`] over handing this
    /// raw pid to a syscall directly — once reaped, the OS may recycle
    /// it for an unrelated process, and only `Child` itself tracks
    /// whether that has happened.
    #[must_use]
    pub fn pid(&self) -> libc::pid_t {
        self.pid
    }

    /// Block until the child exits, returning its exit code.
    ///
    /// A child killed by a signal reports `128 + signal`, matching shell
    /// convention. Drop any piped `stdin` first — a child reading stdin will
    /// not exit while the parent holds the write end open.
    ///
    /// Once the child has been reaped (by a prior `wait` or `try_wait`),
    /// returns the cached exit code without issuing another `waitpid` —
    /// the pid may have since been recycled by the OS.
    ///
    /// Mirrors [`std::process::Child::wait`] in shape.
    pub fn wait(&mut self) -> Result<i32, ProximaError> {
        if let Some(code) = self.exit {
            return Ok(code);
        }

        let mut status: libc::c_int = 0;

        // SAFETY: waitpid is a kernel call taking our own child's pid and a
        // pointer to a stack int; no Rust invariant is at risk. self.exit is
        // still None here, so this pid has not yet been reaped and cannot
        // have been recycled by the OS.
        let waited = unsafe { libc::waitpid(self.pid, &mut status, 0) };

        if waited != self.pid {
            return Err(ProximaError::Body(format!(
                "waitpid returned {waited}, expected {}",
                self.pid
            )));
        }

        let code = exit_code(status);
        self.exit = Some(code);
        Ok(code)
    }

    /// Non-blocking liveness check. `Ok(None)` means still running;
    /// `Ok(Some(code))` means it has exited (or just did), decoded via
    /// the same shell convention as [`Child::wait`].
    ///
    /// Once reaped, subsequent calls return the cached code without
    /// issuing another `waitpid` — the pid may have since been recycled.
    pub fn try_wait(&mut self) -> Result<Option<i32>, ProximaError> {
        if let Some(code) = self.exit {
            return Ok(Some(code));
        }

        let mut status: libc::c_int = 0;

        // SAFETY: waitpid is a kernel call taking our own child's pid and a
        // pointer to a stack int; WNOHANG makes this a non-blocking poll
        // rather than a wait. self.exit is still None here, so this pid has
        // not yet been reaped and cannot have been recycled by the OS.
        let waited = unsafe { libc::waitpid(self.pid, &mut status, libc::WNOHANG) };

        if waited == 0 {
            Ok(None)
        } else if waited == self.pid {
            let code = exit_code(status);
            self.exit = Some(code);
            Ok(Some(code))
        } else {
            Err(ProximaError::Body(format!(
                "waitpid returned {waited}, expected {}",
                self.pid
            )))
        }
    }

    /// Send `SIGKILL`. A no-op if the child has already been reaped —
    /// never signals a pid the OS may have since recycled.
    pub fn kill(&mut self) -> Result<(), ProximaError> {
        if self.exit.is_some() {
            return Ok(());
        }

        // SAFETY: pid is our own spawned child and self.exit is still None,
        // so it has not been reaped and cannot have been recycled by the OS;
        // SIGKILL against our own live child is always a valid operation.
        let result = unsafe { libc::kill(self.pid, libc::SIGKILL) };

        if result < 0 {
            let error = std::io::Error::last_os_error();
            // ESRCH means the process is already gone (e.g. raced with an
            // external reaper) — not a failure for a kill-if-alive call.
            if error.raw_os_error() == Some(libc::ESRCH) {
                return Ok(());
            }
            return Err(ProximaError::Body(format!(
                "kill pid {}: {error}",
                self.pid
            )));
        }

        Ok(())
    }
}

/// Decode a `waitpid` status word into a shell-convention exit code.
fn exit_code(status: libc::c_int) -> i32 {
    if libc::WIFEXITED(status) {
        libc::WEXITSTATUS(status)
    } else if libc::WIFSIGNALED(status) {
        128 + libc::WTERMSIG(status)
    } else {
        -1
    }
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
    spawn_fork(command, options)
}

/// Spawn an ordinary external executable without entering the raw `fork(2)`
/// path. This is the safe route for callers that may already have runtime
/// threads. Options requiring child-side code before `exec` are rejected;
/// they must use [`spawn`] through a single-threaded boundary.
#[cfg(target_os = "linux")]
pub fn spawn_external(command: &CommandDescriptor, options: SpawnOptions) -> Result<Child, ProximaError> {
    if options.dispatch_fd.is_some() || options.controlling_tty || options.umask.is_some() {
        return Err(ProximaError::Body("posix_spawn external route does not support dispatch fd, controlling tty, or umask".to_owned()));
    }
    spawn_posix(command)
}

#[cfg(not(target_os = "linux"))]
pub fn spawn_external(_command: &CommandDescriptor, _options: SpawnOptions) -> Result<Child, ProximaError> {
    Err(ProximaError::Body("posix_spawn external route is unavailable on this platform".to_owned()))
}

#[cfg(target_os = "linux")]
fn spawn_posix(command: &CommandDescriptor) -> Result<Child, ProximaError> {
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
    let argv_pointers: Vec<*mut libc::c_char> = argv_storage
        .iter()
        .map(|value| value.as_ptr() as *mut libc::c_char)
        .chain(std::iter::once(ptr::null_mut()))
        .collect();
    let envp = build_envp(&command.env);
    let mut actions = unsafe { std::mem::zeroed::<libc::posix_spawn_file_actions_t>() };
    let initialized = unsafe { libc::posix_spawn_file_actions_init(&mut actions) };
    if initialized != 0 {
        return Err(posix_spawn_error("file-actions init", initialized));
    }

    let action_result = add_posix_stdio_actions(&mut actions, command, &piped_pairs);
    if let Err(error) = action_result {
        unsafe { libc::posix_spawn_file_actions_destroy(&mut actions) };
        return Err(error);
    }

    let mut pid = 0;
    let spawned = unsafe {
        libc::posix_spawnp(
            &mut pid,
            command.program.as_ptr(),
            &actions,
            ptr::null(),
            argv_pointers.as_ptr(),
            envp.pointers.as_ptr(),
        )
    };
    let destroyed = unsafe { libc::posix_spawn_file_actions_destroy(&mut actions) };
    if spawned != 0 {
        return Err(posix_spawn_error("spawn", spawned));
    }
    if destroyed != 0 {
        return Err(posix_spawn_error("file-actions destroy", destroyed));
    }

    Ok(parent_after_fork(pid, piped_pairs))
}

#[cfg(target_os = "linux")]
fn add_posix_stdio_actions(
    actions: &mut libc::posix_spawn_file_actions_t,
    command: &CommandDescriptor,
    piped_pairs: &[PipedPair],
) -> Result<(), ProximaError> {
    for slot in [Slot::Input, Slot::Output, Slot::Error] {
        let target = slot.child_fd();
        match slot_io(command, slot) {
            Stdio::Inherit => {}
            Stdio::Null => {
                let result = unsafe {
                    libc::posix_spawn_file_actions_addopen(actions, target, c"/dev/null".as_ptr(), libc::O_RDWR, 0)
                };
                if result != 0 {
                    return Err(posix_spawn_error("null stdio action", result));
                }
            }
            Stdio::Fd(source) => add_dup_action(actions, source, target)?,
            Stdio::Piped => {
                let pair = piped_pairs
                    .iter()
                    .find(|pair| pair.slot.child_fd() == target)
                    .ok_or_else(|| ProximaError::Body(format!("missing pipe for child fd {target}")))?;
                let source = pair.child_end.as_raw_fd();
                add_dup_action(actions, source, target)?;
                add_close_action(actions, pair.parent_end.as_raw_fd())?;
                if source != target {
                    add_close_action(actions, source)?;
                }
            }
        }
    }
    if let Some(directory) = command.current_dir.as_ref() {
        let result = unsafe { libc::posix_spawn_file_actions_addchdir_np(actions, directory.as_ptr()) };
        if result != 0 {
            return Err(posix_spawn_error("working-directory action", result));
        }
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn add_dup_action(actions: &mut libc::posix_spawn_file_actions_t, source: RawFd, target: libc::c_int) -> Result<(), ProximaError> {
    if source == target {
        return Ok(());
    }
    let result = unsafe { libc::posix_spawn_file_actions_adddup2(actions, source, target) };
    if result == 0 {
        Ok(())
    } else {
        Err(posix_spawn_error("dup stdio action", result))
    }
}

#[cfg(target_os = "linux")]
fn add_close_action(actions: &mut libc::posix_spawn_file_actions_t, fd: RawFd) -> Result<(), ProximaError> {
    let result = unsafe { libc::posix_spawn_file_actions_addclose(actions, fd) };
    if result == 0 {
        Ok(())
    } else {
        Err(posix_spawn_error("close stdio action", result))
    }
}

#[cfg(target_os = "linux")]
fn posix_spawn_error(operation: &str, code: libc::c_int) -> ProximaError {
    ProximaError::Body(format!("posix_spawn {operation} failed: {}", std::io::Error::from_raw_os_error(code)))
}

fn spawn_fork(command: &CommandDescriptor, options: SpawnOptions) -> Result<Child, ProximaError> {
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
        exit: None,
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
    use std::thread;

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

    /// Poll `try_wait` until it observes the reap, yielding between
    /// attempts instead of sleeping. Needed because a child's fds close
    /// (unblocking a pipe-EOF read) a moment before the kernel finishes
    /// the zombie transition that `waitpid` observes — the gap is real
    /// but not time-bounded, so we poll a live condition rather than
    /// guess a duration.
    fn poll_until_reaped(child: &mut Child) -> i32 {
        for _ in 0..100_000 {
            if let Some(code) = child.try_wait().expect("try_wait") {
                return code;
            }
            thread::yield_now();
        }
        panic!("child did not reach a reapable state after stdout EOF");
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

    #[cfg(target_os = "linux")]
    #[test]
    fn posix_spawn_supports_concurrent_external_children() {
        let workers: Vec<_> = (0..8)
            .map(|worker| {
                thread::spawn(move || {
                    let mut command = CommandDescriptor::new(cstr("/bin/echo"));
                    command.arg(cstr(&format!("worker {worker}"))).stdout(Stdio::Piped);
                    let mut spawned = spawn_external(&command, SpawnOptions::default()).expect("posix_spawn");
                    let output = drain_to_string(spawned.stdout.take().expect("piped output"));
                    assert_eq!(output, format!("worker {worker}\n"));
                    let status = wait_child(spawned.pid);
                    assert!(libc::WIFEXITED(status) && libc::WEXITSTATUS(status) == 0);
                })
            })
            .collect();
        for worker in workers {
            worker.join().expect("concurrent child worker");
        }
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
            CommandDescriptor::new(cstr("/usr/bin/true")).inherit_current_env(),
            SpawnOptions::default(),
        )
        .expect("spawn /bin/true");
        assert!(spawned.stdin.is_none());
        assert!(spawned.stdout.is_none());
        assert!(spawned.stderr.is_none());
        wait_child(spawned.pid);
    }

    #[test]
    fn try_wait_reports_running_then_exited() {
        let mut command = CommandDescriptor::new(cstr("/bin/sh"));
        command
            .arg(cstr("-c"))
            .arg(cstr("read line; exit 0"))
            .stdin(Stdio::Piped)
            .stdout(Stdio::Piped);

        let mut spawned = spawn(&command, SpawnOptions::default()).expect("spawn");
        assert_eq!(
            spawned.try_wait().expect("try_wait while running"),
            None,
            "child is blocked on read and has not been given any input yet"
        );

        let input_fd = spawned.stdin.take().expect("piped stdin");
        let raw = input_fd.into_raw_fd();
        let mut writer = unsafe { std::fs::File::from_raw_fd(raw) };
        writer.write_all(b"go\n").expect("write to shell stdin");
        drop(writer);

        let output_fd = spawned.stdout.take().expect("piped stdout");
        let _ = drain_to_string(output_fd); // blocks until the shell exits and closes stdout

        let exit_code = poll_until_reaped(&mut spawned);
        assert_eq!(exit_code, 0, "shell should exit 0 after reading its line");
    }

    #[test]
    fn kill_terminates_indefinitely_blocked_child_and_try_wait_reports_signal_death() {
        let mut command = CommandDescriptor::new(cstr("/bin/sh"));
        command
            .arg(cstr("-c"))
            .arg(cstr("read line"))
            .stdin(Stdio::Piped);

        let mut spawned = spawn(&command, SpawnOptions::default()).expect("spawn");
        assert_eq!(
            spawned.try_wait().expect("try_wait while running"),
            None,
            "child is blocked reading stdin that nothing ever writes to"
        );

        spawned.kill().expect("kill");
        let waited_code = spawned.wait().expect("wait after kill");
        assert_eq!(
            waited_code,
            128 + libc::SIGKILL,
            "SIGKILL death reported via shell 128+signal convention"
        );

        assert_eq!(
            spawned.try_wait().expect("try_wait after kill"),
            Some(waited_code),
            "try_wait must report the cached signal-death code once reaped"
        );
    }

    #[test]
    fn try_wait_after_exit_is_cached_and_idempotent() {
        let mut command = CommandDescriptor::new(cstr("/usr/bin/true"));
        command.stdout(Stdio::Piped);
        let mut spawned = spawn(&command, SpawnOptions::default()).expect("spawn");

        let output_fd = spawned.stdout.take().expect("piped stdout");
        let _ = drain_to_string(output_fd); // blocks until the child exits and closes stdout

        let first = poll_until_reaped(&mut spawned);
        let second = spawned
            .try_wait()
            .expect("second try_wait")
            .expect("second try_wait must still report Some after reap");
        assert_eq!(first, 0);
        assert_eq!(
            second, first,
            "second call must return the identical cached code, not re-issue waitpid"
        );
    }

    #[test]
    fn kill_after_reap_is_a_safe_no_op() {
        let mut command = CommandDescriptor::new(cstr("/usr/bin/true"));
        command.stdout(Stdio::Piped);
        let mut spawned = spawn(&command, SpawnOptions::default()).expect("spawn");

        let output_fd = spawned.stdout.take().expect("piped stdout");
        let _ = drain_to_string(output_fd);
        let reaped_code = poll_until_reaped(&mut spawned);

        spawned
            .kill()
            .expect("kill after reap must be Ok, never signal a recycled pid");

        assert_eq!(
            spawned.try_wait().expect("try_wait after no-op kill"),
            Some(reaped_code),
            "kill after reap must not disturb the cached exit code"
        );
    }
}
