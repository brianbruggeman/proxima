//! [`spawn_and_dispatch`] — spawn a [`CommandDescriptor`] with a typed
//! dispatch chain attached to `extra_fd[7]`.
//!
//! This is the C8d wiring: it composes [`super::spawn`],
//! [`super::ipc::run_dispatch_loop`], and any
//! [`Pipe`]-shaped dispatcher into one synchronous entry
//! point. The child is spawned with:
//!
//! - `extra_fd[7]` wired to the child end of a unix socketpair.
//! - `PROXIMA_DISPATCH_FD=7` set in the child env.
//!
//! While the child runs, a sibling thread drives the parent end
//! of the socketpair: it reads framed [`super::protocol::ChildRequest`]s,
//! dispatches each via the supplied pipe, writes framed
//! [`super::protocol::ChildResponse`]s back. When the child exits
//! (closing its end), the dispatch thread terminates on EOF.
//!
//! # Without the cdylib
//!
//! `spawn_and_dispatch` works even when no libc shim is loaded —
//! the dispatch fd is just an inert socket from the child's
//! perspective. The dispatch thread terminates cleanly on EOF
//! when the child exits. Test code can simulate a shim by
//! spawning a child that writes a frame explicitly (e.g. via
//! `printf` over fd 7).
//!
//! # With the cdylib (C8c)
//!
//! Once `proxima-process-shim` is loaded into the child via
//! `DYLD_INSERT_LIBRARIES`, libc hooks intercept `uname(2)` etc.
//! and write framed `ChildRequest`s to fd 7. The dispatch
//! thread receives them, routes via the configured pipe, and
//! returns synthesized responses. The child's libc surface is
//! honeypotted without any code change to `spawn_and_dispatch`.

use std::io;
use std::os::fd::{AsRawFd, IntoRawFd, OwnedFd};
use std::os::unix::net::UnixStream;
use std::thread;

use futures::executor::block_on;
use nix::sys::socket::{AddressFamily, SockFlag, SockType, socketpair};
use proxima_primitives::pipe::ProximaError;
use proxima_primitives::pipe::SendPipe;

use super::descriptor::CommandDescriptor;
use super::ipc::run_dispatch_loop;
use super::protocol::{ChildRequest, ChildResponse};
use super::spawn::{Child, SpawnOptions, spawn};

/// Standard fd number for the dispatch socket inside the child.
/// Matches the convention the cdylib shim reads from
/// `PROXIMA_DISPATCH_FD`.
pub const DISPATCH_FD: i32 = 7;

/// Environment variable name conveying the dispatch fd to the
/// child's libc shim. Hardcoded to a stable string.
pub const DISPATCH_FD_ENV: &str = "PROXIMA_DISPATCH_FD";

/// Spawn `command` with the dispatch chain `pipe` attached to
/// `extra_fd[DISPATCH_FD]`. The dispatch chain runs on a sibling
/// thread for the lifetime of the child. Returns the spawned
/// [`Child`] and the joinable dispatch-thread handle; call
/// [`DispatchedChild::wait`] to wait for both to terminate.
///
/// # Errors
///
/// - `ProximaError::Body` if the socketpair allocation fails.
/// - `ProximaError::Body` if `extra_fd[DISPATCH_FD]` is already
///   configured in the command (would clobber).
/// - Anything `spawn` itself returns.
///
/// # Panics
///
/// Does not panic in normal operation.
pub fn spawn_and_dispatch<P>(
    command: &CommandDescriptor,
    pipe: P,
) -> Result<DispatchedChild, ProximaError>
where
    P: SendPipe<In = ChildRequest, Out = ChildResponse, Err = ProximaError> + Send + Sync + 'static,
{
    spawn_and_dispatch_with_options(command, pipe, SpawnOptions::default())
}

/// Same shape as [`spawn_and_dispatch`] but carries through the
/// non-dispatch [`SpawnOptions`] knobs (controlling_tty, umask).
/// `options.dispatch_fd` is ignored — the dispatch fd this function
/// allocates always wins.
pub fn spawn_and_dispatch_with_options<P>(
    command: &CommandDescriptor,
    pipe: P,
    mut options: SpawnOptions,
) -> Result<DispatchedChild, ProximaError>
where
    P: SendPipe<In = ChildRequest, Out = ChildResponse, Err = ProximaError> + Send + Sync + 'static,
{
    let (parent_fd, child_fd) = socketpair(
        AddressFamily::Unix,
        SockType::Stream,
        None,
        SockFlag::empty(),
    )
    .map_err(|err| ProximaError::Body(format!("socketpair: {err}")))?;

    // Clone the command so we can stamp the env var without
    // mutating the caller's value. The dispatch fd itself is
    // threaded through `SpawnOptions::dispatch_fd` — NOT
    // through any field on `CommandDescriptor`, which stays pure data.
    let mut command = command.clone();
    command.env(
        std::ffi::CString::new(DISPATCH_FD_ENV)
            .map_err(|err| ProximaError::Body(format!("env name: {err}")))?,
        std::ffi::CString::new(DISPATCH_FD.to_string())
            .map_err(|err| ProximaError::Body(format!("env value: {err}")))?,
    );

    options.dispatch_fd = Some(child_fd.as_raw_fd());
    let child = spawn(&command, options)?;

    // Parent no longer needs the child end of the socketpair —
    // close it so the child's writes don't loop back to us.
    drop(child_fd);

    let parent_stream = owned_fd_to_unix_stream(parent_fd)?;
    let dispatch_thread = thread::spawn(move || dispatch_thread_body(parent_stream, pipe));

    Ok(DispatchedChild {
        child,
        dispatch_thread,
    })
}

/// Convert an `OwnedFd` (from nix) into a `std::os::unix::net::UnixStream`.
fn owned_fd_to_unix_stream(fd: OwnedFd) -> Result<UnixStream, ProximaError> {
    // SAFETY: nix's OwnedFd guarantees we own the raw fd; into_raw_fd
    // releases ownership; UnixStream::from_raw_fd reclaims it. Net
    // ownership transfer is exact, no double-close.
    let raw = fd.into_raw_fd();
    Ok(unsafe { <UnixStream as std::os::fd::FromRawFd>::from_raw_fd(raw) })
}

/// Body of the dispatch thread: drive `run_dispatch_loop` against
/// the parent socket end until EOF (child closed its end).
fn dispatch_thread_body<P>(stream: UnixStream, pipe: P) -> io::Result<()>
where
    P: SendPipe<In = ChildRequest, Out = ChildResponse, Err = ProximaError> + Sync,
{
    let mut read = stream.try_clone()?;
    let mut write = stream;
    run_dispatch_loop(&mut read, &mut write, |request| {
        block_on(SendPipe::call(&pipe, request))
    })
}

/// A spawned child with a sibling dispatch thread. Use
/// [`Self::wait`] to wait for both to terminate and reap the
/// child.
pub struct DispatchedChild {
    /// Underlying spawned child (pid + piped stdio fds).
    pub child: Child,
    /// Join handle for the dispatch thread. The thread terminates
    /// on the child's EOF; join after the child has exited.
    pub dispatch_thread: thread::JoinHandle<io::Result<()>>,
}

impl DispatchedChild {
    /// Wait for the child to exit, then join the dispatch thread.
    ///
    /// Returns the child's raw `wait(2)` status code. The dispatch
    /// thread's result is propagated via the returned error if it
    /// hit an I/O error mid-flight.
    ///
    /// # Errors
    ///
    /// - `ProximaError::Body` if `waitpid` fails or returns an
    ///   unexpected pid.
    /// - `ProximaError::Body` if the dispatch thread returned an
    ///   I/O error before EOF.
    pub fn wait(self) -> Result<libc::c_int, ProximaError> {
        let pid = self.child.pid;
        let mut status: libc::c_int = 0;
        // SAFETY: waitpid is a kernel call with a u32 pid and i32* status;
        // no Rust-level invariants are at risk.
        let waited = unsafe { libc::waitpid(pid, &mut status, 0) };
        if waited != pid {
            return Err(ProximaError::Body(format!(
                "waitpid returned {waited}, expected {pid}"
            )));
        }

        match self.dispatch_thread.join() {
            Ok(Ok(())) => Ok(status),
            Ok(Err(io_err)) => Err(ProximaError::Body(format!(
                "dispatch thread io error: {io_err}"
            ))),
            Err(_panic) => Err(ProximaError::Body("dispatch thread panicked".into())),
        }
    }
}
