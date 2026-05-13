//! `impl Pipe` that runs a [`CommandDescriptor`] inside a kernel PTY with a
//! typed dispatch chain attached.
//!
//! What you reach for when the child needs to see a TTY: shells,
//! editors, REPLs, agent CLIs that emit ANSI escape codes for
//! prompts / spinners / colors. With this Pipe the child's
//! `isatty(0/1/2)` returns true, `tcgetattr` succeeds, SIGWINCH on
//! resize is delivered, and the controlling-tty session is set up so
//! Ctrl-C reaches the child's process group.
//!
//! # Composition
//!
//! 1. [`open_pty`](super::pty::open_pty) — `libc::openpty` via nix.
//! 2. The descriptor's `input`/`output`/`error` slots are forced to
//!    [`Stdio::Fd(slave_raw)`](super::command::Stdio::Fd) and
//!    `controlling_tty = true`.
//! 3. [`spawn_and_dispatch`] — fork+exec with the dispatch chain
//!    wired to `extra_fd[7]` + `PROXIMA_DISPATCH_FD` in the child
//!    env. dup2s slave onto child fds 0/1/2, `setsid()`,
//!    `ioctl(TIOCSCTTY)`.
//! 4. Parent drops its slave reference.
//! 5. Spawn a **window-size follower** thread (if parent has a
//!    terminal): polls the parent's terminal size every ~200 ms,
//!    `TIOCSWINSZ`'s the master fd on change. The kernel PTY then
//!    delivers SIGWINCH to the child's controlling-tty session
//!    automatically.
//! 6. Wrap the master in [`FdPairPipe`](super::fd_pipe::FdPairPipe)
//!    for the byte shuttle. The response body owns BOTH the
//!    size-follower cancel handle AND the [`DispatchedChild`]; when
//!    the body finishes or is dropped, the follower exits cleanly
//!    and a background thread `wait`s the dispatched child (joins
//!    the dispatch thread + `waitpid`s the pid).
//!
//! # Required surface
//!
//! Construction is fluent and typestate-gated: `command`, `size`,
//! and `dispatch` must all be set before `.build()` is callable.
//! No `new`; for an un-shim'd TTY child, pass
//! [`super::grounds::Empty`] as the dispatch chain.
//!
//! # Why polling instead of signalfd / kqueue EVFILT_SIGNAL
//!
//! `signalfd` (Linux) requires process-wide signal masking; kqueue
//! EVFILT_SIGNAL (macOS) requires installing a signal disposition.
//! Both pull us into `unsafe` (extern statics, sigaction handlers,
//! global signal state). A 200 ms polling loop using
//! [`current_terminal_size`](super::pty::current_terminal_size) +
//! `nix::poll` for cancellation timing is entirely safe, has no
//! process-wide side effects, and the 200 ms latency on resize is
//! invisible to a human user.
use bytes::Bytes;

use std::future::Future;
use std::marker::PhantomData;
use std::os::fd::{AsFd, AsRawFd, OwnedFd};
use std::sync::Arc;
use std::thread;

use futures::stream::{self, StreamExt};
use nix::poll::{PollFd, PollFlags, PollTimeout, poll};
use nix::unistd::pipe;
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::{ProximaError, Request, Response, ResponseStream};

use super::descriptor::{CommandDescriptor, Stdio};
use super::dispatched::DispatchedChild;
use super::fd_pipe::FdPairPipe;
use super::protocol::{ChildRequest, ChildResponse};
use super::pty::{PtySize, current_terminal_size, open_pty, set_pty_size};

/// A `Pipe` whose `call(request)` runs `command` inside a kernel
/// PTY sized `size`. Request body bytes flow into the master fd;
/// the child's TTY stdout streams back as the response body.
/// Resize events on the parent's terminal are forwarded
/// automatically; the dispatch chain handles any framed
/// [`ChildRequest`]s the child emits on `PROXIMA_DISPATCH_FD`.
pub struct PtyCommandPipe<Chain> {
    command: CommandDescriptor,
    size: PtySize,
    chain: Chain,
}

impl PtyCommandPipe<NotBuiltYet> {
    /// Entry to the fluent builder. `.command(..)`, `.size(..)`,
    /// and `.dispatch(..)` must all be called before `.build()`
    /// will type-check.
    #[must_use]
    pub fn builder() -> PtyCommandPipeBuilder<Unset, Unset, Unset> {
        PtyCommandPipeBuilder {
            command: Unset,
            size: Unset,
            chain: Unset,
            _marker: PhantomData,
        }
    }
}

/// Phantom placeholder used purely to give `PtyCommandPipe::builder`
/// a stable receiver type. The real `PtyCommandPipe<C>` is
/// parameterised by the user-supplied chain type.
pub enum NotBuiltYet {}

/// Typestate marker — corresponding field has not been set on the
/// builder.
pub struct Unset;

/// Typestate marker — corresponding field has been set on the
/// builder. Carries the supplied value.
pub struct Set<T>(T);

/// Invariant typestate marker tying the builder to its three state
/// parameters without owning them.
type StateMarker<CommandState, SizeState, ChainState> =
    PhantomData<fn() -> (CommandState, SizeState, ChainState)>;

/// Fluent builder for [`PtyCommandPipe`]. The three type
/// parameters track which required fields have been provided.
/// `.build()` is only defined when all three are `Set<_>`, so
/// forgetting any one is a compile error.
pub struct PtyCommandPipeBuilder<CommandState, SizeState, ChainState> {
    command: CommandState,
    size: SizeState,
    chain: ChainState,
    _marker: StateMarker<CommandState, SizeState, ChainState>,
}

/// Lower any command shape proxima-process exposes to the
/// alloc-tier [`CommandDescriptor`] that the underlying spawn
/// primitive consumes. Implementations:
///
/// - `CommandDescriptor` — identity, infallible.
/// - [`super::command::Command`] — calls `to_descriptor()`,
///   fallible on NUL bytes (deferred error path matching std).
///
/// Add an impl here when a new high-level command surface lands;
/// existing `PtyCommandPipe` call sites pick it up for free.
pub trait IntoCommandDescriptor {
    /// Lower into a [`CommandDescriptor`]. `CommandDescriptor`'s
    /// own impl always returns `Ok`; the std-tier
    /// [`Command`](super::command::Command) impl can fail on NUL
    /// bytes in any field.
    fn into_command_descriptor(self) -> Result<CommandDescriptor, ProximaError>;
}

impl IntoCommandDescriptor for CommandDescriptor {
    fn into_command_descriptor(self) -> Result<CommandDescriptor, ProximaError> {
        Ok(self)
    }
}

impl IntoCommandDescriptor for super::command::Command {
    fn into_command_descriptor(self) -> Result<CommandDescriptor, ProximaError> {
        self.to_descriptor()
    }
}

impl<SizeState, ChainState> PtyCommandPipeBuilder<Unset, SizeState, ChainState> {
    /// Supply the command to spawn inside the PTY. Accepts either
    /// a [`CommandDescriptor`] (alloc-tier, infallible) or a
    /// [`super::command::Command`] (std-tier, fallible on NUL
    /// bytes). The fallibility is the cost of drop-in compat
    /// with `std::process::Command`-shape input.
    pub fn command<C: IntoCommandDescriptor>(
        self,
        command: C,
    ) -> Result<PtyCommandPipeBuilder<Set<CommandDescriptor>, SizeState, ChainState>, ProximaError>
    {
        Ok(PtyCommandPipeBuilder {
            command: Set(command.into_command_descriptor()?),
            size: self.size,
            chain: self.chain,
            _marker: PhantomData,
        })
    }
}

impl<CommandState, ChainState> PtyCommandPipeBuilder<CommandState, Unset, ChainState> {
    /// Supply the initial PTY window size. Subsequent resizes on
    /// the parent's terminal are tracked automatically.
    #[must_use]
    pub fn size(
        self,
        size: PtySize,
    ) -> PtyCommandPipeBuilder<CommandState, Set<PtySize>, ChainState> {
        PtyCommandPipeBuilder {
            command: self.command,
            size: Set(size),
            chain: self.chain,
            _marker: PhantomData,
        }
    }
}

impl<CommandState, SizeState> PtyCommandPipeBuilder<CommandState, SizeState, Unset> {
    /// Supply the dispatch chain. Any [`Pipe`] mapping
    /// [`ChildRequest`] → [`ChildResponse`] works; e.g.
    /// [`super::grounds::Empty`], [`super::grounds::Deny`], or
    /// a [`super::operators::AndThen`] composition.
    #[must_use]
    pub fn dispatch<Chain>(
        self,
        chain: Chain,
    ) -> PtyCommandPipeBuilder<CommandState, SizeState, Set<Chain>>
    where
        Chain: SendPipe<In = ChildRequest, Out = ChildResponse, Err = ProximaError>
            + Clone
            + Send
            + Sync
            + 'static,
    {
        PtyCommandPipeBuilder {
            command: self.command,
            size: self.size,
            chain: Set(chain),
            _marker: PhantomData,
        }
    }
}

impl<Chain> PtyCommandPipeBuilder<Set<CommandDescriptor>, Set<PtySize>, Set<Chain>>
where
    Chain: SendPipe<In = ChildRequest, Out = ChildResponse, Err = ProximaError>
        + Clone
        + Send
        + Sync
        + 'static,
{
    /// Finalise the builder. Only callable when `command`, `size`,
    /// and `dispatch` have all been set — a missing field is a
    /// compile error, not a runtime panic.
    #[must_use]
    pub fn build(self) -> PtyCommandPipe<Chain> {
        PtyCommandPipe {
            command: self.command.0,
            size: self.size.0,
            chain: self.chain.0,
        }
    }
}

impl<Chain> SendPipe for PtyCommandPipe<Chain>
where
    Chain: SendPipe<In = ChildRequest, Out = ChildResponse, Err = ProximaError>
        + Clone
        + Send
        + Sync
        + 'static,
{
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let mut command = self.command.clone();
        let size = self.size;
        let chain = self.chain.clone();
        async move {
            let (master, slave) = open_pty(size)?;
            let slave_raw = slave.as_raw_fd();

            command
                .stdin(Stdio::Fd(slave_raw))
                .stdout(Stdio::Fd(slave_raw))
                .stderr(Stdio::Fd(slave_raw));

            // PTY child must own the session — moved off
            // `CommandDescriptor` into SpawnOptions per the RISC
            // split (descriptor is pure data; runtime knobs ride
            // on SpawnOptions).
            let options = super::spawn::SpawnOptions {
                dispatch_fd: None,
                controlling_tty: true,
                umask: None,
            };
            let dispatched =
                super::dispatched::spawn_and_dispatch_with_options(&command, chain, options)?;
            drop(slave);

            let cancel_handle = if current_terminal_size().is_some() {
                let master_for_winch = master
                    .try_clone()
                    .map_err(|err| ProximaError::Body(format!("dup master for winch: {err}")))?;
                Some(spawn_size_follower(master_for_winch)?)
            } else {
                None
            };

            // FdPairPipe wants two fds (read + write). PTY master
            // is bidirectional; dup gives us a second OwnedFd
            // referencing the same kernel object.
            let master_for_writes = master
                .try_clone()
                .map_err(|err| ProximaError::Body(format!("dup master fd: {err}")))?;

            // child_pid=0: skip FdPairPipe's auto-reap;
            // DispatchedChild::wait (from the body wrapper) is the
            // single reap path.
            let inner_response = SendPipe::call(
                &FdPairPipe::with_child(master_for_writes, master, 0),
                request,
            )
            .await?;

            Ok(Response::streamed(wrap_body(
                inner_response,
                cancel_handle,
                dispatched,
            )))
        }
    }
}


/// Cancel handle for the size-follower thread. Dropping the handle
/// closes the cancel pipe's write end → the follower's poll wakes
/// on POLLHUP → follower exits cleanly.
struct SizeFollowerHandle {
    _cancel_write: OwnedFd,
}

fn spawn_size_follower(master: OwnedFd) -> Result<SizeFollowerHandle, ProximaError> {
    let (cancel_read, cancel_write) =
        pipe().map_err(|err| ProximaError::Body(format!("cancel pipe: {err}")))?;
    let master = Arc::new(master);
    let master_for_thread = Arc::clone(&master);
    thread::spawn(move || run_size_follower(master_for_thread, cancel_read));
    drop(master);
    Ok(SizeFollowerHandle {
        _cancel_write: cancel_write,
    })
}

fn run_size_follower(master: Arc<OwnedFd>, cancel_read: OwnedFd) {
    let mut last_size = current_terminal_size();
    loop {
        let mut fds = [PollFd::new(cancel_read.as_fd(), PollFlags::POLLIN)];
        let timeout = PollTimeout::from(200u16);
        match poll(&mut fds, timeout) {
            Ok(0) => {
                let current = current_terminal_size();
                if current != last_size {
                    if let Some(size) = current {
                        let _ = set_pty_size(master.as_fd(), size);
                    }
                    last_size = current;
                }
            }
            Ok(_) => {
                break;
            }
            Err(_) => break,
        }
    }
    drop(master);
}

// Combined body wrapper: when the stream finishes, drop the
// size-follower cancel handle (signals the follower to exit) and
// spawn a thread to `wait` the dispatched child (joins the
// dispatch thread + waitpids the pid). Done in a thread because
// `libc::waitpid` is blocking and can't run inside the executor.
fn wrap_body(
    response: Response<Bytes>,
    cancel: Option<SizeFollowerHandle>,
    dispatched: DispatchedChild,
) -> ResponseStream {
    let stream = response.into_chunk_stream();
    ResponseStream::new(stream::unfold(
        (stream, cancel, Some(dispatched)),
        |(mut stream, cancel, mut handle)| async move {
            match stream.next().await {
                Some(item) => Some((item, (stream, cancel, handle))),
                None => {
                    drop(cancel);
                    if let Some(dispatched) = handle.take() {
                        thread::spawn(move || {
                            let _ = dispatched.wait();
                        });
                    }
                    None
                }
            }
        },
    ))
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
    use crate::grounds::Empty;
    use bytes::Bytes;
    use futures::executor::block_on;
    use std::ffi::CString;

    fn cstr(text: &str) -> CString {
        CString::new(text).expect("test literal contains no interior NUL")
    }

    fn drain(response: Response<Bytes>) -> Vec<u8> {
        let mut stream = response.into_chunk_stream();
        let mut buffer = Vec::new();
        block_on(async {
            while let Some(chunk) = stream.next().await {
                buffer.extend_from_slice(&chunk.expect("body chunk"));
            }
        });
        buffer
    }

    #[test]
    fn pty_command_pipe_runs_tty_test() {
        let mut command = CommandDescriptor::new(cstr("/bin/sh"));
        command
            .arg(cstr("-c"))
            .arg(cstr(
                "if [ -t 0 ]; then printf isatty-true; else printf isatty-false; fi",
            ))
            .inherit_current_env();

        let pipe = PtyCommandPipe::builder()
            .command(command)
            .expect("lower descriptor")
            .size(PtySize::default())
            .dispatch(Empty)
            .build();
        let request = Request::builder()
            .method("POST")
            .path("/")
            .build()
            .expect("request builds");
        let response = block_on(SendPipe::call(&pipe, request)).expect("pty pipe call");
        let output = drain(response);
        let text = String::from_utf8_lossy(&output);
        assert!(
            text.contains("isatty-true"),
            "expected isatty-true in PTY child output, got: {text:?}"
        );
    }

    #[test]
    fn pty_command_pipe_carries_stdin_to_child() {
        let mut command = CommandDescriptor::new(cstr("/bin/cat"));
        command.inherit_current_env();
        let pipe = PtyCommandPipe::builder()
            .command(command)
            .expect("lower descriptor")
            .size(PtySize::default())
            .dispatch(Empty)
            .build();

        let payload = b"hello pty\n\x04";
        let request = Request::builder()
            .method("POST")
            .path("/")
            .body(Bytes::from_static(payload))
            .build()
            .expect("request builds");

        let response = block_on(SendPipe::call(&pipe, request)).expect("pty pipe call");
        let output = drain(response);
        let text = String::from_utf8_lossy(&output);
        assert!(
            text.contains("hello pty"),
            "expected 'hello pty' in PTY cat output, got: {text:?}"
        );
    }
}
