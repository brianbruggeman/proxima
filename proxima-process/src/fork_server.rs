//! Single-threaded fork-server that lets multi-threaded parents
//! spawn forked `Pipe`s safely.
//!
//! # Why this exists
//!
//! `fork(2)` in a multi-threaded process is hostile by design. After
//! fork, only the calling thread survives in the child; all other
//! parent threads vanish, but the mutexes they were holding stay
//! "locked" in the child's memory with no thread to release them.
//! Any code path in the child that touches those mutexes deadlocks.
//! `malloc`, the standard library's panic machinery, anything with a
//! global registry — all hazards.
//!
//! Workarounds like "do the fork on a fresh thread" do NOT fix this.
//! `fork` copies the whole process state; the thread that issues the
//! syscall doesn't matter.
//!
//! The standard fix (Chromium's zygote, Python's multiprocessing
//! `forkserver` start method): at process startup, BEFORE any other
//! threads exist, fork off a single-threaded helper child. The
//! helper enters an event loop on a Unix socket. The main process
//! can then spawn runtime threads freely. When the main process
//! wants to fork-and-run a `Pipe`, it sends a request over the
//! socket; the helper does the fork (single-threaded → safe) and
//! returns the grandchild's I/O file descriptors via `SCM_RIGHTS`.
//!
//! # API
//!
//! ```ignore
//! use std::sync::Arc;
//! use proxima_primitives::pipe::Pipe;
//! use pty_tester::proxima_process::ForkServer;
//!
//! // At process startup (before other threads):
//! let server = ForkServer::builder()
//!     .register("echo", || EchoPipe)
//!     .register("uppercase", || UppercasePipe)
//!     .start()?;
//!
//! // Later, from any thread:
//! let echo: impl Pipe = server.pipe("echo");
//! let response = echo.call(request).await?;
//! ```
//!
//! Each `pipe.call(...)` invocation forks a fresh grandchild. The
//! grandchild runs the registered factory's `Pipe::call`, with
//! streaming bytes through `pipe(2)` channels established by the
//! fork-server.
//!
//! # Constraints
//!
//! - **Register before start.** Factories registered after `start()`
//!   are not visible to the server child (different process, frozen
//!   memory snapshot from boot).
//! - **Factories must not capture state that's modified after boot.**
//!   The server has a snapshot from when `start()` was called; any
//!   later mutations in the parent don't propagate.
//! - **Pipe identifier is a string name.** The closure type can't
//!   cross the boot-time fork as a runtime value; only the binary's
//!   compiled code does.
//!
//! # Why no conflaguration `Settings` here
//!
//! Per [guiding-principles.md][gp] principle 4, std-tier composition
//! types ought to expose a `conflaguration::Settings`-derived config
//! struct + a `bon::Builder` fluent surface, with a parity test.
//! `ForkServer` is std-tier but its only meaningful "config" — the
//! registered factory closures — is **code**, not data: closures
//! don't deserialize from env vars or TOML. The fluent surface
//! ([`ForkServerBuilder`]) already exists; a parallel `Settings`
//! struct would have nothing to put in it.
//!
//! Per the same principle's caveat: "If the type has zero or one
//! configurable parameter, a free function or constructor is enough
//! — document the reason." This is that documented reason.
//!
//! [gp]: ../../../docs/proxima-pty/guiding-principles.md
use bytes::Bytes;

use std::collections::HashMap;
use std::future::Future;
use std::io::Write;
use std::os::fd::{AsRawFd, FromRawFd, IntoRawFd, OwnedFd, RawFd};
use std::os::unix::net::UnixStream;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use futures::executor::block_on;
use futures::stream::StreamExt;
use nix::cmsg_space;
use nix::sys::socket::{ControlMessage, ControlMessageOwned, MsgFlags, recvmsg, sendmsg};
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::{ProximaError, Request, Response};

use super::fd_pipe::{FdPairPipe, make_pipe};

const STATUS_OK: u8 = 0;
const STATUS_UNKNOWN_PIPE: u8 = 1;
const STATUS_FORK_FAILED: u8 = 2;
const STATUS_PIPE_FAILED: u8 = 3;

/// Boxed-future projection of `Pipe`. Object-safe so we can store
/// heterogeneous `Pipe` factories in a `HashMap`. The returned
/// future borrows from `&self` so the lifetime parameter is needed.
trait DynPipe: Send + Sync + 'static {
    fn dyn_call<'pipe>(
        &'pipe self,
        request: Request<Bytes>,
    ) -> Pin<Box<dyn Future<Output = Result<Response<Bytes>, ProximaError>> + Send + 'pipe>>;
}

impl<P> DynPipe for P
where
    P: SendPipe<In = Request<Bytes>, Out = Response<Bytes>, Err = ProximaError>,
{
    fn dyn_call<'pipe>(
        &'pipe self,
        request: Request<Bytes>,
    ) -> Pin<Box<dyn Future<Output = Result<Response<Bytes>, ProximaError>> + Send + 'pipe>> {
        Box::pin(SendPipe::call(self, request))
    }
}

type PipeFactory = Box<dyn Fn() -> Box<dyn DynPipe> + Send + Sync + 'static>;

/// Builds a [`ForkServer`]. Register factories, then `start()`.
pub struct ForkServerBuilder {
    factories: HashMap<String, PipeFactory>,
}

impl ForkServerBuilder {
    #[must_use]
    pub fn new() -> Self {
        Self {
            factories: HashMap::new(),
        }
    }

    /// Register `factory` under `name`. The factory produces a fresh
    /// `Pipe` each time the server forks for that name.
    pub fn register<F, P>(mut self, name: &str, factory: F) -> Self
    where
        F: Fn() -> P + Send + Sync + 'static,
        P: SendPipe<In = Request<Bytes>, Out = Response<Bytes>, Err = ProximaError>,
    {
        self.factories.insert(
            name.to_string(),
            Box::new(move || Box::new(factory()) as Box<dyn DynPipe>),
        );
        self
    }

    /// Fork off the single-threaded server child. **Call this before
    /// the process spawns any other threads.**
    pub fn start(self) -> Result<ForkServer, ProximaError> {
        let (parent_sock, server_sock) =
            UnixStream::pair().map_err(|err| ProximaError::Body(format!("socketpair: {err}")))?;

        let pid = unsafe { libc::fork() };
        if pid < 0 {
            return Err(ProximaError::Body(format!(
                "fork-server boot fork failed: {}",
                std::io::Error::last_os_error()
            )));
        }

        if pid == 0 {
            drop(parent_sock);
            run_server_loop(server_sock, self.factories);
        }

        drop(server_sock);
        Ok(ForkServer {
            inner: Arc::new(ForkServerInner {
                socket: Mutex::new(parent_sock),
                server_pid: pid,
            }),
        })
    }
}

impl Default for ForkServerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

struct ForkServerInner {
    socket: Mutex<UnixStream>,
    server_pid: libc::pid_t,
}

/// Handle to the fork-server child. `pipe(name)` returns an
/// `impl Pipe` whose every `call` forks a fresh grandchild via the
/// server.
#[derive(Clone)]
pub struct ForkServer {
    inner: Arc<ForkServerInner>,
}

impl ForkServer {
    #[must_use]
    pub fn builder() -> ForkServerBuilder {
        ForkServerBuilder::new()
    }

    /// Returns a `Pipe` whose every call invokes the factory
    /// registered as `name`, in a fresh forked grandchild.
    #[must_use]
    pub fn pipe(&self, name: &str) -> ForkedPipe {
        ForkedPipe {
            server: Arc::clone(&self.inner),
            name: name.to_string(),
        }
    }
}

impl Drop for ForkServerInner {
    fn drop(&mut self) {
        // Dropping the socket gives the server EOF on its read; its
        // event loop exits cleanly. Reap the server pid so we don't
        // leak a zombie.
        let mut status: libc::c_int = 0;
        unsafe {
            libc::kill(self.server_pid, libc::SIGTERM);
            libc::waitpid(self.server_pid, &mut status, 0);
        }
    }
}

/// A `Pipe` backed by the fork-server. Each `call` forks fresh.
pub struct ForkedPipe {
    server: Arc<ForkServerInner>,
    name: String,
}

impl SendPipe for ForkedPipe {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let server = Arc::clone(&self.server);
        let name = self.name.clone();
        async move {
            let (write_fd, read_fd) = fork_via_server(&server, &name)?;
            // Grandchildren are auto-reaped via SA_NOCLDWAIT in the
            // server, so no child_pid is supplied here.
            SendPipe::call(&FdPairPipe::new(write_fd, read_fd), request).await
        }
    }
}


fn fork_via_server(
    server: &ForkServerInner,
    name: &str,
) -> Result<(OwnedFd, OwnedFd), ProximaError> {
    let mut socket = server
        .socket
        .lock()
        .map_err(|_| ProximaError::Body("fork-server socket lock poisoned".into()))?;

    send_request(&mut socket, name)?;
    let (status, fds) = receive_response(&socket)?;

    match status {
        STATUS_OK => {
            if fds.len() != 2 {
                return Err(ProximaError::Body(format!(
                    "fork-server returned {} fds; expected 2",
                    fds.len()
                )));
            }
            let write_fd = unsafe { OwnedFd::from_raw_fd(fds[0]) };
            let read_fd = unsafe { OwnedFd::from_raw_fd(fds[1]) };
            Ok((write_fd, read_fd))
        }
        STATUS_UNKNOWN_PIPE => Err(ProximaError::Body(format!(
            "fork-server has no Pipe registered as {name:?}"
        ))),
        STATUS_FORK_FAILED => Err(ProximaError::Body(
            "fork-server: grandchild fork failed".into(),
        )),
        STATUS_PIPE_FAILED => Err(ProximaError::Body(
            "fork-server: pipe(2) allocation failed".into(),
        )),
        other => Err(ProximaError::Body(format!(
            "fork-server: unknown status byte {other}"
        ))),
    }
}

fn send_request(socket: &mut UnixStream, name: &str) -> Result<(), ProximaError> {
    let bytes = name.as_bytes();
    let length =
        u32::try_from(bytes.len()).map_err(|_| ProximaError::Body("pipe name too long".into()))?;
    socket
        .write_all(&length.to_le_bytes())
        .map_err(|err| ProximaError::Body(format!("fork-server write len: {err}")))?;
    socket
        .write_all(bytes)
        .map_err(|err| ProximaError::Body(format!("fork-server write name: {err}")))?;
    socket
        .flush()
        .map_err(|err| ProximaError::Body(format!("fork-server flush: {err}")))?;
    Ok(())
}

fn receive_response(socket: &UnixStream) -> Result<(u8, Vec<RawFd>), ProximaError> {
    let mut status_buffer = [0u8; 1];
    let mut io_slices = [std::io::IoSliceMut::new(&mut status_buffer)];
    let mut cmsg_buffer = cmsg_space!([RawFd; 2]);

    let message = recvmsg::<()>(
        socket.as_raw_fd(),
        &mut io_slices,
        Some(&mut cmsg_buffer),
        MsgFlags::empty(),
    )
    .map_err(|err| ProximaError::Body(format!("fork-server recvmsg: {err}")))?;

    let mut received_fds: Vec<RawFd> = Vec::new();
    for control_message in message
        .cmsgs()
        .map_err(|err| ProximaError::Body(format!("fork-server cmsgs: {err}")))?
    {
        if let ControlMessageOwned::ScmRights(fds) = control_message {
            received_fds.extend(fds);
        }
    }

    Ok((status_buffer[0], received_fds))
}

fn run_server_loop(server_socket: UnixStream, factories: HashMap<String, PipeFactory>) -> ! {
    // Auto-reap grandchildren so they don't accumulate as zombies.
    unsafe {
        let mut action: libc::sigaction = std::mem::zeroed();
        action.sa_sigaction = libc::SIG_IGN;
        action.sa_flags = libc::SA_NOCLDWAIT;
        libc::sigaction(libc::SIGCHLD, &action, std::ptr::null_mut());
    }

    let mut socket = server_socket;
    loop {
        let name = match read_request(&mut socket) {
            Ok(Some(name)) => name,
            Ok(None) => break,
            Err(_) => break,
        };

        match factories.get(&name) {
            None => {
                let _ = send_response(&socket, STATUS_UNKNOWN_PIPE, &[]);
            }
            Some(factory) => match make_pipe_pair() {
                Err(_) => {
                    let _ = send_response(&socket, STATUS_PIPE_FAILED, &[]);
                }
                Ok((
                    (child_input_read, parent_input_write),
                    (parent_output_read, child_output_write),
                )) => {
                    let grandchild_pid = unsafe { libc::fork() };
                    if grandchild_pid < 0 {
                        let _ = send_response(&socket, STATUS_FORK_FAILED, &[]);
                        continue;
                    }
                    if grandchild_pid == 0 {
                        drop(parent_input_write);
                        drop(parent_output_read);
                        let pipe = factory();
                        let exit_code = run_grandchild(pipe, child_input_read, child_output_write);
                        unsafe { libc::_exit(exit_code) };
                    }
                    drop(child_input_read);
                    drop(child_output_write);
                    let fds = [
                        parent_input_write.as_raw_fd(),
                        parent_output_read.as_raw_fd(),
                    ];
                    let _ = send_response(&socket, STATUS_OK, &fds);
                    drop(parent_input_write);
                    drop(parent_output_read);
                }
            },
        }
    }
    unsafe { libc::_exit(0) };
}

fn read_request(socket: &mut UnixStream) -> Result<Option<String>, std::io::Error> {
    use std::io::Read;
    let mut length_buffer = [0u8; 4];
    if let Err(err) = socket.read_exact(&mut length_buffer) {
        if err.kind() == std::io::ErrorKind::UnexpectedEof {
            return Ok(None);
        }
        return Err(err);
    }
    let length = u32::from_le_bytes(length_buffer) as usize;
    let mut name_bytes = vec![0u8; length];
    socket.read_exact(&mut name_bytes)?;
    String::from_utf8(name_bytes)
        .map(Some)
        .map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidData, err))
}

fn send_response(socket: &UnixStream, status: u8, fds_to_send: &[RawFd]) -> Result<(), nix::Error> {
    let status_buffer = [status];
    let io_slices = [std::io::IoSlice::new(&status_buffer)];
    let cmsgs: Vec<ControlMessage> = if fds_to_send.is_empty() {
        Vec::new()
    } else {
        vec![ControlMessage::ScmRights(fds_to_send)]
    };
    sendmsg::<()>(
        socket.as_raw_fd(),
        &io_slices,
        &cmsgs,
        MsgFlags::empty(),
        None,
    )
    .map(|_| ())
}

#[allow(clippy::field_reassign_with_default, clippy::type_complexity)]
fn make_pipe_pair() -> Result<((OwnedFd, OwnedFd), (OwnedFd, OwnedFd)), ProximaError> {
    let input = make_pipe()?;
    let output = make_pipe()?;
    Ok((input, output))
}

fn run_grandchild(
    pipe: Box<dyn DynPipe>,
    input_read: OwnedFd,
    output_write: OwnedFd,
) -> libc::c_int {
    // The grandchild reads its stdin from input_read and writes its
    // stdout to output_write. We build a streaming request body from
    // input_read, drive the inner Pipe with block_on, then drain the
    // response body chunks to output_write. Wrapping these fds with
    // an FdPairPipe and re-using its plumbing would deadlock here:
    // the request body needs to FEED inner.call, not be the output
    // of it. So we use the same primitives FdPairPipe uses
    // (request-pump and response-builder), but in the inverse
    // direction.
    let input_fd_raw = input_read.into_raw_fd();
    let output_fd_raw = output_write.into_raw_fd();

    let result: Result<(), ProximaError> = (|| {
        let input_body =
            super::fd_pipe::body_from_read_fd(unsafe { OwnedFd::from_raw_fd(input_fd_raw) });

        let request = Request::builder()
            .method("POST")
            .path("/")
            .stream(input_body)
            .build()
            .map_err(|err| ProximaError::Body(err.to_string()))?;

        let response = block_on(pipe.dyn_call(request))?;

        let mut stream = response.into_chunk_stream();
        let mut writer = unsafe { std::fs::File::from_raw_fd(output_fd_raw) };
        block_on(async {
            while let Some(chunk) = stream.next().await {
                let Ok(bytes) = chunk else { return };
                if writer.write_all(&bytes).is_err() {
                    return;
                }
            }
        });
        drop(writer);

        Ok(())
    })();

    if result.is_ok() { 0 } else { 1 }
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
    use std::thread;
    use std::time::Duration;

    #[derive(Clone)]
    struct EchoPipe;
    impl SendPipe for EchoPipe {
        type In = Request<Bytes>;
        type Out = Response<Bytes>;
        type Err = ProximaError;

        async fn call(&self, request: Request<Bytes>) -> Result<Response<Bytes>, ProximaError> {
            let (_, bytes) = request.body_bytes().await?;
            Ok(Response::ok(bytes))
        }
    }

    #[derive(Clone)]
    struct UppercasePipe;
    impl SendPipe for UppercasePipe {
        type In = Request<Bytes>;
        type Out = Response<Bytes>;
        type Err = ProximaError;

        async fn call(&self, request: Request<Bytes>) -> Result<Response<Bytes>, ProximaError> {
            let mut stream = request.into_chunk_stream();
            let mut output = Vec::new();
            while let Some(chunk) = stream.next().await {
                let bytes = chunk?;
                for byte in bytes.iter() {
                    output.push(byte.to_ascii_uppercase());
                }
            }
            Ok(Response::ok(Bytes::from(output)))
        }
    }

    fn drain(response: Response<Bytes>) -> Vec<u8> {
        let mut stream = response.into_chunk_stream();
        let mut buffer = Vec::new();
        block_on(async {
            while let Some(chunk) = stream.next().await {
                buffer.extend_from_slice(&chunk.expect("body chunk ok"));
            }
        });
        buffer
    }

    fn make_request(payload: &[u8]) -> Request<Bytes> {
        Request::builder()
            .method("POST")
            .path("/")
            .body(Bytes::copy_from_slice(payload))
            .build()
            .expect("request builds")
    }

    #[test]
    fn echo_round_trip_single_threaded() {
        let server = ForkServer::builder()
            .register("echo", || EchoPipe)
            .start()
            .expect("server starts");
        let pipe = server.pipe("echo");

        let output = drain(block_on(pipe.call(make_request(b"hello fork-server"))).expect("call"));
        assert_eq!(output, b"hello fork-server");
    }

    #[test]
    fn uppercase_round_trip() {
        let server = ForkServer::builder()
            .register("uppercase", || UppercasePipe)
            .start()
            .expect("server starts");
        let pipe = server.pipe("uppercase");

        let output = drain(block_on(pipe.call(make_request(b"fork-server lives"))).expect("call"));
        assert_eq!(output, b"FORK-SERVER LIVES");
    }

    #[test]
    fn unknown_pipe_name_errors() {
        let server = ForkServer::builder()
            .register("echo", || EchoPipe)
            .start()
            .expect("server starts");
        let pipe = server.pipe("nope");
        let result = block_on(pipe.call(make_request(b"")));
        assert!(result.is_err(), "expected error for unknown pipe");
    }

    #[test]
    fn multi_threaded_parent_concurrent_forks() {
        // The headline test: parent has many threads at fork time,
        // each issues a fork via the server, all succeed.
        let server = ForkServer::builder()
            .register("echo", || EchoPipe)
            .register("uppercase", || UppercasePipe)
            .start()
            .expect("server starts");

        // Burn some threads so the parent is genuinely multi-threaded
        // when ForkedPipe::call runs.
        let burners: Vec<_> = (0..4)
            .map(|_| {
                thread::spawn(|| {
                    thread::sleep(Duration::from_millis(200));
                })
            })
            .collect();

        let mut workers = Vec::new();
        for index in 0..8 {
            let server = server.clone();
            workers.push(thread::spawn(move || {
                let pipe_name = if index % 2 == 0 { "echo" } else { "uppercase" };
                let payload = format!("thread-{index}");
                let expected: Vec<u8> = if pipe_name == "uppercase" {
                    payload.to_ascii_uppercase().into_bytes()
                } else {
                    payload.clone().into_bytes()
                };

                let pipe = server.pipe(pipe_name);
                let output =
                    drain(block_on(pipe.call(make_request(payload.as_bytes()))).expect("call"));
                assert_eq!(output, expected, "thread {index} mismatch");
            }));
        }

        for handle in workers {
            handle.join().expect("worker join");
        }
        for handle in burners {
            handle.join().expect("burner join");
        }
    }
}
