//! `FdPairPipe` — the byte-transport `Pipe` primitive.
//!
//! A `(write_fd, read_fd)` pair IS a Pipe: bytes the caller writes
//! go into the write_fd, bytes the consumer reads come out of the
//! read_fd. This module makes that explicit. The same shape works
//! for kernel `pipe(2)` pairs, Unix-socket pairs, serial-device fds,
//! and anything else where one fd carries input and one carries
//! output.
//!
//! [`CommandPipe`] and [`ForkedPipe`] both produce an `FdPairPipe`
//! internally after they've set up the child process — the
//! subprocess-spawning and fork-server logic chooses HOW the bytes
//! get produced/consumed; once the fds exist, the Pipe shape is
//! uniform.
//!
//! # One-shot
//!
//! Each `FdPairPipe` is single-use. Calling `call()` consumes the
//! held fds (pumps and readers take ownership). A second call
//! returns an error. The outer Pipes (`CommandPipe`, `ForkedPipe`)
//! construct a fresh `FdPairPipe` per outer-call, which matches
//! "fresh child process per outer-call".
//!
//! [`CommandPipe`]: super::command_pipe::CommandPipe
//! [`ForkedPipe`]: super::fork_server::ForkedPipe

use std::future::Future;
use std::io::{Read, Write};
use std::os::fd::{FromRawFd, IntoRawFd, OwnedFd};
use std::sync::Mutex;
use std::thread;

use bytes::Bytes;
use futures::executor::block_on;
use futures::stream::{self, StreamExt};
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::{ChunkStream, ProximaError, Request, RequestStream, Response, ResponseStream};
use proxima_primitives::sync::mpsc;

/// Allocate a `pipe(2)` and return `(read_end, write_end)`.
pub(super) fn make_pipe() -> Result<(OwnedFd, OwnedFd), ProximaError> {
    let mut raw_fds: [libc::c_int; 2] = [0, 0];
    let result = unsafe { libc::pipe(raw_fds.as_mut_ptr()) };
    if result < 0 {
        return Err(ProximaError::Body(format!(
            "libc::pipe failed: {}",
            std::io::Error::last_os_error()
        )));
    }
    let read_fd = unsafe { OwnedFd::from_raw_fd(raw_fds[0]) };
    let write_fd = unsafe { OwnedFd::from_raw_fd(raw_fds[1]) };
    Ok((read_fd, write_fd))
}

/// A `Pipe` whose request body drains into `write_fd` and whose
/// response body streams bytes out of `read_fd`.
///
/// Single-shot: the first `call()` consumes the held fds. Subsequent
/// calls return an error. Construct fresh per round.
///
/// Behaviour during `call()`:
/// - A worker thread is spawned to drain `request.body` into
///   `write_fd`; when the request body ends, the thread closes the
///   write end, signalling EOF to whoever reads it.
/// - The returned `Response`'s body is backed by a channel fed by a
///   reader thread that reads `read_fd` until EOF.
/// - If `child_pid > 0`, the reader thread `waitpid`s the child once
///   read EOF arrives. Pass `0` for no reaping (e.g., when the
///   fork-server's `SIGCHLD` handler will reap auto).
pub struct FdPairPipe {
    state: Mutex<Option<FdPairState>>,
}

struct FdPairState {
    write_fd: OwnedFd,
    read_fd: OwnedFd,
    child_pid: libc::pid_t,
}

impl FdPairPipe {
    /// Wrap an fd pair without an associated child to reap.
    #[must_use]
    pub fn new(write_fd: OwnedFd, read_fd: OwnedFd) -> Self {
        Self::with_child(write_fd, read_fd, 0)
    }

    /// Wrap an fd pair AND a child pid that should be `waitpid`-reaped
    /// once the read end reaches EOF. Pass `0` for the pid to opt out
    /// of reaping (e.g., when an external SIGCHLD disposition is
    /// auto-reaping).
    #[must_use]
    pub fn with_child(write_fd: OwnedFd, read_fd: OwnedFd, child_pid: libc::pid_t) -> Self {
        Self {
            state: Mutex::new(Some(FdPairState {
                write_fd,
                read_fd,
                child_pid,
            })),
        }
    }
}

impl SendPipe for FdPairPipe {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let taken = self.state.lock().ok().and_then(|mut guard| guard.take());
        async move {
            let FdPairState {
                write_fd,
                read_fd,
                child_pid,
            } = taken.ok_or_else(|| {
                ProximaError::Body("FdPairPipe::call invoked after single use".into())
            })?;
            spawn_request_pump(write_fd, request);
            let body = build_response_body(read_fd, child_pid);
            Ok(Response::streamed(body))
        }
    }
}


/// Drain `request_body` into `write_end` on a worker thread.
/// Closing `write_end` (dropped on closure exit) signals EOF to
/// whatever reads the other end.
fn spawn_request_pump(write_end: OwnedFd, request: Request<Bytes>) {
    let raw_fd = write_end.into_raw_fd();
    thread::spawn(move || {
        let mut writer = unsafe { std::fs::File::from_raw_fd(raw_fd) };
        let mut stream = request.into_chunk_stream();
        block_on(async {
            while let Some(result) = stream.next().await {
                let Ok(bytes) = result else { break };
                if writer.write_all(&bytes).is_err() {
                    break;
                }
            }
        });
    });
}

/// Wrap a read-end fd as a [`RequestStream`] (no child pid to reap).
/// Used inside the fork-server's grandchild where its input fd needs
/// to become the inner Pipe's request body. Exposed at `pub(super)`
/// so the grandchild runner can compose the same primitive.
pub(super) fn body_from_read_fd(read_end: OwnedFd) -> RequestStream {
    RequestStream::from_chunk_stream(byte_stream_from_read_fd(read_end, 0))
}

/// Wrap `read_end` as a [`ResponseStream`] backed by an unbounded
/// channel that a worker thread feeds in 4 KiB chunks.
fn build_response_body(read_end: OwnedFd, child_pid: libc::pid_t) -> ResponseStream {
    ResponseStream::from_chunk_stream(byte_stream_from_read_fd(read_end, child_pid))
}

/// Build a [`ChunkStream`] from `read_end`, fed by a worker thread in
/// 4 KiB chunks. After read EOF the worker reaps `child_pid` if
/// non-zero, then drops the sender so the stream ends.
fn byte_stream_from_read_fd(read_end: OwnedFd, child_pid: libc::pid_t) -> ChunkStream {
    let raw_fd = read_end.into_raw_fd();
    let (sender, receiver) = mpsc::unbounded_channel::<Vec<u8>>();

    thread::spawn(move || {
        let mut reader = unsafe { std::fs::File::from_raw_fd(raw_fd) };
        let mut buffer = [0u8; 4096];
        loop {
            match reader.read(&mut buffer) {
                Ok(0) => break,
                Ok(bytes_read) => {
                    if sender.send(buffer[..bytes_read].to_vec()).is_err() {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
        if child_pid > 0 {
            let mut status: libc::c_int = 0;
            unsafe {
                libc::waitpid(child_pid, &mut status, 0);
            }
        }
    });

    Box::pin(stream::unfold(receiver, |mut rx| async move {
        rx.recv()
            .await
            .map(|chunk| (Ok::<Bytes, ProximaError>(Bytes::from(chunk)), rx))
    }))
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

    fn empty_request() -> Request<Bytes> {
        Request::builder()
            .method("POST")
            .path("/")
            .build()
            .expect("request builds")
    }

    #[test]
    fn fd_pair_round_trips_bytes_via_loopback() {
        // Loopback test: parent writes to write_fd, parent reads from
        // read_fd. With kernel pipes that's two separate pairs wired
        // back-to-back via a tee process — too much. Simpler: just
        // verify FdPairPipe::call shuttles bytes when we wire a
        // single pipe's write end to one input and read end to a
        // verification reader.
        //
        // Here: write_fd and read_fd are OPPOSITE ENDS of the same
        // pipe(2) pair. We drain a Body into write_fd; the reader
        // thread reads the same bytes from read_fd.
        let (read_fd, write_fd) = make_pipe().expect("pipe");
        let pipe = FdPairPipe::new(write_fd, read_fd);

        let payload = b"loopback bytes through FdPairPipe";
        let request = Request::builder()
            .method("POST")
            .path("/")
            .body(Bytes::copy_from_slice(payload))
            .build()
            .expect("request builds");

        let response = block_on(pipe.call(request)).expect("call");
        let output = drain(response);
        assert_eq!(output, payload);
    }

    #[test]
    fn second_call_returns_error() {
        let (read_fd, write_fd) = make_pipe().expect("pipe");
        let pipe = FdPairPipe::new(write_fd, read_fd);

        let _ = block_on(pipe.call(empty_request())).expect("first call");
        let second = block_on(pipe.call(empty_request()));
        assert!(second.is_err(), "second call should error");
    }
}
