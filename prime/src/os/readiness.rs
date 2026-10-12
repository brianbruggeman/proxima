//! Read-readiness for an externally-owned source over the per-core reactor.
//! On Windows `RawSource` is a Winsock socket, never a filesystem `HANDLE`.
//! File operations belong on the background pool; they cannot be registered here.
//!
//! Some callers own a raw fd whose readiness the kernel signals directly
//! (`POLLIN`) but that isn't itself a proxima `TcpStream`/`TcpListener` — an
//! AF_XDP socket, a signalfd, any other single-fd source. Rather than
//! busy-poll (self-waking the task every turn), [`Readiness`] registers the
//! fd on the worker's reactor for read interest and parks until the reactor
//! fires. This reuses prime's existing epoll/kqueue source (via
//! [`with_current_reactor`]) — it does not invent a new source kind, and it
//! is generic over any `RawSource`, not tied to a particular caller.
//!
//! prime's reactor is edge-triggered (`EPOLLET` on Linux), so the caller
//! MUST fully drain its own readable state before arming; otherwise a
//! residual descriptor produces no fresh edge and the wake is lost.
//! [`Readiness::poll`] returns [`ReadyState::Retry`] on a fresh readiness
//! epoch (caller re-drains) and [`ReadyState::Parked`] once armed. Off a
//! proxima worker (no reactor) it returns [`ReadyState::OffWorker`] so the
//! caller falls back to busy-poll — which keeps plain-`block_on` callers
//! working unchanged.

#[cfg(unix)]
use super::reactor::RawSource;
use std::io;
#[cfg(target_os = "macos")]
use std::num::NonZeroI32;
#[cfg(unix)]
use std::os::fd::{AsRawFd, FromRawFd};
#[cfg(windows)]
use std::os::windows::io::AsSocket;
use std::task::Context;

use super::core_shard::with_current_reactor;
use super::reactor::{Interest, OwnedSource, Reactor, SourceKey, Wakeup};

/// Outcome of arming read-readiness for a registered fd.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadyState {
    /// A fresh readiness edge is available — the caller should re-drain
    /// whatever it reads from the fd before parking again.
    Retry,
    /// Armed and waiting; the reactor will wake the task on the next
    /// `POLLIN`.
    Parked,
    /// Not on a proxima worker thread (no reactor) — the caller falls back
    /// to busy-poll (`cx.waker().wake_by_ref()` + `Pending`).
    OffWorker,
}

/// What the reactor is asked to watch. `Spent` is the state after `Drop` moved
/// the registration out.
enum Registration {
    #[cfg(unix)]
    Descriptor(RawSource),
    Owned(OwnedSource),
    #[cfg(target_os = "macos")]
    Process(NonZeroI32),
    Spent,
}

/// Reactor registration for one externally-owned fd. Deregisters on drop.
pub struct Readiness {
    registration: Registration,
    interest: Interest,
    owner: Option<Wakeup>,
    source: Option<SourceKey>,
    last_blocked_epoch: Option<u32>,
}

impl Readiness {
    #[cfg(unix)]
    #[must_use]
    pub fn new(fd: RawSource) -> Self {
        Self::with_registration(Registration::Descriptor(fd), Interest::Read)
    }

    /// Gate that opens when `fd` can accept a write (`POLLOUT`/`EVFILT_WRITE`).
    ///
    /// This is the write-side twin of [`Readiness::new`], over the same
    /// reactor source and the same epoch handshake: [`Readiness::poll`] arms the
    /// reactor's write waker (`Reactor::register_write_waker_ref`) and compares
    /// `Reactor::write_ready_epoch` where [`Readiness::new`] uses the read pair
    /// (`register_read_waker_ref` / `read_ready_epoch`). The composition is a
    /// sink pipe: do one nonblocking write, and on `EAGAIN` poll this gate;
    /// [`ReadyState::Retry`] means write again, [`ReadyState::Parked`] means
    /// return `Pending`. Edge-triggered like the read gate, so attempt the write
    /// until `EAGAIN` before arming.
    #[cfg(unix)]
    #[must_use]
    pub fn writable(fd: RawSource) -> Self {
        Self::with_registration(Registration::Descriptor(fd), Interest::Write)
    }

    /// Gate that opens when the child `pid` exits (EVFILT_PROC/NOTE_EXIT on the
    /// worker's own kqueue). A pid already past its knote fails the first
    /// `poll` with the raw `ESRCH`; the caller then reaps it immediately with
    /// [`child::reap_reported`](super::child::reap_reported). Linux gates a
    /// pidfd through [`Readiness::new`] instead.
    #[cfg(target_os = "macos")]
    #[must_use]
    pub fn process_exit(pid: NonZeroI32) -> Self {
        Self::with_registration(Registration::Process(pid), Interest::Read)
    }

    #[cfg(windows)]
    pub fn new(socket: &impl AsSocket) -> io::Result<Self> {
        Ok(Self::with_registration(
            Registration::Owned(socket.as_socket().try_clone_to_owned()?),
            Interest::Read,
        ))
    }

    /// Gate that opens when the socket can accept a write; the write-side twin
    /// of [`Readiness::new`], armed through the same reactor `rearm` path. A
    /// sink pipe does one nonblocking write, and on `WouldBlock` polls this gate.
    #[cfg(windows)]
    pub fn writable(socket: &impl AsSocket) -> io::Result<Self> {
        Ok(Self::with_registration(
            Registration::Owned(socket.as_socket().try_clone_to_owned()?),
            Interest::Write,
        ))
    }

    fn with_registration(registration: Registration, interest: Interest) -> Self {
        Self {
            registration,
            interest,
            owner: None,
            source: None,
            last_blocked_epoch: None,
        }
    }

    /// Arm read-readiness for the fd against the current worker's reactor.
    ///
    /// # Errors
    /// Returns an [`io::Error`] if the epoll/kqueue registration fails or
    /// the reactor source went stale.
    pub fn poll(&mut self, context: &Context<'_>) -> io::Result<ReadyState> {
        let outcome = with_current_reactor(|reactor| {
            if self
                .owner
                .as_ref()
                .is_some_and(|owner| !reactor.owns(owner))
            {
                return Err(io::Error::other(
                    "readiness moved to a different prime worker",
                ));
            }
            #[cfg(unix)]
            if let Registration::Descriptor(fd) = self.registration {
                // safety: fcntl validates the descriptor; the duplicate belongs to this readiness.
                let duplicate = unsafe { libc::fcntl(fd, libc::F_DUPFD_CLOEXEC, 0) };
                if duplicate < 0 {
                    return Err(io::Error::last_os_error());
                }
                self.registration = Registration::Owned(unsafe { OwnedSource::from_raw_fd(duplicate) });
            }

            let key = match self.source {
                Some(existing) => existing,
                None => {
                    let registered = self.register(reactor)?;
                    self.source = Some(registered);
                    self.owner = Some(reactor.wakeup());
                    registered
                }
            };
            #[cfg(windows)]
            reactor.rearm(key, self.interest)?;
            let armed = match self.interest {
                Interest::Write => reactor.register_write_waker_ref(key, context.waker()),
                Interest::Read | Interest::ReadWrite => {
                    reactor.register_read_waker_ref(key, context.waker())
                }
            };
            if !armed {
                return Err(stale_registration());
            }
            let epoch = match self.interest {
                Interest::Write => reactor.write_ready_epoch(key),
                Interest::Read | Interest::ReadWrite => reactor.read_ready_epoch(key),
            };
            epoch.ok_or_else(stale_registration)
        });
        let Some(epoch_result) = outcome else {
            if self.owner.is_some() {
                return Err(io::Error::other(
                    "registered readiness used off its owning worker",
                ));
            }
            return Ok(ReadyState::OffWorker);
        };
        let epoch = epoch_result?;
        match self.last_blocked_epoch {
            Some(previous) if previous == epoch => Ok(ReadyState::Parked),
            _ => {
                self.last_blocked_epoch = Some(epoch);
                Ok(ReadyState::Retry)
            }
        }
    }

    fn register(&self, reactor: &mut Reactor) -> io::Result<SourceKey> {
        match &self.registration {
            Registration::Owned(socket) => {
                #[cfg(unix)]
                let source = socket.as_raw_fd();
                #[cfg(windows)]
                let source = socket;
                reactor.register(source, self.interest)
            }
            #[cfg(target_os = "macos")]
            Registration::Process(pid) => reactor.register_process_exit(*pid),
            #[cfg(unix)]
            Registration::Descriptor(_) => Err(io::Error::other("readiness source not yet owned")),
            Registration::Spent => Err(io::Error::other("readiness source missing")),
        }
    }
}

impl Drop for Readiness {
    fn drop(&mut self) {
        let registration = std::mem::replace(&mut self.registration, Registration::Spent);
        let (Some(key), Some(owner)) = (self.source.take(), self.owner.take()) else {
            return;
        };
        let local = with_current_reactor(|reactor| {
            reactor.owns(&owner) && {
                let _ = reactor.deregister(key);
                true
            }
        })
        .unwrap_or(false);
        if local {
            return;
        }
        match registration {
            Registration::Owned(socket) => owner.release_source(key, socket),
            #[cfg(target_os = "macos")]
            Registration::Process(_) => owner.release_process_exit(key),
            #[cfg(unix)]
            Registration::Descriptor(_) => {}
            Registration::Spent => {}
        }
    }
}

fn stale_registration() -> io::Error {
    io::Error::from(io::ErrorKind::NotFound)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn readiness_off_worker_returns_off_worker_state() {
        #[cfg(unix)]
        let mut readiness = Readiness::new(0);
        #[cfg(windows)]
        let mut readiness =
            Readiness::new(&std::net::UdpSocket::bind("127.0.0.1:0").expect("bind socket"))
                .expect("own readiness source");
        let waker = std::task::Waker::noop();
        let context = Context::from_waker(waker);

        let outcome = readiness
            .poll(&context)
            .expect("polling off-worker never fails");

        assert_eq!(outcome, ReadyState::OffWorker);
    }

    #[cfg(unix)]
    fn fill_until_would_block(fd: std::os::fd::RawFd) -> usize {
        let chunk = [0u8; 4096];
        let mut written = 0;
        loop {
            // safety: fd is a live nonblocking socket and chunk outlives the call.
            let count = unsafe { libc::write(fd, chunk.as_ptr().cast(), chunk.len()) };
            if count < 0 {
                assert_eq!(io::Error::last_os_error().kind(), io::ErrorKind::WouldBlock);
                return written;
            }
            written += count as usize;
        }
    }

    #[cfg(unix)]
    fn drain_all(fd: std::os::fd::RawFd) -> usize {
        let mut sink = [0u8; 65536];
        let mut drained = 0;
        loop {
            // safety: fd is a live nonblocking socket and sink outlives the call.
            let count = unsafe { libc::read(fd, sink.as_mut_ptr().cast(), sink.len()) };
            if count <= 0 {
                return drained;
            }
            drained += count as usize;
        }
    }

    #[cfg(unix)]
    #[test]
    fn writable_gate_parks_until_peer_drains() {
        use super::super::core_shard::launch_with_lanes;
        use proxima_runtime::CoreId;
        use std::os::unix::net::UnixStream;
        use std::sync::mpsc;

        let (writer, peer) = UnixStream::pair().expect("socketpair");
        writer.set_nonblocking(true).expect("nonblocking writer");
        peer.set_nonblocking(true).expect("nonblocking peer");
        let (sender, receiver) = mpsc::channel();
        let handle = launch_with_lanes(CoreId(0), None, 2, 16).expect("launch worker");
        handle
            .dispatch_send(Box::pin(async move {
                let writer_fd = writer.as_raw_fd();
                let peer_fd = peer.as_raw_fd();
                let filled = fill_until_would_block(writer_fd);
                let mut gate = Readiness::writable(writer_fd);
                let mut states = Vec::new();
                for _ in 0..2 {
                    let state = std::future::poll_fn(|context| {
                        std::task::Poll::Ready(gate.poll(context))
                    })
                    .await
                    .expect("poll write gate");
                    states.push(state);
                }
                let drained = drain_all(peer_fd);
                let opened = std::future::poll_fn(|context| match gate.poll(context) {
                    Ok(ReadyState::Retry) => std::task::Poll::Ready(Ok(())),
                    Ok(_) => std::task::Poll::Pending,
                    Err(error) => std::task::Poll::Ready(Err(error)),
                })
                .await;
                let rewritten = fill_until_would_block(writer_fd);
                sender
                    .send((filled, states, drained, opened.is_ok(), rewritten))
                    .expect("report to test thread");
            }))
            .expect("dispatch probe to worker");

        let (filled, states, drained, opened, rewritten) =
            receiver.recv().expect("worker reports gate outcome");

        assert!(filled > 0, "writer accepted bytes before EAGAIN");
        assert_eq!(states, vec![ReadyState::Retry, ReadyState::Parked]);
        assert_eq!(drained, filled, "peer drained exactly what was written");
        assert!(opened, "gate reopened after the peer drained");
        assert!(rewritten > 0, "write succeeds after the gate opened");
        handle.shutdown_and_join().expect("shutdown worker");
    }

    #[cfg(unix)]
    #[test]
    fn registration_error_keeps_raw_errno() {
        use super::super::core_shard::launch_with_lanes;
        use proxima_runtime::CoreId;
        use std::sync::mpsc;

        let (sender, receiver) = mpsc::channel();
        let handle = launch_with_lanes(CoreId(0), None, 2, 16).expect("launch worker");
        handle
            .dispatch_send(Box::pin(async move {
                let mut readiness = Readiness::new(-1);
                let outcome =
                    std::future::poll_fn(|context| std::task::Poll::Ready(readiness.poll(context)))
                        .await;
                let errno = outcome.err().and_then(|error| error.raw_os_error());
                sender.send(errno).expect("report errno to test thread");
            }))
            .expect("dispatch probe to worker");

        let errno = receiver.recv().expect("worker reports registration result");

        assert_eq!(errno, Some(libc::EBADF));
        handle.shutdown_and_join().expect("shutdown worker");
    }
}
