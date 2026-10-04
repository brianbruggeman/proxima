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
#[cfg(unix)]
use std::os::fd::{AsRawFd, FromRawFd};
#[cfg(windows)]
use std::os::windows::io::AsSocket;
use std::task::Context;

use super::core_shard::with_current_reactor;
use super::reactor::{Interest, OwnedSource, SourceKey, Wakeup};

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

/// Reactor registration for one externally-owned fd. Deregisters on drop.
pub struct Readiness {
    #[cfg(unix)]
    fd: RawSource,
    socket: Option<OwnedSource>,
    owner: Option<Wakeup>,
    source: Option<SourceKey>,
    last_blocked_epoch: Option<u32>,
}

impl Readiness {
    #[cfg(unix)]
    #[must_use]
    pub fn new(fd: RawSource) -> Self {
        Self {
            fd,
            socket: None,
            owner: None,
            source: None,
            last_blocked_epoch: None,
        }
    }

    #[cfg(windows)]
    pub fn new(socket: &impl AsSocket) -> io::Result<Self> {
        Ok(Self {
            socket: Some(socket.as_socket().try_clone_to_owned()?),
            owner: None,
            source: None,
            last_blocked_epoch: None,
        })
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
            if self.socket.is_none() {
                // safety: fcntl validates the descriptor; the duplicate belongs to this readiness.
                let duplicate = unsafe { libc::fcntl(self.fd, libc::F_DUPFD_CLOEXEC, 0) };
                if duplicate < 0 {
                    return Err(io::Error::last_os_error());
                }
                self.socket = Some(unsafe { OwnedSource::from_raw_fd(duplicate) });
            }
            let socket = self
                .socket
                .as_ref()
                .ok_or_else(|| io::Error::other("readiness source missing"))?;
            #[cfg(unix)]
            let fd = socket.as_raw_fd();
            #[cfg(windows)]
            let fd = socket;

            let key = match self.source {
                Some(existing) => existing,
                None => {
                    let registered = reactor
                        .register(fd, Interest::Read)
                        .map_err(|err| registration_failed(errno_of(&err)))?;
                    self.source = Some(registered);
                    self.owner = Some(reactor.wakeup());
                    registered
                }
            };
            #[cfg(windows)]
            reactor.rearm(key, Interest::Read)?;
            if !reactor.register_read_waker_ref(key, context.waker()) {
                return Err(registration_failed(-1));
            }
            reactor
                .read_ready_epoch(key)
                .ok_or_else(|| registration_failed(-1))
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
}

impl Drop for Readiness {
    fn drop(&mut self) {
        if let (Some(key), Some(owner), Some(socket)) =
            (self.source.take(), self.owner.take(), self.socket.take())
        {
            let local = with_current_reactor(|reactor| {
                if reactor.owns(&owner) {
                    let _ = reactor.deregister(key);
                    true
                } else {
                    false
                }
            })
            .unwrap_or(false);
            if !local {
                owner.release_source(key, socket);
            }
        }
    }
}

fn errno_of(error: &io::Error) -> i32 {
    error.raw_os_error().unwrap_or(-1)
}

fn registration_failed(errno: i32) -> io::Error {
    io::Error::other(format!(
        "reactor read-readiness registration failed (errno {errno})"
    ))
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
}
