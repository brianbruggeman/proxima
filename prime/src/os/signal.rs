//! Process-wide SIGINT/SIGTERM wait, built on the reactor's
//! externally-owned-fd readiness primitive ([`super::readiness::Readiness`])
//! via the classic self-pipe trick — the exact "signalfd, any other
//! single-fd source" case that module's doc comment names as the intended
//! generalization.
//!
//! A signal handler cannot safely allocate, lock, or call anything that
//! isn't async-signal-safe, so it does the one thing that is: `write(2)` one
//! byte to a pre-opened, non-blocking pipe. [`wait_for_signal`] then parks on
//! the pipe's read end via the reactor, the same way any other prime I/O
//! source parks — no dedicated thread, no busy loop.
//!
//! The pipe and both `sigaction` installs happen once per process
//! (`OnceLock`); every call to [`wait_for_signal`] shares the same
//! self-pipe. Only one waiter is expected to actually consume the wake byte
//! at a time (the server shutdown path calls this once) — concurrent
//! waiters would race for the single byte, which is out of scope here the
//! same way it is for `tokio::signal::ctrl_c()`.

use std::io;
use std::os::fd::RawFd;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicI32, Ordering};

use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll};

use super::readiness::{Readiness, ReadyState};

/// The signal handler's only channel to the async side: the self-pipe's
/// write end. `-1` until [`ensure_installed`] runs; the handler checks
/// before writing so a signal delivered before install (impossible in
/// practice — install happens before the handler is registered) is a no-op
/// instead of a write to a bogus fd.
static WRITE_FD: AtomicI32 = AtomicI32::new(-1);

/// Wait for the first of SIGINT or SIGTERM. Resolves exactly once per
/// process-wide delivery; a second call after the first resolves parks on
/// the next delivery (the pipe is a stream, not a one-shot).
///
/// # Errors
/// Returns an [`io::Error`] if the self-pipe or `sigaction` install fails
/// (checked once, cached for the process), or if the reactor read fails.
pub async fn wait_for_signal() -> io::Result<()> {
    let read_fd = ensure_installed()?;
    WaitForSignal {
        fd: read_fd,
        readiness: Readiness::new(read_fd),
    }
    .await
}

fn ensure_installed() -> io::Result<RawFd> {
    static READ_FD: OnceLock<Result<RawFd, i32>> = OnceLock::new();
    READ_FD
        .get_or_init(install)
        .map_err(io::Error::from_raw_os_error)
}

fn install() -> Result<RawFd, i32> {
    let mut fds: [RawFd; 2] = [-1, -1];
    // SAFETY: `fds` is a valid 2-element out-array for pipe(2).
    if unsafe { libc::pipe(fds.as_mut_ptr()) } != 0 {
        return Err(current_errno());
    }
    let [read_fd, write_fd] = fds;
    set_nonblocking_cloexec(read_fd)?;
    set_nonblocking_cloexec(write_fd)?;
    WRITE_FD.store(write_fd, Ordering::Release);
    install_handler(libc::SIGINT)?;
    install_handler(libc::SIGTERM)?;
    Ok(read_fd)
}

fn set_nonblocking_cloexec(fd: RawFd) -> Result<(), i32> {
    // SAFETY: fd is a just-created, owned pipe endpoint; F_GETFL/F_SETFL/
    // F_SETFD are plain fcntl(2) calls on a valid descriptor.
    unsafe {
        let flags = libc::fcntl(fd, libc::F_GETFL, 0);
        if flags < 0 {
            return Err(current_errno());
        }
        if libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) < 0 {
            return Err(current_errno());
        }
        if libc::fcntl(fd, libc::F_SETFD, libc::FD_CLOEXEC) < 0 {
            return Err(current_errno());
        }
    }
    Ok(())
}

fn install_handler(signal: libc::c_int) -> Result<(), i32> {
    let mut action: libc::sigaction = unsafe { core::mem::zeroed() };
    action.sa_sigaction = write_wake_byte as *const () as usize;
    // SA_RESTART: a delivered SIGINT/SIGTERM must not turn every other
    // blocking syscall in the process (reactor's kevent/epoll_wait among
    // them) into a spurious EINTR error — kevent's own EINTR case in
    // reactor.rs is a defense-in-depth backstop, this is the primary one.
    action.sa_flags = libc::SA_RESTART;
    // SAFETY: `action.sa_mask` is a valid, owned `sigset_t` local.
    unsafe {
        libc::sigemptyset(&raw mut action.sa_mask);
    }
    // SAFETY: `action` is a fully-initialized sigaction with a valid,
    // 'static function pointer; sigaction(2) on a process-wide signal
    // number is the documented way to install a handler.
    if unsafe { libc::sigaction(signal, &action, core::ptr::null_mut()) } != 0 {
        return Err(current_errno());
    }
    Ok(())
}

extern "C" fn write_wake_byte(_signal: libc::c_int) {
    let fd = WRITE_FD.load(Ordering::Acquire);
    if fd < 0 {
        return;
    }
    let byte: u8 = 1;
    // SAFETY: async-signal-safe write(2) to a fd this module owns; the
    // pipe is non-blocking so this cannot stall inside the handler, and
    // the result is intentionally discarded — EAGAIN just means a wake
    // byte is already pending, and a handler must never panic.
    unsafe {
        let _ = libc::write(fd, (&raw const byte).cast(), 1);
    }
}

fn current_errno() -> i32 {
    io::Error::last_os_error().raw_os_error().unwrap_or(-1)
}

struct WaitForSignal {
    fd: RawFd,
    readiness: Readiness,
}

impl Future for WaitForSignal {
    type Output = io::Result<()>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        let this = self.get_mut();
        loop {
            let mut byte = [0u8; 1];
            // SAFETY: `this.fd` is the self-pipe's non-blocking read end,
            // owned for the process lifetime.
            let read_result = unsafe { libc::read(this.fd, byte.as_mut_ptr().cast(), 1) };
            if read_result > 0 {
                return Poll::Ready(Ok(()));
            }
            if read_result < 0 {
                let err = io::Error::last_os_error();
                if err.kind() != io::ErrorKind::WouldBlock {
                    return Poll::Ready(Err(err));
                }
            }
            match this.readiness.poll(cx) {
                Ok(ReadyState::Retry) => continue,
                Ok(ReadyState::Parked) => return Poll::Pending,
                Ok(ReadyState::OffWorker) => {
                    // no reactor on this thread (plain block_on, no
                    // worker) — busy-poll like every other off-worker
                    // Readiness caller (see packet_listener.rs).
                    cx.waker().wake_by_ref();
                    return Poll::Pending;
                }
                Err(error) => return Poll::Ready(Err(error)),
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn self_pipe_installs_once_and_reuses_the_same_fd() {
        let first = ensure_installed().expect("install");
        let second = ensure_installed().expect("install");
        assert_eq!(first, second, "the self-pipe is process-wide, not per-call");
    }

    #[test]
    fn raising_sigint_wakes_the_pipe() {
        let read_fd = ensure_installed().expect("install");
        // drain any byte left over from a previous test in this process.
        let mut drain = [0u8; 8];
        while unsafe { libc::read(read_fd, drain.as_mut_ptr().cast(), drain.len()) } > 0 {}

        // SAFETY: raising a signal against our own process is the documented
        // use of raise(2); the handler installed by `ensure_installed` above
        // only performs an async-signal-safe write(2).
        let raised = unsafe { libc::raise(libc::SIGINT) };
        assert_eq!(raised, 0, "raise(SIGINT) must succeed");

        let mut byte = [0u8; 1];
        let mut read_result = unsafe { libc::read(read_fd, byte.as_mut_ptr().cast(), 1) };
        // the handler runs asynchronously w.r.t. raise(2) returning on some
        // platforms; give it a bounded number of immediate retries rather
        // than sleeping (raise(2) is documented to deliver before returning
        // on the raising thread for a synchronous signal like this, so this
        // loop is a belt-and-suspenders formality, not a real wait).
        let mut attempts = 0;
        while read_result <= 0 && attempts < 1000 {
            read_result = unsafe { libc::read(read_fd, byte.as_mut_ptr().cast(), 1) };
            attempts += 1;
        }
        assert_eq!(
            read_result, 1,
            "the SIGINT handler must have written one wake byte"
        );
        assert_eq!(byte[0], 1);
    }
}
