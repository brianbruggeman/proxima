//! Single bounded OS calls that turn a child's exit signal into its status.
//!
//! The gate lives in [`Readiness`](super::readiness::Readiness): kqueue
//! `EVFILT_PROC` on macOS ([`Readiness::process_exit`](super::readiness::Readiness::process_exit)),
//! a pidfd on Linux ([`open_pidfd`] then [`Readiness::new`](super::readiness::Readiness::new)).
//! On Windows the gate is [`Readiness::signaled`](super::readiness::Readiness::signaled) over
//! the process handle and the read step is `exit_status`.
//! A caller composes `gate.and_then(reap)`; these functions are the reap step and
//! never park. Statuses are `std::process::ExitStatus`, so signals and core dumps
//! survive in the raw wait status.

use std::io;
#[cfg(unix)]
use std::num::NonZeroI32;
#[cfg(target_os = "linux")]
use std::os::fd::{AsRawFd, BorrowedFd, FromRawFd, OwnedFd, RawFd};
#[cfg(unix)]
use std::os::unix::process::ExitStatusExt;
#[cfg(windows)]
use std::os::windows::io::{AsRawHandle, BorrowedHandle};
#[cfg(windows)]
use std::os::windows::process::ExitStatusExt;
use std::process::ExitStatus;

#[cfg(windows)]
use windows_sys::Win32::Foundation::{HANDLE, STILL_ACTIVE, WAIT_OBJECT_0};
#[cfg(windows)]
use windows_sys::Win32::System::Threading::{GetExitCodeProcess, WaitForSingleObject};

#[cfg(unix)]
/// Reaps `pid` if it has exited, without blocking.
///
/// # Errors
/// The raw `waitpid` errno, for example `ECHILD` when `pid` is not our child.
pub fn try_reap(pid: NonZeroI32) -> io::Result<Option<ExitStatus>> {
    let mut raw_status: libc::c_int = 0;
    loop {
        // SAFETY: raw_status outlives the call; WNOHANG never blocks.
        let reaped = unsafe { libc::waitpid(pid.get(), &raw mut raw_status, libc::WNOHANG) };
        match reaped {
            0 => return Ok(None),
            reaped if reaped == pid.get() => return Ok(Some(ExitStatus::from_raw(raw_status))),
            _ => {
                let error = io::Error::last_os_error();
                if error.raw_os_error() != Some(libc::EINTR) {
                    return Err(error);
                }
            }
        }
    }
}

/// Reaps `pid` after the kernel reported its exit.
///
/// The caller contract is that `EVFILT_PROC` fired for `pid`, or registration
/// returned `ESRCH`. `NOTE_EXIT` can arrive before `waitpid` can reap, so this
/// waits, bounded by the kernel finishing the exit and not by the child's life.
///
/// # Errors
/// The raw `waitpid` errno.
#[cfg(target_os = "macos")]
pub fn reap_reported(pid: NonZeroI32) -> io::Result<ExitStatus> {
    let mut raw_status: libc::c_int = 0;
    loop {
        // SAFETY: raw_status outlives the call.
        let reaped = unsafe { libc::waitpid(pid.get(), &raw mut raw_status, 0) };
        if reaped == pid.get() {
            return Ok(ExitStatus::from_raw(raw_status));
        }
        let error = io::Error::last_os_error();
        if error.raw_os_error() != Some(libc::EINTR) {
            return Err(error);
        }
    }
}

/// Opens a pidfd for `pid`; readable once the process exits.
///
/// # Errors
/// The raw errno. `ENOSYS`, `EINVAL` and `EPERM` mean the kernel (Linux 5.3+
/// required) or its sandbox has no pidfd.
#[cfg(target_os = "linux")]
pub fn open_pidfd(pid: NonZeroI32) -> io::Result<OwnedFd> {
    // SAFETY: pidfd_open takes a pid and flags and returns a new descriptor or -1.
    let descriptor = unsafe { libc::syscall(libc::SYS_pidfd_open, pid.get(), 0_u32) };
    if descriptor < 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: the kernel just returned this descriptor and nothing else owns it.
    Ok(unsafe { OwnedFd::from_raw_fd(descriptor as RawFd) })
}

/// Reaps the process behind `pidfd` if it has exited, without blocking.
///
/// # Errors
/// The raw `waitid` errno.
#[cfg(target_os = "linux")]
pub fn try_reap_pidfd(pidfd: BorrowedFd<'_>) -> io::Result<Option<ExitStatus>> {
    // SAFETY: a zeroed siginfo_t is the documented "nothing reported" input.
    let mut info: libc::siginfo_t = unsafe { std::mem::zeroed() };
    loop {
        // SAFETY: info outlives the call; WNOHANG never blocks.
        let result = unsafe {
            libc::waitid(
                libc::P_PIDFD,
                pidfd.as_raw_fd() as libc::id_t,
                &raw mut info,
                libc::WEXITED | libc::WNOHANG,
            )
        };
        if result == 0 {
            break;
        }
        let error = io::Error::last_os_error();
        if error.raw_os_error() != Some(libc::EINTR) {
            return Err(error);
        }
    }
    // SAFETY: waitid with WEXITED fills si_pid and si_status for a reaped child.
    let (child_pid, status) = unsafe { (info.si_pid(), info.si_status()) };
    if child_pid == 0 {
        return Ok(None);
    }
    let raw_status = match info.si_code {
        libc::CLD_EXITED => (status & 0xff) << 8,
        libc::CLD_DUMPED => (status & 0x7f) | 0x80,
        _ => status & 0x7f,
    };
    Ok(Some(ExitStatus::from_raw(raw_status)))
}

/// Reads the exit code of `process` once its handle has signaled.
///
/// The gate is [`Readiness::signaled`](super::readiness::Readiness::signaled) over the
/// process handle; a caller composes `gate.and_then(exit_status)`. This is one bounded
/// `GetExitCodeProcess` call and never parks. The code travels as a
/// `std::process::ExitStatus`, so `status.code()` is the child's own code.
///
/// # Errors
/// The raw `GetExitCodeProcess` error, or `WouldBlock` when `process` has not exited.
#[cfg(windows)]
pub fn exit_status(process: BorrowedHandle<'_>) -> io::Result<ExitStatus> {
    let raw = process.as_raw_handle() as HANDLE;
    let mut code = 0u32;
    // SAFETY: `code` is a valid out-pointer and `raw` is borrowed for the call.
    if unsafe { GetExitCodeProcess(raw, &mut code) } == 0 {
        return Err(io::Error::last_os_error());
    }
    // an exit code of 259 is indistinguishable from a live process until the handle is waited on
    // SAFETY: a zero timeout polls the handle's signaled state and never parks.
    if code == STILL_ACTIVE as u32 && unsafe { WaitForSingleObject(raw, 0) } != WAIT_OBJECT_0 {
        return Err(io::Error::from(io::ErrorKind::WouldBlock));
    }
    Ok(ExitStatus::from_raw(code))
}
