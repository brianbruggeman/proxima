//! Kernel PTY pair allocation + window-size ioctls.
//!
//! [`open_pty`] returns `(master, slave)`. The caller hands the slave
//! to a [`Command`](super::descriptor::CommandDescriptor) via
//! [`Io::Fd`](super::descriptor::Stdio::Fd) on `input`/`output`/`error`
//! and sets `controlling_tty = true`. After `spawn`, the caller drops
//! its slave reference so the kernel PTY closes cleanly when the
//! child exits.
//!
//! # Composition
//!
//! 1. `open_pty(size)` → `(master, slave)`
//! 2. `CommandDescriptor::new(...).input(Io::Fd(slave_raw)).output(Io::Fd(slave_raw)).error(Io::Fd(slave_raw)).controlling_tty(true)`
//! 3. [`spawn(&command)`](super::spawn::spawn) — honors `controlling_tty`
//!    (setsid + `ioctl(TIOCSCTTY)`) and the `Io::Fd` slots
//! 4. Drop the parent's slave
//! 5. Wrap the master in
//!    [`FdPairPipe`](super::fd_pipe::FdPairPipe) — see
//!    [`PtyCommandPipe`](super::pty_pipe::PtyCommandPipe).
//!
//! # Safety surface
//!
//! Uses `nix::pty::posix_openpt` and `nix::ioctl_*!`-generated wrappers
//! for TIOCGWINSZ/TIOCSWINSZ. The ioctl macros emit `unsafe fn`s
//! because the safety contract is per-ioctl (caller asserts the
//! fd kind matches the request). Those unsafe blocks live in this
//! file's private wrappers; the public API
//! ([`terminal_size`] / [`current_terminal_size`] / [`set_pty_size`])
//! is safe.

#[cfg(target_vendor = "apple")]
use std::ffi::CStr;
use std::os::fd::{AsFd, AsRawFd, BorrowedFd, OwnedFd};

use nix::errno::Errno;
#[cfg(not(any(target_os = "linux", target_os = "android", target_vendor = "apple")))]
use nix::fcntl::{FcntlArg, FdFlag, fcntl};
use nix::fcntl::{OFlag, open};
#[cfg(any(target_os = "linux", target_os = "android"))]
use nix::pty::ptsname_r;
#[cfg(not(any(target_os = "linux", target_os = "android", target_vendor = "apple")))]
use nix::pty::{OpenptyResult, openpty};
use nix::pty::{PtyMaster, Winsize, grantpt, posix_openpt, unlockpt};
use nix::sys::stat::Mode;
use proxima_primitives::pipe::ProximaError;

/// Terminal size in character cells.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PtySize {
    /// Number of rows (lines).
    pub rows: u16,
    /// Number of columns.
    pub cols: u16,
}

impl Default for PtySize {
    fn default() -> Self {
        Self { rows: 24, cols: 80 }
    }
}

impl PtySize {
    fn to_winsize(self) -> Winsize {
        Winsize {
            ws_row: self.rows,
            ws_col: self.cols,
            ws_xpixel: 0,
            ws_ypixel: 0,
        }
    }
}

// nix-generated ioctl wrappers. The macros expand to `unsafe fn`s
// because the safety contract is "caller asserts the ioctl is
// appropriate for this fd kind". We wrap them in safe public APIs
// below; the unsafe blocks are internal and scoped to one syscall.
nix::ioctl_read_bad!(get_winsize_ioctl, libc::TIOCGWINSZ, Winsize);
nix::ioctl_write_ptr_bad!(set_winsize_ioctl, libc::TIOCSWINSZ, Winsize);

/// Query the terminal size of `fd` via `TIOCGWINSZ`.
///
/// Returns `None` when `fd` is not a terminal (the ioctl returns
/// ENOTTY) or when the kernel reports zero rows/cols.
pub fn terminal_size(fd: BorrowedFd<'_>) -> Option<PtySize> {
    let mut ws = Winsize {
        ws_row: 0,
        ws_col: 0,
        ws_xpixel: 0,
        ws_ypixel: 0,
    };
    // SAFETY: TIOCGWINSZ on a non-TTY returns ENOTTY rather than
    // misbehaving. BorrowedFd guarantees the fd is valid for the
    // call. nix wraps the raw syscall.
    let result = unsafe { get_winsize_ioctl(fd.as_raw_fd(), &raw mut ws) };
    match result {
        Ok(_) if ws.ws_row > 0 && ws.ws_col > 0 => Some(PtySize {
            rows: ws.ws_row,
            cols: ws.ws_col,
        }),
        _ => None,
    }
}

/// Query the calling process's stdout terminal size.
///
/// Returns `None` if stdout is not a terminal (pipe / file redirect
/// / CI). Common pattern: `current_terminal_size().unwrap_or_default()`
/// for an 80×24 fallback.
pub fn current_terminal_size() -> Option<PtySize> {
    // BorrowedFd from raw STDOUT_FILENO; the fd is process-global
    // and lives for the lifetime of the program.
    let stdout = unsafe { BorrowedFd::borrow_raw(libc::STDOUT_FILENO) };
    terminal_size(stdout)
}

/// Set the terminal size of `fd` via `TIOCSWINSZ`. For a PTY master
/// fd, the kernel propagates SIGWINCH to the slave's controlling-tty
/// process group.
pub fn set_pty_size(fd: BorrowedFd<'_>, size: PtySize) -> Result<(), ProximaError> {
    let ws = size.to_winsize();
    // SAFETY: TIOCSWINSZ on a non-TTY returns ENOTTY rather than
    // misbehaving. BorrowedFd guarantees fd validity for the call.
    unsafe { set_winsize_ioctl(fd.as_raw_fd(), &raw const ws) }
        .map(|_| ())
        .map_err(|err| ProximaError::Body(format!("TIOCSWINSZ: {err}")))
}

/// Allocate a kernel PTY pair sized `size`. Returns `(master, slave)`.
///
/// Both fds are created close-on-exec so a spawned child never inherits
/// a stray handle on its own pty; the spawner dup2s the slave onto
/// 0/1/2, and those copies are not close-on-exec.
#[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
pub fn open_pty(size: PtySize) -> Result<(OwnedFd, OwnedFd), ProximaError> {
    let master = posix_openpt(OFlag::O_RDWR | OFlag::O_NOCTTY | OFlag::O_CLOEXEC)
        .map_err(|err| pty_error("posix_openpt", err))?;
    grantpt(&master).map_err(|err| pty_error("grantpt", err))?;
    unlockpt(&master).map_err(|err| pty_error("unlockpt", err))?;
    let path = slave_path(&master).map_err(|err| pty_error("slave path", err))?;
    let slave_flags = OFlag::O_RDWR | OFlag::O_NOCTTY | OFlag::O_CLOEXEC;
    let slave = open(path.as_str(), slave_flags, Mode::empty())
        .map_err(|err| pty_error("open slave", err))?;
    let master = OwnedFd::from(master);
    set_pty_size(master.as_fd(), size)?;
    Ok((master, slave))
}

/// Fallback for unix targets without a reentrant slave-name lookup: the
/// fds are only marked close-on-exec after creation, so a concurrent
/// fork in another thread can still inherit them.
#[cfg(not(any(target_os = "linux", target_os = "android", target_vendor = "apple")))]
pub fn open_pty(size: PtySize) -> Result<(OwnedFd, OwnedFd), ProximaError> {
    let winsize = size.to_winsize();
    let OpenptyResult { master, slave } =
        openpty(Some(&winsize), None).map_err(|err| pty_error("openpty", err))?;
    for descriptor in [&master, &slave] {
        fcntl(descriptor, FcntlArg::F_SETFD(FdFlag::FD_CLOEXEC))
            .map_err(|err| pty_error("F_SETFD", err))?;
    }
    Ok((master, slave))
}

fn pty_error(step: &str, err: Errno) -> ProximaError {
    ProximaError::Body(format!("pty {step}: {err}"))
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn slave_path(master: &PtyMaster) -> Result<String, Errno> {
    ptsname_r(master)
}

#[cfg(target_vendor = "apple")]
fn slave_path(master: &PtyMaster) -> Result<String, Errno> {
    let mut name = [0 as libc::c_char; 128];
    // SAFETY: TIOCPTYGNAME writes a NUL-terminated name of at most 128
    // bytes into the buffer; the buffer is exactly 128 bytes and the fd
    // is a live pty master borrowed for the call.
    let status = unsafe {
        libc::ioctl(
            master.as_raw_fd(),
            libc::c_ulong::from(libc::TIOCPTYGNAME),
            name.as_mut_ptr(),
        )
    };
    if status < 0 {
        return Err(Errno::last());
    }
    // SAFETY: the kernel NUL-terminated the name within the buffer.
    let cname = unsafe { CStr::from_ptr(name.as_ptr()) };
    Ok(cname.to_string_lossy().into_owned())
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

    use std::ffi::CString;
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;

    use nix::fcntl::{FcntlArg, FdFlag, fcntl};
    use nix::sys::signal::{Signal, kill};
    use nix::unistd::Pid;

    use super::*;
    use crate::descriptor::{CommandDescriptor, Stdio};
    use crate::spawn::{SpawnOptions, spawn};

    fn close_on_exec(descriptor: BorrowedFd<'_>) -> bool {
        let raw = fcntl(descriptor, FcntlArg::F_GETFD).expect("fcntl F_GETFD failed");
        FdFlag::from_bits_truncate(raw).contains(FdFlag::FD_CLOEXEC)
    }

    #[test]
    fn open_pty_fds_are_close_on_exec() {
        let (master, slave) = open_pty(PtySize::default()).expect("openpty");
        assert!(close_on_exec(master.as_fd()), "master lacks FD_CLOEXEC");
        assert!(close_on_exec(slave.as_fd()), "slave lacks FD_CLOEXEC");
    }

    #[test]
    fn dropping_the_master_hangs_up_a_controlling_tty_child() {
        let (master, slave) = open_pty(PtySize::default()).expect("openpty");
        let slave_fd = slave.as_raw_fd();
        let mut command = CommandDescriptor::new(CString::new("/bin/cat").expect("cstring"));
        command
            .stdin(Stdio::Fd(slave_fd))
            .stdout(Stdio::Fd(slave_fd))
            .stderr(Stdio::Fd(slave_fd));
        let options = SpawnOptions {
            controlling_tty: true,
            ..SpawnOptions::default()
        };
        let mut child = spawn(&command, options).expect("spawn cat on the pty slave");
        let pid = Pid::from_raw(child.pid());
        drop(slave);
        drop(master);
        let (sender, receiver) = mpsc::channel();
        thread::spawn(move || {
            let _ = sender.send(child.wait());
        });
        match receiver.recv_timeout(Duration::from_secs(10)) {
            Ok(status) => {
                status.expect("waiting on the child failed");
            }
            Err(_) => {
                kill(pid, Signal::SIGKILL).expect("kill of the stuck child failed");
                panic!("child kept the pty alive after the parent dropped the master");
            }
        }
    }

    #[test]
    fn open_pty_returns_distinct_fds() {
        let (master, slave) = open_pty(PtySize::default()).expect("openpty");
        assert_ne!(master.as_raw_fd(), slave.as_raw_fd());
        assert!(master.as_raw_fd() > 2);
        assert!(slave.as_raw_fd() > 2);
    }

    #[test]
    fn open_pty_honors_size() {
        let target = PtySize {
            rows: 40,
            cols: 132,
        };
        let (master, _slave) = open_pty(target).expect("openpty");
        let size = terminal_size(master.as_fd()).expect("master has size");
        assert_eq!(size, target);
    }

    #[test]
    fn set_pty_size_updates_winsize() {
        let (master, _slave) = open_pty(PtySize::default()).expect("openpty");
        set_pty_size(
            master.as_fd(),
            PtySize {
                rows: 50,
                cols: 200,
            },
        )
        .expect("set_pty_size");
        let size = terminal_size(master.as_fd()).expect("master has size");
        assert_eq!(
            size,
            PtySize {
                rows: 50,
                cols: 200
            }
        );
    }
}
