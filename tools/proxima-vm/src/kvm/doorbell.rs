use core::convert::Infallible;
use core::future::Future;
use std::io;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU8, Ordering};

use proxima_primitives::pipe::Pipe;
use proxima_primitives::sync::blocking::Mutex;

use super::error::{KvmError, KvmResult};

extern "C" fn interrupt_only(_signal: libc::c_int) {}

static HANDLER: OnceLock<Option<i32>> = OnceLock::new();

/// Installed once per process and never restored: its only job is to make a
/// blocked `KVM_RUN` return `EINTR`, so there is no previous action worth
/// putting back, and concurrent users cannot race each other's restore.
fn install_interrupt_handler() -> KvmResult<()> {
    let failure = HANDLER.get_or_init(|| {
        // SAFETY: a zeroed sigaction is a valid empty action; the handler does nothing.
        let mut action: libc::sigaction = unsafe { core::mem::zeroed() };
        action.sa_sigaction = interrupt_only as extern "C" fn(libc::c_int) as usize;
        // SAFETY: `sa_mask` is a plain sigset owned by `action`.
        unsafe { libc::sigemptyset(&raw mut action.sa_mask) };
        // SAFETY: the pointer references a live sigaction; the old action is not wanted.
        match unsafe { libc::sigaction(libc::SIGUSR2, &raw const action, core::ptr::null_mut()) } {
            0 => None,
            _ => Some(io::Error::last_os_error().raw_os_error().unwrap_or(libc::EINVAL)),
        }
    });
    match failure {
        None => Ok(()),
        Some(errno) => Err(KvmError::syscall("sigaction SIGUSR2", io::Error::from_raw_os_error(*errno))),
    }
}

struct Target {
    flag: *mut u8,
    thread: libc::pid_t,
}

// SAFETY: `flag` is a byte in the vCPU's mapped run page, touched only through atomics and
// only while the enclosing mutex holds the target, which `retire` removes before the page is released.
unsafe impl Send for Target {}

/// Wakes a vCPU thread blocked in `KVM_RUN` from any other thread: sets the run
/// page's `immediate_exit` byte, then signals the thread so the ioctl returns
/// `EINTR`. Setting the byte first covers the signal landing before the ioctl
/// is entered. As a sink pipe (`() -> ()`) a call rings it, so a producer
/// composes `AndThen::new(frame_sink, doorbell)` to wake the guest per frame.
pub struct Doorbell {
    target: Mutex<Option<Target>>,
}

impl Doorbell {
    /// Binds the doorbell to the calling thread, which must be the thread that
    /// runs the vCPU. `flag` is that vCPU's `immediate_exit` byte.
    pub fn for_current_thread(flag: *mut u8) -> KvmResult<Self> {
        install_interrupt_handler()?;
        // SAFETY: gettid has no preconditions.
        let thread = unsafe { libc::syscall(libc::SYS_gettid) } as libc::pid_t;
        Ok(Self { target: Mutex::new(Some(Target { flag, thread })) })
    }

    /// Callable from any thread. After `retire` this does nothing.
    pub fn ring(&self) {
        let guard = self.target.lock();
        let Some(target) = guard.as_ref() else {
            return;
        };
        // SAFETY: the run page outlives the target; `retire` takes it out under this lock.
        unsafe { AtomicU8::from_ptr(target.flag) }.store(1, Ordering::Release);
        // SAFETY: tgkill against a live thread of this process; `retire` runs on that thread
        // before it exits, so the id cannot have been recycled while the target is present.
        unsafe { libc::syscall(libc::SYS_tgkill, libc::getpid(), target.thread, libc::SIGUSR2) };
    }

    /// Lowers `immediate_exit`; called on the vCPU thread once it has seen the ring.
    pub fn clear(&self) {
        if let Some(target) = self.target.lock().as_ref() {
            // SAFETY: as in `ring`, the page is live while the target is present.
            unsafe { AtomicU8::from_ptr(target.flag) }.store(0, Ordering::Release);
        }
    }

    /// Called on the vCPU thread before the run page is released; once it
    /// returns no ring can touch the page or signal the thread.
    pub fn retire(&self) {
        *self.target.lock() = None;
    }
}

impl Pipe for Doorbell {
    type In = ();
    type Out = ();
    type Err = Infallible;

    fn call(&self, (): ()) -> impl Future<Output = Result<(), Infallible>> {
        self.ring();
        core::future::ready(Ok(()))
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use futures::FutureExt;

    #[test]
    fn ring_sets_the_flag_and_clear_lowers_it() {
        let mut flag = 0_u8;
        let doorbell = Doorbell::for_current_thread(&raw mut flag).expect("handler installs");

        doorbell.ring();
        // SAFETY: `flag` is live on this stack frame.
        assert_eq!(unsafe { AtomicU8::from_ptr(&raw mut flag) }.load(Ordering::Acquire), 1);
        doorbell.clear();

        assert_eq!(unsafe { AtomicU8::from_ptr(&raw mut flag) }.load(Ordering::Acquire), 0);
    }

    #[test]
    fn ring_after_retire_never_touches_the_flag() {
        let mut flag = 0_u8;
        let doorbell = Doorbell::for_current_thread(&raw mut flag).expect("handler installs");

        doorbell.retire();
        doorbell.ring();

        assert_eq!(unsafe { AtomicU8::from_ptr(&raw mut flag) }.load(Ordering::Acquire), 0);
    }

    #[test]
    fn calling_the_doorbell_as_a_pipe_rings_it() {
        let mut flag = 0_u8;
        let doorbell = Doorbell::for_current_thread(&raw mut flag).expect("handler installs");

        doorbell.call(()).now_or_never().expect("ready").expect("infallible");

        assert_eq!(unsafe { AtomicU8::from_ptr(&raw mut flag) }.load(Ordering::Acquire), 1);
    }

    #[test]
    fn ring_from_another_thread_interrupts_a_blocked_wait() {
        let mut flag = 0_u8;
        let doorbell = Doorbell::for_current_thread(&raw mut flag).expect("handler installs");
        // SAFETY: zeroed sigsets are valid out-parameters, initialised by sigemptyset below.
        let (mut blocked, mut previous): (libc::sigset_t, libc::sigset_t) = unsafe { core::mem::zeroed() };
        // SAFETY: both sets are live locals.
        unsafe {
            libc::sigemptyset(&raw mut blocked);
            libc::sigaddset(&raw mut blocked, libc::SIGUSR2);
            libc::pthread_sigmask(libc::SIG_BLOCK, &raw const blocked, &raw mut previous);
        }

        std::thread::scope(|scope| {
            scope.spawn(|| doorbell.ring());
            // SAFETY: no descriptors; the previous mask unblocks the pending signal atomically, so the
            // ring cannot be lost between the check and the wait.
            let waited = unsafe { libc::ppoll(core::ptr::null_mut(), 0, core::ptr::null(), &raw const previous) };
            assert_eq!(waited, -1);
            assert_eq!(io::Error::last_os_error().raw_os_error(), Some(libc::EINTR));
        });

        // SAFETY: restores the mask captured above.
        unsafe { libc::pthread_sigmask(libc::SIG_SETMASK, &raw const previous, core::ptr::null_mut()) };
    }
}
