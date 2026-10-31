use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use super::error::{KvmError, KvmResult};

extern "C" fn interrupt_only(_signal: libc::c_int) {}

struct ImmediateExit(*mut u8);

// SAFETY: the flag is a byte in the vCPU's mapped run page, written only through atomics.
unsafe impl Send for ImmediateExit {}
unsafe impl Sync for ImmediateExit {}

struct Shared {
    fired: AtomicBool,
    finished: AtomicBool,
    flag: ImmediateExit,
    vcpu_thread: libc::pthread_t,
}

impl Shared {
    fn kick(&self) {
        self.fired.store(true, Ordering::Release);
        // SAFETY: the run page outlives the watchdog, which joins before the vCPU is released.
        unsafe { AtomicU8::from_ptr(self.flag.0) }.store(1, Ordering::Release);
        // SAFETY: the vCPU thread is blocked in, or about to enter, KVM_RUN and outlives the join.
        unsafe { libc::pthread_kill(self.vcpu_thread, libc::SIGUSR2) };
    }
}

pub struct Watchdog {
    shared: Arc<Shared>,
    worker: Option<JoinHandle<()>>,
    previous_action: libc::sigaction,
}

fn install_interrupt_handler() -> KvmResult<libc::sigaction> {
    // SAFETY: zeroed sigaction is a valid empty action; the handler does nothing.
    let mut action: libc::sigaction = unsafe { core::mem::zeroed() };
    action.sa_sigaction = interrupt_only as extern "C" fn(libc::c_int) as usize;
    // SAFETY: `sa_mask` is a plain sigset owned by `action`.
    unsafe { libc::sigemptyset(&raw mut action.sa_mask) };
    // SAFETY: a zeroed sigaction is a valid out-parameter for the previous action.
    let mut previous: libc::sigaction = unsafe { core::mem::zeroed() };
    // SAFETY: both pointers reference live sigaction values.
    match unsafe { libc::sigaction(libc::SIGUSR2, &raw const action, &raw mut previous) } {
        0 => Ok(previous),
        _ => Err(KvmError::last_os("sigaction SIGUSR2")),
    }
}

fn wait_out(shared: &Shared, deadline: Instant) {
    while !shared.finished.load(Ordering::Acquire) {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            shared.kick();
            return;
        }
        thread::park_timeout(remaining);
    }
}

impl Watchdog {
    /// Starts a thread that, after `millis`, sets `immediate_exit` on the run
    /// page `flag` points at and signals the calling thread so a blocked
    /// `KVM_RUN` returns `EINTR`. Setting the flag first covers the signal
    /// landing before the ioctl is entered. `SIGUSR2`'s handler is a no-op
    /// installed without `SA_RESTART`: its only job is to interrupt.
    pub fn arm(flag: *mut u8, millis: u64) -> KvmResult<Option<Self>> {
        if millis == 0 {
            return Ok(None);
        }
        let previous_action = install_interrupt_handler()?;
        let shared = Arc::new(Shared {
            fired: AtomicBool::new(false),
            finished: AtomicBool::new(false),
            flag: ImmediateExit(flag),
            // SAFETY: pthread_self has no preconditions.
            vcpu_thread: unsafe { libc::pthread_self() },
        });
        let deadline = Instant::now() + Duration::from_millis(millis);
        let worker_shared = Arc::clone(&shared);
        let worker = thread::Builder::new()
            .name("proxima-vm-watchdog".to_owned())
            .spawn(move || wait_out(&worker_shared, deadline))
            .map_err(|error| KvmError::syscall("spawn watchdog thread", error))?;
        Ok(Some(Self { shared, worker: Some(worker), previous_action }))
    }

    #[must_use]
    pub fn fired(&self) -> bool {
        self.shared.fired.load(Ordering::Acquire)
    }
}

impl Drop for Watchdog {
    fn drop(&mut self) {
        self.shared.finished.store(true, Ordering::Release);
        if let Some(worker) = self.worker.take() {
            worker.thread().unpark();
            let _ = worker.join();
        }
        // SAFETY: restores exactly the action `arm` replaced.
        unsafe { libc::sigaction(libc::SIGUSR2, &raw const self.previous_action, core::ptr::null_mut()) };
    }
}
