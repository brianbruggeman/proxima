use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use super::doorbell::Doorbell;
use super::error::{KvmError, KvmResult};

struct Shared {
    fired: AtomicBool,
    finished: AtomicBool,
    doorbell: Doorbell,
}

impl Shared {
    fn kick(&self) {
        self.fired.store(true, Ordering::Release);
        self.doorbell.ring();
    }
}

pub struct Watchdog {
    shared: Arc<Shared>,
    worker: Option<JoinHandle<()>>,
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
    /// Starts a thread that, after `millis`, rings a doorbell bound to the
    /// calling thread: `immediate_exit` is set on the run page `flag` points at
    /// and the thread is signalled so a blocked `KVM_RUN` returns `EINTR`.
    pub fn arm(flag: *mut u8, millis: u64) -> KvmResult<Option<Self>> {
        if millis == 0 {
            return Ok(None);
        }
        let shared = Arc::new(Shared {
            fired: AtomicBool::new(false),
            finished: AtomicBool::new(false),
            doorbell: Doorbell::for_current_thread(flag)?,
        });
        let deadline = Instant::now() + Duration::from_millis(millis);
        let worker_shared = Arc::clone(&shared);
        let worker = thread::Builder::new()
            .name("proxima-vm-watchdog".to_owned())
            .spawn(move || wait_out(&worker_shared, deadline))
            .map_err(|error| KvmError::syscall("spawn watchdog thread", error))?;
        Ok(Some(Self { shared, worker: Some(worker) }))
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
        self.shared.doorbell.retire();
    }
}
