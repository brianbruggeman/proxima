use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;

use omega::backend::GpuDriver;
use proxima_telemetry::warn;

/// A driver's one-time setup ([`GpuDriver::warm`]) running on a worker thread
/// while the caller prepares host data; [`Self::settle`] is where the first real
/// evaluation waits for it. No driver, no worker: the runtime is already warm.
///
/// The worker is a plain thread rather than a pipe: it runs one blocking driver
/// call and reports through a flag, with no stream of values to compose.
pub(super) struct WarmUp {
    worker: Option<JoinHandle<()>>,
    done: Arc<AtomicBool>,
    launched: bool,
}

impl WarmUp {
    /// Starts warming `driver` now; `None` starts nothing.
    pub(super) fn start(driver: Option<GpuDriver>) -> Self {
        let done = Arc::new(AtomicBool::new(driver.is_none()));
        let Some(driver) = driver else {
            return Self { worker: None, done, launched: false };
        };
        let flag = Arc::clone(&done);
        let spawned = std::thread::Builder::new()
            .name("omega-warm".to_string())
            .spawn(move || {
                if let Err(error) = driver.warm() {
                    warn!(driver = driver.name(), error = %error, "driver warm-up failed; first evaluation pays setup");
                }
                flag.store(true, Ordering::Release);
            });
        match spawned {
            Ok(worker) => Self { worker: Some(worker), done, launched: true },
            Err(error) => {
                warn!(error = %error, "warm-up thread refused; first evaluation pays setup");
                done.store(true, Ordering::Release);
                Self { worker: None, done, launched: false }
            }
        }
    }

    /// Blocks until the worker has finished; a no-op once settled.
    pub(super) fn settle(&mut self) {
        if let Some(worker) = self.worker.take() && worker.join().is_err() {
            warn!("warm-up thread panicked; first evaluation pays setup");
            self.done.store(true, Ordering::Release);
        }
    }

    /// Whether the warm-up has finished (trivially true when there was no driver).
    pub(super) fn warmed(&self) -> bool {
        self.done.load(Ordering::Acquire)
    }

    /// Whether a worker was started for this runtime.
    pub(super) fn launched(&self) -> bool {
        self.launched
    }
}
