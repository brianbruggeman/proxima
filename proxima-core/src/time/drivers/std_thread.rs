//! Host fallback driver — spawns a `std::thread` per scheduled wake.
//! Cheap to write, suboptimal under load; the right driver under heavy
//! timer workloads is `prime-wheel` or `embassy-time`.

use std::sync::OnceLock;
use std::task::Waker;

use crate::time::{Driver, Instant};

/// Singleton driver instance — referenced by the static binding emitted
/// by `proxima-core`'s `build.rs` when `timer = "std-thread"`.
pub static DRIVER: StdThreadDriver = StdThreadDriver {
    epoch: OnceLock::new(),
};

/// State for the host-thread driver. The epoch is captured lazily on
/// first use so the driver remains construct-in-static-context.
pub struct StdThreadDriver {
    epoch: OnceLock<std::time::Instant>,
}

impl StdThreadDriver {
    fn epoch(&self) -> std::time::Instant {
        *self.epoch.get_or_init(std::time::Instant::now)
    }
}

impl Driver for StdThreadDriver {
    fn now(&self) -> Instant {
        let elapsed = std::time::Instant::now().saturating_duration_since(self.epoch());
        Instant::from_monotonic(elapsed)
    }

    fn schedule_wake(&self, deadline: Instant, waker: Waker) {
        let target = self.epoch() + deadline.into_monotonic();
        std::thread::spawn(move || {
            let now = std::time::Instant::now();
            if now < target {
                std::thread::sleep(target - now);
            }
            waker.wake();
        });
    }
}
