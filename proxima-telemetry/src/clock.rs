use core::sync::atomic::{AtomicU64, Ordering};

/// Monotonic timestamp source, injected by the caller.
///
/// Shared by c5-trace (spans) and c8-log (log records).
/// v1 hosts inject a concrete impl; C9/C13 will wire platform defaults.
pub trait Clock {
    fn now_ns(&self) -> u64;
}

/// Atomic counter clock — always-ascending, no platform syscall.
///
/// Suitable for tests and no_std environments where wall-clock precision is
/// irrelevant; each call returns a value one higher than the last.
pub struct MonotonicCounter(AtomicU64);

impl MonotonicCounter {
    pub const fn new(start: u64) -> Self {
        Self(AtomicU64::new(start))
    }
}

impl Clock for MonotonicCounter {
    fn now_ns(&self) -> u64 {
        self.0.fetch_add(1, Ordering::Relaxed)
    }
}

impl Clock for alloc::sync::Arc<dyn Clock + Send + Sync> {
    fn now_ns(&self) -> u64 {
        self.as_ref().now_ns()
    }
}
