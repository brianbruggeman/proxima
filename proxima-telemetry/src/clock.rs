use core::sync::atomic::{AtomicU64, Ordering};

/// Monotonic timestamp source, injected by the caller.
///
/// Shared by c5-trace (spans) and c8-log (log records). Deliberately
/// object-safe (no associated types): the per-core `Recorder` erases its
/// injected clock behind `Arc<dyn Clock + Send + Sync>` (see `recorder`'s
/// `SystemClock`), which `proxima_primitives::pipe::capabilities::Clock`'s
/// `delay`-capable, associated-`Delay`-typed shape cannot support without
/// boxing. Reach for that pipe-tier `Clock` when a combinator needs to AWAIT
/// a sleep (`Retry`/`RateLimit`/`Delay`); reach for this one to read a
/// timestamp only. `prime::core::timer::Clock` is a third, lower-level seam
/// again — an abstract, resolution-agnostic tick source for `TimerWheel`,
/// deliberately NOT pinned to nanoseconds (its production impl reads
/// milliseconds). Three traits, three incompatible shapes — see each
/// definition site for why none of them collapses into another.
pub trait Clock {
    fn now_ns(&self) -> u64;
}

/// Atomic counter clock — always-ascending, no platform syscall.
///
/// Suitable for tests and no_std environments where wall-clock precision is
/// irrelevant; each call adds `step` to the last value. [`MonotonicCounter::new`]
/// defaults `step` to 1; use [`MonotonicCounter::with_step`] when a test needs
/// a guaranteed delta between two reads larger than some threshold (e.g.
/// proving a span's duration exceeds a budget the real `SystemClock` might
/// measure as 0ns for a fast span).
pub struct MonotonicCounter {
    value: AtomicU64,
    step: u64,
}

impl MonotonicCounter {
    pub const fn new(start: u64) -> Self {
        Self::with_step(start, 1)
    }

    pub const fn with_step(start: u64, step: u64) -> Self {
        Self {
            value: AtomicU64::new(start),
            step,
        }
    }
}

impl Clock for MonotonicCounter {
    fn now_ns(&self) -> u64 {
        self.value.fetch_add(self.step, Ordering::Relaxed)
    }
}

impl Clock for alloc::sync::Arc<dyn Clock + Send + Sync> {
    fn now_ns(&self) -> u64 {
        self.as_ref().now_ns()
    }
}
