use alloc::sync::Arc;
use core::sync::atomic::{AtomicU64, Ordering};

#[cfg(feature = "std")]
use crate::recorder::SystemClock;

/// Monotonic timestamp source, injected by the caller.
///
/// Shared by c5-trace (spans) and c8-log (log records). Deliberately
/// object-safe (no associated types): the per-core `Recorder` is generic over
/// its injected clock (`Recorder<Clk>`, held as `Arc<Clk>` — see `recorder`'s
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

// `Recorder<Clk>::builder()`/`RecorderBuilder::start` need `Clk: Default` as the
// no-explicit-clock fallback; the natural default matches `MonotonicCounter::new(0)`.
impl Default for MonotonicCounter {
    fn default() -> Self {
        Self::new(0)
    }
}

/// Forwarding impl for a shared clock handle: any `Arc<C>` reads through to the
/// clock it wraps. Not a blanket impl over an open foreign set — `C` is still
/// bounded by `Clock`, so this only ever forwards, never adapts an unrelated
/// type. This is what lets `Recorder<Clk>` clone `Arc<Clk>` per span/log
/// (`recorder`'s `SpanBuilderWired`/`LogBuilderWired`) while `SpanGuard::enter`
/// takes its clock by value: the clone is a refcount bump over one shared
/// clock, not an independent copy — load-bearing for a clock with shared
/// mutable state (e.g. `MonotonicCounter`'s `AtomicU64`).
impl<C: Clock + ?Sized> Clock for Arc<C> {
    fn now_ns(&self) -> u64 {
        self.as_ref().now_ns()
    }
}

/// The clock behind the process-wide ambient recorder (`export::DEFAULT_RECORDER`).
///
/// A `static` names exactly one type, so the ambient recorder can't stay
/// generic over `Clk` the way a locally-built `Recorder<Clk>` can — this enum
/// is the runtime-switchable stand-in, replacing what would otherwise need a
/// `dyn Clock`. `System` is the default (real wall-clock) arm; `Virtual` lets
/// a test or a deterministic-replay harness install a `MonotonicCounter` as
/// the ambient clock instead. Adding a third arm later (e.g. a caller-supplied
/// clock) is non-breaking — the same shape `Sink::Writer`/`Sink::Handle` use
/// in `export.rs` to keep `Exporter` open — but nothing needs one today, so it
/// stays out (a `dyn`-carrying arm nobody calls is not a "no dyn" migration).
pub enum GlobalClock {
    #[cfg(feature = "std")]
    System(SystemClock),
    Virtual(MonotonicCounter),
}

impl Clock for GlobalClock {
    fn now_ns(&self) -> u64 {
        match self {
            #[cfg(feature = "std")]
            Self::System(clock) => clock.now_ns(),
            Self::Virtual(clock) => clock.now_ns(),
        }
    }
}
