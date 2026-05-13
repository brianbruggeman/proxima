//! Scoped current span: the `(TraceId, SpanId)` of the span presently entered
//! on this executor thread, so a log/metric emitted inside a span correlates to
//! it without the caller threading the id by hand.
//!
//! Storage is a single `Cell<Option<(TraceId, SpanId)>>`, not a container: on
//! `enter` the caller gets back the parent it displaced and holds it (in the
//! `SpanGuard` for a sync scope, or in a poll-local guard inside
//! [`Spanned::poll`](crate::spanned::Spanned)), then `restore`s it when the scope
//! or poll ends. Nesting composes because each scope's parent lives in its own
//! frame — a LIFO stack distributed across the call/poll frames, with no heap and
//! no per-op container bookkeeping.
//!
//! Sync spans bracket it over their synchronous scope (`SpanGuard::enter` /
//! `Drop`). Async spans bracket it PER POLL — entered just before polling the
//! inner future, restored just after — so between polls (across an `.await`) the
//! current span is NOT this task's, and two tasks interleaving on one executor
//! thread never see each other's span.
//!
//! std-only storage: it is thread-scoped, which is the executor thread on
//! proxima's per-core shared-nothing runtime. On no_std there is no thread-local,
//! so the stubs make `current` an unconditional `None` — correlation degrades to
//! explicit ids, never to a wrong id. (The `Cell` itself needs no `alloc`; only
//! the thread-local keys it std.)

use crate::id::{SpanId, TraceId};

#[cfg(feature = "std")]
mod imp {
    use core::cell::Cell;

    use super::{SpanId, TraceId};

    std::thread_local! {
        // const-initialised so `.with` is the fast TLS path — no lazy-init branch.
        static CURRENT: Cell<Option<(TraceId, SpanId)>> = const { Cell::new(None) };
    }

    // The verbose-buffered bit for the current trace, kept in lockstep with
    // CURRENT so the log macro's below-floor admit branch reads it with a single
    // `Cell::get` — no per-record sampler recompute, no per-record map lookup.
    // Feature-gated so a non-elevation build carries no extra TLS and `enter`/
    // `restore` are byte-identical to today.
    #[cfg(feature = "elevation")]
    std::thread_local! {
        static CURRENT_VERBOSE: Cell<bool> = const { Cell::new(false) };
    }

    #[cfg(feature = "elevation")]
    use core::sync::atomic::{AtomicU8, AtomicU64, Ordering};

    // 0 = elevation off / no trace verbose. Set once at elevation install from the
    // sample ratio; read per span-enter (not per record).
    #[cfg(feature = "elevation")]
    static VERBOSE_THRESHOLD: AtomicU64 = AtomicU64::new(0);

    // The severity floor down to which below-gate records are admitted for a
    // verbose trace (the elevation `elevated` depth). `u8::MAX` admits nothing
    // until install sets the real floor.
    #[cfg(feature = "elevation")]
    static VERBOSE_ADMIT_FLOOR: AtomicU8 = AtomicU8::new(u8::MAX);

    #[cfg(feature = "elevation")]
    pub fn set_verbose_ratio(ratio: f64) {
        // same threshold math as sampler::TraceIdRatioBased, so verbose sampling
        // matches the OTel ratio semantics.
        let clamped = ratio.clamp(0.0, 1.0);
        let threshold = (clamped * u64::MAX as f64) as u64;
        VERBOSE_THRESHOLD.store(threshold, Ordering::Relaxed);
    }

    #[cfg(feature = "elevation")]
    pub fn is_verbose_trace(trace: TraceId) -> bool {
        let threshold = VERBOSE_THRESHOLD.load(Ordering::Relaxed);
        if threshold == 0 {
            return false;
        }
        let bytes = trace.to_bytes();
        let value = u64::from_le_bytes([
            bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15],
        ]);
        value < threshold
    }

    #[cfg(feature = "elevation")]
    fn set_verbose_for(context: Option<(TraceId, SpanId)>) {
        let verbose = context.is_some_and(|(trace, _)| is_verbose_trace(trace));
        CURRENT_VERBOSE.with(|cell| cell.set(verbose));
    }

    #[cfg(feature = "elevation")]
    pub fn is_current_verbose() -> bool {
        CURRENT_VERBOSE.with(Cell::get)
    }

    #[cfg(feature = "elevation")]
    pub fn set_verbose_admit_floor(severity: u8) {
        VERBOSE_ADMIT_FLOOR.store(severity, Ordering::Relaxed);
    }

    // the log macro's below-floor admit test: in a verbose trace AND at/above the
    // elevated replay depth. One `Cell::get` + one atomic load; no map, no sampler.
    #[cfg(feature = "elevation")]
    pub fn should_admit_below_floor(severity: u8) -> bool {
        is_current_verbose() && severity >= VERBOSE_ADMIT_FLOOR.load(Ordering::Relaxed)
    }

    pub fn enter(trace: TraceId, span: SpanId) -> Option<(TraceId, SpanId)> {
        #[cfg(feature = "elevation")]
        set_verbose_for(Some((trace, span)));
        CURRENT.with(|cell| cell.replace(Some((trace, span))))
    }

    pub fn restore(parent: Option<(TraceId, SpanId)>) {
        #[cfg(feature = "elevation")]
        set_verbose_for(parent);
        CURRENT.with(|cell| cell.set(parent));
    }

    pub fn current() -> Option<(TraceId, SpanId)> {
        CURRENT.with(Cell::get)
    }
}

#[cfg(not(feature = "std"))]
mod imp {
    use super::{SpanId, TraceId};

    pub fn enter(_trace: TraceId, _span: SpanId) -> Option<(TraceId, SpanId)> {
        None
    }

    pub fn restore(_parent: Option<(TraceId, SpanId)>) {}

    pub fn current() -> Option<(TraceId, SpanId)> {
        None
    }
}

/// Enter `(trace, span)` as the current span, returning the parent it displaced
/// so the caller can [`restore`] it when the scope or poll ends.
pub fn enter(trace: TraceId, span: SpanId) -> Option<(TraceId, SpanId)> {
    imp::enter(trace, span)
}

/// Restore the parent captured by [`enter`].
pub fn restore(parent: Option<(TraceId, SpanId)>) {
    imp::restore(parent);
}

/// The span presently entered on this thread, if any.
#[must_use]
pub fn current() -> Option<(TraceId, SpanId)> {
    imp::current()
}

/// Set the elevation verbose-sampling ratio (fraction of traces admitted to
/// verbose-buffered mode). Called once when the `elevation` policy is installed;
/// `0.0` disables it. Deterministic on `trace_id`, matching
/// [`crate::sampler::TraceIdRatioBased`].
#[cfg(feature = "elevation")]
pub fn set_verbose_ratio(ratio: f64) {
    imp::set_verbose_ratio(ratio);
}

/// Whether `trace` is in the verbose-sampled fraction. Decided once per trace at
/// span-enter and cached in the current-span context; the log macro reads the
/// cache, not this.
#[cfg(feature = "elevation")]
#[must_use]
pub fn is_verbose_trace(trace: TraceId) -> bool {
    imp::is_verbose_trace(trace)
}

/// Whether the current trace is verbose-buffered — the cached bit the log
/// macro's below-floor admit branch reads (one `Cell::get`).
#[cfg(feature = "elevation")]
#[must_use]
pub fn is_current_verbose() -> bool {
    imp::is_current_verbose()
}

/// Set the elevated replay depth: below-gate records at or above this severity
/// are admitted for verbose traces. Called once at elevation install.
#[cfg(feature = "elevation")]
pub fn set_verbose_admit_floor(elevated: crate::level::Level) {
    imp::set_verbose_admit_floor(elevated.severity());
}

/// The log macro's admit test — see [`imp::should_admit_below_floor`]. Taking a
/// [`Level`](crate::level::Level) keeps the macro call site level-typed.
#[cfg(feature = "elevation")]
#[must_use]
pub fn should_admit_below_floor(level: crate::level::Level) -> bool {
    imp::should_admit_below_floor(level.severity())
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::{current, enter, restore};
    use crate::id::{SpanId, TraceId};

    fn ids(byte: u8) -> (TraceId, SpanId) {
        (TraceId::from_bytes([byte; 16]), SpanId::from_bytes([byte; 8]))
    }

    // enter displaces the current span and hands back the parent; restore puts it
    // back — the LIFO nesting lives in the returned parents, not a container.
    #[test]
    fn enter_restore_nests() {
        assert_eq!(current(), None, "empty: no current span");

        let (outer_trace, outer_span) = ids(1);
        let (inner_trace, inner_span) = ids(2);

        let root_parent = enter(outer_trace, outer_span);
        assert_eq!(root_parent, None, "outer displaced no parent");
        assert_eq!(current(), Some((outer_trace, outer_span)));

        let outer_parent = enter(inner_trace, inner_span);
        assert_eq!(outer_parent, Some((outer_trace, outer_span)), "inner displaced outer");
        assert_eq!(current(), Some((inner_trace, inner_span)));

        restore(outer_parent);
        assert_eq!(current(), Some((outer_trace, outer_span)), "restore brings outer back");

        restore(root_parent);
        assert_eq!(current(), None, "restore empties back to no current span");
    }

    // restore(None) on an empty cell is a no-op — a defensive floor for an
    // unbalanced drop.
    #[test]
    fn restore_none_is_noop() {
        assert_eq!(current(), None);
        restore(None);
        assert_eq!(current(), None);
    }

    // the verbose bit tracks the current trace in lockstep with the span cell:
    // ratio=1.0 makes every trace verbose; restoring to None clears it.
    #[cfg(feature = "elevation")]
    #[test]
    fn verbose_bit_follows_current_trace() {
        use super::{is_current_verbose, set_verbose_ratio};

        set_verbose_ratio(1.0);
        let (trace, span) = ids(7);
        assert!(!is_current_verbose(), "no current trace: not verbose");
        let parent = enter(trace, span);
        assert!(is_current_verbose(), "inside a verbose-sampled trace");
        restore(parent);
        assert!(!is_current_verbose(), "restored to no trace: cleared");

        // ratio=0.0 admits nothing.
        set_verbose_ratio(0.0);
        let parent = enter(trace, span);
        assert!(!is_current_verbose(), "ratio 0 admits no trace");
        restore(parent);
    }
}
