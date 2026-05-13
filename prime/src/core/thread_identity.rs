//! Per-thread identity abstraction.
//!
//! Two production sites in `prime/src/core/` reach for thread identity today:
//! [`inbox.rs`](super::inbox) (lane allocation for `try_send_mpsc`) and
//! [`local_executor.rs`](super::local_executor) (deciding whether a waker
//! routes to `local_ready` or `remote_ready`). Both use `std::thread_local!`
//! directly, which makes them std-only and blocks a `#![no_std]` flip of
//! the `prime/src/core/` subtree.
//!
//! This trait gives those sites — and future consumers — a portable handle:
//! the std impl assigns each thread a unique `u64` via `std::thread_local!`,
//! and the no_std stub returns `()` (single-thread by construction).
//!
//! The trait is the prerequisite (gate-row C1 of the no_std + alloc cliff
//! plan, `woolly-watching-cupcake`); the consumer-site refactors that
//! actually retire the existing `thread_local!` declarations live in C2
//! (lane-ticket) and C3 (reactor-direct-wake). This commit only lands the
//! abstraction.

/// Per-thread identity.
///
/// Implementations choose how to assign and read IDs. The trait is zero-cost
/// when monomorphized — concrete impls inline to a TLS read on std and a
/// constant on no_std.
pub trait ThreadIdentity {
    /// Opaque identity. `Copy + Eq` so callers can stash a captured id and
    /// compare cheaply against subsequent `current()` results.
    type Id: Copy + Eq;

    /// Returns the calling thread's id. Stable for the lifetime of the thread.
    fn current() -> Self::Id;

    /// Whether `other` is the calling thread's id. Default impl is
    /// `Self::current() == other`; implementations may override if they have
    /// a faster path.
    #[inline]
    fn is_owning(other: Self::Id) -> bool {
        Self::current() == other
    }
}

/// No_std single-thread stub. All calls share one identity.
///
/// On a single-core MCU or an RTOS task that exclusively owns the runtime,
/// "this thread" is the only thread that exists; identity collapses to a
/// unit value.
pub struct SingleThreadIdentity;

impl ThreadIdentity for SingleThreadIdentity {
    type Id = ();

    #[inline]
    fn current() -> Self::Id {}

    #[inline]
    fn is_owning(_: Self::Id) -> bool {
        true
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn single_thread_stub_returns_unit_identity() {
        let id_a = SingleThreadIdentity::current();
        let id_b = SingleThreadIdentity::current();
        assert_eq!(id_a, id_b);
    }

    #[test]
    fn single_thread_stub_is_owning_always_true() {
        let captured = SingleThreadIdentity::current();
        assert!(SingleThreadIdentity::is_owning(captured));
    }

    #[test]
    fn single_thread_stub_id_type_is_copy_and_eq() {
        fn assert_copy<T: Copy>() {}
        fn assert_eq_bound<T: Eq>() {}
        assert_copy::<<SingleThreadIdentity as ThreadIdentity>::Id>();
        assert_eq_bound::<<SingleThreadIdentity as ThreadIdentity>::Id>();
    }
}
