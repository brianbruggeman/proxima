//! Request-level admission — the per-request twin of [`super::ListenerCore`].
//!
//! `ListenerCore` is deliberately `&mut self`, single-threaded: one accept
//! loop drives it serially. Request admission cannot share that shape — a
//! multiplexed protocol (h2 streams, pgwire messages, redis commands) calls
//! request-level admit/release from inside a connection's own task, which
//! may be running on a different core than the accept loop entirely (see
//! `proxima_listen::dispatch_handler`'s `Route::Peer` spawn). So this is the
//! atomics-backed, freely-`Clone`-and-share sibling: one [`ConnAdmission`]
//! instance is created ONCE per listener and cloned into every accepted
//! connection, threaded into [`crate::any::AnyProtocol::drive`] as
//! `admission: &ConnAdmission`.
//!
//! Every protocol calls `request_admit()`/`request_release()` at its own
//! request boundary (h1 per request, h2 per stream, pgwire per message,
//! redis per command) and renders its own wire-specific rejection on
//! `Shed` — the listener owns the uniform policy (capacity, quiesce,
//! drain); the protocol only reports boundaries and renders the reply.
//! This dissolves the old per-protocol `Arc<AtomicU64>` +
//! `Arc<AtomicBool>` + protocol-specific `QuiesceResponse` triple every
//! caller used to hand-roll and thread separately.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use super::state::ShedReason;

/// Generous default so a listener that never configures a request cap
/// behaves as "unbounded" (matching [`super::ListenerCore::new`]'s default
/// connection capacity of `usize::MAX` on the alloc tier).
const DEFAULT_MAX_IN_FLIGHT_REQUESTS: u64 = u64::MAX;

/// Outcome of [`ConnAdmission::request_admit`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestAdmit {
    /// Proceed: dispatch the request/stream/message/command to the
    /// business handler, then call [`ConnAdmission::request_release`] on
    /// completion.
    Admit,
    /// Do not dispatch. The protocol renders its own wire-specific
    /// rejection (in-band 503 for h1/h2, an error packet for pgwire, `-ERR`
    /// for redis) instead of calling the handler.
    Shed {
        /// Why.
        reason: ShedReason,
    },
}

struct Inner {
    in_flight: Arc<AtomicU64>,
    quiescing: Arc<AtomicBool>,
    max_in_flight: u64,
    draining: AtomicBool,
}

/// Listener-wide request-admission handle. Cheap to `Clone` (one `Arc`
/// bump); every accepted connection gets a clone, threaded into
/// [`crate::any::AnyProtocol::drive`].
#[derive(Clone)]
pub struct ConnAdmission(Arc<Inner>);

impl ConnAdmission {
    /// `max_in_flight` bounds the TOTAL number of concurrently-admitted
    /// requests across every connection this listener owns (not per
    /// connection) — the request-level twin of
    /// [`super::ListenerCore::with_capacity`]'s connection-level bound.
    #[must_use]
    pub fn new(max_in_flight: usize) -> Self {
        Self(Arc::new(Inner {
            in_flight: Arc::new(AtomicU64::new(0)),
            quiescing: Arc::new(AtomicBool::new(false)),
            max_in_flight: u64::try_from(max_in_flight).unwrap_or(DEFAULT_MAX_IN_FLIGHT_REQUESTS),
            draining: AtomicBool::new(false),
        }))
    }

    /// Unbounded (`usize::MAX`) request cap — the default for a listener
    /// that never configures one, mirroring [`super::ListenerCore::new`].
    #[must_use]
    pub fn unbounded() -> Self {
        Self::new(usize::MAX)
    }

    /// Decide whether to admit the next request at this connection's own
    /// request boundary. Draining takes priority over quiescing (a hard
    /// shutdown always wins over a courtesy window); quiescing takes
    /// priority over capacity (a listener told to wind down should not
    /// keep admitting up to its cap in the meantime).
    #[must_use]
    pub fn request_admit(&self) -> RequestAdmit {
        if self.0.draining.load(Ordering::Acquire) {
            return RequestAdmit::Shed {
                reason: ShedReason::Draining,
            };
        }
        if self.0.quiescing.load(Ordering::Acquire) {
            return RequestAdmit::Shed {
                reason: ShedReason::Quiescing,
            };
        }
        if self.0.in_flight.load(Ordering::Acquire) >= self.0.max_in_flight {
            return RequestAdmit::Shed {
                reason: ShedReason::AtCapacity,
            };
        }
        self.0.in_flight.fetch_add(1, Ordering::AcqRel);
        RequestAdmit::Admit
    }

    /// Release a request admitted by [`Self::request_admit`]. Every
    /// `Admit` must be paired with exactly one `request_release` call
    /// (success or failure alike) or the in-flight count never drains to
    /// zero and shutdown blocks until its timeout.
    pub fn request_release(&self) {
        self.0.in_flight.fetch_sub(1, Ordering::AcqRel);
    }

    /// Begin the courtesy quiesce window: new requests are shed
    /// (`ShedReason::Quiescing`) but connections stay open and already
    /// in-flight requests complete normally.
    pub fn begin_quiesce(&self) {
        self.0.quiescing.store(true, Ordering::Release);
    }

    /// Begin hard drain: new requests are shed (`ShedReason::Draining`).
    /// Idempotent with [`Self::begin_quiesce`] — draining always takes
    /// priority in [`Self::request_admit`] regardless of call order.
    pub fn begin_drain(&self) {
        self.0.draining.store(true, Ordering::Release);
    }

    /// Current count of admitted-but-not-yet-released requests. Polled by
    /// the listener's shutdown path to decide when the request-level half
    /// of a graceful drain is done.
    #[must_use]
    pub fn in_flight(&self) -> u64 {
        self.0.in_flight.load(Ordering::Acquire)
    }

    #[must_use]
    pub fn is_quiescing(&self) -> bool {
        self.0.quiescing.load(Ordering::Acquire)
    }

    #[must_use]
    pub fn is_draining(&self) -> bool {
        self.0.draining.load(Ordering::Acquire)
    }

    /// Bridge for protocols whose existing per-request loop already takes
    /// these exact atomics positionally (h1's `serve_connection`) — a
    /// clone of the SAME shared counter `request_admit`/`request_release`
    /// operate on, not a fresh one, so driving the legacy signature
    /// through this handle keeps the listener-wide count accurate. New
    /// integrations should call `request_admit`/`request_release`
    /// directly instead of reaching for this.
    #[must_use]
    pub fn in_flight_counter(&self) -> Arc<AtomicU64> {
        self.0.in_flight.clone()
    }

    /// Bridge for protocols whose existing loop already takes a bare
    /// `Arc<AtomicBool>` quiesce flag positionally (h1's
    /// `serve_connection`). See [`Self::in_flight_counter`]'s doc.
    #[must_use]
    pub fn quiescing_flag(&self) -> Arc<AtomicBool> {
        self.0.quiescing.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn admits_until_capacity_then_sheds_at_capacity() {
        let admission = ConnAdmission::new(2);
        assert_eq!(admission.request_admit(), RequestAdmit::Admit);
        assert_eq!(admission.request_admit(), RequestAdmit::Admit);
        assert_eq!(
            admission.request_admit(),
            RequestAdmit::Shed {
                reason: ShedReason::AtCapacity
            }
        );
        assert_eq!(admission.in_flight(), 2);
    }

    #[test]
    fn release_frees_a_capacity_slot() {
        let admission = ConnAdmission::new(1);
        assert_eq!(admission.request_admit(), RequestAdmit::Admit);
        assert_eq!(
            admission.request_admit(),
            RequestAdmit::Shed {
                reason: ShedReason::AtCapacity
            }
        );
        admission.request_release();
        assert_eq!(admission.in_flight(), 0);
        assert_eq!(admission.request_admit(), RequestAdmit::Admit);
    }

    #[test]
    fn quiesce_sheds_new_requests_without_affecting_in_flight_count() {
        let admission = ConnAdmission::unbounded();
        assert_eq!(admission.request_admit(), RequestAdmit::Admit);
        admission.begin_quiesce();
        assert_eq!(
            admission.request_admit(),
            RequestAdmit::Shed {
                reason: ShedReason::Quiescing
            }
        );
        assert_eq!(admission.in_flight(), 1, "the earlier admit is unaffected");
    }

    #[test]
    fn drain_takes_priority_over_quiesce() {
        let admission = ConnAdmission::unbounded();
        admission.begin_quiesce();
        admission.begin_drain();
        assert_eq!(
            admission.request_admit(),
            RequestAdmit::Shed {
                reason: ShedReason::Draining
            }
        );
    }

    #[test]
    fn clone_shares_the_same_underlying_counters() {
        let admission = ConnAdmission::new(1);
        let cloned = admission.clone();
        assert_eq!(admission.request_admit(), RequestAdmit::Admit);
        assert_eq!(
            cloned.request_admit(),
            RequestAdmit::Shed {
                reason: ShedReason::AtCapacity
            },
            "a clone observes the same shared in-flight counter"
        );
    }

    #[test]
    fn legacy_atomics_bridge_shares_state_with_request_admit() {
        let admission = ConnAdmission::new(2);
        let counter = admission.in_flight_counter();
        counter.fetch_add(1, Ordering::Relaxed);
        assert_eq!(
            admission.in_flight(),
            1,
            "the bridged Arc<AtomicU64> is the SAME counter, not a copy"
        );
        let flag = admission.quiescing_flag();
        flag.store(true, Ordering::Relaxed);
        assert!(admission.is_quiescing());
    }
}
