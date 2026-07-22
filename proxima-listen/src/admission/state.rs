//! The listener admission state machine and its decision types.

use core::net::IpAddr;

#[cfg(not(feature = "alloc"))]
use heapless::index_map::FnvIndexMap;

use super::sized;

/// Caller-opaque per-connection handle, minted by the core on admit. The
/// adapter stores whatever it needs against it (the spawned task, the socket);
/// the core only uses it to track liveness and release.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ConnectionHandle(pub u32);

/// Where an admitted connection should be driven. Runtime-agnostic — the
/// adapter maps [`Route::Inline`] onto its current core and [`Route::Peer`]
/// onto a spawn to the named peer core index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Route {
    /// Drive on the accepting core (the reactor's current thread).
    Inline,
    /// Drive on peer core `index`.
    Peer(u16),
}

/// How the core routes each admitted connection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DispatchPolicy {
    /// Every connection runs inline on the accepting core.
    Inline,
    /// Round-robin across peer cores `1..num_cores`, reserving core 0 for the
    /// acceptor. `num_cores <= 1` collapses to core 0.
    SpreadToPeers {
        /// Total cores including the reserved acceptor core 0.
        num_cores: u16,
    },
}

impl DispatchPolicy {
    /// The reserve-core-0 round-robin decision, given an external
    /// monotonically increasing `cursor`. This is the SOLE implementation of
    /// that decision: [`ListenerCore::admit`] calls it with its own internal
    /// cursor; a caller that only wants routing (no admission/capacity/drain
    /// tracking) calls it directly with a cursor it owns, so there is never a
    /// second, re-derived copy of this logic.
    #[must_use]
    pub fn route(&self, cursor: &mut usize) -> Route {
        match *self {
            DispatchPolicy::Inline => Route::Inline,
            DispatchPolicy::SpreadToPeers { num_cores } => {
                if num_cores <= 1 {
                    return Route::Peer(0);
                }
                let slot = *cursor;
                *cursor = cursor.wrapping_add(1);
                Route::Peer(1 + (slot % (num_cores as usize - 1)) as u16)
            }
        }
    }
}

/// Why a connection was shed rather than admitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ShedReason {
    /// The live-connection table is full (bounded tier at cap, or an explicit
    /// `alloc`-tier capacity reached).
    AtCapacity,
    /// The listener is draining or closed; it no longer admits.
    Draining,
    /// This peer already has `per_peer_cap` live connections — the DoS knob.
    /// Shed even when the listener's global capacity has room left.
    PerPeerLimit,
    /// Request-level only (`ConnAdmission::request_admit`, the std-tier
    /// request-admission sibling of this connection-level core): the
    /// listener is in its courtesy quiesce window — new requests are shed,
    /// but (unlike [`ShedReason::Draining`]) connections stay open and
    /// already-admitted requests complete normally.
    Quiescing,
    /// Accept-edge only (`proxima_http::any_listener::AnyListenProtocol`'s
    /// blacklist gate, std-tier): this peer tripped a DoS-blacklist strike
    /// threshold (a `DenySignature` match, or enough unclassifiable
    /// rejects) and is still within its ban window — see
    /// `crate::admission::blacklist::BlacklistTable`. Shed BEFORE
    /// `ListenerCore::admit` is ever called, so a banned peer never commits
    /// a table slot.
    Blacklisted,
}

/// The decision [`ListenerCore::admit`] returns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Admission {
    /// Admitted: drive the connection on `route`, release it later with
    /// `handle`.
    Admit {
        /// The minted handle to release with on close.
        handle: ConnectionHandle,
        /// Where to drive it.
        route: Route,
    },
    /// Refused before any resource was committed.
    Shed {
        /// Why.
        reason: ShedReason,
    },
}

/// Lifecycle phase of the listener.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// Admitting new connections.
    Accepting,
    /// Not admitting; waiting for live connections to drain to zero.
    Draining,
    /// Drained and done.
    Closed,
}

/// The outcome of [`ListenerCore::release`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReleaseOutcome {
    /// Released; the listener keeps running.
    Released,
    /// Released the last in-flight connection while draining — now closed.
    ReleasedNowClosed,
    /// The handle was not tracked (double release / never admitted).
    Unknown,
}

/// The outcome of [`ListenerCore::begin_drain`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrainOutcome {
    /// Now draining; waiting on live connections.
    Draining,
    /// Nothing was in flight, so it closed immediately.
    ClosedImmediately,
}

/// Live-connection table: handle -> the peer it came from, so [`release`]
/// knows whose live count to decrement.
///
/// [`release`]: ListenerCore::release
#[cfg(feature = "alloc")]
type Table = hashbrown::HashMap<u32, IpAddr>;
#[cfg(not(feature = "alloc"))]
type Table = FnvIndexMap<u32, IpAddr, { sized::ADMISSION_TABLE_CAP }>;

#[cfg(feature = "alloc")]
type PeerTable = hashbrown::HashMap<IpAddr, u32>;
#[cfg(not(feature = "alloc"))]
type PeerTable = FnvIndexMap<IpAddr, u32, { sized::PEER_TABLE_CAP }>;

/// Accept-layer admission + routing + drain state machine.
///
/// Drive it from a reactor loop: on an accepted connection call [`admit`]
/// with the peer's address; if it returns [`Admission::Admit`], spawn the
/// handler on `route` and keep the `handle`; when the handler finishes call
/// [`release`]; on shutdown call [`begin_drain`] and stop accepting once
/// [`is_closed`] is true.
///
/// [`admit`]: ListenerCore::admit
/// [`release`]: ListenerCore::release
/// [`begin_drain`]: ListenerCore::begin_drain
/// [`is_closed`]: ListenerCore::is_closed
#[derive(Debug)]
pub struct ListenerCore {
    table: Table,
    peer_counts: PeerTable,
    policy: DispatchPolicy,
    capacity: usize,
    per_peer_cap: usize,
    cursor: usize,
    next_handle: u32,
    phase: Phase,
}

impl ListenerCore {
    /// Construct with the default capacity (unbounded on `alloc`,
    /// [`crate::sized::ADMISSION_TABLE_CAP`] on bare `no_std + no_alloc`) and
    /// the default per-peer cap ([`crate::sized::PER_PEER_CAP_DEFAULT`]).
    #[must_use]
    pub fn new(policy: DispatchPolicy) -> Self {
        #[cfg(feature = "alloc")]
        let capacity = usize::MAX;
        #[cfg(not(feature = "alloc"))]
        let capacity = sized::ADMISSION_TABLE_CAP;
        Self::with_caps(policy, capacity, sized::PER_PEER_CAP_DEFAULT)
    }

    /// Construct with an explicit live-connection bound and the default
    /// per-peer cap. On the no-alloc tier the bound is clamped to
    /// [`crate::sized::ADMISSION_TABLE_CAP`] (the backing map's fixed size).
    #[must_use]
    pub fn with_capacity(policy: DispatchPolicy, capacity: usize) -> Self {
        Self::with_caps(policy, capacity, sized::PER_PEER_CAP_DEFAULT)
    }

    /// Construct with an explicit live-connection bound and an explicit
    /// per-peer cap — the full knob set, mainly for tests that need a low
    /// `per_peer_cap` without waiting out the conflag default.
    #[must_use]
    pub fn with_caps(policy: DispatchPolicy, capacity: usize, per_peer_cap: usize) -> Self {
        #[cfg(not(feature = "alloc"))]
        let capacity = capacity.min(sized::ADMISSION_TABLE_CAP);
        Self {
            table: Table::new(),
            peer_counts: PeerTable::new(),
            policy,
            capacity,
            per_peer_cap,
            cursor: 0,
            next_handle: 0,
            phase: Phase::Accepting,
        }
    }

    /// Decide an accepted connection from `peer`: admit + route, or shed.
    /// Commits a table slot (and bumps `peer`'s live count) only on
    /// [`Admission::Admit`].
    pub fn admit(&mut self, peer: IpAddr) -> Admission {
        if self.phase != Phase::Accepting {
            return Admission::Shed {
                reason: ShedReason::Draining,
            };
        }
        if self.table.len() >= self.capacity {
            return Admission::Shed {
                reason: ShedReason::AtCapacity,
            };
        }
        if self.peer_live_count(peer) >= self.per_peer_cap {
            return Admission::Shed {
                reason: ShedReason::PerPeerLimit,
            };
        }
        let handle = self.mint_handle();
        let route = self.next_route();
        self.insert(handle.0, peer);
        self.bump_peer_count(peer);
        Admission::Admit { handle, route }
    }

    /// Release a previously admitted connection, decrementing its peer's
    /// live count. On the last release while draining, transitions to
    /// [`Phase::Closed`].
    pub fn release(&mut self, handle: ConnectionHandle) -> ReleaseOutcome {
        let Some(peer) = self.remove(handle.0) else {
            return ReleaseOutcome::Unknown;
        };
        self.drop_peer_count(peer);
        if self.phase == Phase::Draining && self.table.is_empty() {
            self.phase = Phase::Closed;
            return ReleaseOutcome::ReleasedNowClosed;
        }
        ReleaseOutcome::Released
    }

    /// Stop admitting and begin graceful drain. Closes immediately when
    /// nothing is in flight.
    pub fn begin_drain(&mut self) -> DrainOutcome {
        if self.phase == Phase::Closed {
            return DrainOutcome::ClosedImmediately;
        }
        if self.table.is_empty() {
            self.phase = Phase::Closed;
            return DrainOutcome::ClosedImmediately;
        }
        self.phase = Phase::Draining;
        DrainOutcome::Draining
    }

    /// Current lifecycle phase.
    #[must_use]
    pub fn phase(&self) -> Phase {
        self.phase
    }

    /// Number of live (admitted, not yet released) connections.
    #[must_use]
    pub fn live(&self) -> usize {
        self.table.len()
    }

    /// `true` once drained and done.
    #[must_use]
    pub fn is_closed(&self) -> bool {
        self.phase == Phase::Closed
    }

    fn mint_handle(&mut self) -> ConnectionHandle {
        let handle = ConnectionHandle(self.next_handle);
        self.next_handle = self.next_handle.wrapping_add(1);
        handle
    }

    fn next_route(&mut self) -> Route {
        self.policy.route(&mut self.cursor)
    }

    fn peer_live_count(&self, peer: IpAddr) -> usize {
        let count = self.peer_counts.get(&peer).copied().unwrap_or(0);
        usize::try_from(count).unwrap_or(usize::MAX)
    }

    fn bump_peer_count(&mut self, peer: IpAddr) {
        if let Some(count) = self.peer_counts.get_mut(&peer) {
            *count = count.saturating_add(1);
            return;
        }
        // A fresh peer table slot was checked against `per_peer_cap` (never
        // more entries than admitted connections) in `admit`, so the
        // no-alloc insert cannot overflow; discard the (impossible)
        // full-table Result there.
        #[cfg(feature = "alloc")]
        {
            self.peer_counts.insert(peer, 1);
        }
        #[cfg(not(feature = "alloc"))]
        {
            let _ = self.peer_counts.insert(peer, 1);
        }
    }

    fn drop_peer_count(&mut self, peer: IpAddr) {
        let Some(count) = self.peer_counts.get_mut(&peer) else {
            return;
        };
        if *count > 1 {
            *count -= 1;
            return;
        }
        self.remove_peer(peer);
    }

    fn insert(&mut self, handle: u32, peer: IpAddr) {
        // capacity was checked in `admit`, so the no-alloc insert cannot
        // overflow; discard the (impossible) full-table Result there.
        #[cfg(feature = "alloc")]
        {
            self.table.insert(handle, peer);
        }
        #[cfg(not(feature = "alloc"))]
        {
            let _ = self.table.insert(handle, peer);
        }
    }

    fn remove(&mut self, handle: u32) -> Option<IpAddr> {
        #[cfg(feature = "alloc")]
        {
            self.table.remove(&handle)
        }
        #[cfg(not(feature = "alloc"))]
        {
            self.table.swap_remove(&handle)
        }
    }

    fn remove_peer(&mut self, peer: IpAddr) {
        #[cfg(feature = "alloc")]
        {
            self.peer_counts.remove(&peer);
        }
        #[cfg(not(feature = "alloc"))]
        {
            self.peer_counts.swap_remove(&peer);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PEER_A: IpAddr = IpAddr::V4(core::net::Ipv4Addr::new(127, 0, 0, 1));
    const PEER_B: IpAddr = IpAddr::V4(core::net::Ipv4Addr::new(127, 0, 0, 2));

    #[test]
    fn inline_policy_admits_and_routes_inline() {
        let mut core = ListenerCore::new(DispatchPolicy::Inline);
        let outcome = core.admit(PEER_A);
        assert_eq!(
            outcome,
            Admission::Admit {
                handle: ConnectionHandle(0),
                route: Route::Inline,
            }
        );
        assert_eq!(core.live(), 1);
    }

    #[test]
    fn spread_reserves_core_zero_and_round_robins() {
        let mut core = ListenerCore::new(DispatchPolicy::SpreadToPeers { num_cores: 3 });
        // cores 1,2 alternate; core 0 (the acceptor) is never handed a peer.
        let routes: [Route; 4] = core::array::from_fn(|_| match core.admit(PEER_A) {
            Admission::Admit { route, .. } => route,
            other => panic!("expected admit, got {other:?}"),
        });
        assert_eq!(
            routes,
            [
                Route::Peer(1),
                Route::Peer(2),
                Route::Peer(1),
                Route::Peer(2)
            ]
        );
    }

    #[test]
    fn single_core_spread_collapses_to_core_zero() {
        let mut core = ListenerCore::new(DispatchPolicy::SpreadToPeers { num_cores: 1 });
        match core.admit(PEER_A) {
            Admission::Admit { route, .. } => assert_eq!(route, Route::Peer(0)),
            other => panic!("expected admit, got {other:?}"),
        }
    }

    #[test]
    fn admits_until_capacity_then_sheds() {
        let mut core = ListenerCore::with_capacity(DispatchPolicy::Inline, 2);
        assert!(matches!(core.admit(PEER_A), Admission::Admit { .. }));
        assert!(matches!(core.admit(PEER_B), Admission::Admit { .. }));
        assert_eq!(
            core.admit(PEER_A),
            Admission::Shed {
                reason: ShedReason::AtCapacity,
            }
        );
        assert_eq!(core.live(), 2);
    }

    #[test]
    fn releasing_below_capacity_readmits() {
        let mut core = ListenerCore::with_capacity(DispatchPolicy::Inline, 1);
        let handle = match core.admit(PEER_A) {
            Admission::Admit { handle, .. } => handle,
            other => panic!("expected admit, got {other:?}"),
        };
        assert_eq!(
            core.admit(PEER_A),
            Admission::Shed {
                reason: ShedReason::AtCapacity,
            }
        );
        assert_eq!(core.release(handle), ReleaseOutcome::Released);
        assert!(matches!(core.admit(PEER_A), Admission::Admit { .. }));
    }

    #[test]
    fn releasing_unknown_handle_reports_unknown() {
        let mut core = ListenerCore::new(DispatchPolicy::Inline);
        assert_eq!(core.release(ConnectionHandle(7)), ReleaseOutcome::Unknown);
    }

    #[test]
    fn draining_sheds_new_connections() {
        let mut core = ListenerCore::new(DispatchPolicy::Inline);
        let _held = core.admit(PEER_A);
        assert_eq!(core.begin_drain(), DrainOutcome::Draining);
        assert_eq!(core.phase(), Phase::Draining);
        assert_eq!(
            core.admit(PEER_A),
            Admission::Shed {
                reason: ShedReason::Draining,
            }
        );
    }

    #[test]
    fn last_release_while_draining_closes() {
        let mut core = ListenerCore::new(DispatchPolicy::Inline);
        let first = match core.admit(PEER_A) {
            Admission::Admit { handle, .. } => handle,
            other => panic!("expected admit, got {other:?}"),
        };
        let second = match core.admit(PEER_B) {
            Admission::Admit { handle, .. } => handle,
            other => panic!("expected admit, got {other:?}"),
        };
        core.begin_drain();
        assert_eq!(core.release(first), ReleaseOutcome::Released);
        assert!(!core.is_closed());
        assert_eq!(core.release(second), ReleaseOutcome::ReleasedNowClosed);
        assert!(core.is_closed());
    }

    #[test]
    fn drain_with_nothing_in_flight_closes_immediately() {
        let mut core = ListenerCore::new(DispatchPolicy::Inline);
        assert_eq!(core.begin_drain(), DrainOutcome::ClosedImmediately);
        assert!(core.is_closed());
    }

    #[test]
    fn per_peer_limit_sheds_before_global_capacity() {
        // global capacity is generous; only the one peer's own cap binds.
        let mut core = ListenerCore::with_caps(DispatchPolicy::Inline, 100, 2);
        assert!(matches!(core.admit(PEER_A), Admission::Admit { .. }));
        assert!(matches!(core.admit(PEER_A), Admission::Admit { .. }));
        assert_eq!(
            core.admit(PEER_A),
            Admission::Shed {
                reason: ShedReason::PerPeerLimit,
            }
        );
        assert_eq!(
            core.live(),
            2,
            "shed connection must not commit a table slot"
        );
    }

    #[test]
    fn per_peer_limit_is_independent_per_peer() {
        let mut core = ListenerCore::with_caps(DispatchPolicy::Inline, 100, 1);
        assert!(matches!(core.admit(PEER_A), Admission::Admit { .. }));
        assert_eq!(
            core.admit(PEER_A),
            Admission::Shed {
                reason: ShedReason::PerPeerLimit,
            }
        );
        // a different peer is unaffected by peer A's cap.
        assert!(matches!(core.admit(PEER_B), Admission::Admit { .. }));
    }

    #[test]
    fn release_decrements_peer_count_and_readmits() {
        let mut core = ListenerCore::with_caps(DispatchPolicy::Inline, 100, 1);
        let handle = match core.admit(PEER_A) {
            Admission::Admit { handle, .. } => handle,
            other => panic!("expected admit, got {other:?}"),
        };
        assert_eq!(
            core.admit(PEER_A),
            Admission::Shed {
                reason: ShedReason::PerPeerLimit,
            }
        );
        assert_eq!(core.release(handle), ReleaseOutcome::Released);
        assert!(
            matches!(core.admit(PEER_A), Admission::Admit { .. }),
            "release must decrement the peer's live count, freeing its slot"
        );
    }
}
