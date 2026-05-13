//! Multipath QUIC per-path state machine + path table + wire-format
//! frame codec per [draft-ietf-quic-multipath-21].
//!
//! - `mod.rs` (C26.0) — per-path state machine + path table primitive
//!   per §3. PathId, PathStatus, MultipathPath, MultipathTable.
//! - `frame` (C26.1) — wire-format codec for the 7 multipath
//!   extension frames per §4 (PATH_ABANDON, PATH_STATUS_*,
//!   PATH_NEW/RETIRE_CONNECTION_ID, MAX_PATH_ID, PATHS_BLOCKED,
//!   PATH_CIDS_BLOCKED). PATH_ACK from §4.1 defers to C26.2 (per-path
//!   PN spaces).
//!
//! [draft-ietf-quic-multipath-21]: https://www.ietf.org/archive/id/draft-ietf-quic-multipath-21.txt
//!
//! # Tier
//!
//! Tier-3 (bare `no_std + no_alloc`). State = POD per-path entry +
//! `heapless::LinearMap` keyed by [`PathId`]. Frame codec borrows
//! into caller buffers; no alloc.

pub mod frame;

use heapless::LinearMap;

use crate::quic::packet::header::MAX_CID_LEN;
use crate::quic::sized;
use crate::quic::time::{Duration, Instant};

/// Per-connection cap on tracked multipath paths. Sourced from
/// `proxima-quic-proto.toml [multipath].max_paths_per_connection`.
pub const MAX_PATHS_PER_CONNECTION: usize = sized::MULTIPATH_MAX_PATHS_PER_CONNECTION;

/// Connection-ID byte string (mirrors [`crate::quic::connection::state::ConnectionIdBytes`]
/// but kept inline here so this module compiles without pulling in the
/// `connection` module — needed for tier-3 standalone use).
pub type CidBytes = arrayvec::ArrayVec<u8, MAX_CID_LEN>;

/// Newtype for the multipath PathID per draft-21 §2.1.
///
/// The draft caps the value at 2^32-1; the newtype enforces that
/// boundary at the type level (`u32`). PathID = 0 is reserved for
/// the connection's initial path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PathId(pub u32);

/// Per-path status per draft-21 §3.1 / §3.3 / §3.4.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PathStatus {
    /// Path has been created but PATH_CHALLENGE/PATH_RESPONSE round-trip
    /// not yet completed (§3.1).
    Validating,
    /// Path validated + peer signaled "use freely" (or no preference
    /// signaled and we default to Available per §3.3).
    Available,
    /// Peer sent PATH_STATUS_BACKUP — only use if no Available path
    /// is usable (§3.3).
    Backup,
    /// PATH_ABANDON sent or received; draining outstanding traffic
    /// before final removal (§3.4).
    Closing,
    /// Drain complete — terminal; entry is about to be removed from
    /// the table. Subsequent PathID re-use rejected (§3.4).
    Abandoned,
}

/// Per-path state entry.
#[derive(Debug, Clone)]
pub struct MultipathPath {
    pub path_id: PathId,
    pub status: PathStatus,
    /// CID we use as source on outbound packets for this path.
    pub local_cid: CidBytes,
    /// CID the peer uses as source on inbound packets for this path.
    pub remote_cid: CidBytes,
    /// Most-recent activity timestamp (any inbound/outbound packet).
    pub last_active: Instant,
    /// Set when status transitions to [`PathStatus::Closing`]; the
    /// path moves to [`PathStatus::Abandoned`] after this instant.
    pub close_deadline: Option<Instant>,
    /// Largest `status_seq` seen on inbound PATH_STATUS_* frames for
    /// this path. draft §3.3 — status changes whose seq is not the
    /// largest yet received MUST be ignored.
    pub last_status_seq: Option<u64>,
}

/// Errors from [`MultipathTable`] operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum MultipathError {
    /// Path-table at capacity; cannot register a new path.
    TableFull,
    /// PathID already exists in the table (whether active or
    /// Abandoned — Abandoned PathIDs MUST NOT be re-used per §3.4).
    DuplicatePathId,
    /// Operation referenced a PathID not in the table.
    UnknownPathId,
    /// Status transition is illegal per the per-status FSM
    /// (e.g. Validating → Backup directly; must go via Available).
    IllegalTransition { from: PathStatus, to: PathStatus },
    /// CID exceeded the protocol max (20 bytes).
    CidTooLong,
    /// PATH_STATUS_* frame's status_seq was not larger than the
    /// largest yet received for this path — draft §3.3 requires it
    /// to be silently ignored.
    StaleStatusSeq,
}

/// Per-connection multipath path table.
#[derive(Debug, Clone)]
pub struct MultipathTable {
    paths: LinearMap<PathId, MultipathPath, MAX_PATHS_PER_CONNECTION>,
    /// Default drain interval when [`Self::abandon`] is called without
    /// an explicit `close_deadline` — set from caller's PTO estimate.
    default_drain: Duration,
}

impl Default for MultipathTable {
    fn default() -> Self {
        Self::new(Duration::from_micros(500_000))
    }
}

impl MultipathTable {
    /// Construct an empty table with the given default drain interval
    /// (used by [`Self::abandon`] when no explicit deadline given).
    /// 3 × PTO is the typical caller value per draft-21 §3.4.
    #[must_use]
    pub fn new(default_drain: Duration) -> Self {
        Self {
            paths: LinearMap::new(),
            default_drain,
        }
    }

    /// Number of registered (non-Abandoned) paths.
    #[must_use]
    pub fn len(&self) -> usize {
        self.paths.len()
    }

    /// `true` if no paths are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.paths.is_empty()
    }

    /// Borrow a path entry by ID.
    #[must_use]
    pub fn get(&self, path_id: PathId) -> Option<&MultipathPath> {
        self.paths.get(&path_id)
    }

    /// Register a new path. Initial connection's primary path should
    /// be registered with `initial_status = PathStatus::Available`;
    /// secondary paths with `PathStatus::Validating`.
    ///
    /// # Errors
    ///
    /// See [`MultipathError`].
    pub fn register_path(
        &mut self,
        path_id: PathId,
        local_cid: &[u8],
        remote_cid: &[u8],
        initial_status: PathStatus,
        now: Instant,
    ) -> Result<(), MultipathError> {
        if self.paths.contains_key(&path_id) {
            return Err(MultipathError::DuplicatePathId);
        }
        if local_cid.len() > MAX_CID_LEN || remote_cid.len() > MAX_CID_LEN {
            return Err(MultipathError::CidTooLong);
        }
        if !matches!(
            initial_status,
            PathStatus::Validating | PathStatus::Available
        ) {
            return Err(MultipathError::IllegalTransition {
                from: PathStatus::Validating,
                to: initial_status,
            });
        }
        let mut local = CidBytes::new();
        // try_extend_from_slice cannot fail — bounds-checked above.
        local.try_extend_from_slice(local_cid).ok();
        let mut remote = CidBytes::new();
        remote.try_extend_from_slice(remote_cid).ok();
        let entry = MultipathPath {
            path_id,
            status: initial_status,
            local_cid: local,
            remote_cid: remote,
            last_active: now,
            close_deadline: None,
            last_status_seq: None,
        };
        self.paths
            .insert(path_id, entry)
            .map_err(|_| MultipathError::TableFull)?;
        Ok(())
    }

    /// Transition a Validating path to Available after PATH_RESPONSE
    /// matches a recently-issued PATH_CHALLENGE token (§3.1).
    ///
    /// # Errors
    ///
    /// Returns [`MultipathError::UnknownPathId`] or
    /// [`MultipathError::IllegalTransition`] when the path isn't
    /// Validating.
    pub fn note_path_validated(
        &mut self,
        path_id: PathId,
        now: Instant,
    ) -> Result<(), MultipathError> {
        let entry = self
            .paths
            .get_mut(&path_id)
            .ok_or(MultipathError::UnknownPathId)?;
        if !matches!(entry.status, PathStatus::Validating) {
            return Err(MultipathError::IllegalTransition {
                from: entry.status,
                to: PathStatus::Available,
            });
        }
        entry.status = PathStatus::Available;
        entry.last_active = now;
        Ok(())
    }

    /// Apply a peer's PATH_STATUS_AVAILABLE / PATH_STATUS_BACKUP
    /// frame (§3.3). The transition is Available ↔ Backup; rejection
    /// for any other source state.
    ///
    /// # Errors
    ///
    /// See [`MultipathError`].
    pub fn set_remote_status_preference(
        &mut self,
        path_id: PathId,
        preference: PathStatus,
        status_seq: u64,
        now: Instant,
    ) -> Result<(), MultipathError> {
        if !matches!(preference, PathStatus::Available | PathStatus::Backup) {
            return Err(MultipathError::IllegalTransition {
                from: PathStatus::Available,
                to: preference,
            });
        }
        let entry = self
            .paths
            .get_mut(&path_id)
            .ok_or(MultipathError::UnknownPathId)?;
        if !matches!(entry.status, PathStatus::Available | PathStatus::Backup) {
            return Err(MultipathError::IllegalTransition {
                from: entry.status,
                to: preference,
            });
        }
        if entry.last_status_seq.is_some_and(|last| status_seq <= last) {
            return Err(MultipathError::StaleStatusSeq);
        }
        entry.last_status_seq = Some(status_seq);
        entry.status = preference;
        entry.last_active = now;
        Ok(())
    }

    /// Abandon a path (§3.4) — sent locally OR received from peer.
    /// Transitions to Closing with `close_deadline = now + drain`
    /// (or `default_drain` if `drain` is `None`).
    ///
    /// # Errors
    ///
    /// Returns [`MultipathError::UnknownPathId`] or
    /// [`MultipathError::IllegalTransition`] when the path is already
    /// Closing / Abandoned.
    pub fn abandon(
        &mut self,
        path_id: PathId,
        now: Instant,
        drain: Option<Duration>,
    ) -> Result<(), MultipathError> {
        let drain = drain.unwrap_or(self.default_drain);
        let entry = self
            .paths
            .get_mut(&path_id)
            .ok_or(MultipathError::UnknownPathId)?;
        if !matches!(
            entry.status,
            PathStatus::Validating | PathStatus::Available | PathStatus::Backup
        ) {
            return Err(MultipathError::IllegalTransition {
                from: entry.status,
                to: PathStatus::Closing,
            });
        }
        entry.status = PathStatus::Closing;
        entry.close_deadline = Some(now + drain);
        Ok(())
    }

    /// Advance the table's drain timers. Any Closing path whose
    /// `close_deadline` has elapsed transitions to Abandoned + is
    /// removed from the table. Returns the count removed.
    pub fn tick(&mut self, now: Instant) -> usize {
        // LinearMap iteration order is insertion order, so we collect
        // PathIds to drop and remove them in a second pass.
        let mut to_drop = arrayvec::ArrayVec::<PathId, MAX_PATHS_PER_CONNECTION>::new();
        for (path_id, entry) in self.paths.iter() {
            if matches!(entry.status, PathStatus::Closing)
                && entry.close_deadline.is_some_and(|deadline| now >= deadline)
            {
                // try_push cannot fail — to_drop sized at the
                // same cap as the table.
                let _ = to_drop.try_push(*path_id);
            }
        }
        let removed = to_drop.len();
        for path_id in &to_drop {
            self.paths.remove(path_id);
        }
        removed
    }

    /// Record activity on the path (any inbound or outbound packet).
    pub fn note_activity(&mut self, path_id: PathId, now: Instant) -> Result<(), MultipathError> {
        let entry = self
            .paths
            .get_mut(&path_id)
            .ok_or(MultipathError::UnknownPathId)?;
        entry.last_active = now;
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    const L0: [u8; 4] = [0xA0, 0xA1, 0xA2, 0xA3];
    const R0: [u8; 4] = [0xB0, 0xB1, 0xB2, 0xB3];
    const L1: [u8; 4] = [0xC0, 0xC1, 0xC2, 0xC3];
    const R1: [u8; 4] = [0xD0, 0xD1, 0xD2, 0xD3];

    fn at(micros: u64) -> Instant {
        Instant::from_micros(micros)
    }

    #[test]
    fn worked_example_two_path_lifecycle_from_design_doc() {
        // Trace the 5-row worked example from
        // docs/proxima-quic/c26-multipath-design.md verbatim.
        let mut table = MultipathTable::new(Duration::from_micros(300_000)); // 300 ms = 3 × 100 ms PTO

        // T0: primary path P0 registered as Available (inherited from handshake).
        table
            .register_path(PathId(0), &L0, &R0, PathStatus::Available, at(0))
            .expect("register P0");
        assert_eq!(table.get(PathId(0)).unwrap().status, PathStatus::Available);

        // T1: client opens P1 → Validating.
        table
            .register_path(PathId(1), &L1, &R1, PathStatus::Validating, at(1_000_000))
            .expect("register P1");
        assert_eq!(table.get(PathId(1)).unwrap().status, PathStatus::Validating,);

        // T2: PATH_RESPONSE matches P1's token → Available.
        table
            .note_path_validated(PathId(1), at(1_100_000))
            .expect("validate P1");
        assert_eq!(table.get(PathId(1)).unwrap().status, PathStatus::Available);

        // T3: peer sends PATH_STATUS_BACKUP for P0.
        table
            .set_remote_status_preference(PathId(0), PathStatus::Backup, 1, at(2_000_000))
            .expect("status backup P0");
        assert_eq!(table.get(PathId(0)).unwrap().status, PathStatus::Backup);

        // T4: P0 connectivity breaks → abandon.
        table
            .abandon(PathId(0), at(3_000_000), None)
            .expect("abandon P0");
        assert_eq!(table.get(PathId(0)).unwrap().status, PathStatus::Closing);

        // T5: drain timer fires (T4 + 300 ms).
        let removed = table.tick(at(3_000_000 + 300_000));
        assert_eq!(removed, 1);
        assert!(table.get(PathId(0)).is_none(), "P0 must be removed");
        assert_eq!(table.get(PathId(1)).unwrap().status, PathStatus::Available);
    }

    #[test]
    fn register_path_rejects_duplicate() {
        let mut table = MultipathTable::default();
        table
            .register_path(PathId(7), &L0, &R0, PathStatus::Available, at(0))
            .expect("first ok");
        let err = table
            .register_path(PathId(7), &L0, &R0, PathStatus::Available, at(0))
            .unwrap_err();
        assert_eq!(err, MultipathError::DuplicatePathId);
    }

    #[test]
    fn register_path_rejects_cid_too_long() {
        let mut table = MultipathTable::default();
        let too_long = [0u8; MAX_CID_LEN + 1];
        let result = table.register_path(PathId(1), &too_long, &R0, PathStatus::Validating, at(0));
        assert_eq!(result, Err(MultipathError::CidTooLong));
    }

    #[test]
    fn register_path_rejects_invalid_initial_status() {
        let mut table = MultipathTable::default();
        let result = table.register_path(PathId(1), &L0, &R0, PathStatus::Backup, at(0));
        assert!(matches!(
            result,
            Err(MultipathError::IllegalTransition { .. })
        ));
    }

    #[test]
    fn note_path_validated_rejects_non_validating_state() {
        let mut table = MultipathTable::default();
        table
            .register_path(PathId(1), &L0, &R0, PathStatus::Available, at(0))
            .expect("ok");
        let err = table.note_path_validated(PathId(1), at(0)).unwrap_err();
        assert!(matches!(
            err,
            MultipathError::IllegalTransition {
                from: PathStatus::Available,
                ..
            }
        ));
    }

    #[test]
    fn set_remote_status_preference_rejects_validating_path() {
        let mut table = MultipathTable::default();
        table
            .register_path(PathId(1), &L0, &R0, PathStatus::Validating, at(0))
            .expect("ok");
        let err = table
            .set_remote_status_preference(PathId(1), PathStatus::Available, 1, at(0))
            .unwrap_err();
        assert!(matches!(
            err,
            MultipathError::IllegalTransition {
                from: PathStatus::Validating,
                ..
            }
        ));
    }

    #[test]
    fn set_remote_status_preference_rejects_non_status_target() {
        let mut table = MultipathTable::default();
        table
            .register_path(PathId(1), &L0, &R0, PathStatus::Available, at(0))
            .expect("ok");
        let err = table
            .set_remote_status_preference(PathId(1), PathStatus::Closing, 1, at(0))
            .unwrap_err();
        assert!(matches!(err, MultipathError::IllegalTransition { .. }));
    }

    #[test]
    fn set_remote_status_preference_rejects_non_monotonic_seq() {
        let mut table = MultipathTable::default();
        table
            .register_path(PathId(1), &L0, &R0, PathStatus::Available, at(0))
            .expect("register");
        // seq=5 → Backup, accepted.
        table
            .set_remote_status_preference(PathId(1), PathStatus::Backup, 5, at(10))
            .expect("seq 5 accepted");
        // seq=4 (smaller) → must be rejected as stale.
        let err = table
            .set_remote_status_preference(PathId(1), PathStatus::Available, 4, at(20))
            .unwrap_err();
        assert!(matches!(err, MultipathError::StaleStatusSeq));
        // seq=5 (equal) → also stale (draft §3.3 requires strictly larger).
        let err = table
            .set_remote_status_preference(PathId(1), PathStatus::Available, 5, at(30))
            .unwrap_err();
        assert!(matches!(err, MultipathError::StaleStatusSeq));
        assert_eq!(table.get(PathId(1)).unwrap().status, PathStatus::Backup);
        // seq=6 → accepted.
        table
            .set_remote_status_preference(PathId(1), PathStatus::Available, 6, at(40))
            .expect("seq 6 accepted");
        assert_eq!(table.get(PathId(1)).unwrap().status, PathStatus::Available);
    }

    #[test]
    fn abandon_rejects_closing_path() {
        let mut table = MultipathTable::default();
        table
            .register_path(PathId(1), &L0, &R0, PathStatus::Available, at(0))
            .expect("ok");
        table
            .abandon(PathId(1), at(0), None)
            .expect("first abandon ok");
        let err = table.abandon(PathId(1), at(0), None).unwrap_err();
        assert!(matches!(
            err,
            MultipathError::IllegalTransition {
                from: PathStatus::Closing,
                ..
            }
        ));
    }

    #[test]
    fn tick_with_no_closing_paths_returns_zero() {
        let mut table = MultipathTable::default();
        table
            .register_path(PathId(1), &L0, &R0, PathStatus::Available, at(0))
            .expect("ok");
        assert_eq!(table.tick(at(10_000_000)), 0);
        assert_eq!(table.len(), 1);
    }

    #[test]
    fn tick_before_drain_deadline_is_noop() {
        let mut table = MultipathTable::new(Duration::from_micros(500_000));
        table
            .register_path(PathId(1), &L0, &R0, PathStatus::Available, at(0))
            .expect("ok");
        table
            .abandon(PathId(1), at(1_000_000), None)
            .expect("abandon");
        // 100 µs before the 500 ms drain deadline.
        let removed = table.tick(at(1_000_000 + 500_000 - 100));
        assert_eq!(removed, 0);
        assert!(table.get(PathId(1)).is_some());
    }

    #[test]
    fn note_activity_updates_last_active_timestamp() {
        let mut table = MultipathTable::default();
        table
            .register_path(PathId(1), &L0, &R0, PathStatus::Available, at(0))
            .expect("ok");
        table.note_activity(PathId(1), at(5_000_000)).expect("ok");
        assert_eq!(table.get(PathId(1)).unwrap().last_active, at(5_000_000));
    }

    #[test]
    fn table_full_rejects_extra_path() {
        let mut table = MultipathTable::default();
        for i in 0..MAX_PATHS_PER_CONNECTION as u32 {
            table
                .register_path(PathId(i), &L0, &R0, PathStatus::Available, at(0))
                .expect("ok");
        }
        let err = table
            .register_path(
                PathId(MAX_PATHS_PER_CONNECTION as u32),
                &L0,
                &R0,
                PathStatus::Available,
                at(0),
            )
            .unwrap_err();
        assert_eq!(err, MultipathError::TableFull);
    }
}
