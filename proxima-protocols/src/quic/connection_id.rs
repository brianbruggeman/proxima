//! Connection-ID storage per [RFC 9000 §5.1].
//!
//! Each endpoint maintains two CID sets per connection:
//!
//! - **Local CIDs** — connection IDs we have issued; the peer uses them
//!   as the Destination Connection ID for packets sent to us. New ones
//!   are issued via the [`NEW_CONNECTION_ID`] frame and retired via the
//!   peer's [`RETIRE_CONNECTION_ID`] frame.
//! - **Remote CIDs** — connection IDs the peer has issued; we use them
//!   as the Destination Connection ID for packets sent to the peer.
//!
//! Both sets are bounded by the `active_connection_id_limit` transport
//! parameter (RFC 9000 §18.2, minimum 2, default 2; `prime-runtime.toml`
//! `[quic].active_connection_id_limit` is 4). The exact cap is provided
//! by the caller as a const generic on [`CidQueue`].
//!
//! # Tier
//!
//! Tier-3 (bare `no_std + no_alloc`). Storage is `arrayvec::ArrayVec`
//! with a const-generic capacity; the per-entry CID bytes live in a
//! `[u8; MAX_CID_LEN]` inline buffer (no heap).
//!
//! [RFC 9000 §5.1]: https://www.rfc-editor.org/rfc/rfc9000#section-5.1
//! [`NEW_CONNECTION_ID`]: crate::quic::frame::Frame::NewConnectionId
//! [`RETIRE_CONNECTION_ID`]: crate::quic::frame::Frame::RetireConnectionId

use arrayvec::ArrayVec;

use crate::quic::packet::header::MAX_CID_LEN;

/// Length of the stateless-reset token per RFC 9000 §10.3.
pub const STATELESS_RESET_TOKEN_LEN: usize = 16;

/// One CID-store entry: sequence number + CID bytes + stateless-reset token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CidEntry {
    pub sequence: u64,
    /// CID bytes; length implied by `connection_id_len`.
    connection_id: [u8; MAX_CID_LEN],
    connection_id_len: u8,
    pub stateless_reset_token: [u8; STATELESS_RESET_TOKEN_LEN],
}

impl CidEntry {
    /// Construct an entry from a borrowed CID slice + token.
    ///
    /// # Errors
    ///
    /// Returns [`CidStoreError::CidTooLong`] when `cid.len() > MAX_CID_LEN`.
    pub fn new(
        sequence: u64,
        cid: &[u8],
        stateless_reset_token: [u8; STATELESS_RESET_TOKEN_LEN],
    ) -> Result<Self, CidStoreError> {
        if cid.len() > MAX_CID_LEN {
            return Err(CidStoreError::CidTooLong);
        }
        let mut connection_id = [0u8; MAX_CID_LEN];
        connection_id[..cid.len()].copy_from_slice(cid);
        Ok(Self {
            sequence,
            connection_id,
            connection_id_len: cid.len() as u8,
            stateless_reset_token,
        })
    }

    /// Borrow the active connection-ID slice.
    #[must_use]
    pub fn cid(&self) -> &[u8] {
        &self.connection_id[..usize::from(self.connection_id_len)]
    }

    /// Connection-ID length in bytes (0..=20).
    #[must_use]
    pub fn cid_len(&self) -> usize {
        usize::from(self.connection_id_len)
    }
}

/// Failure modes for [`CidQueue`] operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum CidStoreError {
    /// Insertion would exceed the const-generic capacity. Per RFC 9000
    /// §5.1.1 the peer MUST NOT issue more CIDs than the active limit;
    /// receiving more is a transport-error condition (CONNECTION_ID_LIMIT_ERROR).
    Full,
    /// An entry with this sequence number already exists. RFC 9000 §5.1.1
    /// treats this as a `PROTOCOL_VIOLATION`.
    DuplicateSequence,
    /// The named sequence number is not present.
    NotFound,
    /// CID length exceeded RFC 9000 §17.2 maximum of 20 bytes.
    CidTooLong,
}

/// Bounded queue of active connection-ID entries.
///
/// Entries are kept sorted by sequence number ascending — newly issued
/// CIDs append; retiring removes from anywhere in the middle. Lookup by
/// sequence or by CID bytes is linear (small CAP — typically 2..=8 per
/// `prime-runtime.toml` sizing). For higher caps a sorted-vec binary
/// search would be the natural opt-sweep.
#[derive(Debug, Clone)]
pub struct CidQueue<const CAP: usize> {
    entries: ArrayVec<CidEntry, CAP>,
    /// Lowest sequence number that may still be issued. Tracks
    /// `retire_prior_to` from received NEW_CONNECTION_ID per RFC 9000
    /// §19.15. Helps the connection state machine detect duplicate
    /// retire requests + out-of-order sequence acceptance.
    retire_prior_to: u64,
}

impl<const CAP: usize> Default for CidQueue<CAP> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const CAP: usize> CidQueue<CAP> {
    /// New empty queue.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            entries: ArrayVec::new_const(),
            retire_prior_to: 0,
        }
    }

    /// Number of active entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the queue has zero active entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Whether the queue is at its capacity bound.
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.entries.is_full()
    }

    /// Current `retire_prior_to` watermark. Entries with `sequence`
    /// strictly less than this value have been retired by the peer.
    #[must_use]
    pub fn retire_prior_to(&self) -> u64 {
        self.retire_prior_to
    }

    /// Iterate the active entries in sequence-number order.
    pub fn iter(&self) -> impl Iterator<Item = &CidEntry> {
        self.entries.iter()
    }

    /// Insert an entry. Maintains sequence-number sorted order.
    ///
    /// # Errors
    ///
    /// - [`CidStoreError::Full`] when the queue is at capacity.
    /// - [`CidStoreError::DuplicateSequence`] when the sequence number
    ///   collides with an existing entry.
    pub fn insert(&mut self, entry: CidEntry) -> Result<(), CidStoreError> {
        if self.entries.is_full() {
            return Err(CidStoreError::Full);
        }
        // binary-insert keeps the queue sorted; linear scan is fine for
        // typical small CAP (2..=8). For CAP > 32 a binary_search would
        // be the opt-sweep.
        let position = match self
            .entries
            .iter()
            .position(|existing| existing.sequence == entry.sequence)
        {
            Some(_) => return Err(CidStoreError::DuplicateSequence),
            None => self
                .entries
                .iter()
                .position(|existing| existing.sequence > entry.sequence)
                .unwrap_or(self.entries.len()),
        };
        self.entries.insert(position, entry);
        Ok(())
    }

    /// Remove the entry matching `sequence`. Returns the removed entry.
    ///
    /// # Errors
    ///
    /// [`CidStoreError::NotFound`] if no entry matches.
    pub fn retire(&mut self, sequence: u64) -> Result<CidEntry, CidStoreError> {
        let position = self
            .entries
            .iter()
            .position(|existing| existing.sequence == sequence)
            .ok_or(CidStoreError::NotFound)?;
        Ok(self.entries.remove(position))
    }

    /// Advance `retire_prior_to` and remove every entry with `sequence`
    /// strictly less than `threshold`. Returns the count of removed
    /// entries.
    ///
    /// Per RFC 9000 §19.15 the receiver of a NEW_CONNECTION_ID with
    /// `retire_prior_to = T` must retire all CIDs with sequence < T and
    /// send RETIRE_CONNECTION_ID frames for each. The caller is
    /// responsible for emitting those frames; this function just
    /// updates the local store.
    pub fn retire_prior_to_threshold(&mut self, threshold: u64) -> usize {
        if threshold <= self.retire_prior_to {
            return 0;
        }
        self.retire_prior_to = threshold;
        let before = self.entries.len();
        self.entries.retain(|entry| entry.sequence >= threshold);
        before - self.entries.len()
    }

    /// Look up an entry by sequence number.
    #[must_use]
    pub fn find_by_sequence(&self, sequence: u64) -> Option<&CidEntry> {
        self.entries.iter().find(|entry| entry.sequence == sequence)
    }

    /// Look up an entry by CID bytes — used by the endpoint demux to
    /// route incoming packets to the right connection state.
    #[must_use]
    pub fn find_by_cid(&self, cid: &[u8]) -> Option<&CidEntry> {
        self.entries.iter().find(|entry| entry.cid() == cid)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    const TOKEN_A: [u8; STATELESS_RESET_TOKEN_LEN] = [0xa1; STATELESS_RESET_TOKEN_LEN];
    const TOKEN_B: [u8; STATELESS_RESET_TOKEN_LEN] = [0xb2; STATELESS_RESET_TOKEN_LEN];
    const TOKEN_C: [u8; STATELESS_RESET_TOKEN_LEN] = [0xc3; STATELESS_RESET_TOKEN_LEN];

    #[test]
    fn empty_queue_has_no_entries() {
        let queue: CidQueue<4> = CidQueue::new();
        assert!(queue.is_empty());
        assert_eq!(queue.len(), 0);
        assert!(!queue.is_full());
    }

    #[test]
    fn insert_then_find_by_sequence() {
        let mut queue: CidQueue<4> = CidQueue::new();
        let entry = CidEntry::new(7, &[1, 2, 3, 4], TOKEN_A).unwrap();
        queue.insert(entry).unwrap();
        let found = queue.find_by_sequence(7).unwrap();
        assert_eq!(found.cid(), &[1, 2, 3, 4]);
        assert_eq!(found.stateless_reset_token, TOKEN_A);
    }

    #[test]
    fn insert_then_find_by_cid() {
        let mut queue: CidQueue<4> = CidQueue::new();
        let cid = [9, 8, 7, 6, 5];
        queue
            .insert(CidEntry::new(0, &cid, TOKEN_A).unwrap())
            .unwrap();
        let found = queue.find_by_cid(&cid).unwrap();
        assert_eq!(found.sequence, 0);
    }

    #[cfg(feature = "quic-alloc")]
    #[test]
    fn insert_maintains_sequence_order() {
        let mut queue: CidQueue<4> = CidQueue::new();
        queue
            .insert(CidEntry::new(2, &[2], TOKEN_A).unwrap())
            .unwrap();
        queue
            .insert(CidEntry::new(0, &[0], TOKEN_B).unwrap())
            .unwrap();
        queue
            .insert(CidEntry::new(1, &[1], TOKEN_C).unwrap())
            .unwrap();
        let sequences: alloc::vec::Vec<u64> = queue.iter().map(|entry| entry.sequence).collect();
        assert_eq!(sequences, alloc::vec![0, 1, 2]);
    }

    #[test]
    fn duplicate_sequence_rejected() {
        let mut queue: CidQueue<4> = CidQueue::new();
        queue
            .insert(CidEntry::new(0, &[0], TOKEN_A).unwrap())
            .unwrap();
        assert_eq!(
            queue.insert(CidEntry::new(0, &[1], TOKEN_B).unwrap()),
            Err(CidStoreError::DuplicateSequence)
        );
    }

    #[test]
    fn capacity_full_rejected() {
        let mut queue: CidQueue<2> = CidQueue::new();
        queue
            .insert(CidEntry::new(0, &[0], TOKEN_A).unwrap())
            .unwrap();
        queue
            .insert(CidEntry::new(1, &[1], TOKEN_B).unwrap())
            .unwrap();
        assert!(queue.is_full());
        assert_eq!(
            queue.insert(CidEntry::new(2, &[2], TOKEN_C).unwrap()),
            Err(CidStoreError::Full)
        );
    }

    #[test]
    fn retire_removes_named_sequence() {
        let mut queue: CidQueue<4> = CidQueue::new();
        queue
            .insert(CidEntry::new(0, &[0], TOKEN_A).unwrap())
            .unwrap();
        queue
            .insert(CidEntry::new(1, &[1], TOKEN_B).unwrap())
            .unwrap();
        let removed = queue.retire(0).unwrap();
        assert_eq!(removed.sequence, 0);
        assert_eq!(queue.len(), 1);
        assert!(queue.find_by_sequence(0).is_none());
        assert!(queue.find_by_sequence(1).is_some());
    }

    #[test]
    fn retire_not_found() {
        let mut queue: CidQueue<4> = CidQueue::new();
        assert_eq!(queue.retire(99), Err(CidStoreError::NotFound));
    }

    #[cfg(feature = "quic-alloc")]
    #[test]
    fn retire_prior_to_threshold_removes_below() {
        let mut queue: CidQueue<8> = CidQueue::new();
        for seq in 0..6 {
            queue
                .insert(CidEntry::new(seq, &[seq as u8], TOKEN_A).unwrap())
                .unwrap();
        }
        let removed = queue.retire_prior_to_threshold(3);
        assert_eq!(removed, 3);
        assert_eq!(queue.len(), 3);
        let remaining: alloc::vec::Vec<u64> = queue.iter().map(|entry| entry.sequence).collect();
        assert_eq!(remaining, alloc::vec![3, 4, 5]);
        assert_eq!(queue.retire_prior_to(), 3);
    }

    #[test]
    fn retire_prior_to_threshold_idempotent_for_older_thresholds() {
        let mut queue: CidQueue<4> = CidQueue::new();
        queue
            .insert(CidEntry::new(5, &[5], TOKEN_A).unwrap())
            .unwrap();
        queue.retire_prior_to_threshold(10);
        assert_eq!(queue.retire_prior_to(), 10);
        // a later, older threshold is a no-op
        let removed = queue.retire_prior_to_threshold(3);
        assert_eq!(removed, 0);
        assert_eq!(queue.retire_prior_to(), 10);
    }

    #[test]
    fn cid_too_long_rejected() {
        let too_long = [0u8; MAX_CID_LEN + 1];
        assert_eq!(
            CidEntry::new(0, &too_long, TOKEN_A),
            Err(CidStoreError::CidTooLong)
        );
    }

    #[test]
    fn zero_length_cid_supported() {
        let entry = CidEntry::new(0, &[], TOKEN_A).unwrap();
        assert_eq!(entry.cid(), &[] as &[u8]);
        assert_eq!(entry.cid_len(), 0);
    }

    #[test]
    fn max_length_cid_supported() {
        let max_cid = [0xab; MAX_CID_LEN];
        let entry = CidEntry::new(0, &max_cid, TOKEN_A).unwrap();
        assert_eq!(entry.cid(), &max_cid);
        assert_eq!(entry.cid_len(), MAX_CID_LEN);
    }

    #[test]
    fn find_by_cid_returns_none_for_missing() {
        let queue: CidQueue<4> = CidQueue::new();
        assert!(queue.find_by_cid(&[1, 2, 3]).is_none());
    }
}
