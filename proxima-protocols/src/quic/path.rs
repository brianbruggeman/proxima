//! Path validation per [RFC 9000 §8.2] + [§9].
//!
//! Per-path challenger state: outstanding PATH_CHALLENGE tokens we
//! issued, the single pending inbound PATH_RESPONSE awaiting emission,
//! and the validated flag. The 1-RTT egress drains pending tokens via
//! `take_pending_outbound_challenge` / `take_pending_path_response`.
//!
//! [RFC 9000 §8.2]: https://www.rfc-editor.org/rfc/rfc9000#section-8.2
//! [§9]: https://www.rfc-editor.org/rfc/rfc9000#section-9
//!
//! # Tier
//!
//! Tier-3 (bare `no_std + no_alloc`). Outstanding-token storage uses
//! `arrayvec::ArrayVec` with the const-generic cap from
//! `proxima-quic-proto.toml [path].max_outstanding_challenges` per
//! principle 12.

use arrayvec::ArrayVec;
use rand_core::{CryptoRng, Rng};

use crate::quic::frame::PATH_CHALLENGE_LEN;
use crate::quic::sized;
use crate::quic::time::Instant;

/// Maximum outstanding PATH_CHALLENGE tokens per path. Sourced from
/// `proxima-quic-proto.toml [path].max_outstanding_challenges`.
pub const MAX_OUTSTANDING_CHALLENGES: usize = sized::PATH_MAX_OUTSTANDING_CHALLENGES;

/// Per-path challenger state per RFC 9000 §8.2.
#[derive(Debug, Clone)]
pub struct PathChallenger {
    /// Tokens we've sent and are awaiting a response for.
    outstanding: ArrayVec<OutstandingChallenge, MAX_OUTSTANDING_CHALLENGES>,
    /// `true` once any inbound PATH_RESPONSE matched an outstanding token.
    validated: bool,
    /// Pending PATH_RESPONSE to emit (response to an inbound CHALLENGE).
    /// One slot — RFC 9000 §8.2.2 says we MAY drop subsequent challenges
    /// if we're already queueing a response (rate-limiting).
    pending_response: Option<[u8; PATH_CHALLENGE_LEN]>,
    /// Pending outbound PATH_CHALLENGE token to emit on the next
    /// outbound packet (set by [`Self::issue`]). One slot — the most
    /// recent issued token. The matching outstanding entry tracks
    /// the response separately.
    pending_outbound_challenge: Option<[u8; PATH_CHALLENGE_LEN]>,
}

/// One outstanding PATH_CHALLENGE awaiting response.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutstandingChallenge {
    pub token: [u8; PATH_CHALLENGE_LEN],
    pub sent_at: Instant,
}

impl Default for PathChallenger {
    fn default() -> Self {
        Self::new()
    }
}

impl PathChallenger {
    /// Construct a fresh unvalidated challenger.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            outstanding: ArrayVec::new_const(),
            validated: false,
            pending_response: None,
            pending_outbound_challenge: None,
        }
    }

    /// Is this path validated? Once `true`, never flips back.
    #[must_use]
    pub const fn validated(&self) -> bool {
        self.validated
    }

    /// Issue a new PATH_CHALLENGE token via the caller-supplied
    /// CryptoRng + Rng. Returns the 8-byte token to embed in the
    /// outbound PATH_CHALLENGE frame, or `None` if we're at the
    /// outstanding-cap (caller should retry after a previous token
    /// is matched/abandoned).
    pub fn issue<R: CryptoRng + Rng>(
        &mut self,
        rng: &mut R,
        now: Instant,
    ) -> Option<[u8; PATH_CHALLENGE_LEN]> {
        if self.outstanding.is_full() {
            return None;
        }
        let mut token = [0u8; PATH_CHALLENGE_LEN];
        rng.fill_bytes(&mut token);
        let _ = self.outstanding.try_push(OutstandingChallenge {
            token,
            sent_at: now,
        });
        // Queue for outbound emission. Overrides any prior pending
        // outbound challenge (only the most-recent token is emitted;
        // the older one stays in outstanding[] for matching).
        self.pending_outbound_challenge = Some(token);
        Some(token)
    }

    /// Drain the pending outbound PATH_CHALLENGE token if any (caller
    /// will emit a PATH_CHALLENGE frame on the next outbound packet).
    pub fn take_pending_outbound_challenge(&mut self) -> Option<[u8; PATH_CHALLENGE_LEN]> {
        self.pending_outbound_challenge.take()
    }

    /// Restash a pending outbound challenge (used by the egress path
    /// when the caller drains via take_pending_outbound_challenge but
    /// then has nothing to actually emit this turn).
    pub fn queue_outbound_challenge(&mut self, token: [u8; PATH_CHALLENGE_LEN]) {
        self.pending_outbound_challenge = Some(token);
    }

    /// Match an inbound PATH_RESPONSE token against outstanding
    /// challenges. On match, removes the token AND marks the path
    /// validated; returns `true`. On no match, returns `false`.
    pub fn record_response(&mut self, response_token: &[u8; PATH_CHALLENGE_LEN]) -> bool {
        let position = self
            .outstanding
            .iter()
            .position(|outstanding| outstanding.token.as_slice() == response_token.as_slice());
        let Some(index) = position else {
            return false;
        };
        self.outstanding.remove(index);
        self.validated = true;
        true
    }

    /// Record an inbound PATH_CHALLENGE — queue the matching
    /// PATH_RESPONSE for the next outbound packet. If a response is
    /// already queued (rate-limit per RFC 9000 §8.2.2), drop silently.
    pub fn note_inbound_challenge(&mut self, challenge_token: [u8; PATH_CHALLENGE_LEN]) {
        if self.pending_response.is_none() {
            self.pending_response = Some(challenge_token);
        }
    }

    /// Drain the pending PATH_RESPONSE token if any (caller will
    /// emit a PATH_RESPONSE frame with this data).
    pub fn take_pending_response(&mut self) -> Option<[u8; PATH_CHALLENGE_LEN]> {
        self.pending_response.take()
    }

    /// Number of outstanding tokens awaiting response.
    #[must_use]
    pub fn outstanding_count(&self) -> usize {
        self.outstanding.len()
    }

    /// Abandon outstanding tokens whose `sent_at` is older than
    /// `now - max_age`. Called periodically from `handle_timeout`.
    pub fn abandon_expired(&mut self, now: Instant, max_age: crate::quic::time::Duration) {
        let cutoff = now.saturating_sub(max_age);
        self.outstanding
            .retain(|challenge| challenge.sent_at >= cutoff);
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::quic::time::Duration;
    use rand_core::SeedableRng;

    fn deterministic_rng() -> rand_chacha::ChaCha8Rng {
        rand_chacha::ChaCha8Rng::seed_from_u64(0xC0FFEE)
    }

    fn at(micros: u64) -> Instant {
        Instant::from_micros(micros)
    }

    #[test]
    fn new_challenger_is_unvalidated_with_no_outstanding() {
        let challenger = PathChallenger::new();
        assert!(!challenger.validated());
        assert_eq!(challenger.outstanding_count(), 0);
    }

    #[test]
    fn issue_returns_8_byte_token_and_increments_outstanding() {
        let mut challenger = PathChallenger::new();
        let mut rng = deterministic_rng();
        let token = challenger.issue(&mut rng, at(1_000_000)).expect("issued");
        assert_eq!(token.len(), 8);
        assert_eq!(challenger.outstanding_count(), 1);
    }

    #[test]
    fn issue_at_cap_returns_none() {
        let mut challenger = PathChallenger::new();
        let mut rng = deterministic_rng();
        for _ in 0..MAX_OUTSTANDING_CHALLENGES {
            challenger
                .issue(&mut rng, at(1_000_000))
                .expect("within cap");
        }
        assert!(challenger.issue(&mut rng, at(1_000_000)).is_none());
    }

    #[test]
    fn record_response_matching_token_validates_path() {
        let mut challenger = PathChallenger::new();
        let mut rng = deterministic_rng();
        let token = challenger.issue(&mut rng, at(1_000_000)).expect("issued");
        assert!(challenger.record_response(&token));
        assert!(challenger.validated());
        assert_eq!(challenger.outstanding_count(), 0);
    }

    #[test]
    fn record_response_unknown_token_returns_false() {
        let mut challenger = PathChallenger::new();
        let unknown = [0xAA; 8];
        assert!(!challenger.record_response(&unknown));
        assert!(!challenger.validated());
    }

    #[test]
    fn note_inbound_challenge_queues_one_pending_response() {
        let mut challenger = PathChallenger::new();
        let token = [0xDE; 8];
        challenger.note_inbound_challenge(token);
        let next_token = [0xAD; 8];
        // Second challenge dropped while first response still pending.
        challenger.note_inbound_challenge(next_token);
        let drained = challenger.take_pending_response().expect("pending");
        assert_eq!(drained, token);
        assert!(challenger.take_pending_response().is_none());
    }

    #[test]
    fn abandon_expired_drops_old_tokens() {
        let mut challenger = PathChallenger::new();
        let mut rng = deterministic_rng();
        challenger.issue(&mut rng, at(1_000_000)).expect("issued");
        challenger.issue(&mut rng, at(2_000_000)).expect("issued");
        challenger.abandon_expired(at(3_000_000), Duration::from_micros(500_000));
        // Cutoff = 2_500_000; t=1_000_000 token abandoned, t=2_000_000 abandoned (also older).
        assert_eq!(challenger.outstanding_count(), 0);
    }

    #[test]
    fn worked_example_from_design_doc() {
        // docs/proxima-quic/c21-path-migration-design.md walked example.
        let mut challenger = PathChallenger::new();
        let mut rng = deterministic_rng();
        // Issue T1 + T2.
        let _t1 = challenger.issue(&mut rng, at(1_000_000)).expect("T1");
        let t2 = challenger.issue(&mut rng, at(1_000_000)).expect("T2");
        assert_eq!(challenger.outstanding_count(), 2);
        // Inbound PATH_CHALLENGE C1 → queue response.
        let c1 = [0x11; 8];
        challenger.note_inbound_challenge(c1);
        assert_eq!(challenger.take_pending_response(), Some(c1));
        // Inbound PATH_RESPONSE T2 → match, validate.
        assert!(challenger.record_response(&t2));
        assert!(challenger.validated());
        assert_eq!(challenger.outstanding_count(), 1);
        // Unknown PATH_RESPONSE → no match.
        let unknown = [0xFF; 8];
        assert!(!challenger.record_response(&unknown));
        // T1 still outstanding.
        assert_eq!(challenger.outstanding_count(), 1);
        // Abandon expired (max_age=500ms; t1 sent at 1s, now 2s).
        challenger.abandon_expired(at(2_000_000), Duration::from_micros(500_000));
        assert_eq!(challenger.outstanding_count(), 0);
    }
}
