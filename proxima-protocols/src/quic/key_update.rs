//! 1-RTT key update per [RFC 9001 §6].
//!
//! The TLS provider trait already exposes `initiate_key_update` +
//! pushes new [`EpochSecrets`] via the sink with `generation = N+1`
//! when a key update completes. This module adds the per-connection
//! state machine that:
//!
//! 1. Enforces the RFC §6.1 preconditions (handshake confirmed +
//!    current key phase ACK'd).
//! 2. Stages pending next-phase secrets until the first 1-RTT egress
//!    or peer-initiated swap.
//! 3. Recognises peer-initiated key updates by the Key Phase bit flip
//!    per RFC §5.4.1 + §6.2.
//! 4. Refuses consecutive updates without intervening peer ACK
//!    (RFC §6.2 — KEY_UPDATE_ERROR condition).
//!
//! [RFC 9001 §6]: https://www.rfc-editor.org/rfc/rfc9001#section-6
//!
//! # Tier
//!
//! Tier-3 (bare `no_std + no_alloc`). State is plain POD + an
//! `Option<EpochSecrets>` pending slot.

use crate::quic::sized;
use crate::quic::time::{Duration, Instant};
use crate::quic::tls::EpochSecrets;

/// Minimum interval between client-initiated key updates BEYOND the
/// RFC-mandated "current-phase-ACK'd" precondition. Sourced from
/// `proxima-quic-proto.toml [key_update].min_initiation_interval_micros`.
pub const MIN_INITIATION_INTERVAL_MICROS: u64 = sized::KEY_UPDATE_MIN_INITIATION_INTERVAL_MICROS;

/// Decision outcome from [`KeyUpdateManager::observe_inbound_key_phase`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum KeyChoice {
    /// Inbound key phase bit matches our current generation — use current keys.
    Current,
    /// Inbound key phase bit differs — try the staged next-phase keys.
    /// On successful unprotect, caller MUST call [`KeyUpdateManager::confirm_peer_initiated_update`]
    /// to swap the send-side keys per RFC §6.2 BEFORE acking the trigger packet.
    Next,
    /// Inbound bit differs but no next-phase keys are staged — drop the
    /// packet per RFC §6.3. Caller has options: stage the next keys
    /// proactively (recommended), or wait for the TLS provider to push
    /// them.
    DropNoNextKeys,
}

/// Errors from [`KeyUpdateManager::initiate`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum KeyUpdateError {
    /// Handshake not yet confirmed — RFC §6.1.
    HandshakeNotConfirmed,
    /// Haven't received an ACK for any packet sent in the current key
    /// phase — RFC §6.1 ("MUST NOT initiate a subsequent key update
    /// unless it has received an acknowledgment...").
    CurrentPhaseUnacked,
    /// Update was initiated too recently per the local
    /// `min_initiation_interval` floor.
    TooSoon { earliest: Instant },
    /// A previous key update is still pending (peer hasn't ACK'd our
    /// switch to the new phase).
    UpdateInProgress,
}

/// Per-connection 1-RTT key update state machine.
#[derive(Debug, Clone)]
pub struct KeyUpdateManager {
    /// Monotonic generation count. Key phase bit = (generation & 1) as u8.
    generation: u8,
    /// Set true once the TLS provider fires `TlsEvent::HandshakeConfirmed`.
    handshake_confirmed: bool,
    /// `true` once we've received an ACK for any packet sent in the
    /// current key phase. Reset to `false` on every key swap.
    current_phase_acked: bool,
    /// Pending next-phase secrets staged by the TLS provider's
    /// `initiate_key_update` (arrives via sink on_new_secrets with
    /// `generation = current + 1`), OR proactively pre-derived per
    /// RFC §6.3 to be ready for peer-initiated updates.
    pending_next: Option<EpochSecrets>,
    /// Earliest time we may initiate another update. Defaults to
    /// `Instant::ZERO` (i.e. allowed immediately) until the first
    /// initiation.
    next_update_allowed_at: Instant,
    /// Configured floor on the time between client-initiated updates
    /// beyond the RFC's MUST-ACK precondition.
    min_initiation_interval: Duration,
}

impl Default for KeyUpdateManager {
    fn default() -> Self {
        Self::new()
    }
}

impl KeyUpdateManager {
    /// Construct a fresh manager at generation 0, handshake unconfirmed.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            generation: 0,
            handshake_confirmed: false,
            current_phase_acked: false,
            pending_next: None,
            next_update_allowed_at: Instant::ZERO,
            min_initiation_interval: Duration::from_micros(MIN_INITIATION_INTERVAL_MICROS),
        }
    }

    /// Override the default min-initiation-interval floor.
    pub fn set_min_initiation_interval(&mut self, interval: Duration) {
        self.min_initiation_interval = interval;
    }

    /// Current generation. Key phase bit = `generation & 1`.
    #[must_use]
    pub const fn generation(&self) -> u8 {
        self.generation
    }

    /// Current key phase bit per RFC 9001 §5.4.1 — `0` or `1`.
    #[must_use]
    pub const fn key_phase(&self) -> u8 {
        self.generation & 1
    }

    /// Mark handshake confirmation (called when the TLS provider fires
    /// `TlsEvent::HandshakeConfirmed`).
    pub fn note_handshake_confirmed(&mut self) {
        self.handshake_confirmed = true;
    }

    /// Has the handshake been confirmed?
    #[must_use]
    pub const fn handshake_confirmed(&self) -> bool {
        self.handshake_confirmed
    }

    /// Has the current key-phase received an ACK from the peer?
    #[must_use]
    pub const fn current_phase_acked(&self) -> bool {
        self.current_phase_acked
    }

    /// Earliest time another key update may be initiated, post the
    /// min_initiation_interval floor.
    #[must_use]
    pub const fn next_update_allowed_at(&self) -> Instant {
        self.next_update_allowed_at
    }

    /// Record that we've received an ACK for any packet sent in the
    /// current key phase. Required by RFC §6.1 before another update
    /// may be initiated.
    pub fn note_current_phase_acked(&mut self) {
        self.current_phase_acked = true;
    }

    /// Are the per-RFC preconditions met to initiate a key update?
    pub fn may_initiate(&self, now: Instant) -> Result<(), KeyUpdateError> {
        if !self.handshake_confirmed {
            return Err(KeyUpdateError::HandshakeNotConfirmed);
        }
        if !self.current_phase_acked {
            return Err(KeyUpdateError::CurrentPhaseUnacked);
        }
        if self.pending_next.is_some() {
            return Err(KeyUpdateError::UpdateInProgress);
        }
        if now < self.next_update_allowed_at {
            return Err(KeyUpdateError::TooSoon {
                earliest: self.next_update_allowed_at,
            });
        }
        Ok(())
    }

    /// Record a TLS-provider-pushed new EpochSecrets at
    /// `generation = self.generation + 1`. Returns `true` if accepted;
    /// `false` if the generation doesn't advance (caller should treat
    /// as a TLS provider error).
    pub fn stage_pending(&mut self, secrets: EpochSecrets) -> bool {
        if secrets.generation != self.generation.saturating_add(1) {
            return false;
        }
        self.pending_next = Some(secrets);
        true
    }

    /// Borrow the pending next-phase secrets without consuming them.
    /// Used to attempt unprotect of an inbound packet whose key phase
    /// bit differs from ours.
    pub fn pending_next(&self) -> Option<&EpochSecrets> {
        self.pending_next.as_ref()
    }

    /// Decide which keys to try for an inbound 1-RTT packet given its
    /// observed key phase bit (extracted from the first byte AFTER
    /// header protection removal).
    #[must_use]
    pub fn observe_inbound_key_phase(&self, inbound_key_phase: u8) -> KeyChoice {
        if inbound_key_phase & 1 == self.key_phase() {
            KeyChoice::Current
        } else if self.pending_next.is_some() {
            KeyChoice::Next
        } else {
            KeyChoice::DropNoNextKeys
        }
    }

    /// Swap to the staged next-phase secrets. Called by the FSM when:
    ///
    /// - Locally initiated update: just before emitting the first 1-RTT
    ///   packet with the new key phase bit.
    /// - Peer-initiated update: BEFORE sending the ACK for the
    ///   packet that triggered the update (RFC §6.2).
    ///
    /// Returns the new current-phase EpochSecrets, or `None` if no
    /// pending update was staged.
    pub fn swap_to_pending(&mut self, now: Instant) -> Option<EpochSecrets> {
        let next = self.pending_next.take()?;
        self.generation = self.generation.saturating_add(1);
        self.current_phase_acked = false;
        self.next_update_allowed_at = now + self.min_initiation_interval;
        Some(next)
    }

    /// Convenience: caller wants to confirm a peer-initiated update.
    /// Identical to `swap_to_pending(now)` but the semantic is
    /// explicit so future-readers don't confuse the call sites.
    pub fn confirm_peer_initiated_update(&mut self, now: Instant) -> Option<EpochSecrets> {
        self.swap_to_pending(now)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::quic::crypto::initial_keys::{QUIC_HP_LEN, QUIC_IV_LEN, QUIC_KEY_LEN};
    use crate::quic::tls::{DirectionalKeys, Epoch, EpochSecrets, HeaderKeyMaterial, PacketKeyMaterial};

    fn at(micros: u64) -> Instant {
        Instant::from_micros(micros)
    }

    fn synthetic_app_secrets(generation: u8) -> EpochSecrets {
        let keys = DirectionalKeys {
            packet: PacketKeyMaterial::Aes128Gcm {
                key: [generation; QUIC_KEY_LEN],
                iv: [generation; QUIC_IV_LEN],
            },
            header: HeaderKeyMaterial::Aes128 {
                hp: [generation; QUIC_HP_LEN],
            },
        };
        EpochSecrets {
            epoch: Epoch::Application,
            generation,
            local: keys.clone(),
            remote: keys,
        }
    }

    #[test]
    fn new_manager_at_generation_zero() {
        let mgr = KeyUpdateManager::new();
        assert_eq!(mgr.generation(), 0);
        assert_eq!(mgr.key_phase(), 0);
        assert!(mgr.pending_next().is_none());
    }

    #[test]
    fn may_initiate_rejects_pre_handshake() {
        let mgr = KeyUpdateManager::new();
        assert_eq!(
            mgr.may_initiate(at(1_000_000)),
            Err(KeyUpdateError::HandshakeNotConfirmed)
        );
    }

    #[test]
    fn may_initiate_rejects_current_phase_unacked() {
        let mut mgr = KeyUpdateManager::new();
        mgr.note_handshake_confirmed();
        assert_eq!(
            mgr.may_initiate(at(1_000_000)),
            Err(KeyUpdateError::CurrentPhaseUnacked)
        );
    }

    #[test]
    fn may_initiate_accepts_after_preconditions() {
        let mut mgr = KeyUpdateManager::new();
        mgr.note_handshake_confirmed();
        mgr.note_current_phase_acked();
        assert!(mgr.may_initiate(at(1_000_000)).is_ok());
    }

    #[test]
    fn stage_pending_rejects_wrong_generation() {
        let mut mgr = KeyUpdateManager::new();
        mgr.note_handshake_confirmed();
        mgr.note_current_phase_acked();
        let wrong = synthetic_app_secrets(5);
        assert!(!mgr.stage_pending(wrong));
        assert!(mgr.pending_next().is_none());
    }

    #[test]
    fn stage_pending_accepts_generation_plus_one() {
        let mut mgr = KeyUpdateManager::new();
        let next = synthetic_app_secrets(1);
        assert!(mgr.stage_pending(next));
        assert_eq!(mgr.pending_next().unwrap().generation, 1);
    }

    #[test]
    fn may_initiate_rejects_while_update_in_progress() {
        let mut mgr = KeyUpdateManager::new();
        mgr.note_handshake_confirmed();
        mgr.note_current_phase_acked();
        mgr.stage_pending(synthetic_app_secrets(1));
        assert_eq!(
            mgr.may_initiate(at(1_000_000)),
            Err(KeyUpdateError::UpdateInProgress)
        );
    }

    #[test]
    fn swap_to_pending_advances_generation_and_resets_ack() {
        let mut mgr = KeyUpdateManager::new();
        mgr.note_handshake_confirmed();
        mgr.note_current_phase_acked();
        mgr.stage_pending(synthetic_app_secrets(1));
        let swapped = mgr.swap_to_pending(at(2_000_000)).expect("swap");
        assert_eq!(swapped.generation, 1);
        assert_eq!(mgr.generation(), 1);
        assert_eq!(mgr.key_phase(), 1);
        assert!(!mgr.current_phase_acked);
        assert!(mgr.pending_next().is_none());
    }

    #[test]
    fn may_initiate_rejects_too_soon_after_recent_update() {
        let mut mgr = KeyUpdateManager::new();
        mgr.note_handshake_confirmed();
        mgr.note_current_phase_acked();
        mgr.stage_pending(synthetic_app_secrets(1));
        mgr.swap_to_pending(at(2_000_000));
        mgr.note_current_phase_acked();
        let result = mgr.may_initiate(at(2_000_100));
        assert!(matches!(result, Err(KeyUpdateError::TooSoon { .. })));
        let allowed = at(2_000_000 + MIN_INITIATION_INTERVAL_MICROS);
        assert!(mgr.may_initiate(allowed).is_ok());
    }

    #[test]
    fn observe_inbound_matching_phase_returns_current() {
        let mgr = KeyUpdateManager::new();
        // Generation 0 → key phase 0.
        assert_eq!(mgr.observe_inbound_key_phase(0), KeyChoice::Current);
    }

    #[test]
    fn observe_inbound_differing_phase_with_no_pending_returns_drop() {
        let mgr = KeyUpdateManager::new();
        assert_eq!(mgr.observe_inbound_key_phase(1), KeyChoice::DropNoNextKeys);
    }

    #[test]
    fn observe_inbound_differing_phase_with_pending_returns_next() {
        let mut mgr = KeyUpdateManager::new();
        mgr.stage_pending(synthetic_app_secrets(1));
        assert_eq!(mgr.observe_inbound_key_phase(1), KeyChoice::Next);
    }

    #[test]
    fn client_initiated_walked_example_from_design_doc() {
        // docs/proxima-quic/c23-key-update-design.md client-initiated worked example.
        let mut mgr = KeyUpdateManager::new();
        // Handshake confirmed + first ACK in phase 0 received.
        mgr.note_handshake_confirmed();
        mgr.note_current_phase_acked();
        // Initiation OK.
        assert!(mgr.may_initiate(at(1_000_000)).is_ok());
        // TLS provider pushes next secrets.
        mgr.stage_pending(synthetic_app_secrets(1));
        // Swap before next 1-RTT egress.
        let new_secrets = mgr.swap_to_pending(at(1_001_000)).expect("swap");
        assert_eq!(new_secrets.generation, 1);
        assert_eq!(mgr.key_phase(), 1);
        // ACK arrives for phase-1 packet.
        mgr.note_current_phase_acked();
        // Next initiation refused until interval passes.
        assert!(matches!(
            mgr.may_initiate(at(1_001_500)),
            Err(KeyUpdateError::TooSoon { .. })
        ));
    }

    #[test]
    fn peer_initiated_walked_example_from_design_doc() {
        // docs/proxima-quic/c23-key-update-design.md peer-initiated worked example.
        let mut mgr = KeyUpdateManager::new();
        // Proactively stage next-phase keys per RFC §6.3.
        mgr.stage_pending(synthetic_app_secrets(1));
        // Inbound packet at key phase 1 (we're at phase 0) → Next.
        assert_eq!(mgr.observe_inbound_key_phase(1), KeyChoice::Next);
        // After successful unprotect with pending keys: confirm + swap.
        let new_secrets = mgr
            .confirm_peer_initiated_update(at(2_000_000))
            .expect("confirm");
        assert_eq!(new_secrets.generation, 1);
        assert_eq!(mgr.key_phase(), 1);
    }
}
