//! Per-connection 0-RTT state machine + replay-protection policy
//! per [RFC 9001 §4.6] + §9.2.
//!
//! Holds the per-connection ResumptionTicket (opaque server-private
//! bytes), tracks the early-data status (NotAttempted / Attempting
//! / Accepted / Rejected / Disabled), and gates outbound 0-RTT egress
//! via the caller-chosen [`ZeroRttPolicy`].
//!
//! Per RFC §9.2: "Disabling 0-RTT entirely is the most effective
//! defense against replay attack." Default policy is [`ZeroRttPolicy::Disabled`];
//! caller MUST opt in.
//!
//! [RFC 9001 §4.6]: https://www.rfc-editor.org/rfc/rfc9001#section-4.6
//!
//! # Tier
//!
//! Tier-3 (bare `no_std + no_alloc`). State = POD policy + status +
//! bounded `ArrayVec<u8, MAX_RESUMPTION_TICKET_LEN>`.

use arrayvec::ArrayVec;

use crate::quic::sized;

/// Maximum length of a server-issued NewSessionTicket carried as the
/// client's resumption credential. Sourced from
/// `proxima-quic-proto.toml [zero_rtt].max_resumption_ticket_len`.
pub const MAX_RESUMPTION_TICKET_LEN: usize = sized::ZERO_RTT_MAX_RESUMPTION_TICKET_LEN;

/// Inline byte stash for a TLS 1.3 NewSessionTicket.
pub type ResumptionTicket = ArrayVec<u8, MAX_RESUMPTION_TICKET_LEN>;

/// Replay-protection policy for 0-RTT data. The application protocol
/// MUST attest to its own per-RFC-§9.2 replay-mitigation analysis
/// before selecting [`Required`] or [`Allowed`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum ZeroRttPolicy {
    /// 0-RTT MUST be used if the resumption ticket allows it.
    /// Caller attests its application protocol has adequate
    /// replay-mitigation (e.g. HTTP/3 idempotent-methods-only).
    Required,
    /// 0-RTT is permitted if the server accepts it. Caller has
    /// implemented per-RFC-§9.2 replay-mitigation analysis.
    Allowed,
    /// 0-RTT is disabled even if the resumption ticket would allow
    /// it. The most effective defense per RFC §9.2 — the default.
    #[default]
    Disabled,
}

/// Current per-connection 0-RTT status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum ZeroRttStatus {
    /// No resumption ticket OR policy = [`ZeroRttPolicy::Disabled`].
    #[default]
    NotAttempted,
    /// Client side, pending the server's EncryptedExtensions answer.
    Attempting,
    /// Server's EncryptedExtensions included the `early_data` extension.
    Accepted,
    /// Server's EncryptedExtensions omitted `early_data`, or server
    /// sent HelloRetryRequest (RFC §4.6.2).
    Rejected,
    /// Policy is [`ZeroRttPolicy::Disabled`] — 0-RTT will never be
    /// attempted regardless of ticket availability.
    Disabled,
}

/// Errors from [`ZeroRttManager`] operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ZeroRttError {
    /// `prepare_resumption` rejected because policy is `Disabled`.
    PolicyDisabled,
    /// Resumption ticket exceeded [`MAX_RESUMPTION_TICKET_LEN`].
    TicketTooLong,
    /// `record_zero_rtt_ack_received` was called while status is
    /// [`ZeroRttStatus::Rejected`] — RFC §4.6.2 mandates the client
    /// SHOULD treat this as `PROTOCOL_VIOLATION` and close the
    /// connection.
    AckForRejectedZeroRtt,
    /// Method was invoked from an illegal status (e.g.
    /// `note_server_accepted` from `NotAttempted`).
    IllegalStatusTransition {
        from: ZeroRttStatus,
        attempted: &'static str,
    },
}

/// Per-connection 0-RTT state machine.
#[derive(Debug, Clone)]
pub struct ZeroRttManager {
    policy: ZeroRttPolicy,
    status: ZeroRttStatus,
    resumption_ticket: Option<ResumptionTicket>,
}

impl Default for ZeroRttManager {
    fn default() -> Self {
        Self::new(ZeroRttPolicy::default())
    }
}

impl ZeroRttManager {
    /// Construct with the given replay-protection policy.
    /// `Disabled` is the safest default per RFC §9.2.
    #[must_use]
    pub const fn new(policy: ZeroRttPolicy) -> Self {
        let status = match policy {
            ZeroRttPolicy::Disabled => ZeroRttStatus::Disabled,
            _ => ZeroRttStatus::NotAttempted,
        };
        Self {
            policy,
            status,
            resumption_ticket: None,
        }
    }

    /// Current policy.
    #[must_use]
    pub const fn policy(&self) -> ZeroRttPolicy {
        self.policy
    }

    /// Current status.
    #[must_use]
    pub const fn status(&self) -> ZeroRttStatus {
        self.status
    }

    /// Borrow the resumption ticket bytes, if any.
    #[must_use]
    pub fn resumption_ticket(&self) -> Option<&[u8]> {
        self.resumption_ticket.as_deref()
    }

    /// Client-side: stash a NewSessionTicket from a prior connection
    /// for use in the next ClientHello's pre_shared_key extension.
    ///
    /// Transitions [`ZeroRttStatus::NotAttempted`] →
    /// [`ZeroRttStatus::Attempting`] when policy is not `Disabled`.
    ///
    /// # Errors
    ///
    /// - [`ZeroRttError::PolicyDisabled`] when policy is `Disabled`.
    /// - [`ZeroRttError::TicketTooLong`] when `ticket` exceeds
    ///   [`MAX_RESUMPTION_TICKET_LEN`].
    pub fn prepare_resumption(&mut self, ticket: &[u8]) -> Result<(), ZeroRttError> {
        if matches!(self.policy, ZeroRttPolicy::Disabled) {
            return Err(ZeroRttError::PolicyDisabled);
        }
        if ticket.len() > MAX_RESUMPTION_TICKET_LEN {
            return Err(ZeroRttError::TicketTooLong);
        }
        let mut buf = ResumptionTicket::new();
        // try_extend_from_slice cannot fail — bounds-checked above.
        buf.try_extend_from_slice(ticket).ok();
        self.resumption_ticket = Some(buf);
        self.status = ZeroRttStatus::Attempting;
        Ok(())
    }

    /// Server's EncryptedExtensions included the `early_data`
    /// extension — 0-RTT accepted per RFC §4.6.2.
    ///
    /// # Errors
    ///
    /// Returns [`ZeroRttError::IllegalStatusTransition`] when called
    /// from a non-Attempting status.
    pub fn note_server_accepted(&mut self) -> Result<(), ZeroRttError> {
        if !matches!(self.status, ZeroRttStatus::Attempting) {
            return Err(ZeroRttError::IllegalStatusTransition {
                from: self.status,
                attempted: "note_server_accepted",
            });
        }
        self.status = ZeroRttStatus::Accepted;
        Ok(())
    }

    /// Server's EncryptedExtensions omitted `early_data` OR server
    /// sent a HelloRetryRequest — 0-RTT rejected per RFC §4.6.2.
    ///
    /// # Errors
    ///
    /// Returns [`ZeroRttError::IllegalStatusTransition`] when called
    /// from a non-Attempting status.
    pub fn note_server_rejected(&mut self) -> Result<(), ZeroRttError> {
        if !matches!(self.status, ZeroRttStatus::Attempting) {
            return Err(ZeroRttError::IllegalStatusTransition {
                from: self.status,
                attempted: "note_server_rejected",
            });
        }
        self.status = ZeroRttStatus::Rejected;
        Ok(())
    }

    /// Gate for outbound 0-RTT egress. Returns `true` iff status is
    /// `Attempting` (client probing) or `Accepted` (server confirmed).
    /// False for `NotAttempted`, `Rejected`, `Disabled`.
    #[must_use]
    pub const fn may_send_zero_rtt(&self) -> bool {
        matches!(
            self.status,
            ZeroRttStatus::Attempting | ZeroRttStatus::Accepted
        )
    }

    /// Record an inbound ACK for a 0-RTT-epoch packet. Returns
    /// [`ZeroRttError::AckForRejectedZeroRtt`] when status is
    /// `Rejected` — caller MUST emit `CONNECTION_CLOSE` with
    /// `PROTOCOL_VIOLATION` per RFC §4.6.2.
    ///
    /// # Errors
    ///
    /// See [`ZeroRttError::AckForRejectedZeroRtt`].
    pub fn record_zero_rtt_ack_received(&self) -> Result<(), ZeroRttError> {
        if matches!(self.status, ZeroRttStatus::Rejected) {
            return Err(ZeroRttError::AckForRejectedZeroRtt);
        }
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    const TICKET: &[u8] = b"opaque-tls-1.3-newsessionticket-bytes-from-prior-connection";

    #[test]
    fn default_policy_is_disabled() {
        let manager = ZeroRttManager::default();
        assert_eq!(manager.policy(), ZeroRttPolicy::Disabled);
        assert_eq!(manager.status(), ZeroRttStatus::Disabled);
        assert!(!manager.may_send_zero_rtt());
    }

    #[test]
    fn disabled_policy_rejects_prepare_resumption() {
        let mut manager = ZeroRttManager::new(ZeroRttPolicy::Disabled);
        let result = manager.prepare_resumption(TICKET);
        assert_eq!(result, Err(ZeroRttError::PolicyDisabled));
        assert!(manager.resumption_ticket().is_none());
    }

    #[test]
    fn allowed_policy_accepts_resumption_and_attempts() {
        let mut manager = ZeroRttManager::new(ZeroRttPolicy::Allowed);
        assert_eq!(manager.status(), ZeroRttStatus::NotAttempted);
        manager.prepare_resumption(TICKET).expect("ok");
        assert_eq!(manager.status(), ZeroRttStatus::Attempting);
        assert_eq!(manager.resumption_ticket(), Some(TICKET));
        assert!(manager.may_send_zero_rtt());
    }

    #[test]
    fn required_policy_accepts_resumption_and_attempts() {
        let mut manager = ZeroRttManager::new(ZeroRttPolicy::Required);
        manager.prepare_resumption(TICKET).expect("ok");
        assert_eq!(manager.status(), ZeroRttStatus::Attempting);
        assert!(manager.may_send_zero_rtt());
    }

    #[test]
    fn prepare_resumption_rejects_oversized_ticket() {
        let mut manager = ZeroRttManager::new(ZeroRttPolicy::Allowed);
        let too_long = alloc::vec![0u8; MAX_RESUMPTION_TICKET_LEN + 1];
        let result = manager.prepare_resumption(&too_long);
        assert_eq!(result, Err(ZeroRttError::TicketTooLong));
    }

    extern crate alloc;

    #[test]
    fn worked_example_accept_path_from_design_doc() {
        // docs/proxima-quic/c24-zero-rtt-design.md accept worked example.
        let mut manager = ZeroRttManager::new(ZeroRttPolicy::Allowed);
        // T1: prepare resumption.
        manager.prepare_resumption(TICKET).expect("ok");
        // T2: gate true.
        assert!(manager.may_send_zero_rtt());
        // T4: server accepts.
        manager.note_server_accepted().expect("accept ok");
        assert_eq!(manager.status(), ZeroRttStatus::Accepted);
        // T5: gate still true.
        assert!(manager.may_send_zero_rtt());
    }

    #[test]
    fn worked_example_reject_path_from_design_doc() {
        // docs/proxima-quic/c24-zero-rtt-design.md reject worked example.
        let mut manager = ZeroRttManager::new(ZeroRttPolicy::Allowed);
        manager.prepare_resumption(TICKET).expect("ok");
        // T2: server rejects.
        manager.note_server_rejected().expect("reject ok");
        assert_eq!(manager.status(), ZeroRttStatus::Rejected);
        // T3: gate now false.
        assert!(!manager.may_send_zero_rtt());
        // T4: ack for 0-RTT packet → PROTOCOL_VIOLATION.
        let result = manager.record_zero_rtt_ack_received();
        assert_eq!(result, Err(ZeroRttError::AckForRejectedZeroRtt));
    }

    #[test]
    fn worked_example_disabled_path_from_design_doc() {
        // docs/proxima-quic/c24-zero-rtt-design.md disabled worked example.
        let mut manager = ZeroRttManager::new(ZeroRttPolicy::Disabled);
        assert_eq!(manager.status(), ZeroRttStatus::Disabled);
        // T1: prepare rejected.
        assert_eq!(
            manager.prepare_resumption(TICKET),
            Err(ZeroRttError::PolicyDisabled)
        );
        // T2: gate false.
        assert!(!manager.may_send_zero_rtt());
    }

    #[test]
    fn note_server_accepted_rejects_from_not_attempted() {
        let mut manager = ZeroRttManager::new(ZeroRttPolicy::Allowed);
        let err = manager.note_server_accepted().unwrap_err();
        assert!(matches!(
            err,
            ZeroRttError::IllegalStatusTransition {
                from: ZeroRttStatus::NotAttempted,
                ..
            }
        ));
    }

    #[test]
    fn note_server_rejected_rejects_from_accepted() {
        let mut manager = ZeroRttManager::new(ZeroRttPolicy::Allowed);
        manager.prepare_resumption(TICKET).expect("ok");
        manager.note_server_accepted().expect("ok");
        let err = manager.note_server_rejected().unwrap_err();
        assert!(matches!(
            err,
            ZeroRttError::IllegalStatusTransition {
                from: ZeroRttStatus::Accepted,
                ..
            }
        ));
    }

    #[test]
    fn record_ack_in_accepted_is_ok() {
        let mut manager = ZeroRttManager::new(ZeroRttPolicy::Allowed);
        manager.prepare_resumption(TICKET).expect("ok");
        manager.note_server_accepted().expect("ok");
        assert!(manager.record_zero_rtt_ack_received().is_ok());
    }

    #[test]
    fn record_ack_in_not_attempted_is_ok() {
        let manager = ZeroRttManager::new(ZeroRttPolicy::Allowed);
        // No prior egress + no rejection — receiving an ack here is
        // weird but not the §4.6.2 violation specifically (that's
        // gated on Rejected status). Caller can decide.
        assert!(manager.record_zero_rtt_ack_received().is_ok());
    }
}
