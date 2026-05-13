//! Per-stream Send + Recv state per [RFC 9000 §3].
//!
//! [RFC 9000 §3]: https://www.rfc-editor.org/rfc/rfc9000#section-3

use arrayvec::ArrayVec;

use crate::quic::sized;

/// Maximum inline bytes per stream's send buffer. Sourced from
/// `proxima-quic-proto.toml [streams].send_buffer_inline_bytes`
/// (override via `PROXIMA_QUIC_PROTO_STREAMS_SEND_BUFFER_INLINE_BYTES`
/// at build time).
pub const STREAM_SEND_INLINE: usize = sized::STREAMS_SEND_BUFFER_INLINE_BYTES;

/// Maximum inline bytes per stream's recv reassembly buffer. Sourced
/// from `proxima-quic-proto.toml [streams].recv_buffer_inline_bytes`.
pub const STREAM_RECV_INLINE: usize = sized::STREAMS_RECV_BUFFER_INLINE_BYTES;

/// Sender-side stream state per RFC 9000 §3.1.
//
// `large_enum_variant` is intentional: the `Send` variant carries the
// inline send buffer (16 KiB) per principle 11 (sans-IO state owns
// data inline; no `Box<dyn>`). The enum's stack size dominates the
// per-stream cost — acceptable because StreamTable caps the total
// stream count and transitions are infrequent.
#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)]
#[non_exhaustive]
pub enum SendState {
    /// Sender has not yet queued data.
    Ready,
    /// Sender has data queued or in flight.
    Send {
        send_buffer: ArrayVec<u8, STREAM_SEND_INLINE>,
        offset_next: u64,
        offset_acked: u64,
        /// `close_send` was called while bytes were still buffered.
        /// On the next `collect_stream_emission` that drains the buffer
        /// the emission carries `fin=true` and the state advances to
        /// `DataSent`. Without this flag the stream would either lose
        /// the buffered bytes (transition before drain) or emit endless
        /// duplicate FINs (transition before fin emission).
        fin_pending: bool,
    },
    /// Caller has called `close_send`; FIN is queued or in flight.
    DataSent {
        offset_final: u64,
        offset_acked: u64,
    },
    /// Peer has acknowledged everything through `offset_final`.
    /// Terminal (success path).
    DataRecvd { offset_final: u64 },
    /// Caller called [`SendState::reset`] OR peer sent STOP_SENDING;
    /// RESET_STREAM frame is queued or in flight (RFC 9000 §3.1).
    ResetSent { offset_final: u64, error_code: u64 },
    /// Peer acknowledged the RESET_STREAM frame. Terminal (reset path).
    ResetRecvd { offset_final: u64, error_code: u64 },
}

/// Errors from sender-side state transitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SendStateError {
    /// `reset` called from a state that cannot transition to ResetSent
    /// — DataRecvd (success-terminal) and ResetRecvd (reset-terminal)
    /// are both terminal per RFC §3.1.
    AlreadyTerminal,
    /// `note_reset_acked` called from a non-ResetSent state.
    NotResetSent,
}

impl SendState {
    /// Construct a fresh `Ready` state.
    #[must_use]
    pub const fn new() -> Self {
        Self::Ready
    }

    /// Largest offset ever observed by this side (next-byte index).
    #[must_use]
    pub fn current_offset(&self) -> u64 {
        match self {
            Self::Ready => 0,
            Self::Send { offset_next, .. } => *offset_next,
            Self::DataSent { offset_final, .. } | Self::DataRecvd { offset_final } => *offset_final,
            Self::ResetSent { offset_final, .. } | Self::ResetRecvd { offset_final, .. } => {
                *offset_final
            }
        }
    }

    /// Has the caller already called `close_send`?
    #[must_use]
    pub fn is_closed(&self) -> bool {
        matches!(self, Self::DataSent { .. } | Self::DataRecvd { .. })
    }

    /// Has the stream entered a reset state (sent or acknowledged)?
    #[must_use]
    pub fn is_reset(&self) -> bool {
        matches!(self, Self::ResetSent { .. } | Self::ResetRecvd { .. })
    }

    /// Has the stream reached a terminal state (DataRecvd success OR
    /// ResetRecvd)?
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::DataRecvd { .. } | Self::ResetRecvd { .. })
    }

    /// Transition to `ResetSent` per RFC 9000 §3.1. Called when the
    /// caller invokes the local-reset API OR when a STOP_SENDING
    /// frame arrives from the peer (§3.5 — the receiver requests
    /// the sender abort).
    ///
    /// Records `error_code` and the current offset as `offset_final`
    /// (the largest offset the sender has produced; subsequent peer
    /// state machines rely on it to know how much data was promised
    /// before reset).
    ///
    /// # Errors
    ///
    /// [`SendStateError::AlreadyTerminal`] when called from
    /// `DataRecvd` or `ResetRecvd` — terminal states are immutable.
    pub fn reset(&mut self, error_code: u64) -> Result<(), SendStateError> {
        self.reset_with_final_cap(error_code, u64::MAX)
    }

    /// Reset variant that clips `offset_final` to `max_final` —
    /// callers with access to per-stream flow-control state pass
    /// `credit_send` so the emitted RESET_STREAM final_size cannot
    /// exceed the peer's advertised MAX_STREAM_DATA (RFC 9000 §4.5
    /// + §19.4 forbid `final_size > credit_send`).
    ///
    /// # Errors
    /// See [`Self::reset`].
    pub fn reset_with_final_cap(
        &mut self,
        error_code: u64,
        max_final: u64,
    ) -> Result<(), SendStateError> {
        let offset_final = core::cmp::min(self.current_offset(), max_final);
        match self {
            Self::Ready | Self::Send { .. } | Self::DataSent { .. } => {
                *self = Self::ResetSent {
                    offset_final,
                    error_code,
                };
                Ok(())
            }
            Self::ResetSent { .. } => {
                // Idempotent — already ResetSent. Per RFC §3.5 a
                // sender SHOULD send only one RESET_STREAM; later
                // calls are no-ops with the existing error_code.
                Ok(())
            }
            Self::DataRecvd { .. } | Self::ResetRecvd { .. } => {
                Err(SendStateError::AlreadyTerminal)
            }
        }
    }

    /// Transition `ResetSent → ResetRecvd` when the peer ACKs the
    /// RESET_STREAM frame.
    ///
    /// # Errors
    ///
    /// [`SendStateError::NotResetSent`] when called from any other state.
    pub fn note_reset_acked(&mut self) -> Result<(), SendStateError> {
        match self {
            Self::ResetSent {
                offset_final,
                error_code,
            } => {
                *self = Self::ResetRecvd {
                    offset_final: *offset_final,
                    error_code: *error_code,
                };
                Ok(())
            }
            _ => Err(SendStateError::NotResetSent),
        }
    }
}

impl Default for SendState {
    fn default() -> Self {
        Self::new()
    }
}

/// Receiver-side stream state per RFC 9000 §3.2.
//
// Same `large_enum_variant` rationale as `SendState`.
#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)]
#[non_exhaustive]
pub enum RecvState {
    /// Receiving STREAM bytes; no FIN seen yet.
    ///
    /// `recv_buffer` holds the CONTIGUOUS-bytes head — `[0, offset_next)`.
    /// `reassembly` holds any out-of-order fragments awaiting earlier
    /// bytes to fill the gap.
    Recv {
        recv_buffer: ArrayVec<u8, STREAM_RECV_INLINE>,
        offset_next: u64,
        reassembly: super::reassembly::ReassemblyQueue,
    },
    /// FIN seen; receiver knows the final size but may still have
    /// gaps in the reassembly buffer. The reassembly queue carries
    /// any out-of-order fragments that arrived BEFORE the FIN — they
    /// drain into `recv_buffer` as the missing earlier bytes arrive.
    ///
    /// `offset_next` tracks the highest contiguous byte offset the
    /// receiver has assembled into `recv_buffer`; it advances on
    /// each successful reassembly insert. `apply_inbound_stream`
    /// uses it for the reassembly cursor; `read_stream` reads from
    /// the head of `recv_buffer` WITHOUT advancing `offset_next`
    /// (the two are orthogonal: `offset_next` tracks "fully
    /// assembled up through", `recv_buffer.len()` tracks "buffered
    /// but not yet drained by the app").
    SizeKnown {
        recv_buffer: ArrayVec<u8, STREAM_RECV_INLINE>,
        offset_final: u64,
        offset_next: u64,
        reassembly: super::reassembly::ReassemblyQueue,
    },
    /// All bytes through `offset_final` are present in the buffer
    /// (or already drained).
    DataRecvd { offset_final: u64 },
    /// Caller has drained the recv buffer past `offset_final`.
    /// Terminal (success path).
    DataRead { offset_final: u64 },
    /// Peer sent RESET_STREAM with `offset_final` and `error_code`;
    /// receiver has registered the reset (RFC 9000 §3.2).
    ResetRecvd { offset_final: u64, error_code: u64 },
    /// Caller has acknowledged the reset (consumed the error code).
    /// Terminal (reset path).
    ResetRead { offset_final: u64, error_code: u64 },
}

/// Errors from receiver-side state transitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RecvStateError {
    /// `note_inbound_reset` called from a terminal state (DataRead
    /// or ResetRead) — terminal states are immutable.
    AlreadyTerminal,
    /// `note_reset_read` called from a non-ResetRecvd state.
    NotResetRecvd,
    /// A RESET_STREAM / FIN declared a final size that conflicts with
    /// data already received or a prior final-size declaration per
    /// RFC 9000 §4.5 — MUST be a connection error of type
    /// `FINAL_SIZE_ERROR` (0x06).
    FinalSizeConflict { declared: u64, observed: u64 },
}

impl RecvState {
    /// Construct a fresh `Recv` state.
    #[must_use]
    pub fn new() -> Self {
        Self::Recv {
            recv_buffer: ArrayVec::new(),
            offset_next: 0,
            reassembly: super::reassembly::ReassemblyQueue::new(),
        }
    }

    /// Has the receiver seen the FIN bit?
    #[must_use]
    pub fn fin_seen(&self) -> bool {
        matches!(
            self,
            Self::SizeKnown { .. } | Self::DataRecvd { .. } | Self::DataRead { .. }
        )
    }

    /// Has the receiver registered a peer reset (RESET_STREAM)?
    #[must_use]
    pub fn is_reset(&self) -> bool {
        matches!(self, Self::ResetRecvd { .. } | Self::ResetRead { .. })
    }

    /// Has the receiver reached a terminal state (DataRead success OR
    /// ResetRead)?
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::DataRead { .. } | Self::ResetRead { .. })
    }

    /// Record an inbound RESET_STREAM frame per RFC 9000 §3.2.
    /// Transitions any non-terminal state into `ResetRecvd`.
    ///
    /// Per RFC §19.4 the `offset_final` field is the largest offset
    /// the sender promised before reset; receivers MUST treat it as
    /// the new final size of the stream. The receiver MAY discard
    /// any buffered data per §3.2.
    ///
    /// # Errors
    ///
    /// [`RecvStateError::AlreadyTerminal`] when called from
    /// `DataRead` or `ResetRead`.
    pub fn note_inbound_reset(
        &mut self,
        offset_final: u64,
        error_code: u64,
    ) -> Result<(), RecvStateError> {
        match self {
            Self::Recv { offset_next, .. } => {
                // RFC 9000 §4.5 — "An endpoint that receives a
                // RESET_STREAM frame can ignore the error code …
                // however … If a RESET_STREAM frame is received for
                // a … stream that has already been partially received,
                // an endpoint MUST check that the Final Size field
                // matches the final size already known."
                //
                // FINAL_SIZE_ERROR if final_size is LOWER than data
                // we've already accepted (peer can't un-send bytes).
                if offset_final < *offset_next {
                    return Err(RecvStateError::FinalSizeConflict {
                        declared: offset_final,
                        observed: *offset_next,
                    });
                }
                *self = Self::ResetRecvd {
                    offset_final,
                    error_code,
                };
                Ok(())
            }
            Self::SizeKnown {
                offset_final: known_final,
                ..
            }
            | Self::DataRecvd {
                offset_final: known_final,
                ..
            } => {
                // FIN already established a final size — the reset
                // MUST agree with it.
                if offset_final != *known_final {
                    return Err(RecvStateError::FinalSizeConflict {
                        declared: offset_final,
                        observed: *known_final,
                    });
                }
                *self = Self::ResetRecvd {
                    offset_final,
                    error_code,
                };
                Ok(())
            }
            Self::ResetRecvd {
                offset_final: prior_final,
                ..
            } => {
                // Subsequent reset MUST have the same final_size.
                if offset_final != *prior_final {
                    return Err(RecvStateError::FinalSizeConflict {
                        declared: offset_final,
                        observed: *prior_final,
                    });
                }
                // Idempotent — receiver SHOULD ignore subsequent
                // RESET_STREAM frames for the same stream per RFC §3.5.
                Ok(())
            }
            Self::DataRead { .. } | Self::ResetRead { .. } => Err(RecvStateError::AlreadyTerminal),
        }
    }

    /// Transition `ResetRecvd → ResetRead` when the caller drains the
    /// reset notification (consumes the error code).
    ///
    /// # Errors
    ///
    /// [`RecvStateError::NotResetRecvd`] when called from any other state.
    pub fn note_reset_read(&mut self) -> Result<(), RecvStateError> {
        match self {
            Self::ResetRecvd {
                offset_final,
                error_code,
            } => {
                *self = Self::ResetRead {
                    offset_final: *offset_final,
                    error_code: *error_code,
                };
                Ok(())
            }
            _ => Err(RecvStateError::NotResetRecvd),
        }
    }
}

impl Default for RecvState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn ready_default_state() {
        let send = SendState::new();
        assert!(matches!(send, SendState::Ready));
        assert!(!send.is_closed());
        assert_eq!(send.current_offset(), 0);
    }

    #[test]
    fn data_sent_reports_closed() {
        let send = SendState::DataSent {
            offset_final: 100,
            offset_acked: 50,
        };
        assert!(send.is_closed());
        assert_eq!(send.current_offset(), 100);
    }

    #[test]
    fn recv_default_no_fin_seen() {
        let recv = RecvState::new();
        assert!(matches!(recv, RecvState::Recv { .. }));
        assert!(!recv.fin_seen());
    }

    #[test]
    fn size_known_reports_fin_seen() {
        let recv = RecvState::SizeKnown {
            recv_buffer: ArrayVec::new(),
            offset_final: 100,
            offset_next: 0,
            reassembly: crate::quic::streams::ReassemblyQueue::new(),
        };
        assert!(recv.fin_seen());
    }

    // ---- C12.4 — RESET_STREAM / STOP_SENDING transitions ----

    #[test]
    fn send_reset_from_ready_records_zero_offset() {
        let mut send = SendState::new();
        send.reset(0x42).expect("ok");
        assert!(send.is_reset());
        assert!(!send.is_terminal());
        match send {
            SendState::ResetSent {
                offset_final,
                error_code,
            } => {
                assert_eq!(offset_final, 0);
                assert_eq!(error_code, 0x42);
            }
            other => panic!("expected ResetSent, got {other:?}"),
        }
    }

    #[test]
    fn send_reset_from_send_carries_current_offset() {
        let mut send = SendState::Send {
            send_buffer: ArrayVec::new(),
            offset_next: 64,
            offset_acked: 32,
            fin_pending: false,
        };
        send.reset(0x7).expect("ok");
        match send {
            SendState::ResetSent {
                offset_final,
                error_code,
            } => {
                assert_eq!(offset_final, 64);
                assert_eq!(error_code, 0x7);
            }
            other => panic!("expected ResetSent, got {other:?}"),
        }
    }

    #[test]
    fn send_reset_from_data_recvd_rejected_as_terminal() {
        let mut send = SendState::DataRecvd { offset_final: 100 };
        assert_eq!(send.reset(0x1), Err(SendStateError::AlreadyTerminal));
    }

    #[test]
    fn send_reset_idempotent_from_reset_sent() {
        let mut send = SendState::ResetSent {
            offset_final: 50,
            error_code: 0x42,
        };
        // Second reset call should be a no-op (already in ResetSent).
        send.reset(0x99).expect("idempotent ok");
        // Original error_code preserved per RFC §3.5 — sender SHOULD
        // send only one RESET_STREAM.
        match send {
            SendState::ResetSent {
                offset_final,
                error_code,
            } => {
                assert_eq!(offset_final, 50);
                assert_eq!(error_code, 0x42);
            }
            other => panic!("expected ResetSent unchanged, got {other:?}"),
        }
    }

    #[test]
    fn send_note_reset_acked_transitions_to_terminal() {
        let mut send = SendState::ResetSent {
            offset_final: 50,
            error_code: 0x42,
        };
        send.note_reset_acked().expect("ok");
        assert!(send.is_terminal());
        match send {
            SendState::ResetRecvd {
                offset_final,
                error_code,
            } => {
                assert_eq!(offset_final, 50);
                assert_eq!(error_code, 0x42);
            }
            other => panic!("expected ResetRecvd, got {other:?}"),
        }
    }

    #[test]
    fn send_note_reset_acked_rejects_non_reset_state() {
        let mut send = SendState::new();
        assert_eq!(send.note_reset_acked(), Err(SendStateError::NotResetSent));
    }

    #[test]
    fn recv_note_inbound_reset_from_recv_records_state() {
        let mut recv = RecvState::new();
        recv.note_inbound_reset(120, 0x42).expect("ok");
        assert!(recv.is_reset());
        match recv {
            RecvState::ResetRecvd {
                offset_final,
                error_code,
            } => {
                assert_eq!(offset_final, 120);
                assert_eq!(error_code, 0x42);
            }
            other => panic!("expected ResetRecvd, got {other:?}"),
        }
    }

    #[test]
    fn recv_note_inbound_reset_from_data_read_rejected_as_terminal() {
        let mut recv = RecvState::DataRead { offset_final: 100 };
        assert_eq!(
            recv.note_inbound_reset(120, 0x1),
            Err(RecvStateError::AlreadyTerminal)
        );
    }

    #[test]
    fn recv_note_inbound_reset_idempotent_from_reset_recvd_same_final_size() {
        let mut recv = RecvState::ResetRecvd {
            offset_final: 50,
            error_code: 0x42,
        };
        // Same final_size → idempotent (RFC §3.5); original fields preserved.
        recv.note_inbound_reset(50, 0x99)
            .expect("idempotent with same final_size");
        match recv {
            RecvState::ResetRecvd {
                offset_final,
                error_code,
            } => {
                assert_eq!(offset_final, 50);
                assert_eq!(error_code, 0x42);
            }
            other => panic!("expected unchanged ResetRecvd, got {other:?}"),
        }
    }

    #[test]
    fn recv_note_inbound_reset_conflicting_final_size_returns_error() {
        let mut recv = RecvState::ResetRecvd {
            offset_final: 50,
            error_code: 0x42,
        };
        // Different final_size → FINAL_SIZE_ERROR per RFC 9000 §4.5.
        let err = recv.note_inbound_reset(99, 0x99).expect_err("conflict");
        assert!(
            matches!(
                err,
                RecvStateError::FinalSizeConflict {
                    declared: 99,
                    observed: 50
                }
            ),
            "expected FinalSizeConflict, got {err:?}"
        );
    }

    #[test]
    fn recv_note_reset_read_transitions_to_terminal() {
        let mut recv = RecvState::ResetRecvd {
            offset_final: 50,
            error_code: 0x42,
        };
        recv.note_reset_read().expect("ok");
        assert!(recv.is_terminal());
        match recv {
            RecvState::ResetRead {
                offset_final,
                error_code,
            } => {
                assert_eq!(offset_final, 50);
                assert_eq!(error_code, 0x42);
            }
            other => panic!("expected ResetRead, got {other:?}"),
        }
    }

    #[test]
    fn recv_note_reset_read_rejects_non_reset_state() {
        let mut recv = RecvState::new();
        assert_eq!(recv.note_reset_read(), Err(RecvStateError::NotResetRecvd));
    }

    #[test]
    fn send_stop_sending_triggers_reset_per_rfc_3_5() {
        // RFC 9000 §3.5: peer sends STOP_SENDING → sender SHOULD
        // reset the stream with the error code from STOP_SENDING.
        // We model this by the caller invoking SendState::reset with
        // the peer's error_code; no separate API needed because the
        // sender FSM end-result is identical.
        let mut send = SendState::Send {
            send_buffer: ArrayVec::new(),
            offset_next: 32,
            offset_acked: 0,
            fin_pending: false,
        };
        send.reset(0xBEEF)
            .expect("peer STOP_SENDING translates to local reset");
        assert!(send.is_reset());
    }
}
