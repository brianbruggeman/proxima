//! Push-only event sink fed by [`TlsProvider::read_handshake`].
//!
//! TLS 1.3 flights can emit multiple "things the proto layer cares
//! about" in a single `read_handshake` call (Handshake secrets +
//! peer transport parameters + maybe Application secrets, all in
//! the same TLS record buffer). A pull-style `take_*` API can lose
//! events if the proto layer forgets to drain it; the push sink
//! makes the lossless ordering native.
//!
//! Events carry borrowed slices into the transcript with a
//! `'transcript` lifetime — the sink callback executes synchronously
//! within `read_handshake`, so the borrow is sound. Sinks that need
//! to retain a slice copy into an inline owned buffer
//! ([`TlsEventOwned`]) keeping the proto layer alloc-free on the hot
//! path.
//!
//! [`TlsProvider::read_handshake`]: super::TlsProvider::read_handshake

use arrayvec::ArrayVec;

use super::secrets::EpochSecrets;

/// Maximum bytes copied into [`TlsEventOwned::PeerTransportParameters`]
/// (or AlpnNegotiated / PeerCertificate / SessionTicket variants).
///
/// 256 B comfortably exceeds the ~150 B canonical TP set; oversized
/// inputs are truncated and the sink raises [`TlsEventOwned::Overflow`].
pub const TLS_EVENT_INLINE_BYTES: usize = 256;

/// Maximum number of events the [`InlineEventSink`] buffers per
/// `read_handshake` call. TLS 1.3 single-flight peaks at ~4 events
/// (HandshakeData + PeerTransportParameters + HandshakeConfirmed +
/// SessionTicket) — 8 is comfortable headroom.
///
/// On overflow the [`InlineEventSink`] sets the `overflowed` flag and
/// the proto layer maps it to `CONNECTION_CLOSE` with `INTERNAL_ERROR
/// (0x01)` per the TlsProvider resolution.
pub const MAX_INLINE_EVENTS: usize = 8;

/// TLS alert severity per RFC 8446 §6.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum AlertLevel {
    Warning,
    Fatal,
}

/// Event yielded by the TLS provider during `read_handshake`.
///
/// All borrowed-slice variants carry data with a `'transcript` lifetime
/// scoped to the sink callback. Sinks that need to retain the bytes
/// copy into [`TlsEventOwned`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TlsEvent<'transcript> {
    /// The provider consumed handshake data; no further state-machine
    /// action required (used for keep-alive / progress telemetry).
    HandshakeDataReceived,

    /// TLS handshake has been confirmed per RFC 9001 §4.1.2 — peer
    /// has cryptographically demonstrated possession of the application
    /// traffic secret. Cleared by the proto layer to advance to
    /// `EstablishedState`.
    HandshakeConfirmed,

    /// Peer's transport parameters per RFC 9000 §18.
    PeerTransportParameters(&'transcript [u8]),

    /// ALPN protocol negotiated per RFC 7301; the bytes are the wire
    /// representation of the selected protocol (e.g. `b"h3"`).
    AlpnNegotiated(&'transcript [u8]),

    /// Peer's certificate chain in TLS-wire format (one entry per
    /// `CertificateEntry`).
    PeerCertificate(&'transcript [u8]),

    /// TLS 1.3 NewSessionTicket bytes; the application MAY persist
    /// for 0-RTT resumption.
    SessionTicket(&'transcript [u8]),
}

/// Owned form of [`TlsEvent`] suitable for buffering past the
/// `read_handshake` lifetime.
///
/// Inline-allocated up to [`TLS_EVENT_INLINE_BYTES`] per slice variant;
/// no heap. The proto layer drains the buffer once `read_handshake`
/// returns.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum TlsEventOwned {
    HandshakeDataReceived,
    HandshakeConfirmed,
    PeerTransportParameters(ArrayVec<u8, TLS_EVENT_INLINE_BYTES>),
    AlpnNegotiated(ArrayVec<u8, TLS_EVENT_INLINE_BYTES>),
    PeerCertificate(ArrayVec<u8, TLS_EVENT_INLINE_BYTES>),
    SessionTicket(ArrayVec<u8, TLS_EVENT_INLINE_BYTES>),
    /// A borrowed slice exceeded [`TLS_EVENT_INLINE_BYTES`].
    /// The proto layer treats this as a fatal violation per the
    /// TlsProvider resolution.
    Overflow {
        event_kind: TlsEventKind,
    },
}

/// Kind discriminant of [`TlsEvent`] / [`TlsEventOwned`].
///
/// Used to disambiguate [`TlsEventOwned::Overflow`] and to script mock
/// events in tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TlsEventKind {
    HandshakeDataReceived,
    HandshakeConfirmed,
    PeerTransportParameters,
    AlpnNegotiated,
    PeerCertificate,
    SessionTicket,
}

impl TlsEvent<'_> {
    /// The kind discriminant of `self`.
    #[must_use]
    pub const fn kind(&self) -> TlsEventKind {
        match self {
            Self::HandshakeDataReceived => TlsEventKind::HandshakeDataReceived,
            Self::HandshakeConfirmed => TlsEventKind::HandshakeConfirmed,
            Self::PeerTransportParameters(_) => TlsEventKind::PeerTransportParameters,
            Self::AlpnNegotiated(_) => TlsEventKind::AlpnNegotiated,
            Self::PeerCertificate(_) => TlsEventKind::PeerCertificate,
            Self::SessionTicket(_) => TlsEventKind::SessionTicket,
        }
    }
}

impl TlsEventOwned {
    /// The kind discriminant of `self`.
    #[must_use]
    pub const fn kind(&self) -> TlsEventKind {
        match self {
            Self::HandshakeDataReceived => TlsEventKind::HandshakeDataReceived,
            Self::HandshakeConfirmed => TlsEventKind::HandshakeConfirmed,
            Self::PeerTransportParameters(_) => TlsEventKind::PeerTransportParameters,
            Self::AlpnNegotiated(_) => TlsEventKind::AlpnNegotiated,
            Self::PeerCertificate(_) => TlsEventKind::PeerCertificate,
            Self::SessionTicket(_) => TlsEventKind::SessionTicket,
            Self::Overflow { event_kind } => *event_kind,
        }
    }
}

/// Sink callback for events the [`TlsProvider`] pushes during
/// `read_handshake`.
///
/// The trait is `dyn`-compatible by design: `read_handshake` accepts
/// `&mut dyn TlsEventSink`, so any backend can implement it without
/// monomorphising the provider on the sink type.
///
/// Methods are infallible at the trait level — overflow handling is
/// recorded in the sink's internal state (e.g. [`InlineEventSink`]'s
/// `overflowed` flag) and surfaced by the consumer after the
/// `read_handshake` call returns.
///
/// [`TlsProvider`]: super::TlsProvider
pub trait TlsEventSink {
    /// Record a non-secrets event (transcript-borrowed).
    fn on_event(&mut self, event: TlsEvent<'_>);

    /// Record newly-installed AEAD + header-protection keys for an
    /// epoch and key-update generation.
    fn on_new_secrets(&mut self, secrets: EpochSecrets);
}

/// Bounded, inline-allocated event sink usable from any caller of
/// `read_handshake`.
///
/// Stores up to [`MAX_INLINE_EVENTS`] events and 2 secret installs per
/// call. Overflow is recorded as a flag that the proto layer checks
/// after the call returns.
#[derive(Debug, Default)]
pub struct InlineEventSink {
    events: ArrayVec<TlsEventOwned, MAX_INLINE_EVENTS>,
    secrets: ArrayVec<EpochSecrets, 2>,
    overflowed: bool,
}

impl InlineEventSink {
    /// Construct an empty sink.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            events: ArrayVec::new_const(),
            secrets: ArrayVec::new_const(),
            overflowed: false,
        }
    }

    /// Has any push exceeded the inline buffer (events OR secrets OR
    /// inline-byte caps)? When true the proto layer MUST emit
    /// `CONNECTION_CLOSE` with `INTERNAL_ERROR (0x01)` per the
    /// TlsProvider resolution.
    #[must_use]
    pub const fn overflowed(&self) -> bool {
        self.overflowed
    }

    /// Slice over buffered events in insertion order.
    #[must_use]
    pub fn events(&self) -> &[TlsEventOwned] {
        &self.events
    }

    /// Slice over buffered secret installs in insertion order.
    #[must_use]
    pub fn secrets(&self) -> &[EpochSecrets] {
        &self.secrets
    }

    /// Drain and return the buffered events, leaving the sink empty
    /// for re-use.
    pub fn take_events(&mut self) -> ArrayVec<TlsEventOwned, MAX_INLINE_EVENTS> {
        core::mem::take(&mut self.events)
    }

    /// Drain and return the buffered secret installs, leaving the
    /// sink empty for re-use.
    pub fn take_secrets(&mut self) -> ArrayVec<EpochSecrets, 2> {
        core::mem::take(&mut self.secrets)
    }

    /// Reset both buffers AND the overflow flag.
    pub fn clear(&mut self) {
        self.events.clear();
        self.secrets.clear();
        self.overflowed = false;
    }

    fn copy_payload(
        &mut self,
        src: &[u8],
        kind: TlsEventKind,
    ) -> Option<ArrayVec<u8, TLS_EVENT_INLINE_BYTES>> {
        if src.len() > TLS_EVENT_INLINE_BYTES {
            self.overflowed = true;
            let owned = TlsEventOwned::Overflow { event_kind: kind };
            if self.events.try_push(owned).is_err() {
                self.overflowed = true;
            }
            return None;
        }
        let mut buf: ArrayVec<u8, TLS_EVENT_INLINE_BYTES> = ArrayVec::new();
        // try_extend_from_slice cannot fail because the length check above bounds it.
        buf.try_extend_from_slice(src).ok();
        Some(buf)
    }
}

impl TlsEventSink for InlineEventSink {
    fn on_event(&mut self, event: TlsEvent<'_>) {
        let owned = match event {
            TlsEvent::HandshakeDataReceived => TlsEventOwned::HandshakeDataReceived,
            TlsEvent::HandshakeConfirmed => TlsEventOwned::HandshakeConfirmed,
            TlsEvent::PeerTransportParameters(bytes) => {
                let Some(buf) = self.copy_payload(bytes, TlsEventKind::PeerTransportParameters)
                else {
                    return;
                };
                TlsEventOwned::PeerTransportParameters(buf)
            }
            TlsEvent::AlpnNegotiated(bytes) => {
                let Some(buf) = self.copy_payload(bytes, TlsEventKind::AlpnNegotiated) else {
                    return;
                };
                TlsEventOwned::AlpnNegotiated(buf)
            }
            TlsEvent::PeerCertificate(bytes) => {
                let Some(buf) = self.copy_payload(bytes, TlsEventKind::PeerCertificate) else {
                    return;
                };
                TlsEventOwned::PeerCertificate(buf)
            }
            TlsEvent::SessionTicket(bytes) => {
                let Some(buf) = self.copy_payload(bytes, TlsEventKind::SessionTicket) else {
                    return;
                };
                TlsEventOwned::SessionTicket(buf)
            }
        };
        if self.events.try_push(owned).is_err() {
            self.overflowed = true;
        }
    }

    fn on_new_secrets(&mut self, secrets: EpochSecrets) {
        if self.secrets.try_push(secrets).is_err() {
            self.overflowed = true;
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::super::secrets::{DirectionalKeys, Epoch, HeaderKeyMaterial, PacketKeyMaterial};
    use super::*;
    use crate::quic::crypto::initial_keys::{QUIC_HP_LEN, QUIC_IV_LEN, QUIC_KEY_LEN};

    fn sample_secrets(epoch: Epoch, generation: u8) -> EpochSecrets {
        let keys = DirectionalKeys {
            packet: PacketKeyMaterial::Aes128Gcm {
                key: [0xAA; QUIC_KEY_LEN],
                iv: [0xBB; QUIC_IV_LEN],
            },
            header: HeaderKeyMaterial::Aes128 {
                hp: [0xCC; QUIC_HP_LEN],
            },
        };
        EpochSecrets {
            epoch,
            generation,
            local: keys.clone(),
            remote: keys,
        }
    }

    #[test]
    fn inline_sink_records_events_in_order() {
        let mut sink = InlineEventSink::new();
        sink.on_event(TlsEvent::HandshakeDataReceived);
        sink.on_event(TlsEvent::PeerTransportParameters(&[0x00, 0x01, 0x02]));
        sink.on_event(TlsEvent::HandshakeConfirmed);
        let events = sink.events();
        assert_eq!(events.len(), 3);
        assert!(matches!(events[0], TlsEventOwned::HandshakeDataReceived));
        assert!(matches!(
            events[1],
            TlsEventOwned::PeerTransportParameters(_)
        ));
        assert!(matches!(events[2], TlsEventOwned::HandshakeConfirmed));
        assert!(!sink.overflowed());
    }

    #[test]
    fn inline_sink_copies_transcript_borrowed_payload() {
        let mut sink = InlineEventSink::new();
        let borrowed = [0xDE, 0xAD, 0xBE, 0xEF];
        sink.on_event(TlsEvent::PeerTransportParameters(&borrowed));
        let TlsEventOwned::PeerTransportParameters(stored) = &sink.events()[0] else {
            panic!("expected PeerTransportParameters");
        };
        assert_eq!(stored.as_slice(), &borrowed);
    }

    #[test]
    fn inline_sink_records_secrets_independently() {
        let mut sink = InlineEventSink::new();
        sink.on_new_secrets(sample_secrets(Epoch::Handshake, 0));
        sink.on_new_secrets(sample_secrets(Epoch::Application, 0));
        let secrets = sink.secrets();
        assert_eq!(secrets.len(), 2);
        assert_eq!(secrets[0].epoch, Epoch::Handshake);
        assert_eq!(secrets[1].epoch, Epoch::Application);
    }

    #[test]
    fn inline_sink_marks_overflow_on_oversize_payload() {
        let mut sink = InlineEventSink::new();
        let huge = [0xAA; TLS_EVENT_INLINE_BYTES + 1];
        sink.on_event(TlsEvent::PeerTransportParameters(&huge));
        assert!(sink.overflowed());
        assert!(matches!(
            sink.events()[0],
            TlsEventOwned::Overflow {
                event_kind: TlsEventKind::PeerTransportParameters
            }
        ));
    }

    #[test]
    fn inline_sink_marks_overflow_on_too_many_events() {
        let mut sink = InlineEventSink::new();
        for _ in 0..(MAX_INLINE_EVENTS + 1) {
            sink.on_event(TlsEvent::HandshakeDataReceived);
        }
        assert!(sink.overflowed());
        assert_eq!(sink.events().len(), MAX_INLINE_EVENTS);
    }

    #[test]
    fn inline_sink_take_events_drains_buffer() {
        let mut sink = InlineEventSink::new();
        sink.on_event(TlsEvent::HandshakeConfirmed);
        let drained = sink.take_events();
        assert_eq!(drained.len(), 1);
        assert!(sink.events().is_empty());
    }

    #[test]
    fn tls_event_kind_round_trips_through_owned() {
        let borrowed = [0xAB];
        let event = TlsEvent::PeerTransportParameters(&borrowed);
        let mut sink = InlineEventSink::new();
        sink.on_event(event);
        assert_eq!(
            sink.events()[0].kind(),
            TlsEventKind::PeerTransportParameters
        );
        assert_eq!(event.kind(), TlsEventKind::PeerTransportParameters);
    }
}
