//! Sans-IO TLS 1.3 provider abstraction for the QUIC state machine.
//!
//! The proto layer never validates a certificate, never compares an
//! ALPN string, never derives a key — those concerns belong to the
//! TLS stack the provider wraps. The trait surface in this module
//! defines exactly the interface QUIC needs to drive a TLS 1.3
//! handshake forward: bytes in via `read_handshake`, bytes out via
//! `write_handshake`, secrets + events pushed via [`TlsEventSink`].
//!
//! Concrete providers ship in sibling crates: [`MockTlsProvider`] for
//! deterministic tests (this crate), `RustlsProvider` (std-tier
//! `proxima-quic-rustls`), `EmbeddedTlsProvider` (tier-1
//! `proxima-quic-embedded-tls`). The proto crate compiles with **zero**
//! TLS-stack knowledge — only the trait surface here.
//!
//! Decisive shape per `/research-rigor` self-play tournament documented
//! in [`docs/proxima-quic/edges.md`].
//!
//! [`MockTlsProvider`]: mock::MockTlsProvider
//! [`docs/proxima-quic/edges.md`]: ../../docs/proxima-quic/edges.md

use core::fmt;
use core::ops::Range;

use crate::quic::side::Side;

pub mod event;
pub mod secrets;

#[cfg(any(test, feature = "quic-mock-tls"))]
pub mod mock;

#[cfg(feature = "quic-tls-rustls")]
pub mod rustls_provider;

pub use event::{
    AlertLevel, InlineEventSink, MAX_INLINE_EVENTS, TLS_EVENT_INLINE_BYTES, TlsEvent, TlsEventKind,
    TlsEventOwned, TlsEventSink,
};
pub use secrets::{
    Direction, DirectionalKeys, Epoch, EpochSecrets, HeaderKeyMaterial, PacketKeyMaterial,
};
#[cfg(feature = "quic-tls-rustls")]
pub use secrets::{ExternalHeaderKey, ExternalPacketKey};

/// Closed error enum returned by all [`TlsProvider`] methods.
///
/// Three distinguishable buckets:
///
/// - **Wire-protocol** (`Alert`, `UnexpectedMessage`, `DecryptError`,
///   `BadCertificate`, `UnsupportedExtension`, `NoApplicationProtocol`)
///   — peer violated TLS; proto layer maps to `CONNECTION_CLOSE` with
///   `0x0100 | description` per RFC 9001 §4.8.
/// - **Local bug or budget** (`BufferTooSmall`) — caller has work to do
///   before retrying.
/// - **Asynchronous state** (`NotReady`, `Aborted`) — provider is not
///   in a state to satisfy this call right now.
///
/// `ProviderInternal(u16)` is reserved for provider-specific failures
/// that don't map cleanly to one of the above (e.g. internal sequencing
/// errors). The `u16` discriminates per-provider error codes; the proto
/// layer surfaces it as an opaque value in tracing.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum TlsError {
    /// Peer alerted us; map to `CONNECTION_CLOSE` per RFC 9001 §4.8.
    Alert { level: AlertLevel, description: u8 },
    /// Peer sent an out-of-sequence handshake message.
    UnexpectedMessage,
    /// Provider failed to decrypt the supplied input.
    DecryptError,
    /// Certificate chain validation failed.
    BadCertificate,
    /// Peer asked for an extension we cannot honour.
    UnsupportedExtension,
    /// ALPN negotiation produced no overlap.
    NoApplicationProtocol,
    /// Caller's output buffer was too small to hold the next handshake
    /// flight; resize and retry. No bytes are consumed from the
    /// provider's internal queue.
    BufferTooSmall { needed: usize },
    /// Provider is not in a state to satisfy this call yet. The proto
    /// layer treats this as "loop back and wait for more data" rather
    /// than an error condition.
    NotReady,
    /// Provider has been aborted via [`TlsProvider::abort`].
    Aborted { code: u16 },
    /// Provider-specific internal error; opaque `u16` discriminant.
    ProviderInternal(u16),
}

impl fmt::Display for TlsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Alert { level, description } => {
                write!(f, "tls alert (level {level:?}, description {description})")
            }
            Self::UnexpectedMessage => f.write_str("tls unexpected message"),
            Self::DecryptError => f.write_str("tls decrypt error"),
            Self::BadCertificate => f.write_str("tls bad certificate"),
            Self::UnsupportedExtension => f.write_str("tls unsupported extension"),
            Self::NoApplicationProtocol => f.write_str("tls no application protocol overlap"),
            Self::BufferTooSmall { needed } => {
                write!(f, "tls buffer too small (needed {needed} bytes)")
            }
            Self::NotReady => f.write_str("tls provider not ready"),
            Self::Aborted { code } => write!(f, "tls aborted (code {code})"),
            Self::ProviderInternal(code) => write!(f, "tls provider internal error ({code})"),
        }
    }
}

/// Sans-IO AEAD interface a [`TlsProvider`] exposes for packet protect /
/// unprotect.
///
/// The trait is intentionally minimal — the proto layer composes header
/// protection + nonce build + AEAD here. Future implementations may wire
/// hardware crypto, FFI to system providers, etc.
pub trait AeadProvider {
    /// Encrypt `plaintext_then_tag` in place; on success returns the
    /// total length written (plaintext + 16-byte AEAD tag).
    ///
    /// `plaintext_then_tag[..plaintext_len]` holds the plaintext on
    /// entry; on success `[..plaintext_len + 16]` holds ciphertext +
    /// tag.
    ///
    /// # Errors
    /// Returns [`TlsError::BufferTooSmall`] if the buffer is shorter
    /// than `plaintext_len + 16`.
    fn seal_in_place(
        &self,
        nonce: &[u8; 12],
        aad: &[u8],
        plaintext_then_tag: &mut [u8],
        plaintext_len: usize,
    ) -> Result<usize, TlsError>;

    /// Decrypt `ciphertext_with_tag` in place; on success returns the
    /// plaintext length (ciphertext_len - 16).
    ///
    /// # Errors
    /// Returns [`TlsError::DecryptError`] on authentication failure.
    fn open_in_place(
        &self,
        nonce: &[u8; 12],
        aad: &[u8],
        ciphertext_with_tag: &mut [u8],
    ) -> Result<usize, TlsError>;
}

/// The sans-IO TLS 1.3 provider interface QUIC drives.
///
/// **Provider-perspective** semantics throughout: `Direction::Local` =
/// "I protect outbound with this", `Direction::Remote` = "I unprotect
/// inbound with this", regardless of whether `SIDE` is `Client` or
/// `Server`. This makes the symmetry of TLS 1.3 explicit: server's RX
/// key IS client's TX key.
pub trait TlsProvider: Sized {
    /// Provider-specific configuration (TLS roots, cert chain, ALPN
    /// list, server cert resolver, session ticket store, etc.).
    /// `Clone` not `Copy` because rustls / embedded-tls configs are
    /// `Arc`-wrapped.
    type Config: Clone;

    /// Provider-specific AEAD implementation reached via `aead_for`.
    type Aead: AeadProvider;

    /// Connection side baked into the provider at construction.
    const SIDE: Side;

    /// Construct a provider from its config and the local transport
    /// parameters as TLS-extension wire bytes.
    ///
    /// # Errors
    /// Returns [`TlsError::ProviderInternal`] on provider init failure.
    fn new(config: Self::Config, local_transport_params: &[u8]) -> Result<Self, TlsError>;

    /// Derive Initial-packet keys from the client's first DCID per
    /// RFC 9001 §5.2.
    ///
    /// Associated function — Initial keys are a fixed-salt + DCID
    /// derivation; no stateful TLS context required. Folded in from a
    /// separate Factory trait per the TlsProvider resolution.
    ///
    /// # Errors
    /// Returns [`TlsError::ProviderInternal`] on key-derivation failure.
    fn initial_keys(destination_cid: &[u8]) -> Result<EpochSecrets, TlsError>;

    /// Pump TLS handshake bytes outward.
    ///
    /// Writes contiguous bytes starting at `out[0]`; returns the
    /// written range. **Atomic-or-nothing**: on
    /// [`TlsError::BufferTooSmall`] no bytes are consumed from the
    /// provider's internal queue — caller can resize and retry.
    ///
    /// # Errors
    /// See [`TlsError`].
    fn write_handshake(&mut self, epoch: Epoch, out: &mut [u8]) -> Result<Range<usize>, TlsError>;

    /// Feed TLS handshake bytes inward.
    ///
    /// The provider MAY call `sink.on_event` and `sink.on_new_secrets`
    /// zero or more times synchronously during this call. Multi-event
    /// flights (Handshake-secrets + peer transport parameters +
    /// HandshakeConfirmed) MUST be expressed via multiple sink calls
    /// in the correct order — pull-style APIs are not provided.
    ///
    /// # Errors
    /// See [`TlsError`].
    fn read_handshake(
        &mut self,
        epoch: Epoch,
        input: &[u8],
        sink: &mut dyn TlsEventSink,
    ) -> Result<(), TlsError>;

    /// Locally derive the next-generation 1-RTT keys per RFC 9001
    /// §6.1 (HKDF-Expand-Label with label "quic ku" over the current
    /// application secret).
    ///
    /// The returned [`EpochSecrets`] MUST have `epoch == Epoch::Application`
    /// and `generation == current_app_generation + 1`. The caller
    /// stages these via [`crate::quic::key_update::KeyUpdateManager::stage_pending`]
    /// and swaps to them either on the next outbound 1-RTT packet
    /// (locally-initiated) or on the next inbound packet whose
    /// key-phase bit flipped (peer-initiated; the proactive
    /// derivation per RFC §6.3).
    ///
    /// # Errors
    /// Returns [`TlsError::NotReady`] if the provider is still in the
    /// handshake.
    fn initiate_key_update(&mut self) -> Result<EpochSecrets, TlsError>;

    /// Return the AEAD for a given epoch + generation + direction.
    ///
    /// # Errors
    /// Returns [`TlsError::NotReady`] if the requested keys have not
    /// been installed yet (e.g. Application before HandshakeConfirmed).
    fn aead_for(
        &self,
        epoch: Epoch,
        generation: u8,
        direction: Direction,
    ) -> Result<&Self::Aead, TlsError>;

    /// Is the TLS handshake still in progress?
    fn is_handshaking(&self) -> bool;

    /// Has TLS handshake confirmation occurred (RFC 9001 §4.1.2)?
    fn is_confirmed(&self) -> bool;

    /// Abort the provider with the given QUIC transport error code.
    /// The proto layer maps this to `CONNECTION_CLOSE` per RFC 9000
    /// §20.1. Drop runs Rust destructors; no `close()` method.
    fn abort(&mut self, code: u16);

    /// Hand the provider a server-issued Retry token to attach to the
    /// next ClientHello's Initial packet Token field (per RFC 9000
    /// §17.2.5 and §17.2.2). Called by the connection state machine
    /// after `InitialState::reset_for_retry` has accepted a Retry
    /// packet.
    ///
    /// Default implementation is a no-op so providers that don't
    /// support Retry (e.g. server-only or test stubs) compile clean.
    /// Providers that DO support Retry override this to record the
    /// token + ensure the next emitted ClientHello includes it.
    ///
    /// The token bytes are server-private and opaque to the provider;
    /// it must not parse, modify, or interpret them.
    fn set_retry_token(&mut self, token: &[u8]) {
        let _ = token;
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn tls_error_alert_carries_discriminants() {
        let err = TlsError::Alert {
            level: AlertLevel::Fatal,
            description: 40,
        };
        match err {
            TlsError::Alert { level, description } => {
                assert_eq!(level, AlertLevel::Fatal);
                assert_eq!(description, 40);
            }
            _ => panic!("expected alert variant"),
        }
    }

    #[test]
    fn tls_error_buffer_too_small_carries_needed_size() {
        let err = TlsError::BufferTooSmall { needed: 1234 };
        match err {
            TlsError::BufferTooSmall { needed } => assert_eq!(needed, 1234),
            _ => panic!("expected BufferTooSmall variant"),
        }
    }
}
