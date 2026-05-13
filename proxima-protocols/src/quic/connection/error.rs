//! Errors returned by the connection state machine.

use core::fmt;

use arrayvec::ArrayVec;

use crate::quic::crypto::aead::AeadError;
use crate::quic::crypto::expand_label::ExpandError;
use crate::quic::crypto::packet_protection::PacketProtectionError;
use crate::quic::frame::DecodeError as FrameDecodeError;
use crate::quic::packet::header::DecodeError as HeaderDecodeError;
use crate::quic::packet_number::PacketNumberError;
use crate::quic::sized;
use crate::quic::time::Instant;
use crate::quic::tls::TlsError;

/// Maximum versions reported in [`ConnectionError::VersionNegotiationRequested`] /
/// `VersionNegotiationFailed`. Sourced from
/// `proxima-quic-proto.toml [connection].vn_max_offered_versions`.
pub const MAX_VN_OFFERED_VERSIONS: usize = sized::CONNECTION_VN_MAX_OFFERED_VERSIONS;

/// Errors the connection state machine surfaces to its caller.
///
/// Three buckets per the C11 FSM design pass:
///
/// - **Caller-bug-ish** (`IllegalInState`, `NonMonotonicTime`) — the
///   caller misused the API.
/// - **Wire / crypto** (`Tls`, `PacketProtection`, `Frame`, `Header`,
///   `Aead`, `PacketNumber`) — the protocol layer found bad bytes.
/// - **Capacity** (`BufferTooSmall`, `EventOverflow`) — bounded buffer
///   exceeded.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ConnectionError {
    /// Method invoked in a state where it has no defined behavior.
    /// `current` is a static label for the state at the time of call.
    IllegalInState {
        current: &'static str,
        method: &'static str,
    },

    /// `now` was earlier than the most recent `now` seen by an ingress
    /// entry point. Caller's three documented responses:
    /// 1. tear down handshake (rare),
    /// 2. clamp on Established (most common),
    /// 3. log-and-drop best-effort.
    NonMonotonicTime {
        previous: Instant,
        supplied: Instant,
    },

    /// TLS provider returned an error.
    Tls(TlsError),

    /// Packet protect/unprotect failed.
    PacketProtection(PacketProtectionError),

    /// Underlying AEAD failed.
    Aead(AeadError),

    /// Frame parse failed.
    Frame(FrameDecodeError),

    /// Packet header parse failed.
    Header(HeaderDecodeError),

    /// Packet number assign/encode/decode failed.
    PacketNumber(PacketNumberError),

    /// Initial-keys derivation failed.
    InitialKeys(ExpandError),

    /// Caller-supplied buffer was too small for the next outbound
    /// datagram or for the requested operation.
    BufferTooSmall { needed: usize },

    /// The TlsEventSink filled past its bounded capacity. Maps to
    /// `CONNECTION_CLOSE` with `INTERNAL_ERROR (0x01)` per the
    /// TlsProvider resolution.
    EventOverflow,

    /// Peer's first datagram or handshake violated a protocol invariant
    /// (e.g. wrong epoch order, malformed Initial header).
    ProtocolViolation { reason: &'static str },

    /// Peer sent more stream / connection-level data than the limits we
    /// advertised (RFC 9000 §4.1 / §4.5) — MUST be a connection error
    /// of type `FLOW_CONTROL_ERROR` (0x03).
    FlowControlError { reason: &'static str },

    /// Inbound STREAM data was within the advertised flow-control limit
    /// but exceeded the receiver's actual reassembly buffer capacity.
    /// This is a **local** under-provisioning condition — the receiver
    /// advertised more credit than its `recv_buffer_inline_bytes` +
    /// pending fragment cap could hold. The packet MUST NOT be ACKed
    /// (peer will retransmit once loss detection fires; by then the
    /// application should have drained); the connection is otherwise
    /// healthy and MUST NOT be closed.
    ///
    /// Callers (e.g. the I/O facade) treat this as a silent-drop per
    /// the same RFC 9000 §10.3 reasoning as undecryptable packets —
    /// not the peer's fault, no wire response.
    TransientRecvBufferFull {
        stream_id: u64,
        dropped_bytes: usize,
    },

    /// All concurrent bidi/uni stream slots for this direction are in use,
    /// OR the peer's cumulative stream limit (initial_max_streams_* /
    /// MAX_STREAMS) blocks opening another local stream. The connection is
    /// healthy — caller must wait for in-flight streams to complete and, for
    /// the cumulative limit, for the peer to issue a MAX_STREAMS frame.
    /// NOT a protocol violation; never close the connection on this error.
    PeerStreamLimitExhausted,

    /// Component path not yet implemented in this build.
    /// Used to firewall later-component capability behind a clean
    /// error rather than a panic during C11→C27 incremental landing.
    NotImplemented { component: &'static str },

    /// Server sent a Version Negotiation packet (RFC 9000 §6); at
    /// least one of the offered versions is supported by us. Caller's
    /// decision whether to restart the handshake with one of them.
    VersionNegotiationRequested {
        offered: ArrayVec<u32, MAX_VN_OFFERED_VERSIONS>,
    },

    /// Server sent a Version Negotiation packet but none of the
    /// offered versions are supported. Caller's only path forward is
    /// to fail the connection.
    VersionNegotiationFailed {
        offered: ArrayVec<u32, MAX_VN_OFFERED_VERSIONS>,
    },
}

impl fmt::Display for ConnectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IllegalInState { current, method } => {
                write!(f, "{method} illegal in state {current}")
            }
            Self::NonMonotonicTime { previous, supplied } => write!(
                f,
                "non-monotonic time: previous {:?} supplied {:?}",
                previous, supplied
            ),
            Self::Tls(err) => write!(f, "tls: {err}"),
            Self::PacketProtection(err) => write!(f, "packet protection: {err:?}"),
            Self::Aead(err) => write!(f, "aead: {err:?}"),
            Self::Frame(err) => write!(f, "frame decode: {err:?}"),
            Self::Header(err) => write!(f, "header decode: {err:?}"),
            Self::PacketNumber(err) => write!(f, "packet number: {err:?}"),
            Self::InitialKeys(err) => write!(f, "initial keys: {err:?}"),
            Self::BufferTooSmall { needed } => write!(f, "buffer too small (needed {needed})"),
            Self::EventOverflow => f.write_str("tls event sink overflowed"),
            Self::ProtocolViolation { reason } => write!(f, "protocol violation: {reason}"),
            Self::FlowControlError { reason } => write!(f, "flow control error: {reason}"),
            Self::TransientRecvBufferFull {
                stream_id,
                dropped_bytes,
            } => {
                write!(
                    f,
                    "transient recv buffer full on stream {stream_id} (dropped {dropped_bytes} bytes — packet not ACKed; peer will retransmit)"
                )
            }
            Self::PeerStreamLimitExhausted => f.write_str(
                "peer stream limit exhausted: wait for open streams to close and peer MAX_STREAMS",
            ),
            Self::NotImplemented { component } => write!(f, "not implemented: {component}"),
            Self::VersionNegotiationRequested { offered } => {
                write!(
                    f,
                    "version negotiation requested ({} offered)",
                    offered.len()
                )
            }
            Self::VersionNegotiationFailed { offered } => {
                write!(
                    f,
                    "version negotiation failed ({} offered, none supported)",
                    offered.len()
                )
            }
        }
    }
}

impl From<TlsError> for ConnectionError {
    fn from(err: TlsError) -> Self {
        Self::Tls(err)
    }
}

impl From<PacketProtectionError> for ConnectionError {
    fn from(err: PacketProtectionError) -> Self {
        Self::PacketProtection(err)
    }
}

impl From<AeadError> for ConnectionError {
    fn from(err: AeadError) -> Self {
        Self::Aead(err)
    }
}

impl From<FrameDecodeError> for ConnectionError {
    fn from(err: FrameDecodeError) -> Self {
        Self::Frame(err)
    }
}

impl From<HeaderDecodeError> for ConnectionError {
    fn from(err: HeaderDecodeError) -> Self {
        Self::Header(err)
    }
}

impl From<PacketNumberError> for ConnectionError {
    fn from(err: PacketNumberError) -> Self {
        Self::PacketNumber(err)
    }
}

impl From<ExpandError> for ConnectionError {
    fn from(err: ExpandError) -> Self {
        Self::InitialKeys(err)
    }
}

/// Short result type used internally by the FSM dispatcher.
pub type ConnectionResult<T> = Result<T, ConnectionError>;
