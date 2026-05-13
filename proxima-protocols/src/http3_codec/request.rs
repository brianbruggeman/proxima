//! Per-request state machines per [RFC 9114 §4.1] — request stream
//! lifecycle.
//!
//! Server-side: receives HEADERS then optional DATA frames then
//! optional trailing HEADERS then FIN. Symmetric on the response side
//! (server emits HEADERS + DATA + optional trailing HEADERS + FIN).
//!
//! Client-side mirror — sends request, receives response.
//!
//! Per the proxima-h3 axiom B (guiding-principles overlay): typestate
//! type parameters would be the ideal shape since there's exactly one
//! path through the states. The runtime FSM here is a discriminated
//! enum carried inside the per-request slot — typestate gets us
//! compile-time enforcement only across owned `Request<S>` handles,
//! which the server-side `Connection` can't easily provide (it
//! manages many concurrent requests in a stream-id-keyed table). So
//! the FSM lives inside the connection entry; the enum + exhaustive
//! match on every transition gives equivalent compile-time
//! enforcement of legal transitions.
//!
//! [RFC 9114 §4.1]: https://www.rfc-editor.org/rfc/rfc9114#section-4.1

use alloc::vec::Vec;

use crate::http3_codec::frame;
use crate::http3_codec::qpack::decoder::DecodedField;

/// Receive-side state for one H3 request stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecvState {
    /// No bytes consumed yet on this stream.
    Idle,
    /// HEADERS frame parsed + decoded; awaiting DATA or trailing
    /// HEADERS or FIN.
    HeadersReceived { headers: Vec<DecodedField> },
    /// Receiving DATA frames; trailing HEADERS or FIN ends the body. The
    /// server accumulates `body_so_far` to hand the full request body to its
    /// handler; the client leaves it empty (it delivers body bytes per-frame
    /// via `ResponseData` and never reads the accumulation).
    BodyReceiving {
        headers: Vec<DecodedField>,
        body_so_far: Vec<u8>,
    },
    /// Received an optional trailing HEADERS after the body.
    TrailersReceived {
        headers: Vec<DecodedField>,
        body: Vec<u8>,
        trailers: Vec<DecodedField>,
    },
    /// FIN received; the stream is closed inbound. The previous body /
    /// trailers state has already been emitted to the caller.
    Done,
}

/// Send-side state for one H3 request stream (server: response;
/// client: request).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SendState {
    /// No frames emitted yet.
    Idle,
    /// HEADERS emitted; awaiting data or close.
    HeadersSent,
    /// DATA frames in flight; trailing HEADERS or close ends the body.
    BodyStreaming { bytes_sent: u64 },
    /// Trailing HEADERS emitted; only close remains.
    TrailersSent,
    /// FIN emitted; outbound side complete.
    Done,
}

/// Errors from per-request state-machine transitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RequestError {
    /// Frame received in a state where it isn't permitted by RFC 9114
    /// §4.1 (e.g. DATA before HEADERS, second HEADERS before body, etc).
    UnexpectedFrame,
    /// Bytes received after FIN.
    AfterFin,
    /// Frame codec error.
    Frame(frame::FrameError),
}

impl From<frame::FrameError> for RequestError {
    fn from(err: frame::FrameError) -> Self {
        Self::Frame(err)
    }
}

impl RecvState {
    /// Construct in `Idle`.
    #[must_use]
    pub const fn new() -> Self {
        Self::Idle
    }

    /// Borrow currently-decoded headers (None until HEADERS received).
    #[must_use]
    pub fn headers(&self) -> Option<&[DecodedField]> {
        match self {
            Self::HeadersReceived { headers }
            | Self::BodyReceiving { headers, .. }
            | Self::TrailersReceived { headers, .. } => Some(headers),
            _ => None,
        }
    }

    /// Borrow accumulated body bytes (empty until BodyReceiving).
    #[must_use]
    pub fn body(&self) -> &[u8] {
        match self {
            Self::BodyReceiving { body_so_far, .. } => body_so_far,
            Self::TrailersReceived { body, .. } => body,
            _ => &[],
        }
    }
}

impl Default for RecvState {
    fn default() -> Self {
        Self::new()
    }
}

impl SendState {
    /// Construct in `Idle`.
    #[must_use]
    pub const fn new() -> Self {
        Self::Idle
    }
}

impl Default for SendState {
    fn default() -> Self {
        Self::new()
    }
}
