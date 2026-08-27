#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

#[cfg(feature = "std")]
pub mod factory;
pub mod share_buf;

use alloc::vec::Vec;
use core::net::SocketAddr;

use bytes::{Bytes, BytesMut};
use proxima_core::ProximaError;
use thiserror::Error;

#[cfg(feature = "std")]
use std::cell::RefCell;

#[cfg(feature = "std")]
pub use factory::{
    BytesPassthroughCodecFactory, BytesPassthroughDynCodec, CodecBuildFuture, CodecFactory,
    CodecRegistry, DynCodec, DynCodecFactory, DynCodecHandle, JsonCodecFactory, JsonDynCodec,
};
pub use share_buf::ShareBuf;

// per-thread scratch buffer reused across simd-json decodes. simd-json
// mutates its input in place, so the codec must own a mutable copy. a
// thread_local Vec amortizes the allocation across requests on the
// same worker without needing a synchronized BufferPool dance.
#[cfg(feature = "std")]
thread_local! {
    static DECODE_SCRATCH: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

#[cfg(feature = "std")]
pub(crate) fn decode_through_scratch<T>(bytes: &[u8]) -> Result<T, ProximaError>
where
    T: serde::de::DeserializeOwned,
{
    DECODE_SCRATCH.with(|cell| {
        let mut buf = cell.borrow_mut();
        buf.clear();
        buf.extend_from_slice(bytes);
        simd_json::serde::from_slice(&mut buf)
            .map_err(|error| ProximaError::Decode(format!("json: {error}")))
    })
}

/// Request/response full-message codec — schema-driven, single owned
/// Input/Output value pair (JSON, protobuf, CBOR, etc.). The historical
/// `Codec` trait — renamed when the codec-trait family landed so the
/// sibling shapes (`FrameCodec`, `StatefulCodec`, `WireCodec`) read as
/// peers rather than specializations.
pub trait MessageCodec: Send + Sync + 'static {
    type Input: Send + Sync;
    type Output: Send + Sync;

    fn decode_input(&self, bytes: &[u8]) -> Result<Self::Input, ProximaError>;
    fn encode_output(&self, output: &Self::Output) -> Result<Bytes, ProximaError>;

    fn content_type(&self) -> &str {
        "application/octet-stream"
    }
}

/// Stateless, borrow-only frame codec for length-delimited wire formats
/// (HTTP/1, HTTP/2, HTTP/3, QUIC, gRPC, WebSocket). Each `parse_frame`
/// call returns a borrowed view of the input slice plus the number of
/// bytes consumed — no allocation on the inner loop. Each `encode_frame`
/// call appends to a caller-owned `Vec<u8>` so the framer can hand the
/// composed bytes to a transport without an extra copy.
///
/// Compared to [`MessageCodec`]: the input and output of a `FrameCodec`
/// are the SAME frame shape (parser and serializer roundtrip the wire),
/// whereas a `MessageCodec` may decode into one schema type and encode
/// from another. Compared to [`StatefulCodec`]: there is no per-codec
/// state — same parser called twice yields the same result.
pub trait FrameCodec: Send + Sync + 'static {
    type Frame<'a>;
    type Error: core::error::Error + Send + Sync + 'static;

    fn parse_frame<'a>(&self, buf: &'a [u8]) -> Result<(Self::Frame<'a>, usize), Self::Error>;

    fn encode_frame(&self, frame: &Self::Frame<'_>, dest: &mut Vec<u8>) -> Result<(), Self::Error>;
}

/// Stateful encoder/decoder factory for codecs that carry per-session
/// state (HPACK's dynamic table, QPACK's encoder/decoder streams). The
/// trait itself is a factory: `new_encoder` and `new_decoder` mint
/// distinct instances so callers control state ownership and lifetime.
///
/// Compared to [`FrameCodec`]: the encoder and decoder must be split
/// because they need to track wire state (table indices, eviction)
/// across many calls. A `FrameCodec` can be called from any number of
/// threads on the same `&Self`; a `StatefulCodec`'s encoder/decoder are
/// per-session and not necessarily `Sync`.
pub trait StatefulCodec: Send + Sync + 'static {
    type Encoder: Send;
    type Decoder: Send;

    fn new_encoder(&self) -> Result<Self::Encoder, ProximaError>;
    fn new_decoder(&self) -> Result<Self::Decoder, ProximaError>;
}

/// Wire-level field iterator for codecs that walk a buffer field-at-a-
/// time (protobuf, future thrift/avro). Different from [`FrameCodec`]
/// in that there is no notion of a "complete frame" — a protobuf message
/// is just a sequence of (tag, value) pairs, parsed one at a time, with
/// the caller deciding when to stop.
pub trait WireCodec: Send + Sync + 'static {
    type Field<'a>;
    type Error: core::error::Error + Send + Sync + 'static;

    fn parse_field<'a>(&self, buf: &'a [u8]) -> Result<(Self::Field<'a>, usize), Self::Error>;

    fn iter_fields<'a>(
        &self,
        buf: &'a [u8],
    ) -> impl Iterator<Item = Result<Self::Field<'a>, Self::Error>>;
}

/// A message paired with the peer [`SocketAddr`] it travels with.
/// Connectionless transports (UDP, DTLS-over-UDP, QUIC datagram
/// frames) have no persistent peer identity the way a TCP stream
/// does — the peer address rides with every packet, both directions,
/// or a reply has nowhere to go. `decode` fills `peer` from where the
/// packet arrived; `encode` reads it as where to send.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Addressed<M> {
    pub peer: SocketAddr,
    pub message: M,
}

/// Codec for connectionless, atomic wire messages — UDP-carried
/// protocols (DNS, memcached's UDP mode, syslog, RADIUS, a
/// bencode-over-UDP or fixed-binary trackerless-DHT-style datagram
/// protocol) where one `recvfrom` hands the codec exactly one whole
/// message.
///
/// `Datagram` composes two things [`MessageCodec`] and [`FrameCodec`]
/// each have only one of: [`MessageCodec`]'s zero-copy borrow is
/// missing (owned-only `Input`/`Output`, `decode_input(&self, bytes:
/// &[u8])` can't return a view into `bytes`) — that would regress
/// `proxima-protocols`'s `dns` module's lazy `Name` (19× slower with
/// an eagerly-collected `Vec<&[u8]>` per that module's own bench
/// history). [`FrameCodec`] has the zero-copy `Frame<'a>`
/// GAT this trait borrows the same shape from, but its
/// `parse_frame`/[`FrameError::Incomplete`] contract means "read more
/// bytes and retry" — wrong for a datagram, where the kernel already
/// delivered the packet atomically and a short buffer can never grow.
/// Neither trait carries a peer address.
///
/// So: `Message<'a>` is [`FrameCodec::Frame`]'s zero-copy GAT (an
/// owned-only impl sets `Message<'a> = Owned` and ignores `'a`, same
/// escape hatch [`FrameCodec`] impls use); [`Addressed`] is the one
/// addition neither sibling trait has. Everything else — the hard
/// per-call `Result`, the caller-owned `&mut Vec<u8>` encode
/// destination — matches [`FrameCodec::parse_frame`]/`encode_frame`
/// exactly, so a `Datagram` impl reads like a `FrameCodec` impl to
/// anyone who already knows that trait.
pub trait Datagram: Send + Sync + 'static {
    /// Zero-copy impls borrow the packet buffer (same shape as
    /// [`FrameCodec::Frame`]); owned impls set `Message<'a> = Owned`
    /// and ignore the lifetime.
    type Message<'a>;
    /// Every failure is a hard, per-packet error — see the trait docs.
    type Error: core::error::Error + Send + Sync + 'static;

    /// Decode the WHOLE packet as one message — a UDP datagram is
    /// delivered atomically, so there is no partial-message state to
    /// carry between calls. `peer` is the address the packet arrived
    /// from; it rides along in the returned [`Addressed`] so a reply
    /// knows where to go. A buffer too short or malformed to hold a
    /// complete message is [`Self::Error`] — never "read more."
    ///
    /// Reassembling a message split across multiple datagrams (e.g.
    /// memcached UDP mode's request-id/sequence header) is a stateful
    /// concern the caller owns, keyed by `(peer, request_id)`, ABOVE
    /// this call — out of scope for a stateless per-packet decode.
    fn decode<'a>(
        &self,
        peer: SocketAddr,
        bytes: &'a [u8],
    ) -> Result<Addressed<Self::Message<'a>>, Self::Error>;

    /// Encode one message into `dest`, a caller-owned scratch buffer
    /// that this call only appends to — no allocation inside the
    /// codec, mirroring [`FrameCodec::encode_frame`]. `addressed.peer`
    /// is the destination the transport sends `dest`'s bytes to after
    /// this call returns.
    fn encode(
        &self,
        addressed: &Addressed<Self::Message<'_>>,
        dest: &mut Vec<u8>,
    ) -> Result<(), Self::Error>;
}

pub struct BytesPassthrough;

impl MessageCodec for BytesPassthrough {
    type Input = Bytes;
    type Output = Bytes;

    fn decode_input(&self, bytes: &[u8]) -> Result<Self::Input, ProximaError> {
        Ok(Bytes::copy_from_slice(bytes))
    }

    fn encode_output(&self, output: &Self::Output) -> Result<Bytes, ProximaError> {
        Ok(Bytes::clone(output))
    }
}

#[cfg(feature = "std")]
pub struct JsonCodec<Input, Output>(core::marker::PhantomData<(Input, Output)>);

#[cfg(feature = "std")]
impl<Input, Output> Default for JsonCodec<Input, Output> {
    fn default() -> Self {
        Self(core::marker::PhantomData)
    }
}

#[cfg(feature = "std")]
impl<Input, Output> JsonCodec<Input, Output> {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

#[cfg(feature = "std")]
impl<Input, Output> MessageCodec for JsonCodec<Input, Output>
where
    Input: serde::de::DeserializeOwned + Send + Sync + 'static,
    Output: serde::Serialize + Send + Sync + 'static,
{
    type Input = Input;
    type Output = Output;

    fn decode_input(&self, bytes: &[u8]) -> Result<Self::Input, ProximaError> {
        decode_through_scratch(bytes)
    }

    fn encode_output(&self, output: &Self::Output) -> Result<Bytes, ProximaError> {
        simd_json::serde::to_vec(output)
            .map(Bytes::from)
            .map_err(|error| ProximaError::Encode(format!("json: {error}")))
    }

    fn content_type(&self) -> &str {
        "application/json"
    }
}

/// Per-frame limits for length-delimited framing. Lets a consumer cap
/// frame size tighter than a transport default and reject zero-length
/// frames — a zero-length frame carries no payload and only serves as a
/// free keepalive for an attacker holding a connection slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameLimits {
    /// Largest accepted payload length. Real-world: a JSON RPC daemon
    /// caps this at 16 MiB so a malformed length prefix can't trick the
    /// server into a multi-GB allocation.
    pub max_frame_bytes: usize,
    /// When true, a declared length of 0 is rejected as
    /// [`FrameError::ZeroLength`].
    pub reject_zero_len: bool,
}

impl FrameLimits {
    /// Permissive default cap: 64 MiB, zero-length allowed. Transports
    /// with a tighter requirement pass their own via [`Self::new`].
    pub const DEFAULT_MAX_FRAME_BYTES: usize = 64 * 1024 * 1024;

    #[must_use]
    pub const fn new(max_frame_bytes: usize, reject_zero_len: bool) -> Self {
        Self {
            max_frame_bytes,
            reject_zero_len,
        }
    }
}

impl Default for FrameLimits {
    fn default() -> Self {
        Self {
            max_frame_bytes: Self::DEFAULT_MAX_FRAME_BYTES,
            reject_zero_len: false,
        }
    }
}

/// `[u32 BE len][payload]` length-delimited [`FrameCodec`]. Zero-copy:
/// `parse_frame` returns a borrowed `&[u8]` view into the caller's
/// buffer plus the total bytes consumed (header + payload). The IO loop
/// that owns the read buffer (a listener or driver) reads more on
/// [`FrameError::Incomplete`] and retries — keeping this codec sans-IO.
///
/// ```
/// use proxima_codec::{FrameCodec, FrameError, FrameLimits, LengthDelimitedCodec};
///
/// let codec = LengthDelimitedCodec::new(FrameLimits::new(16 * 1024 * 1024, true));
///
/// let mut wire = Vec::new();
/// codec.encode_frame(&&b"{\"op\":\"ping\"}"[..], &mut wire)?;
///
/// // one whole frame plus the first two bytes of the next one.
/// wire.extend_from_slice(&[0, 0]);
/// let (frame, consumed) = codec.parse_frame(&wire)?;
/// assert_eq!(frame, b"{\"op\":\"ping\"}");
/// assert_eq!(consumed, 4 + frame.len());
///
/// // the read loop advances by `consumed` and asks again; a short tail is
/// // "read more bytes", not a failure.
/// assert_eq!(codec.parse_frame(&wire[consumed..]), Err(FrameError::Incomplete));
/// # Ok::<(), FrameError>(())
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct LengthDelimitedCodec {
    limits: FrameLimits,
}

impl LengthDelimitedCodec {
    /// Length-prefix header size (`u32` big-endian).
    pub const HEADER_BYTES: usize = 4;

    #[must_use]
    pub const fn new(limits: FrameLimits) -> Self {
        Self { limits }
    }

    #[must_use]
    pub const fn limits(self) -> FrameLimits {
        self.limits
    }

    /// Encode the length prefix for `payload_len` into a stack `[u8; 4]`
    /// — no allocation. The no_alloc encode path: the caller writes these
    /// 4 bytes then the payload to its own buffer / socket. The
    /// [`FrameCodec::encode_frame`] impl reuses this.
    ///
    /// # Errors
    ///
    /// [`FrameError::FrameTooLarge`] when `payload_len` exceeds
    /// `max_frame_bytes` or `u32::MAX`.
    pub fn encode_header(
        &self,
        payload_len: usize,
    ) -> Result<[u8; Self::HEADER_BYTES], FrameError> {
        if payload_len > self.limits.max_frame_bytes {
            return Err(FrameError::FrameTooLarge { len: payload_len });
        }
        let len_u32 = u32::try_from(payload_len)
            .map_err(|_| FrameError::FrameTooLarge { len: payload_len })?;
        Ok(len_u32.to_be_bytes())
    }

    /// Decode a 4-byte length prefix into the payload length, applying the
    /// configured limits (zero-length rejection + cap). Companion of
    /// [`Self::encode_header`] for read loops that read the header then the
    /// payload separately (vs. [`FrameCodec::parse_frame`], which needs the
    /// whole frame buffered).
    ///
    /// # Errors
    ///
    /// [`FrameError::ZeroLength`] / [`FrameError::FrameTooLarge`] per limits.
    pub fn decode_header(&self, bytes: [u8; Self::HEADER_BYTES]) -> Result<usize, FrameError> {
        let len = u32::from_be_bytes(bytes) as usize;
        if self.limits.reject_zero_len && len == 0 {
            return Err(FrameError::ZeroLength);
        }
        if len > self.limits.max_frame_bytes {
            return Err(FrameError::FrameTooLarge { len });
        }
        Ok(len)
    }
}

/// Errors and control signals from [`LengthDelimitedCodec`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum FrameError {
    /// The buffer does not yet hold a complete frame — read more bytes
    /// and retry. The normal partial-read signal, not a failure.
    #[error("incomplete frame")]
    Incomplete,
    /// A zero-length frame was declared while `reject_zero_len` is set.
    #[error("zero-length frame rejected")]
    ZeroLength,
    /// The declared payload length exceeds `max_frame_bytes`.
    #[error("declared frame size {len} exceeds max_frame_bytes")]
    FrameTooLarge { len: usize },
}

impl FrameCodec for LengthDelimitedCodec {
    type Frame<'a> = &'a [u8];
    type Error = FrameError;

    fn parse_frame<'a>(&self, buf: &'a [u8]) -> Result<(&'a [u8], usize), FrameError> {
        if buf.len() < Self::HEADER_BYTES {
            return Err(FrameError::Incomplete);
        }
        let mut header = [0_u8; Self::HEADER_BYTES];
        header.copy_from_slice(&buf[..Self::HEADER_BYTES]);
        let len = u32::from_be_bytes(header) as usize;
        if self.limits.reject_zero_len && len == 0 {
            return Err(FrameError::ZeroLength);
        }
        if len > self.limits.max_frame_bytes {
            return Err(FrameError::FrameTooLarge { len });
        }
        let total = Self::HEADER_BYTES + len;
        if buf.len() < total {
            return Err(FrameError::Incomplete);
        }
        Ok((&buf[Self::HEADER_BYTES..total], total))
    }

    fn encode_frame(&self, frame: &&[u8], dest: &mut Vec<u8>) -> Result<(), FrameError> {
        let header = self.encode_header(frame.len())?;
        dest.extend_from_slice(&header);
        dest.extend_from_slice(frame);
        Ok(())
    }
}

/// Delimiter-terminated [`FrameCodec`] — RESP (`\r\n`), NDJSON (`\n`),
/// pgwire startup-parameter pairs (NUL) all frame this way instead of a
/// length prefix. Zero-copy like [`LengthDelimitedCodec`]: `parse_frame`
/// returns a borrowed `&[u8]` view up to (excluding) the delimiter, plus
/// the bytes consumed (frame + delimiter).
///
/// ```
/// use proxima_codec::{DelimiterCodec, FrameCodec, FrameError};
///
/// let codec = DelimiterCodec::unbounded(b"\n");
/// let (frame, consumed) = codec.parse_frame(b"{\"op\":\"ping\"}\nrest").expect("parse");
/// assert_eq!(frame, b"{\"op\":\"ping\"}");
/// assert_eq!(consumed, frame.len() + 1);
/// assert_eq!(codec.parse_frame(b"no newline yet"), Err(FrameError::Incomplete));
/// ```
#[derive(Debug, Clone, Copy)]
pub struct DelimiterCodec {
    delimiter: &'static [u8],
    limits: FrameLimits,
}

impl DelimiterCodec {
    #[must_use]
    pub const fn new(delimiter: &'static [u8], limits: FrameLimits) -> Self {
        Self { delimiter, limits }
    }

    /// A cap of `usize::MAX` so a real long line is never rejected as
    /// oversized — the caller trusts the transport to bound total memory
    /// some other way (e.g. a connection-level read cap).
    #[must_use]
    pub const fn unbounded(delimiter: &'static [u8]) -> Self {
        Self {
            delimiter,
            limits: FrameLimits::new(usize::MAX, false),
        }
    }

    #[must_use]
    pub const fn limits(self) -> FrameLimits {
        self.limits
    }

    /// Scans `buf[search_start..]` for the delimiter, returning the
    /// absolute offset where it starts. Shared by [`Self::parse_frame`]
    /// (always `search_start = 0`) and [`DelimiterFraming::next_frame`]
    /// (a backed-up `search_start` so a resumed scan never rereads bytes
    /// already proven delimiter-free). An empty delimiter never matches —
    /// matching at every position would either return a zero-length frame
    /// forever or require special-casing; treating it as "never found"
    /// keeps the caller's Incomplete/retry loop the single source of
    /// truth instead of a second code path.
    fn find_frame_end(&self, buf: &[u8], search_start: usize) -> Option<usize> {
        if self.delimiter.is_empty() {
            return None;
        }
        let start = search_start.min(buf.len());
        memchr::memmem::find(&buf[start..], self.delimiter).map(|offset| start + offset)
    }
}

impl FrameCodec for DelimiterCodec {
    type Frame<'a> = &'a [u8];
    type Error = FrameError;

    fn parse_frame<'a>(&self, buf: &'a [u8]) -> Result<(&'a [u8], usize), FrameError> {
        match self.find_frame_end(buf, 0) {
            Some(end) => Ok((&buf[..end], end + self.delimiter.len())),
            None if buf.len() > self.limits.max_frame_bytes => {
                Err(FrameError::FrameTooLarge { len: buf.len() })
            }
            None => Err(FrameError::Incomplete),
        }
    }

    fn encode_frame(&self, frame: &&[u8], dest: &mut Vec<u8>) -> Result<(), FrameError> {
        dest.extend_from_slice(frame);
        dest.extend_from_slice(self.delimiter);
        Ok(())
    }
}

/// Frame-boundary scan state for [`DelimiterCodec`], modeled as a
/// discriminated-enum FSM rather than a struct with a `scanned: usize`
/// field sitting beside a buffer that also mutates: a struct shape lets a
/// caller (or a future refactor) advance the buffer without advancing the
/// cursor, silently reintroducing the whole-buffer rescan this type
/// exists to rule out. Folding the cursor into the enum's own variant
/// makes that bug unrepresentable — there is no state in which "the
/// buffer changed" and "the scan position" can drift apart.
#[derive(Debug, Clone)]
pub enum DelimiterFraming {
    /// No delimiter found in `buf[..scanned]` yet — a resumed scan starts
    /// at (a small backup before) `scanned`, never at 0.
    Scanning { buf: BytesMut, scanned: usize },
    /// A frame was found and handed back via [`DelimiterFraming::next_frame`]'s
    /// `Some`; `rest` is the buffered tail after the delimiter, not yet
    /// scanned. No `scanned` field — a fresh tail starts its scan at 0.
    FrameReady { frame: Bytes, rest: BytesMut },
}

impl DelimiterFraming {
    #[must_use]
    pub fn new() -> Self {
        Self::Scanning {
            buf: BytesMut::new(),
            scanned: 0,
        }
    }

    /// Infallible append — never scans. Appends to whichever buffer the
    /// current state is tracking (`Scanning::buf` or `FrameReady::rest`),
    /// so bytes that arrive before the previous frame is drained via
    /// [`Self::next_frame`] are not lost.
    #[must_use]
    pub fn push(self, chunk: &[u8]) -> Self {
        match self {
            Self::Scanning { mut buf, scanned } => {
                buf.extend_from_slice(chunk);
                Self::Scanning { buf, scanned }
            }
            Self::FrameReady { frame, mut rest } => {
                rest.extend_from_slice(chunk);
                Self::FrameReady { frame, rest }
            }
        }
    }

    /// Advances the FSM by one step. From `FrameReady`, the cached frame
    /// was already handed to the caller by the call that produced this
    /// state, so this call discards it and resumes scanning `rest` from
    /// 0. From `Scanning`, resumes at `scanned` minus a `delimiter.len() -
    /// 1` backup — mandatory, not an optimization: a delimiter split
    /// across two `push` calls has its first byte inside `buf[..scanned]`
    /// and would be silently missed without the backup.
    ///
    /// # Errors
    ///
    /// [`FrameError::FrameTooLarge`] when the buffered, still-undelimited
    /// tail exceeds `codec`'s `max_frame_bytes`.
    pub fn next_frame(self, codec: DelimiterCodec) -> Result<(Self, Option<Bytes>), FrameError> {
        let (buf, scanned) = match self {
            Self::Scanning { buf, scanned } => (buf, scanned),
            Self::FrameReady { rest, .. } => (rest, 0),
        };
        let backup = codec.delimiter.len().saturating_sub(1);
        let search_start = scanned.saturating_sub(backup);
        match codec.find_frame_end(&buf, search_start) {
            Some(end) => {
                let mut buf = buf;
                let mut rest = buf.split_off(end);
                let _ = rest.split_to(codec.delimiter.len());
                let frame = buf.freeze();
                Ok((
                    Self::FrameReady {
                        frame: frame.clone(),
                        rest,
                    },
                    Some(frame),
                ))
            }
            None if buf.len() > codec.limits.max_frame_bytes => {
                Err(FrameError::FrameTooLarge { len: buf.len() })
            }
            None => {
                let scanned = buf.len();
                Ok((Self::Scanning { buf, scanned }, None))
            }
        }
    }
}

impl Default for DelimiterFraming {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// Fixed-width 4-byte [`Datagram`] — a trivial POD message, borrowed
    /// zero-copy from the packet buffer, to exercise the trait shape
    /// without pulling in a real protocol's parsing logic.
    #[derive(Debug, Clone, Copy, Default)]
    struct FixedFourCodec;

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
    #[error("expected exactly 4 bytes, got {got_len}")]
    struct FixedFourError {
        got_len: usize,
    }

    impl Datagram for FixedFourCodec {
        type Message<'a> = &'a [u8];
        type Error = FixedFourError;

        fn decode<'a>(
            &self,
            peer: SocketAddr,
            bytes: &'a [u8],
        ) -> Result<Addressed<&'a [u8]>, FixedFourError> {
            if bytes.len() != 4 {
                return Err(FixedFourError {
                    got_len: bytes.len(),
                });
            }
            Ok(Addressed {
                peer,
                message: bytes,
            })
        }

        fn encode(
            &self,
            addressed: &Addressed<&[u8]>,
            dest: &mut Vec<u8>,
        ) -> Result<(), FixedFourError> {
            if addressed.message.len() != 4 {
                return Err(FixedFourError {
                    got_len: addressed.message.len(),
                });
            }
            dest.extend_from_slice(addressed.message);
            Ok(())
        }
    }

    /// Owned-message [`Datagram`] — demonstrates the `Message<'a> = Owned`
    /// escape hatch a future owned-`Message` protocol (a bencode-over-UDP
    /// or fixed-binary datagram format that decodes straight into an owned
    /// value, never borrows) would use: `Message<'a>` ignores `'a`
    /// entirely.
    #[derive(Debug, Clone, Copy, Default)]
    struct OwnedIdCodec;

    impl Datagram for OwnedIdCodec {
        type Message<'a> = u32;
        type Error = FixedFourError;

        fn decode(&self, peer: SocketAddr, bytes: &[u8]) -> Result<Addressed<u32>, FixedFourError> {
            let array: [u8; 4] = bytes.try_into().map_err(|_| FixedFourError {
                got_len: bytes.len(),
            })?;
            Ok(Addressed {
                peer,
                message: u32::from_be_bytes(array),
            })
        }

        fn encode(
            &self,
            addressed: &Addressed<u32>,
            dest: &mut Vec<u8>,
        ) -> Result<(), FixedFourError> {
            dest.extend_from_slice(&addressed.message.to_be_bytes());
            Ok(())
        }
    }

    #[cfg(feature = "std")]
    #[derive(Debug, serde::Deserialize, serde::Serialize, PartialEq, Eq)]
    struct Sample {
        name: alloc::string::String,
        count: u32,
    }

    #[cfg(feature = "std")]
    #[test]
    fn json_codec_roundtrips_struct() {
        let codec: JsonCodec<Sample, Sample> = JsonCodec::new();
        let original = Sample {
            name: "alice".into(),
            count: 7,
        };
        let encoded = codec
            .encode_output(&original)
            .expect("encode should succeed");
        let decoded = codec.decode_input(&encoded).expect("decode should succeed");
        assert_eq!(decoded, original);
    }

    #[cfg(feature = "std")]
    #[test]
    fn json_codec_decode_error_returns_decode_variant() {
        let codec: JsonCodec<Sample, Sample> = JsonCodec::new();
        let outcome = codec.decode_input(b"not json");
        assert!(matches!(outcome, Err(ProximaError::Decode(_))));
    }

    #[cfg(feature = "std")]
    #[test]
    fn json_content_type_is_application_json() {
        let codec: JsonCodec<Sample, Sample> = JsonCodec::new();
        assert_eq!(codec.content_type(), "application/json");
    }

    #[test]
    fn bytes_passthrough_roundtrips() {
        let codec = BytesPassthrough;
        let original = Bytes::from_static(b"\x00\x01\xff");
        let encoded = codec
            .encode_output(&original)
            .expect("encode should succeed");
        let decoded = codec.decode_input(&encoded).expect("decode should succeed");
        assert_eq!(decoded, original);
    }

    #[test]
    fn bytes_passthrough_content_type_is_octet_stream() {
        let codec = BytesPassthrough;
        assert_eq!(codec.content_type(), "application/octet-stream");
    }

    #[test]
    fn length_delimited_parses_complete_frame_zero_copy() {
        let codec = LengthDelimitedCodec::default();
        // [0,0,0,3] "abc" then trailing bytes of a second frame.
        let buf = [0, 0, 0, 3, b'a', b'b', b'c', 0, 0];
        let (frame, consumed) = codec.parse_frame(&buf).expect("parse");
        assert_eq!(frame, b"abc");
        assert_eq!(consumed, 7);
        // borrowed view points into the caller buffer (zero-copy).
        assert_eq!(frame.as_ptr(), buf[4..].as_ptr());
    }

    #[test]
    fn length_delimited_signals_incomplete_for_short_header_and_payload() {
        let codec = LengthDelimitedCodec::default();
        assert_eq!(codec.parse_frame(&[0, 0]), Err(FrameError::Incomplete));
        // header says 10 but only 3 payload bytes present.
        assert_eq!(
            codec.parse_frame(&[0, 0, 0, 10, 1, 2, 3]),
            Err(FrameError::Incomplete)
        );
    }

    #[test]
    fn length_delimited_zero_length_policy() {
        let allow = LengthDelimitedCodec::default();
        let (frame, consumed) = allow.parse_frame(&[0, 0, 0, 0]).expect("zero allowed");
        assert!(frame.is_empty());
        assert_eq!(consumed, 4);

        let reject = LengthDelimitedCodec::new(FrameLimits::new(64, true));
        assert_eq!(
            reject.parse_frame(&[0, 0, 0, 0]),
            Err(FrameError::ZeroLength)
        );
    }

    #[test]
    fn length_delimited_enforces_cap() {
        let codec = LengthDelimitedCodec::new(FrameLimits::new(16 * 1024 * 1024, true));
        let over = ((16 * 1024 * 1024_u32) + 1).to_be_bytes();
        let mut buf = [0_u8; 8];
        buf[..4].copy_from_slice(&over);
        assert_eq!(
            codec.parse_frame(&buf),
            Err(FrameError::FrameTooLarge {
                len: 16 * 1024 * 1024 + 1
            })
        );
    }

    #[test]
    fn length_delimited_encode_round_trips_and_header_is_no_alloc() {
        let codec = LengthDelimitedCodec::default();
        assert_eq!(codec.encode_header(7).expect("header"), [0, 0, 0, 7]);

        let mut dest = Vec::new();
        let payload: &[u8] = b"hello world";
        codec.encode_frame(&payload, &mut dest).expect("encode");
        let (frame, consumed) = codec.parse_frame(&dest).expect("parse back");
        assert_eq!(frame, payload);
        assert_eq!(consumed, dest.len());
    }

    #[test]
    fn length_delimited_decode_header_roundtrips_and_applies_limits() {
        let codec = LengthDelimitedCodec::default();
        let header = codec.encode_header(7).expect("encode");
        assert_eq!(codec.decode_header(header).expect("decode"), 7);

        let strict = LengthDelimitedCodec::new(FrameLimits::new(16, true));
        assert!(matches!(
            strict.decode_header([0, 0, 0, 0]),
            Err(FrameError::ZeroLength)
        ));
        assert!(matches!(
            strict.decode_header([0, 0, 0, 20]),
            Err(FrameError::FrameTooLarge { .. })
        ));
    }

    #[test]
    fn frame_error_display_is_parity_stable() {
        // the downstream consumer's incumbent wire error is exactly these strings;
        // the listener maps FrameError -> io::Error with this Display.
        use alloc::string::ToString;
        assert_eq!(FrameError::Incomplete.to_string(), "incomplete frame");
        assert_eq!(
            FrameError::ZeroLength.to_string(),
            "zero-length frame rejected"
        );
        assert_eq!(
            FrameError::FrameTooLarge { len: 99 }.to_string(),
            "declared frame size 99 exceeds max_frame_bytes"
        );
    }

    fn loopback_peer() -> SocketAddr {
        SocketAddr::from((core::net::Ipv4Addr::LOCALHOST, 11211))
    }

    #[test]
    fn datagram_decode_encode_round_trips_pod_message() {
        let codec = FixedFourCodec;
        let peer = loopback_peer();
        let packet = [1_u8, 2, 3, 4];

        let addressed = codec.decode(peer, &packet).expect("decode should succeed");
        assert_eq!(addressed.peer, peer);
        // zero-copy: the decoded message borrows straight into the caller's
        // packet buffer, no intermediate allocation.
        assert_eq!(addressed.message.as_ptr(), packet.as_ptr());

        let mut dest = Vec::new();
        codec
            .encode(&addressed, &mut dest)
            .expect("encode should succeed");
        assert_eq!(dest, packet);
    }

    #[test]
    fn datagram_malformed_buffer_is_hard_error_not_incomplete() {
        let codec = FixedFourCodec;
        let peer = loopback_peer();
        // a real recvfrom() never hands the codec a short buffer to
        // "read more" from — the kernel already delivered the whole
        // datagram. FixedFourError has no Incomplete/retry variant at
        // all (unlike FrameError above): this is the ONE call the
        // packet gets.
        let short_packet = [1_u8, 2, 3];

        let outcome = codec.decode(peer, &short_packet);
        assert_eq!(outcome.unwrap_err(), FixedFourError { got_len: 3 });
    }

    #[test]
    fn datagram_owned_message_escape_hatch_round_trips() {
        let codec = OwnedIdCodec;
        let peer = loopback_peer();
        let packet = 42_u32.to_be_bytes();

        let addressed = codec.decode(peer, &packet).expect("decode should succeed");
        assert_eq!(addressed.message, 42);

        let mut dest = Vec::new();
        codec
            .encode(&addressed, &mut dest)
            .expect("encode should succeed");
        assert_eq!(dest, packet);
    }

    // ── DelimiterCodec (REBUILD 1) ───────────────────────────────────

    #[proxima::test]
    #[case::resp_simple_string(&b"\r\n"[..], &b"+PONG\r\ntrailing"[..], &b"+PONG"[..], 7)]
    #[case::ndjson_line(&b"\n"[..], b"{\"op\":\"ping\"}\n{\"op\":\"pong\"}", b"{\"op\":\"ping\"}", 14)]
    #[case::pgwire_nul_terminated(&b"\0"[..], b"user\0rest", b"user", 5)]
    async fn delimiter_codec_parses_complete_frame_zero_copy(
        #[case] delimiter: &'static [u8],
        #[case] wire: &[u8],
        #[case] expected_frame: &[u8],
        #[case] expected_consumed: usize,
    ) {
        let codec = DelimiterCodec::unbounded(delimiter);
        let (frame, consumed) = codec.parse_frame(wire).expect("parse");
        assert_eq!(frame, expected_frame);
        assert_eq!(consumed, expected_consumed);
        assert_eq!(frame.as_ptr(), wire.as_ptr());
    }

    #[test]
    fn delimiter_codec_signals_incomplete_for_missing_delimiter() {
        let codec = DelimiterCodec::unbounded(b"\r\n");
        assert_eq!(
            codec.parse_frame(b"+PONG without a terminator"),
            Err(FrameError::Incomplete)
        );
    }

    #[test]
    fn delimiter_codec_enforces_cap_under_new_but_not_unbounded() {
        let long_line = alloc::vec![b'x'; 100];
        let bounded = DelimiterCodec::new(b"\n", FrameLimits::new(16, false));
        assert_eq!(
            bounded.parse_frame(&long_line),
            Err(FrameError::FrameTooLarge { len: 100 })
        );

        let unbounded = DelimiterCodec::unbounded(b"\n");
        assert_eq!(
            unbounded.parse_frame(&long_line),
            Err(FrameError::Incomplete)
        );
    }

    #[test]
    fn delimiter_codec_empty_delimiter_is_permanently_incomplete_never_loops() {
        let codec = DelimiterCodec::unbounded(b"");
        assert_eq!(
            codec.parse_frame(b"anything at all"),
            Err(FrameError::Incomplete)
        );
        assert_eq!(codec.parse_frame(b""), Err(FrameError::Incomplete));
    }

    #[test]
    fn delimiter_codec_encode_round_trips() {
        let codec = DelimiterCodec::unbounded(b"\r\n");
        let mut dest = Vec::new();
        let payload: &[u8] = b"+PONG";
        codec.encode_frame(&payload, &mut dest).expect("encode");
        assert_eq!(dest, b"+PONG\r\n");
        let (frame, consumed) = codec.parse_frame(&dest).expect("parse back");
        assert_eq!(frame, payload);
        assert_eq!(consumed, dest.len());
    }

    // ── DelimiterFraming (REBUILD 2) ─────────────────────────────────

    #[test]
    fn delimiter_framing_finds_delimiter_straddling_a_push_boundary() {
        // without the mandatory backup in `next_frame`, the first push's
        // trailing `\r` and the second push's leading `\n` never meet in
        // one scan window, and "+PONG" is silently swallowed.
        let codec = DelimiterCodec::unbounded(b"\r\n");
        let state = DelimiterFraming::new().push(b"+PONG\r");

        let (state, first_attempt) = state.next_frame(codec).expect("scan");
        assert!(first_attempt.is_none(), "no delimiter buffered yet");

        let state = state.push(b"\n+PANG\r\n");

        let (state, first_frame) = state.next_frame(codec).expect("scan");
        assert_eq!(first_frame, Some(Bytes::from_static(b"+PONG")));

        let (_state, second_frame) = state.next_frame(codec).expect("scan");
        assert_eq!(second_frame, Some(Bytes::from_static(b"+PANG")));
    }

    #[test]
    fn delimiter_framing_returns_none_on_partial_buffer() {
        let codec = DelimiterCodec::unbounded(b"\n");
        let state = DelimiterFraming::new().push(b"no newline yet");
        let (_state, frame) = state.next_frame(codec).expect("scan");
        assert!(frame.is_none());
    }

    #[test]
    fn delimiter_framing_drains_multiple_frames_from_one_push() {
        let codec = DelimiterCodec::unbounded(b"\n");
        let state = DelimiterFraming::new().push(b"one\ntwo\nthree\n");

        let (state, first) = state.next_frame(codec).expect("scan");
        assert_eq!(first, Some(Bytes::from_static(b"one")));
        let (state, second) = state.next_frame(codec).expect("scan");
        assert_eq!(second, Some(Bytes::from_static(b"two")));
        let (state, third) = state.next_frame(codec).expect("scan");
        assert_eq!(third, Some(Bytes::from_static(b"three")));
        let (_state, none) = state.next_frame(codec).expect("scan");
        assert!(none.is_none());
    }
}
