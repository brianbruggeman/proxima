#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::vec::Vec;
use core::net::SocketAddr;

use bytes::Bytes;
use proxima_core::ProximaError;

#[cfg(feature = "std")]
use std::cell::RefCell;

// per-thread scratch buffer reused across simd-json decodes. simd-json
// mutates its input in place, so the codec must own a mutable copy. a
// thread_local Vec amortizes the allocation across requests on the
// same worker without needing a synchronized BufferPool dance.
#[cfg(feature = "std")]
thread_local! {
    static DECODE_SCRATCH: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

#[cfg(feature = "std")]
fn decode_through_scratch<T>(bytes: &[u8]) -> Result<T, ProximaError>
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
pub struct JsonCodec<Input, Output>(std::marker::PhantomData<(Input, Output)>);

#[cfg(feature = "std")]
impl<Input, Output> Default for JsonCodec<Input, Output> {
    fn default() -> Self {
        Self(std::marker::PhantomData)
    }
}

#[cfg(feature = "std")]
impl<Input, Output> JsonCodec<Input, Output> {
    #[must_use]
    pub fn new() -> Self {
        Self(std::marker::PhantomData)
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameError {
    /// The buffer does not yet hold a complete frame — read more bytes
    /// and retry. The normal partial-read signal, not a failure.
    Incomplete,
    /// A zero-length frame was declared while `reject_zero_len` is set.
    ZeroLength,
    /// The declared payload length exceeds `max_frame_bytes`.
    FrameTooLarge { len: usize },
}

impl core::fmt::Display for FrameError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Incomplete => write!(formatter, "incomplete frame"),
            Self::ZeroLength => write!(formatter, "zero-length frame rejected"),
            Self::FrameTooLarge { len } => {
                write!(
                    formatter,
                    "declared frame size {len} exceeds max_frame_bytes"
                )
            }
        }
    }
}

impl core::error::Error for FrameError {}

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

/// Fixed-width 4-byte [`Datagram`] used only by this module's own unit
/// tests — a trivial POD message, borrowed zero-copy from the packet
/// buffer, to exercise the trait shape without pulling in a real
/// protocol's parsing logic.
#[cfg(test)]
#[derive(Debug, Clone, Copy, Default)]
struct FixedFourCodec;

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FixedFourError {
    got_len: usize,
}

#[cfg(test)]
impl core::fmt::Display for FixedFourError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(formatter, "expected exactly 4 bytes, got {}", self.got_len)
    }
}

#[cfg(test)]
impl core::error::Error for FixedFourError {}

#[cfg(test)]
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

/// Owned-message [`Datagram`] used only by this module's own unit
/// tests — demonstrates the `Message<'a> = Owned` escape hatch a
/// future owned-`Message` protocol (a bencode-over-UDP or
/// fixed-binary datagram format that decodes straight into an owned
/// value, never borrows) would use: `Message<'a>` ignores `'a`
/// entirely.
#[cfg(test)]
#[derive(Debug, Clone, Copy, Default)]
struct OwnedIdCodec;

#[cfg(test)]
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

    fn encode(&self, addressed: &Addressed<u32>, dest: &mut Vec<u8>) -> Result<(), FixedFourError> {
        dest.extend_from_slice(&addressed.message.to_be_bytes());
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Deserialize, Serialize, PartialEq, Eq)]
    struct Sample {
        name: String,
        count: u32,
    }

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

    #[test]
    fn json_codec_decode_error_returns_decode_variant() {
        let codec: JsonCodec<Sample, Sample> = JsonCodec::new();
        let outcome = codec.decode_input(b"not json");
        assert!(matches!(outcome, Err(ProximaError::Decode(_))));
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
    fn json_content_type_is_application_json() {
        let codec: JsonCodec<Sample, Sample> = JsonCodec::new();
        assert_eq!(codec.content_type(), "application/json");
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
    fn frame_error_zero_length_message_is_parity_stable() {
        // the downstream consumer's incumbent wire error is exactly this string; the listener
        // maps FrameError -> io::Error with this Display.
        assert_eq!(
            FrameError::ZeroLength.to_string(),
            "zero-length frame rejected"
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
}

#[cfg(feature = "std")]
pub mod factory;
#[cfg(feature = "std")]
pub use factory::{
    BytesPassthroughCodecFactory, BytesPassthroughDynCodec, CodecBuildFuture, CodecFactory,
    CodecRegistry, DynCodec, DynCodecFactory, DynCodecHandle, JsonCodecFactory, JsonDynCodec,
};

pub mod share_buf;
pub use share_buf::ShareBuf;
