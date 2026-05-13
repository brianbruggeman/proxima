//! HTTP/2 framing layer (RFC 7540 §4 + §6).
//!
//! ```text
//! +-----------------------------------------------+
//! |                 Length (24)                   |
//! +---------------+---------------+---------------+
//! |   Type (8)    |   Flags (8)   |
//! +-+-------------+---------------+-------------------------------+
//! |R|                 Stream Identifier (31)                      |
//! +=+=============================================================+
//! |                   Frame Payload (0...)                      ...
//! +---------------------------------------------------------------+
//! ```
//!
//! Pure bytes-in / bytes-out: no IO, no state machine. The framer
//! parses one frame at a time from a buffer and serializes frames
//! back to a buffer. Higher layers (stream state machine, connection
//! driver) consume the parsed frames and decide what to do.

use alloc::vec::Vec;

use bytes::{Buf, BufMut, Bytes, BytesMut};
use smallvec::SmallVec;

/// Frame header is fixed at 9 bytes per RFC 7540 §4.1.
pub const FRAME_HEADER_LEN: usize = 9;

/// Default maximum frame payload length per the spec (§4.2). Peers
/// can negotiate larger via SETTINGS_MAX_FRAME_SIZE.
pub const DEFAULT_MAX_FRAME_SIZE: u32 = 16_384;

/// HTTP/2 connection preface bytes (§3.5). Sent by the client first;
/// server reads and verifies before any frames flow.
pub const CONNECTION_PREFACE: &[u8] = b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n";

/// Frame type byte (§11.2). One byte on the wire.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameType {
    Data,
    Headers,
    Priority,
    RstStream,
    Settings,
    PushPromise,
    Ping,
    GoAway,
    WindowUpdate,
    Continuation,
    Unknown(u8),
}

impl FrameType {
    #[inline]
    pub fn from_u8(byte: u8) -> Self {
        match byte {
            0x0 => Self::Data,
            0x1 => Self::Headers,
            0x2 => Self::Priority,
            0x3 => Self::RstStream,
            0x4 => Self::Settings,
            0x5 => Self::PushPromise,
            0x6 => Self::Ping,
            0x7 => Self::GoAway,
            0x8 => Self::WindowUpdate,
            0x9 => Self::Continuation,
            other => Self::Unknown(other),
        }
    }

    #[inline]
    pub fn to_u8(self) -> u8 {
        match self {
            Self::Data => 0x0,
            Self::Headers => 0x1,
            Self::Priority => 0x2,
            Self::RstStream => 0x3,
            Self::Settings => 0x4,
            Self::PushPromise => 0x5,
            Self::Ping => 0x6,
            Self::GoAway => 0x7,
            Self::WindowUpdate => 0x8,
            Self::Continuation => 0x9,
            Self::Unknown(byte) => byte,
        }
    }
}

/// Flag bits — meaning depends on the frame type (§6).
pub mod flags {
    pub const END_STREAM: u8 = 0x1;
    pub const ACK: u8 = 0x1;
    pub const END_HEADERS: u8 = 0x4;
    pub const PADDED: u8 = 0x8;
    pub const PRIORITY: u8 = 0x20;
}

/// Raw frame header without payload interpretation. Consumers either
/// re-parse the payload via the typed [`FramePayload`] enum or handle
/// the raw bytes directly (e.g. forwarding DATA without HPACK touch).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameHeader {
    pub length: u32,
    pub frame_type: FrameType,
    pub flags: u8,
    pub stream_id: u32,
}

impl FrameHeader {
    /// Parse a 9-byte header from the front of `buf`. Returns `None`
    /// when fewer than 9 bytes are available.
    #[inline]
    pub fn parse(buf: &[u8]) -> Option<Self> {
        if buf.len() < FRAME_HEADER_LEN {
            return None;
        }
        // Reads via from_be_bytes compile to a single 4-byte big-endian
        // load + a single bswap on ARM/x86 — better than 4 byte loads +
        // shifts for the length field. The length is 24-bit so we mask.
        let head_word = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]);
        let length = head_word >> 8; // top 24 bits
        let frame_type = FrameType::from_u8((head_word & 0xff) as u8);
        let flags = buf[4];
        let stream_word = u32::from_be_bytes([buf[5], buf[6], buf[7], buf[8]]);
        let stream_id = stream_word & 0x7FFF_FFFF; // top bit reserved (§4.1)
        Some(Self {
            length,
            frame_type,
            flags,
            stream_id,
        })
    }

    /// Serialize into 9 bytes appended to `dst`. One `extend_from_slice`
    /// of a stack array — single capacity check + memcpy of 9 bytes,
    /// vs the 6 separate `put_u8`/`put_u32` calls the prior version did.
    #[inline]
    pub fn encode(&self, dst: &mut Vec<u8>) {
        dst.extend_from_slice(&self.to_bytes());
    }

    /// Pack the header into a 9-byte stack array. Useful for the
    /// vectored encode path — no allocation, no atomic op.
    #[inline]
    #[must_use]
    pub fn to_bytes(&self) -> [u8; FRAME_HEADER_LEN] {
        let stream = self.stream_id & 0x7FFF_FFFF;
        [
            (self.length >> 16) as u8,
            (self.length >> 8) as u8,
            self.length as u8,
            self.frame_type.to_u8(),
            self.flags,
            (stream >> 24) as u8,
            (stream >> 16) as u8,
            (stream >> 8) as u8,
            stream as u8,
        ]
    }

    #[inline]
    pub fn has_flag(&self, flag: u8) -> bool {
        self.flags & flag != 0
    }
}

/// Per-frame-type payload after stripping the header. Variants carry
/// only what's specific to the type — flags + stream_id stay on the
/// [`FrameHeader`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FramePayload {
    /// `DATA` (§6.1). Carries application bytes. May include padding;
    /// `data` excludes the pad-length octet and the padding bytes.
    Data { data: Bytes },
    /// `HEADERS` (§6.2). Carries an HPACK-encoded header block
    /// fragment. May be split across CONTINUATION frames if too
    /// large; the framing layer surfaces the fragment as-is.
    Headers {
        /// 5-byte priority block if the PRIORITY flag was set on the
        /// header. RFC 7540 §6.3 — deprecated in §6.3 of 9113 but
        /// still present on the wire.
        priority: Option<PriorityBlock>,
        block_fragment: Bytes,
    },
    /// `PRIORITY` (§6.3). Stream-priority hint. Always exactly 5 bytes
    /// of payload.
    Priority(PriorityBlock),
    /// `RST_STREAM` (§6.4). Immediate cancellation with an error code.
    RstStream { error_code: u32 },
    /// `SETTINGS` (§6.5). Either a parsed [`StandardSettings`] struct
    /// (each known setting in a typed slot, plus a small extension
    /// vec for unknown ids) or — when the ACK flag is on the header —
    /// a default-constructed (all-None) settings carrying no entries.
    Settings(StandardSettings),
    /// `PUSH_PROMISE` (§6.6). Server push announcement.
    PushPromise {
        promised_stream_id: u32,
        block_fragment: Bytes,
    },
    /// `PING` (§6.7). Always exactly 8 bytes of opaque data.
    Ping { opaque: [u8; 8] },
    /// `GOAWAY` (§6.8). Graceful (or not) connection termination
    /// notice. Carries the last stream id the sender will process.
    GoAway {
        last_stream_id: u32,
        error_code: u32,
        debug_data: Bytes,
    },
    /// `WINDOW_UPDATE` (§6.9). Flow-control credit. `stream_id == 0`
    /// means connection-level; nonzero means stream-level.
    WindowUpdate { increment: u32 },
    /// `CONTINUATION` (§6.10). HPACK fragment continuation after a
    /// HEADERS or PUSH_PROMISE that didn't fit in one frame.
    Continuation { block_fragment: Bytes },
    /// Frame type the framer doesn't recognize. Surfaced so callers
    /// can choose to ignore (§4.1 says implementations MUST ignore
    /// unknown frame types) or forward.
    Unknown { raw_type: u8, payload: Bytes },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PriorityBlock {
    pub exclusive: bool,
    pub stream_dependency: u32,
    pub weight: u8,
}

impl PriorityBlock {
    pub fn parse(buf: &[u8]) -> Result<Self, FrameError> {
        if buf.len() < 5 {
            return Err(FrameError::PriorityTooShort);
        }
        let raw = (u32::from(buf[0]) << 24)
            | (u32::from(buf[1]) << 16)
            | (u32::from(buf[2]) << 8)
            | u32::from(buf[3]);
        Ok(Self {
            exclusive: raw & 0x8000_0000 != 0,
            stream_dependency: raw & 0x7FFF_FFFF,
            weight: buf[4],
        })
    }

    pub fn encode(&self, dst: &mut Vec<u8>) {
        let mut raw = self.stream_dependency & 0x7FFF_FFFF;
        if self.exclusive {
            raw |= 0x8000_0000;
        }
        dst.put_u32(raw);
        dst.put_u8(self.weight);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SettingEntry {
    pub identifier: u16,
    pub value: u32,
}

/// Fixed-shape SETTINGS payload (RFC 7540 §6.5.2). Each known
/// identifier has its own typed `Option<...>` slot — `Some(value)`
/// means the peer sent that setting in this frame, `None` means it
/// wasn't present. Unknown identifiers spill to a small `extensions`
/// vec so encode/decode round-trip; per RFC §6.5.2 receivers MUST
/// ignore unknown settings, but the framer preserves them for the
/// upper layer to handle.
///
/// This shape mirrors the `h2` crate's `Settings` struct. Parse is
/// allocation-free for the common case (≤6 standard ids), beating
/// the previous `SmallVec<SettingEntry>` design that allocated a
/// growing collection per parse.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StandardSettings {
    pub header_table_size: Option<u32>,
    pub enable_push: Option<bool>,
    pub max_concurrent_streams: Option<u32>,
    pub initial_window_size: Option<u32>,
    pub max_frame_size: Option<u32>,
    pub max_header_list_size: Option<u32>,
    /// Non-standard / unrecognized identifiers, preserved for
    /// encode/decode fidelity. Most h2 traffic has zero extensions;
    /// the `SmallVec` keeps that case stack-only.
    pub extensions: SmallVec<[SettingEntry; 4]>,
}

impl StandardSettings {
    /// Apply one (identifier, value) pair, validating per-id rules
    /// when the spec specifies them. ENABLE_PUSH must be 0 or 1
    /// (§6.5.2); other settings accept any u32 in [0, 2^31-1]
    /// (the framer enforces nothing here — that's the connection
    /// driver's job to honor/reject under SETTINGS_FLOW_CONTROL_ERROR).
    #[inline]
    fn apply(&mut self, entry: SettingEntry) -> Result<(), FrameError> {
        match entry.identifier {
            settings_id::HEADER_TABLE_SIZE => {
                self.header_table_size = Some(entry.value);
            }
            settings_id::ENABLE_PUSH => match entry.value {
                0 => self.enable_push = Some(false),
                1 => self.enable_push = Some(true),
                other => {
                    return Err(FrameError::SettingsEnablePushInvalid { value: other });
                }
            },
            settings_id::MAX_CONCURRENT_STREAMS => {
                self.max_concurrent_streams = Some(entry.value);
            }
            settings_id::INITIAL_WINDOW_SIZE => {
                self.initial_window_size = Some(entry.value);
            }
            settings_id::MAX_FRAME_SIZE => {
                self.max_frame_size = Some(entry.value);
            }
            settings_id::MAX_HEADER_LIST_SIZE => {
                self.max_header_list_size = Some(entry.value);
            }
            _ => {
                self.extensions.push(entry);
            }
        }
        Ok(())
    }

    /// Write all set settings + extensions to `dst` as the wire
    /// payload (each (id, value) is 6 bytes). Encode order matches
    /// the standard identifier numbering (1 through 6) followed by
    /// extensions in insertion order.
    #[inline]
    fn encode<B: BufMut>(&self, dst: &mut B) -> usize {
        let mut written = 0_usize;
        if let Some(value) = self.header_table_size {
            dst.put_u16(settings_id::HEADER_TABLE_SIZE);
            dst.put_u32(value);
            written += 6;
        }
        if let Some(enable) = self.enable_push {
            dst.put_u16(settings_id::ENABLE_PUSH);
            dst.put_u32(u32::from(enable));
            written += 6;
        }
        if let Some(value) = self.max_concurrent_streams {
            dst.put_u16(settings_id::MAX_CONCURRENT_STREAMS);
            dst.put_u32(value);
            written += 6;
        }
        if let Some(value) = self.initial_window_size {
            dst.put_u16(settings_id::INITIAL_WINDOW_SIZE);
            dst.put_u32(value);
            written += 6;
        }
        if let Some(value) = self.max_frame_size {
            dst.put_u16(settings_id::MAX_FRAME_SIZE);
            dst.put_u32(value);
            written += 6;
        }
        if let Some(value) = self.max_header_list_size {
            dst.put_u16(settings_id::MAX_HEADER_LIST_SIZE);
            dst.put_u32(value);
            written += 6;
        }
        for entry in &self.extensions {
            dst.put_u16(entry.identifier);
            dst.put_u32(entry.value);
            written += 6;
        }
        written
    }

    /// Number of slots set (counts `Some` fields + extensions).
    /// Used to size encode buffers and answer `is_empty`.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.header_table_size.is_some() as usize
            + self.enable_push.is_some() as usize
            + self.max_concurrent_streams.is_some() as usize
            + self.initial_window_size.is_some() as usize
            + self.max_frame_size.is_some() as usize
            + self.max_header_list_size.is_some() as usize
            + self.extensions.len()
    }

    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Standard SETTINGS identifiers (§6.5.2).
pub mod settings_id {
    pub const HEADER_TABLE_SIZE: u16 = 0x1;
    pub const ENABLE_PUSH: u16 = 0x2;
    pub const MAX_CONCURRENT_STREAMS: u16 = 0x3;
    pub const INITIAL_WINDOW_SIZE: u16 = 0x4;
    pub const MAX_FRAME_SIZE: u16 = 0x5;
    pub const MAX_HEADER_LIST_SIZE: u16 = 0x6;
}

/// Error codes (§7).
pub mod error_code {
    pub const NO_ERROR: u32 = 0x0;
    pub const PROTOCOL_ERROR: u32 = 0x1;
    pub const INTERNAL_ERROR: u32 = 0x2;
    pub const FLOW_CONTROL_ERROR: u32 = 0x3;
    pub const SETTINGS_TIMEOUT: u32 = 0x4;
    pub const STREAM_CLOSED: u32 = 0x5;
    pub const FRAME_SIZE_ERROR: u32 = 0x6;
    pub const REFUSED_STREAM: u32 = 0x7;
    pub const CANCEL: u32 = 0x8;
    pub const COMPRESSION_ERROR: u32 = 0x9;
    pub const CONNECT_ERROR: u32 = 0xa;
    pub const ENHANCE_YOUR_CALM: u32 = 0xb;
    pub const INADEQUATE_SECURITY: u32 = 0xc;
    pub const HTTP_1_1_REQUIRED: u32 = 0xd;
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum FrameError {
    #[error("frame payload length exceeds peer-advertised SETTINGS_MAX_FRAME_SIZE: {got} > {max}")]
    FrameSizeExceeded { got: u32, max: u32 },
    #[error("DATA frame requires stream id != 0")]
    DataOnConnectionStream,
    #[error("HEADERS frame requires stream id != 0")]
    HeadersOnConnectionStream,
    #[error("CONTINUATION frame requires stream id != 0")]
    ContinuationOnConnectionStream,
    #[error("RST_STREAM payload must be exactly 4 bytes; got {got}")]
    RstStreamWrongSize { got: usize },
    #[error("SETTINGS payload must be a multiple of 6 bytes; got {got}")]
    SettingsWrongSize { got: usize },
    #[error("SETTINGS ACK frame must have empty payload; got {got} bytes")]
    SettingsAckNonEmpty { got: usize },
    #[error("SETTINGS_ENABLE_PUSH must be 0 or 1; got {value}")]
    SettingsEnablePushInvalid { value: u32 },
    #[error("PING payload must be exactly 8 bytes; got {got}")]
    PingWrongSize { got: usize },
    #[error("GOAWAY payload must be at least 8 bytes; got {got}")]
    GoAwayWrongSize { got: usize },
    #[error("WINDOW_UPDATE payload must be exactly 4 bytes; got {got}")]
    WindowUpdateWrongSize { got: usize },
    #[error("WINDOW_UPDATE increment must be 1..=2^31-1; got 0")]
    WindowUpdateZero,
    #[error("PRIORITY block must be exactly 5 bytes")]
    PriorityTooShort,
    #[error("PADDED frame: pad length {pad_length} exceeds payload {payload_len}")]
    PaddingOverflow { pad_length: u8, payload_len: usize },
    #[error("frame payload truncated; need {need} more bytes")]
    PayloadTruncated { need: usize },
}

/// Parse a frame payload from `payload`. Zero-copy: all `Bytes`-typed
/// sub-payloads are produced via `payload.slice(..)` (refcount bump,
/// no memcpy). Caller passes exactly `header.length` bytes as a
/// `Bytes` slice; this function never copies the wire data.
///
/// `#[inline]` because the connection driver calls this in a tight
/// loop per frame; LLVM specializes the match dispatch better when it
/// can see the call site's frame_type pattern.
#[inline]
pub fn parse_payload(header: &FrameHeader, payload: &Bytes) -> Result<FramePayload, FrameError> {
    let length = header.length as usize;
    if payload.len() < length {
        return Err(FrameError::PayloadTruncated {
            need: length - payload.len(),
        });
    }
    let payload_bytes = payload.as_ref();
    let payload_view = &payload_bytes[..length];
    match header.frame_type {
        FrameType::Data => {
            if header.stream_id == 0 {
                return Err(FrameError::DataOnConnectionStream);
            }
            let (data_start, data_end) = strip_padding_range(header, payload_view)?;
            Ok(FramePayload::Data {
                data: payload.slice(data_start..data_end),
            })
        }
        FrameType::Headers => {
            if header.stream_id == 0 {
                return Err(FrameError::HeadersOnConnectionStream);
            }
            let (start, end) = strip_padding_range(header, payload_view)?;
            let unpadded = &payload_bytes[start..end];
            let (priority, fragment_offset) = if header.has_flag(flags::PRIORITY) {
                (Some(PriorityBlock::parse(unpadded)?), 5)
            } else {
                (None, 0)
            };
            Ok(FramePayload::Headers {
                priority,
                block_fragment: payload.slice(start + fragment_offset..end),
            })
        }
        FrameType::Priority => Ok(FramePayload::Priority(PriorityBlock::parse(payload_view)?)),
        FrameType::RstStream => {
            if payload_view.len() != 4 {
                return Err(FrameError::RstStreamWrongSize {
                    got: payload_view.len(),
                });
            }
            let mut bytes = payload_view;
            Ok(FramePayload::RstStream {
                error_code: bytes.get_u32(),
            })
        }
        FrameType::Settings => {
            if header.has_flag(flags::ACK) {
                if !payload_view.is_empty() {
                    return Err(FrameError::SettingsAckNonEmpty {
                        got: payload_view.len(),
                    });
                }
                return Ok(FramePayload::Settings(StandardSettings::default()));
            }
            if !payload_view.len().is_multiple_of(6) {
                return Err(FrameError::SettingsWrongSize {
                    got: payload_view.len(),
                });
            }
            // chunks_exact gives the compiler proof of 6-byte slices,
            // so inner from_be_bytes calls compile to inline loads
            // with no bounds checks. apply() routes each (id, value)
            // into its typed slot — zero heap allocation for the
            // common case (≤6 standard ids; extensions land in a
            // 4-inline SmallVec).
            let mut settings = StandardSettings::default();
            for chunk in payload_view.chunks_exact(6) {
                let identifier = u16::from_be_bytes([chunk[0], chunk[1]]);
                let value = u32::from_be_bytes([chunk[2], chunk[3], chunk[4], chunk[5]]);
                settings.apply(SettingEntry { identifier, value })?;
            }
            Ok(FramePayload::Settings(settings))
        }
        FrameType::PushPromise => {
            let (start, end) = strip_padding_range(header, payload_view)?;
            let unpadded = &payload_bytes[start..end];
            if unpadded.len() < 4 {
                return Err(FrameError::PayloadTruncated {
                    need: 4 - unpadded.len(),
                });
            }
            let mut head = &unpadded[..4];
            let promised_stream_id = head.get_u32() & 0x7FFF_FFFF;
            Ok(FramePayload::PushPromise {
                promised_stream_id,
                block_fragment: payload.slice(start + 4..end),
            })
        }
        FrameType::Ping => {
            if payload_view.len() != 8 {
                return Err(FrameError::PingWrongSize {
                    got: payload_view.len(),
                });
            }
            let mut opaque = [0_u8; 8];
            opaque.copy_from_slice(payload_view);
            Ok(FramePayload::Ping { opaque })
        }
        FrameType::GoAway => {
            if payload_view.len() < 8 {
                return Err(FrameError::GoAwayWrongSize {
                    got: payload_view.len(),
                });
            }
            let mut head = &payload_view[..8];
            let last_stream_id = head.get_u32() & 0x7FFF_FFFF;
            let error_code = head.get_u32();
            Ok(FramePayload::GoAway {
                last_stream_id,
                error_code,
                debug_data: payload.slice(8..length),
            })
        }
        FrameType::WindowUpdate => {
            if payload_view.len() != 4 {
                return Err(FrameError::WindowUpdateWrongSize {
                    got: payload_view.len(),
                });
            }
            let mut bytes = payload_view;
            let increment = bytes.get_u32() & 0x7FFF_FFFF;
            if increment == 0 {
                return Err(FrameError::WindowUpdateZero);
            }
            Ok(FramePayload::WindowUpdate { increment })
        }
        FrameType::Continuation => {
            if header.stream_id == 0 {
                return Err(FrameError::ContinuationOnConnectionStream);
            }
            Ok(FramePayload::Continuation {
                block_fragment: payload.slice(0..length),
            })
        }
        FrameType::Unknown(byte) => Ok(FramePayload::Unknown {
            raw_type: byte,
            payload: payload.slice(0..length),
        }),
    }
}

/// Encode a frame (header + payload) into `dst`. Convenience for
/// callers that own typed payloads; the framing layer doesn't impose
/// per-stream flag invariants — those live in the connection driver.
pub fn encode_frame(
    frame_type: FrameType,
    flags: u8,
    stream_id: u32,
    payload: &FramePayload,
    dst: &mut Vec<u8>,
) {
    let payload_start = dst.len() + FRAME_HEADER_LEN;
    let header = FrameHeader {
        length: 0, // patched below once payload is encoded
        frame_type,
        flags,
        stream_id,
    };
    header.encode(dst);
    encode_payload(payload, dst);
    let payload_len = dst.len() - payload_start;
    // patch the 24-bit length at the start of the header
    let length_bytes = payload_len as u32;
    let header_start = payload_start - FRAME_HEADER_LEN;
    dst[header_start] = (length_bytes >> 16) as u8;
    dst[header_start + 1] = (length_bytes >> 8) as u8;
    dst[header_start + 2] = length_bytes as u8;
}

/// Output container for vectored encode. Holds up to 4 segments
/// inline (stack) — the maximum any real-world frame needs:
///
/// - DATA: header + payload = 2 segments
/// - HEADERS w/ priority: header + priority block + fragment = 3 segments
///   (header + priority share one `Bytes` from the scratch buffer, so
///   it's actually 2 segments in practice)
/// - SETTINGS / PING / RST_STREAM / WINDOW_UPDATE / GOAWAY: 1 segment
///
/// SmallVec falls back to heap on overflow but the inline-4 path never
/// allocates on the hot path.
pub type FrameSegments = SmallVec<[Bytes; 4]>;

/// Vectored encode: write the frame as a sequence of refcount-shared
/// segments. No memcpy of the payload bytes regardless of size.
///
/// `scratch` is a reusable `BytesMut` buffer for the small header /
/// fixed-size payload encoding. Across many calls the same buffer
/// allocation is reused (split_to + freeze hand out shared views,
/// growing the buffer only when capacity runs out).
///
/// For frames whose entire payload is small (PING, RST_STREAM,
/// SETTINGS, WINDOW_UPDATE, GOAWAY-with-empty-debug), the result is
/// one segment with everything inline. For DATA/HEADERS/CONTINUATION
/// the payload `Bytes` is borrowed in place — zero copy of the
/// data path.
///
/// Pair this with `tokio::io::AsyncWriteExt::write_vectored` /
/// `IORING_OP_WRITEV` / DPDK mbuf-chain TX to get end-to-end zero-copy
/// on the write side.
#[inline]
pub fn encode_frame_vectored(
    frame_type: FrameType,
    flags: u8,
    stream_id: u32,
    payload: &FramePayload,
    scratch: &mut BytesMut,
) -> FrameSegments {
    let mut segments: FrameSegments = SmallVec::new();
    let scratch_start = scratch.len();

    // Phase 1: encode header placeholder (length patched after payload
    // size is known) + any structured payload bytes into scratch.
    let header_offset = scratch_start;
    scratch.put_u8(0); // length byte 0 — patched
    scratch.put_u8(0); // length byte 1 — patched
    scratch.put_u8(0); // length byte 2 — patched
    scratch.put_u8(frame_type.to_u8());
    scratch.put_u8(flags);
    scratch.put_u32(stream_id & 0x7FFF_FFFF);

    // Phase 2: encode structured payload bytes + capture borrowed segments.
    let mut payload_len: u32 = 0;
    let borrow = encode_payload_vectored(payload, scratch, &mut payload_len);

    // Phase 3: patch the length field in the header.
    let length = payload_len;
    scratch[header_offset] = (length >> 16) as u8;
    scratch[header_offset + 1] = (length >> 8) as u8;
    scratch[header_offset + 2] = length as u8;

    // Phase 4: freeze the scratch range we wrote (header + structured
    // payload, if any) and emit it as the first segment.
    let frozen = scratch.split_to(scratch.len() - scratch_start).freeze();
    segments.push(frozen);

    // Phase 5: append any borrowed Bytes segments after the inline part.
    if let Some(borrowed) = borrow {
        segments.push(borrowed);
    }
    segments
}

/// Encode the payload portion of a frame in the vectored model:
/// structured bytes go into `scratch`; one optional borrowed `Bytes`
/// payload is returned to be appended as a separate segment. Updates
/// `payload_len` with the total wire-level payload size so the caller
/// can patch the frame header's length field.
///
/// Every `Bytes` payload borrows via `.slice(..)` over its own full
/// range rather than `.clone()` — matches the parse side's idiom
/// (`parse_payload` borrows via `payload.slice(start..end)`) so a
/// reader scanning this file sees ONE convention for "share this
/// buffer, don't copy it" rather than two spellings of the same
/// operation. `bytes::Bytes::slice(range)` is defined as `self.clone()`
/// plus a pointer/length adjustment (verified against `bytes` 1.12's
/// source), so for a full-range slice this is a genuinely 0-cost
/// rename, not a performance fix — measured via `stats_alloc`, see
/// `docs/proxima-h2/discipline.md`'s "h2 HPACK borrowing decode"
/// entry for the exact before/after allocation counts.
fn encode_payload_vectored(
    payload: &FramePayload,
    scratch: &mut BytesMut,
    payload_len: &mut u32,
) -> Option<Bytes> {
    match payload {
        FramePayload::Data { data } => {
            *payload_len = data.len() as u32;
            Some(data.slice(..))
        }
        FramePayload::Headers {
            priority,
            block_fragment,
        } => {
            let mut size = 0_u32;
            if let Some(priority) = priority {
                let before = scratch.len();
                priority.encode_bytesmut(scratch);
                size += (scratch.len() - before) as u32;
            }
            size += block_fragment.len() as u32;
            *payload_len = size;
            Some(block_fragment.slice(..))
        }
        FramePayload::Priority(priority) => {
            let before = scratch.len();
            priority.encode_bytesmut(scratch);
            *payload_len = (scratch.len() - before) as u32;
            None
        }
        FramePayload::RstStream { error_code } => {
            scratch.put_u32(*error_code);
            *payload_len = 4;
            None
        }
        FramePayload::Settings(settings) => {
            *payload_len = settings.encode(scratch) as u32;
            None
        }
        FramePayload::PushPromise {
            promised_stream_id,
            block_fragment,
        } => {
            scratch.put_u32(*promised_stream_id & 0x7FFF_FFFF);
            *payload_len = 4 + block_fragment.len() as u32;
            Some(block_fragment.slice(..))
        }
        FramePayload::Ping { opaque } => {
            scratch.extend_from_slice(opaque);
            *payload_len = 8;
            None
        }
        FramePayload::GoAway {
            last_stream_id,
            error_code,
            debug_data,
        } => {
            scratch.put_u32(*last_stream_id & 0x7FFF_FFFF);
            scratch.put_u32(*error_code);
            *payload_len = 8 + debug_data.len() as u32;
            Some(debug_data.slice(..))
        }
        FramePayload::WindowUpdate { increment } => {
            scratch.put_u32(*increment & 0x7FFF_FFFF);
            *payload_len = 4;
            None
        }
        FramePayload::Continuation { block_fragment } => {
            *payload_len = block_fragment.len() as u32;
            Some(block_fragment.slice(..))
        }
        FramePayload::Unknown { payload, .. } => {
            *payload_len = payload.len() as u32;
            Some(payload.slice(..))
        }
    }
}

/// Helper to write segments out to a single Vec for callers that need
/// a contiguous buffer (testing, debug). On the hot path callers
/// should hand segments to `write_vectored` / mbuf chain TX directly.
pub fn flatten_segments(segments: &FrameSegments, dst: &mut Vec<u8>) {
    let total: usize = segments.iter().map(|segment| segment.len()).sum();
    dst.reserve(total);
    for segment in segments {
        dst.extend_from_slice(segment);
    }
}

impl PriorityBlock {
    fn encode_bytesmut(&self, dst: &mut BytesMut) {
        let mut raw = self.stream_dependency & 0x7FFF_FFFF;
        if self.exclusive {
            raw |= 0x8000_0000;
        }
        dst.put_u32(raw);
        dst.put_u8(self.weight);
    }
}

fn encode_payload(payload: &FramePayload, dst: &mut Vec<u8>) {
    match payload {
        FramePayload::Data { data } => dst.extend_from_slice(data),
        FramePayload::Headers {
            priority,
            block_fragment,
        } => {
            if let Some(priority) = priority {
                priority.encode(dst);
            }
            dst.extend_from_slice(block_fragment);
        }
        FramePayload::Priority(priority) => priority.encode(dst),
        FramePayload::RstStream { error_code } => dst.put_u32(*error_code),
        FramePayload::Settings(settings) => {
            settings.encode(dst);
        }
        FramePayload::PushPromise {
            promised_stream_id,
            block_fragment,
        } => {
            dst.put_u32(*promised_stream_id & 0x7FFF_FFFF);
            dst.extend_from_slice(block_fragment);
        }
        FramePayload::Ping { opaque } => dst.extend_from_slice(opaque),
        FramePayload::GoAway {
            last_stream_id,
            error_code,
            debug_data,
        } => {
            dst.put_u32(*last_stream_id & 0x7FFF_FFFF);
            dst.put_u32(*error_code);
            dst.extend_from_slice(debug_data);
        }
        FramePayload::WindowUpdate { increment } => dst.put_u32(*increment & 0x7FFF_FFFF),
        FramePayload::Continuation { block_fragment } => dst.extend_from_slice(block_fragment),
        FramePayload::Unknown { payload, .. } => dst.extend_from_slice(payload),
    }
}

/// Returns `(start, end)` offsets into `payload` such that
/// `payload[start..end]` is the unpadded data. When PADDED is not
/// set, returns `(0, payload.len())`. When set, returns
/// `(1, payload.len() - pad_length)`, skipping the leading pad-length
/// octet and the trailing pad bytes — all in zero-copy index math.
fn strip_padding_range(header: &FrameHeader, payload: &[u8]) -> Result<(usize, usize), FrameError> {
    if !header.has_flag(flags::PADDED) {
        return Ok((0, payload.len()));
    }
    if payload.is_empty() {
        return Err(FrameError::PaddingOverflow {
            pad_length: 0,
            payload_len: 0,
        });
    }
    let pad_length = payload[0];
    let pad_total = pad_length as usize + 1; // +1 for the length octet itself
    if pad_total > payload.len() {
        return Err(FrameError::PaddingOverflow {
            pad_length,
            payload_len: payload.len(),
        });
    }
    Ok((1, payload.len() - pad_length as usize))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn frame_header_round_trips() {
        let original = FrameHeader {
            length: 0x12_3456,
            frame_type: FrameType::Data,
            flags: flags::END_STREAM,
            stream_id: 5,
        };
        let mut buf = Vec::new();
        original.encode(&mut buf);
        assert_eq!(buf.len(), FRAME_HEADER_LEN);
        let parsed = FrameHeader::parse(&buf).expect("parse");
        assert_eq!(parsed, original);
    }

    #[test]
    fn frame_header_strips_reserved_bit_on_stream_id() {
        // stream id with top bit set; spec says receivers MUST ignore.
        let bytes = [
            0x00, 0x00, 0x00, // length 0
            0x00, // type DATA
            0x00, // flags
            0x80, 0x00, 0x00, 0x01, // reserved=1, stream_id=1
        ];
        let parsed = FrameHeader::parse(&bytes).expect("parse");
        assert_eq!(parsed.stream_id, 1, "reserved bit must be ignored");
    }

    #[test]
    fn data_frame_round_trip() {
        let payload = FramePayload::Data {
            data: Bytes::from_static(b"hello"),
        };
        let mut buf = Vec::new();
        encode_frame(FrameType::Data, flags::END_STREAM, 7, &payload, &mut buf);
        let header = FrameHeader::parse(&buf).expect("header");
        assert_eq!(header.length, 5);
        assert_eq!(header.frame_type, FrameType::Data);
        assert_eq!(header.stream_id, 7);
        assert!(header.has_flag(flags::END_STREAM));
        let parsed = parse_payload(&header, &Bytes::copy_from_slice(&buf[FRAME_HEADER_LEN..]))
            .expect("payload");
        assert_eq!(parsed, payload);
    }

    #[test]
    fn data_frame_with_padding_strips_pad_bytes() {
        // PADDED flag set; payload is [pad_len=2, h, i, 0, 0]; total len 5,
        // effective data = [h, i].
        let bytes = [
            0x00,
            0x00,
            0x05,          // length 5
            0x00,          // type DATA
            flags::PADDED, // PADDED flag
            0x00,
            0x00,
            0x00,
            0x01, // stream_id 1
            0x02,
            b'h',
            b'i',
            0x00,
            0x00, // payload
        ];
        let header = FrameHeader::parse(&bytes).expect("header");
        let parsed = parse_payload(&header, &Bytes::copy_from_slice(&bytes[FRAME_HEADER_LEN..]))
            .expect("payload");
        match parsed {
            FramePayload::Data { data } => assert_eq!(&data[..], b"hi"),
            other => panic!("expected Data, got {other:?}"),
        }
    }

    #[test]
    fn data_frame_rejects_stream_zero() {
        let header = FrameHeader {
            length: 1,
            frame_type: FrameType::Data,
            flags: 0,
            stream_id: 0,
        };
        let outcome = parse_payload(&header, &Bytes::from_static(&[0xAA]));
        assert!(matches!(outcome, Err(FrameError::DataOnConnectionStream)));
    }

    #[test]
    fn headers_frame_without_priority_round_trips() {
        let payload = FramePayload::Headers {
            priority: None,
            block_fragment: Bytes::from_static(b"\x82\x86\x84"),
        };
        let mut buf = Vec::new();
        encode_frame(
            FrameType::Headers,
            flags::END_HEADERS | flags::END_STREAM,
            13,
            &payload,
            &mut buf,
        );
        let header = FrameHeader::parse(&buf).expect("header");
        let parsed = parse_payload(&header, &Bytes::copy_from_slice(&buf[FRAME_HEADER_LEN..]))
            .expect("payload");
        assert_eq!(parsed, payload);
    }

    #[test]
    fn headers_frame_with_priority_round_trips() {
        let priority = PriorityBlock {
            exclusive: true,
            stream_dependency: 3,
            weight: 200,
        };
        let payload = FramePayload::Headers {
            priority: Some(priority),
            block_fragment: Bytes::from_static(b"\x82"),
        };
        let mut buf = Vec::new();
        encode_frame(
            FrameType::Headers,
            flags::END_HEADERS | flags::PRIORITY,
            5,
            &payload,
            &mut buf,
        );
        let header = FrameHeader::parse(&buf).expect("header");
        let parsed = parse_payload(&header, &Bytes::copy_from_slice(&buf[FRAME_HEADER_LEN..]))
            .expect("payload");
        assert_eq!(parsed, payload);
    }

    #[test]
    fn settings_frame_with_entries_round_trips() {
        let settings = StandardSettings {
            initial_window_size: Some(65535),
            max_concurrent_streams: Some(100),
            ..Default::default()
        };
        let payload = FramePayload::Settings(settings.clone());
        let mut buf = Vec::new();
        encode_frame(FrameType::Settings, 0, 0, &payload, &mut buf);
        let header = FrameHeader::parse(&buf).expect("header");
        assert_eq!(header.length, (settings.len() * 6) as u32);
        let parsed = parse_payload(&header, &Bytes::copy_from_slice(&buf[FRAME_HEADER_LEN..]))
            .expect("payload");
        assert_eq!(parsed, payload);
    }

    #[test]
    fn settings_ack_must_be_empty() {
        let header = FrameHeader {
            length: 6,
            frame_type: FrameType::Settings,
            flags: flags::ACK,
            stream_id: 0,
        };
        let outcome = parse_payload(&header, &Bytes::from_static(&[0_u8; 6]));
        assert!(matches!(
            outcome,
            Err(FrameError::SettingsAckNonEmpty { got: 6 })
        ));
    }

    #[test]
    fn settings_payload_must_be_multiple_of_six() {
        let header = FrameHeader {
            length: 5,
            frame_type: FrameType::Settings,
            flags: 0,
            stream_id: 0,
        };
        let outcome = parse_payload(&header, &Bytes::from_static(&[0_u8; 5]));
        assert!(matches!(
            outcome,
            Err(FrameError::SettingsWrongSize { got: 5 })
        ));
    }

    #[test]
    fn ping_round_trip() {
        let payload = FramePayload::Ping {
            opaque: *b"opaque!!",
        };
        let mut buf = Vec::new();
        encode_frame(FrameType::Ping, 0, 0, &payload, &mut buf);
        let header = FrameHeader::parse(&buf).expect("header");
        let parsed = parse_payload(&header, &Bytes::copy_from_slice(&buf[FRAME_HEADER_LEN..]))
            .expect("payload");
        assert_eq!(parsed, payload);
    }

    #[test]
    fn ping_wrong_size_errors() {
        let header = FrameHeader {
            length: 7,
            frame_type: FrameType::Ping,
            flags: 0,
            stream_id: 0,
        };
        let outcome = parse_payload(&header, &Bytes::from_static(&[0_u8; 7]));
        assert!(matches!(outcome, Err(FrameError::PingWrongSize { got: 7 })));
    }

    #[test]
    fn rst_stream_round_trip() {
        let payload = FramePayload::RstStream {
            error_code: error_code::CANCEL,
        };
        let mut buf = Vec::new();
        encode_frame(FrameType::RstStream, 0, 11, &payload, &mut buf);
        let header = FrameHeader::parse(&buf).expect("header");
        let parsed = parse_payload(&header, &Bytes::copy_from_slice(&buf[FRAME_HEADER_LEN..]))
            .expect("payload");
        assert_eq!(parsed, payload);
    }

    #[test]
    fn goaway_round_trip_with_debug_data() {
        let payload = FramePayload::GoAway {
            last_stream_id: 13,
            error_code: error_code::INTERNAL_ERROR,
            debug_data: Bytes::from_static(b"crash"),
        };
        let mut buf = Vec::new();
        encode_frame(FrameType::GoAway, 0, 0, &payload, &mut buf);
        let header = FrameHeader::parse(&buf).expect("header");
        let parsed = parse_payload(&header, &Bytes::copy_from_slice(&buf[FRAME_HEADER_LEN..]))
            .expect("payload");
        assert_eq!(parsed, payload);
    }

    #[test]
    fn window_update_zero_increment_is_protocol_error() {
        let header = FrameHeader {
            length: 4,
            frame_type: FrameType::WindowUpdate,
            flags: 0,
            stream_id: 0,
        };
        let outcome = parse_payload(&header, &Bytes::from_static(&[0_u8; 4]));
        assert!(matches!(outcome, Err(FrameError::WindowUpdateZero)));
    }

    #[test]
    fn window_update_round_trip() {
        let payload = FramePayload::WindowUpdate { increment: 12345 };
        let mut buf = Vec::new();
        encode_frame(FrameType::WindowUpdate, 0, 7, &payload, &mut buf);
        let header = FrameHeader::parse(&buf).expect("header");
        let parsed = parse_payload(&header, &Bytes::copy_from_slice(&buf[FRAME_HEADER_LEN..]))
            .expect("payload");
        assert_eq!(parsed, payload);
    }

    #[test]
    fn unknown_frame_type_round_trips_via_unknown_variant() {
        // Use type 0xFF — not assigned by the spec.
        let bytes = [
            0x00, 0x00, 0x03, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, // header
            0xAA, 0xBB, 0xCC, // payload
        ];
        let header = FrameHeader::parse(&bytes).expect("header");
        let parsed = parse_payload(&header, &Bytes::copy_from_slice(&bytes[FRAME_HEADER_LEN..]))
            .expect("payload");
        match parsed {
            FramePayload::Unknown { raw_type, payload } => {
                assert_eq!(raw_type, 0xFF);
                assert_eq!(&payload[..], &[0xAA, 0xBB, 0xCC]);
            }
            other => panic!("expected Unknown, got {other:?}"),
        }
    }

    #[test]
    fn payload_truncated_when_buf_shorter_than_length() {
        let header = FrameHeader {
            length: 10,
            frame_type: FrameType::Data,
            flags: 0,
            stream_id: 1,
        };
        let outcome = parse_payload(&header, &Bytes::from_static(&[0_u8; 5]));
        assert!(matches!(
            outcome,
            Err(FrameError::PayloadTruncated { need: 5 })
        ));
    }

    #[test]
    fn data_frame_payload_is_zero_copy_view_into_source() {
        // Build a DATA frame inside a single Bytes allocation. Parse it.
        // Assert the returned `Data { data }` shares its backing pointer
        // with the source — i.e., it's a refcount slice, not a memcpy.
        let mut wire = Vec::new();
        let payload = FramePayload::Data {
            data: Bytes::from_static(b"hello-zero-copy"),
        };
        encode_frame(FrameType::Data, 0, 1, &payload, &mut wire);
        let wire_bytes = Bytes::from(wire);
        let header = FrameHeader::parse(&wire_bytes).expect("header");
        let payload_view = wire_bytes.slice(FRAME_HEADER_LEN..);
        let parsed = parse_payload(&header, &payload_view).expect("payload");
        let FramePayload::Data { data } = parsed else {
            panic!("expected Data variant");
        };
        // Both `data` and `payload_view` should slice into the same
        // arc-backed allocation as `wire_bytes`. Asserting equality of
        // base pointers proves no copy happened.
        let wire_base = wire_bytes.as_ptr();
        let data_base = data.as_ptr();
        let offset = unsafe { data_base.offset_from(wire_base) };
        assert!(
            offset >= 0 && (offset as usize) >= FRAME_HEADER_LEN,
            "Data payload must be a slice inside the wire Bytes — offset {offset}"
        );
    }

    #[test]
    fn vectored_encode_matches_contiguous_encode_for_data() {
        let data = Bytes::from(vec![b'A'; 4096]);
        let payload = FramePayload::Data { data: data.clone() };

        let mut contiguous = Vec::new();
        encode_frame(
            FrameType::Data,
            flags::END_STREAM,
            7,
            &payload,
            &mut contiguous,
        );

        let mut scratch = BytesMut::with_capacity(64);
        let segments = encode_frame_vectored(
            FrameType::Data,
            flags::END_STREAM,
            7,
            &payload,
            &mut scratch,
        );
        let mut flat = Vec::new();
        flatten_segments(&segments, &mut flat);

        assert_eq!(
            flat, contiguous,
            "vectored encode must produce identical wire bytes"
        );
        assert_eq!(
            segments.len(),
            2,
            "DATA should emit 2 segments: header + borrowed payload"
        );
    }

    #[test]
    fn vectored_encode_borrows_data_payload_zero_copy() {
        // Build a unique source Bytes and prove the segment shares its
        // backing pointer (no memcpy).
        let data = Bytes::from(vec![b'Z'; 16 * 1024]);
        let source_ptr = data.as_ptr();
        let payload = FramePayload::Data { data: data.clone() };

        let mut scratch = BytesMut::with_capacity(64);
        let segments = encode_frame_vectored(FrameType::Data, 0, 1, &payload, &mut scratch);

        assert_eq!(segments.len(), 2);
        let payload_segment = &segments[1];
        assert_eq!(
            payload_segment.as_ptr(),
            source_ptr,
            "payload must be borrowed, not copied"
        );
    }

    #[test]
    fn vectored_encode_inlines_small_frames_to_one_segment() {
        let payload = FramePayload::Ping {
            opaque: *b"opaque!!",
        };
        let mut scratch = BytesMut::with_capacity(64);
        let segments = encode_frame_vectored(FrameType::Ping, 0, 0, &payload, &mut scratch);
        assert_eq!(
            segments.len(),
            1,
            "PING fits entirely in the inline segment"
        );
    }

    #[test]
    fn vectored_encode_scratch_buffer_reuses_allocation_across_calls() {
        // Across many encode calls the scratch BytesMut should grow once
        // then stay at peak capacity — no realloc on the hot path.
        let mut scratch = BytesMut::with_capacity(64);
        let payload = FramePayload::Data {
            data: Bytes::from(vec![b'A'; 1024]),
        };
        // Prime
        let _ = encode_frame_vectored(FrameType::Data, 0, 1, &payload, &mut scratch);
        let cap_after_first = scratch.capacity();
        for _ in 0..1000 {
            // Each call freezes the prior segment, leaving scratch empty
            // but at the same capacity.
            let _ = encode_frame_vectored(FrameType::Data, 0, 1, &payload, &mut scratch);
        }
        // capacity may grow once to amortize but should not grow per call.
        assert!(
            scratch.capacity() <= cap_after_first * 2,
            "scratch capacity should be amortized; got {} after 1000 frames",
            scratch.capacity()
        );
    }

    #[test]
    fn vectored_encode_round_trips_all_typed_frames() {
        let cases: Vec<(FrameType, u8, u32, FramePayload)> = vec![
            (
                FrameType::Data,
                flags::END_STREAM,
                3,
                FramePayload::Data {
                    data: Bytes::from_static(b"hello"),
                },
            ),
            (
                FrameType::Headers,
                flags::END_HEADERS,
                5,
                FramePayload::Headers {
                    priority: None,
                    block_fragment: Bytes::from_static(b"\x82\x86\x84"),
                },
            ),
            (
                FrameType::Headers,
                flags::END_HEADERS | flags::PRIORITY,
                7,
                FramePayload::Headers {
                    priority: Some(PriorityBlock {
                        exclusive: false,
                        stream_dependency: 1,
                        weight: 100,
                    }),
                    block_fragment: Bytes::from_static(b"\x82"),
                },
            ),
            (
                FrameType::Settings,
                0,
                0,
                FramePayload::Settings(StandardSettings {
                    initial_window_size: Some(65535),
                    ..Default::default()
                }),
            ),
            (
                FrameType::Ping,
                0,
                0,
                FramePayload::Ping {
                    opaque: *b"abcdefgh",
                },
            ),
            (
                FrameType::RstStream,
                0,
                9,
                FramePayload::RstStream {
                    error_code: error_code::CANCEL,
                },
            ),
            (
                FrameType::WindowUpdate,
                0,
                3,
                FramePayload::WindowUpdate { increment: 1234 },
            ),
            (
                FrameType::GoAway,
                0,
                0,
                FramePayload::GoAway {
                    last_stream_id: 11,
                    error_code: error_code::NO_ERROR,
                    debug_data: Bytes::from_static(b"bye"),
                },
            ),
        ];
        let mut scratch = BytesMut::with_capacity(64);
        for (frame_type, flags, stream_id, payload) in cases {
            scratch.clear();
            let segments =
                encode_frame_vectored(frame_type, flags, stream_id, &payload, &mut scratch);
            let mut flat = Vec::new();
            flatten_segments(&segments, &mut flat);
            let mut expected = Vec::new();
            encode_frame(frame_type, flags, stream_id, &payload, &mut expected);
            assert_eq!(
                flat, expected,
                "vectored must match contiguous for {frame_type:?}"
            );
        }
    }

    #[test]
    fn padding_overflow_errors() {
        // PADDED frame claims pad_length=5 but payload is only 3 bytes.
        let bytes = [
            0x00,
            0x00,
            0x03,          // length 3
            0x00,          // type DATA
            flags::PADDED, // PADDED
            0x00,
            0x00,
            0x00,
            0x01, // stream_id 1
            0x05,
            0xAA,
            0xBB, // payload [pad_len=5, then 2 data bytes — overflow]
        ];
        let header = FrameHeader::parse(&bytes).expect("header");
        let outcome = parse_payload(&header, &Bytes::copy_from_slice(&bytes[FRAME_HEADER_LEN..]));
        assert!(matches!(outcome, Err(FrameError::PaddingOverflow { .. })));
    }
}
