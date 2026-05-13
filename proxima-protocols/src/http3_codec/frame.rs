//! HTTP/3 frame codec per [RFC 9114 §7].
//!
//! Every H3 frame is `<type varint> <length varint> <payload bytes>`.
//! The codec parses one frame at a time from a borrowed byte slice +
//! encodes one frame at a time into a caller-supplied `&mut [u8]`.
//! Output views borrow from the input — zero alloc on the hot path.
//!
//! # Frame types covered (RFC 9114 §7.2)
//!
//! - `DATA` (0x00) §7.2.1
//! - `HEADERS` (0x01) §7.2.2 — payload is QPACK-encoded header block
//! - `CANCEL_PUSH` (0x03) §7.2.3
//! - `SETTINGS` (0x04) §7.2.4 — payload is a series of (id, value) varint pairs
//! - `PUSH_PROMISE` (0x05) §7.2.5
//! - `GOAWAY` (0x07) §7.2.6
//! - `MAX_PUSH_ID` (0x0d) §7.2.7
//!
//! Reserved types 0x02 / 0x06 / 0x08 / 0x09 / 0x0a / 0x0b / 0x0c are
//! treated as ignorable per §7.2.8: parser returns `H3Frame::Reserved`
//! carrying the raw type byte + length + payload slice, so the caller
//! can choose to skip them.
//!
//! [RFC 9114 §7]: https://www.rfc-editor.org/rfc/rfc9114#section-7
//!
//! # Tier
//!
//! Tier-3 (bare `no_std + no_alloc`). All parse paths borrow into
//! `&'a [u8]`; encode paths write into caller-owned `&mut [u8]`.

use crate::quic::varint;

/// `DATA` frame — RFC 9114 §7.2.1.
pub const FRAME_TYPE_DATA: u64 = 0x00;
/// `HEADERS` frame — RFC 9114 §7.2.2.
pub const FRAME_TYPE_HEADERS: u64 = 0x01;
/// `CANCEL_PUSH` — RFC 9114 §7.2.3.
pub const FRAME_TYPE_CANCEL_PUSH: u64 = 0x03;
/// `SETTINGS` — RFC 9114 §7.2.4.
pub const FRAME_TYPE_SETTINGS: u64 = 0x04;
/// `PUSH_PROMISE` — RFC 9114 §7.2.5.
pub const FRAME_TYPE_PUSH_PROMISE: u64 = 0x05;
/// `GOAWAY` — RFC 9114 §7.2.6.
pub const FRAME_TYPE_GOAWAY: u64 = 0x07;
/// `MAX_PUSH_ID` — RFC 9114 §7.2.7.
pub const FRAME_TYPE_MAX_PUSH_ID: u64 = 0x0d;

/// HTTP/2 frame types that are explicitly **reserved** in HTTP/3 per
/// RFC 9114 §11.2.1 — the receiver MUST treat any of these as a
/// connection error of type `H3_FRAME_UNEXPECTED`. They cannot be
/// silently ignored as GREASE; the RFC carves them out from the
/// §7.2.8 "ignore reserved types" rule explicitly.
///
/// - `0x02` HTTP/2 PRIORITY
/// - `0x06` HTTP/2 PING
/// - `0x08` HTTP/2 WINDOW_UPDATE
/// - `0x09` HTTP/2 CONTINUATION
pub const HTTP2_RESERVED_FRAME_TYPES: &[u64] = &[0x02, 0x06, 0x08, 0x09];

/// `true` if `frame_type` is one of the four HTTP/2 frame types that
/// HTTP/3 reserves per RFC 9114 §11.2.1 — see
/// [`HTTP2_RESERVED_FRAME_TYPES`]. Callers map these to a connection
/// error regardless of which stream class they appear on.
#[must_use]
pub fn is_http2_reserved(frame_type: u64) -> bool {
    matches!(frame_type, 0x02 | 0x06 | 0x08 | 0x09)
}

/// `true` if `frame_type` matches the RFC 9114 §7.2.8 GREASE pattern
/// `0x1f * N + 0x21` — the receiver MUST ignore these (and any other
/// truly-unknown type, but the GREASE pattern is the canonical
/// "ignored on purpose" signal). Excludes the four HTTP/2-reserved
/// types per [`is_http2_reserved`], which §11.2.1 carves out.
#[must_use]
pub fn is_grease(frame_type: u64) -> bool {
    // 0x1f * N + 0x21 — N = 0 yields 0x21; further values cover
    // 0x40, 0x5f, 0x7e, … up to the varint cap. The four HTTP/2
    // reserved types are NOT in this set by construction (they're
    // all < 0x21), but the explicit guard makes the precedence
    // obvious to readers.
    !is_http2_reserved(frame_type) && frame_type >= 0x21 && (frame_type - 0x21).is_multiple_of(0x1f)
}

/// Frame-codec failure modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum FrameError {
    /// Input ended before the type/length varint or the payload body
    /// could be read in full.
    Truncated,
    /// A varint inside the frame failed to decode (out-of-range or
    /// malformed encoding per RFC 9000 §16).
    InvalidVarint,
    /// Frame payload length exceeded the H3-permitted maximum or the
    /// surrounding buffer. The bound on the actual maximum is set by
    /// the connection's negotiated `SETTINGS_MAX_FIELD_SECTION_SIZE`
    /// and similar caps — enforced one layer up; the codec rejects
    /// only lengths it could not even read.
    PayloadTooLong,
    /// Caller-supplied output buffer is too small for the encoded
    /// frame. `needed` is the byte count required to encode this
    /// frame's full type + length varints + payload.
    BufferTooSmall { needed: usize },
}

impl From<varint::DecodeError> for FrameError {
    fn from(err: varint::DecodeError) -> Self {
        // DecodeError is #[non_exhaustive] for external crates, but varint
        // now lives in this same crate (folded from proxima-quic-proto) —
        // within the defining crate the attribute doesn't force a wildcard
        // arm, so the match is exhaustive over the two known variants.
        match err {
            varint::DecodeError::Empty | varint::DecodeError::Truncated => Self::Truncated,
        }
    }
}

impl From<varint::EncodeError> for FrameError {
    fn from(_: varint::EncodeError) -> Self {
        Self::BufferTooSmall { needed: 0 }
    }
}

/// HTTP/3 wire-format frame. Borrowed views into the caller's input
/// buffer for zero-copy parse + zero-copy re-emission.
///
/// Reserved-type frames (per RFC 9114 §7.2.8) are preserved as
/// [`H3Frame::Reserved`] so the caller can either skip them or surface
/// them as an `H3_FRAME_UNEXPECTED` error depending on stream context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum H3Frame<'a> {
    /// §7.2.1 — DATA frame, request/response body bytes. Allowed only
    /// on request streams + push streams.
    Data { payload: &'a [u8] },
    /// §7.2.2 — HEADERS frame. Payload is the QPACK-encoded header
    /// block; decode via `qpack::Decoder`.
    Headers { header_block: &'a [u8] },
    /// §7.2.3 — CANCEL_PUSH on the control stream.
    CancelPush { push_id: u64 },
    /// §7.2.4 — SETTINGS frame, list of (id, value) varint pairs. The
    /// raw payload bytes are exposed so the connection-level settings
    /// applier can iterate via [`SettingsIter`].
    Settings { payload: &'a [u8] },
    /// §7.2.5 — PUSH_PROMISE on a request stream.
    PushPromise {
        push_id: u64,
        header_block: &'a [u8],
    },
    /// §7.2.6 — GOAWAY on the control stream. Carries the largest
    /// stream/push ID the sender will process.
    GoAway { id: u64 },
    /// §7.2.7 — MAX_PUSH_ID on the control stream.
    MaxPushId { push_id: u64 },
    /// §7.2.8 — reserved type. Preserved verbatim; the caller decides
    /// whether to ignore (control stream) or error (request stream).
    Reserved { frame_type: u64, payload: &'a [u8] },
}

/// Parse one HTTP/3 frame from `input`. Returns the parsed frame +
/// the total bytes consumed (so the caller can slice past the frame
/// for the next parse call).
///
/// # Errors
///
/// See [`FrameError`].
pub fn parse(input: &[u8]) -> Result<(H3Frame<'_>, usize), FrameError> {
    let (frame_type, type_bytes) = varint::decode(input)?;
    let length_bytes_input = &input[type_bytes..];
    let (length, length_bytes) = varint::decode(length_bytes_input)?;
    let length_usize = usize::try_from(length).map_err(|_| FrameError::PayloadTooLong)?;
    let payload_start = type_bytes + length_bytes;
    let payload_end = payload_start
        .checked_add(length_usize)
        .ok_or(FrameError::PayloadTooLong)?;
    if input.len() < payload_end {
        return Err(FrameError::Truncated);
    }
    let payload = &input[payload_start..payload_end];

    let frame = match frame_type {
        FRAME_TYPE_DATA => H3Frame::Data { payload },
        FRAME_TYPE_HEADERS => H3Frame::Headers {
            header_block: payload,
        },
        FRAME_TYPE_CANCEL_PUSH => {
            let (push_id, consumed) = varint::decode(payload)?;
            if consumed != payload.len() {
                return Err(FrameError::InvalidVarint);
            }
            H3Frame::CancelPush { push_id }
        }
        FRAME_TYPE_SETTINGS => H3Frame::Settings { payload },
        FRAME_TYPE_PUSH_PROMISE => {
            let (push_id, consumed) = varint::decode(payload)?;
            let header_block = &payload[consumed..];
            H3Frame::PushPromise {
                push_id,
                header_block,
            }
        }
        FRAME_TYPE_GOAWAY => {
            let (id, consumed) = varint::decode(payload)?;
            if consumed != payload.len() {
                return Err(FrameError::InvalidVarint);
            }
            H3Frame::GoAway { id }
        }
        FRAME_TYPE_MAX_PUSH_ID => {
            let (push_id, consumed) = varint::decode(payload)?;
            if consumed != payload.len() {
                return Err(FrameError::InvalidVarint);
            }
            H3Frame::MaxPushId { push_id }
        }
        other => H3Frame::Reserved {
            frame_type: other,
            payload,
        },
    };
    Ok((frame, payload_end))
}

/// Encode one HTTP/3 frame into `output`. Returns the number of bytes
/// written.
///
/// # Errors
///
/// Returns [`FrameError::BufferTooSmall`] when `output.len()` is less
/// than the encoded frame size. The `needed` field carries that exact
/// minimum.
pub fn encode(frame: &H3Frame<'_>, output: &mut [u8]) -> Result<usize, FrameError> {
    match *frame {
        H3Frame::Data { payload } => write_bytes_frame(FRAME_TYPE_DATA, payload, output),
        H3Frame::Headers { header_block } => {
            write_bytes_frame(FRAME_TYPE_HEADERS, header_block, output)
        }
        H3Frame::CancelPush { push_id } => {
            write_varint_frame(FRAME_TYPE_CANCEL_PUSH, push_id, output)
        }
        H3Frame::Settings { payload } => write_bytes_frame(FRAME_TYPE_SETTINGS, payload, output),
        H3Frame::PushPromise {
            push_id,
            header_block,
        } => {
            let push_id_len = varint::encoded_len(push_id);
            let payload_len = push_id_len + header_block.len();
            let frame_overhead = varint::encoded_len(FRAME_TYPE_PUSH_PROMISE)
                + varint::encoded_len(payload_len as u64);
            let needed = frame_overhead + payload_len;
            if output.len() < needed {
                return Err(FrameError::BufferTooSmall { needed });
            }
            let mut cursor = varint::encode(FRAME_TYPE_PUSH_PROMISE, output)?;
            cursor += varint::encode(payload_len as u64, &mut output[cursor..])?;
            cursor += varint::encode(push_id, &mut output[cursor..])?;
            output[cursor..cursor + header_block.len()].copy_from_slice(header_block);
            cursor += header_block.len();
            Ok(cursor)
        }
        H3Frame::GoAway { id } => write_varint_frame(FRAME_TYPE_GOAWAY, id, output),
        H3Frame::MaxPushId { push_id } => {
            write_varint_frame(FRAME_TYPE_MAX_PUSH_ID, push_id, output)
        }
        H3Frame::Reserved {
            frame_type,
            payload,
        } => write_bytes_frame(frame_type, payload, output),
    }
}

fn write_bytes_frame(
    frame_type: u64,
    payload: &[u8],
    output: &mut [u8],
) -> Result<usize, FrameError> {
    let type_len = varint::encoded_len(frame_type);
    let length_len = varint::encoded_len(payload.len() as u64);
    let needed = type_len + length_len + payload.len();
    if output.len() < needed {
        return Err(FrameError::BufferTooSmall { needed });
    }
    let mut cursor = varint::encode(frame_type, output)?;
    cursor += varint::encode(payload.len() as u64, &mut output[cursor..])?;
    output[cursor..cursor + payload.len()].copy_from_slice(payload);
    Ok(cursor + payload.len())
}

fn write_varint_frame(frame_type: u64, value: u64, output: &mut [u8]) -> Result<usize, FrameError> {
    let type_len = varint::encoded_len(frame_type);
    let value_len = varint::encoded_len(value);
    let needed = type_len + varint::encoded_len(value_len as u64) + value_len;
    if output.len() < needed {
        return Err(FrameError::BufferTooSmall { needed });
    }
    let mut cursor = varint::encode(frame_type, output)?;
    cursor += varint::encode(value_len as u64, &mut output[cursor..])?;
    cursor += varint::encode(value, &mut output[cursor..])?;
    Ok(cursor)
}

/// Streaming iterator over the (identifier, value) pairs of a SETTINGS
/// frame's payload per RFC 9114 §7.2.4. Borrowed-only, zero-copy.
///
/// Caller passes the [`H3Frame::Settings`] payload slice (the bytes
/// after the SETTINGS frame's length varint) — the iterator yields
/// `Result<(id, value), FrameError>` per pair.
///
/// # Example
///
/// ```ignore
/// for pair in SettingsIter::new(payload) {
///     let (id, value) = pair?;
///     // apply per RFC 9114 §7.2.4.1
/// }
/// ```
#[derive(Debug, Clone, Copy)]
pub struct SettingsIter<'a> {
    cursor: usize,
    payload: &'a [u8],
}

impl<'a> SettingsIter<'a> {
    /// Construct over a SETTINGS-frame payload.
    #[must_use]
    pub const fn new(payload: &'a [u8]) -> Self {
        Self { cursor: 0, payload }
    }
}

impl Iterator for SettingsIter<'_> {
    type Item = Result<(u64, u64), FrameError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor >= self.payload.len() {
            return None;
        }
        let (id, id_len) = match varint::decode(&self.payload[self.cursor..]) {
            Ok(decoded) => decoded,
            Err(err) => return Some(Err(err.into())),
        };
        self.cursor += id_len;
        if self.cursor >= self.payload.len() {
            return Some(Err(FrameError::Truncated));
        }
        let (value, value_len) = match varint::decode(&self.payload[self.cursor..]) {
            Ok(decoded) => decoded,
            Err(err) => return Some(Err(err.into())),
        };
        self.cursor += value_len;
        Some(Ok((id, value)))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    extern crate alloc;
    use alloc::vec;
    use alloc::vec::Vec;

    fn roundtrip(frame: H3Frame<'_>) -> Vec<u8> {
        let mut buf = vec![0u8; 256];
        let written = encode(&frame, &mut buf).expect("encode");
        buf.truncate(written);
        let (parsed, consumed) = parse(&buf).expect("parse");
        assert_eq!(consumed, buf.len(), "consume full frame");
        assert_eq!(parsed, frame);
        buf
    }

    #[test]
    fn data_frame_roundtrip() {
        let payload = b"hello world";
        roundtrip(H3Frame::Data { payload });
    }

    #[test]
    fn headers_frame_roundtrip() {
        let header_block = b"\x00\x00\xc1\xd1";
        roundtrip(H3Frame::Headers { header_block });
    }

    #[test]
    fn cancel_push_roundtrip() {
        roundtrip(H3Frame::CancelPush { push_id: 42 });
    }

    #[test]
    fn settings_roundtrip_with_iter() {
        // RFC 9114 §7.2.4 SETTINGS payload — two pairs:
        // (SETTINGS_QPACK_MAX_TABLE_CAPACITY=0x01, 4096)
        // (SETTINGS_MAX_FIELD_SECTION_SIZE=0x06, 16384)
        let mut payload_buf = vec![0u8; 16];
        let mut cursor = 0;
        cursor += varint::encode(0x01, &mut payload_buf[cursor..]).unwrap();
        cursor += varint::encode(4096, &mut payload_buf[cursor..]).unwrap();
        cursor += varint::encode(0x06, &mut payload_buf[cursor..]).unwrap();
        cursor += varint::encode(16384, &mut payload_buf[cursor..]).unwrap();
        payload_buf.truncate(cursor);

        let encoded = roundtrip(H3Frame::Settings {
            payload: &payload_buf,
        });
        let (parsed, _) = parse(&encoded).unwrap();
        let H3Frame::Settings { payload } = parsed else {
            panic!("expected Settings");
        };
        let pairs: Vec<(u64, u64)> = SettingsIter::new(payload).map(Result::unwrap).collect();
        assert_eq!(pairs, vec![(0x01, 4096), (0x06, 16384)]);
    }

    #[test]
    fn push_promise_roundtrip() {
        let header_block = b"\x00\xc2";
        roundtrip(H3Frame::PushPromise {
            push_id: 7,
            header_block,
        });
    }

    #[test]
    fn goaway_roundtrip() {
        roundtrip(H3Frame::GoAway { id: 0 });
        roundtrip(H3Frame::GoAway { id: 1024 });
    }

    #[test]
    fn max_push_id_roundtrip() {
        roundtrip(H3Frame::MaxPushId { push_id: 100 });
    }

    #[test]
    fn reserved_frame_preserved() {
        let payload = b"opaque";
        roundtrip(H3Frame::Reserved {
            frame_type: 0x21,
            payload,
        });
    }

    #[test]
    fn truncated_frame_rejected() {
        // type=DATA, length=5, but only 3 payload bytes follow.
        let buf = vec![0x00, 0x05, 0xAA, 0xBB, 0xCC];
        assert_eq!(parse(&buf), Err(FrameError::Truncated));
    }

    #[test]
    fn empty_input_rejected() {
        assert!(matches!(
            parse(&[]),
            Err(FrameError::InvalidVarint | FrameError::Truncated)
        ));
    }

    #[test]
    fn encode_buffer_too_small_returns_needed() {
        let payload = b"abcdefg";
        let frame = H3Frame::Data { payload };
        let mut tiny = [0u8; 2];
        let err = encode(&frame, &mut tiny).unwrap_err();
        let FrameError::BufferTooSmall { needed } = err else {
            panic!("expected BufferTooSmall, got {err:?}");
        };
        // type(1) + length varint(1) + payload(7) = 9.
        assert_eq!(needed, 9);
    }

    #[test]
    fn cancel_push_with_trailing_garbage_rejected() {
        // type=CANCEL_PUSH(0x03) + length(2) + push_id varint=0x05 (1 byte)
        // + trailing 0xFF. push_id parses but doesn't consume the whole
        // payload — codec MUST reject per RFC 9114 §7.2.3.
        let buf = vec![0x03, 0x02, 0x05, 0xFF];
        assert_eq!(parse(&buf), Err(FrameError::InvalidVarint));
    }

    #[test]
    fn settings_iter_truncated_pair_rejected() {
        // SETTINGS payload with one orphan id (no value).
        let payload = vec![0x42u8];
        let mut iter = SettingsIter::new(&payload);
        let first = iter.next().unwrap();
        assert_eq!(first, Err(FrameError::Truncated));
    }

    /// RFC 9114 §7.2.4.1 — receiving an unknown SETTINGS id MUST be
    /// ignored. The iterator surfaces the (id, value) pair to the
    /// caller without judgement; the policy lives one layer up.
    #[test]
    fn settings_iter_yields_unknown_ids_for_caller_policy() {
        let mut payload_buf = vec![0u8; 8];
        let mut cursor = 0;
        cursor += varint::encode(0xDEAD, &mut payload_buf[cursor..]).unwrap();
        cursor += varint::encode(0xBEEF, &mut payload_buf[cursor..]).unwrap();
        payload_buf.truncate(cursor);
        let pairs: Vec<(u64, u64)> = SettingsIter::new(&payload_buf)
            .map(Result::unwrap)
            .collect();
        assert_eq!(pairs, vec![(0xDEAD, 0xBEEF)]);
    }

    /// RFC 9114 §11.2.1 — the four HTTP/2 frame types reserved in
    /// HTTP/3 (PRIORITY, PING, WINDOW_UPDATE, CONTINUATION) MUST be
    /// classified as connection errors. `is_http2_reserved` returns
    /// `true` exactly for those four; `is_grease` returns `false`
    /// for the same set (the carve-out makes the precedence explicit).
    #[test]
    fn http2_reserved_classification_matches_rfc_9114_11_2_1() {
        for &reserved in HTTP2_RESERVED_FRAME_TYPES {
            assert!(is_http2_reserved(reserved), "0x{reserved:02x} is reserved");
            assert!(!is_grease(reserved), "0x{reserved:02x} is NOT grease");
        }
    }

    /// RFC 9114 §7.2.8 GREASE pattern: 0x1f * N + 0x21. Spot-check
    /// the first six values (covers N = 0..5) and that no known
    /// in-use frame type falls into the set.
    #[test]
    fn grease_classification_matches_rfc_9114_7_2_8_pattern() {
        let expected_grease = [0x21, 0x40, 0x5f, 0x7e, 0x9d, 0xbc];
        for value in expected_grease {
            assert!(is_grease(value), "0x{value:02x} should be GREASE");
        }
        for known in [
            FRAME_TYPE_DATA,
            FRAME_TYPE_HEADERS,
            FRAME_TYPE_CANCEL_PUSH,
            FRAME_TYPE_SETTINGS,
            FRAME_TYPE_PUSH_PROMISE,
            FRAME_TYPE_GOAWAY,
            FRAME_TYPE_MAX_PUSH_ID,
        ] {
            assert!(
                !is_grease(known),
                "known type 0x{known:02x} must NOT match the GREASE pattern"
            );
        }
    }
}
