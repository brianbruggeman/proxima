//! WebSocket frame parser (sans-IO).
//!
//! Tracked as P11 in `docs/protocol-gap/discipline.md`. RFC 6455
//! defines the WebSocket framing layer separately from the rest of
//! the protocol (handshake, ping/pong semantics, close codes). This
//! module is the framing layer alone — substrate middleware can
//! inspect frames without depending on `async-tungstenite`'s io
//! plumbing.
//!
//! Frame layout:
//!
//! ```text
//!  0                   1                   2                   3
//!  0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
//! +-+-+-+-+-------+-+-------------+-------------------------------+
//! |F|R|R|R| opcode|M| Payload len |    Extended payload length    |
//! |I|S|S|S|  (4)  |A|     (7)     |             (16/64)           |
//! |N|V|V|V|       |S|             |   (if payload len==126/127)   |
//! | |1|2|3|       |K|             |                               |
//! +-+-+-+-+-------+-+-------------+ - - - - - - - - - - - - - - - +
//! |     Extended payload length continued, if payload len == 127  |
//! + - - - - - - - - - - - - - - - +-------------------------------+
//! |                               |Masking-key, if MASK set to 1  |
//! +-------------------------------+-------------------------------+
//! | Masking-key (continued)       |          Payload Data         |
//! +-------------------------------- - - - - - - - - - - - - - - - +
//! ```
//!
//! Reference: `tungstenite::protocol::frame::FrameHeader` —
//! scope-matched ecosystem baseline. The substrate parser borrows
//! from the source buffer; the payload reference is masked-as-is
//! per the wire (callers that need unmasked bytes call
//! [`unmask_in_place`] on a `&mut [u8]` copy).
//!
//! Sub-flag: `websocket-frame` (default off).


use alloc::vec::Vec;

/// WebSocket frame opcode (low nibble of the first header byte).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Opcode {
    Continuation = 0x0,
    Text = 0x1,
    Binary = 0x2,
    Close = 0x8,
    Ping = 0x9,
    Pong = 0xA,
}

impl Opcode {
    #[inline]
    fn from_bits(bits: u8) -> Result<Self, ParseError> {
        match bits {
            0x0 => Ok(Self::Continuation),
            0x1 => Ok(Self::Text),
            0x2 => Ok(Self::Binary),
            0x8 => Ok(Self::Close),
            0x9 => Ok(Self::Ping),
            0xA => Ok(Self::Pong),
            other => Err(ParseError::UnknownOpcode(other)),
        }
    }
}

/// Parsed WebSocket frame. Borrows the payload slice from the source
/// buffer. Mask key (when present) is also borrowed — callers that
/// need to unmask the payload do so via [`unmask_in_place`] on a
/// caller-owned copy.
#[derive(Debug, Clone, Copy)]
pub struct Frame<'a> {
    pub fin: bool,
    pub opcode: Opcode,
    /// RSV1 — set on the first frame of a permessage-deflate (RFC 7692)
    /// message. `true` ⇒ payload is DEFLATE-compressed and the caller
    /// must inflate before interpreting it.
    pub compressed: bool,
    /// Masking key — RFC 6455 §5.3 requires clients to mask, servers
    /// to send unmasked. `None` ⇒ payload is plaintext on the wire.
    pub mask: Option<[u8; 4]>,
    pub payload: &'a [u8],
}

#[derive(Debug, proxima_macros::Error)]
pub enum ParseError {
    #[error("buffer ended mid-frame")]
    Short,
    #[error("reserved bits RSV1/RSV2/RSV3 set — extensions not negotiated")]
    ReservedBits,
    #[error("opcode 0x{0:X} is reserved or undefined")]
    UnknownOpcode(u8),
    #[error("control frame payload exceeds 125 bytes (got {0})")]
    OversizedControl(u64),
    #[error("declared payload length {0} exceeds buffer")]
    PartialPayload(u64),
    #[error("declared payload length {0} exceeds usize bounds on this platform")]
    PayloadTooLarge(u64),
}

/// Parse one full WebSocket frame at the start of `buf`, with no extensions
/// negotiated. Per RFC 6455 §5.2 any set reserved bit (RSV1/2/3) is a
/// must-fail, so this strict variant rejects all of them. Returns the parsed
/// frame and total bytes consumed (header + mask + payload).
#[inline]
pub fn parse_frame(buf: &[u8]) -> Result<(Frame<'_>, usize), ParseError> {
    parse_frame_with_extensions(buf, false)
}

/// Parse one full WebSocket frame, honoring a negotiated permessage-deflate
/// (RFC 7692) extension. When `permessage_deflate` is true, RSV1 is the
/// per-message-compressed bit and surfaces as [`Frame::compressed`]; RSV2/RSV3
/// remain undefined and are rejected. When false this is identical to the
/// strict [`parse_frame`]. The caller learns the negotiation from the
/// handshake's `Sec-WebSocket-Extensions` response header.
#[inline]
pub fn parse_frame_with_extensions(
    buf: &[u8],
    permessage_deflate: bool,
) -> Result<(Frame<'_>, usize), ParseError> {
    if buf.len() < 2 {
        return Err(ParseError::Short);
    }
    let first = buf[0];
    let second = buf[1];
    let fin = first & 0x80 != 0;
    // RFC 6455 §5.2: a set reserved bit with no negotiated extension defining
    // it MUST fail the connection. RSV1 (0x40) is defined only when
    // permessage-deflate was negotiated; RSV2/RSV3 (0x30) have no extension we
    // support, so they always fail.
    let rsv1 = first & 0x40 != 0;
    if first & 0x30 != 0 || (rsv1 && !permessage_deflate) {
        return Err(ParseError::ReservedBits);
    }
    let compressed = rsv1;
    let opcode = Opcode::from_bits(first & 0x0F)?;
    let masked = second & 0x80 != 0;
    let len_field = second & 0x7F;

    let (payload_len, after_len_off) = match len_field {
        126 => {
            if buf.len() < 4 {
                return Err(ParseError::Short);
            }
            let value = u16::from_be_bytes([buf[2], buf[3]]) as u64;
            (value, 4)
        }
        127 => {
            if buf.len() < 10 {
                return Err(ParseError::Short);
            }
            let value = u64::from_be_bytes([
                buf[2], buf[3], buf[4], buf[5], buf[6], buf[7], buf[8], buf[9],
            ]);
            (value, 10)
        }
        n => (u64::from(n), 2),
    };

    // Control frames (0x8/0x9/0xA) must be ≤125 bytes per RFC 6455 §5.5.
    if matches!(opcode, Opcode::Close | Opcode::Ping | Opcode::Pong) && payload_len > 125 {
        return Err(ParseError::OversizedControl(payload_len));
    }

    let payload_len_usize: usize = payload_len
        .try_into()
        .map_err(|_| ParseError::PayloadTooLarge(payload_len))?;

    let (mask, header_end) = if masked {
        if buf.len() < after_len_off + 4 {
            return Err(ParseError::Short);
        }
        let key = [
            buf[after_len_off],
            buf[after_len_off + 1],
            buf[after_len_off + 2],
            buf[after_len_off + 3],
        ];
        (Some(key), after_len_off + 4)
    } else {
        (None, after_len_off)
    };

    let payload_end = header_end + payload_len_usize;
    if buf.len() < payload_end {
        return Err(ParseError::PartialPayload(payload_len));
    }
    Ok((
        Frame {
            fin,
            opcode,
            compressed,
            mask,
            payload: &buf[header_end..payload_end],
        },
        payload_end,
    ))
}

/// Apply or remove WebSocket masking in place. RFC 6455 §5.3:
/// `unmasked[i] = masked[i] ^ key[i % 4]`. Idempotent — running
/// twice returns the original bytes.
#[inline]
pub fn unmask_in_place(payload: &mut [u8], key: [u8; 4]) {
    for (i, byte) in payload.iter_mut().enumerate() {
        *byte ^= key[i & 0x03];
    }
}

/// Encoded header length given payload byte count + whether the
/// frame is masked. Useful for buffer sizing before encoding.
#[inline]
#[must_use]
pub fn encoded_header_len(payload_len: usize, masked: bool) -> usize {
    let base = match payload_len {
        0..=125 => 2,
        126..=0xFFFF => 4,
        _ => 10,
    };
    if masked { base + 4 } else { base }
}

/// Encode the frame header into `dest`. Does NOT write the payload —
/// caller writes the payload bytes (already masked if `mask` is set).
#[inline]
pub fn encode_header(
    fin: bool,
    opcode: Opcode,
    payload_len: usize,
    mask: Option<[u8; 4]>,
    dest: &mut Vec<u8>,
) {
    let mut first = opcode as u8;
    if fin {
        first |= 0x80;
    }
    dest.push(first);

    let mut second: u8 = if mask.is_some() { 0x80 } else { 0 };
    match payload_len {
        0..=125 => {
            second |= payload_len as u8;
            dest.push(second);
        }
        126..=0xFFFF => {
            second |= 126;
            dest.push(second);
            dest.extend_from_slice(&(payload_len as u16).to_be_bytes());
        }
        _ => {
            second |= 127;
            dest.push(second);
            dest.extend_from_slice(&(payload_len as u64).to_be_bytes());
        }
    }
    if let Some(key) = mask {
        dest.extend_from_slice(&key);
    }
}

#[cfg(feature = "websocket_frame-codec-trait")]
pub mod codec_trait;
#[cfg(feature = "websocket_frame-codec-trait")]
pub use codec_trait::WebSocketFrameCodec;

#[cfg(feature = "websocket_frame-frame-pipe")]
pub mod frame_codec_pipe;
#[cfg(feature = "websocket_frame-frame-pipe")]
pub use frame_codec_pipe::OwnedWsFrame;

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn parses_small_unmasked_text_frame() {
        // FIN + text, no mask, 5-byte payload "hello"
        let mut buf = vec![0x81, 0x05];
        buf.extend_from_slice(b"hello");
        let (frame, used) = parse_frame(&buf).unwrap();
        assert_eq!(used, buf.len());
        assert!(frame.fin);
        assert_eq!(frame.opcode, Opcode::Text);
        assert!(frame.mask.is_none());
        assert_eq!(frame.payload, b"hello");
    }

    #[test]
    fn parses_masked_binary_frame() {
        let key = [0x12, 0x34, 0x56, 0x78];
        let payload = b"hi";
        let mut buf = vec![0x82, 0x82];
        buf.extend_from_slice(&key);
        for (i, byte) in payload.iter().enumerate() {
            buf.push(byte ^ key[i & 0x03]);
        }
        let (frame, _) = parse_frame(&buf).unwrap();
        assert_eq!(frame.opcode, Opcode::Binary);
        assert_eq!(frame.mask, Some(key));
        let mut payload_copy = frame.payload.to_vec();
        unmask_in_place(&mut payload_copy, key);
        assert_eq!(payload_copy, b"hi");
    }

    #[test]
    fn parses_16bit_extended_length() {
        // 200-byte payload requires 16-bit extended length encoding.
        let payload = vec![0xAB; 200];
        let mut buf = vec![0x82, 126];
        buf.extend_from_slice(&200u16.to_be_bytes());
        buf.extend_from_slice(&payload);
        let (frame, _) = parse_frame(&buf).unwrap();
        assert_eq!(frame.payload.len(), 200);
    }

    #[test]
    fn parses_64bit_extended_length() {
        let payload = vec![0xCD; 70_000];
        let mut buf = vec![0x82, 127];
        buf.extend_from_slice(&(70_000u64).to_be_bytes());
        buf.extend_from_slice(&payload);
        let (frame, _) = parse_frame(&buf).unwrap();
        assert_eq!(frame.payload.len(), 70_000);
    }

    #[test]
    fn parses_ping_pong_close() {
        for (opcode_byte, expected) in [
            (0x89, Opcode::Ping),
            (0x8A, Opcode::Pong),
            (0x88, Opcode::Close),
        ] {
            let buf = [opcode_byte, 0];
            let (frame, _) = parse_frame(&buf).unwrap();
            assert!(frame.fin);
            assert_eq!(frame.opcode, expected);
            assert!(frame.payload.is_empty());
        }
    }

    #[test]
    fn fragmented_text_continuation() {
        // First frame: FIN=0, Text, 1 byte
        let mut first = vec![0x01, 0x01, b'h'];
        // Continuation: FIN=1, Continuation, 1 byte
        let cont = vec![0x80, 0x01, b'i'];
        first.extend_from_slice(&cont);
        let (frame_a, used_a) = parse_frame(&first).unwrap();
        assert!(!frame_a.fin);
        assert_eq!(frame_a.opcode, Opcode::Text);
        let (frame_b, _) = parse_frame(&first[used_a..]).unwrap();
        assert!(frame_b.fin);
        assert_eq!(frame_b.opcode, Opcode::Continuation);
    }

    #[test]
    fn rejects_reserved_bits() {
        // RSV1 set with no extension negotiated -> RFC 6455 must-fail
        let buf = [0xC1, 0x00];
        match parse_frame(&buf) {
            Err(ParseError::ReservedBits) => {}
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn strict_parse_marks_frames_uncompressed() {
        let mut buf = vec![0x81, 0x05];
        buf.extend_from_slice(b"hello");
        let (frame, _) = parse_frame(&buf).unwrap();
        assert!(!frame.compressed, "no RSV1 -> not compressed");
    }

    #[test]
    fn permessage_deflate_accepts_rsv1_as_compressed() {
        // FIN + RSV1 + text, 3-byte (opaque deflate) payload
        let mut buf = vec![0xC1, 0x03];
        buf.extend_from_slice(&[0xAA, 0xBB, 0xCC]);
        let (frame, used) = parse_frame_with_extensions(&buf, true).unwrap();
        assert_eq!(used, buf.len());
        assert!(
            frame.compressed,
            "RSV1 under permessage-deflate -> compressed"
        );
        assert_eq!(frame.opcode, Opcode::Text);
        assert_eq!(frame.payload, &[0xAA, 0xBB, 0xCC]);
    }

    #[test]
    fn permessage_deflate_still_rejects_rsv2_and_rsv3() {
        // RSV2 (0x20) and RSV3 (0x10) have no negotiated extension even when
        // permessage-deflate is on -> must still fail.
        for first in [0xA1u8, 0x91u8] {
            match parse_frame_with_extensions(&[first, 0x00], true) {
                Err(ParseError::ReservedBits) => {}
                other => panic!("expected ReservedBits for {first:#x}, got {other:?}"),
            }
        }
    }

    #[test]
    fn rejects_unknown_opcode() {
        let buf = [0x83, 0x00]; // opcode 0x3 (reserved non-control)
        match parse_frame(&buf) {
            Err(ParseError::UnknownOpcode(0x3)) => {}
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn rejects_oversized_control() {
        let mut buf = vec![0x89, 126];
        buf.extend_from_slice(&200u16.to_be_bytes());
        buf.extend_from_slice(&[0u8; 200]);
        match parse_frame(&buf) {
            Err(ParseError::OversizedControl(200)) => {}
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn short_buffer_returns_short() {
        match parse_frame(&[0x82]) {
            Err(ParseError::Short) => {}
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn partial_payload_returns_partial() {
        let buf = [0x82, 10]; // declares 10 byte payload, none supplied
        match parse_frame(&buf) {
            Err(ParseError::PartialPayload(10)) => {}
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn encode_round_trips() {
        for (payload_len, expected_header) in
            [(0, 2), (125, 2), (126, 4), (0xFFFF, 4), (0x10000, 10)]
        {
            let mut dest = Vec::new();
            encode_header(true, Opcode::Binary, payload_len, None, &mut dest);
            assert_eq!(dest.len(), expected_header);
            assert_eq!(encoded_header_len(payload_len, false), expected_header);
        }
        // Masked variant adds 4 bytes
        let mut dest = Vec::new();
        encode_header(true, Opcode::Text, 5, Some([1, 2, 3, 4]), &mut dest);
        assert_eq!(dest.len(), 2 + 4);
        assert_eq!(encoded_header_len(5, true), 6);
    }

    #[test]
    fn unmask_is_idempotent() {
        let key = [0xAB, 0xCD, 0xEF, 0x01];
        let original = b"the quick brown fox";
        let mut buf = original.to_vec();
        unmask_in_place(&mut buf, key);
        assert_ne!(buf, original);
        unmask_in_place(&mut buf, key);
        assert_eq!(buf, original);
    }
}
