//! RFC 9000 §19 QUIC frame codec + RFC 9221 §4 DATAGRAM frame.
//!
//! Every frame variant on the wire is represented by a discriminated enum
//! [`Frame`] with per-variant borrowed slices for variable-length payloads.
//! All fields are tier-3 (bare `no_std + no_alloc`): inputs borrowed,
//! outputs written into the caller's buffer, no `Vec`, no `Bytes`.
//!
//! Frame types covered:
//!
//! | Code  | Frame | RFC 9000 § |
//! |-------|-------|------------|
//! | 0x00  | PADDING | §19.1 |
//! | 0x01  | PING | §19.2 |
//! | 0x02-0x03 | ACK (with optional ECN counts) | §19.3 |
//! | 0x04  | RESET_STREAM | §19.4 |
//! | 0x05  | STOP_SENDING | §19.5 |
//! | 0x06  | CRYPTO | §19.6 |
//! | 0x07  | NEW_TOKEN | §19.7 |
//! | 0x08-0x0f | STREAM (OFF/LEN/FIN bits) | §19.8 |
//! | 0x10  | MAX_DATA | §19.9 |
//! | 0x11  | MAX_STREAM_DATA | §19.10 |
//! | 0x12-0x13 | MAX_STREAMS (bi/uni) | §19.11 |
//! | 0x14  | DATA_BLOCKED | §19.12 |
//! | 0x15  | STREAM_DATA_BLOCKED | §19.13 |
//! | 0x16-0x17 | STREAMS_BLOCKED (bi/uni) | §19.14 |
//! | 0x18  | NEW_CONNECTION_ID | §19.15 |
//! | 0x19  | RETIRE_CONNECTION_ID | §19.16 |
//! | 0x1a  | PATH_CHALLENGE | §19.17 |
//! | 0x1b  | PATH_RESPONSE | §19.18 |
//! | 0x1c-0x1d | CONNECTION_CLOSE (quic/app) | §19.19 |
//! | 0x1e  | HANDSHAKE_DONE | §19.20 |
//! | 0x30-0x31 | DATAGRAM (with/without length) | RFC 9221 §4 |
//!
//! ACK frame ranges and CONNECTION_CLOSE reason phrase are surfaced as
//! borrowed slices; the caller iterates ranges via
//! [`AckRanges`] or interprets the reason as UTF-8 only when the
//! application semantically requires it (parsers stay tier-3, no UTF-8
//! validation in the hot path).

use crate::quic::varint;

/// PATH_CHALLENGE / PATH_RESPONSE carry an 8-byte opaque value (RFC 9000 §19.17).
pub const PATH_CHALLENGE_LEN: usize = 8;
/// NEW_CONNECTION_ID carries a 16-byte stateless-reset token (RFC 9000 §19.15).
pub const STATELESS_RESET_TOKEN_LEN: usize = 16;

/// ECN counts attached to an ACK frame (RFC 9000 §19.3.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EcnCounts {
    pub ect0: u64,
    pub ect1: u64,
    pub ecn_ce: u64,
}

/// Parsed QUIC frame. All variable-length fields borrow into the input slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Frame<'a> {
    /// RFC 9000 §19.1 — one or more PADDING frames coalesce into a single
    /// variant carrying the count of consecutive padding bytes consumed.
    Padding { count: usize },
    /// RFC 9000 §19.2.
    Ping,
    /// RFC 9000 §19.3.
    Ack {
        largest: u64,
        delay: u64,
        first_range: u64,
        /// Raw bytes for the (gap, ack_range_length) varint pairs.
        /// Use [`AckRanges::new`] to iterate.
        ranges_raw: &'a [u8],
        /// Number of (gap, range) pairs encoded in `ranges_raw`.
        range_count: u64,
        ecn: Option<EcnCounts>,
    },
    /// RFC 9000 §19.4.
    ResetStream {
        stream_id: u64,
        error_code: u64,
        final_size: u64,
    },
    /// RFC 9000 §19.5.
    StopSending { stream_id: u64, error_code: u64 },
    /// RFC 9000 §19.6.
    Crypto { offset: u64, data: &'a [u8] },
    /// RFC 9000 §19.7.
    NewToken { token: &'a [u8] },
    /// RFC 9000 §19.8.
    Stream {
        stream_id: u64,
        offset: u64,
        data: &'a [u8],
        fin: bool,
    },
    /// RFC 9000 §19.9.
    MaxData { maximum: u64 },
    /// RFC 9000 §19.10.
    MaxStreamData { stream_id: u64, maximum: u64 },
    /// RFC 9000 §19.11.
    MaxStreams { bidi: bool, maximum: u64 },
    /// RFC 9000 §19.12.
    DataBlocked { maximum: u64 },
    /// RFC 9000 §19.13.
    StreamDataBlocked { stream_id: u64, maximum: u64 },
    /// RFC 9000 §19.14.
    StreamsBlocked { bidi: bool, maximum: u64 },
    /// RFC 9000 §19.15.
    NewConnectionId {
        sequence: u64,
        retire_prior_to: u64,
        connection_id: &'a [u8],
        stateless_reset_token: &'a [u8; STATELESS_RESET_TOKEN_LEN],
    },
    /// RFC 9000 §19.16.
    RetireConnectionId { sequence: u64 },
    /// RFC 9000 §19.17.
    PathChallenge { data: [u8; PATH_CHALLENGE_LEN] },
    /// RFC 9000 §19.18.
    PathResponse { data: [u8; PATH_CHALLENGE_LEN] },
    /// RFC 9000 §19.19.
    ConnectionClose {
        error_code: u64,
        /// `Some(t)` for a transport-error close (frame type 0x1c, with
        /// `frame_type` set to the frame that triggered the error or 0);
        /// `None` for an application-error close (frame type 0x1d).
        frame_type: Option<u64>,
        reason: &'a [u8],
    },
    /// RFC 9000 §19.20.
    HandshakeDone,
    /// RFC 9221 §4 — extension carries opaque datagram bytes; length is
    /// either varint-prefixed (0x31) or implicit to end-of-packet (0x30).
    Datagram { data: &'a [u8] },
}

/// Iterator over (gap, ack_range_length) pairs encoded in [`Frame::Ack::ranges_raw`].
///
/// Each iteration yields the next `(gap, ack_range_length)` pair as varints
/// per RFC 9000 §19.3.1. The caller is responsible for reconstructing the
/// actual acknowledged packet-number ranges relative to `largest_ack` /
/// `first_ack_range`.
pub struct AckRanges<'a> {
    raw: &'a [u8],
    remaining: u64,
}

impl<'a> AckRanges<'a> {
    /// Construct an iterator over the `range_count` (gap, length) pairs.
    #[must_use]
    pub fn new(raw: &'a [u8], range_count: u64) -> Self {
        Self {
            raw,
            remaining: range_count,
        }
    }
}

impl Iterator for AckRanges<'_> {
    type Item = Result<(u64, u64), DecodeError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;
        let (gap, consumed) = match varint::decode(self.raw) {
            Ok(pair) => pair,
            Err(_) => return Some(Err(DecodeError::Truncated)),
        };
        self.raw = &self.raw[consumed..];
        let (length, consumed) = match varint::decode(self.raw) {
            Ok(pair) => pair,
            Err(_) => return Some(Err(DecodeError::Truncated)),
        };
        self.raw = &self.raw[consumed..];
        Some(Ok((gap, length)))
    }
}

/// Parse failures for [`parse`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DecodeError {
    /// Input buffer is empty (no frame type byte).
    Empty,
    /// Input ran out before a required field completed.
    Truncated,
    /// Frame type byte is not a known type per RFC 9000 §19 or RFC 9221.
    UnknownFrameType(u64),
    /// A length-prefix varint exceeded the remaining input buffer.
    LengthOverflowsBuffer,
}

/// Encode failures for [`Frame::encode`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum EncodeError {
    /// Output buffer too small for the encoded frame.
    BufferTooSmall,
    /// A field exceeded the varint encoding range (2^62 - 1).
    ValueTooLarge,
}

/// Parse the next frame from `input`. Returns the parsed frame and the
/// number of bytes consumed.
///
/// PADDING frames coalesce — a run of N consecutive 0x00 bytes yields one
/// [`Frame::Padding`] variant with `count = N`.
///
/// # Errors
///
/// See [`DecodeError`].
#[allow(clippy::too_many_lines)]
pub fn parse(input: &[u8]) -> Result<(Frame<'_>, usize), DecodeError> {
    let Some(&first) = input.first() else {
        return Err(DecodeError::Empty);
    };
    // PADDING is special — the frame type IS 0x00 and consecutive 0x00s
    // are treated as one logical PADDING run per RFC 9000 §19.1.
    if first == 0x00 {
        let mut count = 0usize;
        while count < input.len() && input[count] == 0x00 {
            count += 1;
        }
        return Ok((Frame::Padding { count }, count));
    }

    // For non-PADDING, the frame type is itself a varint (RFC 9000 §12.4).
    let (frame_type, type_len) = varint::decode(input).map_err(|_| DecodeError::Truncated)?;
    let body = &input[type_len..];

    match frame_type {
        0x01 => Ok((Frame::Ping, type_len)),
        0x02 | 0x03 => parse_ack(body, type_len, frame_type == 0x03),
        0x04 => parse_reset_stream(body, type_len),
        0x05 => parse_stop_sending(body, type_len),
        0x06 => parse_crypto(body, type_len),
        0x07 => parse_new_token(body, type_len),
        ty @ 0x08..=0x0f => parse_stream(body, type_len, ty as u8),
        0x10 => parse_max_data(body, type_len),
        0x11 => parse_max_stream_data(body, type_len),
        0x12 | 0x13 => parse_max_streams(body, type_len, frame_type == 0x12),
        0x14 => parse_data_blocked(body, type_len),
        0x15 => parse_stream_data_blocked(body, type_len),
        0x16 | 0x17 => parse_streams_blocked(body, type_len, frame_type == 0x16),
        0x18 => parse_new_connection_id(body, type_len),
        0x19 => parse_retire_connection_id(body, type_len),
        0x1a => parse_path_challenge(body, type_len),
        0x1b => parse_path_response(body, type_len),
        0x1c => parse_connection_close(body, type_len, true),
        0x1d => parse_connection_close(body, type_len, false),
        0x1e => Ok((Frame::HandshakeDone, type_len)),
        // RFC 9221 — DATAGRAM with implicit length consumes the rest of input
        0x30 => Ok((Frame::Datagram { data: body }, type_len + body.len())),
        // RFC 9221 — DATAGRAM with explicit length prefix
        0x31 => parse_datagram_with_length(body, type_len),
        other => Err(DecodeError::UnknownFrameType(other)),
    }
}

// ---- per-variant parse helpers ---------------------------------------------

fn read_varint(input: &[u8], cursor: &mut usize) -> Result<u64, DecodeError> {
    let slice = input.get(*cursor..).ok_or(DecodeError::Truncated)?;
    let (value, consumed) = varint::decode(slice).map_err(|_| DecodeError::Truncated)?;
    *cursor += consumed;
    Ok(value)
}

fn read_slice<'a>(
    input: &'a [u8],
    cursor: &mut usize,
    len: usize,
) -> Result<&'a [u8], DecodeError> {
    let start = *cursor;
    let end = start.checked_add(len).ok_or(DecodeError::Truncated)?;
    let slice = input.get(start..end).ok_or(DecodeError::Truncated)?;
    *cursor = end;
    Ok(slice)
}

fn parse_ack(
    body: &[u8],
    type_len: usize,
    with_ecn: bool,
) -> Result<(Frame<'_>, usize), DecodeError> {
    let mut cursor = 0;
    let largest = read_varint(body, &mut cursor)?;
    let delay = read_varint(body, &mut cursor)?;
    let range_count = read_varint(body, &mut cursor)?;
    let first_range = read_varint(body, &mut cursor)?;

    // skip past `range_count` (gap, ack_range_length) varint pairs to
    // determine the ranges_raw slice and the total ACK frame length.
    let ranges_start = cursor;
    for _index in 0..range_count {
        let _gap = read_varint(body, &mut cursor)?;
        let _length = read_varint(body, &mut cursor)?;
    }
    let ranges_raw = &body[ranges_start..cursor];

    let ecn = if with_ecn {
        let ect0 = read_varint(body, &mut cursor)?;
        let ect1 = read_varint(body, &mut cursor)?;
        let ecn_ce = read_varint(body, &mut cursor)?;
        Some(EcnCounts { ect0, ect1, ecn_ce })
    } else {
        None
    };

    Ok((
        Frame::Ack {
            largest,
            delay,
            first_range,
            ranges_raw,
            range_count,
            ecn,
        },
        type_len + cursor,
    ))
}

fn parse_reset_stream(body: &[u8], type_len: usize) -> Result<(Frame<'_>, usize), DecodeError> {
    let mut cursor = 0;
    let stream_id = read_varint(body, &mut cursor)?;
    let error_code = read_varint(body, &mut cursor)?;
    let final_size = read_varint(body, &mut cursor)?;
    Ok((
        Frame::ResetStream {
            stream_id,
            error_code,
            final_size,
        },
        type_len + cursor,
    ))
}

fn parse_stop_sending(body: &[u8], type_len: usize) -> Result<(Frame<'_>, usize), DecodeError> {
    let mut cursor = 0;
    let stream_id = read_varint(body, &mut cursor)?;
    let error_code = read_varint(body, &mut cursor)?;
    Ok((
        Frame::StopSending {
            stream_id,
            error_code,
        },
        type_len + cursor,
    ))
}

fn parse_crypto(body: &[u8], type_len: usize) -> Result<(Frame<'_>, usize), DecodeError> {
    let mut cursor = 0;
    let offset = read_varint(body, &mut cursor)?;
    let length = read_varint(body, &mut cursor)?;
    let length = usize::try_from(length).map_err(|_| DecodeError::LengthOverflowsBuffer)?;
    let data = read_slice(body, &mut cursor, length)?;
    Ok((Frame::Crypto { offset, data }, type_len + cursor))
}

fn parse_new_token(body: &[u8], type_len: usize) -> Result<(Frame<'_>, usize), DecodeError> {
    let mut cursor = 0;
    let length = read_varint(body, &mut cursor)?;
    let length = usize::try_from(length).map_err(|_| DecodeError::LengthOverflowsBuffer)?;
    let token = read_slice(body, &mut cursor, length)?;
    Ok((Frame::NewToken { token }, type_len + cursor))
}

fn parse_stream(
    body: &[u8],
    type_len: usize,
    type_byte: u8,
) -> Result<(Frame<'_>, usize), DecodeError> {
    // bits: 0x08 | (OFF << 2) | (LEN << 1) | FIN
    let off_present = type_byte & 0b100 != 0;
    let len_present = type_byte & 0b010 != 0;
    let fin = type_byte & 0b001 != 0;

    let mut cursor = 0;
    let stream_id = read_varint(body, &mut cursor)?;
    let offset = if off_present {
        read_varint(body, &mut cursor)?
    } else {
        0
    };
    let data = if len_present {
        let length = read_varint(body, &mut cursor)?;
        let length = usize::try_from(length).map_err(|_| DecodeError::LengthOverflowsBuffer)?;
        read_slice(body, &mut cursor, length)?
    } else {
        // length-implicit — consumes the rest of the input slice
        let rest = body.get(cursor..).ok_or(DecodeError::Truncated)?;
        cursor += rest.len();
        rest
    };
    Ok((
        Frame::Stream {
            stream_id,
            offset,
            data,
            fin,
        },
        type_len + cursor,
    ))
}

fn parse_max_data(body: &[u8], type_len: usize) -> Result<(Frame<'_>, usize), DecodeError> {
    let mut cursor = 0;
    let maximum = read_varint(body, &mut cursor)?;
    Ok((Frame::MaxData { maximum }, type_len + cursor))
}

fn parse_max_stream_data(body: &[u8], type_len: usize) -> Result<(Frame<'_>, usize), DecodeError> {
    let mut cursor = 0;
    let stream_id = read_varint(body, &mut cursor)?;
    let maximum = read_varint(body, &mut cursor)?;
    Ok((
        Frame::MaxStreamData { stream_id, maximum },
        type_len + cursor,
    ))
}

fn parse_max_streams(
    body: &[u8],
    type_len: usize,
    bidi: bool,
) -> Result<(Frame<'_>, usize), DecodeError> {
    let mut cursor = 0;
    let maximum = read_varint(body, &mut cursor)?;
    Ok((Frame::MaxStreams { bidi, maximum }, type_len + cursor))
}

fn parse_data_blocked(body: &[u8], type_len: usize) -> Result<(Frame<'_>, usize), DecodeError> {
    let mut cursor = 0;
    let maximum = read_varint(body, &mut cursor)?;
    Ok((Frame::DataBlocked { maximum }, type_len + cursor))
}

fn parse_stream_data_blocked(
    body: &[u8],
    type_len: usize,
) -> Result<(Frame<'_>, usize), DecodeError> {
    let mut cursor = 0;
    let stream_id = read_varint(body, &mut cursor)?;
    let maximum = read_varint(body, &mut cursor)?;
    Ok((
        Frame::StreamDataBlocked { stream_id, maximum },
        type_len + cursor,
    ))
}

fn parse_streams_blocked(
    body: &[u8],
    type_len: usize,
    bidi: bool,
) -> Result<(Frame<'_>, usize), DecodeError> {
    let mut cursor = 0;
    let maximum = read_varint(body, &mut cursor)?;
    Ok((Frame::StreamsBlocked { bidi, maximum }, type_len + cursor))
}

fn parse_new_connection_id(
    body: &[u8],
    type_len: usize,
) -> Result<(Frame<'_>, usize), DecodeError> {
    let mut cursor = 0;
    let sequence = read_varint(body, &mut cursor)?;
    let retire_prior_to = read_varint(body, &mut cursor)?;
    let cid_len = body.get(cursor).copied().ok_or(DecodeError::Truncated)?;
    cursor += 1;
    let cid_len_usize = usize::from(cid_len);
    let connection_id = read_slice(body, &mut cursor, cid_len_usize)?;
    let token_slice = read_slice(body, &mut cursor, STATELESS_RESET_TOKEN_LEN)?;
    let stateless_reset_token: &[u8; STATELESS_RESET_TOKEN_LEN] =
        token_slice.try_into().map_err(|_| DecodeError::Truncated)?;
    Ok((
        Frame::NewConnectionId {
            sequence,
            retire_prior_to,
            connection_id,
            stateless_reset_token,
        },
        type_len + cursor,
    ))
}

fn parse_retire_connection_id(
    body: &[u8],
    type_len: usize,
) -> Result<(Frame<'_>, usize), DecodeError> {
    let mut cursor = 0;
    let sequence = read_varint(body, &mut cursor)?;
    Ok((Frame::RetireConnectionId { sequence }, type_len + cursor))
}

fn parse_path_challenge(body: &[u8], type_len: usize) -> Result<(Frame<'_>, usize), DecodeError> {
    let mut cursor = 0;
    let slice = read_slice(body, &mut cursor, PATH_CHALLENGE_LEN)?;
    let mut data = [0u8; PATH_CHALLENGE_LEN];
    data.copy_from_slice(slice);
    Ok((Frame::PathChallenge { data }, type_len + cursor))
}

fn parse_path_response(body: &[u8], type_len: usize) -> Result<(Frame<'_>, usize), DecodeError> {
    let mut cursor = 0;
    let slice = read_slice(body, &mut cursor, PATH_CHALLENGE_LEN)?;
    let mut data = [0u8; PATH_CHALLENGE_LEN];
    data.copy_from_slice(slice);
    Ok((Frame::PathResponse { data }, type_len + cursor))
}

fn parse_connection_close(
    body: &[u8],
    type_len: usize,
    is_transport: bool,
) -> Result<(Frame<'_>, usize), DecodeError> {
    let mut cursor = 0;
    let error_code = read_varint(body, &mut cursor)?;
    let frame_type = if is_transport {
        Some(read_varint(body, &mut cursor)?)
    } else {
        None
    };
    let reason_len = read_varint(body, &mut cursor)?;
    let reason_len = usize::try_from(reason_len).map_err(|_| DecodeError::LengthOverflowsBuffer)?;
    let reason = read_slice(body, &mut cursor, reason_len)?;
    Ok((
        Frame::ConnectionClose {
            error_code,
            frame_type,
            reason,
        },
        type_len + cursor,
    ))
}

fn parse_datagram_with_length(
    body: &[u8],
    type_len: usize,
) -> Result<(Frame<'_>, usize), DecodeError> {
    let mut cursor = 0;
    let length = read_varint(body, &mut cursor)?;
    let length = usize::try_from(length).map_err(|_| DecodeError::LengthOverflowsBuffer)?;
    let data = read_slice(body, &mut cursor, length)?;
    Ok((Frame::Datagram { data }, type_len + cursor))
}

// ---- encoder ---------------------------------------------------------------

impl Frame<'_> {
    /// Encode `self` into `output`. Returns the number of bytes written.
    ///
    /// # Errors
    ///
    /// See [`EncodeError`].
    #[allow(clippy::too_many_lines)]
    pub fn encode(&self, output: &mut [u8]) -> Result<usize, EncodeError> {
        let mut cursor = 0;
        match self {
            Self::Padding { count } => {
                let slot = output
                    .get_mut(..*count)
                    .ok_or(EncodeError::BufferTooSmall)?;
                slot.fill(0x00);
                cursor = *count;
            }
            Self::Ping => {
                write_type(output, &mut cursor, 0x01)?;
            }
            Self::Ack {
                largest,
                delay,
                first_range,
                ranges_raw,
                range_count,
                ecn,
            } => {
                let type_byte: u64 = if ecn.is_some() { 0x03 } else { 0x02 };
                write_type(output, &mut cursor, type_byte)?;
                write_varint(output, &mut cursor, *largest)?;
                write_varint(output, &mut cursor, *delay)?;
                write_varint(output, &mut cursor, *range_count)?;
                write_varint(output, &mut cursor, *first_range)?;
                write_slice(output, &mut cursor, ranges_raw)?;
                if let Some(ecn) = ecn {
                    write_varint(output, &mut cursor, ecn.ect0)?;
                    write_varint(output, &mut cursor, ecn.ect1)?;
                    write_varint(output, &mut cursor, ecn.ecn_ce)?;
                }
            }
            Self::ResetStream {
                stream_id,
                error_code,
                final_size,
            } => {
                write_type(output, &mut cursor, 0x04)?;
                write_varint(output, &mut cursor, *stream_id)?;
                write_varint(output, &mut cursor, *error_code)?;
                write_varint(output, &mut cursor, *final_size)?;
            }
            Self::StopSending {
                stream_id,
                error_code,
            } => {
                write_type(output, &mut cursor, 0x05)?;
                write_varint(output, &mut cursor, *stream_id)?;
                write_varint(output, &mut cursor, *error_code)?;
            }
            Self::Crypto { offset, data } => {
                write_type(output, &mut cursor, 0x06)?;
                write_varint(output, &mut cursor, *offset)?;
                write_varint(output, &mut cursor, data.len() as u64)?;
                write_slice(output, &mut cursor, data)?;
            }
            Self::NewToken { token } => {
                write_type(output, &mut cursor, 0x07)?;
                write_varint(output, &mut cursor, token.len() as u64)?;
                write_slice(output, &mut cursor, token)?;
            }
            Self::Stream {
                stream_id,
                offset,
                data,
                fin,
            } => {
                let mut type_byte: u8 = 0x08;
                if *offset != 0 {
                    type_byte |= 0b100;
                }
                // always emit LEN for unambiguous wire shape; an end-of-packet
                // STREAM frame can omit it (encoder consumer's choice for now).
                type_byte |= 0b010;
                if *fin {
                    type_byte |= 0b001;
                }
                write_type(output, &mut cursor, u64::from(type_byte))?;
                write_varint(output, &mut cursor, *stream_id)?;
                if *offset != 0 {
                    write_varint(output, &mut cursor, *offset)?;
                }
                write_varint(output, &mut cursor, data.len() as u64)?;
                write_slice(output, &mut cursor, data)?;
            }
            Self::MaxData { maximum } => {
                write_type(output, &mut cursor, 0x10)?;
                write_varint(output, &mut cursor, *maximum)?;
            }
            Self::MaxStreamData { stream_id, maximum } => {
                write_type(output, &mut cursor, 0x11)?;
                write_varint(output, &mut cursor, *stream_id)?;
                write_varint(output, &mut cursor, *maximum)?;
            }
            Self::MaxStreams { bidi, maximum } => {
                let type_byte: u64 = if *bidi { 0x12 } else { 0x13 };
                write_type(output, &mut cursor, type_byte)?;
                write_varint(output, &mut cursor, *maximum)?;
            }
            Self::DataBlocked { maximum } => {
                write_type(output, &mut cursor, 0x14)?;
                write_varint(output, &mut cursor, *maximum)?;
            }
            Self::StreamDataBlocked { stream_id, maximum } => {
                write_type(output, &mut cursor, 0x15)?;
                write_varint(output, &mut cursor, *stream_id)?;
                write_varint(output, &mut cursor, *maximum)?;
            }
            Self::StreamsBlocked { bidi, maximum } => {
                let type_byte: u64 = if *bidi { 0x16 } else { 0x17 };
                write_type(output, &mut cursor, type_byte)?;
                write_varint(output, &mut cursor, *maximum)?;
            }
            Self::NewConnectionId {
                sequence,
                retire_prior_to,
                connection_id,
                stateless_reset_token,
            } => {
                write_type(output, &mut cursor, 0x18)?;
                write_varint(output, &mut cursor, *sequence)?;
                write_varint(output, &mut cursor, *retire_prior_to)?;
                let cid_len =
                    u8::try_from(connection_id.len()).map_err(|_| EncodeError::ValueTooLarge)?;
                if cursor >= output.len() {
                    return Err(EncodeError::BufferTooSmall);
                }
                output[cursor] = cid_len;
                cursor += 1;
                write_slice(output, &mut cursor, connection_id)?;
                write_slice(output, &mut cursor, *stateless_reset_token)?;
            }
            Self::RetireConnectionId { sequence } => {
                write_type(output, &mut cursor, 0x19)?;
                write_varint(output, &mut cursor, *sequence)?;
            }
            Self::PathChallenge { data } => {
                write_type(output, &mut cursor, 0x1a)?;
                write_slice(output, &mut cursor, data)?;
            }
            Self::PathResponse { data } => {
                write_type(output, &mut cursor, 0x1b)?;
                write_slice(output, &mut cursor, data)?;
            }
            Self::ConnectionClose {
                error_code,
                frame_type,
                reason,
            } => {
                let type_byte: u64 = if frame_type.is_some() { 0x1c } else { 0x1d };
                write_type(output, &mut cursor, type_byte)?;
                write_varint(output, &mut cursor, *error_code)?;
                if let Some(frame_type) = frame_type {
                    write_varint(output, &mut cursor, *frame_type)?;
                }
                write_varint(output, &mut cursor, reason.len() as u64)?;
                write_slice(output, &mut cursor, reason)?;
            }
            Self::HandshakeDone => {
                write_type(output, &mut cursor, 0x1e)?;
            }
            Self::Datagram { data } => {
                write_type(output, &mut cursor, 0x31)?;
                write_varint(output, &mut cursor, data.len() as u64)?;
                write_slice(output, &mut cursor, data)?;
            }
        }
        Ok(cursor)
    }
}

fn write_type(output: &mut [u8], cursor: &mut usize, frame_type: u64) -> Result<(), EncodeError> {
    write_varint(output, cursor, frame_type)
}

fn write_varint(output: &mut [u8], cursor: &mut usize, value: u64) -> Result<(), EncodeError> {
    let slice = output
        .get_mut(*cursor..)
        .ok_or(EncodeError::BufferTooSmall)?;
    let written = varint::encode(value, slice).map_err(|err| match err {
        varint::EncodeError::ValueTooLarge => EncodeError::ValueTooLarge,
        varint::EncodeError::BufferTooSmall => EncodeError::BufferTooSmall,
    })?;
    *cursor += written;
    Ok(())
}

fn write_slice(output: &mut [u8], cursor: &mut usize, source: &[u8]) -> Result<(), EncodeError> {
    let start = *cursor;
    let end = start
        .checked_add(source.len())
        .ok_or(EncodeError::BufferTooSmall)?;
    let slot = output
        .get_mut(start..end)
        .ok_or(EncodeError::BufferTooSmall)?;
    slot.copy_from_slice(source);
    *cursor = end;
    Ok(())
}

#[cfg(all(test, feature = "quic-alloc"))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn round_trip(frame: Frame<'_>) -> Frame<'static> {
        // Unsafe to reuse storage across lifetime — instead, re-encode into a
        // module-static buffer-equivalent via a vec, then leak via Box::leak.
        // For tests this is fine; the buffer outlives the test.
        let mut buffer = alloc::vec![0u8; 4096];
        let written = frame.encode(&mut buffer).expect("encode");
        buffer.truncate(written);
        let leaked: &'static [u8] = alloc::boxed::Box::leak(buffer.into_boxed_slice());
        let (parsed, consumed) = parse(leaked).expect("parse");
        assert_eq!(consumed, leaked.len(), "frame length mismatch");
        match parsed {
            Frame::Padding { count } => Frame::Padding { count },
            Frame::Ping => Frame::Ping,
            Frame::HandshakeDone => Frame::HandshakeDone,
            Frame::Ack {
                largest,
                delay,
                first_range,
                ranges_raw,
                range_count,
                ecn,
            } => Frame::Ack {
                largest,
                delay,
                first_range,
                ranges_raw,
                range_count,
                ecn,
            },
            Frame::Crypto { offset, data } => Frame::Crypto { offset, data },
            Frame::Stream {
                stream_id,
                offset,
                data,
                fin,
            } => Frame::Stream {
                stream_id,
                offset,
                data,
                fin,
            },
            Frame::NewToken { token } => Frame::NewToken { token },
            Frame::MaxData { maximum } => Frame::MaxData { maximum },
            Frame::MaxStreamData { stream_id, maximum } => {
                Frame::MaxStreamData { stream_id, maximum }
            }
            Frame::MaxStreams { bidi, maximum } => Frame::MaxStreams { bidi, maximum },
            Frame::DataBlocked { maximum } => Frame::DataBlocked { maximum },
            Frame::StreamDataBlocked { stream_id, maximum } => {
                Frame::StreamDataBlocked { stream_id, maximum }
            }
            Frame::StreamsBlocked { bidi, maximum } => Frame::StreamsBlocked { bidi, maximum },
            Frame::ResetStream {
                stream_id,
                error_code,
                final_size,
            } => Frame::ResetStream {
                stream_id,
                error_code,
                final_size,
            },
            Frame::StopSending {
                stream_id,
                error_code,
            } => Frame::StopSending {
                stream_id,
                error_code,
            },
            Frame::NewConnectionId {
                sequence,
                retire_prior_to,
                connection_id,
                stateless_reset_token,
            } => Frame::NewConnectionId {
                sequence,
                retire_prior_to,
                connection_id,
                stateless_reset_token,
            },
            Frame::RetireConnectionId { sequence } => Frame::RetireConnectionId { sequence },
            Frame::PathChallenge { data } => Frame::PathChallenge { data },
            Frame::PathResponse { data } => Frame::PathResponse { data },
            Frame::ConnectionClose {
                error_code,
                frame_type,
                reason,
            } => Frame::ConnectionClose {
                error_code,
                frame_type,
                reason,
            },
            Frame::Datagram { data } => Frame::Datagram { data },
        }
    }

    #[test]
    fn padding_run_coalesces() {
        let bytes = [0x00u8; 100];
        let (frame, consumed) = parse(&bytes).expect("parse");
        assert!(matches!(frame, Frame::Padding { count } if count == 100));
        assert_eq!(consumed, 100);
    }

    #[test]
    fn padding_followed_by_other_frame() {
        // 10 padding bytes followed by a PING (0x01)
        let mut bytes = [0x00u8; 11];
        bytes[10] = 0x01;
        let (frame, consumed) = parse(&bytes).expect("parse");
        assert!(matches!(frame, Frame::Padding { count } if count == 10));
        assert_eq!(consumed, 10);
        let (next, consumed_next) = parse(&bytes[10..]).expect("parse PING");
        assert_eq!(next, Frame::Ping);
        assert_eq!(consumed_next, 1);
    }

    #[test]
    fn ping_round_trip() {
        let frame = Frame::Ping;
        let result = round_trip(frame);
        assert_eq!(result, Frame::Ping);
    }

    #[test]
    fn handshake_done_round_trip() {
        let frame = Frame::HandshakeDone;
        let result = round_trip(frame);
        assert_eq!(result, Frame::HandshakeDone);
    }

    #[test]
    fn ack_without_ecn_round_trip() {
        // largest=10, delay=20, first_range=2, then 2 (gap, length) pairs:
        // (gap=0, length=1), (gap=1, length=2)
        let mut ranges_raw = alloc::vec![0u8; 16];
        let mut cursor = 0;
        cursor += varint::encode(0, &mut ranges_raw[cursor..]).unwrap();
        cursor += varint::encode(1, &mut ranges_raw[cursor..]).unwrap();
        cursor += varint::encode(1, &mut ranges_raw[cursor..]).unwrap();
        cursor += varint::encode(2, &mut ranges_raw[cursor..]).unwrap();
        ranges_raw.truncate(cursor);

        let frame = Frame::Ack {
            largest: 10,
            delay: 20,
            first_range: 2,
            ranges_raw: &ranges_raw,
            range_count: 2,
            ecn: None,
        };
        let mut buffer = alloc::vec![0u8; 64];
        let written = frame.encode(&mut buffer).expect("encode");
        let (parsed, consumed) = parse(&buffer[..written]).expect("parse");
        assert_eq!(consumed, written);
        match parsed {
            Frame::Ack {
                largest,
                delay,
                first_range,
                range_count,
                ecn,
                ..
            } => {
                assert_eq!(largest, 10);
                assert_eq!(delay, 20);
                assert_eq!(first_range, 2);
                assert_eq!(range_count, 2);
                assert_eq!(ecn, None);
            }
            other => panic!("expected Ack, got {other:?}"),
        }
    }

    #[test]
    fn ack_with_ecn_round_trip() {
        let frame = Frame::Ack {
            largest: 5,
            delay: 7,
            first_range: 5,
            ranges_raw: &[],
            range_count: 0,
            ecn: Some(EcnCounts {
                ect0: 1,
                ect1: 2,
                ecn_ce: 3,
            }),
        };
        let mut buffer = alloc::vec![0u8; 32];
        let written = frame.encode(&mut buffer).expect("encode");
        let (parsed, _consumed) = parse(&buffer[..written]).expect("parse");
        if let Frame::Ack { ecn, .. } = parsed {
            assert_eq!(
                ecn,
                Some(EcnCounts {
                    ect0: 1,
                    ect1: 2,
                    ecn_ce: 3,
                })
            );
        } else {
            panic!("expected Ack");
        }
    }

    #[test]
    fn stream_round_trip_all_flag_combinations() {
        let payload: [u8; 4] = [0xa, 0xb, 0xc, 0xd];
        for fin in [false, true] {
            for offset in [0u64, 7, 1024] {
                let frame = Frame::Stream {
                    stream_id: 4,
                    offset,
                    data: &payload,
                    fin,
                };
                let result = round_trip(frame);
                match result {
                    Frame::Stream {
                        stream_id,
                        offset: o,
                        data,
                        fin: f,
                    } => {
                        assert_eq!(stream_id, 4);
                        assert_eq!(o, offset);
                        assert_eq!(data, &payload);
                        assert_eq!(f, fin);
                    }
                    _ => panic!("expected Stream"),
                }
            }
        }
    }

    #[test]
    fn crypto_round_trip() {
        let data = alloc::vec![0u8; 32];
        let frame = Frame::Crypto {
            offset: 16,
            data: &data,
        };
        let result = round_trip(frame);
        assert!(matches!(
            result,
            Frame::Crypto { offset: 16, data: d } if d == &data[..]
        ));
    }

    #[test]
    fn new_token_round_trip() {
        let token = alloc::vec![0xab; 24];
        let frame = Frame::NewToken { token: &token };
        let result = round_trip(frame);
        assert!(matches!(result, Frame::NewToken { token: t } if t == &token[..]));
    }

    #[test]
    fn flow_control_frames_round_trip() {
        for frame in [
            Frame::MaxData { maximum: 1024 },
            Frame::MaxStreamData {
                stream_id: 4,
                maximum: 8192,
            },
            Frame::MaxStreams {
                bidi: true,
                maximum: 100,
            },
            Frame::MaxStreams {
                bidi: false,
                maximum: 50,
            },
            Frame::DataBlocked { maximum: 256 },
            Frame::StreamDataBlocked {
                stream_id: 8,
                maximum: 4096,
            },
            Frame::StreamsBlocked {
                bidi: true,
                maximum: 10,
            },
            Frame::StreamsBlocked {
                bidi: false,
                maximum: 20,
            },
        ] {
            let result = round_trip(frame);
            assert_eq!(result, frame);
        }
    }

    #[test]
    fn reset_and_stop_round_trip() {
        let reset = Frame::ResetStream {
            stream_id: 4,
            error_code: 7,
            final_size: 1024,
        };
        assert_eq!(round_trip(reset), reset);
        let stop = Frame::StopSending {
            stream_id: 8,
            error_code: 3,
        };
        assert_eq!(round_trip(stop), stop);
    }

    #[test]
    fn new_connection_id_round_trip() {
        let cid: [u8; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
        let token: [u8; STATELESS_RESET_TOKEN_LEN] = [0xab; STATELESS_RESET_TOKEN_LEN];
        let frame = Frame::NewConnectionId {
            sequence: 3,
            retire_prior_to: 1,
            connection_id: &cid,
            stateless_reset_token: &token,
        };
        let result = round_trip(frame);
        match result {
            Frame::NewConnectionId {
                sequence,
                retire_prior_to,
                connection_id,
                stateless_reset_token,
            } => {
                assert_eq!(sequence, 3);
                assert_eq!(retire_prior_to, 1);
                assert_eq!(connection_id, &cid);
                assert_eq!(stateless_reset_token, &token);
            }
            _ => panic!("expected NewConnectionId"),
        }
    }

    #[test]
    fn retire_connection_id_round_trip() {
        let frame = Frame::RetireConnectionId { sequence: 5 };
        assert_eq!(round_trip(frame), frame);
    }

    #[test]
    fn path_challenge_and_response_round_trip() {
        let data: [u8; PATH_CHALLENGE_LEN] = [0xde, 0xad, 0xbe, 0xef, 0xca, 0xfe, 0xba, 0xbe];
        let challenge = Frame::PathChallenge { data };
        assert_eq!(round_trip(challenge), challenge);
        let response = Frame::PathResponse { data };
        assert_eq!(round_trip(response), response);
    }

    #[test]
    fn connection_close_transport_round_trip() {
        let reason = b"transport error here";
        let frame = Frame::ConnectionClose {
            error_code: 7,
            frame_type: Some(0x06), // CRYPTO triggered the close
            reason,
        };
        let result = round_trip(frame);
        match result {
            Frame::ConnectionClose {
                error_code,
                frame_type,
                reason: r,
            } => {
                assert_eq!(error_code, 7);
                assert_eq!(frame_type, Some(0x06));
                assert_eq!(r, reason);
            }
            _ => panic!("expected ConnectionClose"),
        }
    }

    #[test]
    fn connection_close_application_round_trip() {
        let reason = b"application close";
        let frame = Frame::ConnectionClose {
            error_code: 0,
            frame_type: None,
            reason,
        };
        let result = round_trip(frame);
        match result {
            Frame::ConnectionClose {
                frame_type: None,
                reason: r,
                ..
            } => {
                assert_eq!(r, reason);
            }
            _ => panic!("expected app ConnectionClose"),
        }
    }

    #[test]
    fn datagram_with_length_round_trip() {
        let payload = alloc::vec![0xab; 200];
        let frame = Frame::Datagram { data: &payload };
        let result = round_trip(frame);
        assert!(matches!(result, Frame::Datagram { data: d } if d == &payload[..]));
    }

    #[test]
    fn datagram_implicit_length_parse() {
        // 0x30 = DATAGRAM with implicit length
        let mut bytes = alloc::vec![0x30u8];
        bytes.extend_from_slice(&[0xcd; 50]);
        let (frame, consumed) = parse(&bytes).expect("parse");
        assert!(matches!(frame, Frame::Datagram { data } if data == &[0xcd; 50][..]));
        assert_eq!(consumed, bytes.len());
    }

    #[test]
    fn unknown_frame_type_rejected() {
        // 0x40 = 2-byte varint with value 0; choose 0x4040 (large) — not a known type
        let bytes = [0x4f, 0xff];
        let err = parse(&bytes).unwrap_err();
        assert!(matches!(err, DecodeError::UnknownFrameType(_)));
    }

    #[test]
    fn truncated_crypto_rejected() {
        // CRYPTO type byte + offset varint + length varint that claims 5 bytes
        // but only 2 bytes of payload provided
        let mut bytes = alloc::vec![0x06u8, 0x00, 0x05, 0x01, 0x02];
        let err = parse(&bytes).unwrap_err();
        assert!(matches!(err, DecodeError::Truncated));
        bytes.clear();
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(parse(&[]), Err(DecodeError::Empty));
    }

    #[test]
    fn ack_ranges_iterates() {
        // build an ACK with 3 (gap, length) pairs and verify iteration
        let pairs: [(u64, u64); 3] = [(0, 1), (3, 7), (15, 31)];
        let mut ranges_raw = alloc::vec![0u8; 32];
        let mut cursor = 0;
        for (gap, length) in pairs {
            cursor += varint::encode(gap, &mut ranges_raw[cursor..]).unwrap();
            cursor += varint::encode(length, &mut ranges_raw[cursor..]).unwrap();
        }
        ranges_raw.truncate(cursor);
        let iter = AckRanges::new(&ranges_raw, 3);
        let collected: alloc::vec::Vec<(u64, u64)> = iter.map(|res| res.unwrap()).collect();
        assert_eq!(collected, pairs);
    }
}
