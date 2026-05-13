//! Multipath QUIC wire-format frames per [draft-ietf-quic-multipath-21] §4.
//!
//! Ships the codec for the 7 new frame types added by the multipath
//! extension:
//!
//! - `PATH_ABANDON` (type 0x3e75) — §4.2
//! - `PATH_STATUS_BACKUP` (type 0x3e76) — §4.3
//! - `PATH_STATUS_AVAILABLE` (type 0x3e77) — §4.3
//! - `PATH_NEW_CONNECTION_ID` (type 0x3e78) — §4.4
//! - `PATH_RETIRE_CONNECTION_ID` (type 0x3e79) — §4.5
//! - `MAX_PATH_ID` (type 0x3e7a) — §4.6
//! - `PATHS_BLOCKED` (type 0x3e7b) — §4.7
//! - `PATH_CIDS_BLOCKED` (type 0x3e7c) — §4.7
//! - `PATH_ACK` (type 0x15228c00) / `PATH_ACK_ECN` (0x15228c01) — §4.1.
//!   Same structure as RFC 9000 ACK / ACK_ECN with a Path Identifier
//!   prefix. Parser surfaces ranges as a borrowed slice; the per-path
//!   AckScheduler wires the ranges into loss detection when a
//!   non-zero path has live in-flight packets.
//!
//! [draft-ietf-quic-multipath-21]: https://www.ietf.org/archive/id/draft-ietf-quic-multipath-21.txt
//!
//! # Tier
//!
//! Tier-3 (bare `no_std + no_alloc`). All parse paths return
//! borrowed views; encode paths write into caller-owned `&mut [u8]`.

use crate::quic::packet::header::MAX_CID_LEN;
use crate::quic::varint;

/// Wire-format frame type IDs per draft §4.
pub const FRAME_TYPE_PATH_ABANDON: u64 = 0x3e75;
pub const FRAME_TYPE_PATH_STATUS_BACKUP: u64 = 0x3e76;
pub const FRAME_TYPE_PATH_STATUS_AVAILABLE: u64 = 0x3e77;
pub const FRAME_TYPE_PATH_NEW_CONNECTION_ID: u64 = 0x3e78;
pub const FRAME_TYPE_PATH_RETIRE_CONNECTION_ID: u64 = 0x3e79;
pub const FRAME_TYPE_MAX_PATH_ID: u64 = 0x3e7a;
pub const FRAME_TYPE_PATHS_BLOCKED: u64 = 0x3e7b;
pub const FRAME_TYPE_PATH_CIDS_BLOCKED: u64 = 0x3e7c;
pub const FRAME_TYPE_PATH_ACK: u64 = 0x1528_c000;
pub const FRAME_TYPE_PATH_ACK_ECN: u64 = 0x1528_c001;

/// Stateless reset token length per RFC 9000 §10.3.
pub const STATELESS_RESET_TOKEN_LEN: usize = 16;

/// Errors from multipath frame parsing/encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum FrameError {
    /// Input ended before the frame body completed.
    Truncated,
    /// Varint decoded out-of-range or invalid encoding.
    InvalidVarint,
    /// CID length byte was out of bounds (< 1 or > 20 per RFC 9000 §5.1.1).
    InvalidCidLen,
    /// Caller-provided output buffer too small for the encoded frame.
    BufferTooSmall { needed: usize },
}

impl From<varint::DecodeError> for FrameError {
    fn from(_: varint::DecodeError) -> Self {
        Self::InvalidVarint
    }
}

impl From<varint::EncodeError> for FrameError {
    fn from(_: varint::EncodeError) -> Self {
        Self::BufferTooSmall { needed: 0 }
    }
}

/// Multipath wire-format frame.
///
/// Frame Type prefix is NOT included in the variant — the parse / encode
/// helpers handle it. Borrowed slices reference into the caller's
/// datagram buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum MultipathFrame<'a> {
    /// §4.2 — inform peer to abandon the named path.
    PathAbandon { path_id: u64, error_code: u64 },
    /// §4.3 — peer prefers this path NOT be used for sending.
    PathStatusBackup { path_id: u64, status_seq: u64 },
    /// §4.3 — peer signals path is freely available for sending.
    PathStatusAvailable { path_id: u64, status_seq: u64 },
    /// §4.4 — issue a path-specific CID + stateless-reset token.
    PathNewConnectionId {
        path_id: u64,
        sequence_number: u64,
        retire_prior_to: u64,
        connection_id: &'a [u8],
        stateless_reset_token: &'a [u8; STATELESS_RESET_TOKEN_LEN],
    },
    /// §4.5 — retire a path-specific CID by sequence number.
    PathRetireConnectionId { path_id: u64, sequence_number: u64 },
    /// §4.6 — advertise the maximum path ID we accept.
    MaxPathId { maximum_path_identifier: u64 },
    /// §4.7 — sender wants to open a new path but is at the peer's limit.
    PathsBlocked { maximum_path_identifier: u64 },
    /// §4.7 — sender wants a new CID for the named path but has none.
    PathCidsBlocked {
        path_id: u64,
        next_sequence_number: u64,
    },
    /// §4.1 — per-path generalization of RFC 9000 ACK / ACK_ECN.
    /// `ranges` is the borrowed slice of (largest, ack_delay,
    /// range_count, first_range, ack_ranges, [ecn_counts]) bytes —
    /// the same on-wire form as an RFC 9000 ACK frame following the
    /// type byte. Parser keeps it as a borrowed slice so the
    /// per-path AckScheduler can re-parse with its own range-set.
    PathAck {
        path_id: u64,
        with_ecn: bool,
        ranges: &'a [u8],
    },
}

/// Decode a multipath frame from `input`. Reads the frame-type varint
/// then dispatches to the per-frame parser. Returns the parsed frame
/// + total bytes consumed.
///
/// # Errors
///
/// See [`FrameError`].
pub fn parse(input: &[u8]) -> Result<(MultipathFrame<'_>, usize), FrameError> {
    let (frame_type, type_bytes) = varint::decode(input)?;
    let body = &input[type_bytes..];
    let mut cursor = 0usize;
    let frame = match frame_type {
        FRAME_TYPE_PATH_ABANDON => {
            let path_id = take_varint(body, &mut cursor)?;
            let error_code = take_varint(body, &mut cursor)?;
            MultipathFrame::PathAbandon {
                path_id,
                error_code,
            }
        }
        FRAME_TYPE_PATH_STATUS_BACKUP => {
            let path_id = take_varint(body, &mut cursor)?;
            let status_seq = take_varint(body, &mut cursor)?;
            MultipathFrame::PathStatusBackup {
                path_id,
                status_seq,
            }
        }
        FRAME_TYPE_PATH_STATUS_AVAILABLE => {
            let path_id = take_varint(body, &mut cursor)?;
            let status_seq = take_varint(body, &mut cursor)?;
            MultipathFrame::PathStatusAvailable {
                path_id,
                status_seq,
            }
        }
        FRAME_TYPE_PATH_NEW_CONNECTION_ID => {
            let path_id = take_varint(body, &mut cursor)?;
            let sequence_number = take_varint(body, &mut cursor)?;
            let retire_prior_to = take_varint(body, &mut cursor)?;
            if cursor >= body.len() {
                return Err(FrameError::Truncated);
            }
            let cid_len = body[cursor] as usize;
            cursor += 1;
            if !(1..=MAX_CID_LEN).contains(&cid_len) {
                return Err(FrameError::InvalidCidLen);
            }
            if body.len() < cursor + cid_len + STATELESS_RESET_TOKEN_LEN {
                return Err(FrameError::Truncated);
            }
            let connection_id = &body[cursor..cursor + cid_len];
            cursor += cid_len;
            let token_slice: &[u8; STATELESS_RESET_TOKEN_LEN] = body
                [cursor..cursor + STATELESS_RESET_TOKEN_LEN]
                .try_into()
                .map_err(|_| FrameError::Truncated)?;
            cursor += STATELESS_RESET_TOKEN_LEN;
            MultipathFrame::PathNewConnectionId {
                path_id,
                sequence_number,
                retire_prior_to,
                connection_id,
                stateless_reset_token: token_slice,
            }
        }
        FRAME_TYPE_PATH_RETIRE_CONNECTION_ID => {
            let path_id = take_varint(body, &mut cursor)?;
            let sequence_number = take_varint(body, &mut cursor)?;
            MultipathFrame::PathRetireConnectionId {
                path_id,
                sequence_number,
            }
        }
        FRAME_TYPE_MAX_PATH_ID => {
            let maximum_path_identifier = take_varint(body, &mut cursor)?;
            MultipathFrame::MaxPathId {
                maximum_path_identifier,
            }
        }
        FRAME_TYPE_PATHS_BLOCKED => {
            let maximum_path_identifier = take_varint(body, &mut cursor)?;
            MultipathFrame::PathsBlocked {
                maximum_path_identifier,
            }
        }
        FRAME_TYPE_PATH_CIDS_BLOCKED => {
            let path_id = take_varint(body, &mut cursor)?;
            let next_sequence_number = take_varint(body, &mut cursor)?;
            MultipathFrame::PathCidsBlocked {
                path_id,
                next_sequence_number,
            }
        }
        FRAME_TYPE_PATH_ACK | FRAME_TYPE_PATH_ACK_ECN => {
            let with_ecn = frame_type == FRAME_TYPE_PATH_ACK_ECN;
            let path_id = take_varint(body, &mut cursor)?;
            let ack_body_start = cursor;
            // Per draft §4.1 the remaining fields are exactly an
            // RFC 9000 ACK / ACK_ECN body (largest, delay, range_count,
            // first_range, ranges, [ecn_counts]). Parse just enough
            // to know where the frame ends so we can surface the
            // ack-body byte slice for the per-path AckScheduler to
            // re-decode with its own range-set type.
            let _largest = take_varint(body, &mut cursor)?;
            let _delay = take_varint(body, &mut cursor)?;
            let range_count = take_varint(body, &mut cursor)?;
            let _first = take_varint(body, &mut cursor)?;
            for _ in 0..range_count {
                let _gap = take_varint(body, &mut cursor)?;
                let _len = take_varint(body, &mut cursor)?;
            }
            if with_ecn {
                let _ect0 = take_varint(body, &mut cursor)?;
                let _ect1 = take_varint(body, &mut cursor)?;
                let _ce = take_varint(body, &mut cursor)?;
            }
            let ranges = &body[ack_body_start..cursor];
            MultipathFrame::PathAck {
                path_id,
                with_ecn,
                ranges,
            }
        }
        _ => return Err(FrameError::InvalidVarint),
    };
    Ok((frame, type_bytes + cursor))
}

/// Encode a multipath frame into `output`. Returns the number of bytes
/// written. Writes the frame-type varint prefix + the per-frame body.
///
/// # Errors
///
/// Returns [`FrameError::BufferTooSmall`] when `output` is too short.
pub fn encode(frame: &MultipathFrame<'_>, output: &mut [u8]) -> Result<usize, FrameError> {
    match frame {
        MultipathFrame::PathAbandon {
            path_id,
            error_code,
        } => write_frame_with_two_varints(FRAME_TYPE_PATH_ABANDON, *path_id, *error_code, output),
        MultipathFrame::PathStatusBackup {
            path_id,
            status_seq,
        } => write_frame_with_two_varints(
            FRAME_TYPE_PATH_STATUS_BACKUP,
            *path_id,
            *status_seq,
            output,
        ),
        MultipathFrame::PathStatusAvailable {
            path_id,
            status_seq,
        } => write_frame_with_two_varints(
            FRAME_TYPE_PATH_STATUS_AVAILABLE,
            *path_id,
            *status_seq,
            output,
        ),
        MultipathFrame::PathNewConnectionId {
            path_id,
            sequence_number,
            retire_prior_to,
            connection_id,
            stateless_reset_token,
        } => encode_path_new_connection_id(
            *path_id,
            *sequence_number,
            *retire_prior_to,
            connection_id,
            stateless_reset_token,
            output,
        ),
        MultipathFrame::PathRetireConnectionId {
            path_id,
            sequence_number,
        } => write_frame_with_two_varints(
            FRAME_TYPE_PATH_RETIRE_CONNECTION_ID,
            *path_id,
            *sequence_number,
            output,
        ),
        MultipathFrame::MaxPathId {
            maximum_path_identifier,
        } => write_frame_with_one_varint(FRAME_TYPE_MAX_PATH_ID, *maximum_path_identifier, output),
        MultipathFrame::PathsBlocked {
            maximum_path_identifier,
        } => {
            write_frame_with_one_varint(FRAME_TYPE_PATHS_BLOCKED, *maximum_path_identifier, output)
        }
        MultipathFrame::PathCidsBlocked {
            path_id,
            next_sequence_number,
        } => write_frame_with_two_varints(
            FRAME_TYPE_PATH_CIDS_BLOCKED,
            *path_id,
            *next_sequence_number,
            output,
        ),
        MultipathFrame::PathAck {
            path_id,
            with_ecn,
            ranges,
        } => {
            let frame_type = if *with_ecn {
                FRAME_TYPE_PATH_ACK_ECN
            } else {
                FRAME_TYPE_PATH_ACK
            };
            let needed =
                varint::encoded_len(frame_type) + varint::encoded_len(*path_id) + ranges.len();
            if output.len() < needed {
                return Err(FrameError::BufferTooSmall { needed });
            }
            let mut cursor = varint::encode(frame_type, output)?;
            cursor += varint::encode(*path_id, &mut output[cursor..])?;
            output[cursor..cursor + ranges.len()].copy_from_slice(ranges);
            cursor += ranges.len();
            Ok(cursor)
        }
    }
}

fn take_varint(buf: &[u8], cursor: &mut usize) -> Result<u64, FrameError> {
    let (value, consumed) = varint::decode(&buf[*cursor..])?;
    *cursor += consumed;
    Ok(value)
}

fn write_frame_with_one_varint(
    frame_type: u64,
    value: u64,
    output: &mut [u8],
) -> Result<usize, FrameError> {
    let needed = varint::encoded_len(frame_type) + varint::encoded_len(value);
    if output.len() < needed {
        return Err(FrameError::BufferTooSmall { needed });
    }
    let mut cursor = varint::encode(frame_type, output)?;
    cursor += varint::encode(value, &mut output[cursor..])?;
    Ok(cursor)
}

fn write_frame_with_two_varints(
    frame_type: u64,
    a: u64,
    b: u64,
    output: &mut [u8],
) -> Result<usize, FrameError> {
    let needed = varint::encoded_len(frame_type) + varint::encoded_len(a) + varint::encoded_len(b);
    if output.len() < needed {
        return Err(FrameError::BufferTooSmall { needed });
    }
    let mut cursor = varint::encode(frame_type, output)?;
    cursor += varint::encode(a, &mut output[cursor..])?;
    cursor += varint::encode(b, &mut output[cursor..])?;
    Ok(cursor)
}

fn encode_path_new_connection_id(
    path_id: u64,
    sequence_number: u64,
    retire_prior_to: u64,
    connection_id: &[u8],
    stateless_reset_token: &[u8; STATELESS_RESET_TOKEN_LEN],
    output: &mut [u8],
) -> Result<usize, FrameError> {
    if connection_id.is_empty() || connection_id.len() > MAX_CID_LEN {
        return Err(FrameError::InvalidCidLen);
    }
    let needed = varint::encoded_len(FRAME_TYPE_PATH_NEW_CONNECTION_ID)
        + varint::encoded_len(path_id)
        + varint::encoded_len(sequence_number)
        + varint::encoded_len(retire_prior_to)
        + 1
        + connection_id.len()
        + STATELESS_RESET_TOKEN_LEN;
    if output.len() < needed {
        return Err(FrameError::BufferTooSmall { needed });
    }
    let mut cursor = varint::encode(FRAME_TYPE_PATH_NEW_CONNECTION_ID, output)?;
    cursor += varint::encode(path_id, &mut output[cursor..])?;
    cursor += varint::encode(sequence_number, &mut output[cursor..])?;
    cursor += varint::encode(retire_prior_to, &mut output[cursor..])?;
    // length byte must fit u8 (MAX_CID_LEN = 20).
    #[allow(clippy::cast_possible_truncation)]
    {
        output[cursor] = connection_id.len() as u8;
    }
    cursor += 1;
    output[cursor..cursor + connection_id.len()].copy_from_slice(connection_id);
    cursor += connection_id.len();
    output[cursor..cursor + STATELESS_RESET_TOKEN_LEN].copy_from_slice(stateless_reset_token);
    cursor += STATELESS_RESET_TOKEN_LEN;
    Ok(cursor)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn roundtrip(frame: MultipathFrame<'_>) -> (alloc::vec::Vec<u8>, usize) {
        let mut buf = alloc::vec![0u8; 256];
        let written = encode(&frame, &mut buf).expect("encode");
        buf.truncate(written);
        let (parsed, consumed) = parse(&buf).expect("parse");
        assert_eq!(
            consumed,
            buf.len(),
            "parse must consume exactly encoded bytes"
        );
        // PartialEq enforces field-by-field structural parity including
        // borrowed connection_id + stateless_reset_token byte arrays.
        assert_eq!(parsed, frame, "roundtrip must preserve frame identically");
        (buf, written)
    }

    extern crate alloc;

    #[test]
    fn path_abandon_roundtrip() {
        let frame = MultipathFrame::PathAbandon {
            path_id: 1,
            error_code: 0x3e, // APPLICATION_ABANDON_PATH per §4.2.1
        };
        roundtrip(frame);
    }

    #[test]
    fn path_status_backup_roundtrip() {
        let frame = MultipathFrame::PathStatusBackup {
            path_id: 2,
            status_seq: 7,
        };
        roundtrip(frame);
    }

    #[test]
    fn path_status_available_roundtrip() {
        let frame = MultipathFrame::PathStatusAvailable {
            path_id: 3,
            status_seq: 11,
        };
        roundtrip(frame);
    }

    #[test]
    fn path_new_connection_id_roundtrip() {
        let cid = [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x00, 0x11];
        let token = [0x42; STATELESS_RESET_TOKEN_LEN];
        let frame = MultipathFrame::PathNewConnectionId {
            path_id: 1,
            sequence_number: 5,
            retire_prior_to: 2,
            connection_id: &cid,
            stateless_reset_token: &token,
        };
        roundtrip(frame);
    }

    #[test]
    fn path_retire_connection_id_roundtrip() {
        let frame = MultipathFrame::PathRetireConnectionId {
            path_id: 1,
            sequence_number: 7,
        };
        roundtrip(frame);
    }

    #[test]
    fn max_path_id_roundtrip() {
        let frame = MultipathFrame::MaxPathId {
            maximum_path_identifier: 4,
        };
        roundtrip(frame);
    }

    #[test]
    fn paths_blocked_roundtrip() {
        let frame = MultipathFrame::PathsBlocked {
            maximum_path_identifier: 4,
        };
        roundtrip(frame);
    }

    #[test]
    fn path_cids_blocked_roundtrip() {
        let frame = MultipathFrame::PathCidsBlocked {
            path_id: 3,
            next_sequence_number: 10,
        };
        roundtrip(frame);
    }

    #[test]
    fn path_ack_roundtrip() {
        // ranges = RFC 9000 ACK body: largest(0x0a) + delay(0x00)
        // + range_count(0x00) + first_range(0x00).
        let ranges: &[u8] = &[0x0a, 0x00, 0x00, 0x00];
        let frame = MultipathFrame::PathAck {
            path_id: 2,
            with_ecn: false,
            ranges,
        };
        let (buf, written) = roundtrip(frame);
        let (parsed, consumed) = parse(&buf).expect("re-parse");
        assert_eq!(consumed, written);
        match parsed {
            MultipathFrame::PathAck {
                path_id,
                with_ecn,
                ranges: parsed_ranges,
            } => {
                assert_eq!(path_id, 2);
                assert!(!with_ecn);
                assert_eq!(parsed_ranges, ranges);
            }
            other => panic!("expected PathAck, got {other:?}"),
        }
    }

    #[test]
    fn path_ack_ecn_roundtrip() {
        // ranges = largest(0x10) + delay(0x00) + range_count(0x00)
        // + first(0x00) + ect0(0x01) + ect1(0x00) + ce(0x00)
        let ranges: &[u8] = &[0x10, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00];
        let frame = MultipathFrame::PathAck {
            path_id: 7,
            with_ecn: true,
            ranges,
        };
        let (buf, _) = roundtrip(frame);
        let (parsed, _) = parse(&buf).expect("re-parse");
        match parsed {
            MultipathFrame::PathAck {
                path_id,
                with_ecn,
                ranges: parsed_ranges,
            } => {
                assert_eq!(path_id, 7);
                assert!(with_ecn);
                assert_eq!(parsed_ranges, ranges);
            }
            other => panic!("expected PathAck, got {other:?}"),
        }
    }

    #[test]
    fn parse_rejects_unknown_frame_type() {
        let mut buf = alloc::vec![0u8; 8];
        // Frame type 0x40 is RFC 9000's STREAM base — NOT a multipath
        // frame type. The multipath parser must reject it.
        let written = varint::encode(0x40, &mut buf).expect("encode");
        buf.truncate(written);
        assert_eq!(parse(&buf), Err(FrameError::InvalidVarint));
    }

    #[test]
    fn parse_truncated_path_abandon_rejected() {
        let mut buf = alloc::vec![0u8; 8];
        let written = varint::encode(FRAME_TYPE_PATH_ABANDON, &mut buf).expect("encode");
        buf.truncate(written);
        // Frame type only, no path_id / error_code → Truncated.
        assert_eq!(parse(&buf), Err(FrameError::InvalidVarint));
    }

    #[test]
    fn parse_invalid_cid_len_zero_rejected() {
        // PATH_NEW_CONNECTION_ID with CID length = 0 is invalid per
        // the frame's CID length field (0 means "no CID" which doesn't
        // make sense in a NEW_CONNECTION_ID context).
        let mut buf = alloc::vec![0u8; 64];
        let mut cursor = varint::encode(FRAME_TYPE_PATH_NEW_CONNECTION_ID, &mut buf).unwrap();
        cursor += varint::encode(1, &mut buf[cursor..]).unwrap(); // path_id
        cursor += varint::encode(5, &mut buf[cursor..]).unwrap(); // sequence_number
        cursor += varint::encode(2, &mut buf[cursor..]).unwrap(); // retire_prior_to
        buf[cursor] = 0; // CID length = 0 (invalid)
        cursor += 1;
        // Add 16 bytes of stateless-reset-token padding so length-
        // checking passes the boundary check but the cid_len = 0 check
        // fires.
        for _ in 0..STATELESS_RESET_TOKEN_LEN {
            buf[cursor] = 0;
            cursor += 1;
        }
        buf.truncate(cursor);
        assert_eq!(parse(&buf), Err(FrameError::InvalidCidLen));
    }

    #[test]
    fn encode_rejects_empty_cid_in_path_new_connection_id() {
        let token = [0x00; STATELESS_RESET_TOKEN_LEN];
        let mut buf = alloc::vec![0u8; 64];
        let frame = MultipathFrame::PathNewConnectionId {
            path_id: 1,
            sequence_number: 1,
            retire_prior_to: 0,
            connection_id: &[],
            stateless_reset_token: &token,
        };
        assert_eq!(encode(&frame, &mut buf), Err(FrameError::InvalidCidLen));
    }

    #[test]
    fn encode_rejects_buffer_too_small() {
        let frame = MultipathFrame::PathAbandon {
            path_id: 1,
            error_code: 0,
        };
        let mut buf = alloc::vec![0u8; 1]; // too small
        assert!(matches!(
            encode(&frame, &mut buf),
            Err(FrameError::BufferTooSmall { .. })
        ));
    }

    #[test]
    fn worked_example_path_abandon_wire_bytes() {
        // Spot-check the on-wire layout for PATH_ABANDON with concrete
        // values, per RFC 9000 §16 varint encoding:
        //
        //   frame_type = 0x3e75 (16021 < 16384 → 2-byte varint):
        //     high byte = 0x40 | (0x3e & 0x3f) = 0x7e
        //     low byte  = 0x75
        //   path_id    = 0       (1-byte varint: 0x00)
        //   error_code = 0x3e    (1-byte varint: 0x3e)
        let frame = MultipathFrame::PathAbandon {
            path_id: 0,
            error_code: 0x3e,
        };
        let mut buf = alloc::vec![0u8; 16];
        let written = encode(&frame, &mut buf).unwrap();
        assert_eq!(written, 2 + 1 + 1);
        assert_eq!(&buf[..written], &[0x7e, 0x75, 0x00, 0x3e]);
    }
}
