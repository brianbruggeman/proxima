//! RFC 9000 §18 Transport Parameters + extension parameters.
//!
//! Transport parameters are negotiated as a TLS extension during the
//! handshake (RFC 9001 §8.2). On the wire each parameter is encoded as
//! `(id: varint) | (length: varint) | (value: length bytes)`. This module
//! parses the entire parameter set into a [`TransportParameters`] struct
//! and re-encodes it back to bytes.
//!
//! # Tier
//!
//! Tier-3 (bare `no_std + no_alloc`). All variable-length fields
//! (CIDs, stateless-reset tokens, the preferred-address payload) are
//! borrowed slices into the caller's input. No `Vec`, no `Bytes`.
//!
//! # Parameter IDs (RFC 9000 §18.2 + extensions)
//!
//! | ID    | Name | Type | Default |
//! |-------|------|------|---------|
//! | 0x00  | original_destination_connection_id | CID (server-only) | absent |
//! | 0x01  | max_idle_timeout (ms) | varint | 0 (disabled) |
//! | 0x02  | stateless_reset_token | 16 bytes (server-only) | absent |
//! | 0x03  | max_udp_payload_size | varint (>= 1200) | 65527 |
//! | 0x04  | initial_max_data | varint | 0 |
//! | 0x05  | initial_max_stream_data_bidi_local | varint | 0 |
//! | 0x06  | initial_max_stream_data_bidi_remote | varint | 0 |
//! | 0x07  | initial_max_stream_data_uni | varint | 0 |
//! | 0x08  | initial_max_streams_bidi | varint | 0 |
//! | 0x09  | initial_max_streams_uni | varint | 0 |
//! | 0x0a  | ack_delay_exponent | varint (<= 20) | 3 |
//! | 0x0b  | max_ack_delay (ms) | varint (< 2^14) | 25 |
//! | 0x0c  | disable_active_migration | empty | false |
//! | 0x0d  | preferred_address | structure (server-only) | absent |
//! | 0x0e  | active_connection_id_limit | varint (>= 2) | 2 |
//! | 0x0f  | initial_source_connection_id | CID | absent |
//! | 0x10  | retry_source_connection_id | CID (server-only) | absent |
//! | 0x20  | max_datagram_frame_size (RFC 9221 §3) | varint | absent (= disabled) |
//! | 0x3e  | initial_max_path_id (draft-ietf-quic-multipath-21 §2.1) | varint (<= 2^32-1) | absent (= multipath disabled) |

use crate::quic::varint;

/// Length of the stateless-reset token per RFC 9000 §10.3.
pub const STATELESS_RESET_TOKEN_LEN: usize = 16;

/// Length of a preferred-address IPv4 socket-address blob (4 octets + 2 port).
pub const PREFERRED_ADDRESS_IPV4_LEN: usize = 6;

/// Length of a preferred-address IPv6 socket-address blob (16 octets + 2 port).
pub const PREFERRED_ADDRESS_IPV6_LEN: usize = 18;

// Parameter IDs.
const PID_ORIGINAL_DCID: u64 = 0x00;
const PID_MAX_IDLE_TIMEOUT_MS: u64 = 0x01;
const PID_STATELESS_RESET_TOKEN: u64 = 0x02;
const PID_MAX_UDP_PAYLOAD_SIZE: u64 = 0x03;
const PID_INITIAL_MAX_DATA: u64 = 0x04;
const PID_INITIAL_MAX_STREAM_DATA_BIDI_LOCAL: u64 = 0x05;
const PID_INITIAL_MAX_STREAM_DATA_BIDI_REMOTE: u64 = 0x06;
const PID_INITIAL_MAX_STREAM_DATA_UNI: u64 = 0x07;
const PID_INITIAL_MAX_STREAMS_BIDI: u64 = 0x08;
const PID_INITIAL_MAX_STREAMS_UNI: u64 = 0x09;
const PID_ACK_DELAY_EXPONENT: u64 = 0x0a;
const PID_MAX_ACK_DELAY_MS: u64 = 0x0b;
const PID_DISABLE_ACTIVE_MIGRATION: u64 = 0x0c;
const PID_PREFERRED_ADDRESS: u64 = 0x0d;
const PID_ACTIVE_CID_LIMIT: u64 = 0x0e;
const PID_INITIAL_SOURCE_CID: u64 = 0x0f;
const PID_RETRY_SOURCE_CID: u64 = 0x10;
const PID_MAX_DATAGRAM_FRAME_SIZE: u64 = 0x20;
const PID_INITIAL_MAX_PATH_ID: u64 = 0x3e;

/// Server-supplied preferred address (RFC 9000 §18.2.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PreferredAddress<'a> {
    /// 6-byte IPv4 socket address (4 octets + 2-byte port BE).
    /// `[0u8; 6]` indicates the server is not advertising IPv4.
    pub ipv4: [u8; PREFERRED_ADDRESS_IPV4_LEN],
    /// 18-byte IPv6 socket address (16 octets + 2-byte port BE).
    /// `[0u8; 18]` indicates the server is not advertising IPv6.
    pub ipv6: [u8; PREFERRED_ADDRESS_IPV6_LEN],
    /// Connection ID for the alternate path.
    pub connection_id: &'a [u8],
    /// Stateless-reset token for the new CID.
    pub stateless_reset_token: &'a [u8; STATELESS_RESET_TOKEN_LEN],
}

/// Parsed transport parameter set. Every field defaults to `None` (or
/// `false` for [`Self::disable_active_migration`]) when the parameter
/// was not present in the encoded input. The RFC-defined default applies
/// when the field is `None`; callers should consult RFC 9000 §18.2 for
/// the per-parameter default behavior.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TransportParameters<'a> {
    pub original_destination_connection_id: Option<&'a [u8]>,
    pub max_idle_timeout_ms: Option<u64>,
    pub stateless_reset_token: Option<&'a [u8; STATELESS_RESET_TOKEN_LEN]>,
    pub max_udp_payload_size: Option<u64>,
    pub initial_max_data: Option<u64>,
    pub initial_max_stream_data_bidi_local: Option<u64>,
    pub initial_max_stream_data_bidi_remote: Option<u64>,
    pub initial_max_stream_data_uni: Option<u64>,
    pub initial_max_streams_bidi: Option<u64>,
    pub initial_max_streams_uni: Option<u64>,
    pub ack_delay_exponent: Option<u64>,
    pub max_ack_delay_ms: Option<u64>,
    pub disable_active_migration: bool,
    pub preferred_address: Option<PreferredAddress<'a>>,
    pub active_connection_id_limit: Option<u64>,
    pub initial_source_connection_id: Option<&'a [u8]>,
    pub retry_source_connection_id: Option<&'a [u8]>,
    pub max_datagram_frame_size: Option<u64>,
    /// draft-ietf-quic-multipath-21 §2.1 — maximum path ID the
    /// endpoint is willing to maintain. `None` means the peer
    /// either didn't advertise OR doesn't support the multipath
    /// extension. Value `Some(0)` means multipath supported but no
    /// extra paths immediately (peer reserves the right to send
    /// MAX_PATH_ID later to raise the cap).
    pub initial_max_path_id: Option<u64>,
}

/// Parse failures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DecodeError {
    /// Input ran out before a parameter completed.
    Truncated,
    /// A parameter's declared length exceeded the remaining input.
    LengthOverflowsBuffer,
    /// A parameter that should have a fixed length was the wrong size.
    BadParameterLength { id: u64, length: u64 },
    /// A parameter that expected a varint payload was malformed.
    MalformedVarintPayload { id: u64 },
    /// RFC 9000 §18 — a parameter ID appeared more than once.
    DuplicateParameter { id: u64 },
    /// RFC 9000 §18.2 — a value violates its documented bounds
    /// (e.g. `max_udp_payload_size < 1200`, `ack_delay_exponent > 20`,
    /// `active_connection_id_limit < 2`).
    ValueOutOfRange { id: u64, value: u64 },
}

/// Encode failures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum EncodeError {
    /// Output buffer too small.
    BufferTooSmall,
    /// A parameter value exceeded the varint encoding range.
    ValueTooLarge,
}

/// Parse a transport-parameters extension blob into a typed struct.
///
/// Unknown parameter IDs are skipped per RFC 9000 §7.4.2 ("greasing").
///
/// # Errors
///
/// See [`DecodeError`].
#[allow(clippy::too_many_lines)]
pub fn parse(input: &[u8]) -> Result<TransportParameters<'_>, DecodeError> {
    let mut out = TransportParameters::default();
    let mut cursor = 0usize;
    // RFC 9000 §18 — "An endpoint MUST NOT send a parameter more
    // than once in a given transport parameters extension." Track
    // the 18 known parameter IDs (0x00..0x10 + 0x20 + 0x3e) in a
    // u64 bitset (sufficient for ids < 64).
    let mut seen: u64 = 0;
    while cursor < input.len() {
        let (id, id_len) = varint::decode(&input[cursor..]).map_err(|_| DecodeError::Truncated)?;
        cursor += id_len;
        let (length, length_len) =
            varint::decode(&input[cursor..]).map_err(|_| DecodeError::Truncated)?;
        cursor += length_len;
        let length_usize =
            usize::try_from(length).map_err(|_| DecodeError::LengthOverflowsBuffer)?;
        let value = input
            .get(cursor..cursor + length_usize)
            .ok_or(DecodeError::LengthOverflowsBuffer)?;
        cursor += length_usize;

        // Duplicate detection for known IDs (< 64).
        if id < 64 {
            let bit = 1u64 << id;
            if seen & bit != 0 {
                return Err(DecodeError::DuplicateParameter { id });
            }
            seen |= bit;
        }

        match id {
            PID_ORIGINAL_DCID => out.original_destination_connection_id = Some(value),
            PID_MAX_IDLE_TIMEOUT_MS => {
                out.max_idle_timeout_ms = Some(decode_varint_payload(id, value)?);
            }
            PID_STATELESS_RESET_TOKEN => {
                let token: &[u8; STATELESS_RESET_TOKEN_LEN] = value
                    .try_into()
                    .map_err(|_| DecodeError::BadParameterLength { id, length })?;
                out.stateless_reset_token = Some(token);
            }
            PID_MAX_UDP_PAYLOAD_SIZE => {
                let val = decode_varint_payload(id, value)?;
                if val < 1200 {
                    return Err(DecodeError::ValueOutOfRange { id, value: val });
                }
                out.max_udp_payload_size = Some(val);
            }
            PID_INITIAL_MAX_DATA => {
                out.initial_max_data = Some(decode_varint_payload(id, value)?);
            }
            PID_INITIAL_MAX_STREAM_DATA_BIDI_LOCAL => {
                out.initial_max_stream_data_bidi_local = Some(decode_varint_payload(id, value)?);
            }
            PID_INITIAL_MAX_STREAM_DATA_BIDI_REMOTE => {
                out.initial_max_stream_data_bidi_remote = Some(decode_varint_payload(id, value)?);
            }
            PID_INITIAL_MAX_STREAM_DATA_UNI => {
                out.initial_max_stream_data_uni = Some(decode_varint_payload(id, value)?);
            }
            PID_INITIAL_MAX_STREAMS_BIDI => {
                out.initial_max_streams_bidi = Some(decode_varint_payload(id, value)?);
            }
            PID_INITIAL_MAX_STREAMS_UNI => {
                out.initial_max_streams_uni = Some(decode_varint_payload(id, value)?);
            }
            PID_ACK_DELAY_EXPONENT => {
                let val = decode_varint_payload(id, value)?;
                if val > 20 {
                    return Err(DecodeError::ValueOutOfRange { id, value: val });
                }
                out.ack_delay_exponent = Some(val);
            }
            PID_MAX_ACK_DELAY_MS => {
                out.max_ack_delay_ms = Some(decode_varint_payload(id, value)?);
            }
            PID_DISABLE_ACTIVE_MIGRATION => {
                if !value.is_empty() {
                    return Err(DecodeError::BadParameterLength { id, length });
                }
                out.disable_active_migration = true;
            }
            PID_PREFERRED_ADDRESS => {
                out.preferred_address = Some(parse_preferred_address(value)?);
            }
            PID_ACTIVE_CID_LIMIT => {
                let val = decode_varint_payload(id, value)?;
                if val < 2 {
                    return Err(DecodeError::ValueOutOfRange { id, value: val });
                }
                out.active_connection_id_limit = Some(val);
            }
            PID_INITIAL_SOURCE_CID => out.initial_source_connection_id = Some(value),
            PID_RETRY_SOURCE_CID => out.retry_source_connection_id = Some(value),
            PID_MAX_DATAGRAM_FRAME_SIZE => {
                out.max_datagram_frame_size = Some(decode_varint_payload(id, value)?);
            }
            PID_INITIAL_MAX_PATH_ID => {
                let v = decode_varint_payload(id, value)?;
                // draft-21 §2.1: value MUST NOT exceed 2^32-1.
                if v > u64::from(u32::MAX) {
                    return Err(DecodeError::MalformedVarintPayload { id });
                }
                out.initial_max_path_id = Some(v);
            }
            // unknown / reserved — skip per RFC 9000 §7.4.2 greasing rules
            _other => {}
        }
    }
    Ok(out)
}

fn decode_varint_payload(id: u64, value: &[u8]) -> Result<u64, DecodeError> {
    let (decoded, consumed) =
        varint::decode(value).map_err(|_| DecodeError::MalformedVarintPayload { id })?;
    if consumed != value.len() {
        return Err(DecodeError::MalformedVarintPayload { id });
    }
    Ok(decoded)
}

fn parse_preferred_address(value: &[u8]) -> Result<PreferredAddress<'_>, DecodeError> {
    // layout per RFC 9000 §18.2.1:
    //   ipv4 addr (4) | ipv4 port (2) | ipv6 addr (16) | ipv6 port (2)
    //   | cid len (1) | cid (var) | stateless-reset token (16)
    let fixed_prefix = PREFERRED_ADDRESS_IPV4_LEN + PREFERRED_ADDRESS_IPV6_LEN + 1;
    if value.len() < fixed_prefix + STATELESS_RESET_TOKEN_LEN {
        return Err(DecodeError::BadParameterLength {
            id: PID_PREFERRED_ADDRESS,
            length: value.len() as u64,
        });
    }
    let mut ipv4 = [0u8; PREFERRED_ADDRESS_IPV4_LEN];
    ipv4.copy_from_slice(&value[..PREFERRED_ADDRESS_IPV4_LEN]);
    let mut ipv6 = [0u8; PREFERRED_ADDRESS_IPV6_LEN];
    let ipv6_start = PREFERRED_ADDRESS_IPV4_LEN;
    ipv6.copy_from_slice(&value[ipv6_start..ipv6_start + PREFERRED_ADDRESS_IPV6_LEN]);
    let cid_len_idx = ipv6_start + PREFERRED_ADDRESS_IPV6_LEN;
    let cid_len = usize::from(value[cid_len_idx]);
    let cid_start = cid_len_idx + 1;
    let cid_end = cid_start + cid_len;
    if value.len() < cid_end + STATELESS_RESET_TOKEN_LEN {
        return Err(DecodeError::BadParameterLength {
            id: PID_PREFERRED_ADDRESS,
            length: value.len() as u64,
        });
    }
    let connection_id = &value[cid_start..cid_end];
    let token_slice = &value[cid_end..cid_end + STATELESS_RESET_TOKEN_LEN];
    let stateless_reset_token: &[u8; STATELESS_RESET_TOKEN_LEN] =
        token_slice
            .try_into()
            .map_err(|_| DecodeError::BadParameterLength {
                id: PID_PREFERRED_ADDRESS,
                length: value.len() as u64,
            })?;
    Ok(PreferredAddress {
        ipv4,
        ipv6,
        connection_id,
        stateless_reset_token,
    })
}

impl TransportParameters<'_> {
    /// Encode the parameters into `output`. Returns bytes written.
    ///
    /// # Errors
    ///
    /// See [`EncodeError`].
    #[allow(clippy::too_many_lines)]
    pub fn encode(&self, output: &mut [u8]) -> Result<usize, EncodeError> {
        let mut cursor = 0usize;
        if let Some(value) = self.original_destination_connection_id {
            write_bytes_param(output, &mut cursor, PID_ORIGINAL_DCID, value)?;
        }
        if let Some(value) = self.max_idle_timeout_ms {
            write_varint_param(output, &mut cursor, PID_MAX_IDLE_TIMEOUT_MS, value)?;
        }
        if let Some(value) = self.stateless_reset_token {
            write_bytes_param(output, &mut cursor, PID_STATELESS_RESET_TOKEN, value)?;
        }
        if let Some(value) = self.max_udp_payload_size {
            write_varint_param(output, &mut cursor, PID_MAX_UDP_PAYLOAD_SIZE, value)?;
        }
        if let Some(value) = self.initial_max_data {
            write_varint_param(output, &mut cursor, PID_INITIAL_MAX_DATA, value)?;
        }
        if let Some(value) = self.initial_max_stream_data_bidi_local {
            write_varint_param(
                output,
                &mut cursor,
                PID_INITIAL_MAX_STREAM_DATA_BIDI_LOCAL,
                value,
            )?;
        }
        if let Some(value) = self.initial_max_stream_data_bidi_remote {
            write_varint_param(
                output,
                &mut cursor,
                PID_INITIAL_MAX_STREAM_DATA_BIDI_REMOTE,
                value,
            )?;
        }
        if let Some(value) = self.initial_max_stream_data_uni {
            write_varint_param(output, &mut cursor, PID_INITIAL_MAX_STREAM_DATA_UNI, value)?;
        }
        if let Some(value) = self.initial_max_streams_bidi {
            write_varint_param(output, &mut cursor, PID_INITIAL_MAX_STREAMS_BIDI, value)?;
        }
        if let Some(value) = self.initial_max_streams_uni {
            write_varint_param(output, &mut cursor, PID_INITIAL_MAX_STREAMS_UNI, value)?;
        }
        if let Some(value) = self.ack_delay_exponent {
            write_varint_param(output, &mut cursor, PID_ACK_DELAY_EXPONENT, value)?;
        }
        if let Some(value) = self.max_ack_delay_ms {
            write_varint_param(output, &mut cursor, PID_MAX_ACK_DELAY_MS, value)?;
        }
        if self.disable_active_migration {
            write_empty_param(output, &mut cursor, PID_DISABLE_ACTIVE_MIGRATION)?;
        }
        if let Some(addr) = &self.preferred_address {
            write_preferred_address(output, &mut cursor, addr)?;
        }
        if let Some(value) = self.active_connection_id_limit {
            write_varint_param(output, &mut cursor, PID_ACTIVE_CID_LIMIT, value)?;
        }
        if let Some(value) = self.initial_source_connection_id {
            write_bytes_param(output, &mut cursor, PID_INITIAL_SOURCE_CID, value)?;
        }
        if let Some(value) = self.retry_source_connection_id {
            write_bytes_param(output, &mut cursor, PID_RETRY_SOURCE_CID, value)?;
        }
        if let Some(value) = self.max_datagram_frame_size {
            write_varint_param(output, &mut cursor, PID_MAX_DATAGRAM_FRAME_SIZE, value)?;
        }
        if let Some(value) = self.initial_max_path_id {
            if value > u64::from(u32::MAX) {
                return Err(EncodeError::ValueTooLarge);
            }
            write_varint_param(output, &mut cursor, PID_INITIAL_MAX_PATH_ID, value)?;
        }
        Ok(cursor)
    }
}

fn write_varint_param(
    output: &mut [u8],
    cursor: &mut usize,
    id: u64,
    value: u64,
) -> Result<(), EncodeError> {
    let value_len = varint::encoded_len(value);
    write_param_header(output, cursor, id, value_len as u64)?;
    let slot = output
        .get_mut(*cursor..*cursor + value_len)
        .ok_or(EncodeError::BufferTooSmall)?;
    varint::encode(value, slot).map_err(|err| match err {
        varint::EncodeError::ValueTooLarge => EncodeError::ValueTooLarge,
        varint::EncodeError::BufferTooSmall => EncodeError::BufferTooSmall,
    })?;
    *cursor += value_len;
    Ok(())
}

fn write_bytes_param(
    output: &mut [u8],
    cursor: &mut usize,
    id: u64,
    value: &[u8],
) -> Result<(), EncodeError> {
    write_param_header(output, cursor, id, value.len() as u64)?;
    let end = cursor
        .checked_add(value.len())
        .ok_or(EncodeError::BufferTooSmall)?;
    let slot = output
        .get_mut(*cursor..end)
        .ok_or(EncodeError::BufferTooSmall)?;
    slot.copy_from_slice(value);
    *cursor = end;
    Ok(())
}

fn write_empty_param(output: &mut [u8], cursor: &mut usize, id: u64) -> Result<(), EncodeError> {
    write_param_header(output, cursor, id, 0)
}

fn write_param_header(
    output: &mut [u8],
    cursor: &mut usize,
    id: u64,
    length: u64,
) -> Result<(), EncodeError> {
    write_varint_to(output, cursor, id)?;
    write_varint_to(output, cursor, length)
}

fn write_varint_to(output: &mut [u8], cursor: &mut usize, value: u64) -> Result<(), EncodeError> {
    let slot = output
        .get_mut(*cursor..)
        .ok_or(EncodeError::BufferTooSmall)?;
    let written = varint::encode(value, slot).map_err(|err| match err {
        varint::EncodeError::ValueTooLarge => EncodeError::ValueTooLarge,
        varint::EncodeError::BufferTooSmall => EncodeError::BufferTooSmall,
    })?;
    *cursor += written;
    Ok(())
}

fn write_preferred_address(
    output: &mut [u8],
    cursor: &mut usize,
    addr: &PreferredAddress<'_>,
) -> Result<(), EncodeError> {
    let cid_len = addr.connection_id.len();
    let value_len = PREFERRED_ADDRESS_IPV4_LEN
        + PREFERRED_ADDRESS_IPV6_LEN
        + 1
        + cid_len
        + STATELESS_RESET_TOKEN_LEN;
    write_param_header(output, cursor, PID_PREFERRED_ADDRESS, value_len as u64)?;
    let end = cursor
        .checked_add(value_len)
        .ok_or(EncodeError::BufferTooSmall)?;
    let slot = output
        .get_mut(*cursor..end)
        .ok_or(EncodeError::BufferTooSmall)?;
    let mut inner = 0usize;
    slot[inner..inner + PREFERRED_ADDRESS_IPV4_LEN].copy_from_slice(&addr.ipv4);
    inner += PREFERRED_ADDRESS_IPV4_LEN;
    slot[inner..inner + PREFERRED_ADDRESS_IPV6_LEN].copy_from_slice(&addr.ipv6);
    inner += PREFERRED_ADDRESS_IPV6_LEN;
    let cid_len_u8 = u8::try_from(cid_len).map_err(|_| EncodeError::ValueTooLarge)?;
    slot[inner] = cid_len_u8;
    inner += 1;
    slot[inner..inner + cid_len].copy_from_slice(addr.connection_id);
    inner += cid_len;
    slot[inner..inner + STATELESS_RESET_TOKEN_LEN].copy_from_slice(addr.stateless_reset_token);
    *cursor = end;
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_decodes_to_defaults() {
        let parsed = parse(&[]).expect("parse");
        assert_eq!(parsed, TransportParameters::default());
    }

    #[test]
    fn varint_param_round_trip() {
        let original = TransportParameters {
            initial_max_data: Some(1_000_000),
            initial_max_streams_bidi: Some(100),
            initial_max_streams_uni: Some(50),
            max_idle_timeout_ms: Some(30_000),
            ..Default::default()
        };
        let mut buffer = [0u8; 64];
        let written = original.encode(&mut buffer).expect("encode");
        let parsed = parse(&buffer[..written]).expect("parse");
        assert_eq!(parsed, original);
    }

    #[test]
    fn stateless_reset_token_round_trip() {
        let token: [u8; STATELESS_RESET_TOKEN_LEN] = [0x42; STATELESS_RESET_TOKEN_LEN];
        let original = TransportParameters {
            stateless_reset_token: Some(&token),
            ..Default::default()
        };
        let mut buffer = [0u8; 32];
        let written = original.encode(&mut buffer).expect("encode");
        let parsed = parse(&buffer[..written]).expect("parse");
        assert_eq!(parsed.stateless_reset_token, Some(&token));
    }

    #[test]
    fn cids_round_trip() {
        let original_dcid: [u8; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
        let initial_scid: [u8; 4] = [9, 10, 11, 12];
        let retry_scid: [u8; 4] = [13, 14, 15, 16];
        let original = TransportParameters {
            original_destination_connection_id: Some(&original_dcid),
            initial_source_connection_id: Some(&initial_scid),
            retry_source_connection_id: Some(&retry_scid),
            ..Default::default()
        };
        let mut buffer = [0u8; 64];
        let written = original.encode(&mut buffer).expect("encode");
        let parsed = parse(&buffer[..written]).expect("parse");
        assert_eq!(parsed, original);
    }

    #[test]
    fn disable_active_migration_round_trip() {
        let original = TransportParameters {
            disable_active_migration: true,
            ..Default::default()
        };
        let mut buffer = [0u8; 8];
        let written = original.encode(&mut buffer).expect("encode");
        let parsed = parse(&buffer[..written]).expect("parse");
        assert!(parsed.disable_active_migration);
    }

    #[test]
    fn preferred_address_round_trip() {
        let cid: [u8; 4] = [0xa, 0xb, 0xc, 0xd];
        let token: [u8; STATELESS_RESET_TOKEN_LEN] = [0x99; STATELESS_RESET_TOKEN_LEN];
        let ipv4: [u8; PREFERRED_ADDRESS_IPV4_LEN] = [10, 0, 0, 1, 0x01, 0xbb];
        let ipv6: [u8; PREFERRED_ADDRESS_IPV6_LEN] = [
            0x20, 0x01, 0x0d, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x01, 0x01, 0xbb,
        ];
        let pa = PreferredAddress {
            ipv4,
            ipv6,
            connection_id: &cid,
            stateless_reset_token: &token,
        };
        let original = TransportParameters {
            preferred_address: Some(pa),
            ..Default::default()
        };
        let mut buffer = [0u8; 128];
        let written = original.encode(&mut buffer).expect("encode");
        let parsed = parse(&buffer[..written]).expect("parse");
        assert_eq!(parsed.preferred_address, Some(pa));
    }

    #[test]
    fn unknown_parameter_is_ignored() {
        // build: param 0x1f (unknown), length 4, value [1,2,3,4]
        // then param 0x04 (initial_max_data), length 1, value 42
        let mut buffer = [0u8; 16];
        let mut cursor = 0;
        cursor += varint::encode(0x1f, &mut buffer[cursor..]).unwrap();
        cursor += varint::encode(4, &mut buffer[cursor..]).unwrap();
        buffer[cursor..cursor + 4].copy_from_slice(&[1, 2, 3, 4]);
        cursor += 4;
        cursor += varint::encode(PID_INITIAL_MAX_DATA, &mut buffer[cursor..]).unwrap();
        cursor += varint::encode(1, &mut buffer[cursor..]).unwrap();
        cursor += varint::encode(42, &mut buffer[cursor..]).unwrap();
        let parsed = parse(&buffer[..cursor]).expect("parse");
        assert_eq!(parsed.initial_max_data, Some(42));
    }

    #[test]
    fn truncated_input_rejected() {
        // parameter header says length 10 but only 3 bytes remain
        let mut buffer = [0u8; 8];
        let mut cursor = 0;
        cursor += varint::encode(PID_INITIAL_MAX_DATA, &mut buffer[cursor..]).unwrap();
        cursor += varint::encode(10, &mut buffer[cursor..]).unwrap();
        buffer[cursor..cursor + 3].copy_from_slice(&[1, 2, 3]);
        cursor += 3;
        assert_eq!(
            parse(&buffer[..cursor]),
            Err(DecodeError::LengthOverflowsBuffer)
        );
    }

    #[test]
    fn bad_stateless_reset_token_length_rejected() {
        // length 8 (not 16) for stateless reset token
        let mut buffer = [0u8; 16];
        let mut cursor = 0;
        cursor += varint::encode(PID_STATELESS_RESET_TOKEN, &mut buffer[cursor..]).unwrap();
        cursor += varint::encode(8, &mut buffer[cursor..]).unwrap();
        buffer[cursor..cursor + 8].copy_from_slice(&[0u8; 8]);
        cursor += 8;
        assert_eq!(
            parse(&buffer[..cursor]),
            Err(DecodeError::BadParameterLength {
                id: PID_STATELESS_RESET_TOKEN,
                length: 8,
            })
        );
    }

    #[test]
    fn disable_active_migration_with_payload_rejected() {
        let mut buffer = [0u8; 8];
        let mut cursor = 0;
        cursor += varint::encode(PID_DISABLE_ACTIVE_MIGRATION, &mut buffer[cursor..]).unwrap();
        cursor += varint::encode(1, &mut buffer[cursor..]).unwrap();
        buffer[cursor] = 0xff;
        cursor += 1;
        assert_eq!(
            parse(&buffer[..cursor]),
            Err(DecodeError::BadParameterLength {
                id: PID_DISABLE_ACTIVE_MIGRATION,
                length: 1,
            })
        );
    }

    #[test]
    fn max_datagram_frame_size_round_trip() {
        let original = TransportParameters {
            max_datagram_frame_size: Some(1200),
            ..Default::default()
        };
        let mut buffer = [0u8; 8];
        let written = original.encode(&mut buffer).expect("encode");
        let parsed = parse(&buffer[..written]).expect("parse");
        assert_eq!(parsed.max_datagram_frame_size, Some(1200));
    }

    #[test]
    fn initial_max_path_id_round_trip() {
        let original = TransportParameters {
            initial_max_path_id: Some(4),
            ..Default::default()
        };
        let mut buffer = [0u8; 16];
        let written = original.encode(&mut buffer).expect("encode");
        let parsed = parse(&buffer[..written]).expect("parse");
        assert_eq!(parsed.initial_max_path_id, Some(4));
    }

    #[test]
    fn initial_max_path_id_rejects_overflow_on_encode() {
        let original = TransportParameters {
            initial_max_path_id: Some(u64::from(u32::MAX) + 1),
            ..Default::default()
        };
        let mut buffer = [0u8; 32];
        assert_eq!(
            original.encode(&mut buffer),
            Err(EncodeError::ValueTooLarge)
        );
    }

    #[test]
    fn full_set_round_trip() {
        let token: [u8; STATELESS_RESET_TOKEN_LEN] = [0x42; STATELESS_RESET_TOKEN_LEN];
        let dcid: [u8; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
        let initial_scid: [u8; 4] = [9, 10, 11, 12];
        let original = TransportParameters {
            original_destination_connection_id: Some(&dcid),
            max_idle_timeout_ms: Some(30_000),
            stateless_reset_token: Some(&token),
            max_udp_payload_size: Some(1452),
            initial_max_data: Some(16_777_216),
            initial_max_stream_data_bidi_local: Some(1_048_576),
            initial_max_stream_data_bidi_remote: Some(1_048_576),
            initial_max_stream_data_uni: Some(1_048_576),
            initial_max_streams_bidi: Some(100),
            initial_max_streams_uni: Some(100),
            ack_delay_exponent: Some(3),
            max_ack_delay_ms: Some(25),
            disable_active_migration: true,
            preferred_address: None,
            active_connection_id_limit: Some(4),
            initial_source_connection_id: Some(&initial_scid),
            retry_source_connection_id: None,
            max_datagram_frame_size: Some(1200),
            initial_max_path_id: Some(4),
        };
        let mut buffer = [0u8; 256];
        let written = original.encode(&mut buffer).expect("encode");
        let parsed = parse(&buffer[..written]).expect("parse");
        assert_eq!(parsed, original);
    }
}
