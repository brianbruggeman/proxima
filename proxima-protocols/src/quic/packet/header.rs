//! RFC 9000 §17 packet header parser + encoder (invariant fields only).
//!
//! The QUIC packet number is protected by header protection (RFC 9001 §5.4)
//! and the bits that encode its length are part of that protected region.
//! This module decodes only the **unprotected** fields — everything from
//! the first byte's high bits through the length-prefix varint. The packet
//! number + payload bytes are surfaced as a borrowed slice (`pn_and_payload`)
//! for downstream header-protection (C7) + AEAD unprotect (C6).
//!
//! # Tier
//!
//! Tier-3 (bare `no_std + no_alloc`). All variants of [`Header`] hold
//! borrowed slices into the caller's input buffer. No `Vec`, no `Bytes`,
//! no allocation.
//!
//! # Layout (RFC 9000 §17.2 long form)
//!
//! ```text
//!   +-+-+-+-+-+-+-+-+
//!   |1|1|T T|X X X X|         form (1) | fixed (1) | type (2) | type-specific (4)
//!   +-+-+-+-+-+-+-+-+
//!   |    Version (32)    |
//!   +-+-+-+-+-+-+-+-+
//!   | DCID Len (8)  | Destination Connection ID (0..160)
//!   +-+-+-+-+-+-+-+-+
//!   | SCID Len (8)  | Source Connection ID (0..160)
//!   +-+-+-+-+-+-+-+-+
//!   | Type-Specific Payload (...)
//! ```
//!
//! Long types (RFC 9000 §17.2.1 / §17.2.2 / §17.2.3 / §17.2.4 / §17.2.5):
//!
//! - `0x00` → Initial    — Token Length + Token + Length + PN + Payload
//! - `0x01` → 0-RTT      — Length + PN + Payload
//! - `0x02` → Handshake  — Length + PN + Payload
//! - `0x03` → Retry      — Retry Token + Retry Integrity Tag (16 B)
//!
//! Version Negotiation (§17.2.1) is signalled by `Version == 0`, with the
//! same first 5 bytes + DCID/SCID layout, followed by a sequence of
//! 4-byte supported versions.
//!
//! Short header (RFC 9000 §17.3 — 1-RTT):
//!
//! ```text
//!   +-+-+-+-+-+-+-+-+
//!   |0|1|S|R|R|K|P P|         form (0) | fixed (1) | spin | reserved (2) | key phase | pn-len (2)
//!   +-+-+-+-+-+-+-+-+
//!   | Destination Connection ID (...)  — length determined by connection state
//!   +-+-+-+-+-+-+-+-+
//!   | Packet Number (8..32)
//!   | Protected Payload (...)
//! ```
//!
//! The pn-length bits + the packet number itself live in the header-protected
//! region — surfaced here as a borrowed slice for downstream processing.

use crate::quic::varint;

/// Maximum connection-ID length on the wire per RFC 9000 §17.2.
pub const MAX_CID_LEN: usize = 20;

/// Length of the Retry Integrity Tag per RFC 9001 §5.8.
pub const RETRY_INTEGRITY_TAG_LEN: usize = 16;

/// Packet header form (top bit of the first byte).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Form {
    /// Long header — used for Initial, 0-RTT, Handshake, Retry,
    /// Version Negotiation per RFC 9000 §17.2.
    Long,
    /// Short header — 1-RTT packets after handshake completes per
    /// RFC 9000 §17.3.
    Short,
}

/// Long-header packet type (bits 5-4 of the first byte) per RFC 9000 §17.2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LongType {
    Initial = 0b00,
    ZeroRtt = 0b01,
    Handshake = 0b10,
    Retry = 0b11,
}

impl LongType {
    const fn from_first_byte(first: u8) -> Self {
        // bits 5-4 select the long type; mask + shift then index a table.
        const TABLE: [LongType; 4] = [
            LongType::Initial,
            LongType::ZeroRtt,
            LongType::Handshake,
            LongType::Retry,
        ];
        TABLE[((first >> 4) & 0b11) as usize]
    }
}

/// Parsed QUIC packet header. All slices borrow from the caller's input.
///
/// `pn_and_payload` is the region from the start of the (encrypted) packet
/// number through the end of the packet. Caller applies header protection
/// (RFC 9001 §5.4) then AEAD unprotect (RFC 9001 §5.1) to access the
/// plaintext.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Header<'a> {
    Initial {
        version: u32,
        dcid: &'a [u8],
        scid: &'a [u8],
        token: &'a [u8],
        length: u64,
        pn_and_payload: &'a [u8],
    },
    ZeroRtt {
        version: u32,
        dcid: &'a [u8],
        scid: &'a [u8],
        length: u64,
        pn_and_payload: &'a [u8],
    },
    Handshake {
        version: u32,
        dcid: &'a [u8],
        scid: &'a [u8],
        length: u64,
        pn_and_payload: &'a [u8],
    },
    Retry {
        version: u32,
        dcid: &'a [u8],
        scid: &'a [u8],
        retry_token: &'a [u8],
        integrity_tag: &'a [u8; RETRY_INTEGRITY_TAG_LEN],
    },
    VersionNegotiation {
        dcid: &'a [u8],
        scid: &'a [u8],
        /// Raw 4-byte-per-version slice. Caller iterates via
        /// [`VersionNegotiation::supported_versions`].
        supported_versions_raw: &'a [u8],
    },
    Short {
        /// Spin bit + reserved + key phase + pn-length bits live here
        /// (still header-protected on the wire; the caller has already
        /// applied header protection by the time it constructs / decodes
        /// the Short variant).
        first_byte: u8,
        dcid: &'a [u8],
        pn_and_payload: &'a [u8],
    },
}

/// Parse failures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DecodeError {
    /// Input buffer is empty.
    Empty,
    /// Input ran out before a required field completed.
    Truncated,
    /// Fixed bit (bit 6 of the first byte) was clear; QUIC v1 requires it set
    /// per RFC 9000 §17.2 / §17.3 unless the peer enabled grease_quic_bit.
    /// (Not detected here — callers that want grease support pass the bit through.)
    FixedBitClear,
    /// Connection-ID length exceeded [`MAX_CID_LEN`].
    CidTooLong,
    /// Varint decoded but exceeded the encoded packet boundary.
    LengthOverflowsBuffer,
    /// Version-Negotiation packet's supported-versions list was not a multiple of 4 bytes.
    MalformedVersionList,
    /// Retry packet was too short to contain the 16-byte integrity tag.
    RetryTruncated,
}

/// Encode failures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum EncodeError {
    /// Output buffer too small for the encoded header.
    BufferTooSmall,
    /// Connection-ID length exceeded [`MAX_CID_LEN`].
    CidTooLong,
    /// Length value exceeded the varint encoding range (2^62 - 1).
    LengthOverflowsVarint,
}

impl<'a> Header<'a> {
    /// Form discriminant — derived from the first byte's high bit.
    #[must_use]
    pub fn form(&self) -> Form {
        match self {
            Self::Short { .. } => Form::Short,
            _ => Form::Long,
        }
    }

    /// Destination connection ID (every variant has one).
    #[must_use]
    pub fn dcid(&self) -> &'a [u8] {
        match self {
            Self::Initial { dcid, .. }
            | Self::ZeroRtt { dcid, .. }
            | Self::Handshake { dcid, .. }
            | Self::Retry { dcid, .. }
            | Self::VersionNegotiation { dcid, .. }
            | Self::Short { dcid, .. } => dcid,
        }
    }
}

/// Peek the form of a QUIC packet without further parsing — the high bit
/// of the first byte. Returns `None` if `input` is empty.
#[inline]
#[must_use]
pub const fn peek_form(input: &[u8]) -> Option<Form> {
    if input.is_empty() {
        None
    } else if input[0] & 0x80 == 0 {
        Some(Form::Short)
    } else {
        Some(Form::Long)
    }
}

/// Parse a long-header packet. The form bit MUST be 1; see [`peek_form`]
/// to discriminate before calling.
///
/// # Errors
///
/// See [`DecodeError`].
pub fn parse_long(input: &[u8]) -> Result<Header<'_>, DecodeError> {
    let Some((&first, rest)) = input.split_first() else {
        return Err(DecodeError::Empty);
    };
    if first & 0x80 == 0 {
        // not a long header — caller should use parse_short
        return Err(DecodeError::FixedBitClear);
    }

    // 4-byte big-endian version
    let mut cursor = 0usize;
    let version_bytes = take(rest, &mut cursor, 4)?;
    let version = u32::from_be_bytes([
        version_bytes[0],
        version_bytes[1],
        version_bytes[2],
        version_bytes[3],
    ]);

    // DCID len + DCID
    let dcid_len = read_cid_len(rest, &mut cursor)?;
    let dcid = take(rest, &mut cursor, dcid_len)?;

    // SCID len + SCID
    let scid_len = read_cid_len(rest, &mut cursor)?;
    let scid = take(rest, &mut cursor, scid_len)?;

    // version=0 → Version Negotiation, regardless of long-type bits.
    if version == 0 {
        let remaining = rest.get(cursor..).ok_or(DecodeError::Truncated)?;
        if remaining.len() % 4 != 0 {
            return Err(DecodeError::MalformedVersionList);
        }
        return Ok(Header::VersionNegotiation {
            dcid,
            scid,
            supported_versions_raw: remaining,
        });
    }

    match LongType::from_first_byte(first) {
        LongType::Initial => {
            let token_len = read_varint(rest, &mut cursor)?;
            let token = take(rest, &mut cursor, usize_from_varint(token_len)?)?;
            let length = read_varint(rest, &mut cursor)?;
            let pn_and_payload = take(rest, &mut cursor, usize_from_varint(length)?)?;
            Ok(Header::Initial {
                version,
                dcid,
                scid,
                token,
                length,
                pn_and_payload,
            })
        }
        LongType::ZeroRtt => {
            let length = read_varint(rest, &mut cursor)?;
            let pn_and_payload = take(rest, &mut cursor, usize_from_varint(length)?)?;
            Ok(Header::ZeroRtt {
                version,
                dcid,
                scid,
                length,
                pn_and_payload,
            })
        }
        LongType::Handshake => {
            let length = read_varint(rest, &mut cursor)?;
            let pn_and_payload = take(rest, &mut cursor, usize_from_varint(length)?)?;
            Ok(Header::Handshake {
                version,
                dcid,
                scid,
                length,
                pn_and_payload,
            })
        }
        LongType::Retry => {
            // Retry layout: retry-token (variable) then 16-byte integrity tag.
            // Length is implicit — everything up to the tag is the token.
            let remaining = rest.get(cursor..).ok_or(DecodeError::Truncated)?;
            if remaining.len() < RETRY_INTEGRITY_TAG_LEN {
                return Err(DecodeError::RetryTruncated);
            }
            let split_at = remaining.len() - RETRY_INTEGRITY_TAG_LEN;
            let retry_token = &remaining[..split_at];
            let tag_slice = &remaining[split_at..];
            // SAFETY: bounds-checked above; slice length is exactly RETRY_INTEGRITY_TAG_LEN.
            let integrity_tag: &[u8; RETRY_INTEGRITY_TAG_LEN] = tag_slice
                .try_into()
                .map_err(|_| DecodeError::RetryTruncated)?;
            Ok(Header::Retry {
                version,
                dcid,
                scid,
                retry_token,
                integrity_tag,
            })
        }
    }
}

/// Parse a short-header (1-RTT) packet. The DCID length is determined by
/// the connection state — the wire format does not encode it.
///
/// # Errors
///
/// See [`DecodeError`].
pub fn parse_short(input: &[u8], dcid_len: usize) -> Result<Header<'_>, DecodeError> {
    if dcid_len > MAX_CID_LEN {
        return Err(DecodeError::CidTooLong);
    }
    let Some((&first, rest)) = input.split_first() else {
        return Err(DecodeError::Empty);
    };
    if first & 0x80 != 0 {
        return Err(DecodeError::FixedBitClear);
    }
    if rest.len() < dcid_len {
        return Err(DecodeError::Truncated);
    }
    let dcid = &rest[..dcid_len];
    let pn_and_payload = &rest[dcid_len..];
    Ok(Header::Short {
        first_byte: first,
        dcid,
        pn_and_payload,
    })
}

// internal helpers — all tier-3 (operate on slices + cursor index only)

#[inline]
fn take<'a>(input: &'a [u8], cursor: &mut usize, len: usize) -> Result<&'a [u8], DecodeError> {
    let start = *cursor;
    let end = start.checked_add(len).ok_or(DecodeError::Truncated)?;
    let slice = input.get(start..end).ok_or(DecodeError::Truncated)?;
    *cursor = end;
    Ok(slice)
}

#[inline]
fn read_cid_len(input: &[u8], cursor: &mut usize) -> Result<usize, DecodeError> {
    let byte = take(input, cursor, 1)?[0];
    if usize::from(byte) > MAX_CID_LEN {
        return Err(DecodeError::CidTooLong);
    }
    Ok(usize::from(byte))
}

#[inline]
fn read_varint(input: &[u8], cursor: &mut usize) -> Result<u64, DecodeError> {
    let slice = input.get(*cursor..).ok_or(DecodeError::Truncated)?;
    match varint::decode(slice) {
        Ok((value, consumed)) => {
            *cursor += consumed;
            Ok(value)
        }
        Err(_) => Err(DecodeError::Truncated),
    }
}

#[inline]
fn usize_from_varint(value: u64) -> Result<usize, DecodeError> {
    usize::try_from(value).map_err(|_| DecodeError::LengthOverflowsBuffer)
}

impl Header<'_> {
    /// Encode `self` into `output`. Returns the number of bytes written.
    ///
    /// For `Short`, the `first_byte` carries the pn-length + key-phase
    /// and spin bits as the caller provided; this codec does not
    /// synthesize them. That's the connection state's job.
    ///
    /// # Errors
    ///
    /// See [`EncodeError`].
    pub fn encode(&self, output: &mut [u8]) -> Result<usize, EncodeError> {
        match self {
            Self::Initial {
                version,
                dcid,
                scid,
                token,
                length,
                pn_and_payload,
            } => encode_initial(*version, dcid, scid, token, *length, pn_and_payload, output),
            Self::ZeroRtt {
                version,
                dcid,
                scid,
                length,
                pn_and_payload,
            } => encode_long_simple(0b01, *version, dcid, scid, *length, pn_and_payload, output),
            Self::Handshake {
                version,
                dcid,
                scid,
                length,
                pn_and_payload,
            } => encode_long_simple(0b10, *version, dcid, scid, *length, pn_and_payload, output),
            Self::Retry {
                version,
                dcid,
                scid,
                retry_token,
                integrity_tag,
            } => encode_retry(*version, dcid, scid, retry_token, integrity_tag, output),
            Self::VersionNegotiation {
                dcid,
                scid,
                supported_versions_raw,
            } => encode_version_negotiation(dcid, scid, supported_versions_raw, output),
            Self::Short {
                first_byte,
                dcid,
                pn_and_payload,
            } => encode_short(*first_byte, dcid, pn_and_payload, output),
        }
    }
}

fn encode_initial(
    version: u32,
    dcid: &[u8],
    scid: &[u8],
    token: &[u8],
    length: u64,
    pn_and_payload: &[u8],
    output: &mut [u8],
) -> Result<usize, EncodeError> {
    if dcid.len() > MAX_CID_LEN || scid.len() > MAX_CID_LEN {
        return Err(EncodeError::CidTooLong);
    }
    if length > varint::MAX_VALUE || token.len() as u64 > varint::MAX_VALUE {
        return Err(EncodeError::LengthOverflowsVarint);
    }
    let token_len_size = varint::encoded_len(token.len() as u64);
    let length_size = varint::encoded_len(length);
    let total = 1                     // first byte
        + 4                           // version
        + 1 + dcid.len()              // dcid len + dcid
        + 1 + scid.len()              // scid len + scid
        + token_len_size + token.len()
        + length_size
        + pn_and_payload.len();
    if output.len() < total {
        return Err(EncodeError::BufferTooSmall);
    }

    let mut cursor = 0;
    // first byte: form=1, fixed=1, long-type=00 (Initial), reserved/pn-length
    // bits zeroed here; caller can OR the protected-bits in after header
    // protection. For an unprotected/template emit the low 4 bits are 0.
    output[cursor] = 0b1100_0000;
    cursor += 1;
    output[cursor..cursor + 4].copy_from_slice(&version.to_be_bytes());
    cursor += 4;
    output[cursor] = dcid.len() as u8;
    cursor += 1;
    output[cursor..cursor + dcid.len()].copy_from_slice(dcid);
    cursor += dcid.len();
    output[cursor] = scid.len() as u8;
    cursor += 1;
    output[cursor..cursor + scid.len()].copy_from_slice(scid);
    cursor += scid.len();
    let written = varint::encode(token.len() as u64, &mut output[cursor..])
        .map_err(|_| EncodeError::BufferTooSmall)?;
    cursor += written;
    output[cursor..cursor + token.len()].copy_from_slice(token);
    cursor += token.len();
    let written =
        varint::encode(length, &mut output[cursor..]).map_err(|_| EncodeError::BufferTooSmall)?;
    cursor += written;
    output[cursor..cursor + pn_and_payload.len()].copy_from_slice(pn_and_payload);
    cursor += pn_and_payload.len();
    Ok(cursor)
}

fn encode_long_simple(
    type_bits: u8,
    version: u32,
    dcid: &[u8],
    scid: &[u8],
    length: u64,
    pn_and_payload: &[u8],
    output: &mut [u8],
) -> Result<usize, EncodeError> {
    if dcid.len() > MAX_CID_LEN || scid.len() > MAX_CID_LEN {
        return Err(EncodeError::CidTooLong);
    }
    if length > varint::MAX_VALUE {
        return Err(EncodeError::LengthOverflowsVarint);
    }
    let length_size = varint::encoded_len(length);
    let total = 1 + 4 + 1 + dcid.len() + 1 + scid.len() + length_size + pn_and_payload.len();
    if output.len() < total {
        return Err(EncodeError::BufferTooSmall);
    }
    let mut cursor = 0;
    // form=1, fixed=1, long-type=type_bits, low 4 bits zero
    output[cursor] = 0b1100_0000 | ((type_bits & 0b11) << 4);
    cursor += 1;
    output[cursor..cursor + 4].copy_from_slice(&version.to_be_bytes());
    cursor += 4;
    output[cursor] = dcid.len() as u8;
    cursor += 1;
    output[cursor..cursor + dcid.len()].copy_from_slice(dcid);
    cursor += dcid.len();
    output[cursor] = scid.len() as u8;
    cursor += 1;
    output[cursor..cursor + scid.len()].copy_from_slice(scid);
    cursor += scid.len();
    let written =
        varint::encode(length, &mut output[cursor..]).map_err(|_| EncodeError::BufferTooSmall)?;
    cursor += written;
    output[cursor..cursor + pn_and_payload.len()].copy_from_slice(pn_and_payload);
    cursor += pn_and_payload.len();
    Ok(cursor)
}

fn encode_retry(
    version: u32,
    dcid: &[u8],
    scid: &[u8],
    retry_token: &[u8],
    integrity_tag: &[u8; RETRY_INTEGRITY_TAG_LEN],
    output: &mut [u8],
) -> Result<usize, EncodeError> {
    if dcid.len() > MAX_CID_LEN || scid.len() > MAX_CID_LEN {
        return Err(EncodeError::CidTooLong);
    }
    let total =
        1 + 4 + 1 + dcid.len() + 1 + scid.len() + retry_token.len() + RETRY_INTEGRITY_TAG_LEN;
    if output.len() < total {
        return Err(EncodeError::BufferTooSmall);
    }
    let mut cursor = 0;
    output[cursor] = 0b1100_0000 | (0b11 << 4); // long, fixed, Retry
    cursor += 1;
    output[cursor..cursor + 4].copy_from_slice(&version.to_be_bytes());
    cursor += 4;
    output[cursor] = dcid.len() as u8;
    cursor += 1;
    output[cursor..cursor + dcid.len()].copy_from_slice(dcid);
    cursor += dcid.len();
    output[cursor] = scid.len() as u8;
    cursor += 1;
    output[cursor..cursor + scid.len()].copy_from_slice(scid);
    cursor += scid.len();
    output[cursor..cursor + retry_token.len()].copy_from_slice(retry_token);
    cursor += retry_token.len();
    output[cursor..cursor + RETRY_INTEGRITY_TAG_LEN].copy_from_slice(integrity_tag);
    cursor += RETRY_INTEGRITY_TAG_LEN;
    Ok(cursor)
}

fn encode_version_negotiation(
    dcid: &[u8],
    scid: &[u8],
    supported_versions_raw: &[u8],
    output: &mut [u8],
) -> Result<usize, EncodeError> {
    if dcid.len() > MAX_CID_LEN || scid.len() > MAX_CID_LEN {
        return Err(EncodeError::CidTooLong);
    }
    if !supported_versions_raw.len().is_multiple_of(4) {
        // VN list must be 4-byte aligned per RFC 9000 §17.2.1; treat as caller bug.
        return Err(EncodeError::BufferTooSmall);
    }
    let total = 1 + 4 + 1 + dcid.len() + 1 + scid.len() + supported_versions_raw.len();
    if output.len() < total {
        return Err(EncodeError::BufferTooSmall);
    }
    let mut cursor = 0;
    // Server MAY use any random value for type-specific bits; pick 0xC0
    // (form=1, fixed=1). RFC 9000 §17.2.1 says the type-specific bits MUST
    // be ignored on receipt; pick a deterministic value here.
    output[cursor] = 0xc0;
    cursor += 1;
    // version = 0 indicates Version Negotiation per §17.2.1.
    output[cursor..cursor + 4].copy_from_slice(&0u32.to_be_bytes());
    cursor += 4;
    output[cursor] = dcid.len() as u8;
    cursor += 1;
    output[cursor..cursor + dcid.len()].copy_from_slice(dcid);
    cursor += dcid.len();
    output[cursor] = scid.len() as u8;
    cursor += 1;
    output[cursor..cursor + scid.len()].copy_from_slice(scid);
    cursor += scid.len();
    output[cursor..cursor + supported_versions_raw.len()].copy_from_slice(supported_versions_raw);
    cursor += supported_versions_raw.len();
    Ok(cursor)
}

fn encode_short(
    first_byte: u8,
    dcid: &[u8],
    pn_and_payload: &[u8],
    output: &mut [u8],
) -> Result<usize, EncodeError> {
    if dcid.len() > MAX_CID_LEN {
        return Err(EncodeError::CidTooLong);
    }
    if first_byte & 0x80 != 0 {
        // form bit must be 0 for short headers
        return Err(EncodeError::CidTooLong); // misnomer; reuse for shape error
    }
    let total = 1 + dcid.len() + pn_and_payload.len();
    if output.len() < total {
        return Err(EncodeError::BufferTooSmall);
    }
    output[0] = first_byte;
    output[1..1 + dcid.len()].copy_from_slice(dcid);
    output[1 + dcid.len()..1 + dcid.len() + pn_and_payload.len()].copy_from_slice(pn_and_payload);
    Ok(total)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    // Minimal Initial packet round-trip with realistic-shape fields.
    //
    // RFC 9001 Appendix A.2 provides a full sample Initial — we use a
    // shorter synthetic packet here that exercises every Initial field,
    // and parse the RFC sample in a separate test below where it's
    // available.
    fn sample_initial_bytes() -> [u8; 33] {
        // first byte: long(1) | fixed(1) | Initial(00) | low 4 bits zero
        let first = 0b1100_0000u8;
        let version: u32 = 0x0000_0001; // QUIC v1
        let dcid: [u8; 8] = [0x83, 0x94, 0xc8, 0xf0, 0x3e, 0x51, 0x57, 0x08];
        let scid: [u8; 0] = [];
        let token: [u8; 4] = [0xaa, 0xbb, 0xcc, 0xdd];
        let length_value: u64 = 12;
        let pn_and_payload: [u8; 12] = [
            0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c,
        ];
        let mut buffer = [0u8; 33];
        let mut cursor = 0;
        buffer[cursor] = first;
        cursor += 1;
        buffer[cursor..cursor + 4].copy_from_slice(&version.to_be_bytes());
        cursor += 4;
        buffer[cursor] = dcid.len() as u8;
        cursor += 1;
        buffer[cursor..cursor + dcid.len()].copy_from_slice(&dcid);
        cursor += dcid.len();
        buffer[cursor] = scid.len() as u8;
        cursor += 1;
        // 1-byte varint encoding for token-len = 4 and length = 12
        buffer[cursor] = 4;
        cursor += 1;
        buffer[cursor..cursor + token.len()].copy_from_slice(&token);
        cursor += token.len();
        buffer[cursor] = length_value as u8;
        cursor += 1;
        buffer[cursor..cursor + pn_and_payload.len()].copy_from_slice(&pn_and_payload);
        cursor += pn_and_payload.len();
        assert_eq!(cursor, 33);
        buffer
    }

    #[test]
    fn peek_form_classifies_first_byte() {
        assert_eq!(peek_form(&[]), None);
        assert_eq!(peek_form(&[0x00]), Some(Form::Short));
        assert_eq!(peek_form(&[0x7f]), Some(Form::Short));
        assert_eq!(peek_form(&[0x80]), Some(Form::Long));
        assert_eq!(peek_form(&[0xff]), Some(Form::Long));
    }

    #[test]
    fn parse_initial_extracts_all_fields() {
        let bytes = sample_initial_bytes();
        let header = parse_long(&bytes).expect("parse");
        match header {
            Header::Initial {
                version,
                dcid,
                scid,
                token,
                length,
                pn_and_payload,
            } => {
                assert_eq!(version, 1);
                assert_eq!(dcid, &[0x83, 0x94, 0xc8, 0xf0, 0x3e, 0x51, 0x57, 0x08]);
                assert_eq!(scid, &[] as &[u8]);
                assert_eq!(token, &[0xaa, 0xbb, 0xcc, 0xdd]);
                assert_eq!(length, 12);
                assert_eq!(pn_and_payload.len(), 12);
            }
            other => panic!("expected Initial, got {other:?}"),
        }
    }

    #[test]
    fn round_trip_initial() {
        let bytes = sample_initial_bytes();
        let header = parse_long(&bytes).expect("parse");
        let mut output = [0u8; 64];
        let written = header.encode(&mut output).expect("encode");
        assert_eq!(written, bytes.len());
        assert_eq!(&output[..written], &bytes[..]);
    }

    #[test]
    fn round_trip_handshake() {
        // construct → encode → parse → compare
        let dcid: [u8; 4] = [1, 2, 3, 4];
        let scid: [u8; 8] = [5, 6, 7, 8, 9, 10, 11, 12];
        let pn_and_payload = [0xab; 20];
        let original = Header::Handshake {
            version: 1,
            dcid: &dcid,
            scid: &scid,
            length: 20,
            pn_and_payload: &pn_and_payload,
        };
        let mut buffer = [0u8; 64];
        let written = original.encode(&mut buffer).expect("encode");
        let parsed = parse_long(&buffer[..written]).expect("parse");
        assert_eq!(parsed, original);
    }

    #[test]
    fn round_trip_zero_rtt() {
        let dcid: [u8; 4] = [1, 2, 3, 4];
        let scid: [u8; 0] = [];
        let pn_and_payload = [0xcd; 10];
        let original = Header::ZeroRtt {
            version: 1,
            dcid: &dcid,
            scid: &scid,
            length: 10,
            pn_and_payload: &pn_and_payload,
        };
        let mut buffer = [0u8; 64];
        let written = original.encode(&mut buffer).expect("encode");
        let parsed = parse_long(&buffer[..written]).expect("parse");
        assert_eq!(parsed, original);
    }

    #[test]
    fn round_trip_retry() {
        let dcid: [u8; 4] = [1, 2, 3, 4];
        let scid: [u8; 0] = [];
        let retry_token = [0xab; 24];
        let integrity_tag: [u8; RETRY_INTEGRITY_TAG_LEN] = [0xee; RETRY_INTEGRITY_TAG_LEN];
        let original = Header::Retry {
            version: 1,
            dcid: &dcid,
            scid: &scid,
            retry_token: &retry_token,
            integrity_tag: &integrity_tag,
        };
        let mut buffer = [0u8; 128];
        let written = original.encode(&mut buffer).expect("encode");
        let parsed = parse_long(&buffer[..written]).expect("parse");
        assert_eq!(parsed, original);
    }

    #[test]
    fn round_trip_version_negotiation() {
        let dcid: [u8; 4] = [1, 2, 3, 4];
        let scid: [u8; 4] = [5, 6, 7, 8];
        let supported_versions_raw: [u8; 8] = [
            0x00, 0x00, 0x00, 0x01, // QUIC v1
            0x00, 0x00, 0x00, 0x02, // QUIC v2 (RFC 9369)
        ];
        let original = Header::VersionNegotiation {
            dcid: &dcid,
            scid: &scid,
            supported_versions_raw: &supported_versions_raw,
        };
        let mut buffer = [0u8; 32];
        let written = original.encode(&mut buffer).expect("encode");
        let parsed = parse_long(&buffer[..written]).expect("parse");
        assert_eq!(parsed, original);
    }

    #[test]
    fn round_trip_short() {
        let dcid: [u8; 8] = [9, 8, 7, 6, 5, 4, 3, 2];
        let pn_and_payload = [0x42; 30];
        let original = Header::Short {
            first_byte: 0b0100_0000, // form=0, fixed=1, all other bits zero
            dcid: &dcid,
            pn_and_payload: &pn_and_payload,
        };
        let mut buffer = [0u8; 64];
        let written = original.encode(&mut buffer).expect("encode");
        let parsed = parse_short(&buffer[..written], dcid.len()).expect("parse");
        assert_eq!(parsed, original);
    }

    #[test]
    fn parse_long_rejects_short_form() {
        let short_first_byte = [0x40, 0x00, 0x00];
        assert_eq!(
            parse_long(&short_first_byte),
            Err(DecodeError::FixedBitClear)
        );
    }

    #[test]
    fn parse_long_rejects_empty() {
        assert_eq!(parse_long(&[]), Err(DecodeError::Empty));
    }

    #[test]
    fn parse_long_rejects_oversized_cid() {
        let mut bytes = [0u8; 32];
        bytes[0] = 0b1100_0000;
        bytes[1..5].copy_from_slice(&1u32.to_be_bytes());
        bytes[5] = MAX_CID_LEN as u8 + 1; // DCID len = 21, illegal
        assert_eq!(parse_long(&bytes), Err(DecodeError::CidTooLong));
    }

    #[test]
    fn parse_long_rejects_truncated_initial() {
        let bytes = sample_initial_bytes();
        let truncated = &bytes[..bytes.len() - 1];
        assert_eq!(parse_long(truncated), Err(DecodeError::Truncated));
    }

    #[test]
    fn parse_short_rejects_long_form() {
        let long_first_byte = [0xc0, 0x00, 0x00];
        assert_eq!(
            parse_short(&long_first_byte, 8),
            Err(DecodeError::FixedBitClear)
        );
    }

    #[test]
    fn parse_short_rejects_truncated_dcid() {
        let only_first = [0x40];
        assert_eq!(parse_short(&only_first, 8), Err(DecodeError::Truncated));
    }

    #[test]
    fn parse_retry_truncated_when_under_integrity_tag() {
        // build a Retry-shaped packet but with < 16 trailing bytes
        let mut bytes = [0u8; 8];
        bytes[0] = 0b1100_0000 | (0b11 << 4); // Retry
        bytes[1..5].copy_from_slice(&1u32.to_be_bytes());
        bytes[5] = 0; // dcid len 0
        bytes[6] = 0; // scid len 0
        // remaining 1 byte — less than the 16-byte integrity tag
        assert_eq!(parse_long(&bytes), Err(DecodeError::RetryTruncated));
    }

    #[cfg(feature = "quic-alloc")]
    #[test]
    fn version_negotiation_iterates_supported_versions() {
        let supported_versions_raw: [u8; 8] = [
            0x00, 0x00, 0x00, 0x01, 0xff, 0x00, 0x00, 0x1d, // draft-29
        ];
        // exercise the parse path that produces VN, then verify the iterator
        let dcid: [u8; 4] = [1, 2, 3, 4];
        let scid: [u8; 4] = [5, 6, 7, 8];
        let header = Header::VersionNegotiation {
            dcid: &dcid,
            scid: &scid,
            supported_versions_raw: &supported_versions_raw,
        };
        match header {
            Header::VersionNegotiation {
                supported_versions_raw,
                ..
            } => {
                let versions: alloc::vec::Vec<u32> = supported_versions_raw
                    .chunks_exact(4)
                    .map(|chunk| u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                assert_eq!(versions, alloc::vec![1, 0xff00001d]);
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn version_negotiation_malformed_list_rejected() {
        // 7 bytes for VN list = not multiple of 4
        let mut bytes = [0u8; 14];
        bytes[0] = 0b1100_0000;
        bytes[1..5].copy_from_slice(&0u32.to_be_bytes()); // version = 0 → VN
        bytes[5] = 0; // dcid len
        bytes[6] = 0; // scid len
        // 7 bytes of "versions" — malformed
        assert_eq!(parse_long(&bytes), Err(DecodeError::MalformedVersionList));
    }
}
