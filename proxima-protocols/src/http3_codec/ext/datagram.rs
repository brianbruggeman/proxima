//! HTTP/3 Datagrams per [RFC 9297].
//!
//! H3-Datagrams ride on top of RFC 9221 unreliable QUIC DATAGRAM frames
//! (proxima-quic-proto §RFC 9221 / `datagram` module). The H3 layer
//! adds a **Quarter Stream ID (QSID)** prefix to each datagram so the
//! receiver can demux into the per-request quarter-stream sink.
//!
//! Per RFC 9297 §2.1:
//!
//! ```text
//! H3 Datagram Payload {
//!     Quarter Stream ID (i),
//!     HTTP Datagram Payload (..),
//! }
//! ```
//!
//! The QSID is `stream_id / 4` (varint), where `stream_id` is the
//! client-initiated bidi stream ID for the associated request.
//!
//! Negotiation per §3: both peers MUST advertise `SETTINGS_H3_DATAGRAM
//! = 1` (proxima-h3-proto::settings::SETTINGS_H3_DATAGRAM).
//!
//! [RFC 9297]: https://www.rfc-editor.org/rfc/rfc9297

use alloc::vec::Vec;

use crate::quic::varint;

/// H3 Datagram codec errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DatagramError {
    /// QSID varint decode failed.
    InvalidVarint,
    /// Input ended mid-QSID.
    Truncated,
    /// Output buffer too small for the encoded datagram.
    BufferTooSmall { needed: usize },
    /// Stream ID was not a client-initiated bidi ID (not divisible by 4).
    InvalidStreamId { stream_id: u64 },
}

/// Encode an H3 datagram: prepend the QSID varint to `payload`,
/// write into `output`. Returns total bytes written.
///
/// # Errors
///
/// See [`DatagramError`].
pub fn encode(stream_id: u64, payload: &[u8], output: &mut [u8]) -> Result<usize, DatagramError> {
    if !stream_id.is_multiple_of(4) {
        return Err(DatagramError::InvalidStreamId { stream_id });
    }
    let qsid = stream_id / 4;
    let qsid_len = varint::encoded_len(qsid);
    let needed = qsid_len + payload.len();
    if output.len() < needed {
        return Err(DatagramError::BufferTooSmall { needed });
    }
    let written = varint::encode(qsid, output).map_err(|_| DatagramError::InvalidVarint)?;
    output[written..written + payload.len()].copy_from_slice(payload);
    Ok(written + payload.len())
}

/// Encode into a freshly-allocated `Vec` for caller ergonomics.
///
/// # Errors
///
/// See [`DatagramError`].
pub fn encode_to_vec(stream_id: u64, payload: &[u8]) -> Result<Vec<u8>, DatagramError> {
    if !stream_id.is_multiple_of(4) {
        return Err(DatagramError::InvalidStreamId { stream_id });
    }
    let qsid = stream_id / 4;
    let qsid_len = varint::encoded_len(qsid);
    let mut buf = alloc::vec![0u8; qsid_len + payload.len()];
    let written = encode(stream_id, payload, &mut buf)?;
    buf.truncate(written);
    Ok(buf)
}

/// Decoded H3 datagram view.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct H3Datagram<'a> {
    /// Associated request bidi stream ID (`qsid * 4`).
    pub stream_id: u64,
    pub payload: &'a [u8],
}

/// Parse an inbound H3 datagram — strips the QSID varint prefix.
///
/// # Errors
///
/// See [`DatagramError`].
pub fn parse(input: &[u8]) -> Result<H3Datagram<'_>, DatagramError> {
    // DecodeError is #[non_exhaustive] for external crates, but varint now
    // lives in this same crate (folded from proxima-quic-proto) — within
    // the defining crate the attribute doesn't force a wildcard arm, so
    // the match is exhaustive over the two known variants.
    let (qsid, consumed) = varint::decode(input).map_err(|err| match err {
        varint::DecodeError::Empty | varint::DecodeError::Truncated => DatagramError::Truncated,
    })?;
    Ok(H3Datagram {
        stream_id: qsid.saturating_mul(4),
        payload: &input[consumed..],
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_qsid_zero_stream_id_zero() {
        let payload = b"data-1";
        let encoded = encode_to_vec(0, payload).unwrap();
        let parsed = parse(&encoded).unwrap();
        assert_eq!(parsed.stream_id, 0);
        assert_eq!(parsed.payload, payload);
    }

    #[test]
    fn round_trip_qsid_one_stream_id_four() {
        let payload = b"data-2";
        let encoded = encode_to_vec(4, payload).unwrap();
        let parsed = parse(&encoded).unwrap();
        assert_eq!(parsed.stream_id, 4);
        assert_eq!(parsed.payload, payload);
    }

    #[test]
    fn encode_rejects_non_bidi_stream_id() {
        let err = encode_to_vec(3, b"x").unwrap_err();
        assert!(matches!(
            err,
            DatagramError::InvalidStreamId { stream_id: 3 }
        ));
    }

    #[test]
    fn parse_truncated_input_rejected() {
        assert_eq!(parse(&[]), Err(DatagramError::Truncated));
    }

    #[test]
    fn round_trip_qsid_large_stream_id() {
        let payload = b"x";
        let encoded = encode_to_vec(1024, payload).unwrap();
        let parsed = parse(&encoded).unwrap();
        assert_eq!(parsed.stream_id, 1024);
        assert_eq!(parsed.payload, payload);
    }
}
