//! [`DnsTcpCodec`] — the TCP-direction `proxima_codec::FrameCodec` +
//! `codec_pipe::OwnFrame`/`Incomplete` impl DNS-over-TCP needs to plug into
//! `proxima_listen::any::FramedAny`, the generic stateless `AnyProtocol`
//! driver. Reuses [`super::parse_message`] (decode) UNCHANGED — no wire
//! logic is rewritten here, only wrapped in the trait shapes `FramedAny`
//! composes against. Encoding a reply still goes through
//! [`super::encode::encode_response`] one layer up, at the
//! `proxima-dns` `FramedApp` that owns the typed query/answer pair
//! [`super::encode::encode_response`] needs (`id`/`flags`/question/
//! answers) — this codec only ever sees the fully-rendered bytes.
//!
//! DNS is genuinely symmetric: RFC 1035 §4.2.2 frames a query and a
//! response with the SAME 2-byte-length-prefixed wire shape (a `QR` bit
//! inside the message is what tells them apart, not the framing). So
//! [`FrameCodec::Frame`] here is ONE type, `&'a [u8]` — the message body
//! bytes, minus the 2-byte length prefix — used identically for both
//! directions. No sum type, no unreachable arm (contrast
//! [`super::super::memcached::frame_codec::MemcachedFrame`], whose
//! `Request`/`Reply` split memcached's genuinely asymmetric wire).
//!
//! [`DnsTcpCodec::parse_frame`] only ever answers "is a complete frame
//! buffered yet" — it does NOT validate the message body is a well-formed
//! DNS message; a malformed body still frames successfully (framing and
//! semantic validity are different questions for a length-prefixed wire).
//! [`DnsTcpCodec::own_frame`] does that semantic parse instead, via
//! [`super::parse_message`] + a single-question check, folding a
//! malformed or multi-question body into
//! [`DnsTcpOwnedFrame::Violation`] rather than a hard [`FrameCodec::Error`]
//! — mirroring the deleted `proxima-dns` `handle_one_message`'s own
//! "warn and skip this frame, keep the connection open" contract exactly
//! (see `proxima-dns`'s `framed_app` module for how the App layer answers
//! a `Violation`). A declared length over [`DnsTcpCodec::max_message_bytes`]
//! IS a hard [`DnsTcpFrameError::MessageTooLarge`] — the DoS guard fires
//! before the (possibly attacker-controlled) body is even framed, matching
//! the deleted driver's own "close immediately, don't buffer it" contract.

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

use bytes::Bytes;
use proxima_codec::FrameCodec;

use crate::codec_pipe::{Incomplete, OwnFrame};
use crate::dns::codec_trait::parse_message;

/// RFC 1035 §4.2.2 length-prefix width, in bytes.
const TCP_LENGTH_PREFIX_BYTES: usize = 2;

/// Why [`DnsTcpCodec::parse_frame`] / [`DnsTcpCodec::encode_frame`] could
/// not make progress. Only [`Self::Incomplete`] means "read more and
/// retry" ([`Incomplete::is_incomplete`]) — the other two are hard,
/// connection-closing failures, matching the deleted bespoke driver's own
/// "close immediately" behavior for both an over-declared incoming length
/// and a reply too large for the `u16` length prefix to carry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DnsTcpFrameError {
    /// The buffer does not hold a complete frame yet.
    Incomplete,
    /// The 2-byte length prefix declares more than `max_message_bytes` —
    /// refused before the body is even read into the frame buffer.
    MessageTooLarge { declared: usize, limit: usize },
    /// An encoded reply is too large for the `u16` length prefix to carry
    /// (RFC 1035 §4.2.2's TCP framing tops out at 65535 bytes).
    ReplyTooLarge { len: usize },
}

impl fmt::Display for DnsTcpFrameError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DnsTcpFrameError::Incomplete => {
                formatter.write_str("incomplete: dns-tcp frame not yet fully buffered")
            }
            DnsTcpFrameError::MessageTooLarge { declared, limit } => write!(
                formatter,
                "declared length {declared} exceeds the {limit} byte message limit"
            ),
            DnsTcpFrameError::ReplyTooLarge { len } => write!(
                formatter,
                "encoded reply of {len} bytes exceeds the u16 length-prefix range"
            ),
        }
    }
}

impl core::error::Error for DnsTcpFrameError {}

impl Incomplete for DnsTcpFrameError {
    fn is_incomplete(&self) -> bool {
        matches!(self, DnsTcpFrameError::Incomplete)
    }
}

/// DNS-over-TCP (RFC 1035 §4.2.2) [`FrameCodec`]. Carries
/// [`Self::max_message_bytes`] (mirrors
/// `proxima_dns::config::DnsServerConfig::max_message_bytes`) — the DoS
/// cap [`Self::parse_frame`] enforces directly against the declared
/// length, before the body is ever framed.
#[derive(Debug, Clone, Copy)]
pub struct DnsTcpCodec {
    pub max_message_bytes: usize,
}

impl DnsTcpCodec {
    #[must_use]
    pub const fn new(max_message_bytes: usize) -> Self {
        Self { max_message_bytes }
    }
}

impl FrameCodec for DnsTcpCodec {
    type Frame<'a> = &'a [u8];
    type Error = DnsTcpFrameError;

    fn parse_frame<'a>(&self, buf: &'a [u8]) -> Result<(&'a [u8], usize), DnsTcpFrameError> {
        if buf.len() < TCP_LENGTH_PREFIX_BYTES {
            return Err(DnsTcpFrameError::Incomplete);
        }
        let declared_len = usize::from(u16::from_be_bytes([buf[0], buf[1]]));
        if declared_len > self.max_message_bytes {
            return Err(DnsTcpFrameError::MessageTooLarge {
                declared: declared_len,
                limit: self.max_message_bytes,
            });
        }
        let total = TCP_LENGTH_PREFIX_BYTES + declared_len;
        if buf.len() < total {
            return Err(DnsTcpFrameError::Incomplete);
        }
        Ok((&buf[TCP_LENGTH_PREFIX_BYTES..total], total))
    }

    fn encode_frame(&self, frame: &&[u8], dest: &mut Vec<u8>) -> Result<(), DnsTcpFrameError> {
        let len = u16::try_from(frame.len())
            .map_err(|_| DnsTcpFrameError::ReplyTooLarge { len: frame.len() })?;
        dest.extend_from_slice(&len.to_be_bytes());
        dest.extend_from_slice(frame);
        Ok(())
    }
}

/// One decoded DNS-over-TCP query, owned — the framing-level mirror of
/// `proxima_dns::pipes::DnsQuery` (that richer type can't be referenced
/// here; `proxima-protocols` sits below `proxima-dns` in the dependency
/// graph). `proxima-dns`'s own `FramedApp` re-owns this into its business
/// type at the pipe boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DnsTcpQuery {
    /// Echoed back into the response header (RFC 1035 §4.1.1).
    pub id: u16,
    /// `RD` bit of the query.
    pub recursion_desired: bool,
    /// Dotted question name, e.g. `"example.com."`.
    pub name: String,
    pub qtype: u16,
    pub qclass: u16,
}

/// Why a framed message did not yield a [`DnsTcpQuery`] — the listener
/// drops these and keeps serving (see `proxima-dns`'s `framed_app`
/// module), mirroring the deleted `handle_one_message`'s own
/// warn-and-skip contract for both cases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DnsTcpViolation {
    /// The framed bytes are not a well-formed DNS message.
    Malformed,
    /// The message parsed, but does not carry exactly one question (RFC
    /// 1035 §4.1.2 permits more; no deployed client/server pair does).
    NotSingleQuestion,
}

/// [`OwnFrame::Owned`] for [`DnsTcpCodec`] — either a usable query or a
/// reason it wasn't one. Never constructed for the encode direction; a
/// reply's bytes are handed to [`DnsTcpCodec::encode_frame`] directly as
/// `&[u8]`, with no owning step in between.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DnsTcpOwnedFrame {
    Query(DnsTcpQuery),
    Violation(DnsTcpViolation),
}

impl OwnFrame for DnsTcpCodec {
    type Owned = DnsTcpOwnedFrame;

    fn own_frame(_source: &Bytes, frame: &&[u8]) -> DnsTcpOwnedFrame {
        let message = match parse_message(frame) {
            Ok(message) => message,
            Err(_error) => return DnsTcpOwnedFrame::Violation(DnsTcpViolation::Malformed),
        };
        if message.header.qdcount != 1 {
            return DnsTcpOwnedFrame::Violation(DnsTcpViolation::NotSingleQuestion);
        }
        let question = match message.questions().next() {
            Some(Ok(question)) => question,
            _ => return DnsTcpOwnedFrame::Violation(DnsTcpViolation::NotSingleQuestion),
        };
        DnsTcpOwnedFrame::Query(DnsTcpQuery {
            id: message.header.id,
            recursion_desired: message.header.flags.rd(),
            name: question.name.to_dotted(),
            qtype: question.qtype,
            qclass: question.qclass,
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use alloc::string::ToString;
    use alloc::vec;

    use super::*;
    use crate::dns::encode;

    fn codec() -> DnsTcpCodec {
        DnsTcpCodec::new(65_535)
    }

    fn framed_query(id: u16) -> Vec<u8> {
        let mut message = Vec::new();
        encode::encode_query(
            id,
            true,
            encode::EncodeQuestion {
                name: "example.com.",
                qtype: 1,
                qclass: 1,
            },
            &mut message,
        )
        .unwrap();
        let mut framed = Vec::new();
        framed.extend_from_slice(&u16::try_from(message.len()).unwrap().to_be_bytes());
        framed.extend_from_slice(&message);
        framed
    }

    #[test]
    fn parse_frame_needs_more_bytes_below_the_length_prefix() {
        let outcome = codec().parse_frame(&[0u8]);
        assert_eq!(outcome.unwrap_err(), DnsTcpFrameError::Incomplete);
    }

    #[test]
    fn parse_frame_needs_more_bytes_when_the_body_is_still_short() {
        let framed = framed_query(1234);
        let outcome = codec().parse_frame(&framed[..framed.len() - 1]);
        assert_eq!(outcome.unwrap_err(), DnsTcpFrameError::Incomplete);
    }

    #[test]
    fn parse_frame_returns_the_complete_message_body() {
        let framed = framed_query(1234);
        let (frame, consumed) = codec().parse_frame(&framed).expect("parses");
        assert_eq!(consumed, framed.len());
        assert_eq!(frame, &framed[2..]);
    }

    #[test]
    fn parse_frame_rejects_a_declared_length_over_the_limit() {
        let framed = framed_query(1234);
        let small_codec = DnsTcpCodec::new(4);
        let outcome = small_codec.parse_frame(&framed);
        assert_eq!(
            outcome.unwrap_err(),
            DnsTcpFrameError::MessageTooLarge {
                declared: framed.len() - 2,
                limit: 4,
            }
        );
    }

    #[test]
    fn encode_frame_prefixes_the_length_and_copies_the_body() {
        let mut dest = Vec::new();
        codec().encode_frame(&b"hello".as_slice(), &mut dest).expect("encode");
        assert_eq!(dest, [0u8, 5, b'h', b'e', b'l', b'l', b'o']);
    }

    #[test]
    fn own_frame_reowns_a_well_formed_single_question_query() {
        let framed = framed_query(1234);
        let (frame, _) = codec().parse_frame(&framed).expect("parses");
        let owned = DnsTcpCodec::own_frame(&Bytes::from_static(b"unused"), &frame);
        assert_eq!(
            owned,
            DnsTcpOwnedFrame::Query(DnsTcpQuery {
                id: 1234,
                recursion_desired: true,
                name: "example.com.".to_string(),
                qtype: 1,
                qclass: 1,
            })
        );
    }

    #[test]
    fn own_frame_flags_a_malformed_body_as_a_violation() {
        let owned = DnsTcpCodec::own_frame(&Bytes::new(), &[0u8; 4].as_slice());
        assert_eq!(
            owned,
            DnsTcpOwnedFrame::Violation(DnsTcpViolation::Malformed)
        );
    }

    #[test]
    fn own_frame_flags_a_multi_question_body_as_a_violation() {
        // two well-formed questions — legal wire bytes (RFC 1035 §4.1.2
        // permits qdcount > 1), just not the single-question shape this
        // listener answers.
        let mut body = vec![0u8; 12];
        body[4..6].copy_from_slice(&2u16.to_be_bytes());
        encode::encode_name("example.com.", &mut body).unwrap();
        body.extend_from_slice(&1u16.to_be_bytes());
        body.extend_from_slice(&1u16.to_be_bytes());
        encode::encode_name("example.org.", &mut body).unwrap();
        body.extend_from_slice(&1u16.to_be_bytes());
        body.extend_from_slice(&1u16.to_be_bytes());

        let owned = DnsTcpCodec::own_frame(&Bytes::new(), &body.as_slice());
        assert_eq!(
            owned,
            DnsTcpOwnedFrame::Violation(DnsTcpViolation::NotSingleQuestion)
        );
    }
}
