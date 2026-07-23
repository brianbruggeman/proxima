//! Sans-IO DNS message encoder (RFC 1035 §4.1) — the write-side sibling of
//! [`super::parse_header`] / [`super::parse_question`] / [`super::parse_record`].
//! The parser was the only half that existed after the fold from the former
//! `proxima-dns` crate (see the module doc); a resolver client can't send a
//! query, and a server can't answer one, without building wire bytes from
//! scratch, so this module is the other half `proxima-dns` needs to function.
//!
//! Deliberately does **not** apply RFC 1035 §4.1.4 name compression on
//! encode: compression is an optional wire-size optimization, not part of
//! the framing contract (a decompressed name is legal wire format — every
//! parser, including [`super::parse_name`], accepts it). A typical
//! query (one question) or response (one question, a handful of answers)
//! stays comfortably under the classic 512-byte UDP envelope even with
//! every name spelled out in full. A compressing encoder is a scoped-out
//! future optimization, not a hidden gap in this one.

use alloc::vec::Vec;

use super::{Flags, ParseError};

/// One question-section entry to encode — the mirror of [`super::Question`],
/// but by owned/borrowed `&str` name rather than a parsed [`super::Name`],
/// since an encoder builds a message that doesn't exist on the wire yet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EncodeQuestion<'a> {
    pub name: &'a str,
    pub qtype: u16,
    pub qclass: u16,
}

/// One answer/authority/additional-section record to encode. `rdata` is
/// already-encoded wire bytes (e.g. 4 bytes for an A record) — see
/// [`ipv4_rdata`] / [`ipv6_rdata`] for the common cases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AnswerRecord<'a> {
    pub name: &'a str,
    pub rtype: u16,
    pub rclass: u16,
    pub ttl: u32,
    pub rdata: &'a [u8],
}

/// Failure to encode a message. Distinct from [`ParseError`] (a decode-only
/// type) because the failure modes differ: encoding fails on a caller-
/// supplied value violating an RFC 1035 §2.3.4 size limit, never on
/// truncated/malformed wire bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodeError {
    /// A single label exceeds the 63-byte limit (RFC 1035 §2.3.4).
    LabelTooLong,
    /// The dotted name's total wire encoding exceeds 255 bytes (RFC 1035 §2.3.4).
    NameTooLong,
    /// More than 65535 records in one section — `u16` count field can't
    /// represent it (RFC 1035 §4.1.1).
    TooManyRecords,
}

impl core::fmt::Display for EncodeError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::LabelTooLong => write!(formatter, "dns label exceeds 63 bytes"),
            Self::NameTooLong => write!(formatter, "dns name exceeds 255 bytes"),
            Self::TooManyRecords => write!(formatter, "record count exceeds u16::MAX"),
        }
    }
}

impl core::error::Error for EncodeError {}

/// Encode a dotted name (`"example.com"` or `"example.com."`) into its
/// RFC 1035 §3.1 label sequence, appended to `out`. A trailing dot (the
/// fully-qualified form) is optional and stripped; the root name (`""` or
/// `"."`) encodes to the single terminating zero-length label.
pub fn encode_name(name: &str, out: &mut Vec<u8>) -> Result<(), EncodeError> {
    let trimmed = name.strip_suffix('.').unwrap_or(name);
    if trimmed.is_empty() {
        out.push(0);
        return Ok(());
    }
    let mut total = 0usize;
    for label in trimmed.split('.') {
        let bytes = label.as_bytes();
        // RFC 1035 §2.3.4: a label is at most 63 bytes — `u8::try_from`
        // alone would accept up to 255, so the limit needs its own check.
        if bytes.len() > 63 {
            return Err(EncodeError::LabelTooLong);
        }
        let label_len = u8::try_from(bytes.len()).map_err(|_| EncodeError::LabelTooLong)?;
        // +1 for this label's own length-prefix byte, +1 more (added after
        // the loop) for the terminating zero label.
        total += bytes.len() + 1;
        if total > 254 {
            return Err(EncodeError::NameTooLong);
        }
        out.push(label_len);
        out.extend_from_slice(bytes);
    }
    out.push(0);
    Ok(())
}

/// Write the 12-byte fixed header (RFC 1035 §4.1.1).
fn encode_header(id: u16, flags: Flags, qdcount: u16, ancount: u16, out: &mut Vec<u8>) {
    out.extend_from_slice(&id.to_be_bytes());
    out.extend_from_slice(&flags.0.to_be_bytes());
    out.extend_from_slice(&qdcount.to_be_bytes());
    out.extend_from_slice(&ancount.to_be_bytes());
    out.extend_from_slice(&0u16.to_be_bytes()); // nscount — authority section unused
    out.extend_from_slice(&0u16.to_be_bytes()); // arcount — additional section unused
}

fn encode_question(question: EncodeQuestion<'_>, out: &mut Vec<u8>) -> Result<(), EncodeError> {
    encode_name(question.name, out)?;
    out.extend_from_slice(&question.qtype.to_be_bytes());
    out.extend_from_slice(&question.qclass.to_be_bytes());
    Ok(())
}

fn encode_answer(record: AnswerRecord<'_>, out: &mut Vec<u8>) -> Result<(), EncodeError> {
    encode_name(record.name, out)?;
    out.extend_from_slice(&record.rtype.to_be_bytes());
    out.extend_from_slice(&record.rclass.to_be_bytes());
    out.extend_from_slice(&record.ttl.to_be_bytes());
    let rdlength = u16::try_from(record.rdata.len()).map_err(|_| EncodeError::NameTooLong)?;
    out.extend_from_slice(&rdlength.to_be_bytes());
    out.extend_from_slice(record.rdata);
    Ok(())
}

/// Encode a one-question query message. `id` identifies the query so the
/// matching response (which echoes it, RFC 1035 §4.1.1) can be paired up
/// by a caller tracking multiple in-flight queries. Use [`Flags::for_query`]
/// to build a caller-supplied `flags` word, or call [`encode_query`] for
/// the common "just ask, recursively" case.
pub fn encode_query_with_flags(
    id: u16,
    flags: Flags,
    question: EncodeQuestion<'_>,
    out: &mut Vec<u8>,
) -> Result<(), EncodeError> {
    encode_header(id, flags, 1, 0, out);
    encode_question(question, out)
}

/// Encode a standard recursive query (`RD` set per `recursion_desired`,
/// opcode `QUERY`, RFC 1035 §4.1.1) — the shape every stub resolver sends.
pub fn encode_query(
    id: u16,
    recursion_desired: bool,
    question: EncodeQuestion<'_>,
    out: &mut Vec<u8>,
) -> Result<(), EncodeError> {
    encode_query_with_flags(id, Flags::for_query(recursion_desired), question, out)
}

/// Encode a response message answering `question`: `id` and `question`
/// echo the query being answered (RFC 1035 §4.1.1/§4.1.2 — a response
/// carries the question it answers back in its own question section),
/// `flags` carries `QR=1` plus the caller's `RCODE`/`AA`/`RA`/`RD`-echo
/// (build via [`Flags::for_response`]), and `answers` becomes the answer
/// section.
pub fn encode_response(
    id: u16,
    flags: Flags,
    question: EncodeQuestion<'_>,
    answers: &[AnswerRecord<'_>],
    out: &mut Vec<u8>,
) -> Result<(), EncodeError> {
    let ancount = u16::try_from(answers.len()).map_err(|_| EncodeError::TooManyRecords)?;
    encode_header(id, flags, 1, ancount, out);
    encode_question(question, out)?;
    for answer in answers {
        encode_answer(*answer, out)?;
    }
    Ok(())
}

/// RFC 1035 §3.4.1 A-record rdata: the 4 address octets, network order.
#[must_use]
pub fn ipv4_rdata(addr: core::net::Ipv4Addr) -> [u8; 4] {
    addr.octets()
}

/// RFC 3596 AAAA-record rdata: the 16 address octets, network order.
#[must_use]
pub fn ipv6_rdata(addr: core::net::Ipv6Addr) -> [u8; 16] {
    addr.octets()
}

/// Reduce a decode-time [`ParseError`] into the closest [`EncodeError`] —
/// used by callers that round-trip a parsed name's [`super::Name::to_dotted`]
/// straight back into [`encode_name`] and want one error type at the edge.
impl From<ParseError> for EncodeError {
    fn from(_: ParseError) -> Self {
        Self::NameTooLong
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::dns::{parse_header, parse_question, parse_record};
    use core::net::Ipv4Addr;

    #[test]
    fn encode_name_matches_the_parser_worked_example() {
        // Same fixture the parser's own tests use — "example.com." decodes
        // to 13 bytes: 1+7 ("example") + 1+3 ("com") + 1 (root).
        let mut out = Vec::new();
        encode_name("example.com.", &mut out).unwrap();
        assert_eq!(out.len(), 13);
        assert_eq!(&out[..1], &[7]);
        assert_eq!(&out[1..8], b"example");
        assert_eq!(&out[8..9], &[3]);
        assert_eq!(&out[9..12], b"com");
        assert_eq!(out[12], 0);
    }

    #[test]
    fn encode_name_without_trailing_dot_is_identical() {
        let mut with_dot = Vec::new();
        encode_name("example.com.", &mut with_dot).unwrap();
        let mut without_dot = Vec::new();
        encode_name("example.com", &mut without_dot).unwrap();
        assert_eq!(with_dot, without_dot);
    }

    #[test]
    fn encode_name_rejects_a_label_over_63_bytes() {
        let long_label = "a".repeat(64);
        let mut out = Vec::new();
        assert_eq!(
            encode_name(&long_label, &mut out),
            Err(EncodeError::LabelTooLong)
        );
    }

    #[test]
    fn encode_query_round_trips_through_the_parser() {
        let mut out = Vec::new();
        let question = EncodeQuestion {
            name: "example.com.",
            qtype: 1,
            qclass: 1,
        };
        encode_query(1234, true, question, &mut out).unwrap();

        let header = parse_header(&out).unwrap();
        assert_eq!(header.id, 1234);
        assert!(!header.flags.is_response());
        assert!(header.flags.rd());
        assert_eq!(header.qdcount, 1);

        let (parsed_question, _) = parse_question(&out, 12).unwrap();
        assert_eq!(parsed_question.name.to_dotted(), "example.com.");
        assert_eq!(parsed_question.qtype, 1);
        assert_eq!(parsed_question.qclass, 1);
    }

    #[test]
    fn encode_response_round_trips_through_the_parser_with_an_a_record() {
        let question = EncodeQuestion {
            name: "example.com.",
            qtype: 1,
            qclass: 1,
        };
        let rdata = ipv4_rdata(Ipv4Addr::new(93, 184, 216, 34));
        let answers = [AnswerRecord {
            name: "example.com.",
            rtype: 1,
            rclass: 1,
            ttl: 300,
            rdata: &rdata,
        }];
        let flags = Flags::for_response(true, false, true, 0);

        let mut out = Vec::new();
        encode_response(1234, flags, question, &answers, &mut out).unwrap();

        let header = parse_header(&out).unwrap();
        assert!(header.flags.is_response());
        assert!(header.flags.ra());
        assert!(header.flags.rd());
        assert_eq!(header.flags.rcode(), 0);
        assert_eq!(header.ancount, 1);

        let (parsed_question, used) = parse_question(&out, 12).unwrap();
        assert_eq!(parsed_question.name.to_dotted(), "example.com.");

        let (record, _) = parse_record(&out, 12 + used).unwrap();
        assert_eq!(record.name.to_dotted(), "example.com.");
        assert_eq!(record.ttl, 300);
        assert_eq!(
            record.rdata,
            super::super::RData::A(Ipv4Addr::new(93, 184, 216, 34))
        );
    }

    #[test]
    fn encode_response_rejects_more_than_u16_max_answers() {
        // Cheap to construct: reuse one zero-length rdata slice for all of
        // them — the count check fires before any per-record encoding.
        let question = EncodeQuestion {
            name: ".",
            qtype: 1,
            qclass: 1,
        };
        let template = AnswerRecord {
            name: ".",
            rtype: 1,
            rclass: 1,
            ttl: 0,
            rdata: &[],
        };
        let answers = alloc::vec![template; usize::from(u16::MAX) + 1];
        let mut out = Vec::new();
        assert_eq!(
            encode_response(1, Flags::for_response(false, false, false, 0), question, &answers, &mut out),
            Err(EncodeError::TooManyRecords)
        );
    }
}
