//! The sans-IO half of the resolver client: bytes in, bytes out, no socket
//! (workspace principle 11).
//!
//! One DNS query/response exchange has no intermediate states — a single
//! request, a single reply, matched by the 16-bit query id RFC 1035 §4.1.1
//! already carries. So this half holds no state at all and is two functions
//! rather than a session type: the id is an INPUT, minted by whoever owns the
//! exchange ([`super::pipe::DnsClientUpstream`]), which is also what keeps
//! encoding deterministic and testable at this layer.

use proxima_protocols::dns::codec_trait::parse_message;
use proxima_protocols::dns::encode::{self, EncodeQuestion};

use crate::error::DnsClientError;
use crate::pipes::{DnsAnswer, DnsAnswerWithMetadata};
use crate::wire::message_to_answer_with_metadata;

/// Build query wire bytes for one question under the caller-supplied id (the
/// value RFC 1035 §4.1.1 requires the reply to echo).
///
/// # Errors
/// [`DnsClientError::Wire`] if `name` violates RFC 1035's label or total
/// name length limits.
pub fn encode_query(
    id: u16,
    name: &str,
    qtype: u16,
    qclass: u16,
    recursion_desired: bool,
) -> Result<Vec<u8>, DnsClientError> {
    let mut out = Vec::new();
    encode::encode_query(
        id,
        recursion_desired,
        EncodeQuestion {
            name,
            qtype,
            qclass,
        },
        &mut out,
    )
    .map_err(|error| DnsClientError::Wire(error.to_string()))?;
    Ok(out)
}

/// Decode a reply, verifying its id matches the query it answers.
///
/// # Errors
/// [`DnsClientError::IdMismatch`] when the reply answers a different query,
/// [`DnsClientError::Wire`] when the message or one of its answer records
/// fails to parse.
pub fn decode_response(expected_id: u16, bytes: &[u8]) -> Result<DnsAnswer, DnsClientError> {
    Ok(decode_response_with_metadata(expected_id, bytes)?.answer)
}

/// Decode a reply while retaining the echoed question and DNS TC bit.
pub fn decode_response_with_metadata(
    expected_id: u16,
    bytes: &[u8],
) -> Result<DnsAnswerWithMetadata, DnsClientError> {
    let message = parse_message(bytes).map_err(|error| DnsClientError::Wire(error.to_string()))?;
    if message.header.id != expected_id {
        return Err(DnsClientError::IdMismatch {
            expected: expected_id,
            reply: message.header.id,
        });
    }
    message_to_answer_with_metadata(&message).ok_or_else(|| {
        DnsClientError::Wire("response question or answer record failed to decode".to_string())
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use proxima_protocols::dns::encode::ipv4_rdata;

    #[test]
    fn encode_query_stamps_the_id_the_caller_asked_for() {
        let bytes = encode_query(4242, "example.com.", 1, 1, true).unwrap();
        let message = parse_message(&bytes).unwrap();
        assert_eq!(message.header.id, 4242);
        assert!(message.header.flags.rd());
    }

    #[test]
    fn encode_query_rejects_an_over_long_label() {
        let too_long = "a".repeat(64);
        let outcome = encode_query(1, &format!("{too_long}.example."), 1, 1, true);
        assert!(matches!(outcome, Err(DnsClientError::Wire(_))));
    }

    #[test]
    fn encode_query_round_trips_through_the_listener_side_wire_helper() {
        let id = 7;
        let query_bytes = encode_query(id, "example.com.", 1, 1, true).unwrap();
        assert_eq!(parse_message(&query_bytes).unwrap().header.id, id);

        // build a plausible response the way `crate::wire::answer_to_wire`
        // would, and confirm decode_response reads it back correctly.
        let mut response = Vec::new();
        let flags = proxima_protocols::dns::Flags::for_response(true, false, true, 0);
        let rdata = ipv4_rdata(core::net::Ipv4Addr::new(93, 184, 216, 34));
        let answer_record = encode::AnswerRecord {
            name: "example.com.",
            rtype: 1,
            rclass: 1,
            ttl: 300,
            rdata: &rdata,
        };
        encode::encode_response(
            id,
            flags,
            EncodeQuestion {
                name: "example.com.",
                qtype: 1,
                qclass: 1,
            },
            &[answer_record],
            &mut response,
        )
        .unwrap();

        let detailed = decode_response_with_metadata(id, &response).unwrap();
        assert_eq!(detailed.answer.rcode, 0);
        assert_eq!(detailed.answer.records.len(), 1);
        assert_eq!(detailed.answer.records[0].name, "example.com.");
        assert_eq!(
            detailed.metadata.question,
            Some(crate::pipes::DnsQuery {
                id,
                recursion_desired: true,
                name: "example.com.".to_string(),
                qtype: 1,
                qclass: 1,
            })
        );
        assert!(!detailed.metadata.truncated);

        // The answer-only facade remains byte/API compatible with callers
        // that do not need envelope metadata.
        assert_eq!(decode_response(id, &response).unwrap(), detailed.answer);
    }

    #[test]
    fn decode_response_with_metadata_preserves_the_tc_bit() {
        let id = 8;
        let mut response = Vec::new();
        let base = proxima_protocols::dns::Flags::for_response(true, false, true, 0);
        let flags = proxima_protocols::dns::Flags(base.0 | 0x0200);
        encode::encode_response(
            id,
            flags,
            EncodeQuestion {
                name: "example.com.",
                qtype: 28,
                qclass: 1,
            },
            &[],
            &mut response,
        )
        .unwrap();

        let detailed = decode_response_with_metadata(id, &response).unwrap();
        assert!(detailed.metadata.truncated);
        assert_eq!(detailed.metadata.question.unwrap().qtype, 28);
    }

    #[test]
    fn decode_response_rejects_a_mismatched_id() {
        let mut response = Vec::new();
        let flags = proxima_protocols::dns::Flags::for_response(true, false, true, 0);
        encode::encode_response(
            999,
            flags,
            EncodeQuestion {
                name: "example.com.",
                qtype: 1,
                qclass: 1,
            },
            &[],
            &mut response,
        )
        .unwrap();

        let outcome = decode_response(1, &response);
        assert!(matches!(
            outcome,
            Err(DnsClientError::IdMismatch {
                expected: 1,
                reply: 999
            })
        ));
    }

    #[test]
    fn decode_response_rejects_non_response_or_nonstandard_opcode() {
        let mut response = Vec::new();
        let flags = proxima_protocols::dns::Flags::for_response(true, false, true, 0);
        encode::encode_response(
            7,
            flags,
            EncodeQuestion {
                name: "example.com.",
                qtype: 1,
                qclass: 1,
            },
            &[],
            &mut response,
        )
        .unwrap();

        response[2] &= 0x7f;
        assert!(matches!(
            decode_response_with_metadata(7, &response),
            Err(DnsClientError::Wire(_))
        ));

        response[2] |= 0x80;
        response[2] |= 0x08;
        assert!(matches!(
            decode_response_with_metadata(7, &response),
            Err(DnsClientError::Wire(_))
        ));
    }
}
