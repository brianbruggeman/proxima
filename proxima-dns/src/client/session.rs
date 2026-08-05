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
use crate::pipes::DnsAnswer;
use crate::wire::message_to_answer;

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
    let message = parse_message(bytes).map_err(|error| DnsClientError::Wire(error.to_string()))?;
    if message.header.id != expected_id {
        return Err(DnsClientError::IdMismatch {
            expected: expected_id,
            reply: message.header.id,
        });
    }
    message_to_answer(&message)
        .ok_or_else(|| DnsClientError::Wire("response answer record failed to decode".to_string()))
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

        let answer = decode_response(id, &response).unwrap();
        assert_eq!(answer.rcode, 0);
        assert_eq!(answer.records.len(), 1);
        assert_eq!(answer.records[0].name, "example.com.");
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
}
