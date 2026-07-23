//! `DnsClientSession` — the sans-IO half of the resolver client. Bytes in,
//! bytes out; no socket (workspace principle 11), mirroring
//! `proxima_redis::client::session::ClientSession`'s split from the async
//! transport driver in [`super::pipe`].
//!
//! Unlike `ClientSession` (a multi-step RESP handshake with real states to
//! discriminate), one DNS query/response exchange has no intermediate
//! states — it is a single request, a single reply, matched by the 16-bit
//! query id RFC 1035 §4.1.1 already carries. The state this session holds
//! is exactly that: a monotonically-advancing id counter, so two queries
//! issued back-to-back on the same session never collide in flight. There
//! is no enum FSM here because there is nothing to discriminate between —
//! see [`crate`]'s module doc for why a genuinely stateless exchange stays
//! a plain counter rather than manufacturing states that don't exist.

use proxima_protocols::dns::codec_trait::parse_message;
use proxima_protocols::dns::encode::{self, EncodeQuestion};

use crate::error::DnsClientError;
use crate::pipes::DnsAnswer;
use crate::wire::message_to_answer;

/// Sans-IO DNS client session: builds query bytes, decodes response bytes,
/// tracks the next query id. Holds no transport of its own — [`super::pipe::DnsClientUpstream`]
/// drives it over a real socket.
#[derive(Debug, Default)]
pub struct DnsClientSession {
    next_id: u16,
}

impl DnsClientSession {
    #[must_use]
    pub fn new() -> Self {
        // Starting above zero is purely cosmetic (id 0 is a legal query id,
        // RFC 1035 places no restriction on it) — avoids every fresh
        // session's first query looking suspiciously like an uninitialized
        // field in a packet capture.
        Self { next_id: 1 }
    }

    /// Build query wire bytes for one question, returning the id the reply
    /// must echo (RFC 1035 §4.1.1) so the caller can match it.
    pub fn encode_query(
        &mut self,
        name: &str,
        qtype: u16,
        qclass: u16,
        recursion_desired: bool,
    ) -> Result<(u16, Vec<u8>), DnsClientError> {
        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1);
        let mut out = Vec::new();
        encode::encode_query(id, recursion_desired, EncodeQuestion { name, qtype, qclass }, &mut out)
            .map_err(|error| DnsClientError::Wire(error.to_string()))?;
        Ok((id, out))
    }

    /// Decode a reply, verifying its id matches the query it's answering.
    pub fn decode_response(&self, expected_id: u16, bytes: &[u8]) -> Result<DnsAnswer, DnsClientError> {
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
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use proxima_protocols::dns::encode::ipv4_rdata;

    #[test]
    fn successive_queries_get_distinct_ids() {
        let mut session = DnsClientSession::new();
        let (first_id, _) = session.encode_query("a.example.", 1, 1, true).unwrap();
        let (second_id, _) = session.encode_query("b.example.", 1, 1, true).unwrap();
        assert_ne!(first_id, second_id);
    }

    #[test]
    fn encode_query_round_trips_through_the_listener_side_wire_helper() {
        let mut session = DnsClientSession::new();
        let (id, query_bytes) = session.encode_query("example.com.", 1, 1, true).unwrap();

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
        drop(query_bytes);

        let answer = session.decode_response(id, &response).unwrap();
        assert_eq!(answer.rcode, 0);
        assert_eq!(answer.records.len(), 1);
        assert_eq!(answer.records[0].name, "example.com.");
    }

    #[test]
    fn decode_response_rejects_a_mismatched_id() {
        let session = DnsClientSession::new();
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

        let outcome = session.decode_response(1, &response);
        assert!(matches!(
            outcome,
            Err(DnsClientError::IdMismatch {
                expected: 1,
                reply: 999
            })
        ));
    }
}
