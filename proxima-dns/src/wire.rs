//! Wire <-> typed conversions shared by the listener
//! ([`crate::datagram_protocol`], [`crate::any_protocol`]) and the client
//! ([`crate::client::session`]). Both sides decode with
//! [`proxima_protocols::dns::codec_trait::parse_message`] (zero-copy,
//! borrowed) and hand callers an owned [`crate::pipes::DnsQuery`] /
//! [`crate::pipes::DnsAnswer`] — the boundary where borrowed wire views
//! become plain, `'static` business data (see [`crate::pipes`]'s module
//! doc for why).

#[cfg(feature = "listen")]
use std::net::SocketAddr;

#[cfg(feature = "listen")]
use proxima_codec::Datagram;
#[cfg(feature = "listen")]
use proxima_primitives::pipe::SendPipe;
#[cfg(feature = "client")]
use proxima_protocols::dns::RData;
#[cfg(feature = "listen")]
use proxima_protocols::dns::codec_trait::DnsDatagramCodec;
use proxima_protocols::dns::codec_trait::Message;
use proxima_protocols::dns::encode;
#[cfg(feature = "listen")]
use proxima_telemetry::warn;

use crate::pipes::DnsAnswer;
#[cfg(feature = "client")]
use crate::pipes::{DnsAnswerMetadata, DnsAnswerRecord, DnsAnswerWithMetadata};
#[cfg(feature = "listen")]
use crate::pipes::{DnsPipeHandle, DnsQuery, UDP_TRANSPORT, build_request};

/// Decode a query out of an already-parsed [`Message`]. `None` for anything
/// other than exactly one question (RFC 1035 §4.1.2 permits more; no
/// deployed client sends more than one and no deployed server answers more
/// than one — see [`crate::pipes`]'s module doc) or a question whose name
/// walk fails (a corrupt compression pointer past the header).
/// Listener-side only — the client never receives a query to decode.
#[cfg(feature = "listen")]
pub(crate) fn message_to_query(message: &Message<'_>) -> Option<DnsQuery> {
    if message.header.qdcount != 1 {
        return None;
    }
    let question = message.questions().next()?.ok()?;
    Some(DnsQuery {
        id: message.header.id,
        recursion_desired: message.header.flags.rd(),
        name: question.name.to_dotted(),
        qtype: question.qtype,
        qclass: question.qclass,
    })
}

/// Decode a resolver reply out of an already-parsed [`Message`]: the header
/// flags plus every answer-section record, each re-flattened to raw wire
/// `rdata` bytes via [`rdata_to_bytes`] (undoing the parser's typed A/AAAA/
/// CNAME/NS/PTR decode) so [`DnsAnswerRecord`] stays one flat shape
/// regardless of record type — the same "typed on the wire, owned bytes at
/// the pipe boundary" split [`message_to_query`] uses. `None` only when an
/// answer record's own name walk fails; a query with zero answers (NODATA)
/// is a legal, `Some` result with an empty `records` list. Client-side only
/// — the listener never receives a resolver reply to decode.
#[cfg(feature = "client")]
pub(crate) fn message_to_answer(message: &Message<'_>) -> Option<DnsAnswer> {
    let flags = message.header.flags;
    let mut records = Vec::with_capacity(usize::from(message.header.ancount));
    for answer in message.answers() {
        let record = answer.ok()?;
        records.push(DnsAnswerRecord {
            name: record.name.to_dotted(),
            rtype: record.rtype,
            rclass: record.rclass,
            ttl: record.ttl,
            rdata: rdata_to_bytes(&record.rdata),
        });
    }
    Some(DnsAnswer {
        rcode: flags.rcode(),
        authoritative: flags.aa(),
        recursion_available: flags.ra(),
        records,
    })
}

/// Decode the answer and the response-envelope fields needed by an upstream
/// to validate that it answered the requested question. A response must echo
/// exactly one question; malformed or multi-question responses are rejected.
#[cfg(feature = "client")]
pub(crate) fn message_to_answer_with_metadata(
    message: &Message<'_>,
) -> Option<DnsAnswerWithMetadata> {
    if message.header.qdcount != 1 {
        return None;
    }
    let question = message.questions().next()?.ok()?;
    let query = DnsQuery {
        id: message.header.id,
        recursion_desired: message.header.flags.rd(),
        name: question.name.to_dotted(),
        qtype: question.qtype,
        qclass: question.qclass,
    };
    Some(DnsAnswerWithMetadata {
        answer: message_to_answer(message)?,
        metadata: DnsAnswerMetadata {
            question: Some(query),
            truncated: message.header.flags.tc(),
        },
    })
}

/// Re-flatten a parsed [`RData`] back into raw wire bytes — the inverse of
/// `super::parse_record`'s per-type decode. `Cname`/`Ns`/`Ptr` re-encode the
/// target name uncompressed (see `proxima_protocols::dns::encode`'s module
/// doc on why encode never compresses); `Raw` is already raw bytes.
#[cfg(feature = "client")]
fn rdata_to_bytes(rdata: &RData<'_>) -> Vec<u8> {
    match rdata {
        RData::A(addr) => encode::ipv4_rdata(*addr).to_vec(),
        RData::Aaaa(addr) => encode::ipv6_rdata(*addr).to_vec(),
        RData::Cname(name) | RData::Ns(name) | RData::Ptr(name) => {
            let mut out = Vec::new();
            // A name inside an already-validated parsed message always
            // encodes successfully — `encode_name`'s only failure modes
            // (label/name length limits) are wire-format invariants the
            // parser already enforced to get here.
            let _ = encode::encode_name(&name.to_dotted(), &mut out);
            out
        }
        RData::Raw { bytes, .. } => bytes.to_vec(),
    }
}

/// Build response wire bytes answering `query` with `answer` — the
/// listener side's write path, composing
/// [`proxima_protocols::dns::encode::encode_response`] with the flags
/// [`crate::pipes::DnsAnswer`] carries. Listener-side only.
#[cfg(feature = "listen")]
pub(crate) fn answer_to_wire(
    query: &DnsQuery,
    answer: &DnsAnswer,
    out: &mut Vec<u8>,
) -> Result<(), encode::EncodeError> {
    let flags = proxima_protocols::dns::Flags::for_response(
        query.recursion_desired,
        answer.authoritative,
        answer.recursion_available,
        answer.rcode,
    );
    let question = encode::EncodeQuestion {
        name: &query.name,
        qtype: query.qtype,
        qclass: query.qclass,
    };
    let records: Vec<encode::AnswerRecord<'_>> = answer
        .records
        .iter()
        .map(|record| encode::AnswerRecord {
            name: &record.name,
            rtype: record.rtype,
            rclass: record.rclass,
            ttl: record.ttl,
            rdata: &record.rdata,
        })
        .collect();
    encode::encode_response(query.id, flags, question, &records, out)
}

/// Does this fixed 12-byte header (RFC 1035 §4.1.1) open a query this crate
/// serves? Both `.any()` candidates classify on it, over the raw header for
/// UDP and past the 2-byte length prefix for TCP.
///
/// DNS carries no magic number and query ids are arbitrary — the only signal
/// available to a positive-match probe is that the header's own reserved and
/// count fields agree with a query: `QR == 0` (not a response), `OPCODE == 0`
/// (a standard query, not NOTIFY/UPDATE), and `QDCOUNT == 1` (this crate's
/// single-question contract, see [`crate::pipes`]'s module doc). Checking all
/// three rather than `QDCOUNT` alone is what keeps a stray DNS *reply*, or
/// another protocol's traffic, from claiming the connection.
#[cfg(feature = "listen")]
pub(crate) fn header_is_query(header: &[u8]) -> bool {
    const QR_RESPONSE: u8 = 0b1000_0000;
    const OPCODE_MASK: u8 = 0b0111_1000;

    let Some(flags_high) = header.get(2) else {
        return false;
    };
    let (Some(qdcount_high), Some(qdcount_low)) = (header.get(4), header.get(5)) else {
        return false;
    };
    flags_high & QR_RESPONSE == 0
        && flags_high & OPCODE_MASK == 0
        && u16::from_be_bytes([*qdcount_high, *qdcount_low]) == 1
}

/// The whole DNS-over-UDP unit of work: cap-check, decode, dispatch to
/// `handler`, encode the reply. `None` is every drop case — an oversized,
/// unparseable, or multi-question query, a handler failure, an unencodable
/// answer — each warned where the decision is made, because one bad query
/// must never tear down a connectionless listener.
///
/// Both UDP listeners share this: [`crate::datagram_protocol`]'s dedicated
/// `DatagramProtocol` state machine and [`crate::udp_any_protocol`]'s
/// `.any()`-fanned candidate differ only in where the bytes come from and
/// where the reply goes, never in what a query means.
#[cfg(feature = "listen")]
pub(crate) async fn answer_datagram(
    label: &str,
    handler: &DnsPipeHandle,
    max_message_bytes: usize,
    peer: SocketAddr,
    datagram: &[u8],
) -> Option<Vec<u8>> {
    if datagram.len() > max_message_bytes {
        warn!(
            %label,
            %peer,
            len = datagram.len(),
            limit = max_message_bytes,
            "dns query exceeds message limit; dropping"
        );
        return None;
    }
    let addressed = match DnsDatagramCodec.decode(peer, datagram) {
        Ok(addressed) => addressed,
        Err(error) => {
            warn!(%label, %peer, ?error, "dns query failed to parse; dropping");
            return None;
        }
    };
    let Some(query) = message_to_query(&addressed.message) else {
        warn!(%label, %peer, "dns query is not exactly one question; dropping");
        return None;
    };

    let request = build_request(UDP_TRANSPORT, query.clone());
    let reply = match SendPipe::call(handler, request).await {
        Ok(reply) => reply,
        Err(error) => {
            warn!(%label, %peer, ?error, "dns handler pipe failed; dropping");
            return None;
        }
    };

    // 204 is the DNS listener's explicit no-reply sentinel. It lets a
    // policy pipe implement a true sinkhole/ignore action without inventing
    // a DNS RCODE (all RCODEs are observable replies to a resolver).
    if reply.status == 204 {
        return None;
    }
    let mut out = Vec::new();
    if let Err(error) = answer_to_wire(&query, &reply.payload, &mut out) {
        warn!(%label, %peer, ?error, "dns answer failed to encode; dropping");
        return None;
    }
    Some(out)
}

// The round-trip tests below exercise both halves (listener-side encode,
// client-side decode) against each other, so they only compile/run when
// both features are enabled — `cargo nextest run -p proxima-dns --features
// client,listen`.
#[cfg(all(test, feature = "client", feature = "listen"))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::pipes::{DnsAnswerRecord, DnsQuery};
    use core::net::Ipv4Addr;
    use proxima_protocols::dns::codec_trait::parse_message;

    /// Real-shape query wire bytes: "example.com" A IN, id=1234, rd set.
    fn example_com_query_bytes() -> Vec<u8> {
        let mut out = Vec::new();
        encode::encode_query(
            1234,
            true,
            encode::EncodeQuestion {
                name: "example.com.",
                qtype: 1,
                qclass: 1,
            },
            &mut out,
        )
        .unwrap();
        out
    }

    #[test]
    fn message_to_query_extracts_the_single_question() {
        let bytes = example_com_query_bytes();
        let message = parse_message(&bytes).unwrap();
        let query = message_to_query(&message).unwrap();
        assert_eq!(query.id, 1234);
        assert!(query.recursion_desired);
        assert_eq!(query.name, "example.com.");
        assert_eq!(query.qtype, 1);
        assert_eq!(query.qclass, 1);
    }

    #[test]
    fn message_to_query_rejects_zero_questions() {
        // header only, qdcount=0 — legal wire bytes, just not the shape a
        // resolver server or client ever handles.
        let mut bytes = vec![0u8; 12];
        bytes[0..2].copy_from_slice(&7u16.to_be_bytes());
        let message = parse_message(&bytes).unwrap();
        assert!(message_to_query(&message).is_none());
    }

    #[test]
    fn answer_to_wire_round_trips_through_the_parser() {
        let query = DnsQuery {
            id: 1234,
            recursion_desired: true,
            name: "example.com.".to_string(),
            qtype: 1,
            qclass: 1,
        };
        let answer = DnsAnswer::ok(vec![DnsAnswerRecord {
            name: "example.com.".to_string(),
            rtype: 1,
            rclass: 1,
            ttl: 300,
            rdata: encode::ipv4_rdata(Ipv4Addr::new(93, 184, 216, 34)).to_vec(),
        }]);

        let mut out = Vec::new();
        answer_to_wire(&query, &answer, &mut out).unwrap();

        let message = parse_message(&out).unwrap();
        assert_eq!(message.header.id, 1234);
        assert!(message.header.flags.is_response());
        assert!(message.header.flags.rd());
        assert!(message.header.flags.ra());
        assert_eq!(message.header.ancount, 1);

        let decoded = message_to_answer(&message).unwrap();
        assert_eq!(decoded.rcode, 0);
        assert!(decoded.recursion_available);
        assert_eq!(decoded.records.len(), 1);
        assert_eq!(decoded.records[0].name, "example.com.");
        assert_eq!(
            decoded.records[0].rdata,
            encode::ipv4_rdata(Ipv4Addr::new(93, 184, 216, 34)).to_vec()
        );
    }

    #[test]
    fn message_to_answer_reads_nxdomain_with_no_records() {
        let query = DnsQuery {
            id: 42,
            recursion_desired: true,
            name: "nonexistent.example.".to_string(),
            qtype: 1,
            qclass: 1,
        };
        let answer = DnsAnswer::name_error();
        let mut out = Vec::new();
        answer_to_wire(&query, &answer, &mut out).unwrap();

        let message = parse_message(&out).unwrap();
        let decoded = message_to_answer(&message).unwrap();
        assert_eq!(decoded.rcode, 3);
        assert!(decoded.records.is_empty());
    }
}
