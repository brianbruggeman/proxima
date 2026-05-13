//! `proxima_codec::Datagram` impl for DNS — one UDP packet IS one DNS
//! message ([RFC 1035][1] §4.2.1: a message that doesn't fit in a
//! single datagram is truncated (`TC` bit) and re-sent over TCP, never
//! split across multiple UDP packets), so the trait's atomic-packet
//! contract fits DNS exactly.
//!
//! [`parse_message`] composes the existing [`super::parse_header`] /
//! [`super::parse_question`] / [`super::parse_record`] to validate
//! every question/answer/authority/additional entry once at decode
//! time — a malformed section is a hard [`ParseError`], matching
//! [`Datagram::decode`]'s "whole packet, no `Incomplete`" contract.
//! The walk stores only the four **section start offsets** on
//! [`Message`] (four `usize`s on the stack); it never collects parsed
//! `Question`/`Record` values into a `Vec` — the walk that already
//! regressed 19× when `Name` collected labels eagerly
//! ([`super::Name`]'s own doc comment). [`QuestionIter`] /
//! [`RecordIter`] re-walk from a stored offset on demand, so a caller
//! that only wants `additionals()` still pays zero allocation.
//!
//! [1]: https://www.rfc-editor.org/rfc/rfc1035
//!
//! `Message<'a>` borrows the entire packet buffer (name compression
//! pointers are absolute offsets from the start of the message, so
//! every accessor needs the full buffer, not just the bytes after the
//! header) — which makes [`Datagram::encode`] exact and free: the
//! message never mutates its view, so encoding is copying the
//! borrowed buffer back out, byte for byte.

use alloc::vec::Vec;
use core::net::SocketAddr;

use proxima_codec::{Addressed, Datagram};

use super::{Header, ParseError, Question, Record, parse_header, parse_question, parse_record};

const HEADER_BYTES: usize = 12;

/// Lazily-addressable, zero-copy view of one already-validated DNS
/// message. Holds the decoded [`Header`] plus the message buffer and
/// the four section start offsets computed once by [`parse_message`];
/// [`Self::questions`] / [`Self::answers`] / [`Self::authorities`] /
/// [`Self::additionals`] each build a fresh lazy iterator from a
/// stored offset — no section is materialized until the caller asks
/// for it.
#[derive(Debug, Clone, Copy)]
pub struct Message<'a> {
    pub header: Header,
    buf: &'a [u8],
    questions_offset: usize,
    answers_offset: usize,
    authorities_offset: usize,
    additionals_offset: usize,
}

impl<'a> Message<'a> {
    /// Question section — [`Header::qdcount`] entries starting right
    /// after the 12-byte header.
    #[must_use]
    pub fn questions(&self) -> QuestionIter<'a> {
        QuestionIter {
            buf: self.buf,
            cursor: self.questions_offset,
            remaining: self.header.qdcount,
        }
    }

    /// Answer section — [`Header::ancount`] entries.
    #[must_use]
    pub fn answers(&self) -> RecordIter<'a> {
        RecordIter {
            buf: self.buf,
            cursor: self.answers_offset,
            remaining: self.header.ancount,
        }
    }

    /// Authority section — [`Header::nscount`] entries.
    #[must_use]
    pub fn authorities(&self) -> RecordIter<'a> {
        RecordIter {
            buf: self.buf,
            cursor: self.authorities_offset,
            remaining: self.header.nscount,
        }
    }

    /// Additional section — [`Header::arcount`] entries.
    #[must_use]
    pub fn additionals(&self) -> RecordIter<'a> {
        RecordIter {
            buf: self.buf,
            cursor: self.additionals_offset,
            remaining: self.header.arcount,
        }
    }
}

/// Lazily walks the question section from [`Message::questions`].
/// Stops (and stays stopped) at the first parse error — the buffer
/// was already validated once by [`parse_message`], so an error here
/// signals a bug in the offset bookkeeping, not malformed wire bytes.
pub struct QuestionIter<'a> {
    buf: &'a [u8],
    cursor: usize,
    remaining: u16,
}

impl<'a> Iterator for QuestionIter<'a> {
    type Item = Result<Question<'a>, ParseError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        match parse_question(self.buf, self.cursor) {
            Ok((question, used)) => {
                self.cursor += used;
                self.remaining -= 1;
                Some(Ok(question))
            }
            Err(error) => {
                self.remaining = 0;
                Some(Err(error))
            }
        }
    }
}

/// Lazily walks a record section (answer / authority / additional)
/// from [`Message::answers`] / [`Message::authorities`] /
/// [`Message::additionals`].
pub struct RecordIter<'a> {
    buf: &'a [u8],
    cursor: usize,
    remaining: u16,
}

impl<'a> Iterator for RecordIter<'a> {
    type Item = Result<Record<'a>, ParseError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        match parse_record(self.buf, self.cursor) {
            Ok((record, used)) => {
                self.cursor += used;
                self.remaining -= 1;
                Some(Ok(record))
            }
            Err(error) => {
                self.remaining = 0;
                Some(Err(error))
            }
        }
    }
}

/// Parse and validate one whole DNS message: header plus every
/// question/answer/authority/additional entry, walked once
/// (no `Vec`-collect — see the module doc). Any malformed entry is a
/// hard [`ParseError`], never a partial-message signal — a UDP
/// datagram is delivered atomically, so there is nothing to wait for.
pub fn parse_message(buf: &[u8]) -> Result<Message<'_>, ParseError> {
    let header = parse_header(buf)?;

    let questions_offset = HEADER_BYTES;
    let mut cursor = questions_offset;
    for _ in 0..header.qdcount {
        let (_, used) = parse_question(buf, cursor)?;
        cursor += used;
    }

    let answers_offset = cursor;
    for _ in 0..header.ancount {
        let (_, used) = parse_record(buf, cursor)?;
        cursor += used;
    }

    let authorities_offset = cursor;
    for _ in 0..header.nscount {
        let (_, used) = parse_record(buf, cursor)?;
        cursor += used;
    }

    let additionals_offset = cursor;
    for _ in 0..header.arcount {
        let (_, used) = parse_record(buf, cursor)?;
        cursor += used;
    }

    Ok(Message {
        header,
        buf,
        questions_offset,
        answers_offset,
        authorities_offset,
        additionals_offset,
    })
}

/// [`Datagram`] impl for DNS-over-UDP. Zero-sized; clone freely.
#[derive(Debug, Clone, Copy, Default)]
pub struct DnsDatagramCodec;

impl Datagram for DnsDatagramCodec {
    type Message<'a> = Message<'a>;
    type Error = ParseError;

    fn decode<'a>(
        &self,
        peer: SocketAddr,
        bytes: &'a [u8],
    ) -> Result<Addressed<Message<'a>>, ParseError> {
        let message = parse_message(bytes)?;
        Ok(Addressed { peer, message })
    }

    fn encode(
        &self,
        addressed: &Addressed<Message<'_>>,
        dest: &mut Vec<u8>,
    ) -> Result<(), ParseError> {
        // the message never mutates its view of `buf` — encoding it is
        // exactly the bytes decode borrowed from, copied back out.
        dest.extend_from_slice(addressed.message.buf);
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use core::net::Ipv4Addr;

    fn loopback_peer() -> SocketAddr {
        SocketAddr::from((core::net::Ipv4Addr::LOCALHOST, 53))
    }

    /// Real-shape DNS query for "example.com" A IN, id=1234, rd set —
    /// same fixture the module's own `parse_header`/`parse_question`
    /// tests use.
    fn example_com_query() -> Vec<u8> {
        let mut msg = Vec::new();
        msg.extend_from_slice(&1234u16.to_be_bytes());
        msg.extend_from_slice(&0x0100u16.to_be_bytes());
        msg.extend_from_slice(&1u16.to_be_bytes());
        msg.extend_from_slice(&[0u8; 6]);
        msg.push(7);
        msg.extend_from_slice(b"example");
        msg.push(3);
        msg.extend_from_slice(b"com");
        msg.push(0);
        msg.extend_from_slice(&1u16.to_be_bytes());
        msg.extend_from_slice(&1u16.to_be_bytes());
        msg
    }

    /// Same query, with one A-record answer appended (compressed name
    /// pointing back at the question's qname).
    fn example_com_response() -> Vec<u8> {
        let mut msg = example_com_query();
        msg[2] = 0x81; // response, recursion desired
        msg[3] = 0x80; // recursion available
        msg[6] = 0;
        msg[7] = 1; // ancount = 1
        msg.extend_from_slice(&[0xC0, 0x0C]); // pointer to offset 12
        msg.extend_from_slice(&1u16.to_be_bytes()); // type A
        msg.extend_from_slice(&1u16.to_be_bytes()); // class IN
        msg.extend_from_slice(&300u32.to_be_bytes()); // ttl
        msg.extend_from_slice(&4u16.to_be_bytes()); // rdlength
        msg.extend_from_slice(&[93, 184, 216, 34]);
        msg
    }

    #[test]
    fn decode_exposes_header_and_lazy_question_section() {
        let codec = DnsDatagramCodec;
        let peer = loopback_peer();
        let packet = example_com_query();

        let addressed = codec.decode(peer, &packet).expect("decode should succeed");
        assert_eq!(addressed.peer, peer);
        assert_eq!(addressed.message.header.id, 1234);
        assert_eq!(addressed.message.header.qdcount, 1);

        let questions: Vec<_> = addressed
            .message
            .questions()
            .map(|result| result.expect("question parses"))
            .collect();
        assert_eq!(questions.len(), 1);
        assert_eq!(questions[0].name.to_dotted(), "example.com.");
        assert_eq!(questions[0].qtype, 1);

        assert_eq!(addressed.message.answers().count(), 0);
    }

    #[test]
    fn decode_exposes_lazy_answer_section_after_questions() {
        let codec = DnsDatagramCodec;
        let peer = loopback_peer();
        let packet = example_com_response();

        let addressed = codec.decode(peer, &packet).expect("decode should succeed");
        let answers: Vec<_> = addressed
            .message
            .answers()
            .map(|result| result.expect("answer parses"))
            .collect();
        assert_eq!(answers.len(), 1);
        assert_eq!(answers[0].name.to_dotted(), "example.com.");
        assert_eq!(
            answers[0].rdata,
            super::super::RData::A(Ipv4Addr::new(93, 184, 216, 34))
        );
    }

    #[test]
    fn encode_reproduces_the_exact_decoded_bytes() {
        let codec = DnsDatagramCodec;
        let peer = loopback_peer();
        let packet = example_com_response();

        let addressed = codec.decode(peer, &packet).expect("decode should succeed");
        let mut encoded = Vec::new();
        codec
            .encode(&addressed, &mut encoded)
            .expect("encode should succeed");
        assert_eq!(encoded, packet);
    }

    #[test]
    fn truncated_header_is_hard_error_not_incomplete() {
        // a real recvfrom() never hands the codec a short buffer to
        // "read more" from — the kernel already delivered the whole
        // datagram. ParseError has no Incomplete/retry variant.
        let codec = DnsDatagramCodec;
        let outcome = codec.decode(loopback_peer(), &[0u8; 5]);
        assert_eq!(outcome.unwrap_err(), ParseError::Short);
    }

    #[test]
    fn declared_question_count_beyond_buffer_is_hard_error() {
        // qdcount says 1 but the buffer ends right after the header —
        // the whole-packet validation walk must catch this at decode,
        // not defer it to whichever accessor a caller happens to use.
        let mut packet = Vec::new();
        packet.extend_from_slice(&1u16.to_be_bytes());
        packet.extend_from_slice(&0u16.to_be_bytes());
        packet.extend_from_slice(&1u16.to_be_bytes()); // qdcount = 1
        packet.extend_from_slice(&[0u8; 6]);

        let codec = DnsDatagramCodec;
        let outcome = codec.decode(loopback_peer(), &packet);
        assert!(matches!(outcome, Err(ParseError::Short)));
    }
}
