//! DNS substrate — sans-IO RFC 1035 / RFC 3596 message parser.
//!
//! Tracked as **P1** in `docs/protocol-gap/discipline.md`. Like
//! `proxy_protocol` and `listeners::redis`, the parser is the
//! substrate piece; an actual resolver / authoritative-server
//! listener wires accept loops + answer composition on top.
//!
//! Coverage today (first slice):
//!
//! - Header (12-byte fixed prefix): id, flags, qdcount, ancount,
//!   nscount, arcount.
//! - Question section: name + qtype + qclass.
//! - Answer / Authority / Additional records: name + type + class +
//!   ttl + rdlength + rdata. rdata decoded for the common types:
//!   - A (RFC 1035 §3.4.1) — IPv4 (4 bytes)
//!   - AAAA (RFC 3596) — IPv6 (16 bytes)
//!   - CNAME / NS / PTR (RFC 1035) — compressed domain name
//!   - Other types are kept as raw `&[u8]` rdata for the caller.
//!
//! Name compression (RFC 1035 §4.1.4) handled — pointer offsets
//! refer back into the message buffer; the parser follows them
//! up to a hard depth limit to avoid loop attacks.
//!
//! Folded from the former `proxima-dns` crate into `proxima-protocols` as
//! the `dns` module. The umbrella `proxima` re-exports
//! `proxima_protocols::dns` behind the `dns-substrate` feature for
//! existing call sites.


use alloc::string::String;
use core::net::{Ipv4Addr, Ipv6Addr};

/// Fixed-size 12-byte DNS header prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    pub id: u16,
    pub flags: Flags,
    pub qdcount: u16,
    pub ancount: u16,
    pub nscount: u16,
    pub arcount: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Flags(pub u16);

impl Flags {
    /// QR — 0 = query, 1 = response (bit 15).
    pub fn is_response(self) -> bool {
        self.0 & 0x8000 != 0
    }
    /// OPCODE bits (11-14).
    pub fn opcode(self) -> u8 {
        ((self.0 >> 11) & 0x0F) as u8
    }
    /// RCODE bits (0-3).
    pub fn rcode(self) -> u8 {
        (self.0 & 0x0F) as u8
    }
    /// AA bit (10) — authoritative answer.
    pub fn aa(self) -> bool {
        self.0 & 0x0400 != 0
    }
    /// TC bit (9) — truncated.
    pub fn tc(self) -> bool {
        self.0 & 0x0200 != 0
    }
    /// RD bit (8) — recursion desired.
    pub fn rd(self) -> bool {
        self.0 & 0x0100 != 0
    }
    /// RA bit (7) — recursion available.
    pub fn ra(self) -> bool {
        self.0 & 0x0080 != 0
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Question<'a> {
    pub name: Name<'a>,
    pub qtype: u16,
    pub qclass: u16,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Record<'a> {
    pub name: Name<'a>,
    pub rtype: u16,
    pub rclass: u16,
    pub ttl: u32,
    pub rdata: RData<'a>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RData<'a> {
    A(Ipv4Addr),
    Aaaa(Ipv6Addr),
    Cname(Name<'a>),
    Ns(Name<'a>),
    Ptr(Name<'a>),
    /// Unhandled type — raw rdata bytes for the caller to decode.
    Raw {
        rtype: u16,
        bytes: &'a [u8],
    },
}

/// Domain name parsed out of a DNS message. Labels stored as
/// `&[u8]` slices into the message buffer (after following any
/// compression pointers). Equality compares label sequences,
/// case-insensitive per RFC 1035 §2.3.3 — DNS is case-preserving
/// but case-insensitive.
/// Lazy domain name — stores the start offset into the original
/// message buffer and the on-wire encoded length, without
/// materializing labels. Callers iterate labels on demand via
/// [`Name::labels`]; equality / dotted rendering walks the bytes.
///
/// Iteration follows RFC 1035 compression pointers transparently.
/// Pointer-chain depth is capped at 32 hops by [`parse_name`]; the
/// returned `Name` is therefore safe to iterate without re-checking.
///
/// Bench history: original `Vec<&[u8]>` was 19× slower than the
/// parity baseline; `SmallVec<[&[u8]; 8]>` halved that to 10×;
/// lazy `Name` is the gate-passing tweak — no per-name allocation.
#[derive(Debug, Clone, Copy)]
pub struct Name<'a> {
    message: &'a [u8],
    start: usize,
    encoded_len: usize,
}

impl<'a> Name<'a> {
    /// Iterate labels in order. Follows compression pointers.
    #[must_use]
    pub fn labels(&self) -> LabelIter<'a> {
        LabelIter {
            message: self.message,
            cursor: self.start,
            depth: 0,
        }
    }

    /// Render as dotted ASCII (`example.com.`). Trailing dot
    /// included to disambiguate fully-qualified from relative.
    #[must_use]
    pub fn to_dotted(&self) -> String {
        let mut out = String::new();
        for label in self.labels() {
            out.push_str(&String::from_utf8_lossy(label));
            out.push('.');
        }
        if out.is_empty() {
            out.push('.');
        }
        out
    }

    /// Total bytes the name occupies in the message at its start
    /// offset (NOT including bytes followed via compression
    /// pointers). Used by section parsers to compute the next
    /// element's offset.
    #[must_use]
    pub fn encoded_len(&self) -> usize {
        self.encoded_len
    }
}

/// Case-insensitive label-by-label comparison per RFC 1035 §2.3.3.
impl PartialEq for Name<'_> {
    fn eq(&self, other: &Self) -> bool {
        let mut left = self.labels();
        let mut right = other.labels();
        loop {
            match (left.next(), right.next()) {
                (None, None) => return true,
                (Some(a), Some(b)) => {
                    if !a.eq_ignore_ascii_case(b) {
                        return false;
                    }
                }
                _ => return false,
            }
        }
    }
}

impl Eq for Name<'_> {}

/// Walks labels of a [`Name`] on demand. Follows compression
/// pointers up to a guard depth set by [`parse_name`].
pub struct LabelIter<'a> {
    message: &'a [u8],
    cursor: usize,
    depth: u8,
}

impl<'a> Iterator for LabelIter<'a> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let byte = *self.message.get(self.cursor)?;
            match byte & 0xC0 {
                0x00 => {
                    let len = byte as usize;
                    self.cursor += 1;
                    if len == 0 {
                        return None;
                    }
                    let end = self.cursor + len;
                    if end > self.message.len() {
                        return None;
                    }
                    let label = &self.message[self.cursor..end];
                    self.cursor = end;
                    return Some(label);
                }
                0xC0 => {
                    let high = byte as usize & 0x3F;
                    let low = *self.message.get(self.cursor + 1)? as usize;
                    let pointer = (high << 8) | low;
                    if pointer >= self.message.len() {
                        return None;
                    }
                    self.cursor = pointer;
                    self.depth = self.depth.saturating_add(1);
                    if self.depth > 32 {
                        return None;
                    }
                }
                _ => return None,
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseError {
    /// Buffer too short to complete the structure.
    Short,
    /// Wire byte sequence violates RFC 1035 framing.
    Malformed(&'static str),
}

impl core::fmt::Display for ParseError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Short => write!(formatter, "buffer too short to complete the structure"),
            Self::Malformed(reason) => write!(formatter, "malformed dns message: {reason}"),
        }
    }
}

impl core::error::Error for ParseError {}

/// Parse a DNS message header. The 12-byte fixed prefix is the
/// substrate's foundation; everything else (sections) parses
/// relative to the same buffer with name-compression pointers.
#[inline]
pub fn parse_header(buf: &[u8]) -> Result<Header, ParseError> {
    if buf.len() < 12 {
        return Err(ParseError::Short);
    }
    Ok(Header {
        id: u16::from_be_bytes([buf[0], buf[1]]),
        flags: Flags(u16::from_be_bytes([buf[2], buf[3]])),
        qdcount: u16::from_be_bytes([buf[4], buf[5]]),
        ancount: u16::from_be_bytes([buf[6], buf[7]]),
        nscount: u16::from_be_bytes([buf[8], buf[9]]),
        arcount: u16::from_be_bytes([buf[10], buf[11]]),
    })
}

/// Decode a domain name starting at `offset` in `message`. Returns
/// a [`Name`] (lazy — no label collection) and the number of bytes
/// consumed at the starting position (NOT including bytes followed
/// via compression pointers). RFC 1035 §4.1.4: a name terminates at
/// a 0-length label OR at a pointer.
///
/// This function only validates byte boundaries — it does NOT walk
/// every label. The returned `Name` walks lazily, but bounds-checks
/// each step. Pointer-chain depth gets re-checked on iteration
/// (32-hop cap) so an iterator-side caller is also safe.
#[inline]
pub fn parse_name<'a>(message: &'a [u8], offset: usize) -> Result<(Name<'a>, usize), ParseError> {
    let mut cursor = offset;
    let mut consumed_at_start: Option<usize> = None;
    let mut follow_depth: u8 = 0;
    loop {
        let byte = *message.get(cursor).ok_or(ParseError::Short)?;
        match byte & 0xC0 {
            0x00 => {
                let len = byte as usize;
                cursor += 1;
                if len == 0 {
                    // `cursor - offset` is only valid when we never
                    // followed a pointer (otherwise cursor < offset
                    // after the pointer hop). use the consumed_at_start
                    // captured when we hit the first pointer if it's set.
                    let used = match consumed_at_start {
                        Some(used) => used,
                        None => cursor - offset,
                    };
                    return Ok((
                        Name {
                            message,
                            start: offset,
                            encoded_len: used,
                        },
                        used,
                    ));
                }
                let end = cursor + len;
                if end > message.len() {
                    return Err(ParseError::Short);
                }
                cursor = end;
            }
            0xC0 => {
                let high = byte as usize & 0x3F;
                let low = *message.get(cursor + 1).ok_or(ParseError::Short)? as usize;
                let pointer = (high << 8) | low;
                if pointer >= message.len() {
                    return Err(ParseError::Malformed("name pointer out of range"));
                }
                if consumed_at_start.is_none() {
                    consumed_at_start = Some(cursor + 2 - offset);
                }
                cursor = pointer;
                follow_depth += 1;
                if follow_depth > 32 {
                    return Err(ParseError::Malformed("name pointer chain too deep"));
                }
            }
            _ => {
                return Err(ParseError::Malformed("reserved label type"));
            }
        }
    }
}

/// Parse a single question section entry at `offset`. Returns the
/// question and the number of bytes consumed.
#[inline]
pub fn parse_question<'a>(
    message: &'a [u8],
    offset: usize,
) -> Result<(Question<'a>, usize), ParseError> {
    let (name, name_used) = parse_name(message, offset)?;
    let after_name = offset + name_used;
    if message.len() < after_name + 4 {
        return Err(ParseError::Short);
    }
    let qtype = u16::from_be_bytes([message[after_name], message[after_name + 1]]);
    let qclass = u16::from_be_bytes([message[after_name + 2], message[after_name + 3]]);
    Ok((
        Question {
            name,
            qtype,
            qclass,
        },
        name_used + 4,
    ))
}

/// Parse a single resource record at `offset`. RData decoded for
/// A / AAAA / CNAME / NS / PTR; everything else is kept as Raw.
#[inline]
pub fn parse_record<'a>(
    message: &'a [u8],
    offset: usize,
) -> Result<(Record<'a>, usize), ParseError> {
    let (name, name_used) = parse_name(message, offset)?;
    let after_name = offset + name_used;
    if message.len() < after_name + 10 {
        return Err(ParseError::Short);
    }
    let rtype = u16::from_be_bytes([message[after_name], message[after_name + 1]]);
    let rclass = u16::from_be_bytes([message[after_name + 2], message[after_name + 3]]);
    let ttl = u32::from_be_bytes([
        message[after_name + 4],
        message[after_name + 5],
        message[after_name + 6],
        message[after_name + 7],
    ]);
    let rdlength = u16::from_be_bytes([message[after_name + 8], message[after_name + 9]]) as usize;
    let rdata_start = after_name + 10;
    let rdata_end = rdata_start + rdlength;
    if message.len() < rdata_end {
        return Err(ParseError::Short);
    }
    let rdata_bytes = &message[rdata_start..rdata_end];
    let rdata = match rtype {
        1 if rdlength == 4 => RData::A(Ipv4Addr::new(
            rdata_bytes[0],
            rdata_bytes[1],
            rdata_bytes[2],
            rdata_bytes[3],
        )),
        28 if rdlength == 16 => {
            let mut bytes = [0u8; 16];
            bytes.copy_from_slice(rdata_bytes);
            RData::Aaaa(Ipv6Addr::from(bytes))
        }
        // CNAME=5, NS=2, PTR=12 — all carry a compressed domain name.
        5 | 2 | 12 => {
            let (target, _) = parse_name(message, rdata_start)?;
            match rtype {
                5 => RData::Cname(target),
                2 => RData::Ns(target),
                12 => RData::Ptr(target),
                _ => unreachable!(),
            }
        }
        _ => RData::Raw {
            rtype,
            bytes: rdata_bytes,
        },
    };
    Ok((
        Record {
            name,
            rtype,
            rclass,
            ttl,
            rdata,
        },
        name_used + 10 + rdlength,
    ))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    /// Build a minimal valid DNS query for "example.com" A IN. id=1234,
    /// rd flag set.
    fn example_com_query() -> Vec<u8> {
        let mut msg = Vec::new();
        msg.extend_from_slice(&1234u16.to_be_bytes()); // id
        msg.extend_from_slice(&0x0100u16.to_be_bytes()); // flags: rd
        msg.extend_from_slice(&1u16.to_be_bytes()); // qdcount
        msg.extend_from_slice(&[0u8; 6]); // ancount=0, nscount=0, arcount=0
        // name: "example" "com" 0
        msg.push(7);
        msg.extend_from_slice(b"example");
        msg.push(3);
        msg.extend_from_slice(b"com");
        msg.push(0);
        msg.extend_from_slice(&1u16.to_be_bytes()); // qtype A
        msg.extend_from_slice(&1u16.to_be_bytes()); // qclass IN
        msg
    }

    #[test]
    fn parses_header_fields() {
        let msg = example_com_query();
        let header = parse_header(&msg).unwrap();
        assert_eq!(header.id, 1234);
        assert!(!header.flags.is_response());
        assert!(header.flags.rd());
        assert_eq!(header.qdcount, 1);
        assert_eq!(header.ancount, 0);
    }

    #[test]
    fn header_short_buffer_errors() {
        assert_eq!(parse_header(&[0u8; 5]), Err(ParseError::Short));
    }

    #[test]
    fn parses_uncompressed_name() {
        let msg = example_com_query();
        let (name, used) = parse_name(&msg, 12).unwrap();
        assert_eq!(name.to_dotted(), "example.com.");
        // 1 (len) + 7 ("example") + 1 (len) + 3 ("com") + 1 (root)
        assert_eq!(used, 13);
    }

    #[test]
    fn parses_question() {
        let msg = example_com_query();
        let (question, used) = parse_question(&msg, 12).unwrap();
        assert_eq!(question.name.to_dotted(), "example.com.");
        assert_eq!(question.qtype, 1);
        assert_eq!(question.qclass, 1);
        assert_eq!(used, 17); // 13 (name) + 4 (qtype+qclass)
    }

    #[test]
    fn parses_a_record() {
        // Construct a minimal DNS response with one A record.
        let mut msg = example_com_query();
        msg[6] = 0;
        msg[7] = 1; // ancount = 1
        // Answer: pointer to offset 12 (the qname), type A, class IN,
        // ttl 300, rdlength 4, rdata 93.184.216.34
        msg.extend_from_slice(&[0xC0, 0x0C]); // pointer to offset 12
        msg.extend_from_slice(&1u16.to_be_bytes()); // type A
        msg.extend_from_slice(&1u16.to_be_bytes()); // class IN
        msg.extend_from_slice(&300u32.to_be_bytes()); // ttl
        msg.extend_from_slice(&4u16.to_be_bytes()); // rdlength
        msg.extend_from_slice(&[93, 184, 216, 34]); // rdata

        let after_question = 12 + 17;
        let (record, _) = parse_record(&msg, after_question).unwrap();
        assert_eq!(record.name.to_dotted(), "example.com.");
        assert_eq!(record.rtype, 1);
        assert_eq!(record.rdata, RData::A(Ipv4Addr::new(93, 184, 216, 34)));
        assert_eq!(record.ttl, 300);
    }

    #[test]
    fn rejects_pointer_loop() {
        // Two-byte name at offset 0 that points to itself.
        let msg = vec![0xC0, 0x00];
        let outcome = parse_name(&msg, 0);
        assert!(matches!(outcome, Err(ParseError::Malformed(_))));
    }

    #[test]
    fn rejects_reserved_label_type() {
        // Label byte with reserved high bits 0b10.
        let msg = vec![0x80, 0x00];
        let outcome = parse_name(&msg, 0);
        assert!(matches!(outcome, Err(ParseError::Malformed(_))));
    }
}

#[cfg(feature = "dns-codec-trait")]
pub mod codec_trait;
#[cfg(feature = "dns-codec-trait")]
pub use codec_trait::{DnsDatagramCodec, Message, QuestionIter, RecordIter, parse_message};
