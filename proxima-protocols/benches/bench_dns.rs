#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

//! P1 — DNS message parser micro-bench. Builds the parity baseline
//! into the same bench file so the Compare-bench gate cell starts
//! from an apples-to-apples comparison (lesson from P5's retraction).
//!
//! incumbents (versions pinned in Cargo.toml):
//!   - hickory-proto — full RFC-conformant DNS message parser;
//!     design point is owned typed Record / Message structs supporting
//!     every RDATA variant, EDNS options, DNSSEC records, etc.
//!
//! groups (and design-favors per workload):
//!   - dns_parse_proxima           design-favors: proxima
//!     (borrowed-view decode of header + question + 1 record)
//!   - dns_parse_parity            design-favors: neither
//!     (hand-rolled timing reference matching proxima's scope)
//!   - dns_parse_hickory           design-favors: incumbent
//!     (Message::from_vec — hickory's full owned-typed decode path,
//!     every RDATA variant supported; same wire bytes in, owned
//!     struct out. NOT a like-for-like comparison vs proxima parse
//!     scope — kept as a sizing reference)
//!   - dns_workload_count_a        design-favors: incumbent
//!     (both arms answer the same question — "how many A records" —
//!     for the same input. proxima walks borrowed records; hickory
//!     parses to owned + filters by RecordType. Hickory's design
//!     point IS owned typed traversal, which this engages.)

use std::net::Ipv4Addr;
use std::time::Duration;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};

/// Minimal valid DNS *response* for "example.com IN A" → 93.184.216.34.
/// Header + 1 question + 1 answer (with name compression).
fn example_response() -> Vec<u8> {
    let mut msg = Vec::new();
    msg.extend_from_slice(&1234u16.to_be_bytes()); // id
    msg.extend_from_slice(&0x8180u16.to_be_bytes()); // flags: response, rd, ra
    msg.extend_from_slice(&1u16.to_be_bytes()); // qdcount
    msg.extend_from_slice(&1u16.to_be_bytes()); // ancount
    msg.extend_from_slice(&0u16.to_be_bytes()); // nscount
    msg.extend_from_slice(&0u16.to_be_bytes()); // arcount
    // question: example.com IN A
    msg.push(7);
    msg.extend_from_slice(b"example");
    msg.push(3);
    msg.extend_from_slice(b"com");
    msg.push(0);
    msg.extend_from_slice(&1u16.to_be_bytes()); // qtype A
    msg.extend_from_slice(&1u16.to_be_bytes()); // qclass IN
    // answer: ptr-to-qname, A, IN, ttl 300, rdlen 4, 93.184.216.34
    msg.extend_from_slice(&[0xC0, 0x0C]);
    msg.extend_from_slice(&1u16.to_be_bytes());
    msg.extend_from_slice(&1u16.to_be_bytes());
    msg.extend_from_slice(&300u32.to_be_bytes());
    msg.extend_from_slice(&4u16.to_be_bytes());
    msg.extend_from_slice(&[93, 184, 216, 34]);
    msg
}

fn proxima_parse(criterion: &mut Criterion) {
    use proxima_protocols::dns::{parse_header, parse_question, parse_record};
    let mut group = criterion.benchmark_group("dns_parse_proxima");
    group.measurement_time(Duration::from_secs(3));
    let msg = example_response();
    group.throughput(Throughput::Bytes(msg.len() as u64));
    group.bench_function("response_full", |bencher| {
        bencher.iter(|| {
            let header = parse_header(std::hint::black_box(&msg)).expect("hdr");
            let (question, q_used) = parse_question(&msg, 12).expect("q");
            let (record, _) = parse_record(&msg, 12 + q_used).expect("a");
            std::hint::black_box((header, question, record));
        });
    });
    group.finish();
}

#[allow(dead_code)]
/// Parity baseline. Same scope, **same output shape**, and **same
/// rdata variants** as proxima — header struct + question struct +
/// record struct + `RData` enum covering A / AAAA / CNAME / NS /
/// PTR / Raw. Hand-rolled, zero-alloc, lazy name (offset-based).
///
/// Meaningful parity here: proxima and this baseline pay the same
/// `match rtype` dispatch and the same enum-variant construction.
/// Anything proxima loses by is genuinely parser-shape overhead,
/// not API-feature overhead the baseline avoids.
mod parity {
    use std::net::{Ipv4Addr, Ipv6Addr};

    #[derive(Debug)]
    pub struct Header {
        pub id: u16,
        pub flags: u16,
        pub qdcount: u16,
        pub ancount: u16,
        pub nscount: u16,
        pub arcount: u16,
    }
    #[derive(Debug, Clone, Copy)]
    pub struct LazyName {
        pub start: usize,
        pub encoded_len: usize,
    }
    #[derive(Debug)]
    pub struct Question {
        pub name: LazyName,
        pub qtype: u16,
        pub qclass: u16,
    }
    #[derive(Debug)]
    pub struct Record {
        pub name: LazyName,
        pub rtype: u16,
        pub rclass: u16,
        pub ttl: u32,
        pub rdata: RData,
    }
    #[derive(Debug)]
    pub enum RData {
        A(Ipv4Addr),
        Aaaa(Ipv6Addr),
        Cname(LazyName),
        Ns(LazyName),
        Ptr(LazyName),
        Raw { rtype: u16, len: u16 },
    }
    pub fn parse_response(msg: &[u8]) -> Option<(Header, Question, Record)> {
        if msg.len() < 12 {
            return None;
        }
        let header = Header {
            id: u16::from_be_bytes([msg[0], msg[1]]),
            flags: u16::from_be_bytes([msg[2], msg[3]]),
            qdcount: u16::from_be_bytes([msg[4], msg[5]]),
            ancount: u16::from_be_bytes([msg[6], msg[7]]),
            nscount: u16::from_be_bytes([msg[8], msg[9]]),
            arcount: u16::from_be_bytes([msg[10], msg[11]]),
        };
        let (q_name, q_used) = walk_name(msg, 12)?;
        let q_after = 12 + q_used;
        if msg.len() < q_after + 4 {
            return None;
        }
        let qtype = u16::from_be_bytes([msg[q_after], msg[q_after + 1]]);
        let qclass = u16::from_be_bytes([msg[q_after + 2], msg[q_after + 3]]);
        let question = Question {
            name: q_name,
            qtype,
            qclass,
        };
        let a_offset = q_after + 4;
        let (r_name, r_used) = walk_name(msg, a_offset)?;
        let after_name = a_offset + r_used;
        if msg.len() < after_name + 10 {
            return None;
        }
        let rtype = u16::from_be_bytes([msg[after_name], msg[after_name + 1]]);
        let rclass = u16::from_be_bytes([msg[after_name + 2], msg[after_name + 3]]);
        let ttl = u32::from_be_bytes([
            msg[after_name + 4],
            msg[after_name + 5],
            msg[after_name + 6],
            msg[after_name + 7],
        ]);
        let rdlength = u16::from_be_bytes([msg[after_name + 8], msg[after_name + 9]]) as usize;
        let rdata_start = after_name + 10;
        let rdata_end = rdata_start + rdlength;
        if msg.len() < rdata_end {
            return None;
        }
        let rdata = match rtype {
            1 if rdlength == 4 => RData::A(Ipv4Addr::new(
                msg[rdata_start],
                msg[rdata_start + 1],
                msg[rdata_start + 2],
                msg[rdata_start + 3],
            )),
            28 if rdlength == 16 => {
                let mut bytes = [0u8; 16];
                bytes.copy_from_slice(&msg[rdata_start..rdata_end]);
                RData::Aaaa(Ipv6Addr::from(bytes))
            }
            5 | 2 | 12 => {
                let (target, _) = walk_name(msg, rdata_start)?;
                match rtype {
                    5 => RData::Cname(target),
                    2 => RData::Ns(target),
                    12 => RData::Ptr(target),
                    _ => unreachable!(),
                }
            }
            _ => RData::Raw {
                rtype,
                len: rdlength as u16,
            },
        };
        let record = Record {
            name: r_name,
            rtype,
            rclass,
            ttl,
            rdata,
        };
        Some((header, question, record))
    }
    fn walk_name(msg: &[u8], offset: usize) -> Option<(LazyName, usize)> {
        let mut cursor = offset;
        let mut consumed_at_start: Option<usize> = None;
        loop {
            let byte = *msg.get(cursor)?;
            match byte & 0xC0 {
                0x00 => {
                    cursor += 1;
                    if byte == 0 {
                        let used = consumed_at_start.unwrap_or(cursor - offset);
                        return Some((
                            LazyName {
                                start: offset,
                                encoded_len: used,
                            },
                            used,
                        ));
                    }
                    cursor += byte as usize;
                    if cursor > msg.len() {
                        return None;
                    }
                }
                0xC0 => {
                    if consumed_at_start.is_none() {
                        consumed_at_start = Some(cursor + 2 - offset);
                    }
                    let high = byte as usize & 0x3F;
                    let low = *msg.get(cursor + 1)? as usize;
                    cursor = (high << 8) | low;
                    if cursor >= msg.len() {
                        return None;
                    }
                }
                _ => return None,
            }
        }
    }
}

fn parity_parse(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("dns_parse_parity");
    group.measurement_time(Duration::from_secs(3));
    let msg = example_response();
    group.throughput(Throughput::Bytes(msg.len() as u64));
    group.bench_function("response_full", |bencher| {
        bencher.iter(|| {
            let result = parity::parse_response(std::hint::black_box(&msg)).expect("ok");
            std::hint::black_box(result);
        });
    });
    let _: Ipv4Addr = Ipv4Addr::new(0, 0, 0, 0); // silence dead import on some builds
    group.finish();
}

/// Real-crate reference: `hickory-proto::op::Message::from_vec`.
/// Standalone throughput arm — produces a typed Message struct,
/// allocating owned data. Kept for data-point reference; NOT a
/// gate-passing benchmark on its own because output shapes differ.
fn hickory_parse(criterion: &mut Criterion) {
    use hickory_proto::op::Message;
    let mut group = criterion.benchmark_group("dns_parse_hickory");
    group.measurement_time(Duration::from_secs(3));
    let msg = example_response();
    group.throughput(Throughput::Bytes(msg.len() as u64));
    group.bench_function("response_full", |bencher| {
        bencher.iter(|| {
            let parsed = Message::from_vec(std::hint::black_box(&msg)).expect("hickory");
            std::hint::black_box(parsed);
        });
    });
    group.finish();
}

/// Multi-record fixture for the workload bench. 5 answers: 3 A
/// records, 1 AAAA, 1 CNAME. Same wire bytes feed both proxima and
/// hickory in the workload arms below.
fn mixed_record_response() -> Vec<u8> {
    let mut msg = Vec::new();
    msg.extend_from_slice(&5678u16.to_be_bytes()); // id
    msg.extend_from_slice(&0x8180u16.to_be_bytes()); // flags
    msg.extend_from_slice(&1u16.to_be_bytes()); // qdcount
    msg.extend_from_slice(&5u16.to_be_bytes()); // ancount = 5
    msg.extend_from_slice(&0u16.to_be_bytes()); // nscount
    msg.extend_from_slice(&0u16.to_be_bytes()); // arcount
    // question: example.com IN A
    msg.push(7);
    msg.extend_from_slice(b"example");
    msg.push(3);
    msg.extend_from_slice(b"com");
    msg.push(0);
    msg.extend_from_slice(&1u16.to_be_bytes()); // qtype A
    msg.extend_from_slice(&1u16.to_be_bytes()); // qclass IN

    // helper: write an A record (rdlen=4)
    let push_a = |msg: &mut Vec<u8>, octets: [u8; 4]| {
        msg.extend_from_slice(&[0xC0, 0x0C]); // ptr to qname at offset 12
        msg.extend_from_slice(&1u16.to_be_bytes()); // type A
        msg.extend_from_slice(&1u16.to_be_bytes()); // class IN
        msg.extend_from_slice(&300u32.to_be_bytes()); // ttl
        msg.extend_from_slice(&4u16.to_be_bytes()); // rdlen
        msg.extend_from_slice(&octets);
    };
    push_a(&mut msg, [93, 184, 216, 34]);
    push_a(&mut msg, [93, 184, 216, 35]);
    push_a(&mut msg, [93, 184, 216, 36]);
    // AAAA record (rdlen=16)
    msg.extend_from_slice(&[0xC0, 0x0C]);
    msg.extend_from_slice(&28u16.to_be_bytes()); // type AAAA
    msg.extend_from_slice(&1u16.to_be_bytes());
    msg.extend_from_slice(&300u32.to_be_bytes());
    msg.extend_from_slice(&16u16.to_be_bytes());
    msg.extend_from_slice(&[0x20, 0x01, 0xdb, 0x8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]);
    // CNAME record (rdlen = compressed name "alias.example.com" → ptr)
    msg.extend_from_slice(&[0xC0, 0x0C]);
    msg.extend_from_slice(&5u16.to_be_bytes()); // type CNAME
    msg.extend_from_slice(&1u16.to_be_bytes());
    msg.extend_from_slice(&300u32.to_be_bytes());
    // CNAME target = "alias" + ptr to "example.com"
    // rdlen = 1 + 5 + 2 = 8
    msg.extend_from_slice(&8u16.to_be_bytes());
    msg.push(5);
    msg.extend_from_slice(b"alias");
    msg.extend_from_slice(&[0xC0, 0x0C]); // ptr to "example.com"
    msg
}

/// **Workload**: given a DNS response, count answer records whose
/// type is A (rtype == 1). Feature-parity between proxima and hickory:
/// both impls take `&[u8]`, return `usize`. Implementation strategies
/// differ — proxima walks record headers; hickory parses + filters
/// answers. The benchmark measures total time to deliver the same
/// answer for the same input.
fn workload_count_a_records_proxima(msg: &[u8]) -> usize {
    use proxima_protocols::dns::{RData, parse_header, parse_question, parse_record};
    let header = parse_header(msg).expect("header");
    let (_question, q_used) = parse_question(msg, 12).expect("question");
    let mut cursor = 12 + q_used;
    let mut count = 0usize;
    for _ in 0..header.ancount {
        let (record, used) = parse_record(msg, cursor).expect("record");
        if matches!(record.rdata, RData::A(_)) {
            count += 1;
        }
        cursor += used;
    }
    count
}

fn workload_count_a_records_hickory(msg: &[u8]) -> usize {
    use hickory_proto::op::Message;
    use hickory_proto::rr::RecordType;
    let parsed = Message::from_vec(msg).expect("hickory");
    parsed
        .answers
        .iter()
        .filter(|r| r.record_type() == RecordType::A)
        .count()
}

/// Workload bench: count-A-records, identical feature for both arms.
/// This IS a gate-passing benchmark — both impls answer the same
/// question for the same input. Time difference is implementation
/// strategy, not output-shape difference.
fn workload_count_a_records(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("dns_workload_count_a");
    group.measurement_time(Duration::from_secs(3));
    let msg = mixed_record_response();
    group.throughput(Throughput::Bytes(msg.len() as u64));
    // sanity check: both impls produce the same answer
    let proxima_count = workload_count_a_records_proxima(&msg);
    let hickory_count = workload_count_a_records_hickory(&msg);
    assert_eq!(
        proxima_count, hickory_count,
        "workload result mismatch: proxima={proxima_count}, hickory={hickory_count}"
    );
    group.bench_function("proxima", |bencher| {
        bencher.iter(|| {
            let n = workload_count_a_records_proxima(std::hint::black_box(&msg));
            std::hint::black_box(n);
        });
    });
    group.bench_function("hickory", |bencher| {
        bencher.iter(|| {
            let n = workload_count_a_records_hickory(std::hint::black_box(&msg));
            std::hint::black_box(n);
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    proxima_parse,
    parity_parse,
    hickory_parse,
    workload_count_a_records
);
criterion_main!(benches);
