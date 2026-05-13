#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

//! P5 — Redis RESP3 parser micro-bench.
//!
//! incumbents (versions pinned in Cargo.toml):
//!   - redis-protocol 6.0 — full RESP2 + RESP3 spec, 14+ frame types,
//!     streaming + complete + range + owned + bytes variants. Design
//!     point is a complete Redis client/server codec.
//!
//! groups (and design-favors per workload):
//!   - redis_proxima_parse / redis_tiny_parity   design-favors: proxima
//!     (proxima covers 5 of 14 RESP3 frame types — gate-passing arm
//!     is the hand-rolled parity matching THAT 5-type scope)
//!   - redis_real_redis_protocol                 design-favors: incumbent
//!     (kept as scope-mismatched reference; engages redis-protocol's
//!     full owned-Frame decode design. proxima does strictly less
//!     work, so the delta includes scope reduction — not pure
//!     parser-shape advantage.)
//!
//! REGIME OUT-OF-SCOPE: streaming, aggregate types beyond Array,
//! push/pubsub. redis-protocol supports those; proxima does not.
//! On those frame types no comparison is possible; for our 5-type
//! subset the hand-rolled parity is the honest gate.
//!
//! Original comparison (proxima vs `redis_protocol::resp3::decode`)
//! was scope-mismatched: redis-protocol parses the full RESP2 +
//! RESP3 spec (14+ frame types, streaming + complete + range +
//! owned + bytes variants); proxima parses 5. The measured win
//! was scope reduction, not parser-shape advantage.
//!
//! This bench keeps the redis-protocol arm as a "different scope"
//! reference but **also runs a parity baseline** —
//! `tiny_parity_parse` — covering the same 5 frame types in a
//! hand-rolled minimal parser. That arm is the gate-passing
//! comparison.
//!
//! Six frame shapes covering the substrate's targeted coverage:
//!
//! - `+OK\r\n`                                 SimpleString
//! - `-ERR unknown command 'fizz'\r\n`         Error
//! - `:42\r\n`                                 Integer (small positive)
//! - `$5\r\nhello\r\n`                         BlobString (short)
//! - `*2\r\n$3\r\nGET\r\n$3\r\nfoo\r\n`        Array of blobs
//! - large BlobString (1 KiB payload) — stresses the byte-copy path

use std::time::Duration;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use proxima_protocols::redis::{Frame as ProximaFrame, parse as proxima_parse};

fn payloads() -> Vec<(&'static str, Vec<u8>)> {
    let mut big_blob = b"$1024\r\n".to_vec();
    big_blob.extend(std::iter::repeat_n(b'x', 1024));
    big_blob.extend_from_slice(b"\r\n");
    vec![
        ("simple_string", b"+OK\r\n".to_vec()),
        ("error", b"-ERR unknown command 'fizz'\r\n".to_vec()),
        ("integer", b":42\r\n".to_vec()),
        ("blob_short", b"$5\r\nhello\r\n".to_vec()),
        (
            "array_of_blobs",
            b"*2\r\n$3\r\nGET\r\n$3\r\nfoo\r\n".to_vec(),
        ),
        ("blob_1kb", big_blob),
    ]
}

fn proxima_parser(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("redis_parse_proxima");
    group.measurement_time(Duration::from_secs(3));
    for (label, buf) in &payloads() {
        group.throughput(Throughput::Bytes(buf.len() as u64));
        group.bench_function(*label, |bencher| {
            bencher.iter(|| {
                let (frame, used) = proxima_parse(std::hint::black_box(buf)).expect("ok");
                std::hint::black_box((frame, used));
            });
        });
    }
    group.finish();
}

/// Parity baseline: a minimal RESP3 parser covering only the 5
/// frame types proxima covers. Same scope, hand-rolled, used as
/// the apples-to-apples comparison in the discipline log. If
/// proxima beats this, it's a real parser-shape win; if it ties
/// or loses, the proxima parser doesn't have shape advantage and
/// any prior "win vs redis-protocol" was pure scope reduction.
#[allow(dead_code)]
mod tiny_parity {
    // Frame fields are constructed and `black_box`'d in the bench
    // but never matched on — that's fine; the parser allocates the
    // variant payloads which is what we're measuring. `dead_code`
    // allow at the module level covers all variants uniformly.
    #[derive(Debug)]
    pub enum Frame<'a> {
        SimpleString(&'a [u8]),
        Error(&'a [u8]),
        Integer(i64),
        BlobString(&'a [u8]),
        NullBlob,
        Array(Vec<Frame<'a>>),
        NullArray,
    }
    #[derive(Debug)]
    pub enum ParseError {
        NeedMore,
        Malformed,
    }
    pub fn parse(buf: &[u8]) -> Result<(Frame<'_>, usize), ParseError> {
        if buf.is_empty() {
            return Err(ParseError::NeedMore);
        }
        let tag = buf[0];
        let rest = &buf[1..];
        match tag {
            b'+' => {
                let crlf = find_crlf(rest)?;
                Ok((Frame::SimpleString(&rest[..crlf]), 1 + crlf + 2))
            }
            b'-' => {
                let crlf = find_crlf(rest)?;
                Ok((Frame::Error(&rest[..crlf]), 1 + crlf + 2))
            }
            b':' => {
                let crlf = find_crlf(rest)?;
                let text = std::str::from_utf8(&rest[..crlf]).map_err(|_| ParseError::Malformed)?;
                let value: i64 = text.parse().map_err(|_| ParseError::Malformed)?;
                Ok((Frame::Integer(value), 1 + crlf + 2))
            }
            b'$' => {
                let crlf = find_crlf(rest)?;
                let text = std::str::from_utf8(&rest[..crlf]).map_err(|_| ParseError::Malformed)?;
                let len: i64 = text.parse().map_err(|_| ParseError::Malformed)?;
                if len == -1 {
                    return Ok((Frame::NullBlob, 1 + crlf + 2));
                }
                if len < 0 {
                    return Err(ParseError::Malformed);
                }
                let len = len as usize;
                let total = 1 + crlf + 2 + len + 2;
                if buf.len() < total {
                    return Err(ParseError::NeedMore);
                }
                let payload = &rest[crlf + 2..crlf + 2 + len];
                if rest.get(crlf + 2 + len) != Some(&b'\r')
                    || rest.get(crlf + 2 + len + 1) != Some(&b'\n')
                {
                    return Err(ParseError::Malformed);
                }
                Ok((Frame::BlobString(payload), total))
            }
            b'*' => {
                let crlf = find_crlf(rest)?;
                let text = std::str::from_utf8(&rest[..crlf]).map_err(|_| ParseError::Malformed)?;
                let len: i64 = text.parse().map_err(|_| ParseError::Malformed)?;
                if len == -1 {
                    return Ok((Frame::NullArray, 1 + crlf + 2));
                }
                if len < 0 {
                    return Err(ParseError::Malformed);
                }
                let len = len as usize;
                let mut elems = Vec::with_capacity(len);
                let mut cursor = 1 + crlf + 2;
                for _ in 0..len {
                    let (elem, used) = parse(&buf[cursor..])?;
                    elems.push(elem);
                    cursor += used;
                }
                Ok((Frame::Array(elems), cursor))
            }
            _ => Err(ParseError::Malformed),
        }
    }
    fn find_crlf(buf: &[u8]) -> Result<usize, ParseError> {
        let mut index = 0;
        while index + 1 < buf.len() {
            if buf[index] == b'\r' && buf[index + 1] == b'\n' {
                return Ok(index);
            }
            index += 1;
        }
        Err(ParseError::NeedMore)
    }
}

fn parity_baseline(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("redis_parse_parity_baseline");
    group.measurement_time(Duration::from_secs(3));
    for (label, buf) in &payloads() {
        group.throughput(Throughput::Bytes(buf.len() as u64));
        group.bench_function(*label, |bencher| {
            bencher.iter(|| {
                let (frame, used) = tiny_parity::parse(std::hint::black_box(buf)).expect("ok");
                std::hint::black_box((frame, used));
            });
        });
    }
    group.finish();
}

fn redis_protocol_parser(criterion: &mut Criterion) {
    use redis_protocol::resp3::decode::complete::decode;
    let mut group = criterion.benchmark_group("redis_parse_redis_protocol");
    group.measurement_time(Duration::from_secs(3));
    for (label, buf) in &payloads() {
        group.throughput(Throughput::Bytes(buf.len() as u64));
        group.bench_function(*label, |bencher| {
            bencher.iter(|| {
                let result = decode(std::hint::black_box(buf)).expect("ok");
                std::hint::black_box(result);
            });
        });
    }
    group.finish();
}

fn proxima_round_trip(criterion: &mut Criterion) {
    use proxima_protocols::redis::encode;
    let mut group = criterion.benchmark_group("redis_round_trip_proxima");
    group.measurement_time(Duration::from_secs(3));
    let frames: Vec<(&'static str, ProximaFrame<'static>)> = vec![
        ("simple_string", ProximaFrame::SimpleString(b"OK")),
        ("integer", ProximaFrame::Integer(42)),
        ("blob_short", ProximaFrame::BlobString(b"hello")),
        (
            "array_of_blobs",
            ProximaFrame::Array(vec![
                ProximaFrame::BlobString(b"GET"),
                ProximaFrame::BlobString(b"foo"),
            ]),
        ),
    ];
    for (label, frame) in &frames {
        group.bench_function(*label, |bencher| {
            bencher.iter(|| {
                let bytes = encode(std::hint::black_box(frame));
                let (parsed, _) = proxima_parse(&bytes).expect("ok");
                std::hint::black_box(parsed);
            });
        });
    }
    group.finish();
}

// Workload bench: feature-parity comparison
//
// Workload: given a RESP3 command frame (Array of BlobStrings, where
// position 0 is the command verb), classify the command into a category
// for routing. Categories: READ (GET/MGET/EXISTS), WRITE (SET/MSET/DEL),
// PUBSUB (SUBSCRIBE/PUBLISH), OTHER. Both impls take `&[u8]` and return
// `Option<u8>` (0=READ, 1=WRITE, 2=PUBSUB, 3=OTHER).
//
// Both impls fully parse the array. Difference is output ownership:
// proxima borrows the verb slice from the buffer; redis-protocol
// decodes into owned BulkString (Bytes allocation).

fn classify_verb(verb: &[u8]) -> u8 {
    match verb {
        b"GET" | b"MGET" | b"EXISTS" | b"get" | b"mget" | b"exists" => 0, // READ
        b"SET" | b"MSET" | b"DEL" | b"set" | b"mset" | b"del" => 1,       // WRITE
        b"SUBSCRIBE" | b"PUBLISH" | b"subscribe" | b"publish" => 2,       // PUBSUB
        _ => 3,                                                           // OTHER
    }
}

fn workload_classify_command_proxima(buf: &[u8]) -> Option<u8> {
    let (frame, _) = proxima_parse(buf).ok()?;
    match frame {
        ProximaFrame::Array(items) => {
            let first = items.first()?;
            match first {
                ProximaFrame::BlobString(bytes) => Some(classify_verb(bytes)),
                _ => None,
            }
        }
        _ => None,
    }
}

fn workload_classify_command_redis_protocol(buf: &[u8]) -> Option<u8> {
    use redis_protocol::resp3::decode::complete::decode;
    use redis_protocol::resp3::types::OwnedFrame;
    let (frame, _) = decode(buf).ok().flatten()?;
    match frame {
        OwnedFrame::Array { data, .. } => {
            let first = data.first()?;
            match first {
                OwnedFrame::BlobString { data, .. } => Some(classify_verb(data)),
                _ => None,
            }
        }
        _ => None,
    }
}

fn bench_workload_classify_command(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("redis_workload_classify_command");
    group.measurement_time(Duration::from_secs(2));
    let buf = b"*2\r\n$3\r\nGET\r\n$3\r\nfoo\r\n".to_vec();
    let a = workload_classify_command_proxima(&buf);
    let b = workload_classify_command_redis_protocol(&buf);
    assert_eq!(
        a, b,
        "workload mismatch: proxima={a:?}, redis_protocol={b:?}"
    );
    group.throughput(Throughput::Bytes(buf.len() as u64));
    group.bench_function("proxima", |bencher| {
        bencher.iter(|| {
            let r = workload_classify_command_proxima(std::hint::black_box(&buf));
            std::hint::black_box(r);
        });
    });
    group.bench_function("redis_protocol", |bencher| {
        bencher.iter(|| {
            let r = workload_classify_command_redis_protocol(std::hint::black_box(&buf));
            std::hint::black_box(r);
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    proxima_parser,
    parity_baseline,
    redis_protocol_parser,
    proxima_round_trip,
    bench_workload_classify_command
);
criterion_main!(benches);
