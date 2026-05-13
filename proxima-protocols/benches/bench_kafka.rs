#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

//! P4 — Kafka wire-format parser bench. Apples-to-apples vs an
//! inline hand-rolled parity baseline. Three workloads:
//! length-prefix peek, frame parse, full header parse.
//!
//! incumbents (versions pinned in Cargo.toml):
//!   - kafka-protocol 0.17 — typed Kafka wire-protocol decoder. Design
//!     point is owned `RequestHeader` decoding with an IndexMap of
//!     tagged fields, used by full kafka clients/brokers in Rust.
//!     (Cargo.toml comment about 0.13 failing was stale: upstream
//!     fixed multiple-applicable-items errors by 0.14; 0.17 builds
//!     cleanly with default-features = false.)
//!
//! groups (and design-favors per workload):
//!   - kafka_peek_size           design-favors: neither
//!     (4-byte BE length prefix — primitive op, no incumbent at
//!     this layer; both proxima and parity are trivial)
//!   - kafka_frame_parse         design-favors: neither
//!     (length-prefix + slice borrow; same shape)
//!   - kafka_header_parse        design-favors: incumbent
//!     (kafka_protocol::messages::RequestHeader::decode engages the
//!     incumbent's canonical decode path; proxima borrows from
//!     &[u8] vs incumbent's owned RequestHeader+IndexMap — delta
//!     includes scope reduction, like prost-vs-proxima in P9b.)

use std::time::Duration;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};

use proxima_protocols::kafka::{
    parse_frame as proxima_parse_frame, parse_request_header as proxima_parse_header,
    peek_frame_size as proxima_peek_size,
};

#[allow(dead_code)]
struct ParityHeader<'a> {
    api_key: i16,
    api_version: i16,
    correlation_id: i32,
    client_id: Option<&'a [u8]>,
}

#[inline(always)]
fn parity_peek_size(buf: &[u8]) -> Option<u32> {
    if buf.len() < 4 {
        return None;
    }
    let size = i32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]);
    if size < 0 {
        return None;
    }
    Some(size as u32)
}

#[inline(always)]
fn parity_parse_frame(buf: &[u8]) -> Option<(&[u8], usize)> {
    let size = parity_peek_size(buf)?;
    let total = 4 + size as usize;
    if buf.len() < total {
        return None;
    }
    Some((&buf[4..total], total))
}

#[inline(always)]
fn parity_parse_header(payload: &[u8]) -> Option<(ParityHeader<'_>, usize)> {
    if payload.len() < 10 {
        return None;
    }
    let api_key = i16::from_be_bytes([payload[0], payload[1]]);
    let api_version = i16::from_be_bytes([payload[2], payload[3]]);
    let correlation_id = i32::from_be_bytes([payload[4], payload[5], payload[6], payload[7]]);
    let len = i16::from_be_bytes([payload[8], payload[9]]);
    let (client_id, body_offset) = if len == -1 {
        (None, 10)
    } else if len >= 0 {
        let len = len as usize;
        if payload.len() < 10 + len {
            return None;
        }
        (Some(&payload[10..10 + len]), 10 + len)
    } else {
        return None;
    };
    Some((
        ParityHeader {
            api_key,
            api_version,
            correlation_id,
            client_id,
        },
        body_offset,
    ))
}

fn make_request(api_key: i16, client_id: &[u8], body_len: usize) -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend_from_slice(&api_key.to_be_bytes());
    payload.extend_from_slice(&11i16.to_be_bytes()); // api_version
    payload.extend_from_slice(&42i32.to_be_bytes()); // correlation_id
    payload.extend_from_slice(&(client_id.len() as i16).to_be_bytes());
    payload.extend_from_slice(client_id);
    payload.extend_from_slice(&vec![0xAB; body_len]);

    let mut frame = Vec::new();
    frame.extend_from_slice(&(payload.len() as i32).to_be_bytes());
    frame.extend_from_slice(&payload);
    frame
}

fn bench_peek_size(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("kafka_peek_size");
    group.measurement_time(Duration::from_secs(2));
    let buf = make_request(0, b"client-1", 64);
    group.throughput(Throughput::Bytes(4));
    group.bench_function("proxima", |bencher| {
        bencher.iter(|| {
            let size = proxima_peek_size(std::hint::black_box(&buf[..])).unwrap();
            std::hint::black_box(size);
        });
    });
    group.bench_function("parity", |bencher| {
        bencher.iter(|| {
            let size = parity_peek_size(std::hint::black_box(&buf[..])).unwrap();
            std::hint::black_box(size);
        });
    });
    group.finish();
}

fn bench_frame_parse(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("kafka_frame_parse");
    group.measurement_time(Duration::from_secs(2));
    let buf = make_request(0, b"client-1", 64);
    group.throughput(Throughput::Bytes(buf.len() as u64));
    group.bench_function("proxima", |bencher| {
        bencher.iter(|| {
            let (payload, used) = proxima_parse_frame(std::hint::black_box(&buf)).unwrap();
            std::hint::black_box((payload, used));
        });
    });
    group.bench_function("parity", |bencher| {
        bencher.iter(|| {
            let (payload, used) = parity_parse_frame(std::hint::black_box(&buf)).unwrap();
            std::hint::black_box((payload, used));
        });
    });
    group.finish();
}

fn bench_header_parse(criterion: &mut Criterion) {
    use bytes::Bytes;
    use kafka_protocol::messages::RequestHeader;
    use kafka_protocol::protocol::Decodable;

    let mut group = criterion.benchmark_group("kafka_header_parse");
    group.measurement_time(Duration::from_secs(2));
    let buf = make_request(0, b"client-1", 64);
    let payload = &buf[4..];
    group.throughput(Throughput::Bytes(payload.len() as u64));
    group.bench_function("proxima", |bencher| {
        bencher.iter(|| {
            let (header, offset) = proxima_parse_header(std::hint::black_box(payload)).unwrap();
            std::hint::black_box((header, offset));
        });
    });
    group.bench_function("parity", |bencher| {
        bencher.iter(|| {
            let (header, offset) = parity_parse_header(std::hint::black_box(payload)).unwrap();
            std::hint::black_box((header, offset));
        });
    });
    // header_version=1 matches `make_request` wire layout (i16 client_id
    // length, no tagged-fields varint). Each iteration must re-clone the
    // Bytes since decode advances the cursor.
    group.bench_function("kafka_protocol", |bencher| {
        let payload_bytes = Bytes::copy_from_slice(payload);
        bencher.iter(|| {
            let mut cursor = std::hint::black_box(payload_bytes.clone());
            let header = RequestHeader::decode(&mut cursor, 1).expect("kafka_protocol decode");
            std::hint::black_box(header);
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_peek_size,
    bench_frame_parse,
    bench_header_parse
);
criterion_main!(benches);
