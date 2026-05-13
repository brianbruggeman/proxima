#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

//! P9b — Protobuf wire-format codec micro-bench. Three arms:
//!
//! - `proxima_decode_varint` / `proxima_encode_varint` — `proxima::protobuf::{decode_varint, encode_varint}`.
//! - `parity_decode_varint` / `parity_encode_varint` — hand-rolled inline.
//! - `prost_decode_varint` / `prost_encode_varint` — `prost::encoding::*`, the
//!   standard Rust protobuf crate. Scope-matched (same operation), serves as
//!   the ecosystem reference.
//!
//! incumbents (versions pinned in Cargo.toml):
//!   - prost 0.13 — canonical Rust protobuf codec; design point is the
//!     unrolled LEB128 varint hot path + schema-aware field iteration
//!     used across the protobuf ecosystem.
//!
//! groups (and design-favors per workload):
//!   - protobuf_varint_decode    design-favors: incumbent
//!     (prost's unrolled varint, scope-matched: both arms take &[u8]
//!     and return (u64, usize). Fully engages prost's design point.)
//!   - protobuf_varint_encode    design-favors: incumbent (same shape)
//!   - protobuf_walk_message     design-favors: proxima
//!     (`Fields` iterator vs hand-rolled parity; prost's schema-aware
//!     decode walks via generated structs at a different layer — no
//!     scope-matched incumbent arm at this layer)
//!
//! Plus a `walk_message` bench that exercises the full field iterator
//! against a 7-field message (varint + len + I32 + I64 mix). Same scope
//! hand-rolled in parity.

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use proxima_protocols::protobuf_wire::{
    Fields, decode_varint as proxima_decode_varint, encode_varint as proxima_encode_varint,
};

const VARINT_VALUES: &[u64] = &[0, 1, 127, 128, 16_383, 16_384, 1_000_000, u64::MAX];

#[inline]
fn parity_decode_varint(buf: &[u8]) -> Option<(u64, usize)> {
    let mut value: u64 = 0;
    let mut shift: u32 = 0;
    let mut cursor = 0;
    while cursor < buf.len() && cursor < 10 {
        let byte = buf[cursor];
        cursor += 1;
        value |= u64::from(byte & 0x7F) << shift;
        if byte & 0x80 == 0 {
            return Some((value, cursor));
        }
        shift += 7;
    }
    None
}

#[inline]
fn parity_encode_varint(mut value: u64, dest: &mut Vec<u8>) {
    while value >= 0x80 {
        dest.push((value as u8) | 0x80);
        value >>= 7;
    }
    dest.push(value as u8);
}

fn make_varint_buffer(value: u64) -> Vec<u8> {
    let mut buf = Vec::new();
    proxima_encode_varint(value, &mut buf);
    buf
}

fn bench_decode_varint(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("varint_decode");
    group.measurement_time(Duration::from_secs(2));
    for &value in VARINT_VALUES {
        let buf = make_varint_buffer(value);
        group.throughput(Throughput::Bytes(buf.len() as u64));
        group.bench_with_input(BenchmarkId::new("proxima", value), &buf, |bencher, buf| {
            bencher.iter(|| {
                let (decoded, used) = proxima_decode_varint(std::hint::black_box(buf)).unwrap();
                std::hint::black_box((decoded, used));
            });
        });
        group.bench_with_input(BenchmarkId::new("parity", value), &buf, |bencher, buf| {
            bencher.iter(|| {
                let (decoded, used) = parity_decode_varint(std::hint::black_box(buf)).unwrap();
                std::hint::black_box((decoded, used));
            });
        });
        group.bench_with_input(BenchmarkId::new("prost", value), &buf, |bencher, buf| {
            bencher.iter(|| {
                let mut cursor: &[u8] = std::hint::black_box(buf);
                let decoded = prost::encoding::decode_varint(&mut cursor).unwrap();
                std::hint::black_box(decoded);
            });
        });
    }
    group.finish();
}

fn bench_encode_varint(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("varint_encode");
    group.measurement_time(Duration::from_secs(2));
    for &value in VARINT_VALUES {
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("proxima", value),
            &value,
            |bencher, &value| {
                bencher.iter_with_setup(
                    || Vec::with_capacity(10),
                    |mut buf| {
                        proxima_encode_varint(value, &mut buf);
                        std::hint::black_box(buf);
                    },
                );
            },
        );
        group.bench_with_input(
            BenchmarkId::new("parity", value),
            &value,
            |bencher, &value| {
                bencher.iter_with_setup(
                    || Vec::with_capacity(10),
                    |mut buf| {
                        parity_encode_varint(value, &mut buf);
                        std::hint::black_box(buf);
                    },
                );
            },
        );
        group.bench_with_input(
            BenchmarkId::new("prost", value),
            &value,
            |bencher, &value| {
                bencher.iter_with_setup(
                    || bytes::BytesMut::with_capacity(10),
                    |mut buf| {
                        prost::encoding::encode_varint(value, &mut buf);
                        std::hint::black_box(buf);
                    },
                );
            },
        );
    }
    group.finish();
}

fn build_walk_message() -> Vec<u8> {
    let mut buf = Vec::new();
    buf.push(0x08);
    proxima_encode_varint(150, &mut buf);
    buf.push((2 << 3) | 2);
    proxima_encode_varint(5, &mut buf);
    buf.extend_from_slice(b"hello");
    buf.push((3 << 3) | 5);
    buf.extend_from_slice(&[0xef, 0xbe, 0xad, 0xde]);
    buf.push((4 << 3) | 1);
    buf.extend_from_slice(&[0, 1, 2, 3, 4, 5, 6, 7]);
    buf.push(5 << 3);
    proxima_encode_varint(u64::MAX, &mut buf);
    buf.push((6 << 3) | 2);
    proxima_encode_varint(32, &mut buf);
    buf.extend_from_slice(&[0xAB; 32]);
    buf.push(7 << 3);
    proxima_encode_varint(1, &mut buf);
    buf
}

fn bench_walk_message(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("walk_message");
    group.measurement_time(Duration::from_secs(2));
    let buf = build_walk_message();
    group.throughput(Throughput::Bytes(buf.len() as u64));
    group.bench_function("proxima", |bencher| {
        bencher.iter(|| {
            for field in Fields::new(std::hint::black_box(&buf)) {
                std::hint::black_box(field.unwrap().field_number());
            }
        });
    });
    group.bench_function("parity", |bencher| {
        bencher.iter(|| {
            let mut cursor = std::hint::black_box(&buf[..]);
            while !cursor.is_empty() {
                let (raw, tag_used) = parity_decode_varint(cursor).unwrap();
                let wire = (raw & 0x07) as u8;
                let field = (raw >> 3) as u32;
                cursor = &cursor[tag_used..];
                let consumed = match wire {
                    0 => parity_decode_varint(cursor).unwrap().1,
                    1 => 8,
                    2 => {
                        let (len, len_used) = parity_decode_varint(cursor).unwrap();
                        len_used + len as usize
                    }
                    5 => 4,
                    _ => panic!("bad wire"),
                };
                cursor = &cursor[consumed..];
                std::hint::black_box(field);
            }
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_decode_varint,
    bench_encode_varint,
    bench_walk_message
);
criterion_main!(benches);
