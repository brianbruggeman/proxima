#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

//! C8 of the codec-trait initiative — proxima-protobuf-wire.
//!
//! Compares the trait-routed `WireCodec` walk against the existing
//! `Fields` iterator on the same message. parity baseline; no incumbent
//! comparison at this layer (prost walks via generated structs at a
//! different abstraction).
//!
//! Two arms:
//!
//! - `concrete_walk` — direct `Fields::new(buf)` iterator.
//! - `trait_walk` — `ProtobufWireCodec::iter_fields(buf)` (same
//!   `Fields` under the hood; this bench proves the WireCodec trait
//!   path adds zero overhead vs the bare iterator).
//!
//! Three message sizes (3, 10, 100 fields) — small messages amortize
//! the iterator setup; larger messages expose per-field branching.

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use proxima_codec::WireCodec;
use proxima_protocols::protobuf_wire::{Field, Fields, ProtobufWireCodec, encode_varint};

const FIELD_COUNTS: &[usize] = &[3, 10, 100];

fn build_varint_message(field_count: usize) -> Vec<u8> {
    let mut buf = Vec::with_capacity(field_count * 8);
    for field_number in 1..=field_count as u32 {
        // tag = (field_number << 3) | wire_type=0 (varint)
        encode_varint(u64::from(field_number << 3), &mut buf);
        // value: a varying-width varint so the walk hits both fast-path
        // (single byte) and multi-byte cases.
        encode_varint(u64::from(field_number) * 0xff, &mut buf);
    }
    buf
}

fn bench_walk(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("protobuf_codec_trait_walk");
    group.measurement_time(Duration::from_secs(5));
    for &count in FIELD_COUNTS {
        let bytes = build_varint_message(count);
        group.throughput(Throughput::Bytes(bytes.len() as u64));

        group.bench_with_input(
            BenchmarkId::new("concrete", count),
            &bytes,
            |bencher, buf| {
                bencher.iter(|| {
                    let mut sum: u64 = 0;
                    for field in Fields::new(std::hint::black_box(buf)) {
                        if let Ok(Field::Varint { value, .. }) = field {
                            sum = sum.wrapping_add(value);
                        }
                    }
                    std::hint::black_box(sum);
                });
            },
        );

        let codec = ProtobufWireCodec;
        group.bench_with_input(BenchmarkId::new("trait", count), &bytes, |bencher, buf| {
            bencher.iter(|| {
                let mut sum: u64 = 0;
                for field in codec.iter_fields(std::hint::black_box(buf)) {
                    if let Ok(Field::Varint { value, .. }) = field {
                        sum = sum.wrapping_add(value);
                    }
                }
                std::hint::black_box(sum);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_walk);
criterion_main!(benches);
