// bench fixtures legitimately fail-fast on encoder errors; clippy allowances
// are intentional for bench code paths that should panic loudly on setup bugs.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

//! C1 — varint codec bench arms.
//!
//! Home-turf incumbent: `quinn-proto::VarInt`. Arms cover:
//!
//! - **single-value encode / decode** at each length class (1, 2, 4, 8 bytes).
//!   Tests the per-call overhead — relevant for hot-path parser loops where
//!   one varint per frame field is the cost basis.
//! - **stream encode / decode** at 16 B / 1 KB / 8 KB / 64 KB output sizes.
//!   Tests amortized per-varint cost when packing many values back-to-back.
//! - **adversarial decode**: truncated input — rejection-latency bench.
//!
//! Multi-arch SIMD coverage: this codec is branch-light + table-driven; the
//! per-arch story is dominated by `copy_from_slice` + `to_be_bytes` / `from_be_bytes`,
//! both of which lower to small register moves on x86_64 + aarch64. No
//! hand-rolled SIMD planned — `iai-callgrind` cycle counts will land in the
//! C1 discipline.md row once a Linux x86_64 CI cell exists (deferred per
//! `docs/proxima-quic/edges.md`).
//!
//! CoV expected to be noisy on a contended laptop; treat criterion's
//! `[lower, mean, upper]` 95% CI as the truth and ignore the single number.

use bytes::{Buf, BytesMut};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use proxima_protocols::quic::varint;
use quinn_proto::VarInt;
use quinn_proto::coding::Codec;

/// Representative values across the four length classes.
const VALUES: &[u64] = &[
    37,                      // 1-byte
    15_293,                  // 2-byte
    494_878_333,             // 4-byte
    151_288_809_941_952_652, // 8-byte
];

const STREAM_SIZES: &[usize] = &[16, 1024, 8 * 1024, 64 * 1024];

fn bench_single_encode(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c1_varint_single_encode");
    for &value in VALUES {
        let size = varint::encoded_len(value);
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(
            BenchmarkId::new("proxima_quic_proto", value),
            &value,
            |bencher, &value| {
                let mut buffer = [0u8; varint::MAX_ENCODED_LEN];
                bencher.iter(|| {
                    let written = varint::encode(std::hint::black_box(value), &mut buffer)
                        .expect("varint encode");
                    std::hint::black_box(written);
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("quinn_proto", value),
            &value,
            |bencher, &value| {
                let mut buffer = BytesMut::with_capacity(varint::MAX_ENCODED_LEN);
                bencher.iter(|| {
                    buffer.clear();
                    let varint =
                        VarInt::from_u64(std::hint::black_box(value)).expect("VarInt::from_u64");
                    varint.encode(&mut buffer);
                    std::hint::black_box(buffer.len());
                });
            },
        );
    }
    group.finish();
}

fn bench_single_decode(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c1_varint_single_decode");
    for &value in VALUES {
        let mut encoded = [0u8; varint::MAX_ENCODED_LEN];
        let size = varint::encode(value, &mut encoded).expect("setup: encode");
        let slice = &encoded[..size];
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(
            BenchmarkId::new("proxima_quic_proto", value),
            &slice,
            |bencher, slice| {
                bencher.iter(|| {
                    let (value, consumed) =
                        varint::decode(std::hint::black_box(slice)).expect("decode");
                    std::hint::black_box((value, consumed));
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("quinn_proto", value),
            &slice,
            |bencher, slice| {
                bencher.iter(|| {
                    let mut reader = std::hint::black_box(*slice);
                    let varint = VarInt::decode(&mut reader).expect("decode");
                    std::hint::black_box(varint.into_inner());
                });
            },
        );
    }
    group.finish();
}

fn bench_stream_encode(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c1_varint_stream_encode");
    // pseudo-random 62-bit values via LCG so each arm has the same input shape
    let values = generate_values(64 * 1024);
    for &target in STREAM_SIZES {
        group.throughput(Throughput::Bytes(target as u64));
        group.bench_with_input(
            BenchmarkId::new("proxima_quic_proto", target),
            &target,
            |bencher, &target| {
                let mut buffer = vec![0u8; target + varint::MAX_ENCODED_LEN];
                bencher.iter(|| {
                    let mut cursor = 0usize;
                    for &value in &values {
                        if cursor + varint::MAX_ENCODED_LEN > target {
                            break;
                        }
                        let written = varint::encode(value, &mut buffer[cursor..]).expect("encode");
                        cursor += written;
                    }
                    std::hint::black_box(cursor);
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("quinn_proto", target),
            &target,
            |bencher, &target| {
                let mut buffer = BytesMut::with_capacity(target + 8);
                bencher.iter(|| {
                    buffer.clear();
                    for &value in &values {
                        if buffer.len() + 8 > target {
                            break;
                        }
                        let varint = VarInt::from_u64(value).expect("from_u64");
                        varint.encode(&mut buffer);
                    }
                    std::hint::black_box(buffer.len());
                });
            },
        );
    }
    group.finish();
}

fn bench_stream_decode(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c1_varint_stream_decode");
    let values = generate_values(64 * 1024);
    for &target in STREAM_SIZES {
        // pre-encode the stream so the decode loop is the only thing measured
        let mut encoded = vec![0u8; target + varint::MAX_ENCODED_LEN];
        let mut cursor = 0usize;
        for &value in &values {
            if cursor + varint::MAX_ENCODED_LEN > target {
                break;
            }
            cursor += varint::encode(value, &mut encoded[cursor..]).expect("setup encode");
        }
        encoded.truncate(cursor);
        group.throughput(Throughput::Bytes(encoded.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("proxima_quic_proto", target),
            &encoded,
            |bencher, encoded| {
                bencher.iter(|| {
                    let mut input = encoded.as_slice();
                    let mut count = 0u64;
                    while !input.is_empty() {
                        let (value, consumed) = varint::decode(input).expect("decode");
                        std::hint::black_box(value);
                        input = &input[consumed..];
                        count += 1;
                    }
                    std::hint::black_box(count);
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("quinn_proto", target),
            &encoded,
            |bencher, encoded| {
                bencher.iter(|| {
                    let mut reader = encoded.as_slice();
                    let mut count = 0u64;
                    while reader.has_remaining() {
                        let varint = VarInt::decode(&mut reader).expect("decode");
                        std::hint::black_box(varint.into_inner());
                        count += 1;
                    }
                    std::hint::black_box(count);
                });
            },
        );
    }
    group.finish();
}

fn bench_adversarial_decode(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c1_varint_adversarial");
    // 8-byte length-class prefix followed by 7 bytes (1 short)
    let truncated: [u8; 7] = [0xc0, 0, 0, 0, 0, 0, 0];
    group.bench_function("proxima_quic_proto_truncated_reject", |bencher| {
        bencher.iter(|| {
            let result = varint::decode(std::hint::black_box(&truncated[..]));
            std::hint::black_box(result.is_err());
        });
    });
    group.bench_function("quinn_proto_truncated_reject", |bencher| {
        bencher.iter(|| {
            let mut reader = std::hint::black_box(&truncated[..]);
            let result = VarInt::decode(&mut reader);
            std::hint::black_box(result.is_err());
        });
    });
    group.finish();
}

fn generate_values(count: usize) -> Vec<u64> {
    let mut state: u64 = 0xdeadbeefcafebabe;
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state & varint::MAX_VALUE
        })
        .collect()
}

criterion_group!(
    benches,
    bench_single_encode,
    bench_single_decode,
    bench_stream_encode,
    bench_stream_decode,
    bench_adversarial_decode,
);
criterion_main!(benches);
