#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

//! HPACK variable-length integer codec microbench (RFC 7541 §5.1).
//!
//! Covers the three encoding-length regimes:
//! - 1 byte: value fits in the N-bit prefix
//! - 2 bytes: one continuation byte
//! - 3+ bytes: multi-continuation
//!
//! Plus boundary cases — RFC examples (10, 1337, 42) — and 268_435_710,
//! the largest integer an RFC 7541 decoder accepts with an 8-bit prefix
//! (the 5-octet limit: values needing a 6th octet are an IntegerOverflow
//! error per RFC 7541 §5.1, matching h2's decoder).
//!
//! Apples-to-apples vs h2-0.4.14 vendored encode_int / decode_int.

#[path = "vendored_h2/mod.rs"]
mod h2_vendored;

use std::time::Duration;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use proxima_protocols::hpack::{decode_integer, encode_integer};

fn encode_integer_bench(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("hpack_integer_encode");
    group.measurement_time(Duration::from_secs(3));
    group.throughput(Throughput::Elements(1));
    // (value, prefix_bits) cases spanning byte-length regimes
    let cases: &[(u32, u8, &str)] = &[
        (10, 5, "rfc_c_1_1_10_5b"),
        (1337, 5, "rfc_c_1_2_1337_5b"),
        (42, 8, "rfc_c_1_3_42_8b"),
        (0, 5, "zero_5b"),
        (31, 5, "boundary_5b"),
        (128, 7, "boundary_7b_2byte"),
        (1_000_000, 5, "1m_5b"),
        (268_435_710, 8, "hpack_max5_8b"),
    ];
    for (value, prefix_bits, label) in cases {
        group.bench_function(format!("proxima_native/{label}"), |bencher| {
            let mut out = Vec::with_capacity(8);
            bencher.iter(|| {
                out.clear();
                encode_integer(
                    std::hint::black_box(*value),
                    std::hint::black_box(*prefix_bits),
                    0,
                    &mut out,
                );
                std::hint::black_box(out.len());
            });
        });
        group.bench_function(format!("h2_crate/{label}"), |bencher| {
            let mut out = Vec::with_capacity(8);
            bencher.iter(|| {
                out.clear();
                h2_vendored::integer::encode_int(
                    std::hint::black_box(*value as usize),
                    std::hint::black_box(*prefix_bits as usize),
                    0,
                    &mut out,
                );
                std::hint::black_box(out.len());
            });
        });
    }
    group.finish();
}

fn decode_integer_bench(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("hpack_integer_decode");
    group.measurement_time(Duration::from_secs(3));
    group.throughput(Throughput::Elements(1));
    let cases: &[(u32, u8, &str)] = &[
        (10, 5, "rfc_c_1_1_10_5b"),
        (1337, 5, "rfc_c_1_2_1337_5b"),
        (42, 8, "rfc_c_1_3_42_8b"),
        (0, 5, "zero_5b"),
        (31, 5, "boundary_5b"),
        (128, 7, "boundary_7b_2byte"),
        (1_000_000, 5, "1m_5b"),
        (268_435_710, 8, "hpack_max5_8b"),
    ];
    for (value, prefix_bits, label) in cases {
        // Pre-encode the wire form so the bench only measures decode.
        let mut wire = Vec::with_capacity(8);
        encode_integer(*value, *prefix_bits, 0, &mut wire);
        group.bench_function(format!("proxima_native/{label}"), |bencher| {
            bencher.iter(|| {
                let (decoded, consumed) = decode_integer(
                    std::hint::black_box(&wire),
                    std::hint::black_box(*prefix_bits),
                )
                .expect("decode");
                std::hint::black_box((decoded, consumed));
            });
        });
        group.bench_function(format!("h2_crate/{label}"), |bencher| {
            bencher.iter(|| {
                let (decoded, consumed) = h2_vendored::integer::decode_int(
                    std::hint::black_box(&wire),
                    std::hint::black_box(*prefix_bits),
                )
                .expect("decode");
                std::hint::black_box((decoded, consumed));
            });
        });
    }
    group.finish();
}

criterion_group!(benches, encode_integer_bench, decode_integer_bench);
criterion_main!(benches);
