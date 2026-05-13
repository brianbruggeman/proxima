#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

//! HPACK Huffman codec microbench (RFC 7541 Appendix B).
//!
//! Apples-to-apples comparison: proxima native vs. h2's reference
//! implementation.
//!
//! - **h2-0.4.14** (4-bit nibble table) — vendored from the h2
//!   crate's private `hpack::huffman` module. Same algorithm h2
//!   uses today on the wire. Vendored because h2 keeps `hpack`
//!   `pub(crate)` even under `--features unstable`. MIT-licensed.

#[path = "vendored_h2/mod.rs"]
mod h2_huffman;

use std::time::Duration;

use bytes::BytesMut;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use proxima_protocols::hpack::{huffman_decode, huffman_encode};

fn make_input(label: &str) -> Vec<u8> {
    match label {
        "www_example_com" => b"www.example.com".to_vec(),
        "no_cache" => b"no-cache".to_vec(),
        "custom_value" => b"custom-value".to_vec(),
        "user_agent_chrome" => b"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36".to_vec(),
        "cookie_512b" => vec![b'a'; 512],
        "body_chunk_4kib" => vec![b'A'; 4 * 1024],
        _ => unreachable!(),
    }
}

fn encode_compare(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("hpack_huffman_encode");
    group.measurement_time(Duration::from_secs(3));
    for label in [
        "www_example_com",
        "no_cache",
        "custom_value",
        "user_agent_chrome",
        "cookie_512b",
        "body_chunk_4kib",
    ] {
        let input = make_input(label);
        group.throughput(Throughput::Bytes(input.len() as u64));

        group.bench_function(format!("proxima_native/{label}"), |bencher| {
            let mut out = Vec::with_capacity(input.len());
            bencher.iter(|| {
                out.clear();
                huffman_encode(std::hint::black_box(&input), &mut out);
                std::hint::black_box(out.len());
            });
        });

        // h2-0.4.14 vendored encoder: 40-bit u64 buffer, byte-at-a-
        // time drain, ENCODE_TABLE direct lookup. Same RFC 7541
        // wire output; difference is purely the per-byte bit-pack
        // mechanics.
        group.bench_function(format!("h2_crate/{label}"), |bencher| {
            let mut out = BytesMut::with_capacity(input.len() * 2);
            bencher.iter(|| {
                out.clear();
                h2_huffman::encode(std::hint::black_box(&input), &mut out);
                std::hint::black_box(out.len());
            });
        });
    }
    group.finish();
}

fn decode_compare(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("hpack_huffman_decode");
    group.measurement_time(Duration::from_secs(3));
    for label in [
        "www_example_com",
        "no_cache",
        "custom_value",
        "user_agent_chrome",
        "cookie_512b",
        "body_chunk_4kib",
    ] {
        let input = make_input(label);
        let mut encoded = Vec::new();
        huffman_encode(&input, &mut encoded);
        group.throughput(Throughput::Bytes(input.len() as u64));

        group.bench_function(format!("proxima_native/{label}"), |bencher| {
            let mut out = Vec::with_capacity(input.len());
            bencher.iter(|| {
                out.clear();
                huffman_decode(std::hint::black_box(&encoded), &mut out).expect("decode");
                std::hint::black_box(out.len());
            });
        });

        // h2-0.4.14 vendored reference impl: 4-bit nibble state
        // machine, two table lookups per byte. Same algorithm h2
        // ships today. No tree walk, no bit iteration.
        group.bench_function(format!("h2_crate/{label}"), |bencher| {
            let mut out = BytesMut::with_capacity(input.len() * 2);
            bencher.iter(|| {
                out.clear();
                let _ =
                    h2_huffman::decode(std::hint::black_box(&encoded), &mut out).expect("decode");
                std::hint::black_box(out.len());
            });
        });
    }
    group.finish();
}

criterion_group!(benches, encode_compare, decode_compare);
criterion_main!(benches);
