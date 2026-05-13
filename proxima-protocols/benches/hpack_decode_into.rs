#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

//! `decode` (owned `Bytes` callback) vs `decode_into` (borrowing
//! `FieldSink`) — the C34/C36-style borrowing-decode redesign applied
//! to HPACK. Same realistic request/response workloads as
//! `hpack_block.rs` plus a large-value shape (1KB/8KB cookie) and a
//! malformed-input arm per guiding-principles principle 11's sans-IO
//! bench design point (multiple sizes, adversarial input, home-turf
//! incumbent, alloc count per arm).
//!
//! ## Incumbent arm — scope boundary (already established, not
//! re-litigated here)
//!
//! `hpack_block.rs`'s own docstring already recorded the decision:
//! vendoring `h2` crate's FULL `hpack::Decoder` (its private
//! `Header`/`HeaderName`/`Table` machinery, ~1500 lines) to get a
//! literal block-level h2-crate arm is out of scope; the algorithmic
//! primitives `decode`/`decode_into` compose (integer varint, Huffman,
//! static-table lookup) already have real head-to-head h2-crate arms
//! in `hpack_integer.rs` / `hpack_huffman.rs` / `hpack_static_table.rs`
//! via `benches/vendored_h2/`. This file's incumbent comparison is
//! `decode` vs `decode_into` — the two engines THIS crate ships — plus
//! the alloc-count evidence (measured via `stats_alloc`, not
//! criterion) that motivates `decode_into`'s existence.

use std::time::Duration;

use bytes::{Bytes, BytesMut};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use proxima_protocols::hpack::{
    DecodeError, DynamicTable, decode_block as decode_owned, decode_into, encode_block,
};
use stats_alloc::{Region, StatsAlloc};

#[global_allocator]
static ALLOC: StatsAlloc<std::alloc::System> = StatsAlloc::system();

fn h(name: &'static [u8], value: &'static [u8]) -> (Bytes, Bytes) {
    (Bytes::from_static(name), Bytes::from_static(value))
}

struct Workload {
    label: &'static str,
    headers: Vec<(Bytes, Bytes)>,
}

fn workloads() -> Vec<Workload> {
    let large_cookie_1k = Bytes::from(vec![b'c'; 1024]);
    let large_cookie_8k = Bytes::from(vec![b'c'; 8 * 1024]);
    vec![
        Workload {
            label: "request_minimal",
            headers: vec![
                h(b":method", b"GET"),
                h(b":scheme", b"https"),
                h(b":path", b"/"),
                h(b":authority", b"example.com"),
            ],
        },
        Workload {
            label: "request_browser",
            headers: vec![
                h(b":method", b"GET"),
                h(b":scheme", b"https"),
                h(b":path", b"/index.html"),
                h(b":authority", b"www.example.com"),
                h(b"user-agent", b"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"),
                h(b"accept", b"text/html,application/xhtml+xml,application/xml;q=0.9"),
                h(b"accept-language", b"en-US,en;q=0.9"),
                h(b"accept-encoding", b"gzip, deflate"),
                h(b"cookie", b"session=abc123; user=42; locale=en_US"),
            ],
        },
        Workload {
            label: "request_api",
            headers: vec![
                h(b":method", b"POST"),
                h(b":scheme", b"https"),
                h(b":path", b"/api/v1/users"),
                h(b":authority", b"api.example.com"),
                h(b"content-type", b"application/json"),
                h(b"content-length", b"1024"),
                h(b"authorization", b"Bearer t0k3n-eyJhbGciOiJIUzI1NiJ9"),
                h(b"x-request-id", b"01HAB7PXY9X3K4M5N6P7Q8R9S0"),
            ],
        },
        Workload {
            label: "response_minimal",
            headers: vec![
                h(b":status", b"200"),
                h(b"content-type", b"application/json"),
                h(b"content-length", b"512"),
            ],
        },
        Workload {
            label: "response_cors",
            headers: vec![
                h(b":status", b"200"),
                h(b"content-type", b"application/json"),
                h(b"access-control-allow-origin", b"https://app.example.com"),
                h(b"vary", b"Origin"),
                h(b"server", b"proxima/0.1.0"),
            ],
        },
        Workload {
            label: "request_1kb_cookie",
            headers: vec![
                h(b":method", b"GET"),
                h(b":scheme", b"https"),
                h(b":path", b"/"),
                h(b":authority", b"example.com"),
                (Bytes::from_static(b"cookie"), large_cookie_1k),
            ],
        },
        Workload {
            label: "request_8kb_cookie",
            headers: vec![
                h(b":method", b"GET"),
                h(b":scheme", b"https"),
                h(b":path", b"/"),
                h(b":authority", b"example.com"),
                (Bytes::from_static(b"cookie"), large_cookie_8k),
            ],
        },
    ]
}

fn encode_workload(workload: &Workload) -> Bytes {
    let mut buffer = BytesMut::with_capacity(workload.headers.len() * 64 + 8192);
    let mut table = DynamicTable::new(4096);
    encode_block(workload.headers.clone(), &mut table, &mut buffer);
    buffer.freeze()
}

fn decode_compare(criterion: &mut Criterion) {
    print_alloc_report();
    let mut group = criterion.benchmark_group("hpack_decode_owned_vs_borrowing");
    group.measurement_time(Duration::from_secs(2));
    for workload in workloads() {
        let encoded = encode_workload(&workload);
        group.throughput(Throughput::Bytes(encoded.len() as u64));

        group.bench_function(format!("decode_owned/{}", workload.label), |bencher| {
            let mut table = DynamicTable::new(4096);
            bencher.iter(|| {
                let mut count = 0usize;
                decode_owned(std::hint::black_box(&encoded), &mut table, 4096, |_, _| {
                    count += 1;
                })
                .expect("decode");
                std::hint::black_box(count);
            });
        });

        group.bench_function(format!("decode_into/{}", workload.label), |bencher| {
            let mut table = DynamicTable::new(4096);
            let mut scratch = [0u8; 16_384];
            bencher.iter(|| {
                let mut count = 0usize;
                let mut sink = |_: &[u8], _: &[u8]| -> Result<(), DecodeError> {
                    count += 1;
                    Ok(())
                };
                decode_into(
                    std::hint::black_box(&encoded),
                    &mut table,
                    4096,
                    &mut scratch,
                    &mut sink,
                )
                .expect("decode_into");
                std::hint::black_box(count);
            });
        });
    }
    group.finish();
}

/// Adversarial arm: truncated block (cuts a huffman string's payload
/// short). Both engines must reject cleanly — no panic, no OOM. Times
/// the rejection path since a hostile peer sending malformed HEADERS
/// repeatedly (protocol-fuzzing, DoS probing) makes rejection latency
/// a real cost, not just a correctness footnote.
fn decode_malformed(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("hpack_decode_malformed_input");
    group.measurement_time(Duration::from_secs(2));

    let workload = &workloads()[1]; // request_browser — has a huffman-eligible value
    let mut full = encode_workload(workload).to_vec();
    full.truncate(full.len().saturating_sub(4)); // cut mid-field
    let truncated = Bytes::from(full);
    group.throughput(Throughput::Bytes(truncated.len() as u64));

    group.bench_function("decode_owned/truncated", |bencher| {
        let mut table = DynamicTable::new(4096);
        bencher.iter(|| {
            let result = decode_owned(
                std::hint::black_box(&truncated),
                &mut table,
                4096,
                |_, _| {},
            );
            std::hint::black_box(result.is_err());
        });
    });

    group.bench_function("decode_into/truncated", |bencher| {
        let mut table = DynamicTable::new(4096);
        let mut scratch = [0u8; 16_384];
        bencher.iter(|| {
            let mut sink = |_: &[u8], _: &[u8]| -> Result<(), DecodeError> { Ok(()) };
            let result = decode_into(
                std::hint::black_box(&truncated),
                &mut table,
                4096,
                &mut scratch,
                &mut sink,
            );
            std::hint::black_box(result.is_err());
        });
    });

    group.finish();
}

/// Alloc-count report (not a criterion measurement — a direct
/// `stats_alloc` snapshot printed alongside the ns/op numbers so the
/// discipline-log row can cite both from ONE bench run). Mirrors
/// `proxima-h3-proto`'s `bench_c34_decode.rs::print_alloc_report`
/// pattern (P1 RISC reuse).
fn print_alloc_report() {
    println!("\n--- hpack decode alloc report (stats_alloc, 1 iteration per workload) ---");
    for workload in workloads() {
        let encoded = encode_workload(&workload);

        let mut owned_table = DynamicTable::new(4096);
        let mut via_owned: Vec<(Bytes, Bytes)> = Vec::with_capacity(16);
        let region = Region::new(&ALLOC);
        let before = region.change();
        decode_owned(&encoded, &mut owned_table, 4096, |name, value| {
            via_owned.push((name, value));
        })
        .expect("decode");
        let after = region.change();
        let owned_allocs = after.allocations - before.allocations;

        let mut into_table = DynamicTable::new(4096);
        let mut scratch = [0u8; 16_384];
        let mut field_count = 0usize;
        let before = region.change();
        let mut sink = |_: &[u8], _: &[u8]| -> Result<(), DecodeError> {
            field_count += 1;
            Ok(())
        };
        decode_into(&encoded, &mut into_table, 4096, &mut scratch, &mut sink).expect("decode_into");
        let after = region.change();
        let into_allocs = after.allocations - before.allocations;

        println!(
            "  {:<20} encoded_bytes={:<6} fields={:<2} decode(owned)_allocs={:<3} decode_into(borrowing)_allocs={:<3}",
            workload.label,
            encoded.len(),
            field_count,
            owned_allocs,
            into_allocs,
        );
    }
    println!("--- end alloc report ---\n");
}

criterion_group!(benches, decode_compare, decode_malformed);
criterion_main!(benches);
