#![allow(clippy::unwrap_used, clippy::expect_used)]

//! C34 QPACK-decoder redesign — `decode_into` (borrowing, 0-alloc)
//! vs. `decode_bounded` (owned `Vec<DecodedField>`) vs. the incumbent.
//!
//! # Incumbent arm — verdict: no comparable micro surface
//!
//! `perf-targets.md`'s C34 row names `h3::qpack::Decoder` as the
//! incumbent reference. Checked directly against the vendored source
//! (`~/.cargo/registry/.../h3-0.0.8/src/lib.rs`): `h3`'s `qpack` is
//! `mod qpack;` — **private**, not `pub mod qpack;`. There is no
//! public entry point to drive `h3`'s QPACK decoder standalone from
//! outside its own crate. Per the sans-IO opt-sweep guidance ("if the
//! h3 crate's surface can't be driven standalone, mark the arm 'no
//! comparable micro surface'"): the honest compare for this component
//! is the existing end-to-end rekt->nginx-h3 client bench (a sibling
//! worktree's `examples/rekt_h3_load.rs`), which is what motivated
//! this redesign (allocation ~9%/core in `decode_bounded`) in the
//! first place. This bench file measures the two IN-CRATE arms this
//! redesign controls.
//!
//! # Arms
//!
//! - `decode_into` — borrowing engine, driven with a non-allocating
//!   counting sink.
//! - `decode_bounded` — owned-`Vec<DecodedField>` convenience wrapper
//!   over the same engine.
//!
//! # Input-size sweep
//!
//! - `small` — a single static-indexed field (`:status: 200`), the
//!   16 B/1-2-header floor.
//! - `nginx_response` (home-turf / 80%-case) — a 5-field synthesized
//!   `200` response shaped like nginx's default header set
//!   (`:status`, `server`, `date`, `content-type`, `content-length`)
//!   — the per-response hot path this redesign targets. `design-favors:
//!   decode_into` (0 setup-path Vec churn per response).
//!
//! # Alloc-count per arm
//!
//! A `stats_alloc` global allocator (bench-binary-local; does not
//! affect the crate's own build) reports allocations-per-call for
//! each arm/input-size pair once before the criterion timing groups
//! run — printed to stdout, read off `cargo bench` output.

use std::alloc::System;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use proxima_protocols::http3_codec::qpack::decoder::{DecodeError, decode_bounded, decode_into};
use proxima_protocols::http3_codec::qpack::encoder;
use stats_alloc::{Region, StatsAlloc};

#[global_allocator]
static GLOBAL: StatsAlloc<System> = StatsAlloc::system();

fn small_wire() -> Vec<u8> {
    let mut out = Vec::new();
    encoder::encode_refs([(b":status".as_slice(), b"200".as_slice())], &mut out)
        .expect("encode small field");
    out
}

fn nginx_response_wire() -> Vec<u8> {
    let mut out = Vec::new();
    encoder::encode_refs(
        [
            (b":status".as_slice(), b"200".as_slice()),
            (b"server".as_slice(), b"nginx/1.27.0".as_slice()),
            (
                b"date".as_slice(),
                b"Tue, 30 Jun 2026 00:00:00 GMT".as_slice(),
            ),
            (b"content-type".as_slice(), b"text/html".as_slice()),
            (b"content-length".as_slice(), b"612".as_slice()),
        ],
        &mut out,
    )
    .expect("encode nginx-shaped response header set");
    out
}

fn print_alloc_report(name: &str, wire: &[u8]) {
    let region = Region::new(&GLOBAL);

    let before = region.change();
    let mut field_count = 0usize;
    let mut scratch = [0u8; 256];
    let mut sink = |_name: &[u8], _value: &[u8]| -> Result<(), DecodeError> {
        field_count += 1;
        Ok(())
    };
    decode_into(wire, u64::MAX, &mut scratch, &mut sink).expect("decode_into");
    let into_allocs = (region.change() - before).allocations;

    let before = region.change();
    let decoded = decode_bounded(wire, u64::MAX).expect("decode_bounded");
    let bounded_allocs = (region.change() - before).allocations;

    println!(
        "[C34 alloc-count] {name}: fields={} decode_into.allocations={into_allocs} decode_bounded.allocations={bounded_allocs} (expect 0 / {})",
        decoded.len(),
        1 + 2 * decoded.len()
    );
    assert_eq!(field_count, decoded.len());
}

fn bench_decode_into(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("h3_qpack_c34_decode_into");
    group.measurement_time(Duration::from_secs(5));
    for (label, wire) in [
        ("small", small_wire()),
        ("nginx_response", nginx_response_wire()),
    ] {
        group.throughput(Throughput::Bytes(wire.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("decode_into", label),
            &wire,
            |bencher, buf| {
                let mut scratch = [0u8; 256];
                bencher.iter(|| {
                    let mut sink =
                        |_name: &[u8], _value: &[u8]| -> Result<(), DecodeError> { Ok(()) };
                    decode_into(std::hint::black_box(buf), u64::MAX, &mut scratch, &mut sink)
                        .expect("decode_into")
                });
            },
        );
    }
    group.finish();
}

fn bench_decode_bounded(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("h3_qpack_c34_decode_bounded");
    group.measurement_time(Duration::from_secs(5));
    for (label, wire) in [
        ("small", small_wire()),
        ("nginx_response", nginx_response_wire()),
    ] {
        group.throughput(Throughput::Bytes(wire.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("decode_bounded", label),
            &wire,
            |bencher, buf| {
                bencher.iter(|| {
                    decode_bounded(std::hint::black_box(buf), u64::MAX).expect("decode_bounded")
                });
            },
        );
    }
    group.finish();
}

fn bench_alloc_report(_criterion: &mut Criterion) {
    print_alloc_report("small", &small_wire());
    print_alloc_report("nginx_response", &nginx_response_wire());
}

criterion_group!(
    benches,
    bench_alloc_report,
    bench_decode_into,
    bench_decode_bounded
);
criterion_main!(benches);
