#![allow(clippy::unwrap_used, clippy::expect_used)]

//! `PartSource` step-at-a-time vs `Request::from_source` drain —
//! `docs/proxima-pipe/part-source-sink-design.md` step 1's load-bearing
//! bench. Both arms decode the SAME QPACK field section via the SAME
//! unmodified `decode_into` engine (`HeaderBlockPartSource`); the only
//! difference is whether the caller steps the source directly or drains it
//! into an owned `Request`.
//!
//! # Incumbent arm
//!
//! There is no third-party "borrowed HTTP/3 header source" to compare
//! against on its own design point (h3's `qpack` module is private — see
//! `bench_c34_decode.rs`'s note on the same crate). The comparison this
//! bench proves is in-crate: source-stepping vs the drain it replaces.
//!
//! # Input-size sweep
//!
//! - `small` — one pseudo-header pair (`:method`/`:path` only), the
//!   minimal request shape.
//! - `request` (home-turf / 80%-case) — `:method` + `:path` + 2 ordinary
//!   headers (user-agent, accept), a browser/curl-shaped GET.
//!
//! # Alloc-count per arm
//!
//! A `stats_alloc` global allocator (bench-binary-local) reports
//! allocations-per-call for each arm/input-size pair once before the
//! criterion timing groups run.

use std::alloc::System;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use proxima_protocols::http3_codec::qpack::encoder;
use proxima_protocols::http3_codec::qpack::part_source::HeaderBlockPartSource;
use proxima_primitives::pipe::part::PartSource;
use proxima_primitives::pipe::request::Request;
use stats_alloc::{Region, StatsAlloc};

#[global_allocator]
static GLOBAL: StatsAlloc<System> = StatsAlloc::system();

fn small_wire() -> Vec<u8> {
    let mut out = Vec::new();
    encoder::encode_refs(
        [
            (b":method".as_slice(), b"GET".as_slice()),
            (b":path".as_slice(), b"/".as_slice()),
        ],
        &mut out,
    )
    .expect("encode minimal request header set");
    out
}

fn request_wire() -> Vec<u8> {
    let mut out = Vec::new();
    encoder::encode_refs(
        [
            (b":method".as_slice(), b"GET".as_slice()),
            (b":path".as_slice(), b"/v1/items".as_slice()),
            (b"user-agent".as_slice(), b"curl/8.7.1".as_slice()),
            (b"accept".as_slice(), b"application/json".as_slice()),
        ],
        &mut out,
    )
    .expect("encode request-shaped header set");
    out
}

fn print_alloc_report(name: &str, wire: &[u8]) {
    let region = Region::new(&GLOBAL);

    let before = region.change();
    let mut scratch = [0u8; 256];
    let mut source = HeaderBlockPartSource::new(wire, u64::MAX, &mut scratch).expect("decode");
    let mut parts_seen = 0usize;
    while source.next().is_some() {
        parts_seen += 1;
    }
    let step_allocs = (region.change() - before).allocations;

    let before = region.change();
    let mut drain_scratch = [0u8; 256];
    let mut source_for_drain =
        HeaderBlockPartSource::new(wire, u64::MAX, &mut drain_scratch).expect("decode");
    let request = Request::from_source(&mut source_for_drain);
    let drain_allocs = (region.change() - before).allocations;

    println!(
        "[part_source alloc-count] {name}: parts={parts_seen} step.allocations={step_allocs} \
         drain.allocations={drain_allocs} method={:?} path_len={}",
        request.method,
        request.path.len(),
    );
}

fn bench_step_source(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("h3_part_source_step");
    group.measurement_time(Duration::from_secs(5));
    for (label, wire) in [("small", small_wire()), ("request", request_wire())] {
        group.throughput(Throughput::Bytes(wire.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("step_to_exhaustion", label),
            &wire,
            |bencher, buf| {
                let mut scratch = [0u8; 256];
                bencher.iter(|| {
                    let mut source = HeaderBlockPartSource::new(
                        std::hint::black_box(buf),
                        u64::MAX,
                        &mut scratch,
                    )
                    .expect("decode");
                    let mut count = 0usize;
                    while source.next().is_some() {
                        count += 1;
                    }
                    count
                });
            },
        );
    }
    group.finish();
}

fn bench_drain_to_request(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("h3_part_source_drain_to_request");
    group.measurement_time(Duration::from_secs(5));
    for (label, wire) in [("small", small_wire()), ("request", request_wire())] {
        group.throughput(Throughput::Bytes(wire.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("drain_to_request", label),
            &wire,
            |bencher, buf| {
                let mut scratch = [0u8; 256];
                bencher.iter(|| {
                    let mut source = HeaderBlockPartSource::new(
                        std::hint::black_box(buf),
                        u64::MAX,
                        &mut scratch,
                    )
                    .expect("decode");
                    Request::from_source(&mut source)
                });
            },
        );
    }
    group.finish();
}

fn bench_alloc_report(_criterion: &mut Criterion) {
    print_alloc_report("small", &small_wire());
    print_alloc_report("request", &request_wire());
}

criterion_group!(
    benches,
    bench_alloc_report,
    bench_step_source,
    bench_drain_to_request
);
criterion_main!(benches);
