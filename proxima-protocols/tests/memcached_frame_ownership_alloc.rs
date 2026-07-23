#![allow(clippy::unwrap_used, clippy::expect_used)]

//! Deterministic allocation-count gate for `OwnFrame::own_frame`'s
//! per-request owned-`Vec` lift on the memcached `FrameCodec` path — the
//! CI-facing half of `benches/bench_memcached_frame_ownership.rs` (see its
//! module doc for the full claim and workload rationale;
//! `docs/bench-campaigns/2026-07-23-frame-ownership-baseline.md` carries the
//! sibling ns/op numbers this file does NOT gate on).
//!
//! Allocation COUNTS, unlike nanosecond timings, are a direct product of how
//! many `Vec`s a fixed code path allocates — they do not vary run-to-run or
//! host-to-host the way wall-clock measurements do (the baseline doc's own
//! CoV band is evidence of that gap). That makes alloc count the stable,
//! CI-safe half of this bench's evidence; a hard, hardware-independent
//! `assert_eq!` here is honest where a timing threshold would be flaky.
//!
//! Reuses `stats_alloc::{Region, StatsAlloc}` — the SAME counting substrate
//! the bench's own `print_alloc_report` uses (RISC, P1) — rather than a
//! hand-rolled `AtomicUsize` allocator (the sibling pattern in
//! `tests/pgwire_codec_integration/alloc_counter.rs` counts every
//! `realloc()` as an allocation event, which is right for THAT test's
//! zero-allocation claim but would double-count `Vec` growth reallocations
//! against the `.allocations`-only numbers this file's baseline doc reports;
//! matching the bench's own counting semantics exactly is the point).

use bytes::Bytes;
use proxima_codec::FrameCodec;
use proxima_protocols::codec_pipe::OwnFrame;
use proxima_protocols::memcached::frame_codec::MemcachedCodec;
use stats_alloc::{Region, StatsAlloc};

#[global_allocator]
static ALLOC: StatsAlloc<std::alloc::System> = StatsAlloc::system();

const DEFAULT_MAX_MESSAGE_BYTES: usize = 128 * 1024;

/// Mirrors `benches/bench_memcached_frame_ownership.rs`'s own `get_16_bytes`
/// workload byte-for-byte, so the count asserted here traces to the exact
/// row measured in the baseline doc.
fn get_16_bytes() -> Vec<u8> {
    b"get k123456789\r\n".to_vec()
}

fn set_with_value(value_len: usize) -> Vec<u8> {
    let mut wire = format!("set mykey 0 60 {value_len}\r\n").into_bytes();
    wire.extend(std::iter::repeat_n(b'x', value_len));
    wire.extend_from_slice(b"\r\n");
    wire
}

fn multi_get(key_count: usize) -> Vec<u8> {
    let mut wire = b"get".to_vec();
    for index in 0..key_count {
        wire.push(b' ');
        wire.extend_from_slice(format!("key{index:04}").as_bytes());
    }
    wire.extend_from_slice(b"\r\n");
    wire
}

fn malformed_unknown_verb() -> Vec<u8> {
    b"bogus mykey\r\n".to_vec()
}

fn malformed_oversized_partial() -> Vec<u8> {
    b"set mykey 0 60 999999\r\nabc".to_vec()
}

/// Parses `wire` once (outside the measured window) then counts only the
/// allocations `MemcachedCodec::own_frame` itself performs.
fn own_frame_alloc_count(wire: &[u8], max_message_bytes: usize) -> usize {
    let codec: MemcachedCodec = MemcachedCodec::new(max_message_bytes);
    let raw = Bytes::from(wire.to_vec());
    let (frame, _consumed) = codec.parse_frame(&raw).expect("every workload resolves Ok");

    let region = Region::new(&ALLOC);
    let before = region.change();
    let owned = MemcachedCodec::own_frame(&raw, &frame);
    let after = region.change();
    drop(owned);

    after.allocations - before.allocations
}

#[test]
fn own_frame_allocation_counts_match_the_measured_baseline() {
    let cases: &[(&str, Vec<u8>, usize, usize)] = &[
        ("get_16b", get_16_bytes(), DEFAULT_MAX_MESSAGE_BYTES, 1),
        ("set_1kb_value", set_with_value(1024), DEFAULT_MAX_MESSAGE_BYTES, 1),
        ("set_8kb_value", set_with_value(8 * 1024), DEFAULT_MAX_MESSAGE_BYTES, 1),
        ("set_64kb_value", set_with_value(64 * 1024), DEFAULT_MAX_MESSAGE_BYTES, 1),
        ("multiget_20keys", multi_get(20), DEFAULT_MAX_MESSAGE_BYTES, 1),
        ("malformed_unknown_verb", malformed_unknown_verb(), DEFAULT_MAX_MESSAGE_BYTES, 0),
        ("malformed_oversized", malformed_oversized_partial(), 8, 0),
    ];

    for (label, wire, max_message_bytes, expected_allocs) in cases {
        let observed = own_frame_alloc_count(wire, *max_message_bytes);
        assert_eq!(
            observed, *expected_allocs,
            "{label}: own_frame allocation count drifted from the measured baseline \
             (docs/bench-campaigns/2026-07-23-frame-ownership-baseline.md) — expected \
             {expected_allocs}, observed {observed}. If this is an intentional codec change \
             (e.g. the zero-copy `Bytes::slice_ref` flip), update both this assertion and the \
             baseline doc together."
        );
    }
}
