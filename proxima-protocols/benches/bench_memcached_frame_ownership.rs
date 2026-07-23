#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

//! BASELINE bench for `OwnFrame::own_frame`'s per-request owned-`Vec` copy
//! on the memcached `FrameCodec` path — guiding-principles P11 (sans-IO
//! extreme benching), P18 (profile before the finding), P19 (evidence
//! ladder), P1 (RISC reuse).
//!
//! ## The claim under test
//!
//! `MemcachedCodec::own_frame` (`src/memcached/frame_codec.rs`) lifts a
//! borrowed [`MemcachedFrame::Request(Command<'_>)`] into
//! [`MemcachedRequest`] (`src/memcached/pipe_contract.rs`), whose fields are
//! `Vec<u8>` / `Vec<Vec<u8>>` — every `key`/`value`/multi-`keys` is
//! `.to_vec()`'d out of the `Bytes` window it was parsed from. Sibling
//! codecs on the SAME `OwnFrame` seam (`grpc_framing::frame_codec_pipe`,
//! `http1_codec::frame_codec_pipe`, `websocket_frame::frame_codec_pipe`)
//! instead do `source.slice_ref(field)` — an `Arc` refcount bump, zero
//! bytes copied. Memcached is the OUTLIER on its own seam. The hypothesis:
//! this per-request copy is material. This file produces the number that
//! confirms or refutes it; it changes NO production type.
//!
//! ## Attribution design (P19 — measurement, not bundling)
//!
//! Four stages are benched SEPARATELY so the owned lift's cost is
//! attributed, not folded into a single number:
//!
//! 1. `memcached_parse_frame` — `MemcachedCodec::parse_frame` alone
//!    (borrowed; expected ~0 allocs on the happy path).
//! 2. `memcached_own_frame_owned_vec` — `MemcachedCodec::own_frame` alone
//!    (the CURRENT production lift), timed via `iter_custom` so the
//!    untimed `parse_frame` call inside the loop never bleeds into the
//!    measured window.
//! 3. `memcached_own_frame_zero_copy` — a BENCH-LOCAL counterfactual
//!    `own_frame` doing `Bytes::slice_ref` instead of `.to_vec()` (the
//!    design point `grpc_framing`/`http1_codec`/`websocket_frame` already
//!    ship on the same seam) — the home-turf incumbent arm for THIS
//!    component: not a hypothetical, a pattern already landed elsewhere in
//!    this crate. Proves the codec-owning
//!    `MemcachedRequest`'s `Vec<u8>` field is what causes the allocation and
//!    what its `Bytes`-typed twin would cost instead.
//! 4. `memcached_frame_end_to_end` — the LITERAL pipe
//!    `proxima_listen::any::FramedAny::drive` builds per connection,
//!    `AndThen<FrameCodecPipe<MemcachedCodec>, OnFrame<App>>`, called
//!    directly with a `Bytes` input (no socket — sans-IO per P11; `Pipe`
//!    is polled with a no-op waker, matching
//!    `grpc_framing::frame_codec_pipe`'s own test `block_on`). This is
//!    "bytes in, handler invoked" (P18's sharpening of P11: allocation
//!    must be measured on the real path the component lives in, not only
//!    an isolated micro).
//!
//! ## Explicit scope boundary (named, not hidden — P18)
//!
//! `memcached_frame_end_to_end`'s `App` is `TrivialMemcachedApp`, a
//! zero-alloc stand-in (touches the parsed fields' lengths so the match
//! arms cannot be dead-code-eliminated) — NOT
//! `proxima-memcached::MemcachedFramedApp` (that type additionally builds
//! a `proxima_primitives::pipe::request::Request` and dispatches through a
//! business `Pipe`, which lives in a crate this bench does not depend on
//! and layers its own, separately-measurable allocation cost). This bench
//! also does NOT reproduce `FramedAny::drive`'s own per-loop-attempt
//! `Bytes::copy_from_slice(&buf)` (the re-parse-from-byte-zero buffer
//! copy the generic driver does every attempt) — that is a DIFFERENT,
//! already-known cost orthogonal to `OwnFrame::own_frame`'s claim and is
//! out of scope for this row.
//!
//! ## Workloads
//!
//! `get_16b` (single-key get, exactly 16 wire bytes), `set_1kb` /
//! `set_8kb` / `set_64kb` (the value size `own_frame`'s copy scales
//! with), `multiget_20keys` (exercises `Vec<Vec<u8>>` — one allocation
//! PER KEY, not just per payload), `malformed_unknown_verb` (parse-time
//! rejection — allocates, see the alloc report), `malformed_oversized`
//! (parse-time rejection that does NOT allocate — the frame is folded
//! into `Violation::MessageTooLarge` before any owning ever happens).
//!
//! ## Reuse (P1 RISC)
//!
//! `stats_alloc::{Region, StatsAlloc}` — the SAME alloc-count substrate
//! `hpack_decode_into.rs`/`bench_c34_decode.rs`/`bench_part_source.rs`
//! already use in this crate; no new counting allocator is hand-rolled
//! (see also `tests/pgwire_codec_integration/alloc_counter.rs` for the
//! sibling `AtomicUsize`-based pattern this file does NOT need — criterion
//! benches already own their process, so a `#[global_allocator]` here is
//! the direct, dependency-free answer).

use std::future::Future;
use std::time::{Duration, Instant};

use bytes::Bytes;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use stats_alloc::{Region, StatsAlloc};

use proxima_codec::FrameCodec;
use proxima_primitives::pipe::{AndThen, Pipe};
use proxima_protocols::codec_pipe::{FrameCodecPipe, OnFrame, OwnFrame};
use proxima_protocols::memcached::frame_codec::{MemcachedCodec, MemcachedFrame, MemcachedOwnedFrame, NeedMoreBytes, Violation};
use proxima_protocols::memcached::pipe_contract::{MemcachedRequest, iter_keys};
use proxima_protocols::memcached::{Command, StoreMode};

#[global_allocator]
static ALLOC: StatsAlloc<std::alloc::System> = StatsAlloc::system();

/// Dependency-free executor for the always-ready pipe futures — mirrors
/// `grpc_framing::frame_codec_pipe`'s own test helper (P1 RISC reuse, not
/// a fresh block_on hand-rolled per file).
fn block_on<Fut: Future>(future: Fut) -> Fut::Output {
    let mut pinned = core::pin::pin!(future);
    let mut context = core::task::Context::from_waker(core::task::Waker::noop());
    loop {
        if let core::task::Poll::Ready(output) = pinned.as_mut().poll(&mut context) {
            return output;
        }
    }
}

struct Workload {
    label: &'static str,
    bytes: Vec<u8>,
    max_message_bytes: usize,
}

fn get_16_bytes() -> Vec<u8> {
    let wire = b"get k123456789\r\n".to_vec(); // 4 + 10 + 2 = 16 bytes exactly
    assert_eq!(wire.len(), 16, "get_16b workload must be exactly 16 wire bytes");
    wire
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

/// A still-incomplete `set` (declared value length far exceeds the
/// buffered bytes) whose TOTAL buffered length already exceeds the tiny
/// `max_message_bytes` cap this workload is paired with — folds into
/// `Violation::MessageTooLarge` at parse time rather than waiting for
/// more bytes (see `frame_codec.rs`'s own
/// `parse_frame_partial_value_over_the_cap_is_a_message_too_large_violation`
/// test, mirrored here as a bench workload).
fn malformed_oversized_partial() -> Vec<u8> {
    b"set mykey 0 60 999999\r\nabc".to_vec()
}

const DEFAULT_MAX_MESSAGE_BYTES: usize = 128 * 1024;

fn workloads() -> Vec<Workload> {
    vec![
        Workload {
            label: "get_16b",
            bytes: get_16_bytes(),
            max_message_bytes: DEFAULT_MAX_MESSAGE_BYTES,
        },
        Workload {
            label: "set_1kb_value",
            bytes: set_with_value(1024),
            max_message_bytes: DEFAULT_MAX_MESSAGE_BYTES,
        },
        Workload {
            label: "set_8kb_value",
            bytes: set_with_value(8 * 1024),
            max_message_bytes: DEFAULT_MAX_MESSAGE_BYTES,
        },
        Workload {
            label: "set_64kb_value",
            bytes: set_with_value(64 * 1024),
            max_message_bytes: DEFAULT_MAX_MESSAGE_BYTES,
        },
        Workload {
            label: "multiget_20keys",
            bytes: multi_get(20),
            max_message_bytes: DEFAULT_MAX_MESSAGE_BYTES,
        },
        Workload {
            label: "malformed_unknown_verb",
            bytes: malformed_unknown_verb(),
            max_message_bytes: DEFAULT_MAX_MESSAGE_BYTES,
        },
        Workload {
            label: "malformed_oversized",
            bytes: malformed_oversized_partial(),
            max_message_bytes: 8,
        },
    ]
}

// ---------------------------------------------------------------------
// Counterfactual zero-copy `own_frame` — the home-turf incumbent arm.
// Mirrors `grpc_framing::frame_codec_pipe::OwnFrame::own_frame`'s
// `Bytes::slice_ref` design exactly; a bench-local type, NOT a change to
// `pipe_contract.rs`.
// ---------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
enum ZeroCopyRequest {
    Get {
        keys: Vec<Bytes>,
        gets: bool,
    },
    Store {
        mode: StoreMode,
        key: Bytes,
        flags: u32,
        exptime: u32,
        value: Bytes,
        noreply: bool,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ZeroCopyOwnedFrame {
    Request(ZeroCopyRequest),
    Violation(Violation),
}

fn split_keys_zero_copy(source: &Bytes, joined: &[u8]) -> Vec<Bytes> {
    joined
        .split(|&byte| byte == b' ')
        .filter(|slice| !slice.is_empty())
        .map(|slice| source.slice_ref(slice))
        .collect()
}

fn own_frame_zero_copy(source: &Bytes, frame: &MemcachedFrame<'_>) -> ZeroCopyOwnedFrame {
    match frame {
        MemcachedFrame::Request(Command::Get { keys, gets }) => {
            ZeroCopyOwnedFrame::Request(ZeroCopyRequest::Get {
                keys: split_keys_zero_copy(source, keys),
                gets: *gets,
            })
        }
        MemcachedFrame::Request(Command::Store {
            mode,
            key,
            flags,
            exptime,
            value,
            noreply,
        }) => ZeroCopyOwnedFrame::Request(ZeroCopyRequest::Store {
            mode: *mode,
            key: source.slice_ref(key),
            flags: *flags,
            exptime: *exptime,
            value: source.slice_ref(value),
            noreply: *noreply,
        }),
        MemcachedFrame::Violation(kind) => ZeroCopyOwnedFrame::Violation(*kind),
        other => panic!(
            "baseline bench's zero-copy counterfactual covers only Get/Store/Violation \
             (this file's workloads never produce anything else); got {other:?}"
        ),
    }
}

// ---------------------------------------------------------------------
// End-to-end stage: the literal `AndThen<FrameCodecPipe<C>, OnFrame<App>>`
// `FramedAny::drive` builds, minus the socket loop and admission wrapper
// (see module doc's scope boundary).
// ---------------------------------------------------------------------

#[derive(Debug)]
struct TrivialAppError;

impl From<NeedMoreBytes> for TrivialAppError {
    fn from(_: NeedMoreBytes) -> Self {
        TrivialAppError
    }
}

/// Zero-alloc stand-in business handler — touches every parsed field's
/// length so the compiler cannot dead-code-eliminate the match, without
/// itself allocating (see module doc's scope boundary on why this is NOT
/// `proxima-memcached::MemcachedFramedApp`).
#[derive(Clone, Copy, Default)]
struct TrivialMemcachedApp;

impl Pipe for TrivialMemcachedApp {
    type In = MemcachedOwnedFrame;
    type Out = usize;
    type Err = TrivialAppError;

    fn call(&self, input: MemcachedOwnedFrame) -> impl Future<Output = Result<usize, TrivialAppError>> {
        async move {
            let touched = match &input {
                MemcachedOwnedFrame::Request(MemcachedRequest::Get { keys, .. }) => {
                    iter_keys(keys).map(|key| key.len()).sum()
                }
                MemcachedOwnedFrame::Request(MemcachedRequest::Store { key, value, .. }) => {
                    key.len() + value.len()
                }
                MemcachedOwnedFrame::Request(_) => 0,
                MemcachedOwnedFrame::Violation(_) => 0,
            };
            Ok(touched)
        }
    }
}

fn end_to_end_pipe(codec: MemcachedCodec) -> AndThen<FrameCodecPipe<MemcachedCodec>, OnFrame<TrivialMemcachedApp>> {
    AndThen::new(FrameCodecPipe::new(codec), OnFrame::new(TrivialMemcachedApp))
}

// ---------------------------------------------------------------------
// Criterion arms
// ---------------------------------------------------------------------

fn bench_parse_frame(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("memcached_parse_frame");
    group.measurement_time(Duration::from_millis(700));
    group.sample_size(30);
    for workload in workloads() {
        let codec = MemcachedCodec::new(workload.max_message_bytes);
        let raw = Bytes::from(workload.bytes.clone());
        group.throughput(Throughput::Bytes(raw.len() as u64));
        group.bench_function(workload.label, |bencher| {
            bencher.iter(|| {
                let (frame, consumed) = codec
                    .parse_frame(std::hint::black_box(&raw))
                    .expect("every workload resolves Ok (Request or Violation), never a hard Err");
                std::hint::black_box((frame, consumed));
            });
        });
    }
    group.finish();
}

fn bench_own_frame_owned_vec(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("memcached_own_frame_owned_vec");
    group.measurement_time(Duration::from_millis(700));
    group.sample_size(30);
    for workload in workloads() {
        let codec = MemcachedCodec::new(workload.max_message_bytes);
        let raw = Bytes::from(workload.bytes.clone());
        group.throughput(Throughput::Bytes(raw.len() as u64));
        group.bench_function(workload.label, |bencher| {
            bencher.iter_custom(|iterations| {
                let mut elapsed = Duration::ZERO;
                for _ in 0..iterations {
                    let (frame, _consumed) = codec.parse_frame(&raw).expect("parses or violates");
                    let start = Instant::now();
                    let owned = std::hint::black_box(MemcachedCodec::own_frame(&raw, &frame));
                    elapsed += start.elapsed();
                    std::hint::black_box(owned);
                }
                elapsed
            });
        });
    }
    group.finish();
}

fn bench_own_frame_zero_copy(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("memcached_own_frame_zero_copy");
    group.measurement_time(Duration::from_millis(700));
    group.sample_size(30);
    for workload in workloads() {
        let codec = MemcachedCodec::new(workload.max_message_bytes);
        let raw = Bytes::from(workload.bytes.clone());
        group.throughput(Throughput::Bytes(raw.len() as u64));
        group.bench_function(workload.label, |bencher| {
            bencher.iter_custom(|iterations| {
                let mut elapsed = Duration::ZERO;
                for _ in 0..iterations {
                    let (frame, _consumed) = codec.parse_frame(&raw).expect("parses or violates");
                    let start = Instant::now();
                    let owned = std::hint::black_box(own_frame_zero_copy(&raw, &frame));
                    elapsed += start.elapsed();
                    std::hint::black_box(owned);
                }
                elapsed
            });
        });
    }
    group.finish();
}

fn bench_end_to_end(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("memcached_frame_end_to_end");
    group.measurement_time(Duration::from_millis(700));
    group.sample_size(30);
    for workload in workloads() {
        let pipe = end_to_end_pipe(MemcachedCodec::new(workload.max_message_bytes));
        let raw = Bytes::from(workload.bytes.clone());
        group.throughput(Throughput::Bytes(raw.len() as u64));
        group.bench_function(workload.label, |bencher| {
            bencher.iter(|| {
                let outcome = block_on(Pipe::call(&pipe, std::hint::black_box(raw.clone())));
                let produced =
                    outcome.expect("every workload's frame is complete (Request or Violation), never Incomplete");
                std::hint::black_box(produced);
            });
        });
    }
    group.finish();
}

/// Alloc-count report (not a criterion measurement — a direct
/// `stats_alloc` snapshot printed alongside the ns/op numbers so the
/// discipline-log row can cite both from ONE bench run). Mirrors
/// `hpack_decode_into.rs::print_alloc_report`'s pattern (P1 RISC reuse).
fn print_alloc_report() {
    println!("\n--- memcached frame-ownership alloc report (stats_alloc, 1 iteration per workload) ---");
    println!(
        "  {:<24} {:>6} {:>14} {:>10} {:>18} {:>10} {:>20} {:>10} {:>14} {:>10}",
        "workload",
        "bytes",
        "parse_frame#",
        "p_bytes",
        "own_frame(vec)#",
        "o_bytes",
        "own_frame(0copy)#",
        "z_bytes",
        "end_to_end#",
        "e_bytes"
    );
    for workload in workloads() {
        let codec = MemcachedCodec::new(workload.max_message_bytes);
        let raw = Bytes::from(workload.bytes.clone());
        let region = Region::new(&ALLOC);

        let before = region.change();
        let (frame, _consumed) = codec.parse_frame(&raw).expect("parses or violates");
        let after = region.change();
        let parse_allocs = after.allocations - before.allocations;
        let parse_bytes = after.bytes_allocated - before.bytes_allocated;

        let before = region.change();
        let owned = MemcachedCodec::own_frame(&raw, &frame);
        let after = region.change();
        let own_allocs = after.allocations - before.allocations;
        let own_bytes = after.bytes_allocated - before.bytes_allocated;
        std::hint::black_box(&owned);

        let before = region.change();
        let zero_copy = own_frame_zero_copy(&raw, &frame);
        let after = region.change();
        let zero_copy_allocs = after.allocations - before.allocations;
        let zero_copy_bytes = after.bytes_allocated - before.bytes_allocated;
        std::hint::black_box(&zero_copy);

        let pipe = end_to_end_pipe(MemcachedCodec::new(workload.max_message_bytes));
        let before = region.change();
        let outcome = block_on(Pipe::call(&pipe, raw.clone())).expect("resolves");
        let after = region.change();
        let end_to_end_allocs = after.allocations - before.allocations;
        let end_to_end_bytes = after.bytes_allocated - before.bytes_allocated;
        std::hint::black_box(outcome);

        println!(
            "  {:<24} {:>6} {:>14} {:>10} {:>18} {:>10} {:>20} {:>10} {:>14} {:>10}",
            workload.label,
            raw.len(),
            parse_allocs,
            parse_bytes,
            own_allocs,
            own_bytes,
            zero_copy_allocs,
            zero_copy_bytes,
            end_to_end_allocs,
            end_to_end_bytes,
        );
    }
    println!("--- end alloc report ---\n");
}

fn alloc_report_bench(criterion: &mut Criterion) {
    // Runs once as a `criterion_group` member so `cargo bench` always emits
    // the alloc table alongside the timing groups from a single invocation
    // (mirrors `hpack_decode_into.rs`'s `decode_compare` entry point).
    print_alloc_report();
    let mut group = criterion.benchmark_group("memcached_frame_alloc_report_marker");
    group.bench_function("printed_above", |bencher| {
        bencher.iter(|| std::hint::black_box(1));
    });
    group.finish();
}

criterion_group!(
    benches,
    alloc_report_bench,
    bench_parse_frame,
    bench_own_frame_owned_vec,
    bench_own_frame_zero_copy,
    bench_end_to_end
);
criterion_main!(benches);
