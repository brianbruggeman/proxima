#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

//! P9 — gRPC length-prefix framing micro-bench. Per
//! `docs/protocol-gap/discipline.md`. Apples-to-apples comparison
//! against a hand-rolled equivalent parser/encoder that does the
//! same scope: read 5-byte header (compression flag + u32 BE length),
//! borrow the payload slice, return it. Encode: push 5-byte header
//! then `extend_from_slice` the payload.
//!
//! incumbents:
//!   - tonic (no version pin) — tonic does not expose the bare 5-byte
//!     length-prefix codec publicly; its Streaming decoder bundles
//!     framing with HTTP/2 + state machine at a different layer of
//!     abstraction.
//!
//! groups (and design-favors per workload):
//!   - grpc_decode / grpc_encode   design-favors: proxima
//!     (parity baseline only; no scope-matched incumbent at this layer)
//!
//! REGIME OUT-OF-SCOPE: tonic's home turf is its Streaming decoder
//! over HTTP/2 — covered at the runtime/HTTP layer in
//! `h2_runtime_swap`, not here. At the bare 5-byte framing layer no
//! incumbent comparison can be made; the parity baseline is the
//! gate. See protocol-gap discipline.md.
//!
//! Arms:
//!
//! - `proxima_decode` / `proxima_encode` — `proxima::grpc::{parse, encode}`.
//! - `parity_decode` / `parity_encode` — same scope, inlined in this
//!   bench file. No struct fields beyond what proxima exposes.
//!
//! Three message sizes (16 B, 1 KiB, 64 KiB) so the per-frame fixed
//! cost is visible at small sizes and the memcpy dominates at large.

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use proxima_protocols::grpc_framing::{
    Compression, encode as proxima_encode_frame, parse as proxima_parse_frame,
};

const SIZES: &[usize] = &[16, 1024, 64 * 1024];

/// Same scope as `proxima::grpc::Frame` — borrows the payload slice.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
struct ParityFrame<'a> {
    compression: u8,
    payload: &'a [u8],
}

#[inline]
fn parity_parse(buf: &[u8]) -> Option<(ParityFrame<'_>, usize)> {
    if buf.len() < 5 {
        return None;
    }
    let compression = buf[0];
    if compression > 1 {
        return None;
    }
    let length = u32::from_be_bytes([buf[1], buf[2], buf[3], buf[4]]) as usize;
    if buf.len() < 5 + length {
        return None;
    }
    Some((
        ParityFrame {
            compression,
            payload: &buf[5..5 + length],
        },
        5 + length,
    ))
}

#[inline]
fn parity_encode(message: &[u8], dest: &mut Vec<u8>) {
    dest.reserve(5 + message.len());
    dest.push(0);
    dest.extend_from_slice(&(message.len() as u32).to_be_bytes());
    dest.extend_from_slice(message);
}

fn make_frame(size: usize) -> Vec<u8> {
    let mut buf = Vec::with_capacity(5 + size);
    proxima_encode_frame(&vec![0xAB; size], Compression::None, &mut buf);
    buf
}

fn bench_decode(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("grpc_decode");
    group.measurement_time(Duration::from_secs(2));
    for &size in SIZES {
        let frame = make_frame(size);
        group.throughput(Throughput::Bytes(frame.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("proxima", size),
            &frame,
            |bencher, frame| {
                bencher.iter(|| {
                    let (parsed, used) =
                        proxima_parse_frame(std::hint::black_box(frame)).expect("frame");
                    std::hint::black_box((parsed, used));
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("parity", size),
            &frame,
            |bencher, frame| {
                bencher.iter(|| {
                    let (parsed, used) = parity_parse(std::hint::black_box(frame)).expect("frame");
                    std::hint::black_box((parsed, used));
                });
            },
        );
    }
    group.finish();
}

fn bench_encode(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("grpc_encode");
    group.measurement_time(Duration::from_secs(2));
    for &size in SIZES {
        let payload = vec![0xCD; size];
        group.throughput(Throughput::Bytes(5 + payload.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("proxima", size),
            &payload,
            |bencher, payload| {
                bencher.iter_with_setup(
                    || Vec::with_capacity(5 + payload.len()),
                    |mut buf| {
                        proxima_encode_frame(payload, Compression::None, &mut buf);
                        std::hint::black_box(buf);
                    },
                );
            },
        );
        group.bench_with_input(
            BenchmarkId::new("parity", size),
            &payload,
            |bencher, payload| {
                bencher.iter_with_setup(
                    || Vec::with_capacity(5 + payload.len()),
                    |mut buf| {
                        parity_encode(payload, &mut buf);
                        std::hint::black_box(buf);
                    },
                );
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_decode, bench_encode);
criterion_main!(benches);
