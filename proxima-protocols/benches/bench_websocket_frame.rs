#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

//! P11 — WebSocket frame codec bench. Real-crate comparison only.
//!
//! Reference: `tungstenite::protocol::frame::FrameHeader::parse` —
//! the same operation proxima is doing (header parse from a `Cursor`).
//! Both arms see byte-identical wire input.
//!
//! incumbents (versions pinned in Cargo.toml):
//!   - tungstenite 0.24 — canonical Rust WebSocket frame codec used
//!     by async-tungstenite and tokio-tungstenite. Design point is
//!     RFC 6455 FrameHeader::parse from a Cursor, scope-matched
//!     against proxima.
//!
//! groups (and design-favors per workload):
//!   - ws_frame_small / ws_frame_medium / ws_frame_large
//!     design-favors: incumbent
//!     (tungstenite on its canonical FrameHeader::parse path; three
//!     sizes engage 7-bit / 16-bit / 64-bit length variants + masked
//!     path on the medium arm.)
//!
//! Hand-rolled "parity" baselines are intentionally omitted — they
//! measure nothing (same author writing two impls at bench time
//! proves only the author's consistency, not the implementation's
//! speed against the real ecosystem).
//!
//! Three workloads chosen to span the length-prefix encoding:
//! - small unmasked text frame (5-byte payload, 7-bit length)
//! - medium masked binary frame (200-byte payload, 16-bit length, mask)
//! - large unmasked binary frame (70000-byte payload, 64-bit length)

use std::io::Cursor;
use std::time::Duration;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};

use proxima_protocols::websocket_frame::parse_frame as proxima_parse;

fn make_small_text() -> Vec<u8> {
    let mut buf = vec![0x81, 0x05];
    buf.extend_from_slice(b"hello");
    buf
}

fn make_medium_masked_binary() -> Vec<u8> {
    let key = [0x12, 0x34, 0x56, 0x78];
    let payload = [0xABu8; 200];
    let mut buf = vec![0x82, 0x80 | 126];
    buf.extend_from_slice(&200u16.to_be_bytes());
    buf.extend_from_slice(&key);
    for (i, byte) in payload.iter().enumerate() {
        buf.push(byte ^ key[i & 0x03]);
    }
    buf
}

fn make_large_binary() -> Vec<u8> {
    let payload = vec![0xCDu8; 70_000];
    let mut buf = vec![0x82, 127];
    buf.extend_from_slice(&(70_000u64).to_be_bytes());
    buf.extend_from_slice(&payload);
    buf
}

fn bench_small_text(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("ws_small_text");
    group.measurement_time(Duration::from_secs(2));
    let buf = make_small_text();
    group.throughput(Throughput::Bytes(buf.len() as u64));
    group.bench_function("proxima", |bencher| {
        bencher.iter(|| {
            let (frame, used) = proxima_parse(std::hint::black_box(&buf)).unwrap();
            std::hint::black_box((frame, used));
        });
    });
    group.bench_function("tungstenite", |bencher| {
        bencher.iter(|| {
            let mut cursor = Cursor::new(std::hint::black_box(&buf[..]));
            let header = tungstenite::protocol::frame::FrameHeader::parse(&mut cursor)
                .unwrap()
                .unwrap();
            std::hint::black_box(header);
        });
    });
    group.finish();
}

fn bench_medium_masked(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("ws_medium_masked");
    group.measurement_time(Duration::from_secs(2));
    let buf = make_medium_masked_binary();
    group.throughput(Throughput::Bytes(buf.len() as u64));
    group.bench_function("proxima", |bencher| {
        bencher.iter(|| {
            let (frame, used) = proxima_parse(std::hint::black_box(&buf)).unwrap();
            std::hint::black_box((frame, used));
        });
    });
    group.bench_function("tungstenite", |bencher| {
        bencher.iter(|| {
            let mut cursor = Cursor::new(std::hint::black_box(&buf[..]));
            let header = tungstenite::protocol::frame::FrameHeader::parse(&mut cursor)
                .unwrap()
                .unwrap();
            std::hint::black_box(header);
        });
    });
    group.finish();
}

fn bench_large(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("ws_large");
    group.measurement_time(Duration::from_secs(2));
    let buf = make_large_binary();
    group.throughput(Throughput::Bytes(buf.len() as u64));
    group.bench_function("proxima", |bencher| {
        bencher.iter(|| {
            let (frame, used) = proxima_parse(std::hint::black_box(&buf)).unwrap();
            std::hint::black_box((frame, used));
        });
    });
    group.bench_function("tungstenite", |bencher| {
        bencher.iter(|| {
            let mut cursor = Cursor::new(std::hint::black_box(&buf[..]));
            let header = tungstenite::protocol::frame::FrameHeader::parse(&mut cursor)
                .unwrap()
                .unwrap();
            std::hint::black_box(header);
        });
    });
    group.finish();
}

criterion_group!(benches, bench_small_text, bench_medium_masked, bench_large);
criterion_main!(benches);
