#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

//! Native HTTP/2 framing microbench. Proves the zero-copy claim and
//! sets baselines for future regressions.
//!
//! - `header_parse` / `header_encode` — the 9-byte header alone.
//! - `parse_data_{small,medium,large}` — DATA frame at 64 B / 4 KiB /
//!   64 KiB. **All three must measure roughly the same**: parse is
//!   slice-math + a refcount bump, independent of payload size.
//! - `parse_headers_{no_priority,with_priority}` — HEADERS frame with
//!   and without the 5-byte priority block.
//! - `parse_settings_{1,6,16}_entries` — SETTINGS scales with entry
//!   count (must read each (id, value) pair); useful as a linear-cost
//!   sanity check vs. the constant-cost DATA bench.
//! - `encode_data_{small,large}` — write side. Encode is also slice
//!   math + a Vec::extend; should be flat in `data` size only on the
//!   payload-copy step (which is unavoidable since we don't own the
//!   output buffer at frame-construction time).

use std::time::Duration;

use bytes::{Bytes, BytesMut};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use proxima::h2::frame::{
    FRAME_HEADER_LEN, FrameHeader, FramePayload, FrameType, PriorityBlock, StandardSettings,
    encode_frame, encode_frame_vectored, error_code, flags, parse_payload,
};

fn header_parse(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("h2_native_frame_header");
    group.measurement_time(Duration::from_secs(3));
    group.throughput(Throughput::Elements(1));
    let header = FrameHeader {
        length: 1024,
        frame_type: FrameType::Data,
        flags: flags::END_STREAM,
        stream_id: 5,
    };
    let mut buf = Vec::with_capacity(FRAME_HEADER_LEN);
    header.encode(&mut buf);
    let bytes: [u8; FRAME_HEADER_LEN] = buf.try_into().expect("9 bytes");
    group.bench_function("parse", |bencher| {
        bencher.iter(|| {
            let parsed = FrameHeader::parse(std::hint::black_box(&bytes)).expect("parse");
            std::hint::black_box(parsed.length);
        });
    });
    group.bench_function("encode", |bencher| {
        let mut out = Vec::with_capacity(FRAME_HEADER_LEN);
        bencher.iter(|| {
            out.clear();
            std::hint::black_box(&header).encode(&mut out);
            std::hint::black_box(out.len());
        });
    });
    group.finish();
}

fn parse_data(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("h2_native_frame_parse_data");
    group.measurement_time(Duration::from_secs(3));
    for (label, size) in [("64b", 64_usize), ("4kib", 4 * 1024), ("64kib", 64 * 1024)] {
        let payload = FramePayload::Data {
            data: Bytes::from(vec![b'A'; size]),
        };
        let mut wire = Vec::with_capacity(FRAME_HEADER_LEN + size);
        encode_frame(FrameType::Data, flags::END_STREAM, 1, &payload, &mut wire);
        let wire_bytes = Bytes::from(wire);
        let header = FrameHeader::parse(&wire_bytes).expect("header");
        let payload_slice = wire_bytes.slice(FRAME_HEADER_LEN..);
        group.throughput(Throughput::Elements(1));
        group.bench_function(label, |bencher| {
            bencher.iter(|| {
                let parsed = parse_payload(
                    std::hint::black_box(&header),
                    std::hint::black_box(&payload_slice),
                )
                .expect("payload");
                std::hint::black_box(parsed);
            });
        });
    }
    group.finish();
}

fn parse_headers(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("h2_native_frame_parse_headers");
    group.measurement_time(Duration::from_secs(3));
    group.throughput(Throughput::Elements(1));
    let fragment =
        Bytes::from_static(b"\x82\x86\x84\x41\x8c\xf1\xe3\xc2\xe5\xf2\x3a\x6b\xa0\xab\x90\xf4\xff");

    // No-priority variant
    let payload_np = FramePayload::Headers {
        priority: None,
        block_fragment: fragment.clone(),
    };
    let mut wire = Vec::new();
    encode_frame(
        FrameType::Headers,
        flags::END_HEADERS,
        3,
        &payload_np,
        &mut wire,
    );
    let wire_bytes = Bytes::from(wire);
    let header_np = FrameHeader::parse(&wire_bytes).expect("header");
    let payload_np_slice = wire_bytes.slice(FRAME_HEADER_LEN..);
    group.bench_function("no_priority", |bencher| {
        bencher.iter(|| {
            let parsed = parse_payload(
                std::hint::black_box(&header_np),
                std::hint::black_box(&payload_np_slice),
            )
            .expect("payload");
            std::hint::black_box(parsed);
        });
    });

    // With-priority variant
    let payload_p = FramePayload::Headers {
        priority: Some(PriorityBlock {
            exclusive: true,
            stream_dependency: 1,
            weight: 16,
        }),
        block_fragment: fragment,
    };
    let mut wire = Vec::new();
    encode_frame(
        FrameType::Headers,
        flags::END_HEADERS | flags::PRIORITY,
        3,
        &payload_p,
        &mut wire,
    );
    let wire_bytes = Bytes::from(wire);
    let header_p = FrameHeader::parse(&wire_bytes).expect("header");
    let payload_p_slice = wire_bytes.slice(FRAME_HEADER_LEN..);
    group.bench_function("with_priority", |bencher| {
        bencher.iter(|| {
            let parsed = parse_payload(
                std::hint::black_box(&header_p),
                std::hint::black_box(&payload_p_slice),
            )
            .expect("payload");
            std::hint::black_box(parsed);
        });
    });
    group.finish();
}

fn parse_settings(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("h2_native_frame_parse_settings");
    group.measurement_time(Duration::from_secs(3));
    group.throughput(Throughput::Elements(1));
    for entry_count in [1_usize, 6, 16] {
        // Build a StandardSettings whose serialized length matches the
        // requested entry count by setting that many standard slots
        // (1..=6) and then pushing extra IDs onto the extensions vec.
        let mut settings = StandardSettings::default();
        for index in 0..entry_count {
            match index {
                0 => settings.header_table_size = Some(4096),
                1 => settings.enable_push = Some(true),
                2 => settings.max_concurrent_streams = Some(100),
                3 => settings.initial_window_size = Some(65535),
                4 => settings.max_frame_size = Some(16384),
                5 => settings.max_header_list_size = Some(8192),
                _ => settings.extensions.push(proxima::h2::frame::SettingEntry {
                    identifier: 0x100 + (index as u16),
                    value: 1024 * (index as u32 + 1),
                }),
            }
        }
        let payload = FramePayload::Settings(settings);
        let mut wire = Vec::new();
        encode_frame(FrameType::Settings, 0, 0, &payload, &mut wire);
        let wire_bytes = Bytes::from(wire);
        let header = FrameHeader::parse(&wire_bytes).expect("header");
        let payload_slice = wire_bytes.slice(FRAME_HEADER_LEN..);
        group.bench_function(format!("{entry_count}_entries"), |bencher| {
            bencher.iter(|| {
                let parsed = parse_payload(
                    std::hint::black_box(&header),
                    std::hint::black_box(&payload_slice),
                )
                .expect("payload");
                std::hint::black_box(parsed);
            });
        });
    }
    group.finish();
}

fn parse_lifecycle_frames(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("h2_native_frame_parse_lifecycle");
    group.measurement_time(Duration::from_secs(3));
    group.throughput(Throughput::Elements(1));

    // PING
    let ping_payload = FramePayload::Ping {
        opaque: *b"ping1234",
    };
    let mut wire = Vec::new();
    encode_frame(FrameType::Ping, 0, 0, &ping_payload, &mut wire);
    let ping_wire = Bytes::from(wire);
    let ping_header = FrameHeader::parse(&ping_wire).expect("header");
    let ping_slice = ping_wire.slice(FRAME_HEADER_LEN..);
    group.bench_function("ping", |bencher| {
        bencher.iter(|| {
            let parsed = parse_payload(
                std::hint::black_box(&ping_header),
                std::hint::black_box(&ping_slice),
            )
            .expect("payload");
            std::hint::black_box(parsed);
        });
    });

    // RST_STREAM
    let rst_payload = FramePayload::RstStream {
        error_code: error_code::CANCEL,
    };
    let mut wire = Vec::new();
    encode_frame(FrameType::RstStream, 0, 7, &rst_payload, &mut wire);
    let rst_wire = Bytes::from(wire);
    let rst_header = FrameHeader::parse(&rst_wire).expect("header");
    let rst_slice = rst_wire.slice(FRAME_HEADER_LEN..);
    group.bench_function("rst_stream", |bencher| {
        bencher.iter(|| {
            let parsed = parse_payload(
                std::hint::black_box(&rst_header),
                std::hint::black_box(&rst_slice),
            )
            .expect("payload");
            std::hint::black_box(parsed);
        });
    });

    // WINDOW_UPDATE
    let win_payload = FramePayload::WindowUpdate { increment: 65535 };
    let mut wire = Vec::new();
    encode_frame(FrameType::WindowUpdate, 0, 1, &win_payload, &mut wire);
    let win_wire = Bytes::from(wire);
    let win_header = FrameHeader::parse(&win_wire).expect("header");
    let win_slice = win_wire.slice(FRAME_HEADER_LEN..);
    group.bench_function("window_update", |bencher| {
        bencher.iter(|| {
            let parsed = parse_payload(
                std::hint::black_box(&win_header),
                std::hint::black_box(&win_slice),
            )
            .expect("payload");
            std::hint::black_box(parsed);
        });
    });

    // GOAWAY with no debug data
    let goaway_payload = FramePayload::GoAway {
        last_stream_id: 1,
        error_code: error_code::NO_ERROR,
        debug_data: Bytes::new(),
    };
    let mut wire = Vec::new();
    encode_frame(FrameType::GoAway, 0, 0, &goaway_payload, &mut wire);
    let goaway_wire = Bytes::from(wire);
    let goaway_header = FrameHeader::parse(&goaway_wire).expect("header");
    let goaway_slice = goaway_wire.slice(FRAME_HEADER_LEN..);
    group.bench_function("goaway_empty_debug", |bencher| {
        bencher.iter(|| {
            let parsed = parse_payload(
                std::hint::black_box(&goaway_header),
                std::hint::black_box(&goaway_slice),
            )
            .expect("payload");
            std::hint::black_box(parsed);
        });
    });
    group.finish();
}

fn encode_data(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("h2_native_frame_encode_data");
    group.measurement_time(Duration::from_secs(3));
    for (label, size) in [("64b", 64_usize), ("4kib", 4 * 1024), ("64kib", 64 * 1024)] {
        let data: Bytes = Bytes::from(vec![b'A'; size]);
        let payload = FramePayload::Data { data };
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_function(label, |bencher| {
            let mut out = Vec::with_capacity(FRAME_HEADER_LEN + size);
            bencher.iter(|| {
                out.clear();
                encode_frame(
                    FrameType::Data,
                    flags::END_STREAM,
                    1,
                    std::hint::black_box(&payload),
                    &mut out,
                );
                std::hint::black_box(out.len());
            });
        });
    }
    group.finish();
}

/// Vectored encode bench — proves the zero-copy claim on the write
/// side: encode cost must be flat in payload size because the
/// payload `Bytes` is borrowed by refcount, not memcpy'd.
fn encode_data_vectored(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("h2_native_frame_encode_data_vectored");
    group.measurement_time(Duration::from_secs(3));
    for (label, size) in [("64b", 64_usize), ("4kib", 4 * 1024), ("64kib", 64 * 1024)] {
        let data: Bytes = Bytes::from(vec![b'A'; size]);
        let payload = FramePayload::Data { data };
        group.throughput(Throughput::Elements(1));
        group.bench_function(label, |bencher| {
            let mut scratch = BytesMut::with_capacity(64);
            bencher.iter(|| {
                // The freeze inside encode_frame_vectored hands the
                // scratch bytes out as a refcount; on next call the
                // BytesMut allocates fresh capacity. To prove the
                // amortized-zero-alloc claim, callers in production
                // would reserve and reuse — but criterion's iter
                // loop wants a fresh state each call. Use a generously
                // sized scratch to avoid reallocation noise.
                if scratch.capacity() < 64 {
                    scratch = BytesMut::with_capacity(256);
                }
                let segments = encode_frame_vectored(
                    FrameType::Data,
                    flags::END_STREAM,
                    1,
                    std::hint::black_box(&payload),
                    &mut scratch,
                );
                std::hint::black_box(segments);
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    header_parse,
    parse_data,
    parse_headers,
    parse_settings,
    parse_lifecycle_frames,
    encode_data,
    encode_data_vectored
);
criterion_main!(benches);
