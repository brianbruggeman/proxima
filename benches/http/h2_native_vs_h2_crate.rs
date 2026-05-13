#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

//! Head-to-head: proxima's native h2 framing vs the `h2` crate.
//!
//! Both crates expose their framing layer publicly — the h2 crate via
//! `h2::frame::{Head, Settings, Ping, ...}` and proxima via
//! `proxima::h2::frame::{FrameHeader, parse_payload, ...}`.
//! No connection state, no async runtime — just the framing primitives
//! fed the same bytes.
//!
//! Each bench reports the time to do a single parse or encode. Goal:
//! native should be ≤ reference for like-for-like operations, while
//! also offering zero-copy on parse (flat in payload size) and
//! vectored encode (flat in payload size — Bytes refcount sharing).

use std::time::Duration;

use bytes::{Buf, Bytes, BytesMut};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use proxima::h2::frame::{
    FRAME_HEADER_LEN, FrameHeader, FramePayload, FrameType, StandardSettings, encode_frame,
    encode_frame_vectored, error_code, flags, parse_payload,
};

// ---------- header parse ----------

fn header_parse_compare(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("h2_frame_header_parse");
    group.measurement_time(Duration::from_secs(3));
    group.throughput(Throughput::Elements(1));

    let header_bytes: [u8; FRAME_HEADER_LEN] = [
        0x00, 0x10, 0x00, // length 4096
        0x00, // type DATA
        0x01, // END_STREAM
        0x00, 0x00, 0x00, 0x05, // stream_id 5
    ];

    group.bench_function("proxima_native", |bencher| {
        bencher.iter(|| {
            let parsed = FrameHeader::parse(std::hint::black_box(&header_bytes)).expect("parse");
            std::hint::black_box(parsed);
        });
    });

    group.bench_function("h2_crate", |bencher| {
        bencher.iter(|| {
            let head = h2::frame::Head::parse(std::hint::black_box(&header_bytes));
            std::hint::black_box(head);
        });
    });

    group.finish();
}

// ---------- header encode ----------

fn header_encode_compare(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("h2_frame_header_encode");
    group.measurement_time(Duration::from_secs(3));
    group.throughput(Throughput::Elements(1));

    let header_native = FrameHeader {
        length: 4096,
        frame_type: FrameType::Data,
        flags: flags::END_STREAM,
        stream_id: 5,
    };
    let head_h2 = h2::frame::Head::new(
        h2::frame::Kind::Data,
        flags::END_STREAM,
        h2::frame::StreamId::from(5_u32),
    );

    group.bench_function("proxima_native", |bencher| {
        let mut out = Vec::with_capacity(FRAME_HEADER_LEN);
        bencher.iter(|| {
            out.clear();
            std::hint::black_box(&header_native).encode(&mut out);
            std::hint::black_box(out.len());
        });
    });

    group.bench_function("h2_crate", |bencher| {
        let mut out = BytesMut::with_capacity(FRAME_HEADER_LEN);
        bencher.iter(|| {
            out.clear();
            std::hint::black_box(&head_h2).encode(4096, &mut out);
            std::hint::black_box(out.len());
        });
    });

    group.finish();
}

// ---------- DATA parse — wire bytes → typed payload ----------

fn data_parse_compare(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("h2_frame_data_parse");
    group.measurement_time(Duration::from_secs(3));
    group.throughput(Throughput::Elements(1));

    for (label, size) in [("64b", 64_usize), ("4kib", 4 * 1024), ("64kib", 64 * 1024)] {
        let payload = FramePayload::Data {
            data: Bytes::from(vec![b'A'; size]),
        };
        let mut wire = Vec::with_capacity(FRAME_HEADER_LEN + size);
        encode_frame(FrameType::Data, flags::END_STREAM, 1, &payload, &mut wire);
        let wire_bytes = Bytes::from(wire);
        let header = FrameHeader::parse(&wire_bytes).expect("header");
        let payload_slice = wire_bytes.slice(FRAME_HEADER_LEN..);

        group.bench_function(format!("proxima_native/{label}"), |bencher| {
            bencher.iter(|| {
                let parsed = parse_payload(
                    std::hint::black_box(&header),
                    std::hint::black_box(&payload_slice),
                )
                .expect("payload");
                std::hint::black_box(parsed);
            });
        });

        // h2 crate's DATA path is internal; the closest public surface
        // is to parse the head + treat the payload as opaque bytes,
        // matching what their codec would do at this layer.
        let head_bytes_only: [u8; FRAME_HEADER_LEN] = (&wire_bytes[..FRAME_HEADER_LEN])
            .try_into()
            .expect("9 bytes");
        let payload_slice_for_h2 = payload_slice.clone();
        group.bench_function(format!("h2_crate/{label}"), |bencher| {
            bencher.iter(|| {
                let head = h2::frame::Head::parse(std::hint::black_box(&head_bytes_only));
                // Mirror what their codec does for DATA: refcount-clone
                // the payload Bytes (zero copy, but the Arc::clone is
                // the same cost as our `payload.slice(..)`).
                let payload_clone = std::hint::black_box(&payload_slice_for_h2).clone();
                std::hint::black_box((head, payload_clone));
            });
        });
    }
    group.finish();
}

// ---------- PING parse ----------

fn ping_parse_compare(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("h2_frame_ping_parse");
    group.measurement_time(Duration::from_secs(3));
    group.throughput(Throughput::Elements(1));

    let payload = FramePayload::Ping {
        opaque: *b"ping1234",
    };
    let mut wire = Vec::new();
    encode_frame(FrameType::Ping, 0, 0, &payload, &mut wire);
    let wire_bytes = Bytes::from(wire);
    let header_native = FrameHeader::parse(&wire_bytes).expect("header");
    let payload_slice = wire_bytes.slice(FRAME_HEADER_LEN..);

    group.bench_function("proxima_native", |bencher| {
        bencher.iter(|| {
            let parsed = parse_payload(
                std::hint::black_box(&header_native),
                std::hint::black_box(&payload_slice),
            )
            .expect("payload");
            std::hint::black_box(parsed);
        });
    });

    let head_bytes_only: [u8; FRAME_HEADER_LEN] = (&wire_bytes[..FRAME_HEADER_LEN])
        .try_into()
        .expect("9 bytes");
    let payload_bytes_for_h2: Vec<u8> = payload_slice.to_vec();
    group.bench_function("h2_crate", |bencher| {
        bencher.iter(|| {
            let head = h2::frame::Head::parse(std::hint::black_box(&head_bytes_only));
            let ping = h2::frame::Ping::load(head, std::hint::black_box(&payload_bytes_for_h2))
                .expect("load");
            std::hint::black_box(ping);
        });
    });
    group.finish();
}

// ---------- SETTINGS parse ----------

fn settings_parse_compare(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("h2_frame_settings_parse");
    group.measurement_time(Duration::from_secs(3));
    group.throughput(Throughput::Elements(1));

    for entry_count in [1_usize, 6, 16] {
        // Build a Settings with `entry_count` slots filled: standard
        // ids 1..=6 first (typed slots) then extensions for the rest.
        // Matches the wire layout we want to feed the parsers.
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
                    value: 4096,
                }),
            }
        }
        let payload = FramePayload::Settings(settings);
        let mut wire = Vec::new();
        encode_frame(FrameType::Settings, 0, 0, &payload, &mut wire);
        let wire_bytes = Bytes::from(wire);
        let header_native = FrameHeader::parse(&wire_bytes).expect("header");
        let payload_slice = wire_bytes.slice(FRAME_HEADER_LEN..);

        group.bench_function(format!("proxima_native/{entry_count}_entries"), |bencher| {
            bencher.iter(|| {
                let parsed = parse_payload(
                    std::hint::black_box(&header_native),
                    std::hint::black_box(&payload_slice),
                )
                .expect("payload");
                std::hint::black_box(parsed);
            });
        });

        let head_bytes_only: [u8; FRAME_HEADER_LEN] = (&wire_bytes[..FRAME_HEADER_LEN])
            .try_into()
            .expect("9 bytes");
        let payload_bytes_for_h2: Vec<u8> = payload_slice.to_vec();
        group.bench_function(format!("h2_crate/{entry_count}_entries"), |bencher| {
            bencher.iter(|| {
                let head = h2::frame::Head::parse(std::hint::black_box(&head_bytes_only));
                let settings =
                    h2::frame::Settings::load(head, std::hint::black_box(&payload_bytes_for_h2))
                        .expect("load");
                std::hint::black_box(settings);
            });
        });
    }
    group.finish();
}

// ---------- DATA encode ----------

fn data_encode_compare(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("h2_frame_data_encode");
    group.measurement_time(Duration::from_secs(3));

    for (label, size) in [("64b", 64_usize), ("4kib", 4 * 1024), ("64kib", 64 * 1024)] {
        let data: Bytes = Bytes::from(vec![b'A'; size]);
        let payload_native = FramePayload::Data { data: data.clone() };

        group.throughput(Throughput::Bytes(size as u64));

        // proxima native — contiguous (memcpy path)
        group.bench_function(format!("proxima_native_contiguous/{label}"), |bencher| {
            let mut out = Vec::with_capacity(FRAME_HEADER_LEN + size);
            bencher.iter(|| {
                out.clear();
                encode_frame(
                    FrameType::Data,
                    flags::END_STREAM,
                    1,
                    std::hint::black_box(&payload_native),
                    &mut out,
                );
                std::hint::black_box(out.len());
            });
        });

        // proxima native — vectored (refcount path, flat in size)
        group.throughput(Throughput::Elements(1));
        group.bench_function(format!("proxima_native_vectored/{label}"), |bencher| {
            let mut scratch = BytesMut::with_capacity(256);
            bencher.iter(|| {
                if scratch.capacity() < 64 {
                    scratch = BytesMut::with_capacity(256);
                }
                let segments = encode_frame_vectored(
                    FrameType::Data,
                    flags::END_STREAM,
                    1,
                    std::hint::black_box(&payload_native),
                    &mut scratch,
                );
                std::hint::black_box(segments);
            });
        });

        // h2 crate — Head::encode + payload pushed separately (the
        // pattern their FramedWrite uses for vectored output). To
        // make it apples-to-apples we compare the encode of the head
        // only; the payload is borrowed Bytes either way.
        group.throughput(Throughput::Elements(1));
        let head_h2 = h2::frame::Head::new(
            h2::frame::Kind::Data,
            flags::END_STREAM,
            h2::frame::StreamId::from(1_u32),
        );
        group.bench_function(format!("h2_crate_head_only/{label}"), |bencher| {
            let mut out = BytesMut::with_capacity(FRAME_HEADER_LEN);
            bencher.iter(|| {
                out.clear();
                std::hint::black_box(&head_h2).encode(size, &mut out);
                let payload_clone = std::hint::black_box(&data).clone();
                std::hint::black_box((out.len(), payload_clone));
            });
        });
    }
    group.finish();
}

// ---------- SETTINGS encode ----------

fn settings_encode_compare(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("h2_frame_settings_encode");
    group.measurement_time(Duration::from_secs(3));
    group.throughput(Throughput::Elements(1));

    let settings = StandardSettings {
        header_table_size: Some(4096),
        enable_push: Some(false),
        max_concurrent_streams: Some(100),
        initial_window_size: Some(65535),
        max_frame_size: Some(16384),
        max_header_list_size: Some(8192),
        extensions: Default::default(),
    };
    let payload_native = FramePayload::Settings(settings);

    group.bench_function("proxima_native", |bencher| {
        let mut out = Vec::with_capacity(64);
        bencher.iter(|| {
            out.clear();
            encode_frame(
                FrameType::Settings,
                0,
                0,
                std::hint::black_box(&payload_native),
                &mut out,
            );
            std::hint::black_box(out.len());
        });
    });

    let settings_h2 = {
        let mut settings = h2::frame::Settings::default();
        settings.set_header_table_size(Some(4096));
        settings.set_enable_push(false);
        settings.set_max_concurrent_streams(Some(100));
        settings.set_initial_window_size(Some(65535));
        settings.set_max_frame_size(Some(16384));
        settings.set_max_header_list_size(Some(8192));
        settings
    };
    group.bench_function("h2_crate", |bencher| {
        let mut out = BytesMut::with_capacity(64);
        bencher.iter(|| {
            out.clear();
            std::hint::black_box(&settings_h2).encode(&mut out);
            std::hint::black_box(out.len());
        });
    });
    group.finish();
}

// ---------- silence dead-code lints for unused-in-some-paths --------

#[allow(dead_code)]
fn _force_use_buf() {
    let _: &dyn Buf = &Bytes::new();
    let _: u32 = error_code::CANCEL;
}

criterion_group!(
    benches,
    header_parse_compare,
    header_encode_compare,
    data_parse_compare,
    ping_parse_compare,
    settings_parse_compare,
    data_encode_compare,
    settings_encode_compare,
);
criterion_main!(benches);
