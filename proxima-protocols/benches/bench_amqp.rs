#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

//! P3 — AMQP 0-9-1 frame parser bench. Apples-to-apples vs an
//! inline hand-rolled parity baseline. Three workloads:
//! method (small), body (larger payload), heartbeat (zero body).
//!
//! incumbents (versions pinned in Cargo.toml):
//!   - amq-protocol 7 — full AMQP 0-9-1 codec with per-method typed
//!     decoders. Design point is owned typed Method enums for use in a
//!     full AMQP client/broker.
//!
//! groups (and design-favors per workload):
//!   - amqp_method                 design-favors: proxima  (parity vs borrowed)
//!   - amqp_body                   design-favors: proxima  (parity vs borrowed)
//!   - amqp_heartbeat              design-favors: proxima  (parity vs borrowed)
//!   - amqp_real / amqp_workload   design-favors: incumbent (where present)

use std::time::Duration;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};

use proxima_protocols::amqp::{FRAME_END, Frame as ProximaFrame, parse_frame as proxima_parse};

#[allow(dead_code)]
enum ParityFrame<'a> {
    Method {
        channel: u16,
        class_id: u16,
        method_id: u16,
        args: &'a [u8],
    },
    Header {
        channel: u16,
        class_id: u16,
        weight: u16,
        body_size: u64,
        properties: &'a [u8],
    },
    Body {
        channel: u16,
        payload: &'a [u8],
    },
    Heartbeat {
        channel: u16,
    },
}

fn parity_parse(buf: &[u8]) -> Option<(ParityFrame<'_>, usize)> {
    if buf.len() < 7 {
        return None;
    }
    let frame_type = buf[0];
    let channel = u16::from_be_bytes([buf[1], buf[2]]);
    let size = u32::from_be_bytes([buf[3], buf[4], buf[5], buf[6]]);
    let payload_start = 7;
    let payload_end = payload_start + size as usize;
    let total = payload_end + 1;
    if buf.len() < total {
        return None;
    }
    if buf[payload_end] != 0xCE {
        return None;
    }
    let payload = &buf[payload_start..payload_end];

    let frame = match frame_type {
        1 => {
            if payload.len() < 4 {
                return None;
            }
            ParityFrame::Method {
                channel,
                class_id: u16::from_be_bytes([payload[0], payload[1]]),
                method_id: u16::from_be_bytes([payload[2], payload[3]]),
                args: &payload[4..],
            }
        }
        2 => {
            if payload.len() < 12 {
                return None;
            }
            ParityFrame::Header {
                channel,
                class_id: u16::from_be_bytes([payload[0], payload[1]]),
                weight: u16::from_be_bytes([payload[2], payload[3]]),
                body_size: u64::from_be_bytes([
                    payload[4],
                    payload[5],
                    payload[6],
                    payload[7],
                    payload[8],
                    payload[9],
                    payload[10],
                    payload[11],
                ]),
                properties: &payload[12..],
            }
        }
        3 => ParityFrame::Body { channel, payload },
        8 => ParityFrame::Heartbeat { channel },
        _ => return None,
    };
    Some((frame, total))
}

fn make_frame(frame_type: u8, channel: u16, payload: &[u8]) -> Vec<u8> {
    let mut buf = vec![frame_type];
    buf.extend_from_slice(&channel.to_be_bytes());
    buf.extend_from_slice(&(payload.len() as u32).to_be_bytes());
    buf.extend_from_slice(payload);
    buf.push(FRAME_END);
    buf
}

fn make_method_frame() -> Vec<u8> {
    // connection.close-ok: class_id=10, method_id=51, no args.
    // Picked because amq-protocol validates per-method arg shape and
    // close-ok has zero args, so the test buffer doesn't depend on
    // the full AMQP type-coder being implemented in the test setup.
    let mut payload = Vec::new();
    payload.extend_from_slice(&10u16.to_be_bytes());
    payload.extend_from_slice(&51u16.to_be_bytes());
    make_frame(1, 1, &payload)
}

fn make_body_frame() -> Vec<u8> {
    make_frame(3, 1, &[0xAB; 256])
}

fn make_heartbeat_frame() -> Vec<u8> {
    make_frame(8, 0, &[])
}

fn bench_method(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("amqp_method");
    group.measurement_time(Duration::from_secs(2));
    let buf = make_method_frame();
    group.throughput(Throughput::Bytes(buf.len() as u64));
    group.bench_function("proxima", |bencher| {
        bencher.iter(|| {
            let (frame, used) = proxima_parse(std::hint::black_box(&buf)).unwrap();
            std::hint::black_box((frame, used));
        });
    });
    group.bench_function("parity", |bencher| {
        bencher.iter(|| {
            let (frame, used) = parity_parse(std::hint::black_box(&buf)).unwrap();
            std::hint::black_box((frame, used));
        });
    });
    group.finish();
}

fn bench_body(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("amqp_body");
    group.measurement_time(Duration::from_secs(2));
    let buf = make_body_frame();
    group.throughput(Throughput::Bytes(buf.len() as u64));
    group.bench_function("proxima", |bencher| {
        bencher.iter(|| {
            let (frame, used) = proxima_parse(std::hint::black_box(&buf)).unwrap();
            match frame {
                ProximaFrame::Body { payload, .. } => std::hint::black_box(payload),
                _ => unreachable!(),
            };
            std::hint::black_box(used);
        });
    });
    group.bench_function("parity", |bencher| {
        bencher.iter(|| {
            let (frame, used) = parity_parse(std::hint::black_box(&buf)).unwrap();
            match frame {
                ParityFrame::Body { payload, .. } => std::hint::black_box(payload),
                _ => unreachable!(),
            };
            std::hint::black_box(used);
        });
    });
    group.finish();
}

fn bench_heartbeat(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("amqp_heartbeat");
    group.measurement_time(Duration::from_secs(2));
    let buf = make_heartbeat_frame();
    group.throughput(Throughput::Bytes(buf.len() as u64));
    group.bench_function("proxima", |bencher| {
        bencher.iter(|| {
            let (frame, used) = proxima_parse(std::hint::black_box(&buf)).unwrap();
            std::hint::black_box((frame, used));
        });
    });
    group.bench_function("parity", |bencher| {
        bencher.iter(|| {
            let (frame, used) = parity_parse(std::hint::black_box(&buf)).unwrap();
            std::hint::black_box((frame, used));
        });
    });
    group.finish();
}

fn bench_real_amq_protocol(criterion: &mut Criterion) {
    // Real-crate comparison: `amq-protocol::frame::parse_frame`.
    // amq-protocol is the production Rust AMQP 0-9-1 wire codec used
    // by lapin and others. Scope-mismatched — amq-protocol parses
    // method args / properties into typed enums while proxima leaves
    // them as borrowed slices. The comparison is honest with that
    // caveat called out.
    use amq_protocol::frame::parse_frame as real_parse_frame;
    let mut group = criterion.benchmark_group("amqp_real");
    group.measurement_time(Duration::from_secs(2));
    let method_buf = make_method_frame();
    let body_buf = make_body_frame();
    let heartbeat_buf = make_heartbeat_frame();
    group.bench_function("proxima_method", |bencher| {
        bencher.iter(|| {
            let (frame, used) = proxima_parse(std::hint::black_box(&method_buf)).unwrap();
            std::hint::black_box((frame, used));
        });
    });
    group.bench_function("amq_protocol_method", |bencher| {
        bencher.iter(|| {
            let (_remaining, frame) = real_parse_frame(std::hint::black_box(&method_buf[..]))
                .expect("amq_protocol parse");
            std::hint::black_box(frame);
        });
    });
    group.bench_function("proxima_body", |bencher| {
        bencher.iter(|| {
            let (frame, used) = proxima_parse(std::hint::black_box(&body_buf)).unwrap();
            std::hint::black_box((frame, used));
        });
    });
    group.bench_function("amq_protocol_body", |bencher| {
        bencher.iter(|| {
            let (_remaining, frame) =
                real_parse_frame(std::hint::black_box(&body_buf[..])).expect("amq_protocol parse");
            std::hint::black_box(frame);
        });
    });
    group.bench_function("proxima_heartbeat", |bencher| {
        bencher.iter(|| {
            let (frame, used) = proxima_parse(std::hint::black_box(&heartbeat_buf)).unwrap();
            std::hint::black_box((frame, used));
        });
    });
    group.bench_function("amq_protocol_heartbeat", |bencher| {
        bencher.iter(|| {
            let (_remaining, frame) = real_parse_frame(std::hint::black_box(&heartbeat_buf[..]))
                .expect("amq_protocol parse");
            std::hint::black_box(frame);
        });
    });
    group.finish();
}

// Workload bench: feature-parity comparison
//
// Workload: given an AMQP method frame, classify it by AMQP class_id
// (connection=10 / channel=20 / exchange=40 / queue=50 / basic=60 /
// tx=90 / confirm=85). Substrate-routing use case: deliver to the
// right broker subsystem. Both arms take `&[u8]` in, return
// `Option<u8>` (class category 0-6).
//
// Both impls fully parse the frame. Difference is output ownership:
// proxima borrows class_id from a u16 read; amq-protocol decodes
// the typed AMQPFrame enum with per-method args via nom.

fn classify_class_id(class_id: u16) -> Option<u8> {
    match class_id {
        10 => Some(0), // connection
        20 => Some(1), // channel
        40 => Some(2), // exchange
        50 => Some(3), // queue
        60 => Some(4), // basic
        85 => Some(5), // confirm
        90 => Some(6), // tx
        _ => None,
    }
}

fn workload_classify_method_proxima(buf: &[u8]) -> Option<u8> {
    let (frame, _) = proxima_parse(buf).ok()?;
    match frame {
        ProximaFrame::Method { class_id, .. } => classify_class_id(class_id),
        _ => None,
    }
}

fn workload_classify_method_amq(buf: &[u8]) -> Option<u8> {
    use amq_protocol::frame::AMQPFrame;
    let (_remaining, frame) = amq_protocol::frame::parse_frame(buf).ok()?;
    let class_id = match frame {
        AMQPFrame::Method(_channel, class_method) => {
            // class_method is amq_protocol_types::AMQPClass
            // — get its class id via the helper.
            class_method.get_amqp_class_id()
        }
        _ => return None,
    };
    classify_class_id(class_id)
}

fn bench_workload_classify_method(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("amqp_workload_classify_method");
    group.measurement_time(Duration::from_secs(2));
    let buf = make_method_frame();
    let a = workload_classify_method_proxima(&buf);
    let b = workload_classify_method_amq(&buf);
    assert_eq!(a, b, "workload mismatch: proxima={a:?}, amq={b:?}");
    group.throughput(Throughput::Bytes(buf.len() as u64));
    group.bench_function("proxima", |bencher| {
        bencher.iter(|| {
            let r = workload_classify_method_proxima(std::hint::black_box(&buf));
            std::hint::black_box(r);
        });
    });
    group.bench_function("amq_protocol", |bencher| {
        bencher.iter(|| {
            let r = workload_classify_method_amq(std::hint::black_box(&buf));
            std::hint::black_box(r);
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_method,
    bench_body,
    bench_heartbeat,
    bench_real_amq_protocol,
    bench_workload_classify_method
);
criterion_main!(benches);
