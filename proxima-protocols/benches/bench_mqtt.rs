#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

//! P2 — MQTT v3.1.1 packet parser bench. Apples-to-apples vs a
//! hand-rolled parity baseline matching proxima's scope.
//!
//! incumbents (versions pinned in Cargo.toml):
//!   - mqttbytes 0.6 — owned typed v3.1.1 packet decoder. Design point
//!     is allocating typed `Publish`/`Connect`/etc. with owned String /
//!     Bytes fields, plus validation, for use in a full MQTT client.
//!
//! groups (and design-favors per workload):
//!   - mqtt_publish_qos0           design-favors: proxima  (parity vs borrowed-view)
//!   - mqtt_publish_qos1           design-favors: proxima  (parity vs borrowed-view)
//!   - mqtt_connect                design-favors: proxima  (parity vs borrowed-view)
//!   - mqtt_real                   design-favors: incumbent
//!     (mqttbytes::v4::read on a BytesMut cursor — mqttbytes's
//!     canonical decode path with owned per-variant allocation)
//!   - mqtt_workload_route_publish design-favors: incumbent
//!     (both arms route a PUBLISH packet to one of N brokers by
//!     topic-prefix match. proxima borrows topic; mqttbytes parses
//!     to owned Publish struct. Engages mqttbytes's owned-typed
//!     design point — same input + same output, different strategy.)
//!
//! Three workloads: PUBLISH qos0 (most common), PUBLISH qos1
//! (acked), CONNECT.

use std::time::Duration;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};

use proxima_protocols::mqtt::{Packet as ProximaPacket, parse_packet as proxima_parse};

#[allow(dead_code)]
enum ParityPacket<'a> {
    Publish {
        qos: u8,
        dup: bool,
        retain: bool,
        topic: &'a [u8],
        packet_id: Option<u16>,
        payload: &'a [u8],
    },
    Connect {
        protocol_name: &'a [u8],
        protocol_level: u8,
        connect_flags: u8,
        keep_alive: u16,
        client_id: &'a [u8],
    },
    Ack {
        pkt_type: u8,
        packet_id: u16,
    },
    PingReq,
    PingResp,
    Disconnect,
    Other,
}

fn parity_parse(buf: &[u8]) -> Option<(ParityPacket<'_>, usize)> {
    if buf.is_empty() {
        return None;
    }
    let first = buf[0];
    let packet_type = first >> 4;
    let flags = first & 0x0F;

    // remaining length varint (1-4 bytes)
    let mut value: u32 = 0;
    let mut multiplier: u32 = 1;
    let mut rem_used = 0usize;
    let rem_buf = &buf[1..];
    for (idx, &byte) in rem_buf.iter().take(4).enumerate() {
        value += u32::from(byte & 0x7F) * multiplier;
        if byte & 0x80 == 0 {
            rem_used = idx + 1;
            break;
        }
        multiplier *= 128;
    }
    if rem_used == 0 {
        return None;
    }
    let header_len = 1 + rem_used;
    let total_len = header_len + value as usize;
    if buf.len() < total_len {
        return None;
    }
    let body = &buf[header_len..total_len];

    let packet = match packet_type {
        1 => {
            // CONNECT
            if body.len() < 2 {
                return None;
            }
            let name_len = u16::from_be_bytes([body[0], body[1]]) as usize;
            let rest = &body[2..];
            if rest.len() < name_len + 4 {
                return None;
            }
            let protocol_name = &rest[..name_len];
            let after_name = &rest[name_len..];
            let protocol_level = after_name[0];
            let connect_flags = after_name[1];
            let keep_alive = u16::from_be_bytes([after_name[2], after_name[3]]);
            let after_ka = &after_name[4..];
            if after_ka.len() < 2 {
                return None;
            }
            let client_len = u16::from_be_bytes([after_ka[0], after_ka[1]]) as usize;
            if after_ka.len() < 2 + client_len {
                return None;
            }
            ParityPacket::Connect {
                protocol_name,
                protocol_level,
                connect_flags,
                keep_alive,
                client_id: &after_ka[2..2 + client_len],
            }
        }
        3 => {
            // PUBLISH
            let qos = (flags >> 1) & 0x03;
            let dup = flags & 0x08 != 0;
            let retain = flags & 0x01 != 0;
            if body.len() < 2 {
                return None;
            }
            let topic_len = u16::from_be_bytes([body[0], body[1]]) as usize;
            let rest = &body[2..];
            if rest.len() < topic_len {
                return None;
            }
            let topic = &rest[..topic_len];
            let after_topic = &rest[topic_len..];
            let (packet_id, payload) = if qos > 0 {
                if after_topic.len() < 2 {
                    return None;
                }
                let id = u16::from_be_bytes([after_topic[0], after_topic[1]]);
                (Some(id), &after_topic[2..])
            } else {
                (None, after_topic)
            };
            ParityPacket::Publish {
                qos,
                dup,
                retain,
                topic,
                packet_id,
                payload,
            }
        }
        4 | 5 | 6 | 7 | 11 => {
            if body.len() < 2 {
                return None;
            }
            ParityPacket::Ack {
                pkt_type: packet_type,
                packet_id: u16::from_be_bytes([body[0], body[1]]),
            }
        }
        12 => ParityPacket::PingReq,
        13 => ParityPacket::PingResp,
        14 => ParityPacket::Disconnect,
        2 | 8 | 9 | 10 => ParityPacket::Other,
        _ => return None,
    };
    Some((packet, total_len))
}

fn make_publish_qos0() -> Vec<u8> {
    let mut buf = vec![0x30];
    let body_len = 2 + 5 + 16; // topic_len + "topic" + 16-byte payload
    buf.push(body_len as u8);
    buf.extend_from_slice(&[0, 5]);
    buf.extend_from_slice(b"topic");
    buf.extend_from_slice(&[0xAB; 16]);
    buf
}

fn make_publish_qos1() -> Vec<u8> {
    let mut buf = vec![0x32];
    let body_len = 2 + 5 + 2 + 16;
    buf.push(body_len as u8);
    buf.extend_from_slice(&[0, 5]);
    buf.extend_from_slice(b"topic");
    buf.extend_from_slice(&[0, 1]);
    buf.extend_from_slice(&[0xAB; 16]);
    buf
}

fn make_connect() -> Vec<u8> {
    let mut buf = vec![0x10];
    let body_len = 2 + 4 + 1 + 1 + 2 + 2 + 8;
    buf.push(body_len as u8);
    buf.extend_from_slice(&[0, 4]);
    buf.extend_from_slice(b"MQTT");
    buf.push(4);
    buf.push(0x02);
    buf.extend_from_slice(&[0, 60]);
    buf.extend_from_slice(&[0, 8]);
    buf.extend_from_slice(b"client-1");
    buf
}

fn bench_publish_qos0(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("mqtt_publish_qos0");
    group.measurement_time(Duration::from_secs(2));
    let buf = make_publish_qos0();
    group.throughput(Throughput::Bytes(buf.len() as u64));
    group.bench_function("proxima", |bencher| {
        bencher.iter(|| {
            let (pkt, used) = proxima_parse(std::hint::black_box(&buf)).unwrap();
            match pkt {
                ProximaPacket::Publish { topic, payload, .. } => {
                    std::hint::black_box((topic, payload));
                }
                _ => unreachable!(),
            };
            std::hint::black_box(used);
        });
    });
    group.bench_function("parity", |bencher| {
        bencher.iter(|| {
            let (pkt, used) = parity_parse(std::hint::black_box(&buf)).unwrap();
            match pkt {
                ParityPacket::Publish { topic, payload, .. } => {
                    std::hint::black_box((topic, payload));
                }
                _ => unreachable!(),
            };
            std::hint::black_box(used);
        });
    });
    group.finish();
}

fn bench_publish_qos1(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("mqtt_publish_qos1");
    group.measurement_time(Duration::from_secs(2));
    let buf = make_publish_qos1();
    group.throughput(Throughput::Bytes(buf.len() as u64));
    group.bench_function("proxima", |bencher| {
        bencher.iter(|| {
            let (pkt, used) = proxima_parse(std::hint::black_box(&buf)).unwrap();
            std::hint::black_box((pkt, used));
        });
    });
    group.bench_function("parity", |bencher| {
        bencher.iter(|| {
            let (pkt, used) = parity_parse(std::hint::black_box(&buf)).unwrap();
            std::hint::black_box((pkt, used));
        });
    });
    group.finish();
}

fn bench_connect(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("mqtt_connect");
    group.measurement_time(Duration::from_secs(2));
    let buf = make_connect();
    group.throughput(Throughput::Bytes(buf.len() as u64));
    group.bench_function("proxima", |bencher| {
        bencher.iter(|| {
            let (pkt, used) = proxima_parse(std::hint::black_box(&buf)).unwrap();
            std::hint::black_box((pkt, used));
        });
    });
    group.bench_function("parity", |bencher| {
        bencher.iter(|| {
            let (pkt, used) = parity_parse(std::hint::black_box(&buf)).unwrap();
            std::hint::black_box((pkt, used));
        });
    });
    group.finish();
}

fn bench_real_mqttbytes(criterion: &mut Criterion) {
    // Real-crate comparison: mqttbytes::v4::read. Scope-mismatched —
    // mqttbytes mutates a `BytesMut` cursor and returns typed Packet
    // enum allocating per-variant owned data (Bytes/String). proxima
    // borrows from `&[u8]`. The comparison is honest as long as we
    // call out that mqttbytes does strictly more work.
    let mut group = criterion.benchmark_group("mqtt_real");
    group.measurement_time(Duration::from_secs(2));
    let publish_qos0 = make_publish_qos0();
    let publish_qos1 = make_publish_qos1();
    let connect = make_connect();
    group.throughput(Throughput::Bytes(publish_qos0.len() as u64));
    group.bench_function("proxima_publish_qos0", |bencher| {
        bencher.iter(|| {
            let (pkt, _) = proxima_parse(std::hint::black_box(&publish_qos0)).unwrap();
            std::hint::black_box(pkt);
        });
    });
    group.bench_function("mqttbytes_publish_qos0", |bencher| {
        bencher.iter_with_setup(
            || bytes::BytesMut::from(std::hint::black_box(&publish_qos0[..])),
            |mut buf| {
                let pkt = mqttbytes::v4::read(&mut buf, usize::MAX).unwrap();
                std::hint::black_box(pkt);
            },
        );
    });
    group.bench_function("proxima_publish_qos1", |bencher| {
        bencher.iter(|| {
            let (pkt, _) = proxima_parse(std::hint::black_box(&publish_qos1)).unwrap();
            std::hint::black_box(pkt);
        });
    });
    group.bench_function("mqttbytes_publish_qos1", |bencher| {
        bencher.iter_with_setup(
            || bytes::BytesMut::from(std::hint::black_box(&publish_qos1[..])),
            |mut buf| {
                let pkt = mqttbytes::v4::read(&mut buf, usize::MAX).unwrap();
                std::hint::black_box(pkt);
            },
        );
    });
    group.bench_function("proxima_connect", |bencher| {
        bencher.iter(|| {
            let (pkt, _) = proxima_parse(std::hint::black_box(&connect)).unwrap();
            std::hint::black_box(pkt);
        });
    });
    group.bench_function("mqttbytes_connect", |bencher| {
        bencher.iter_with_setup(
            || bytes::BytesMut::from(std::hint::black_box(&connect[..])),
            |mut buf| {
                let pkt = mqttbytes::v4::read(&mut buf, usize::MAX).unwrap();
                std::hint::black_box(pkt);
            },
        );
    });
    group.finish();
}

// Workload bench: feature-parity comparison
//
// Workload: given an MQTT PUBLISH packet, route to one of N brokers
// based on which `prefix` in a routing table matches the topic.
// Both arms take `&[u8]` in, return `Option<usize>` (route index).
// proxima streams the packet's borrowed topic; mqttbytes parses into
// owned Publish struct. Same featureset, different strategy.

const ROUTES: &[&[u8]] = &[
    b"sensors/",
    b"actuators/",
    b"telemetry/",
    b"control/",
    b"topic", // matches our test packet's topic
];

fn workload_route_publish_proxima(buf: &[u8]) -> Option<usize> {
    let (packet, _) = proxima_parse(buf).ok()?;
    let topic = match packet {
        ProximaPacket::Publish { topic, .. } => topic,
        _ => return None,
    };
    ROUTES.iter().position(|prefix| topic.starts_with(prefix))
}

fn workload_route_publish_mqttbytes(buf: &[u8]) -> Option<usize> {
    let mut bm = bytes::BytesMut::from(buf);
    let packet = mqttbytes::v4::read(&mut bm, usize::MAX).ok()?;
    let topic = match packet {
        mqttbytes::v4::Packet::Publish(publish) => publish.topic,
        _ => return None,
    };
    ROUTES
        .iter()
        .position(|prefix| topic.as_bytes().starts_with(prefix))
}

fn bench_workload_route_publish(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("mqtt_workload_route_publish");
    group.measurement_time(Duration::from_secs(2));
    let buf = make_publish_qos0();
    // sanity: both arms produce the same answer
    let a = workload_route_publish_proxima(&buf);
    let b = workload_route_publish_mqttbytes(&buf);
    assert_eq!(a, b, "workload mismatch: proxima={a:?}, mqttbytes={b:?}");
    group.throughput(Throughput::Bytes(buf.len() as u64));
    group.bench_function("proxima", |bencher| {
        bencher.iter(|| {
            let r = workload_route_publish_proxima(std::hint::black_box(&buf));
            std::hint::black_box(r);
        });
    });
    group.bench_function("mqttbytes", |bencher| {
        bencher.iter(|| {
            let r = workload_route_publish_mqttbytes(std::hint::black_box(&buf));
            std::hint::black_box(r);
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_publish_qos0,
    bench_publish_qos1,
    bench_connect,
    bench_real_mqttbytes,
    bench_workload_route_publish
);
criterion_main!(benches);
