// bench fixtures legitimately fail-fast on encoder errors.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

//! C10 — Initial-packet protect/unprotect compose layer bench arms.
//!
//! Measures the full RFC 9001 §5 pipeline (AEAD + header protection +
//! packet-number encoding) as a single per-packet operation. Direct
//! head-to-head vs `quinn-proto::Connection::handle_packet` defers to
//! C11 once the connection state machine is up.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use proxima_protocols::quic::crypto::initial_keys;
use proxima_protocols::quic::crypto::packet_protection::{protect_initial, unprotect_initial};

const RFC_DCID: [u8; 8] = [0x83, 0x94, 0xc8, 0xf0, 0x3e, 0x51, 0x57, 0x08];
const PAYLOAD_SIZES: &[usize] = &[16, 1200, 1452, 8192];

fn build_packet(plaintext: &[u8]) -> (Vec<u8>, usize) {
    let header_len = 22; // c3 + version(4) + dcid_len(1) + dcid(8) + scid_len(1) + token_len(1) + length(2) + pn(4)
    let total = header_len + plaintext.len() + 16;
    let mut packet = vec![0u8; total];
    packet[0] = 0b1100_0011;
    packet[1..5].copy_from_slice(&1u32.to_be_bytes());
    packet[5] = 8;
    packet[6..14].copy_from_slice(&RFC_DCID);
    packet[14] = 0;
    packet[15] = 0;
    let length_value = (4 + plaintext.len() + 16) as u16 | 0x4000;
    packet[16..18].copy_from_slice(&length_value.to_be_bytes());
    let pn_offset = 18;
    packet[pn_offset..pn_offset + 4].copy_from_slice(&0u32.to_be_bytes());
    packet[22..22 + plaintext.len()].copy_from_slice(plaintext);
    (packet, pn_offset)
}

fn bench_protect(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c10_protect_initial");
    let pair = initial_keys::derive(&RFC_DCID).unwrap();
    for &size in PAYLOAD_SIZES {
        let plaintext = vec![0xabu8; size];
        let (packet_template, pn_offset) = build_packet(&plaintext);
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(
            BenchmarkId::new("proxima_quic_proto", size),
            &size,
            |bencher, &size| {
                bencher.iter(|| {
                    let mut packet = packet_template.clone();
                    protect_initial(
                        std::hint::black_box(&pair.client),
                        std::hint::black_box(0),
                        4,
                        std::hint::black_box(&mut packet),
                        std::hint::black_box(pn_offset),
                        std::hint::black_box(size),
                    )
                    .expect("protect");
                    std::hint::black_box(packet);
                });
            },
        );
    }
    group.finish();
}

fn bench_unprotect(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c10_unprotect_initial");
    let pair = initial_keys::derive(&RFC_DCID).unwrap();
    for &size in PAYLOAD_SIZES {
        let plaintext = vec![0xabu8; size];
        let (mut packet, pn_offset) = build_packet(&plaintext);
        // pre-protect so we measure only the unprotect path
        protect_initial(&pair.client, 0, 4, &mut packet, pn_offset, size).unwrap();
        let protected = packet.clone();
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(
            BenchmarkId::new("proxima_quic_proto", size),
            &size,
            |bencher, _size| {
                bencher.iter(|| {
                    let mut work = protected.clone();
                    let (pn, len) = unprotect_initial(
                        std::hint::black_box(&pair.client),
                        std::hint::black_box(0),
                        std::hint::black_box(&mut work),
                        std::hint::black_box(pn_offset),
                    )
                    .expect("unprotect");
                    std::hint::black_box((pn, len, work));
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_protect, bench_unprotect,);
criterion_main!(benches);
