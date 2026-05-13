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

//! C8 + C9 — CID-queue + packet-number-space bench arms.
//!
//! Both are state primitives with no head-to-head incumbent (quinn's
//! analogous types are `pub(crate)`). The arms below measure the
//! sub-ns to low-ns cost of the typical operations on the steady-state
//! hot path.

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use proxima_protocols::quic::connection_id::{CidEntry, CidQueue, STATELESS_RESET_TOKEN_LEN};
use proxima_protocols::quic::packet_number::{self, RecvSpace, SendSpace};

fn bench_cid_queue_insert_then_find(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c8_cid_queue_insert_then_find");
    let token = [0u8; STATELESS_RESET_TOKEN_LEN];
    let cid = [1u8, 2, 3, 4, 5, 6, 7, 8];
    group.throughput(Throughput::Elements(1));
    group.bench_function("proxima_quic_proto", |bencher| {
        bencher.iter(|| {
            let mut queue: CidQueue<8> = CidQueue::new();
            for sequence in 0..4 {
                let entry = CidEntry::new(sequence, &cid, token).unwrap();
                queue.insert(entry).unwrap();
            }
            let found = queue.find_by_sequence(std::hint::black_box(2));
            std::hint::black_box(found);
        });
    });
    group.finish();
}

fn bench_send_space_assign(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c9_send_space_assign");
    group.throughput(Throughput::Elements(1));
    group.bench_function("proxima_quic_proto", |bencher| {
        let mut space = SendSpace::new();
        bencher.iter(|| {
            let pn = space.assign().unwrap();
            std::hint::black_box(pn);
        });
    });
    group.finish();
}

fn bench_recv_space_record(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c9_recv_space_record");
    group.throughput(Throughput::Elements(1));
    group.bench_function("proxima_quic_proto_in_order", |bencher| {
        let mut space: RecvSpace<128> = RecvSpace::new();
        let mut pn = 0u64;
        bencher.iter(|| {
            let new = space.record_received(std::hint::black_box(pn)).unwrap();
            std::hint::black_box(new);
            pn += 1;
        });
    });
    group.finish();
}

fn bench_encode_packet_number(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c9_encode_packet_number");
    group.throughput(Throughput::Elements(1));
    group.bench_function("proxima_quic_proto", |bencher| {
        let mut pn = 0u64;
        let largest_acked = Some(0u64);
        bencher.iter(|| {
            let (truncated, len) = packet_number::encode_packet_number(
                std::hint::black_box(pn),
                std::hint::black_box(largest_acked),
            )
            .unwrap();
            std::hint::black_box((truncated, len));
            pn += 1;
        });
    });
    group.finish();
}

fn bench_decode_packet_number(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c9_decode_packet_number");
    group.throughput(Throughput::Elements(1));
    group.bench_function("proxima_quic_proto", |bencher| {
        let largest_pn = 0xa82f30eau64;
        bencher.iter(|| {
            let decoded = packet_number::decode_packet_number(
                std::hint::black_box(largest_pn),
                std::hint::black_box(0x9b32),
                std::hint::black_box(16),
            )
            .unwrap();
            std::hint::black_box(decoded);
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_cid_queue_insert_then_find,
    bench_send_space_assign,
    bench_recv_space_record,
    bench_encode_packet_number,
    bench_decode_packet_number,
);
criterion_main!(benches);
