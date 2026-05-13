//! C13 — range_set + AckScheduler bench arms.
//!
//! No comparable incumbent: `quinn-proto::range_set::RangeSet` is
//! `pub(crate)`. Numbers here establish a per-call baseline + a
//! worst-case 32-range wire-encoded ACK frame cost.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use proxima_protocols::quic::ack::AckScheduler;
use proxima_protocols::quic::range_set::ArrayRangeSet;
use proxima_protocols::quic::time::Instant;
use proxima_protocols::quic::varint;

const ORIGIN: Instant = Instant::from_micros(1_000_000);

fn bench_range_set_in_order_insert(criterion: &mut Criterion) {
    criterion.bench_function("c13_range_set_in_order_insert_1000", |bencher| {
        bencher.iter(|| {
            let mut set: ArrayRangeSet<32> = ArrayRangeSet::new();
            for pn in 0..1_000u64 {
                set.insert(black_box(pn));
            }
            black_box(set);
        });
    });
}

fn bench_range_set_random_insert(criterion: &mut Criterion) {
    // Pseudo-random order: stride-23 modulo 1009 visits each residue
    // class once → a 1000-element permutation without RNG state.
    let pns: Vec<u64> = (0..1000u64).map(|index| (index * 23) % 1009).collect();
    criterion.bench_function("c13_range_set_random_insert_1000", |bencher| {
        bencher.iter(|| {
            let mut set: ArrayRangeSet<32> = ArrayRangeSet::new();
            for pn in &pns {
                set.insert(black_box(*pn));
            }
            black_box(set);
        });
    });
}

fn bench_ack_record_received(criterion: &mut Criterion) {
    criterion.bench_function("c13_ack_record_received_in_order_1000", |bencher| {
        bencher.iter(|| {
            let mut scheduler = AckScheduler::new();
            for pn in 0..1_000u64 {
                scheduler.record_received(black_box(pn), true, ORIGIN);
            }
            black_box(scheduler);
        });
    });
}

fn bench_ack_frame_encode_full_32_ranges(criterion: &mut Criterion) {
    // Worst case: 32 disjoint singleton ranges (max capacity reached).
    criterion.bench_function("c13_ack_frame_encode_32_disjoint_ranges", |bencher| {
        let mut scheduler = AckScheduler::new();
        for index in 0..32u64 {
            scheduler.record_received(index * 10, true, ORIGIN);
        }
        bencher.iter(|| {
            let largest = scheduler.largest_for_frame().expect("non-empty");
            let first_range = scheduler.first_range_length().unwrap_or(0);
            let ranges = scheduler.ranges();
            let range_count = ranges.len().saturating_sub(1) as u64;
            let mut buffer = [0u8; 512];
            let mut cursor = 0usize;
            buffer[cursor] = 0x02;
            cursor += 1;
            cursor += varint::encode(largest, &mut buffer[cursor..]).expect("largest");
            cursor += varint::encode(0, &mut buffer[cursor..]).expect("ack_delay");
            cursor += varint::encode(range_count, &mut buffer[cursor..]).expect("range_count");
            cursor += varint::encode(first_range, &mut buffer[cursor..]).expect("first_range");
            for pair in scheduler.ack_range_pairs() {
                cursor += varint::encode(pair.gap, &mut buffer[cursor..]).expect("gap");
                cursor += varint::encode(pair.length, &mut buffer[cursor..]).expect("length");
            }
            black_box(cursor);
        });
    });
}

criterion_group!(
    benches,
    bench_range_set_in_order_insert,
    bench_range_set_random_insert,
    bench_ack_record_received,
    bench_ack_frame_encode_full_32_ranges,
);
criterion_main!(benches);
