#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

use std::time::Duration;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Sample {
    id: String,
    timestamp: u64,
    items: Vec<SampleItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SampleItem {
    sku: String,
    quantity: u32,
    price_cents: u64,
    notes: String,
}

fn fixture_sample() -> Sample {
    Sample {
        id: "ord_01ARZ3NDEKTSV4RRFFQ69G5FAV".into(),
        timestamp: 1_741_390_800_123,
        items: (0..32)
            .map(|index| SampleItem {
                sku: format!("SKU-{index:08}"),
                quantity: (index as u32 + 1) * 7,
                price_cents: 1234 * (index as u64 + 1),
                notes: format!("notes for line {index} — special handling case"),
            })
            .collect(),
    }
}

fn decode_benchmark(criterion: &mut Criterion) {
    let sample = fixture_sample();
    let bytes = serde_json::to_vec(&sample).expect("encode fixture");
    let payload_len = bytes.len();
    let mut group = criterion.benchmark_group("json_decode");
    group.throughput(Throughput::Bytes(payload_len as u64));
    group.measurement_time(Duration::from_secs(3));
    group.bench_function("serde_json_from_slice", |bencher| {
        bencher.iter(|| {
            let parsed: Sample = serde_json::from_slice(&bytes).expect("serde_json");
            std::hint::black_box(parsed);
        });
    });
    group.bench_function("simd_json_from_slice", |bencher| {
        bencher.iter(|| {
            let mut owned = bytes.clone();
            let parsed: Sample = simd_json::serde::from_slice(&mut owned).expect("simd_json");
            std::hint::black_box(parsed);
        });
    });
    group.finish();
}

criterion_group!(benches, decode_benchmark);
criterion_main!(benches);
