#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

//! `CausalIndex::record` and `CausalIndex::explain` microbenches.
//! The causal substrate records (parent, child, byte-range) edges
//! as pipes dispatch; `explain` walks the graph backward from a
//! recorded output byte to its ancestors. Both are hot-path
//! observability primitives.

use std::time::Duration;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use proxima::{ByteRange, CausalEdge, CausalIndex};

fn record_single_edge(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("causal_record_single_edge");
    group.measurement_time(Duration::from_secs(3));
    group.throughput(Throughput::Elements(1));
    let index = CausalIndex::new();
    group.bench_function("record", |bencher| {
        bencher.iter(|| {
            let edge = CausalEdge {
                node_id: "child".into(),
                output_range: ByteRange { start: 0, end: 16 },
                parent_ranges: vec![("parent".into(), ByteRange { start: 0, end: 16 })],
            };
            index.record(edge);
        });
    });
    group.finish();
}

fn explain_walks_chain_of_depth_8(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("causal_explain_chain_depth_8");
    group.measurement_time(Duration::from_secs(3));
    group.throughput(Throughput::Elements(1));
    let index = CausalIndex::new();
    for depth in 0..8 {
        index.record(CausalEdge {
            node_id: format!("node{}", depth + 1),
            output_range: ByteRange { start: 0, end: 16 },
            parent_ranges: vec![(format!("node{depth}"), ByteRange { start: 0, end: 16 })],
        });
    }
    group.bench_function("explain", |bencher| {
        bencher.iter(|| {
            let chain = index.explain("node8", 0);
            std::hint::black_box(chain);
        });
    });
    group.finish();
}

criterion_group!(benches, record_single_edge, explain_walks_chain_of_depth_8);
criterion_main!(benches);
