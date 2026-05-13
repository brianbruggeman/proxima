#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]
//! Typed fan-out micro-bench.
//!
//! Proves the in-process typed fan-out zero-copy property: the drainer
//! builds `Arc<Vec<Arc<Record>>>` once; each downstream consumer clones
//! the outer Arc — ONE atomic bump per branch — then borrows each record
//! (no per-record copy). The bench pins the number.

use std::hint::black_box;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use criterion::{Criterion, criterion_group, criterion_main};
use futures::executor::block_on;
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::{FanOut, IgnoreErrors};

struct BenchRecord {
    name: &'static str,
    start_ns: u64,
    duration_ns: u64,
}

const BATCH: usize = 1000;
const BRANCHES: usize = 8;

// a push sink that mirrors the SharedRing branch's read work: clone the shared
// batch handle (one bump), read every record. As a `SendPipe`, it is what a
// `FanOut` push fan delivers to.
struct ReaderSink(Arc<AtomicU64>);

impl SendPipe for ReaderSink {
    type In = Arc<Vec<Arc<BenchRecord>>>;
    type Out = ();
    type Err = core::convert::Infallible;

    fn call(
        &self,
        batch: Arc<Vec<Arc<BenchRecord>>>,
    ) -> impl core::future::Future<Output = Result<(), core::convert::Infallible>> + Send {
        let sink = Arc::clone(&self.0);
        async move {
            let mut acc = 0u64;
            for record in batch.iter() {
                acc = acc.wrapping_add(record.start_ns ^ record.duration_ns);
                black_box(record.name);
            }
            sink.fetch_add(acc, Ordering::Relaxed);
            Ok(())
        }
    }
}

fn bench_carry_fanout(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("typed_fanout");

    let records: Arc<Vec<Arc<BenchRecord>>> = Arc::new(
        (0..BATCH)
            .map(|index| {
                Arc::new(BenchRecord {
                    name: "bench.span",
                    start_ns: index as u64,
                    duration_ns: 42,
                })
            })
            .collect(),
    );

    // INCUMBENT (design-favors: incumbent) — telemetry's SharedRing pull fan:
    // each branch clones the shared batch handle (one bump) and reads. No pipe,
    // no async. This is telemetry's deliberate zero-copy drain model.
    group.bench_function("sharedring_pull_1000x8", |bencher| {
        bencher.iter(|| {
            let mut acc = 0u64;
            for _ in 0..BRANCHES {
                let branch = black_box(Arc::clone(&records));
                // identical inner work to ReaderSink (same per-record barrier on
                // both arms) — a fair pull-vs-push comparison.
                for record in branch.iter() {
                    acc = acc.wrapping_add(record.start_ns ^ record.duration_ns);
                    black_box(record.name);
                }
            }
            black_box(acc)
        });
    });

    // CHALLENGER (design-favors: proxima) — the generic push FanOut delivering
    // the same shared batch to 8 ReaderSink pipes. Same N Arc bumps; the delta
    // is the async pipe-dispatch the push model adds over the bare pull loop.
    let counter = Arc::new(AtomicU64::new(0));
    let fan = FanOut::<_, IgnoreErrors>::new(
        (0..BRANCHES)
            .map(|_| ReaderSink(Arc::clone(&counter)))
            .collect(),
    );
    group.bench_function("fanout_push_1000x8", |bencher| {
        bencher.iter(|| {
            black_box(block_on(fan.call(black_box(Arc::clone(&records))))).unwrap();
        });
    });
    group.finish();
}

criterion_group!(benches, bench_carry_fanout);
criterion_main!(benches);
