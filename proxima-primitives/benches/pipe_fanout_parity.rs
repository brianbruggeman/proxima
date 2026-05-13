//! Parity gate: does the generic `FanOut<S, Policy>` cost what the hand-rolled
//! concrete fan-out costs? If the monomorphisation ties the concrete struct, the
//! genericity is free and the recording migration is safe (principle 16: prove,
//! don't assert).
//!
//! Three arms, interleaved in one criterion run so box drift cancels:
//!   - `concrete`  — a faithful copy of recording's `FanOut` (`Arc<Vec<S>>`,
//!     clone-per-call, move-into-last via `mem::take`).
//!   - `generic`   — `proxima_primitives::pipe::FanOut<S, AllOrNothing>`.
//!   - `smallvec`  — the concrete shape with the sinks container swapped to
//!     `SmallVec<[S; 4]>`, to put the "smallvec the sinks?" question on the
//!     record. Sinks are iterate-only on the hot path, so this should tie.
//!
//! Payload is `bytes::Bytes` (Arc-backed: clone == refcount bump), mirroring the
//! Arc-backed `RecordingEvent`. A counting global allocator also prints the
//! deterministic per-call allocation count for each arm before the timing run.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::future::Future;
use std::hint::black_box;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

use bytes::Bytes;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use futures::executor::block_on;
use proxima_primitives::pipe::{AllOrNothing, FanOut, IgnoreErrors};
use proxima_primitives::pipe::SendPipe;
use smallvec::{SmallVec, smallvec};

// ── counting allocator (deterministic alloc-count truth, not timing) ──────────

struct Counting;

static ALLOCS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

// ── a trivial sink: counts calls, never fails ────────────────────────────────

struct CountingSink(Arc<AtomicUsize>);

impl SendPipe for CountingSink {
    type In = Bytes;
    type Out = ();
    type Err = std::convert::Infallible;

    fn call(&self, input: Bytes) -> impl Future<Output = Result<(), Self::Err>> + Send {
        let calls = Arc::clone(&self.0);
        async move {
            calls.fetch_add(input.len(), Relaxed);
            Ok(())
        }
    }
}

// ── the concrete incumbent: recording's exact FanOut shape ───────────────────

struct ConcreteFan {
    sinks: Arc<Vec<CountingSink>>,
}

impl ConcreteFan {
    async fn call(&self, mut item: Bytes) {
        let sinks = Arc::clone(&self.sinks);
        let last = sinks.len().saturating_sub(1);
        for (index, sink) in sinks.iter().enumerate() {
            let batch = if index == last {
                core::mem::take(&mut item)
            } else {
                item.clone()
            };
            sink.call(batch).await.unwrap();
        }
    }
}

// ── the smallvec-sinks variant of the concrete shape ─────────────────────────

struct SmallVecFan {
    sinks: Arc<SmallVec<[CountingSink; 4]>>,
}

impl SmallVecFan {
    async fn call(&self, mut item: Bytes) {
        let sinks = Arc::clone(&self.sinks);
        let last = sinks.len().saturating_sub(1);
        for (index, sink) in sinks.iter().enumerate() {
            let batch = if index == last {
                core::mem::take(&mut item)
            } else {
                item.clone()
            };
            sink.call(batch).await.unwrap();
        }
    }
}

fn sinks(count: usize, hits: &Arc<AtomicUsize>) -> Vec<CountingSink> {
    (0..count).map(|_| CountingSink(Arc::clone(hits))).collect()
}

fn payload() -> Bytes {
    Bytes::from_static(b"recording-event-payload-mirror-arc-backed")
}

fn alloc_count_for<Fut: Future>(call: impl Fn() -> Fut) -> usize {
    drop(block_on(call())); // warm any one-time lazy alloc; allocator reads are the observable effect
    let before = ALLOCS.load(Relaxed);
    drop(block_on(call()));
    ALLOCS.load(Relaxed) - before
}

fn report_alloc_parity() {
    let hits = Arc::new(AtomicUsize::new(0));
    let concrete = ConcreteFan {
        sinks: Arc::new(sinks(3, &hits)),
    };
    let generic = FanOut::<_, AllOrNothing>::new(sinks(3, &hits));
    let small: SmallVecFan = SmallVecFan {
        sinks: Arc::new(smallvec![
            CountingSink(Arc::clone(&hits)),
            CountingSink(Arc::clone(&hits)),
            CountingSink(Arc::clone(&hits)),
        ]),
    };

    let concrete_allocs = alloc_count_for(|| concrete.call(payload()));
    let generic_allocs = alloc_count_for(|| generic.call(payload()));
    let small_allocs = alloc_count_for(|| small.call(payload()));

    eprintln!(
        "fanout alloc/call @3 sinks: concrete={concrete_allocs} generic={generic_allocs} smallvec={small_allocs}"
    );
}

fn bench(criterion: &mut Criterion) {
    report_alloc_parity();

    let mut group = criterion.benchmark_group("fanout_parity");
    for count in [1usize, 3, 8] {
        let hits = Arc::new(AtomicUsize::new(0));
        let concrete = ConcreteFan {
            sinks: Arc::new(sinks(count, &hits)),
        };
        let generic = FanOut::<_, AllOrNothing>::new(sinks(count, &hits));
        let ignore = FanOut::<_, IgnoreErrors>::new(sinks(count, &hits));
        let small: SmallVecFan = SmallVecFan {
            sinks: Arc::new(
                (0..count)
                    .map(|_| CountingSink(Arc::clone(&hits)))
                    .collect(),
            ),
        };

        group.bench_with_input(BenchmarkId::new("concrete", count), &count, |bencher, _| {
            bencher.iter(|| block_on(concrete.call(black_box(payload()))));
        });
        group.bench_with_input(BenchmarkId::new("generic", count), &count, |bencher, _| {
            bencher.iter(|| block_on(generic.call(black_box(payload()))).unwrap());
        });
        group.bench_with_input(BenchmarkId::new("smallvec", count), &count, |bencher, _| {
            bencher.iter(|| block_on(small.call(black_box(payload()))));
        });
        // the telemetry fire-and-forget 80% case — IGNORE_ERRORS must const-fold
        // to the same hot path as concrete (no first_err slot kept).
        group.bench_with_input(
            BenchmarkId::new("ignore_errors", count),
            &count,
            |bencher, _| {
                bencher.iter(|| block_on(ignore.call(black_box(payload()))).unwrap());
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
