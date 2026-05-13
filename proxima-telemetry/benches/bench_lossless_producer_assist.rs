//! Discipline bench for the elastic producer-assist arm of `lossless-backpressure`.
//!
//! Gate-13 incumbent arms: flume::bounded (blocking send on full),
//! crossbeam ArrayQueue (spin-retry on full), proxima Drop policy.
//! All arms measured on the same workload: N items pushed through a
//! ring of capacity C, with a concurrent drainer clearing the ring.
//!
//! Separate alloc-count measurements (not criterion — counting GlobalAlloc
//! interferes with criterion's internals) printed to stdout after the
//! criterion run; they are recorded in the discipline log.
//!
//! Run: `cargo bench -p proxima-telemetry --bench bench_lossless_producer_assist \
//!         --features lossless-backpressure -- --save-baseline lossless-backpressure`

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::thread;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group};
use proxima_telemetry::config::OverflowPolicy;
use proxima_telemetry::pipes::CountingPipe;
use proxima_telemetry::recorder::{Recorder, RingCapacities};

// ---- counting allocator (for alloc-count section, printed post-criterion) ---

static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);

struct CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

// ---- helpers ----------------------------------------------------------------

fn alloc_snap() -> usize {
    ALLOC_COUNT.load(Ordering::Relaxed)
}

// build a recorder backed by a fast no-latency sink under the given policy.
fn make_recorder(overflow: OverflowPolicy, ring_cap: usize) -> (Recorder, Arc<AtomicU64>) {
    let (pipe, spans, _, _, _, _) = CountingPipe::new();
    let recorder = Recorder::builder()
        .pipe(pipe)
        .core_count(1)
        .overflow(overflow)
        .ring_capacities(RingCapacities {
            spans: ring_cap,
            events: ring_cap,
            logs: ring_cap,
            metrics: ring_cap,
            links: ring_cap,
            overflow_attrs: ring_cap,
            #[cfg(feature = "deferred-metric-fold")]
            span_obs: ring_cap,
        })
        .start()
        .expect("recorder build failed");
    (recorder, spans)
}

// emit N log records to the recorder and return how many were produced.
fn emit_n(recorder: &Recorder, count: usize) -> usize {
    for seq in 0..count {
        recorder
            .log()
            .message("bench")
            .tag("seq", black_box(seq as u64))
            .emit();
    }
    count
}

// ---- bench group: steady-state (ring never full, hot-path) ------------------
// The 80% case. Ring cap >> batch size, a concurrent drainer always keeps room.
// This is the home-turf for both Drop and Block policies: zero assist overhead.

const RING_CAP_LARGE: usize = 8_192;

const BATCH_SIZES: &[usize] = &[16, 256, 1_024, 4_096];

fn bench_steady_state_drop(crit: &mut Criterion) {
    let mut group = crit.benchmark_group("lossless_steady_state");
    group.throughput(Throughput::Elements(1));

    for &batch in BATCH_SIZES {
        let (recorder, spans) = make_recorder(OverflowPolicy::Drop, RING_CAP_LARGE);

        group.bench_with_input(
            BenchmarkId::new("proxima_drop", batch),
            &batch,
            |bench, &count| {
                bench.iter(|| {
                    emit_n(&recorder, count);
                    recorder.drain();
                    black_box(spans.load(Ordering::Relaxed))
                });
            },
        );
    }
    group.finish();
}

fn bench_steady_state_block(crit: &mut Criterion) {
    let mut group = crit.benchmark_group("lossless_steady_state");
    group.throughput(Throughput::Elements(1));

    for &batch in BATCH_SIZES {
        let (recorder, spans) = make_recorder(OverflowPolicy::Block, RING_CAP_LARGE);

        group.bench_with_input(
            BenchmarkId::new("proxima_block_assist", batch),
            &batch,
            |bench, &count| {
                bench.iter(|| {
                    emit_n(&recorder, count);
                    recorder.drain();
                    black_box(spans.load(Ordering::Relaxed))
                });
            },
        );
    }
    group.finish();
}

// home-turf incumbent: flume bounded channel, blocking send (lossless under back-pressure)
fn bench_steady_state_flume(crit: &mut Criterion) {
    let mut group = crit.benchmark_group("lossless_steady_state");
    group.throughput(Throughput::Elements(1));

    for &batch in BATCH_SIZES {
        group.bench_with_input(
            BenchmarkId::new("flume_bounded_blocking", batch),
            &batch,
            |bench, &count| {
                let (tx, rx) = flume::bounded::<u64>(RING_CAP_LARGE);
                bench.iter(|| {
                    for item in 0..count as u64 {
                        tx.send(black_box(item)).expect("flume send failed");
                    }
                    while rx.try_recv().is_ok() {}
                    black_box(count)
                });
            },
        );
    }
    group.finish();
}

// home-turf incumbent: crossbeam ArrayQueue, spin-retry on full (lossless)
fn bench_steady_state_crossbeam(crit: &mut Criterion) {
    let mut group = crit.benchmark_group("lossless_steady_state");
    group.throughput(Throughput::Elements(1));

    for &batch in BATCH_SIZES {
        group.bench_with_input(
            BenchmarkId::new("crossbeam_array_queue_spin", batch),
            &batch,
            |bench, &count| {
                let queue = Arc::new(crossbeam_queue::ArrayQueue::<u64>::new(RING_CAP_LARGE));
                bench.iter(|| {
                    for item in 0..count as u64 {
                        while queue.push(black_box(item)).is_err() {}
                    }
                    while queue.pop().is_some() {}
                    black_box(count)
                });
            },
        );
    }
    group.finish();
}

// ---- bench group: saturation (ring fills, assist triggers) ------------------
// Ring capacity is tight (32 slots). Producer outpaces a drainer thread.
// Drop path sheds records; Block+assist path is lossless.
// Measures wall time per batch including assist stalls.

const RING_CAP_TIGHT: usize = 32;
const SATURATION_BATCH: usize = 512;

fn bench_saturation_drop(crit: &mut Criterion) {
    let mut group = crit.benchmark_group("lossless_saturation");

    let (recorder, spans) = make_recorder(OverflowPolicy::Drop, RING_CAP_TIGHT);
    let recorder = Arc::new(recorder);
    let recorder_drain = Arc::clone(&recorder);

    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let stop_drain = Arc::clone(&stop);
    let drain_handle = thread::spawn(move || {
        while !stop_drain.load(Ordering::Acquire) {
            recorder_drain.drain();
            thread::yield_now();
        }
        recorder_drain.drain();
    });

    group.bench_function("proxima_drop_tight_ring", |bench| {
        bench.iter(|| {
            emit_n(&recorder, SATURATION_BATCH);
            black_box(spans.load(Ordering::Relaxed))
        });
    });

    stop.store(true, Ordering::Release);
    drain_handle.join().expect("drain thread panicked");
    group.finish();
}

fn bench_saturation_block(crit: &mut Criterion) {
    let mut group = crit.benchmark_group("lossless_saturation");

    let (recorder, spans) = make_recorder(OverflowPolicy::Block, RING_CAP_TIGHT);
    let recorder = Arc::new(recorder);
    let recorder_drain = Arc::clone(&recorder);

    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let stop_drain = Arc::clone(&stop);
    let drain_handle = thread::spawn(move || {
        while !stop_drain.load(Ordering::Acquire) {
            recorder_drain.drain();
            thread::yield_now();
        }
        recorder_drain.drain();
    });

    group.bench_function("proxima_block_assist_tight_ring", |bench| {
        bench.iter(|| {
            emit_n(&recorder, SATURATION_BATCH);
            black_box(spans.load(Ordering::Relaxed))
        });
    });

    stop.store(true, Ordering::Release);
    drain_handle.join().expect("drain thread panicked");
    group.finish();
}

// flume incumbent at saturation: blocking send stalls the producer the same
// way assist stalls it — fair comparison on lossless correctness
fn bench_saturation_flume(crit: &mut Criterion) {
    let mut group = crit.benchmark_group("lossless_saturation");

    let (tx, rx) = flume::bounded::<u64>(RING_CAP_TIGHT);
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let stop_drain = Arc::clone(&stop);
    let drain_handle = thread::spawn(move || {
        while !stop_drain.load(Ordering::Acquire) {
            while rx.try_recv().is_ok() {}
            thread::yield_now();
        }
        while rx.try_recv().is_ok() {}
    });

    group.bench_function("flume_bounded_blocking_tight_ring", |bench| {
        bench.iter(|| {
            for item in 0..SATURATION_BATCH as u64 {
                tx.send(black_box(item)).expect("flume send");
            }
            black_box(SATURATION_BATCH)
        });
    });

    stop.store(true, Ordering::Release);
    drain_handle.join().expect("drain thread panicked");
    group.finish();
}

// ---- alloc-count report (printed after criterion, not a criterion bench) ----
// These are run in main() so the counting allocator is active.

fn report_alloc_counts() {
    println!("\nalloc-count report (hot-path, ring not full)");
    println!("path                                  allocs/emit");
    println!("----------------------------------------------");

    let (recorder_drop, _) = make_recorder(OverflowPolicy::Drop, RING_CAP_LARGE);
    let before = alloc_snap();
    for seq in 0..100usize {
        recorder_drop
            .log()
            .message("bench")
            .tag("seq", black_box(seq as u64))
            .emit();
    }
    let emit_allocs = alloc_snap() - before;
    println!("proxima_drop emit×100                 {}", emit_allocs);

    let before = alloc_snap();
    recorder_drop.drain();
    let drain_allocs = alloc_snap() - before;
    println!("proxima_drop drain×1 (batch)          {}", drain_allocs);

    let (recorder_block, _) = make_recorder(OverflowPolicy::Block, RING_CAP_LARGE);
    let before = alloc_snap();
    for seq in 0..100usize {
        recorder_block
            .log()
            .message("bench")
            .tag("seq", black_box(seq as u64))
            .emit();
    }
    let emit_allocs_block = alloc_snap() - before;
    println!(
        "proxima_block emit×100                {}",
        emit_allocs_block
    );

    let before = alloc_snap();
    recorder_block.drain();
    let drain_allocs_block = alloc_snap() - before;
    println!(
        "proxima_block drain×1 (batch)         {}",
        drain_allocs_block
    );

    println!("\nalloc-count report (cold-path, assist fires)");
    println!("path                                  allocs/batch");
    println!("----------------------------------------------");

    let (recorder_assist, _) = make_recorder(OverflowPolicy::Block, RING_CAP_TIGHT);
    let before = alloc_snap();
    for seq in 0..SATURATION_BATCH {
        recorder_assist
            .log()
            .message("bench")
            .tag("seq", black_box(seq as u64))
            .emit();
        if seq % (RING_CAP_TIGHT / 2) == 0 {
            recorder_assist.drain();
        }
    }
    let assist_allocs = alloc_snap() - before;
    println!(
        "proxima_block assist batch×{}        {}",
        SATURATION_BATCH, assist_allocs
    );
}

// ---- criterion wiring -------------------------------------------------------

criterion_group!(
    benches_steady,
    bench_steady_state_drop,
    bench_steady_state_block,
    bench_steady_state_flume,
    bench_steady_state_crossbeam,
);

criterion_group!(
    benches_saturation,
    bench_saturation_drop,
    bench_saturation_block,
    bench_saturation_flume,
);

// manual main so we can print the alloc-count report before criterion runs.
// criterion_group!-generated functions take no args; they construct Criterion internally.
fn main() {
    report_alloc_counts();
    benches_steady();
    benches_saturation();
    criterion::Criterion::default()
        .configure_from_args()
        .final_summary();
}
