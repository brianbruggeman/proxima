//! Parity gate for the `BoundedQueue` and `Batch` primitives: does the generic
//! cost what the inline hand-rolled logic costs? Like the fanout gate, the
//! migration off recording's inline code is only safe if the primitive ties it
//! (principle 16). Arms interleaved; counting allocator for the deterministic
//! alloc truth.
//!
//!   - bounded enqueue passthrough: generic `BoundedQueue::enqueue`+`dequeue`
//!     vs inline `ArrayQueue::push`+`pop`.
//!   - bounded overflow: generic `enqueue` (DropOldest) vs inline `force_push`
//!     + drop counter, on a full queue.
//!   - batch push+flush: generic `Batch::push` vs inline `Mutex<Vec>` push +
//!     `mem::take`.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed};

use criterion::{Criterion, criterion_group, criterion_main};
use crossbeam_queue::ArrayQueue;
use proxima_primitives::pipe::{Batch, BoundedQueue, EnqueueOutcome, FailMode};

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

// inline mirror of recording's enqueue-with-policy (the pre-extraction code).
struct InlineBounded {
    queue: ArrayQueue<u64>,
    drops: AtomicU64,
}
impl InlineBounded {
    fn new(capacity: usize) -> Self {
        Self {
            queue: ArrayQueue::new(capacity),
            drops: AtomicU64::new(0),
        }
    }
    fn enqueue_drop_oldest(&self, item: u64) {
        if let Err(rejected) = self.queue.push(item) {
            let _evicted = self.queue.force_push(rejected);
            self.drops.fetch_add(1, Relaxed);
        }
    }
}

fn report_alloc_parity() {
    let generic = BoundedQueue::<u64>::new(64, FailMode::DropOldest);
    let inline = InlineBounded::new(64);
    let g_before = ALLOCS.load(Relaxed);
    assert_eq!(generic.enqueue(1), EnqueueOutcome::Enqueued);
    let _ = generic.dequeue();
    let g = ALLOCS.load(Relaxed) - g_before;
    let i_before = ALLOCS.load(Relaxed);
    inline.enqueue_drop_oldest(1);
    let _ = inline.queue.pop();
    let i = ALLOCS.load(Relaxed) - i_before;

    let batch = Batch::<u64>::new(4);
    let mvec: Mutex<Vec<u64>> = Mutex::new(Vec::with_capacity(4));
    let b_before = ALLOCS.load(Relaxed);
    let _ = batch.push(7);
    let b = ALLOCS.load(Relaxed) - b_before;
    let m_before = ALLOCS.load(Relaxed);
    mvec.lock().unwrap().push(7);
    let m = ALLOCS.load(Relaxed) - m_before;

    eprintln!("alloc/op: bounded generic={g} inline={i} | batch generic={b} inline={m}");
}

fn bench(criterion: &mut Criterion) {
    report_alloc_parity();

    let mut bounded = criterion.benchmark_group("bounded_parity");
    let generic = BoundedQueue::<u64>::new(1024, FailMode::DropOldest);
    let inline = InlineBounded::new(1024);
    bounded.bench_function("generic_passthrough", |bencher| {
        bencher.iter(|| {
            let _ = generic.enqueue(black_box(1));
            black_box(generic.dequeue())
        });
    });
    bounded.bench_function("inline_passthrough", |bencher| {
        bencher.iter(|| {
            inline.enqueue_drop_oldest(black_box(1));
            black_box(inline.queue.pop())
        });
    });
    // overflow path: a full queue, every op evicts-and-inserts.
    let full_generic = BoundedQueue::<u64>::new(1, FailMode::DropOldest);
    let _ = full_generic.enqueue(0);
    let full_inline = InlineBounded::new(1);
    full_inline.enqueue_drop_oldest(0);
    bounded.bench_function("generic_overflow", |bencher| {
        bencher.iter(|| black_box(full_generic.enqueue(black_box(9))));
    });
    bounded.bench_function("inline_overflow", |bencher| {
        bencher.iter(|| full_inline.enqueue_drop_oldest(black_box(9)));
    });
    bounded.finish();

    let mut batch_group = criterion.benchmark_group("batch_parity");
    let batch = Batch::<u64>::new(4);
    let mvec: Mutex<Vec<u64>> = Mutex::new(Vec::with_capacity(4));
    batch_group.bench_function("generic_push_flush", |bencher| {
        bencher.iter(|| black_box(batch.push(black_box(7))));
    });
    batch_group.bench_function("inline_push_flush", |bencher| {
        bencher.iter(|| {
            let mut guard = mvec.lock().unwrap();
            guard.push(black_box(7));
            if guard.len() >= 4 {
                black_box(Some(core::mem::take(&mut *guard)))
            } else {
                black_box(None)
            }
        });
    });
    batch_group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
