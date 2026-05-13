//! Fan-in merge: zero-copy `DrainFanIn` vs owned `FanIn` vs the std incumbent
//! `futures::stream::select_all`.
//!
//! The headline claim is ZERO-COPY: the push `DrainFanIn` reads each frame as a
//! borrowed `&[u8]` slot view (no per-frame copy, no alloc), while the owned
//! `FanIn` and `select_all` move each frame out (an `[u8; S]` copy per item).
//! Frames are pre-filled into a stack arena ONCE; each timed iteration only
//! re-presents them (cheap cursor reset / FSM rebuild), so the measurement
//! isolates the merge-read, not the producer fill.
//!
//! design-favors: `select_all` = incumbent (the std N→1 merge on its turf —
//! N sources of items merged into one); `drain`/`fanin_owned` = proxima.
//! Home turf (gate-13): N per-core ring sources of byte frames → one consumer,
//! the telemetry-drain / *DK shape.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::future::Future;
use std::hint::black_box;
use std::ops::ControlFlow;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::task::{Context, Poll, Waker};

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use futures::stream::{self, StreamExt};
use proxima_primitives::pipe::{
    DrainFanIn, DrainSource, DrainState, Exhausted, FanIn, Pipe, Select, UnpinPipe,
};

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

const SOURCES: usize = 8;
const FRAMES: usize = 64;

// zero-copy source: borrows frame views out of a shared pre-filled arena.
#[derive(Clone, Copy)]
struct ZcRing<'arena, const S: usize> {
    arena: &'arena [[u8; S]; FRAMES],
    read: usize,
}

impl<const S: usize> DrainSource for ZcRing<'_, S> {
    type Item = [u8];
    fn drain_ready(&mut self, visitor: &mut dyn FnMut(&[u8]) -> ControlFlow<()>) -> DrainState {
        while self.read < FRAMES {
            let view: &[u8] = &self.arena[self.read][..];
            self.read += 1;
            if visitor(view).is_break() {
                return DrainState::More;
            }
        }
        DrainState::Drained
    }
}

// owned source: copies an `[u8; S]` out of the same arena per item. `read` is
// an atomic cursor because `UnpinPipe::call` takes `&self` (FanIn's sources
// are called through a shared reference, not a mutable one).
struct OwnedRing<'arena, const S: usize> {
    arena: &'arena [[u8; S]; FRAMES],
    read: AtomicUsize,
}

impl<const S: usize> proxima_core::markers::DropSafe for OwnedRing<'_, S> {}

impl<const S: usize> UnpinPipe for OwnedRing<'_, S> {
    type In = ();
    type Out = [u8; S];
    type Err = Exhausted;

    fn call(&self, (): ()) -> impl Future<Output = Result<[u8; S], Exhausted>> + Unpin {
        let read = self.read.load(Relaxed);
        if read >= FRAMES {
            return core::future::ready(Err(Exhausted));
        }
        let frame = self.arena[read]; // [u8; S]: Copy => an S-byte copy
        self.read.store(read + 1, Relaxed);
        core::future::ready(Ok(frame))
    }
}

fn sum_bytes(frame: &[u8]) -> u64 {
    frame.iter().map(|&byte| byte as u64).sum()
}

fn drain_merge<const S: usize>(arena: &[[u8; S]; FRAMES]) -> u64 {
    let mut fan = DrainFanIn::new([ZcRing { arena, read: 0 }; SOURCES]);
    let mut sum = 0u64;
    fan.drain_each(|frame: &[u8]| {
        sum += sum_bytes(frame);
        ControlFlow::Continue(())
    });
    sum
}

fn owned_merge<const S: usize>(arena: &[[u8; S]; FRAMES]) -> u64 {
    let fan = FanIn::new(
        core::array::from_fn::<_, SOURCES, _>(|_| OwnedRing {
            arena,
            read: AtomicUsize::new(0),
        }),
        Select::RoundRobin,
    );
    let waker = Waker::noop();
    let mut cx = Context::from_waker(waker);
    let mut sum = 0u64;
    loop {
        let mut call = Pipe::call(&fan, ());
        match Pin::new(&mut call).poll(&mut cx) {
            Poll::Ready(Ok(frame)) => sum += sum_bytes(&frame),
            Poll::Ready(Err(Exhausted)) => break,
            Poll::Pending => {}
        }
    }
    sum
}

fn select_all_merge<const S: usize>(arena: &[[u8; S]; FRAMES]) -> u64 {
    // the std N→1 merge: each source is a stream yielding owned [u8; S] frames.
    let mut merged = stream::select_all(
        (0..SOURCES).map(|_| stream::iter((0..FRAMES).map(|index| arena[index]))),
    );
    let waker = Waker::noop();
    let mut cx = Context::from_waker(waker);
    let mut sum = 0u64;
    loop {
        match merged.poll_next_unpin(&mut cx) {
            Poll::Ready(Some(frame)) => sum += sum_bytes(&frame),
            Poll::Ready(None) => break,
            Poll::Pending => {}
        }
    }
    sum
}

fn report_allocs<const S: usize>(arena: &[[u8; S]; FRAMES], label: &str) {
    let snap = |f: &dyn Fn() -> u64| {
        let _ = black_box(f()); // warm
        let before = ALLOCS.load(Relaxed);
        let _ = black_box(f());
        ALLOCS.load(Relaxed) - before
    };
    let drain = snap(&|| drain_merge(arena));
    let owned = snap(&|| owned_merge(arena));
    let select = snap(&|| select_all_merge(arena));
    eprintln!("alloc/merge S={S} [{label}]: drain={drain} owned={owned} select_all={select}");
}

fn bench_size<const S: usize>(criterion: &mut Criterion, label: &str) {
    let mut arena = [[0u8; S]; FRAMES];
    for (index, slot) in arena.iter_mut().enumerate() {
        slot[0] = index as u8;
        slot[S / 2] = (index as u8).wrapping_mul(3);
    }
    report_allocs::<S>(&arena, label);

    let mut group = criterion.benchmark_group("fanin_merge");
    group.bench_function(BenchmarkId::new("drain_zerocopy", label), |bencher| {
        bencher.iter(|| black_box(drain_merge(black_box(&arena))));
    });
    group.bench_function(BenchmarkId::new("fanin_owned", label), |bencher| {
        bencher.iter(|| black_box(owned_merge(black_box(&arena))));
    });
    group.bench_function(BenchmarkId::new("select_all", label), |bencher| {
        bencher.iter(|| black_box(select_all_merge(black_box(&arena))));
    });
    group.finish();
}

fn bench(criterion: &mut Criterion) {
    bench_size::<16>(criterion, "16B"); // small frame: copy is cheap
    bench_size::<256>(criterion, "256B"); // realistic frame: copy is the cost
}

criterion_group!(benches, bench);
criterion_main!(benches);
