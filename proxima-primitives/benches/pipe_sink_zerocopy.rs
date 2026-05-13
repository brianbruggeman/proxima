//! Zero-copy `DrainSink` vs an owned-`Vec` sink, on VARIABLE-length frames.
//!
//! The real `DrainSink` property is not raw speed on fixed `[u8; N]` arrays
//! (those copy cheaply, and a first bench attempt with an asymmetric read-back
//! misleadingly showed owned faster — a classic "reason-from-code-around-timing"
//! trap). The property is: a borrowed `&[u8]` of ANY length is written straight
//! into a fixed ring slot with ZERO heap allocation, whereas an owned sink over
//! variable-length data (`SendPipe<In=Vec<u8>>`) must allocate a `Vec` per
//! frame. The counting allocator is the deterministic headline: 0 vs N.
//!
//! design-favors: incumbent = the owned-Vec sink (the std way to consume
//! variable-length items); home turf = per-core ring of variable telemetry
//! frames → one consumer.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use proxima_primitives::pipe::{DrainSink, RingSink};

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

const FRAMES: usize = 256;
const SLOT: usize = 256;

// owned sink over variable-length items — the std shape (a SendPipe<In=Vec<u8>>
// consumer). Each accepted frame is an owned Vec (one heap alloc per frame).
struct OwnedVecSink {
    held: Vec<Vec<u8>>,
}
impl OwnedVecSink {
    fn new() -> Self {
        Self {
            held: Vec::with_capacity(FRAMES),
        }
    }
    fn accept_owned(&mut self, item: Vec<u8>) {
        self.held.push(item);
    }
    fn clear(&mut self) {
        self.held.clear();
    }
}

fn make_source() -> (Vec<u8>, [usize; FRAMES]) {
    // one backing buffer + per-frame lengths (variable: 8..=SLOT bytes).
    let mut lens = [0usize; FRAMES];
    let mut total = 0usize;
    for (index, len) in lens.iter_mut().enumerate() {
        *len = 8 + (index % (SLOT - 8));
        total += *len;
    }
    let mut buf = vec![0u8; total];
    for (index, byte) in buf.iter_mut().enumerate() {
        *byte = index as u8;
    }
    (buf, lens)
}

fn zerocopy_push(buf: &[u8], lens: &[usize; FRAMES], sink: &mut RingSink<FRAMES, SLOT>) -> u64 {
    let mut offset = 0usize;
    let mut acc = 0u64;
    for &len in lens.iter() {
        let frame = &buf[offset..offset + len]; // borrowed view — no alloc
        offset += len;
        if sink.accept(frame).is_break() {
            break;
        }
    }
    while let Some(view) = sink.pop() {
        acc = acc.wrapping_add(view.len() as u64);
    }
    acc
}

fn owned_push(buf: &[u8], lens: &[usize; FRAMES], sink: &mut OwnedVecSink) -> u64 {
    sink.clear();
    let mut offset = 0usize;
    for &len in lens.iter() {
        let owned: Vec<u8> = buf[offset..offset + len].to_vec(); // HEAP alloc per frame
        offset += len;
        sink.accept_owned(owned);
    }
    let mut acc = 0u64;
    for item in &sink.held {
        acc = acc.wrapping_add(item.len() as u64);
    }
    acc
}

fn report_alloc(buf: &[u8], lens: &[usize; FRAMES]) {
    let mut zc: RingSink<FRAMES, SLOT> = RingSink::new();
    let mut owned = OwnedVecSink::new();
    let _ = black_box(zerocopy_push(buf, lens, &mut zc));
    let _ = black_box(owned_push(buf, lens, &mut owned));
    let mut zc2: RingSink<FRAMES, SLOT> = RingSink::new();
    let before = ALLOCS.load(Relaxed);
    let _ = black_box(zerocopy_push(buf, lens, &mut zc2));
    let zc_a = ALLOCS.load(Relaxed) - before;
    let before = ALLOCS.load(Relaxed);
    let _ = black_box(owned_push(buf, lens, &mut owned));
    let owned_a = ALLOCS.load(Relaxed) - before;
    eprintln!("alloc/push (256 variable frames): zerocopy={zc_a} owned_vec={owned_a}");
}

fn bench(criterion: &mut Criterion) {
    let (buf, lens) = make_source();
    report_alloc(&buf, &lens);

    let mut group = criterion.benchmark_group("sink_zerocopy");
    let mut zc: RingSink<FRAMES, SLOT> = RingSink::new();
    group.bench_function(BenchmarkId::new("zerocopy_accept", "var"), |bencher| {
        bencher.iter(|| black_box(zerocopy_push(black_box(&buf), &lens, &mut zc)));
    });
    let mut owned = OwnedVecSink::new();
    group.bench_function(BenchmarkId::new("owned_vec_accept", "var"), |bencher| {
        bencher.iter(|| black_box(owned_push(black_box(&buf), &lens, &mut owned)));
    });
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
