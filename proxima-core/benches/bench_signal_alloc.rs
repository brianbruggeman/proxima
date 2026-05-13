#![allow(clippy::unwrap_used, clippy::expect_used)]
#![cfg(feature = "std")]

//! Per-construction allocation count for [`proxima_core::signal::Signal`] —
//! `Signal::new()`, `Signal::child()`, and one/two `Fired` waker
//! registration cycles (the per-request cost the h1 pipe path pays under
//! `Connection: close` — `proxima-net/src/pipe_connection.rs`'s
//! `drive_frame_pipe` registers a fresh `cancel.fired()` on every loop
//! iteration against the same long-lived `Signal`, and one request runs
//! exactly two iterations). Same `CountingAllocator` idiom as
//! `proxima-http/benches/bench_http1_pipe_serve_alloc.rs` (RISC reuse, P1):
//! a `#[global_allocator]` wrapper bumps an atomic on every
//! `alloc`/`alloc_zeroed`/`realloc`; delta-count / iteration-count gives
//! allocs/op.
//!
//! Not a criterion bench — direct allocator-counter access at the
//! construction boundary is the point, so this crate drives its own loop.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};
use std::task::{Context, Poll, Waker};

use proxima_core::signal::Signal;

static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);

struct CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

const ITERATIONS: u64 = 1000;

fn allocs_per_op(label: &str, mut workload: impl FnMut()) {
    workload();
    let before = ALLOCATIONS.load(Ordering::Relaxed);
    for _ in 0..ITERATIONS {
        workload();
    }
    let after = ALLOCATIONS.load(Ordering::Relaxed);
    let total = after - before;
    #[allow(clippy::cast_precision_loss)]
    let per_op = total as f64 / ITERATIONS as f64;
    println!("{label}: {total} allocations / {ITERATIONS} ops = {per_op:.3} allocs/op");
}

fn main() {
    allocs_per_op("Signal::new()", || {
        let signal = Signal::new();
        std::hint::black_box(&signal);
    });

    allocs_per_op("Signal::new().child()", || {
        let signal = Signal::new();
        let child = signal.child();
        std::hint::black_box(&child);
    });

    // `Waker::noop()` (stable, alloc-free, no Arc) isolates Signal's own
    // registration cost from any waker-construction cost — the h1 pipe
    // alloc bench already charges a real connection waker separately.
    let waker: &Waker = Waker::noop();
    allocs_per_op(
        "Signal::new() + one Fired registration (poll, Pending)",
        || {
            let signal = Signal::new();
            let mut fired = signal.fired();
            let mut cx = Context::from_waker(waker);
            let poll_result = std::pin::Pin::new(&mut fired).poll(&mut cx);
            assert_eq!(poll_result, Poll::Pending);
            std::hint::black_box(&fired);
        },
    );

    allocs_per_op(
        "Signal::new() + Fired registration + drop (unregister)",
        || {
            let signal = Signal::new();
            let mut cx = Context::from_waker(waker);
            {
                let mut fired = signal.fired();
                let poll_result = std::pin::Pin::new(&mut fired).poll(&mut cx);
                assert_eq!(poll_result, Poll::Pending);
            }
        },
    );

    // mirrors `drive_frame_pipe`'s loop shape (proxima-net/src/pipe_connection.rs):
    // a fresh `cancel.fired()` is registered-then-dropped on EACH loop
    // iteration against the SAME long-lived Signal — one request through the
    // h1 pipe path runs exactly two iterations (drain the bytes, then observe
    // the closed read side), so this is the per-request registration count,
    // isolated from Signal::new()'s own one-time cost.
    allocs_per_op(
        "Signal::new() + TWO sequential Fired register+drop cycles (same signal)",
        || {
            let signal = Signal::new();
            let mut cx = Context::from_waker(waker);
            for _ in 0..2 {
                let mut fired = signal.fired();
                let poll_result = std::pin::Pin::new(&mut fired).poll(&mut cx);
                assert_eq!(poll_result, Poll::Pending);
            }
        },
    );
}
