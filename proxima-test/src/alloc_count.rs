//! Zero-allocation and no-leak proof gate: install [`CountingAllocator`] as a
//! test binary's `#[global_allocator]`, then read
//! [`allocations`]/[`live_bytes`]/[`reset`] around the section under proof.
//!
//! Composes with nothing beyond `std::alloc::GlobalAlloc` — `#[global_allocator]`
//! is process-wide, so the caller owns installing it once, at the top of a
//! `tests/*.rs` or `benches/*.rs` binary; `cargo nextest` gives each such
//! binary its own process, so per-binary counts stay clean without `reset`.
//! What a caller gets that they did not have before: the same zero-allocation
//! and leak-check gate in every crate's test/bench binaries without
//! re-minting the `GlobalAlloc` forwarding impl each time.
//!
//! `allocations` proves a hot path allocates nothing; `live_bytes` proves a
//! build-then-drop cycle returns to its pre-build baseline (no leak) — pick
//! the one the claim needs, both come from the one installed allocator.
//!
//! ```
//! # use proxima_test::alloc_count::{CountingAllocator, allocations, live_bytes};
//! #[global_allocator]
//! static ALLOCATOR: CountingAllocator = CountingAllocator;
//!
//! let before = allocations();
//! let live_before = live_bytes();
//! let boxed = Box::new(1u64);
//! assert!(allocations() > before);
//! assert!(live_bytes() > live_before);
//! drop(boxed);
//! assert_eq!(live_bytes(), live_before);
//! ```

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicI64, AtomicUsize, Ordering};

static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
static LIVE_BYTES: AtomicI64 = AtomicI64::new(0);

/// Global allocator that forwards every call to [`System`], counts each
/// `alloc`/`alloc_zeroed`/`realloc` call ([`allocations`]), and tracks the net
/// live byte count ([`live_bytes`]) so a caller can prove both "zero
/// allocations on the hot path" and "no leak over a soak" from the same
/// `#[global_allocator]`. Install with `#[global_allocator]`.
pub struct CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        LIVE_BYTES.fetch_add(layout.size() as i64, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE_BYTES.fetch_sub(layout.size() as i64, Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        LIVE_BYTES.fetch_add(layout.size() as i64, Ordering::Relaxed);
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        LIVE_BYTES.fetch_add(new_size as i64 - layout.size() as i64, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

/// Current allocation count since process start or the last [`reset`].
#[must_use]
pub fn allocations() -> usize {
    ALLOC_COUNT.load(Ordering::Relaxed)
}

/// Net live bytes (bytes allocated minus bytes freed) since process start or
/// the last [`reset`]. A leak check builds something, drops it, and asserts
/// this returns to (approximately) its pre-build baseline.
#[must_use]
pub fn live_bytes() -> i64 {
    LIVE_BYTES.load(Ordering::Relaxed)
}

/// Zeroes both counters. Only meaningful with no concurrent allocation-bearing
/// work in flight — `cargo nextest` isolates each test to its own process, so
/// a fresh counter per test is the common case, and `reset` is for
/// rebaselining mid-binary (e.g. between cases in the same `#[test]` file).
pub fn reset() {
    ALLOC_COUNT.store(0, Ordering::Relaxed);
    LIVE_BYTES.store(0, Ordering::Relaxed);
}
