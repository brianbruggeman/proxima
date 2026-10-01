//! Counts the allocations a closure makes on the calling thread, so a test can
//! assert "zero" over a hot loop instead of arguing it. Installed as the test
//! binary's global allocator; other threads' allocations are not counted.

use core::alloc::{GlobalAlloc, Layout};
use core::cell::Cell;
use std::alloc::System;

std::thread_local! {
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

struct CountingAllocator;

// SAFETY: every call forwards to `System` unchanged; the counter is a
// const-initialised thread-local `Cell`, which never allocates.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = ALLOCATIONS.try_with(|count| count.set(count.get() + 1));
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let _ = ALLOCATIONS.try_with(|count| count.set(count.get() + 1));
        unsafe { System.realloc(pointer, layout, new_size) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let _ = ALLOCATIONS.try_with(|count| count.set(count.get() + 1));
        unsafe { System.alloc_zeroed(layout) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

pub(super) fn allocations_during(work: impl FnOnce()) -> usize {
    let before = ALLOCATIONS.with(Cell::get);
    work();
    ALLOCATIONS.with(Cell::get) - before
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_probe_counts_the_allocation_a_vec_makes() {
        let allocations = allocations_during(|| {
            let held: Vec<u64> = std::hint::black_box(Vec::with_capacity(64));
            drop(held);
        });

        assert_eq!(allocations, 1);
    }
}
