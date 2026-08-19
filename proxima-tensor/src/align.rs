//! An opt-in page-aligned `f32` buffer.
//!
//! Ordinary tensor storage in this crate is a plain `Vec<f32>` — the right
//! default, since most buffers here are small and a page (16 KiB on Apple
//! silicon, queried at runtime rather than assumed) would waste real memory
//! on them. But a `Vec<f32>` is only 4-byte aligned, and a device backend
//! that shares unified memory with the host (`omega::metal`'s
//! `newBufferWithBytesNoCopy` path) can only skip its host->device copy when
//! a buffer's pointer AND byte length both land on a page boundary. Nothing
//! in the ordinary `evaluate`/`evaluate_pooled` path needs that, so this
//! type stays out of it entirely — it exists only for a caller who is about
//! to hand a block straight to that kind of backend and deliberately reaches
//! for it.
//!
//! [`AlignedBuffer`] cannot be a `Vec<f32>`: `Vec<T>`'s allocate and
//! deallocate calls both derive their `Layout` from `T`'s natural alignment
//! (4 bytes for `f32`), never from a larger alignment requested at
//! construction time, so wrapping a manually over-aligned allocation in a
//! `Vec<f32>` would free it with the wrong `Layout` — undefined behavior.
//! This type is the smallest thing that can own such an allocation: it
//! records the exact `Layout` `alloc_zeroed` returned it, and frees with
//! that same `Layout`, nothing more.

use alloc::alloc::{alloc_zeroed, dealloc, handle_alloc_error};
use core::alloc::Layout;
use core::mem::size_of;
use core::ops::{Deref, DerefMut};
use core::ptr::NonNull;
use core::slice;

/// An owned, zero-initialized `f32` buffer whose backing allocation's
/// pointer and byte length are both a multiple of `page_size`.
///
/// The element count actually usable through [`Deref`] is the requested
/// count rounded up to the next page multiple, never the raw request — the
/// no-copy alignment check a caller reaches for this type to satisfy tests
/// the *whole* slice's byte length, so a caller wanting the guarantee to
/// hold must size its tensor input to `buffer.len()`, not the value it
/// originally asked [`AlignedBuffer::new`] for.
pub struct AlignedBuffer {
    ptr: NonNull<f32>,
    len: usize,
    layout: Layout,
}

// SAFETY: `AlignedBuffer` exclusively owns its allocation the same way
// `Vec<f32>` owns its own — no other handle to the memory exists.
unsafe impl Send for AlignedBuffer {}
unsafe impl Sync for AlignedBuffer {}

impl AlignedBuffer {
    /// Allocates room for at least `min_elements` `f32`s, rounded up so the
    /// allocation's byte length is a multiple of `page_size`.
    ///
    /// `page_size` must be a power of two — a real host page size the
    /// caller queried itself (e.g. `omega::metal::page_size`), never
    /// hard-coded here, since it varies across hosts.
    #[must_use]
    pub fn new(min_elements: usize, page_size: usize) -> Self {
        let element_bytes = min_elements
            .checked_mul(size_of::<f32>())
            .unwrap_or_else(|| panic_on_overflow(min_elements));
        let rounded_bytes = element_bytes.next_multiple_of(page_size).max(page_size);
        let layout = Layout::from_size_align(rounded_bytes, page_size)
            .unwrap_or_else(|_| panic_on_bad_layout(rounded_bytes, page_size));
        // SAFETY: `layout`'s size is nonzero — `.max(page_size)` above
        // guarantees at least one page even when `min_elements` is 0.
        let raw = unsafe { alloc_zeroed(layout) };
        let Some(ptr) = NonNull::new(raw.cast::<f32>()) else {
            handle_alloc_error(layout);
        };
        Self {
            ptr,
            len: rounded_bytes / size_of::<f32>(),
            layout,
        }
    }

    /// The full usable element count — always `>= min_elements`, and always
    /// a multiple of `page_size / size_of::<f32>()`.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// `true` only when `page_size` was `0`, which [`AlignedBuffer::new`]
    /// never itself produces (`.max(page_size)` keeps `len` positive
    /// whenever `page_size` is), so this is here only for API symmetry —
    /// a real caller has no `page_size` for which this returns `true`.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Deref for AlignedBuffer {
    type Target = [f32];

    fn deref(&self) -> &[f32] {
        // SAFETY: `ptr` is valid for `len` initialized `f32`s — allocated
        // zeroed and never partially freed — for the lifetime of `self`.
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl DerefMut for AlignedBuffer {
    fn deref_mut(&mut self) -> &mut [f32] {
        // SAFETY: same as `deref`, with exclusive access via `&mut self`.
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        // SAFETY: `self.layout` is exactly the `Layout` this buffer's
        // allocation was made with, and this is the only place it is freed.
        unsafe { dealloc(self.ptr.as_ptr().cast::<u8>(), self.layout) };
    }
}

#[cold]
fn panic_on_overflow(min_elements: usize) -> ! {
    panic!("requested element count {min_elements} overflows a byte length in usize")
}

#[cold]
fn panic_on_bad_layout(rounded_bytes: usize, page_size: usize) -> ! {
    panic!("page_size {page_size} is not a valid allocation alignment for {rounded_bytes} bytes")
}

#[cfg(test)]
mod tests {
    use super::AlignedBuffer;

    const PAGE: usize = 16384;

    #[test]
    fn rounds_up_to_a_page_multiple() {
        let buffer = AlignedBuffer::new(1, PAGE);
        assert_eq!(buffer.len(), PAGE / core::mem::size_of::<f32>());
    }

    #[test]
    fn exact_page_multiple_request_is_unchanged() {
        let elements = PAGE / core::mem::size_of::<f32>();
        let buffer = AlignedBuffer::new(elements, PAGE);
        assert_eq!(buffer.len(), elements);
    }

    #[test]
    fn pointer_is_page_aligned() {
        let buffer = AlignedBuffer::new(4096, PAGE);
        let address = buffer.as_ptr() as usize;
        assert_eq!(address % PAGE, 0, "pointer {address:#x} is not page-aligned");
    }

    #[test]
    fn contents_start_zeroed_and_are_writable() {
        let mut buffer = AlignedBuffer::new(8, PAGE);
        assert!(buffer.iter().all(|&value| value == 0.0));
        buffer[0] = 1.5;
        buffer[7] = 2.5;
        assert_eq!(buffer[0], 1.5);
        assert_eq!(buffer[7], 2.5);
    }

    #[test]
    fn zero_elements_still_allocates_one_page() {
        let buffer = AlignedBuffer::new(0, PAGE);
        assert_eq!(buffer.len(), PAGE / core::mem::size_of::<f32>());
    }
}
