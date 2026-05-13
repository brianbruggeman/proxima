//! Reused byte arena — append bytes, reference them by `(offset, len)`,
//! read them back without copying. One backing allocation grown
//! power-of-two; the store-once / reference-many shape (mirrors
//! a consumer crate's `ByteBuffer`/`HeapBuffer`, but single-writer so no
//! atomics). The consumer owns the reclamation policy: [`reset`] drops
//! everything at once, which is the right primitive for a bounded
//! in-flight window that fully drains (e.g. a per-connection retransmit
//! buffer between quiescent points). A ring / free-list layer can sit on
//! top when out-of-order partial reclaim is needed.
//!
//! [`reset`]: ByteArena::reset

use alloc::vec::Vec;

/// Append-only byte store addressed by `u32` offset. `data.len()` is the
/// (zero-filled) capacity; `cursor` is the live high-water mark — bytes
/// `[0, cursor)` are the appended payloads.
#[derive(Debug, Default, Clone)]
pub struct ByteArena {
    data: Vec<u8>,
    cursor: usize,
}

impl ByteArena {
    /// Empty arena, no allocation until the first append.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            data: Vec::new(),
            cursor: 0,
        }
    }

    /// Empty arena pre-grown to `capacity` so the first appends don't
    /// re-grow.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: alloc::vec![0; capacity],
            cursor: 0,
        }
    }

    /// Append `bytes`, returning the offset they were written at. Grows
    /// the backing power-of-two when the bump would overflow capacity.
    ///
    /// # Panics
    ///
    /// Panics if the resulting high-water mark exceeds `u32::MAX` — the
    /// address width of the returned offset. Callers with a bounded
    /// window (in-flight bytes ≤ congestion window) never reach this.
    pub fn append(&mut self, bytes: &[u8]) -> u32 {
        let offset = self.cursor;
        let end = offset + bytes.len();
        assert!(
            end <= u32::MAX as usize,
            "byte arena exceeded u32 offset space"
        );
        if end > self.data.len() {
            self.data.resize(end.next_power_of_two(), 0);
        }
        self.data[offset..end].copy_from_slice(bytes);
        self.cursor = end;
        offset as u32
    }

    /// Borrow the `len` bytes written at `offset`.
    #[must_use]
    pub fn read(&self, offset: u32, len: u32) -> &[u8] {
        let start = offset as usize;
        &self.data[start..start + len as usize]
    }

    /// Reclaim everything: the next append starts at offset 0 again. The
    /// backing allocation is retained for reuse. Sound only when no live
    /// `(offset, len)` reference into the arena outlives the reset.
    pub fn reset(&mut self) {
        self.cursor = 0;
    }

    /// Live bytes appended since the last [`reset`](Self::reset).
    #[must_use]
    pub fn bytes_used(&self) -> usize {
        self.cursor
    }

    /// Retained backing capacity in bytes.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.data.len()
    }

    /// True when nothing has been appended since the last reset.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cursor == 0
    }
}

#[cfg(test)]
mod tests {
    use super::ByteArena;

    #[test]
    fn append_returns_offsets_and_reads_back_each_payload() {
        let mut arena = ByteArena::new();
        let first = arena.append(b"hello");
        let second = arena.append(b"world!");
        assert_eq!(first, 0);
        assert_eq!(second, 5);
        assert_eq!(arena.read(first, 5), b"hello");
        assert_eq!(arena.read(second, 6), b"world!");
        assert_eq!(arena.bytes_used(), 11);
    }

    #[test]
    fn grows_power_of_two_and_preserves_prior_payloads() {
        let mut arena = ByteArena::with_capacity(4);
        let a = arena.append(b"ab");
        let b = arena.append(b"cdef"); // overflows the initial 4, forces grow
        assert_eq!(arena.read(a, 2), b"ab");
        assert_eq!(arena.read(b, 4), b"cdef");
        assert!(arena.capacity().is_power_of_two());
        assert!(arena.capacity() >= 6);
    }

    #[test]
    fn reset_rewinds_cursor_and_keeps_capacity() {
        let mut arena = ByteArena::new();
        arena.append(b"some bytes");
        let cap = arena.capacity();
        arena.reset();
        assert!(arena.is_empty());
        assert_eq!(arena.bytes_used(), 0);
        assert_eq!(arena.capacity(), cap, "reset retains the allocation");
        // a fresh append reuses offset 0
        assert_eq!(arena.append(b"x"), 0);
    }
}
