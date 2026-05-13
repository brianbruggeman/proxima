//! `BufferPool`/`PooledBuf` — per-worker byte-buffer reservoir. Folded in
//! from the former `proxima-io` satellite crate: BufferPool leans on
//! `thread_local::ThreadLocal` for its per-worker routing, which has no
//! no_std/alloc analog (it resolves OS thread identity), so the whole
//! pool is std-only; alloc-only builds carry no public API here.

#[cfg(feature = "std")]
use alloc::sync::Arc;

#[cfg(feature = "std")]
use bytes::BytesMut;
#[cfg(feature = "std")]
use crossbeam_queue::ArrayQueue;
#[cfg(feature = "std")]
use thread_local::ThreadLocal;

#[cfg(feature = "std")]
pub const DEFAULT_POOL_PER_WORKER: usize = 256;
#[cfg(feature = "std")]
pub const DEFAULT_BUFFER_BYTES: usize = 16 * 1024;

/// Per-worker `BytesMut` reservoir backed by a thread-local
/// `ArrayQueue`. Lock-free in the steady state because each queue is
/// only ever touched by one worker.
#[cfg(feature = "std")]
pub struct BufferPool {
    inner: Arc<BufferPoolInner>,
}

#[cfg(feature = "std")]
struct BufferPoolInner {
    per_worker: ThreadLocal<ArrayQueue<BytesMut>>,
    capacity_per_worker: usize,
    buffer_bytes: usize,
}

#[cfg(feature = "std")]
impl BufferPool {
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(DEFAULT_POOL_PER_WORKER, DEFAULT_BUFFER_BYTES)
    }

    #[must_use]
    pub fn with_config(per_worker_capacity: usize, buffer_bytes: usize) -> Self {
        Self {
            inner: Arc::new(BufferPoolInner {
                per_worker: ThreadLocal::new(),
                capacity_per_worker: per_worker_capacity.max(1),
                buffer_bytes: buffer_bytes.max(1),
            }),
        }
    }

    pub fn acquire(&self) -> PooledBuf {
        let queue = self
            .inner
            .per_worker
            .get_or(|| ArrayQueue::new(self.inner.capacity_per_worker));
        let buffer = queue
            .pop()
            .map(|mut buf| {
                buf.clear();
                buf
            })
            .unwrap_or_else(|| BytesMut::with_capacity(self.inner.buffer_bytes));
        PooledBuf {
            inner: buffer,
            pool: self.inner.clone(),
            return_on_drop: true,
        }
    }

    #[must_use]
    pub fn buffer_bytes(&self) -> usize {
        self.inner.buffer_bytes
    }

    #[must_use]
    pub fn local_inventory(&self) -> usize {
        match self.inner.per_worker.get() {
            Some(queue) => queue.len(),
            None => 0,
        }
    }
}

#[cfg(feature = "std")]
impl Default for BufferPool {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "std")]
impl Clone for BufferPool {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

#[cfg(feature = "std")]
pub struct PooledBuf {
    inner: BytesMut,
    pool: Arc<BufferPoolInner>,
    return_on_drop: bool,
}

#[cfg(feature = "std")]
impl PooledBuf {
    /// Borrow the buffer mutably. The buffer is cleared before being handed
    /// out, so the caller starts with `len() == 0` and the prior allocation.
    pub fn buffer_mut(&mut self) -> &mut BytesMut {
        &mut self.inner
    }

    /// Take ownership of the inner `BytesMut`. After `take`, the `PooledBuf`
    /// is dropped without returning anything to the pool.
    pub fn take(mut self) -> BytesMut {
        self.return_on_drop = false;
        core::mem::take(&mut self.inner)
    }
}

#[cfg(feature = "std")]
impl Drop for PooledBuf {
    fn drop(&mut self) {
        if !self.return_on_drop {
            return;
        }
        let buffer = core::mem::take(&mut self.inner);
        let queue = self
            .pool
            .per_worker
            .get_or(|| ArrayQueue::new(self.pool.capacity_per_worker));
        // push fails (returns Err with the buffer) when the queue is at capacity;
        // dropping the returned buffer frees it through normal allocator path.
        let _ = queue.push(buffer);
    }
}

#[cfg(all(test, feature = "std"))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn acquire_returns_buffer_with_zero_length() {
        let pool = BufferPool::new();
        let buf = pool.acquire();
        assert_eq!(buf.inner.len(), 0);
    }

    #[test]
    fn returned_buffer_is_reused_by_next_acquire_on_same_thread() {
        let pool = BufferPool::with_config(4, 64);
        let mut first = pool.acquire();
        first.buffer_mut().extend_from_slice(b"abc");
        let initial_capacity = first.inner.capacity();
        drop(first);
        assert_eq!(pool.local_inventory(), 1);
        let second = pool.acquire();
        assert_eq!(
            second.inner.capacity(),
            initial_capacity,
            "returned buffer must be reused with its allocated capacity",
        );
        assert_eq!(pool.local_inventory(), 0);
    }

    #[test]
    fn capacity_overflow_drops_extra_buffers() {
        let pool = BufferPool::with_config(2, 64);
        let one = pool.acquire();
        let two = pool.acquire();
        let three = pool.acquire();
        drop(one);
        drop(two);
        drop(three);
        assert!(
            pool.local_inventory() <= 2,
            "local queue must respect its capacity"
        );
    }

    #[test]
    fn take_consumes_buffer_without_returning_to_pool() {
        let pool = BufferPool::with_config(4, 64);
        let buf = pool.acquire();
        let _owned = buf.take();
        assert_eq!(pool.local_inventory(), 0);
    }
}
