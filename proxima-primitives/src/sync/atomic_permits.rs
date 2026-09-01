//! Lock-free counted permits for bounded synchronous admission.
//!
//! [`AtomicPermitPool`] is intentionally smaller than [`super::Semaphore`]:
//! it has no wait queue, close operation, or async acquire. It is the right
//! shape for a caller that only needs `try_acquire` and releases by dropping
//! the returned permit. It is a counted gate, not a [`crate::pipe::Pipe`].

use core::sync::atomic::{AtomicUsize, Ordering};

/// A fixed-capacity pool whose permits are acquired and released without a
/// lock or allocation.
#[derive(Debug)]
pub struct AtomicPermitPool {
    available: AtomicUsize,
}

impl AtomicPermitPool {
    /// Create a pool with `capacity` available permits.
    #[must_use]
    pub const fn new(capacity: usize) -> Self {
        Self {
            available: AtomicUsize::new(capacity),
        }
    }

    /// Attempt to acquire one permit without waiting.
    pub fn try_acquire(&self) -> Option<AtomicPermit<'_>> {
        let mut current = self.available.load(Ordering::Acquire);
        loop {
            if current == 0 {
                return None;
            }
            match self.available.compare_exchange_weak(
                current,
                current - 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Some(AtomicPermit { owner: self }),
                Err(next) => current = next,
            }
        }
    }

    /// Return a snapshot of currently available permits.
    #[must_use]
    pub fn available_permits(&self) -> usize {
        self.available.load(Ordering::Acquire)
    }
}

/// A permit held from an [`AtomicPermitPool`]. Dropping it returns capacity.
#[derive(Debug)]
pub struct AtomicPermit<'a> {
    owner: &'a AtomicPermitPool,
}

impl Drop for AtomicPermit<'_> {
    fn drop(&mut self) {
        self.owner.available.fetch_add(1, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acquire_exhausts_and_drop_releases() {
        let pool = AtomicPermitPool::new(2);
        let first = pool.try_acquire().expect("first permit");
        let second = pool.try_acquire().expect("second permit");
        assert_eq!(pool.available_permits(), 0);
        assert!(pool.try_acquire().is_none());
        drop(first);
        assert_eq!(pool.available_permits(), 1);
        drop(second);
        assert_eq!(pool.available_permits(), 2);
    }

    #[test]
    fn zero_capacity_always_sheds() {
        let pool = AtomicPermitPool::new(0);
        assert!(pool.try_acquire().is_none());
        assert_eq!(pool.available_permits(), 0);
    }

    #[test]
    fn permit_is_raii_scoped() {
        let pool = AtomicPermitPool::new(1);
        {
            let _permit = pool.try_acquire().expect("permit");
            assert_eq!(pool.available_permits(), 0);
        }
        assert_eq!(pool.available_permits(), 1);
    }

    #[test]
    fn concurrent_acquisition_never_exceeds_capacity() {
        let pool = std::sync::Arc::new(AtomicPermitPool::new(4));
        let active = std::sync::Arc::new(AtomicUsize::new(0));
        let exceeded = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let mut threads = Vec::new();
        for _ in 0..8 {
            let pool = std::sync::Arc::clone(&pool);
            let active = std::sync::Arc::clone(&active);
            let exceeded = std::sync::Arc::clone(&exceeded);
            threads.push(std::thread::spawn(move || {
                let Some(_permit) = pool.try_acquire() else {
                    return;
                };
                let held = active.fetch_add(1, Ordering::AcqRel) + 1;
                if held > 4 {
                    exceeded.store(true, Ordering::Release);
                }
                std::thread::yield_now();
                active.fetch_sub(1, Ordering::AcqRel);
            }));
        }
        for thread in threads {
            thread.join().expect("thread");
        }
        assert!(!exceeded.load(Ordering::Acquire));
        assert_eq!(pool.available_permits(), 4);
    }
}
