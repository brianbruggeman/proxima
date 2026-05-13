//! Three-state futex mutex for the no_std+OS tier, fronted by `lock_api` so it
//! exposes the identical `Mutex<T>` surface as the std (parking_lot) tier.
//!
//! State encoding (the canonical futex-mutex protocol): `0` unlocked, `1`
//! locked with no waiters, `2` locked with at least one waiter. The `2` state
//! is what tells `unlock` a wake syscall is actually needed, so the fast
//! uncontended path pays no syscall.

use core::hint;
use core::sync::atomic::{AtomicU32, Ordering};

use lock_api::{GuardSend, RawMutex as RawMutexTrait};

pub type Mutex<T> = lock_api::Mutex<RawFutexMutex, T>;
pub type MutexGuard<'guard, T> = lock_api::MutexGuard<'guard, RawFutexMutex, T>;

pub struct RawFutexMutex {
    state: AtomicU32,
}

unsafe impl RawMutexTrait for RawFutexMutex {
    // interior mutability is the whole point of a lock's INIT constant.
    #[allow(clippy::declare_interior_mutable_const)]
    const INIT: Self = Self {
        state: AtomicU32::new(0),
    };

    type GuardMarker = GuardSend;

    fn lock(&self) {
        if self
            .state
            .compare_exchange(0, 1, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            self.lock_contended();
        }
    }

    fn try_lock(&self) -> bool {
        self.state
            .compare_exchange(0, 1, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
    }

    unsafe fn unlock(&self) {
        // swap to 0; only if a waiter was recorded (state == 2) do we pay a wake.
        if self.state.swap(0, Ordering::Release) == 2 {
            atomic_wait::wake_one(&self.state);
        }
    }
}

impl RawFutexMutex {
    #[cold]
    fn lock_contended(&self) {
        // brief adaptive spin absorbs short critical sections without a syscall.
        let mut spins = 0u32;
        while self.state.load(Ordering::Relaxed) == 1 && spins < 100 {
            spins += 1;
            hint::spin_loop();
        }
        if self
            .state
            .compare_exchange(0, 1, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
        {
            return;
        }
        // record contention (state -> 2) and park until unlock wakes us; the
        // swap-returns-nonzero loop re-checks after every (possibly spurious) wake.
        while self.state.swap(2, Ordering::Acquire) != 0 {
            atomic_wait::wait(&self.state, 2);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::thread;

    use super::Mutex;

    // contended counter: N threads each add M under the lock; a lost wakeup or a
    // torn critical section shows up as a final sum below N*M.
    #[test]
    fn contended_counter_sums_exactly() {
        const THREADS: usize = 8;
        const ADDS: usize = 50_000;

        let counter = Arc::new(Mutex::new(0usize));
        let handles: std::vec::Vec<_> = (0..THREADS)
            .map(|_| {
                let counter = Arc::clone(&counter);
                thread::spawn(move || {
                    for _ in 0..ADDS {
                        *counter.lock() += 1;
                    }
                })
            })
            .collect();
        for handle in handles {
            handle.join().expect("worker thread panicked");
        }

        assert_eq!(*counter.lock(), THREADS * ADDS);
    }

    #[test]
    fn try_lock_excludes_while_held() {
        let mutex = Mutex::new(7u32);
        let guard = mutex.lock();
        assert!(mutex.try_lock().is_none());
        drop(guard);
        assert!(mutex.try_lock().is_some());
    }
}
