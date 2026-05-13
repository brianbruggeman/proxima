//! Std-host implementation of [`ThreadIdentity`].
//!
//! Each thread reads a `std::thread_local!` cell on first call. The cell is
//! initialized from a process-global `AtomicU64` counter (starts at 1, so
//! `0` is reserved as a "not-set" sentinel that consumers can use).
//!
//! Cost: one TLS read per `current()` call (LLVM lowers `with(|id| *id)` to
//! `mov` on Linux/macOS x86_64 and a register-load on aarch64). Monomorphizes
//! through the trait so callers pay no dispatch overhead.

use core::sync::atomic::{AtomicU64, Ordering};

use crate::core::thread_identity::ThreadIdentity;

static NEXT_THREAD_ID: AtomicU64 = AtomicU64::new(1);

std::thread_local! {
    static THREAD_ID: u64 = NEXT_THREAD_ID.fetch_add(1, Ordering::Relaxed);
}

/// Std-host `ThreadIdentity` backed by `std::thread_local!`.
pub struct StdThreadIdentity;

impl ThreadIdentity for StdThreadIdentity {
    type Id = u64;

    #[inline]
    fn current() -> Self::Id {
        THREAD_ID.with(|id| *id)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64 as StdAtomicU64, Ordering as StdOrdering};
    use std::sync::mpsc;
    use std::thread;

    #[test]
    fn std_impl_current_is_stable_within_thread() {
        let id_a = StdThreadIdentity::current();
        let id_b = StdThreadIdentity::current();
        assert_eq!(id_a, id_b);
    }

    #[test]
    fn std_impl_different_threads_get_different_ids() {
        let main_id = StdThreadIdentity::current();
        let (tx, rx) = mpsc::channel();
        let handle = thread::spawn(move || {
            tx.send(StdThreadIdentity::current()).expect("send");
        });
        let other_id = rx.recv().expect("recv");
        handle.join().expect("join");
        assert_ne!(main_id, other_id);
    }

    #[test]
    fn std_impl_is_owning_returns_true_for_current_thread() {
        let captured = StdThreadIdentity::current();
        assert!(StdThreadIdentity::is_owning(captured));
    }

    #[test]
    fn std_impl_is_owning_returns_false_for_other_thread() {
        let (tx, rx) = mpsc::channel();
        let handle = thread::spawn(move || {
            tx.send(StdThreadIdentity::current()).expect("send");
        });
        let other_id = rx.recv().expect("recv");
        handle.join().expect("join");
        assert!(!StdThreadIdentity::is_owning(other_id));
    }

    #[test]
    fn std_impl_ids_are_unique_across_many_threads() {
        const THREAD_COUNT: usize = 16;
        let ids: Arc<std::sync::Mutex<HashSet<u64>>> =
            Arc::new(std::sync::Mutex::new(HashSet::new()));
        let mut handles = Vec::with_capacity(THREAD_COUNT);
        for _ in 0..THREAD_COUNT {
            let ids = ids.clone();
            handles.push(thread::spawn(move || {
                let id = StdThreadIdentity::current();
                ids.lock().expect("lock").insert(id);
            }));
        }
        for handle in handles {
            handle.join().expect("join");
        }
        let collected = ids.lock().expect("lock");
        assert_eq!(
            collected.len(),
            THREAD_COUNT,
            "all thread ids should be unique"
        );
    }

    #[test]
    fn std_impl_trait_route_matches_direct_tls_read() {
        let direct = THREAD_ID.with(|id| *id);
        let via_trait = StdThreadIdentity::current();
        assert_eq!(
            direct, via_trait,
            "trait route must read the same TLS cell as the direct path"
        );
    }

    #[test]
    fn std_impl_concurrent_reads_are_stable_per_thread() {
        const ITERATIONS: usize = 1000;
        let mismatches = Arc::new(StdAtomicU64::new(0));
        let mut handles = Vec::with_capacity(4);
        for _ in 0..4 {
            let mismatches = mismatches.clone();
            handles.push(thread::spawn(move || {
                let first = StdThreadIdentity::current();
                for _ in 0..ITERATIONS {
                    if StdThreadIdentity::current() != first {
                        mismatches.fetch_add(1, StdOrdering::Relaxed);
                    }
                }
            }));
        }
        for handle in handles {
            handle.join().expect("join");
        }
        assert_eq!(mismatches.load(StdOrdering::Relaxed), 0);
    }
}
