//! Rayon-backed `BackgroundPool` impl.
//!
//! Use case: Pipes that need cross-thread CPU-bound work — image
//! decoding, model inference, parallel parsing — without saturating
//! tokio's blocking-thread pool (which is sized for I/O-blocking
//! work and can grow to 512 threads). Rayon's pool is work-stealing
//! across a fixed thread count (default = num_cpus), the right
//! shape for fork-join compute.
//!
//! Plug into `TokioPerCoreRuntime` via `.with_background_pool(...)`.
//!
//! changelog:
//! - v1: dyn-only BackgroundPool trait impl (rayon feature gate)
//! - v2: typed spawn<F, T> fast-path (runtime-prime-bgpool-rayon gate);
//!   mirrors ProximaBackgroundPool API exactly so callers can swap

use alloc::boxed::Box;
use alloc::format;
use alloc::sync::Arc;

use core::future::Future;

use futures::channel::oneshot;
use rayon::ThreadPool;

use crate::{BackgroundHandle, BackgroundPool};
use proxima_core::ProximaError;

pub struct RayonBackgroundPool {
    pool: Arc<ThreadPool>,
}

impl RayonBackgroundPool {
    /// Build a pool with rayon's default thread count (= num_cpus).
    pub fn new() -> Result<Self, ProximaError> {
        let pool = rayon::ThreadPoolBuilder::new()
            .thread_name(|index| format!("proxima-rayon-bg-{index}"))
            .build()
            .map_err(|error| ProximaError::Config(format!("build rayon pool: {error}")))?;
        Ok(Self {
            pool: Arc::new(pool),
        })
    }

    /// Build a pool with `threads` workers.
    pub fn with_threads(threads: usize) -> Result<Self, ProximaError> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|index| format!("proxima-rayon-bg-{index}"))
            .build()
            .map_err(|error| ProximaError::Config(format!("build rayon pool: {error}")))?;
        Ok(Self {
            pool: Arc::new(pool),
        })
    }

    /// Type-specialized fast-path spawn. Mirrors `ProximaBackgroundPool::spawn<F, T>`:
    /// no API-level `Box<dyn FnOnce>` — the closure is pushed directly into
    /// rayon's work-stealing deque. Available whenever the `rayon` feature
    /// is on; callers holding the concrete `RayonBackgroundPool` get the
    /// no-alloc path.
    pub fn spawn<F, T>(
        &self,
        work: F,
    ) -> impl Future<Output = Result<T, ProximaError>> + Send + 'static
    where
        F: FnOnce() -> Result<T, ProximaError> + Send + 'static,
        T: Send + 'static,
    {
        let (tx, rx) = oneshot::channel();
        self.pool.spawn(move || {
            let result = work();
            let _ = tx.send(result);
        });
        async move {
            rx.await.unwrap_or_else(|_| {
                Err(ProximaError::Body(
                    "rayon background task dropped sender".into(),
                ))
            })
        }
    }
}

impl BackgroundPool for RayonBackgroundPool {
    fn spawn(
        &self,
        work: Box<dyn FnOnce() -> Result<Box<dyn core::any::Any + Send>, ProximaError> + Send>,
    ) -> BackgroundHandle<Box<dyn core::any::Any + Send>> {
        let (tx, rx) = oneshot::channel();
        self.pool.spawn(move || {
            let result = work();
            let _ = tx.send(result);
        });
        Box::pin(async move {
            rx.await.unwrap_or_else(|_| {
                Err(ProximaError::Body(
                    "rayon background task dropped sender".into(),
                ))
            })
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[proxima::test]
    async fn rayon_pool_runs_work_and_returns_result() {
        let pool = Arc::new(RayonBackgroundPool::new().expect("build pool"));
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_for_work = counter.clone();
        let handle = pool.spawn(Box::new(move || {
            counter_for_work.fetch_add(1, Ordering::SeqCst);
            Ok(Box::new(42_u32) as Box<dyn core::any::Any + Send>)
        }));
        let result = handle.await.expect("background result");
        let value = result.downcast::<u32>().expect("downcast");
        assert_eq!(*value, 42);
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[proxima::test]
    async fn rayon_pool_runs_many_tasks_in_parallel() {
        let pool = Arc::new(RayonBackgroundPool::with_threads(4).expect("build pool"));
        let mut handles = Vec::new();
        for index in 0..8_u32 {
            handles.push(pool.spawn(Box::new(move || {
                Ok(Box::new(index * 2) as Box<dyn core::any::Any + Send>)
            })));
        }
        let mut results: Vec<u32> = Vec::new();
        for handle in handles {
            let value = handle.await.expect("result");
            results.push(*value.downcast::<u32>().expect("downcast"));
        }
        results.sort_unstable();
        let expected: Vec<u32> = (0..8).map(|index| index * 2).collect();
        assert_eq!(results, expected);
    }
}
