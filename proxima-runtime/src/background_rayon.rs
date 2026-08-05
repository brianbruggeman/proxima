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
    use std::any::Any;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[proxima::test]
    async fn typed_spawn_hands_back_the_concrete_type() {
        let pool = RayonBackgroundPool::new().expect("build pool");
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_for_work = Arc::clone(&counter);
        let value = pool
            .spawn(move || {
                counter_for_work.fetch_add(1, Ordering::SeqCst);
                Ok(42_u32)
            })
            .await
            .expect("background result");
        assert_eq!(value, 42);
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[proxima::test]
    async fn typed_spawn_runs_many_jobs_across_the_pool() {
        let pool = RayonBackgroundPool::with_threads(4).expect("build pool");
        let handles: Vec<_> = (0..8_u32)
            .map(|index| pool.spawn(move || Ok(index * 2)))
            .collect();
        let mut results: Vec<u32> = Vec::new();
        for handle in handles {
            results.push(handle.await.expect("result"));
        }
        results.sort_unstable();
        let expected: Vec<u32> = (0..8).map(|index| index * 2).collect();
        assert_eq!(results, expected);
    }

    /// The erased form `TokioPerCoreRuntime::with_background_pool` stores. Worth
    /// its own case because the two tests above resolve to the INHERENT
    /// `spawn<F, T>` — an inherent method shadows the trait one — so deleting
    /// `impl BackgroundPool for RayonBackgroundPool` left them compiling.
    #[proxima::test]
    async fn erased_pool_round_trips_a_payload_through_the_trait() {
        let pool: Arc<dyn BackgroundPool> =
            Arc::new(RayonBackgroundPool::with_threads(2).expect("build pool"));
        let handle = pool.spawn(Box::new(|| {
            Ok(Box::new(alloc::string::String::from("hello")) as Box<dyn Any + Send>)
        }));
        let boxed = handle.await.expect("background result");
        let value = boxed.downcast::<alloc::string::String>().expect("downcast");
        assert_eq!(*value, "hello");
    }

    /// A job that fails must surface its own error, not the channel's — the arm
    /// that distinguishes `Ok(Err(job))` from a dropped sender.
    #[proxima::test]
    async fn erased_pool_surfaces_the_jobs_own_error() {
        let pool: Arc<dyn BackgroundPool> =
            Arc::new(RayonBackgroundPool::new().expect("build pool"));
        let handle = pool.spawn(Box::new(|| Err(ProximaError::Body("job said no".into()))));
        let error = handle.await.expect_err("job failed");
        assert!(
            error.to_string().contains("job said no"),
            "job error, not a transport error; got {error}"
        );
    }
}
