//! stdlib + crossbeam-only `BackgroundPool` implementation. workers steal
//! from a shared `crossbeam_deque::Injector<Job>` (the same primitive rayon
//! uses for external job injection). workers park on `crossbeam_utils::sync::Parker`
//! when the injector is empty — atomic-only notify, no mutex lock on the
//! producer's notify path. results returned via `futures::channel::oneshot`.
//!
//! changelog:
//! - v1: SegQueue + Condvar (baseline)
//! - v2: notify-elision via `parked_count: AtomicUsize` (+41% over v1)
//! - v3: tried per-worker queues + work-stealing — rolled back, regressed
//!   under single-producer load
//! - v4: Injector-based dispatch + typed `spawn<F, T>` method on the
//!   concrete type to skip the trait's `Box<dyn FnOnce>` allocation at the
//!   API surface
//! - v5: `Job` is now an enum so the dyn-trait spawn doesn't allocate a
//!   wrapper Box around the caller's already-boxed work (paired with the
//!   sender directly in `Job::Dyn`); switched parking from Condvar+Mutex
//!   to round-robin `crossbeam_utils::sync::Parker` (atomic-only notify,
//!   lower variance on the dyn dispatch path)
//!
//! purpose: replace the `rayon` dependency for proxima's cross-thread CPU
//! work, AND beat rayon on the bench. `BackgroundPool` trait stays
//! object-safe (slow path via `spawn_boxed`); concrete callers use
//! `spawn<F, T>` to skip API-level boxing.

#![cfg(feature = "runtime-prime-bgpool")]

use std::any::Any;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::task::{Context, Poll};
use std::thread;

use crossbeam_deque::{Injector, Steal};
use crossbeam_utils::sync::{Parker, Unparker};
use futures::channel::oneshot;

use proxima_core::ProximaError;
use proxima_runtime::{BackgroundHandle, BackgroundPool};

/// queue slot. variants paired with the spawn entry point that produces
/// them so the dyn-trait spawn doesn't allocate a wrapper Box just to
/// pair the caller's `work` with our `sender`: the caller's outer Box
/// is already on hand, store it directly with the oneshot sender.
enum Job {
    /// produced by the typed fast-path (`pub fn spawn<F, T>`): the work
    /// closure already captures its own sender, so just run it.
    Plain(Box<dyn FnOnce() + Send + 'static>),
    /// produced by the object-safe trait method: caller already provided
    /// the boxed work; pair it with the sender here, no extra wrap.
    Dyn {
        work: Box<dyn FnOnce() -> Result<Box<dyn Any + Send>, ProximaError> + Send>,
        sender: oneshot::Sender<Result<Box<dyn Any + Send>, ProximaError>>,
    },
    /// produced by `spawn_async`: an already-pinned future whose body
    /// captures its own sender. workers `block_on` this on their per-thread
    /// tokio current_thread runtime.
    #[cfg(feature = "runtime-prime-bgpool-async")]
    Async(Pin<Box<dyn Future<Output = ()> + Send + 'static>>),
}

struct Inner {
    injector: Injector<Job>,
    /// one unparker per worker, indexed by worker id. producers fire an
    /// unpark on a round-robin worker when `parked_count > 0`. unparking
    /// a non-parked worker is harmless — the token is consumed by the
    /// next park attempt. zero mutex lock on the notify path.
    unparkers: Box<[Unparker]>,
    shutdown: AtomicBool,
    /// number of workers currently in the parked region. producers elide
    /// unpark entirely when zero — common case once workers are running
    /// continuously.
    parked_count: AtomicUsize,
    /// round-robin cursor over `unparkers` for `maybe_notify`. relaxed
    /// reads/writes — biasing is fine.
    notify_cursor: AtomicUsize,
}

pub struct ProximaBackgroundPool {
    inner: Arc<Inner>,
    handles: Vec<Option<thread::JoinHandle<()>>>,
}

impl ProximaBackgroundPool {
    /// build a pool with `num_cpus::get()` workers (or 1 if num_cpus is 0).
    pub fn new() -> Result<Self, ProximaError> {
        Self::with_threads(num_cpus::get().max(1))
    }

    /// worker count. usable for caller-side scheduling decisions (e.g.
    /// the par module derives a default chunk threshold from this so
    /// leaves-per-worker stays in the low-variance zone of the U-curve).
    #[must_use = "worker count is informational; act on it or discard"]
    pub fn workers(&self) -> usize {
        self.inner.unparkers.len()
    }

    /// build a pool with exactly `threads` workers.
    pub fn with_threads(threads: usize) -> Result<Self, ProximaError> {
        let threads = threads.max(1);
        let mut parkers: Vec<Parker> = Vec::with_capacity(threads);
        let mut unparkers: Vec<Unparker> = Vec::with_capacity(threads);
        for _ in 0..threads {
            let parker = Parker::new();
            unparkers.push(parker.unparker().clone());
            parkers.push(parker);
        }
        let inner = Arc::new(Inner {
            injector: Injector::new(),
            unparkers: unparkers.into_boxed_slice(),
            shutdown: AtomicBool::new(false),
            parked_count: AtomicUsize::new(0),
            notify_cursor: AtomicUsize::new(0),
        });
        let mut handles = Vec::with_capacity(threads);
        for (index, parker) in parkers.into_iter().enumerate() {
            let inner = inner.clone();
            let handle = thread::Builder::new()
                .name(format!("proxima-bg-{index}"))
                .spawn(move || worker(inner, parker))
                .map_err(|err| ProximaError::Config(format!("spawn proxima bg worker: {err}")))?;
            handles.push(Some(handle));
        }
        Ok(Self { inner, handles })
    }

    /// type-specialized fast-path spawn. takes `F` directly (no API-level
    /// `Box<dyn FnOnce>`) and returns a typed Future for the result. the
    /// internal job queue still holds erased `Box<dyn FnOnce>` (the pool is
    /// heterogeneous), but callers that hold a concrete `&ProximaBackgroundPool`
    /// skip one boxing vs the trait method's signature.
    ///
    /// returns `impl Future` (an opaque async-block state machine) rather
    /// than the named `JobFuture<T>` adapter — bench showed async-block is
    /// faster on the typed path (the compiler has full visibility to
    /// optimize the state machine when the return type isn't boxed).
    /// `JobFuture` is used only on the dyn-trait path where the return is
    /// `Pin<Box<dyn Future>>` and the optimizer can't see through anyway.
    pub fn spawn<F, T>(
        &self,
        work: F,
    ) -> impl Future<Output = Result<T, ProximaError>> + Send + 'static
    where
        F: FnOnce() -> Result<T, ProximaError> + Send + 'static,
        T: Send + 'static,
    {
        let (sender, receiver) = oneshot::channel();
        let job = Job::Plain(Box::new(move || {
            let result = work();
            let _ = sender.send(result);
        }));
        self.inner.injector.push(job);
        self.maybe_notify();
        async move {
            receiver.await.unwrap_or_else(|_| {
                Err(ProximaError::Body(
                    "proxima background task dropped sender".into(),
                ))
            })
        }
    }

    /// typed async spawn. takes any `Future<Output = Result<T, ProximaError>>`
    /// and returns a Future that completes when the inner future is
    /// driven to completion by a worker's per-thread tokio current_thread
    /// runtime. structurally mirrors [`Self::spawn`] (which is sync) but
    /// the leaf runs on a polling worker instead of being executed inline.
    ///
    /// only available when the `runtime-prime-bgpool-async` feature is on;
    /// without it, workers are sync-only and never construct a tokio
    /// runtime (one less allocation per worker).
    #[cfg(feature = "runtime-prime-bgpool-async")]
    pub fn spawn_async<F, T>(
        &self,
        work: F,
    ) -> impl Future<Output = Result<T, ProximaError>> + Send + 'static
    where
        F: Future<Output = Result<T, ProximaError>> + Send + 'static,
        T: Send + 'static,
    {
        let (sender, receiver) = oneshot::channel();
        // wrap the user's future so it sends its own result on completion;
        // the worker only sees `Future<Output = ()>` and doesn't have to
        // know about `T`. one heap alloc per spawn (the Pin<Box>); the
        // boxed future already exists as a state machine, this just adds
        // the send-on-done shim.
        let job_future: Pin<Box<dyn Future<Output = ()> + Send + 'static>> = Box::pin(async move {
            let result = work.await;
            let _ = sender.send(result);
        });
        self.inner.injector.push(Job::Async(job_future));
        self.maybe_notify();
        async move {
            receiver.await.unwrap_or_else(|_| {
                Err(ProximaError::Body(
                    "proxima background async task dropped sender".into(),
                ))
            })
        }
    }

    fn maybe_notify(&self) {
        if self.inner.parked_count.load(Ordering::Acquire) == 0 {
            return;
        }
        let workers = self.inner.unparkers.len();
        let index = self.inner.notify_cursor.fetch_add(1, Ordering::Relaxed) % workers;
        // unparking a non-parked worker just queues a token consumed by
        // its next park attempt — harmless. for the steady state where
        // workers are all running, the parked_count==0 short-circuit
        // above skips this entirely.
        self.inner.unparkers[index].unpark();
    }
}

impl BackgroundPool for ProximaBackgroundPool {
    fn spawn(
        &self,
        work: Box<dyn FnOnce() -> Result<Box<dyn Any + Send>, ProximaError> + Send>,
    ) -> BackgroundHandle<Box<dyn Any + Send>> {
        // dyn-compatible slow path. caller already paid for one `Box<dyn
        // FnOnce>` (the work itself); we pair it with the sender directly
        // in `Job::Dyn` instead of wrapping in a second Box. the trait
        // return must still be `Pin<Box<dyn Future>>` per the API alias,
        // but the inner is `JobFuture` — a one-field struct that polls the
        // receiver directly (no async-block state machine, no enum tag,
        // no captured state beyond the receiver).
        let (sender, receiver) = oneshot::channel();
        self.inner.injector.push(Job::Dyn { work, sender });
        self.maybe_notify();
        Box::pin(JobFuture { receiver })
    }
}

/// hand-rolled Future for `ProximaBackgroundPool::spawn`. delegates to
/// the inner `oneshot::Receiver` and maps the canceled-sender error to
/// `ProximaError::Body`. structurally identical to what `async move {
/// receiver.await.unwrap_or_else(...) }` compiles to, but without the
/// async-block state machine's enum tag + padding — smaller `Pin<Box>`
/// allocation on the trait return path, fewer state transitions in
/// poll.
pub struct JobFuture<T> {
    receiver: oneshot::Receiver<Result<T, ProximaError>>,
}

impl<T> Future for JobFuture<T> {
    type Output = Result<T, ProximaError>;

    #[inline]
    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        // SAFETY: oneshot::Receiver is Unpin (no self-referential state).
        // we project &mut to the inner receiver without re-pinning.
        let this = unsafe { self.get_unchecked_mut() };
        match Pin::new(&mut this.receiver).poll(context) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(Ok(result)) => Poll::Ready(result),
            Poll::Ready(Err(_)) => Poll::Ready(Err(ProximaError::Body(
                "proxima background task dropped sender".into(),
            ))),
        }
    }
}

impl Drop for ProximaBackgroundPool {
    fn drop(&mut self) {
        self.inner.shutdown.store(true, Ordering::Release);
        // wake every worker so each observes shutdown and exits.
        for unparker in &self.inner.unparkers {
            unparker.unpark();
        }
        for slot in &mut self.handles {
            if let Some(handle) = slot.take() {
                let _ = handle.join();
            }
        }
    }
}

fn worker(inner: Arc<Inner>, parker: Parker) {
    // per-worker tokio current_thread runtime, built only when the async
    // feature is on. drives `Job::Async` futures via `block_on`. each worker
    // owns its own — async state never escapes the worker that started it.
    // a build failure is rare (FD exhaustion or OOM); on failure the worker
    // exits silently and the pool runs with one fewer concurrent slot.
    #[cfg(feature = "runtime-prime-bgpool-async")]
    let async_runtime = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(_) => return,
    };

    loop {
        // fast path: steal from injector. `steal` retries internally on
        // contention; we treat `Empty` as the signal to park.
        match inner.injector.steal() {
            Steal::Success(job) => {
                let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match job {
                    Job::Plain(closure) => closure(),
                    Job::Dyn { work, sender } => {
                        let _ = sender.send(work());
                    }
                    #[cfg(feature = "runtime-prime-bgpool-async")]
                    Job::Async(future) => {
                        async_runtime.block_on(future);
                    }
                }));
                let _ = unwind;
                continue;
            }
            Steal::Retry => continue,
            Steal::Empty => {}
        }
        // park: increment parked_count BEFORE the re-check so a concurrent
        // producer observes `parked_count > 0` and fires its unpark. if
        // the producer pushed between our `Steal::Empty` and the
        // fetch_add, the re-check below catches it without parking.
        inner.parked_count.fetch_add(1, Ordering::AcqRel);
        if !inner.injector.is_empty() {
            inner.parked_count.fetch_sub(1, Ordering::AcqRel);
            continue;
        }
        if inner.shutdown.load(Ordering::Acquire) {
            inner.parked_count.fetch_sub(1, Ordering::AcqRel);
            return;
        }
        // `Parker::park` is edge-triggered: an `unpark` that fired before
        // this call returns immediately. handles the race where a
        // producer's `maybe_notify` raced ahead of our park call.
        parker.park();
        inner.parked_count.fetch_sub(1, Ordering::AcqRel);
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[proxima::test]
    async fn proxima_pool_runs_work_and_returns_result() {
        let pool = Arc::new(ProximaBackgroundPool::new().expect("build pool"));
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_for_work = counter.clone();
        // exercise the typed fast path.
        let handle = pool.spawn(move || {
            counter_for_work.fetch_add(1, Ordering::SeqCst);
            Ok::<u32, ProximaError>(42)
        });
        let value = handle.await.expect("bg result");
        assert_eq!(value, 42);
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[proxima::test]
    async fn proxima_pool_runs_many_tasks_in_parallel() {
        let pool = Arc::new(ProximaBackgroundPool::with_threads(4).expect("build pool"));
        let mut handles = Vec::new();
        for index in 0..16_u32 {
            handles.push(pool.spawn(move || Ok::<u32, ProximaError>(index * 2)));
        }
        let mut results: Vec<u32> = Vec::new();
        for handle in handles {
            let value = handle.await.expect("result");
            results.push(value);
        }
        results.sort_unstable();
        let expected: Vec<u32> = (0..16).map(|index| index * 2).collect();
        assert_eq!(results, expected);
    }

    #[proxima::test]
    async fn panicking_job_does_not_kill_worker() {
        let pool = Arc::new(ProximaBackgroundPool::with_threads(2).expect("build pool"));
        let bad = pool.spawn(|| -> Result<u32, ProximaError> {
            panic!("intentional");
        });
        let outcome = bad.await;
        assert!(outcome.is_err(), "panicking job → err");
        let good = pool.spawn(move || Ok::<u32, ProximaError>(77));
        let value = good.await.expect("good result");
        assert_eq!(value, 77);
    }

    #[proxima::test]
    async fn pool_drop_joins_workers_cleanly() {
        let pool = ProximaBackgroundPool::with_threads(3).expect("build pool");
        let handle = pool.spawn(|| Ok::<u32, ProximaError>(1));
        let _ = handle.await;
        drop(pool);
    }

    #[cfg(feature = "runtime-prime-bgpool-async")]
    #[proxima::test]
    async fn spawn_async_runs_future_on_worker() {
        let pool = Arc::new(ProximaBackgroundPool::with_threads(2).expect("build pool"));
        // exercise the polling-worker path: the future yields (via
        // tokio::task::yield_now) so we know the worker's runtime is
        // actually driving it as a future, not just block_on'ing a
        // ready-immediately leaf.
        let handle = pool.spawn_async(async move {
            tokio::task::yield_now().await;
            Ok::<u32, ProximaError>(123)
        });
        let value = handle.await.expect("async result");
        assert_eq!(value, 123);
    }

    #[cfg(feature = "runtime-prime-bgpool-async")]
    #[proxima::test]
    async fn spawn_async_concurrent_futures() {
        let pool = Arc::new(ProximaBackgroundPool::with_threads(4).expect("build pool"));
        let mut handles = Vec::new();
        for index in 0..16_u32 {
            handles.push(pool.spawn_async(async move {
                tokio::task::yield_now().await;
                Ok::<u32, ProximaError>(index * 3)
            }));
        }
        let mut results: Vec<u32> = Vec::new();
        for handle in handles {
            results.push(handle.await.expect("result"));
        }
        results.sort_unstable();
        let expected: Vec<u32> = (0..16).map(|index| index * 3).collect();
        assert_eq!(results, expected);
    }

    #[proxima::test]
    async fn dyn_trait_path_still_works() {
        // verify the `BackgroundPool::spawn` (object-safe) method also works
        // for callers that hold an `Arc<dyn BackgroundPool>`.
        let pool: Arc<dyn BackgroundPool> =
            Arc::new(ProximaBackgroundPool::with_threads(2).expect("build pool"));
        let handle = BackgroundPool::spawn(
            &*pool,
            Box::new(|| Ok(Box::new(99_u32) as Box<dyn Any + Send>)),
        );
        let value = handle.await.expect("dyn result");
        assert_eq!(*value.downcast::<u32>().expect("downcast"), 99);
    }
}
