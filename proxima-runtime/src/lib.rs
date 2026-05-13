//! Runtime trait surface. Abstract scheduling primitives that
//! concrete runtimes (`TokioPerCoreRuntime`, `PrimeRuntime`,
//! `RayonBackgroundPool`) implement.
//!
//! Moved out of the proxima umbrella during Phase 3 of the
//! decomposition (see `proxima/rust/docs/decomposition/discipline.md`).
//! The trait and value types live here so `prime` can be a separate
//! crate without depending on the umbrella.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "rayon")]
pub mod background_rayon;
#[cfg(feature = "rayon")]
pub use background_rayon::RayonBackgroundPool;

#[cfg(feature = "concurrency")]
pub mod concurrency;

#[cfg(feature = "alloc")]
pub mod primitives;
#[cfg(feature = "alloc")]
pub use primitives::{
    JoinError, JoinSetLike, LocalMutexLike, LocalNotifyLike, MutexLike, NotifyLike, SleepFuture,
};
#[cfg(all(feature = "alloc", feature = "std"))]
pub use primitives::{LocalRuntimeFactory, RuntimeFactory};

// Default `Runtime` impl: N tokio current-thread runtimes pinned one per CPU
// core (folded from the former proxima-runtime-tokio crate — FOLD 1 of the
// runtime backend consolidation). Nested under its own module (not the crate
// root) because it sits alongside this crate's own `primitives` module above
// (the RuntimeFactory traits) — nesting avoids the name collision.
#[cfg(feature = "tokio")]
pub mod tokio;

#[cfg(feature = "alloc")]
use core::convert::Infallible;
#[cfg(feature = "alloc")]
use core::future::Future;
#[cfg(feature = "alloc")]
use core::pin::Pin;

#[cfg(feature = "alloc")]
use proxima_core::ProximaError;

#[cfg(feature = "alloc")]
use alloc::boxed::Box;

/// Cross-core worker dispatch message used by `Runtime` impls. Generic
/// over an `Inline` extension type so prime's typed-future fast path can
/// add a variant (`SendInline`) carrying its `InlineTask`, while tokio's
/// per-core impl uses the default (`Infallible`) which makes the
/// `SendInline` arm unreachable.
///
/// Trace-span context no longer travels embedded in this enum (moved to
/// `telemetry::Spanned<T>`, which wraps the future itself) — see Wave D
/// Phase 1.
#[cfg(feature = "alloc")]
#[allow(clippy::large_enum_variant)]
pub enum SpawnRequest<Inline = Infallible> {
    /// A `Send` future shipped across cores; runs locally on arrival.
    Send(Pin<Box<dyn Future<Output = ()> + Send + 'static>>),
    /// A `Send` closure that builds a `?Send` future on the target core.
    /// Lets cross-thread callers dispatch per-core-only futures.
    Factory(Box<dyn FnOnce() -> Pin<Box<dyn Future<Output = ()> + 'static>> + Send + 'static>),
    /// Runtime-impl-specific extension. Default (`Infallible`) makes this
    /// arm unreachable. Prime sets `Inline = InlineTask` to carry typed
    /// futures without the `Pin<Box<dyn Future>>` allocation.
    SendInline(Inline),
    /// Graceful shutdown signal — the worker breaks its event loop.
    Shutdown,
}

#[cfg(feature = "alloc")]
impl<Inline> core::fmt::Debug for SpawnRequest<Inline> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Send(..) => formatter.write_str("SpawnRequest::Send"),
            Self::Factory(..) => formatter.write_str("SpawnRequest::Factory"),
            Self::SendInline(..) => formatter.write_str("SpawnRequest::SendInline"),
            Self::Shutdown => formatter.write_str("SpawnRequest::Shutdown"),
        }
    }
}

/// Identifier for a pinned core in the runtime. Opaque; not assumed to
/// map 1:1 onto OS CPU ids. Construction is implementation-specific.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CoreId(pub usize);

impl CoreId {
    #[must_use]
    pub fn as_usize(&self) -> usize {
        self.0
    }
}

/// Handle to a task running on the background work-stealing pool. The
/// chain runtime stays per-core; this handle lets a Pipe await a
/// CPU-bound side computation that *does* migrate between threads.
#[cfg(feature = "alloc")]
pub type BackgroundHandle<R> = Pin<Box<dyn Future<Output = Result<R, ProximaError>> + Send>>;

/// Pluggable backend for cross-thread CPU-bound work.
///
/// The chain `Runtime` is per-core (`!Send` futures). When a Pipe
/// needs to do work that *does* migrate threads — image decoding,
/// model inference, parallel parsing, fork-join compute — it routes
/// through here. The pool runs on its own threads; the returned
/// handle is `Send` so the per-core caller can await it across the
/// chain-runtime / pool boundary.
///
/// Default: `tokio::task::spawn_blocking` (already wired into
/// `TokioPerCoreRuntime::spawn_background_blocking` — good for
/// I/O-blocking work, scales to 512 threads). Alternative:
/// `RayonBackgroundPool` (feature `background-rayon`) — work-
/// stealing across a fixed thread count, the right shape for
/// CPU-bound parallel compute.
#[cfg(feature = "alloc")]
pub trait BackgroundPool: Send + Sync + 'static {
    fn spawn(
        &self,
        work: Box<dyn FnOnce() -> Result<Box<dyn core::any::Any + Send>, ProximaError> + Send>,
    ) -> BackgroundHandle<Box<dyn core::any::Any + Send>>;
}

/// Failure modes for cross-core spawn dispatch. Surfaced by
/// `Runtime::spawn_on_core` and `Runtime::spawn_factory_on_core` so callers
/// must explicitly handle back-pressure rather than relying on silent
/// drops — which was the cause of the inbox-saturation hang exposed by
/// `runtime_spawn_on_core_silently_drops_on_inbox_overflow` in earlier
/// versions.
///
/// `InboxFull` is *transient*: retry by rebuilding the future and calling
/// again, ideally with a small backoff. The `spawn_on_core_blocking_with`
/// helper below automates this loop.
///
/// `Disconnected` is *terminal*: the target core has shut down. The
/// future is dropped; retrying will return the same error.
///
/// Implementations that cannot fail on back-pressure (e.g. tokio's
/// `flume::unbounded()` mpsc never returns Full) will simply never
/// produce `InboxFull` and callers see `Ok(())` in the steady state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpawnError {
    /// The target core's inbox lane is at capacity. The future has been
    /// dropped; the caller should rebuild and retry (see
    /// [`spawn_on_core_blocking_with`]).
    InboxFull,
    /// The target core has shut down or never existed (out-of-range
    /// `CoreId`). The future is dropped; retrying is futile.
    Disconnected,
}

impl core::fmt::Display for SpawnError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InboxFull => formatter.write_str("inbox lane full; transient back-pressure"),
            Self::Disconnected => formatter.write_str("target core disconnected"),
        }
    }
}

impl core::error::Error for SpawnError {}

/// Loop on `spawn_on_core` until the future is accepted (`Ok`) or the
/// target core is shut down (`SpawnError::Disconnected`). The factory
/// closure is invoked once per attempt so the future is freshly
/// constructed each time — necessary because `spawn_on_core` consumes
/// the future on every call (including failures).
///
/// This is the canonical helper for batch dispatchers (load generators,
/// bench harnesses, mass-spawn migrations) that genuinely want
/// back-pressure absorbed at the caller via busy-yield rather than
/// surfaced as an error. Production code that has a meaningful response
/// to `InboxFull` (shed load, return 503, etc.) should call
/// `spawn_on_core` directly.
///
/// Returns `Ok(())` on success or `Err(SpawnError::Disconnected)` if the
/// target core is gone.
#[cfg(feature = "alloc")]
pub fn spawn_on_core_blocking_with<F>(
    runtime: &dyn Runtime,
    core_id: CoreId,
    mut factory: F,
) -> Result<(), SpawnError>
where
    F: FnMut() -> Pin<Box<dyn Future<Output = ()> + Send + 'static>>,
{
    loop {
        match runtime.spawn_on_core(core_id, factory()) {
            Ok(()) => return Ok(()),
            Err(SpawnError::InboxFull) => {
                #[cfg(feature = "std")]
                std::thread::yield_now();
                #[cfg(not(feature = "std"))]
                core::hint::spin_loop();
            }
            Err(SpawnError::Disconnected) => return Err(SpawnError::Disconnected),
        }
    }
}

/// Drive `future` to completion on a [`Runtime`] you already HOLD, returning
/// its output. The runtime-backed sibling of the no-runtime
/// `proxima_primitives::block_on` poll loop: same verb ("drive a future to
/// completion"), but the future runs on `runtime`'s core 0 worker instead of
/// the calling thread. Composes [`Runtime::spawn_on_core`] with a
/// `sync_channel(1)` for the thread-park / value handoff — the exact core of
/// the edge `run_prime` driver, lifted off the concrete `PrimeRuntime` onto
/// `&dyn Runtime` (a free fn, not a trait method, so `dyn Runtime` stays
/// object-safe).
///
/// FOREIGN-THREAD entry: call this from a thread that is NOT a worker of
/// `runtime`, or you deadlock the worker (the same rule as
/// `tokio::runtime::Runtime::block_on`). The edge `run_prime` avoids this by
/// booting a dedicated driver core; this primitive does not — it drives on
/// core 0 and parks the caller.
///
/// # Errors
/// Returns `ProximaError::Io` if core 0's inbox rejects the dispatch
/// (`InboxFull`/`Disconnected`) or the worker drops the completion channel
/// without producing a value.
#[cfg(all(feature = "alloc", feature = "std"))]
#[must_use = "block_on returns the future's output or the dispatch error"]
pub fn block_on<F>(runtime: &dyn Runtime, future: F) -> Result<F::Output, ProximaError>
where
    F: Future + Send + 'static,
    F::Output: Send + 'static,
{
    let (sender, receiver) = std::sync::mpsc::sync_channel::<F::Output>(1);
    let task = async move {
        let output = future.await;
        let _ = sender.send(output);
    };
    match runtime.spawn_on_core(CoreId(0), Box::pin(task)) {
        Ok(()) => {}
        Err(SpawnError::InboxFull) => {
            return Err(block_on_dispatch_error(
                "runtime core 0 inbox full on block_on dispatch",
            ));
        }
        Err(SpawnError::Disconnected) => {
            return Err(block_on_dispatch_error(
                "runtime core 0 disconnected on block_on dispatch",
            ));
        }
    }
    receiver
        .recv()
        .map_err(|_| block_on_dispatch_error("runtime worker dropped the block_on completion channel"))
}

#[cfg(all(feature = "alloc", feature = "std"))]
fn block_on_dispatch_error(message: &str) -> ProximaError {
    ProximaError::Io(std::io::Error::other(message))
}

/// Per-core executor abstraction. Implementations select the concrete
/// threading + I/O model:
/// - **Default**: tokio current-thread runtimes, one per pinned CPU core,
///   spawned at app construction (see `runtime::tokio_per_core`, feature `runtime-tokio`).
/// - **Alternatives**: monoio, glommio, custom kernels — all addressable
///   via the same trait surface.
///
/// Futures spawned via `spawn_on_current_core` are `!Send` by intent: each
/// task is pinned to its core for life. `spawn_on_core` and
/// `spawn_background_blocking` are the explicit cross-thread escape hatches.
#[cfg(feature = "alloc")]
pub trait Runtime: Send + Sync + 'static {
    /// Spawn a future on the current core's executor. Future is `!Send` by
    /// default — it lives on this thread until completion. Use this for
    /// per-connection handlers, per-request work, and substrate-internal
    /// bookkeeping.
    fn spawn_on_current_core(&self, future: Pin<Box<dyn Future<Output = ()> + 'static>>);

    /// Spawn a future on a *designated* core's executor. The future must
    /// be `Send` because it traverses the per-core message channel before
    /// the target core picks it up.
    ///
    /// Returns `Err(SpawnError::InboxFull)` when the target core's inbox
    /// lane is at capacity — callers must explicitly choose to retry
    /// (use [`spawn_on_core_blocking_with`]) or shed load. **The future
    /// is consumed on `Err`**; retrying requires rebuilding it.
    /// Returns `Err(SpawnError::Disconnected)` when the target core has
    /// shut down.
    fn spawn_on_core(
        &self,
        core_id: CoreId,
        future: Pin<Box<dyn Future<Output = ()> + Send + 'static>>,
    ) -> Result<(), SpawnError>;

    /// Cross-thread dispatch of a future *factory*: a `Send` closure that
    /// runs on the target core and there constructs a `?Send` future. The
    /// factory and its captures cross the per-core channel; the future
    /// itself is built locally on the destination core and stays there for
    /// life. This is what lets a caller (e.g. `App::run_until_signal` running
    /// on outer tokio) inject a per-core listener loop that internally
    /// awaits `?Send` `Pipe::call`s.
    ///
    /// Same back-pressure semantics as [`Self::spawn_on_core`]: returns
    /// `Err(SpawnError::InboxFull)` on saturation (factory is consumed),
    /// `Err(SpawnError::Disconnected)` on shutdown.
    fn spawn_factory_on_core(
        &self,
        core_id: CoreId,
        factory: Box<dyn FnOnce() -> Pin<Box<dyn Future<Output = ()> + 'static>> + Send + 'static>,
    ) -> Result<(), SpawnError>;

    /// Submit a CPU-bound blocking task to the background work-stealing pool.
    /// The returned handle is `Send` so the awaiter (which lives on a per-core
    /// runtime) can park until the result is ready, then resume on its core.
    fn spawn_background_blocking(
        &self,
        work: Box<dyn FnOnce() -> Result<Box<dyn core::any::Any + Send>, ProximaError> + Send>,
    ) -> BackgroundHandle<Box<dyn core::any::Any + Send>>;

    /// Yield a future that resolves at `deadline`. Used for timeouts and
    /// scheduled work. Implementations drive the timer via their per-core
    /// timer wheel. Only available with `std` because `Instant` requires
    /// std — keeps `&dyn Runtime` object-safe without an associated type.
    #[cfg(feature = "std")]
    fn timer_at(&self, deadline: std::time::Instant)
    -> Pin<Box<dyn Future<Output = ()> + 'static>>;

    /// Number of pinned cores. Stable across the lifetime of the runtime.
    fn num_cores(&self) -> usize;

    /// The current core's id. Implementations panic if called from outside
    /// a runtime worker thread — that's a bug at the call site, not a
    /// recoverable error.
    fn current_core(&self) -> CoreId;
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[cfg(feature = "alloc")]
    use alloc::boxed::Box;

    #[test]
    fn core_id_round_trips_as_usize() {
        let id = CoreId(7);
        assert_eq!(id.as_usize(), 7);
    }

    #[test]
    fn spawn_error_inbox_full_display() {
        let err = SpawnError::InboxFull;
        assert_eq!(err.to_string(), "inbox lane full; transient back-pressure");
    }

    #[test]
    fn spawn_error_disconnected_display() {
        let err = SpawnError::Disconnected;
        assert_eq!(err.to_string(), "target core disconnected");
    }

    #[test]
    fn spawn_error_source_is_none() {
        use core::error::Error;
        assert!(SpawnError::InboxFull.source().is_none());
        assert!(SpawnError::Disconnected.source().is_none());
    }

    #[cfg(feature = "alloc")]
    mod alloc_tests {
        use super::*;
        use core::future::Future;
        use core::pin::Pin;
        use core::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

        struct MockRuntime {
            send_dispatched: core::sync::atomic::AtomicBool,
            factory_dispatched: core::sync::atomic::AtomicBool,
        }

        impl MockRuntime {
            fn new() -> Self {
                Self {
                    send_dispatched: core::sync::atomic::AtomicBool::new(false),
                    factory_dispatched: core::sync::atomic::AtomicBool::new(false),
                }
            }
        }

        impl Runtime for MockRuntime {
            fn spawn_on_current_core(&self, _future: Pin<Box<dyn Future<Output = ()> + 'static>>) {
                self.send_dispatched
                    .store(true, core::sync::atomic::Ordering::Relaxed);
            }

            fn spawn_on_core(
                &self,
                _core_id: CoreId,
                _future: Pin<Box<dyn Future<Output = ()> + Send + 'static>>,
            ) -> Result<(), SpawnError> {
                self.send_dispatched
                    .store(true, core::sync::atomic::Ordering::Relaxed);
                Ok(())
            }

            fn spawn_factory_on_core(
                &self,
                _core_id: CoreId,
                _factory: Box<
                    dyn FnOnce() -> Pin<Box<dyn Future<Output = ()> + 'static>> + Send + 'static,
                >,
            ) -> Result<(), SpawnError> {
                self.factory_dispatched
                    .store(true, core::sync::atomic::Ordering::Relaxed);
                Ok(())
            }

            fn spawn_background_blocking(
                &self,
                work: Box<
                    dyn FnOnce() -> Result<Box<dyn core::any::Any + Send>, ProximaError> + Send,
                >,
            ) -> BackgroundHandle<Box<dyn core::any::Any + Send>> {
                let result = work();
                Box::pin(async move { result })
            }

            fn num_cores(&self) -> usize {
                1
            }

            fn current_core(&self) -> CoreId {
                CoreId(0)
            }

            #[cfg(feature = "std")]
            fn timer_at(
                &self,
                _deadline: std::time::Instant,
            ) -> Pin<Box<dyn Future<Output = ()> + 'static>> {
                Box::pin(async {})
            }
        }

        fn noop_waker() -> Waker {
            const VTABLE: RawWakerVTable =
                RawWakerVTable::new(|ptr| RawWaker::new(ptr, &VTABLE), |_| {}, |_| {}, |_| {});
            unsafe { Waker::from_raw(RawWaker::new(core::ptr::null(), &VTABLE)) }
        }

        #[test]
        fn spawn_request_send_dispatches() {
            let request: SpawnRequest = SpawnRequest::Send(Box::pin(async {}));
            assert!(matches!(request, SpawnRequest::Send(..)));
        }

        #[test]
        fn spawn_request_factory_dispatches() {
            let request: SpawnRequest = SpawnRequest::Factory(Box::new(|| Box::pin(async {})));
            assert!(matches!(request, SpawnRequest::Factory(..)));
        }

        #[test]
        fn spawn_request_shutdown_dispatches() {
            let request: SpawnRequest<core::convert::Infallible> = SpawnRequest::Shutdown;
            assert!(matches!(request, SpawnRequest::Shutdown));
        }

        #[test]
        fn background_handle_round_trip_with_payload() {
            let rt = MockRuntime::new();
            let payload = alloc::string::String::from("hello");
            let mut handle = rt.spawn_background_blocking(Box::new(move || {
                Ok(Box::new(payload) as Box<dyn core::any::Any + Send>)
            }));

            let waker = noop_waker();
            let mut ctx = Context::from_waker(&waker);
            let poll = Pin::new(&mut handle).poll(&mut ctx);

            let result = match poll {
                Poll::Ready(result) => result,
                Poll::Pending => panic!("expected ready"),
            };

            let boxed = result.expect("ok result");
            let value = boxed.downcast::<alloc::string::String>().expect("downcast");
            assert_eq!(*value, "hello");
        }

        #[cfg(feature = "std")]
        #[test]
        fn timer_at_compiles_and_is_callable() {
            let rt = MockRuntime::new();
            let deadline = std::time::Instant::now();
            let _future = rt.timer_at(deadline);
        }
    }
}
