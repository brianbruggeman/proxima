//! Runtime harness for `#[proxima::test]`: the cassette spec the macro emits,
//! the per-test [`TestCtx`] handed to a body, and — behind `tokio-driver` or
//! `test-prime` — the runtime entry points that drive that body and turn its
//! panic into a test failure.
//!
//! std is the finest tier this crate has: the driver installs a `std::panic`
//! hook, reads the environment, and parks a thread on the driving runtime.
//! With neither driver feature it compiles to the macro-facing data surface
//! alone — which is exactly what the runtime-agnostic `proxima/test-support`
//! feature links. No app surface, so foundational crates can dev-dep on this
//! without pulling the proxima umbrella.

use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::{Arc, Mutex, PoisonError};

#[cfg(any(feature = "tokio-driver", feature = "test-prime"))]
mod harness;

#[cfg(any(feature = "tokio-driver", feature = "test-prime"))]
pub use harness::*;

// ---------------------------------------------------------------------------
// async-once
// ---------------------------------------------------------------------------

/// Backs `#[proxima::fixture(once)]`: a value computed once per process and
/// shared as `&'static T`. Backed by `async_lock::OnceCell` (no runtime
/// coupling — `async_lock::OnceCell::new()` is `const`, same as tokio's
/// `Mutex::const_new`, so `static CELL: AsyncOnce<T> = AsyncOnce::new();`
/// keeps working without a tokio dependency).
pub struct AsyncOnce<T>(async_lock::OnceCell<T>);

impl<T: Send + Sync> AsyncOnce<T> {
    #[must_use]
    pub const fn new() -> Self {
        Self(async_lock::OnceCell::new())
    }

    pub async fn get_or_init<Factory, Fut>(&'static self, init: Factory) -> &'static T
    where
        Factory: FnOnce() -> Fut,
        Fut: Future<Output = T>,
    {
        self.0.get_or_init(init).await
    }
}

impl<T: Send + Sync> Default for AsyncOnce<T> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// plan + cassette spec
// ---------------------------------------------------------------------------

/// Cassette directive emitted by the macro from `cassette = "name"`.
pub struct CassetteSpec {
    pub name: &'static str,
    pub case: &'static str,
    pub manifest_dir: &'static str,
}

/// What the macro hands each runtime entry point.
pub struct Plan {
    pub cassette: Option<CassetteSpec>,
}

impl Plan {
    #[must_use]
    pub fn new() -> Self {
        Self { cassette: None }
    }

    #[must_use]
    pub fn with_cassette(spec: CassetteSpec) -> Self {
        Self {
            cassette: Some(spec),
        }
    }
}

impl Default for Plan {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// cassette ctx + mode
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Record,
    Replay,
}

pub struct CassetteCtx {
    pub path: PathBuf,
    pub mode: Mode,
}

// ---------------------------------------------------------------------------
// TestCtx
// ---------------------------------------------------------------------------

// the deferred cleanups are an open set of caller futures with distinct types
// queued in one registry — the case the box-free rule exempts.
type TeardownRegistry = Arc<Mutex<Vec<Pin<Box<dyn Future<Output = ()> + Send>>>>>;

/// Per-test capability handed to the body by value.
pub struct TestCtx {
    cassette: Option<CassetteCtx>,
    teardowns: TeardownRegistry,
}

impl TestCtx {
    pub fn cassette(&self) -> Option<&CassetteCtx> {
        self.cassette.as_ref()
    }

    /// Test-only constructor for creating a `TestCtx` directly in tests that
    /// need to exercise cassette logic without going through the macro harness.
    #[doc(hidden)]
    pub fn __new_for_test(cassette: Option<CassetteCtx>) -> Self {
        Self {
            cassette,
            teardowns: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn defer<Fut>(&self, cleanup: Fut)
    where
        Fut: Future<Output = ()> + Send + 'static,
    {
        self.teardowns
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(Box::pin(cleanup));
    }
}

// ---------------------------------------------------------------------------
// outcome trait
// ---------------------------------------------------------------------------

/// Lets a `#[proxima::test]` body return `()` or `Result<(), E>`.
pub trait IntoTestOutcome {
    fn into_test_outcome(self);
}

impl IntoTestOutcome for () {
    fn into_test_outcome(self) {}
}

impl<Error: core::fmt::Debug> IntoTestOutcome for Result<(), Error> {
    fn into_test_outcome(self) {
        if let Err(error) = self {
            panic!("test returned Err: {error:?}");
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    #[test]
    fn async_once_initializes_exactly_once() {
        static ONCE: AsyncOnce<u32> = AsyncOnce::new();
        static COUNT: AtomicU32 = AtomicU32::new(0);
        futures::executor::block_on(async {
            let first = *ONCE
                .get_or_init(|| async {
                    COUNT.fetch_add(1, Ordering::SeqCst);
                    7
                })
                .await;
            let second = *ONCE
                .get_or_init(|| async {
                    COUNT.fetch_add(1, Ordering::SeqCst);
                    99
                })
                .await;
            assert_eq!(first, 7);
            assert_eq!(second, 7, "second init must be memoized, not re-run");
            assert_eq!(
                COUNT.load(Ordering::SeqCst),
                1,
                "initializer must run exactly once"
            );
        });
    }
}
