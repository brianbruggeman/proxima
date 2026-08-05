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

/// Backs `#[proxima::fixture(once)]`: a value computed once per process and
/// shared as `&'static T`. `async_lock::OnceCell::new()` is `const`, so the
/// cell can be a `static` with no runtime coupling — unlike
/// `tokio::sync::Mutex::const_new`, which would drag tokio into every fixture.
///
/// ```
/// use proxima_test::AsyncOnce;
///
/// static GREETING: AsyncOnce<String> = AsyncOnce::new();
///
/// let shared: &'static String = futures::executor::block_on(
///     GREETING.get_or_init(|| async { "hello from the fixture".to_string() }),
/// );
/// assert_eq!(shared, "hello from the fixture");
/// ```
pub use async_lock::OnceCell as AsyncOnce;

/// Cassette directive emitted by the macro from `cassette = "name"`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CassetteSpec {
    pub name: &'static str,
    pub case: &'static str,
    pub manifest_dir: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Record,
    Replay,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CassetteCtx {
    pub path: PathBuf,
    pub mode: Mode,
}

// the deferred cleanups are an open set of caller futures with distinct types
// queued in one registry — the case the box-free rule exempts.
type TeardownRegistry = Arc<Mutex<Vec<Pin<Box<dyn Future<Output = ()> + Send>>>>>;

/// Per-test capability handed to the body by value.
pub struct TestCtx {
    cassette: Option<CassetteCtx>,
    teardowns: TeardownRegistry,
}

impl TestCtx {
    #[must_use]
    pub fn new(cassette: Option<CassetteCtx>) -> Self {
        Self {
            cassette,
            teardowns: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn cassette(&self) -> Option<&CassetteCtx> {
        self.cassette.as_ref()
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
    use std::panic;
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

    #[test]
    fn ctx_without_a_cassette_reports_none() {
        assert!(TestCtx::new(None).cassette().is_none());
    }

    #[test]
    fn ctx_hands_back_the_cassette_it_was_built_with() {
        let ctx = TestCtx::new(Some(CassetteCtx {
            path: PathBuf::from("tests/cassettes/health.jsonl"),
            mode: Mode::Replay,
        }));
        let cassette = ctx.cassette().expect("ctx was built with a cassette");
        assert_eq!(cassette.mode, Mode::Replay);
        assert_eq!(cassette.path, PathBuf::from("tests/cassettes/health.jsonl"));
    }

    #[test]
    fn err_outcome_panics_with_the_error_rendered() {
        let outcome: Result<(), &str> = Err("upstream refused");
        let failure = panic::catch_unwind(|| outcome.into_test_outcome())
            .expect_err("an Err body must fail the test");
        let message = failure
            .downcast_ref::<String>()
            .expect("panic payload is the rendered error");
        assert!(message.contains("upstream refused"), "got: {message}");
    }

    #[test]
    fn unit_outcome_is_a_pass() {
        ().into_test_outcome();
    }
}
