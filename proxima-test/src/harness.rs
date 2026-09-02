//! The driver half: panic capture, teardown draining, and the per-runtime
//! entry points the `#[proxima::test]` macro calls. Compiled only when a
//! driver feature (`tokio-driver` / `test-prime`) is on.

// test-harness code: a panic IS the failure path here (mirrors
// src/test_support.rs), so expect on these lines is the reporting mechanism.
#![allow(clippy::expect_used)]

use std::cell::{Cell, RefCell};
use std::future::Future;
use std::panic::{self, AssertUnwindSafe, PanicHookInfo};
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::{Once, PoisonError};
use std::task::{Context, Poll};

#[cfg(feature = "test-prime")]
use std::sync::OnceLock;
#[cfg(feature = "test-prime")]
use std::sync::mpsc::{RecvTimeoutError, sync_channel};
#[cfg(feature = "test-prime")]
use std::time::Duration;

#[cfg(feature = "test-prime")]
use prime::os::runtime::PrimeRuntime;
#[cfg(feature = "test-prime")]
use proxima_runtime::{CoreId, Runtime, SpawnError};

use crate::{CassetteCtx, CassetteSpec, Mode, TeardownRegistry, TestCtx};

// ---------------------------------------------------------------------------
// ctx construction
// ---------------------------------------------------------------------------

fn resolve_mode(path: &Path) -> Mode {
    match std::env::var("PROXIMA_CASSETTE").as_deref() {
        Ok("record") => Mode::Record,
        Ok("replay") => Mode::Replay,
        _ if path.exists() => Mode::Replay,
        _ => Mode::Record,
    }
}

fn build_test_ctx(cassette: Option<&CassetteSpec>) -> TestCtx {
    TestCtx::new(cassette.map(|spec| {
        let mut path = PathBuf::from(spec.manifest_dir);
        path.push("tests");
        path.push("cassettes");
        let file = if spec.case.is_empty() {
            format!("{}.jsonl", spec.name)
        } else {
            format!("{}__{}.jsonl", spec.name, spec.case)
        };
        path.push(file);
        let mode = resolve_mode(&path);
        CassetteCtx { path, mode }
    }))
}

// ---------------------------------------------------------------------------
// teardowns
// ---------------------------------------------------------------------------

async fn run_teardowns(teardowns: &TeardownRegistry) -> Option<CapturedPanic> {
    let mut pending = {
        let mut guard = teardowns.lock().unwrap_or_else(PoisonError::into_inner);
        std::mem::take(&mut *guard)
    };
    pending.reverse();
    let mut first_panic = None;
    for cleanup in pending {
        let outcome = CatchUnwind { inner: cleanup }.await;
        if outcome.is_err() && first_panic.is_none() {
            first_panic = Some(take_captured());
        }
    }
    first_panic
}

// ---------------------------------------------------------------------------
// panic capture
// ---------------------------------------------------------------------------

struct CapturedPanic {
    message: String,
    location: Option<String>,
}

thread_local! {
    static ARMED: Cell<bool> = const { Cell::new(false) };
    static SLOT: RefCell<Option<CapturedPanic>> = const { RefCell::new(None) };
}

static HOOK: Once = Once::new();

fn install_panic_hook_once() {
    HOOK.call_once(|| {
        let previous = panic::take_hook();
        panic::set_hook(Box::new(move |info: &PanicHookInfo<'_>| {
            if ARMED.with(Cell::get) {
                let captured = CapturedPanic {
                    message: panic_message(info),
                    location: info
                        .location()
                        .map(|loc| format!("{}:{}:{}", loc.file(), loc.line(), loc.column())),
                };
                SLOT.with(|slot| *slot.borrow_mut() = Some(captured));
            } else {
                previous(info);
            }
        }));
    });
}

static FD_LIMIT: Once = Once::new();

/// Raise the process's open-file soft limit to its hard limit, once. Every
/// test that builds an `App` spins a per-core runtime (reactor + worker
/// fds); with many such tests running concurrently under `cargo test`, the
/// platform default soft `RLIMIT_NOFILE` (as low as 256 on macOS) is easily
/// exhausted well before the process is actually short on real resources.
fn raise_fd_limit_once() {
    FD_LIMIT.call_once(|| {
        #[cfg(unix)]
        {
            let mut limit = libc::rlimit {
                rlim_cur: 0,
                rlim_max: 0,
            };
            // safety: both calls take a pointer to this live, fully initialised
            // local. best-effort: a failure here just leaves the platform
            // default in place, which is the status quo this improves on.
            if unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &mut limit) } == 0 {
                limit.rlim_cur = limit.rlim_max;
                unsafe {
                    libc::setrlimit(libc::RLIMIT_NOFILE, &limit);
                }
            }
        }
    });
}

fn panic_message(info: &PanicHookInfo<'_>) -> String {
    let payload = info.payload();
    if let Some(text) = payload.downcast_ref::<&str>() {
        (*text).to_string()
    } else if let Some(text) = payload.downcast_ref::<String>() {
        text.clone()
    } else {
        "panicked with a non-string payload".to_string()
    }
}

fn arm() {
    ARMED.with(|armed| armed.set(true));
}

fn disarm() {
    ARMED.with(|armed| armed.set(false));
}

fn take_captured() -> CapturedPanic {
    SLOT.with(|slot| slot.borrow_mut().take())
        .unwrap_or(CapturedPanic {
            message: "panicked without a captured message".to_string(),
            location: None,
        })
}

fn finish(outcome: Option<CapturedPanic>) {
    if let Some(captured) = outcome {
        match captured.location {
            Some(location) => panic!("{} (at {location})", captured.message),
            None => panic!("{}", captured.message),
        }
    }
}

fn outcome_of(result: std::thread::Result<()>) -> Option<CapturedPanic> {
    result.err().map(|_| take_captured())
}

// ---------------------------------------------------------------------------
// CatchUnwind future
// ---------------------------------------------------------------------------

struct CatchUnwind<F> {
    inner: F,
}

impl<F: Future> Future for CatchUnwind<F> {
    type Output = std::thread::Result<F::Output>;

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        // safety: `inner` is structurally pinned — `CatchUnwind` is never moved
        // out of, grants no `Unpin` shortcut, and has no `Drop`.
        let inner = unsafe { self.map_unchecked_mut(|wrapper| &mut wrapper.inner) };
        match panic::catch_unwind(AssertUnwindSafe(|| inner.poll(context))) {
            Ok(Poll::Pending) => Poll::Pending,
            Ok(Poll::Ready(output)) => Poll::Ready(Ok(output)),
            Err(payload) => Poll::Ready(Err(payload)),
        }
    }
}

// ---------------------------------------------------------------------------
// runtime entry points
// ---------------------------------------------------------------------------

#[cfg(feature = "test-prime")]
pub fn run<Body, Fut>(cassette: Option<CassetteSpec>, body: Body)
where
    Body: FnOnce(TestCtx) -> Fut + Send + 'static,
    Fut: Future<Output = ()> + 'static,
{
    finish(drive_prime(cassette.as_ref(), body));
}

#[cfg(all(not(feature = "test-prime"), feature = "tokio-driver"))]
pub fn run<Body, Fut>(cassette: Option<CassetteSpec>, body: Body)
where
    Body: FnOnce(TestCtx) -> Fut,
    Fut: Future<Output = ()>,
{
    finish(drive_tokio(cassette.as_ref(), body));
}

#[cfg(feature = "test-prime")]
pub fn run_prime<Body, Fut>(cassette: Option<CassetteSpec>, body: Body)
where
    Body: FnOnce(TestCtx) -> Fut + Send + 'static,
    Fut: Future<Output = ()> + 'static,
{
    finish(drive_prime(cassette.as_ref(), body));
}

#[cfg(feature = "tokio-driver")]
pub fn run_tokio<Body, Fut>(cassette: Option<CassetteSpec>, body: Body)
where
    Body: FnOnce(TestCtx) -> Fut,
    Fut: Future<Output = ()>,
{
    finish(drive_tokio(cassette.as_ref(), body));
}

#[cfg(feature = "tokio-driver")]
pub fn run_tokio_current_thread_paused<Body, Fut>(cassette: Option<CassetteSpec>, body: Body)
where
    Body: FnOnce(TestCtx) -> Fut,
    Fut: Future<Output = ()>,
{
    finish(drive_tokio_inner(cassette.as_ref(), true, body));
}

#[cfg(feature = "tokio-driver")]
pub fn run_tokio_multi_thread<Body, Fut>(
    cassette: Option<CassetteSpec>,
    workers: Option<usize>,
    body: Body,
) where
    Body: FnOnce(TestCtx) -> Fut,
    Fut: Future<Output = ()>,
{
    finish(drive_tokio_multi(cassette.as_ref(), workers, false, body));
}

#[cfg(feature = "tokio-driver")]
pub fn run_tokio_multi_thread_paused<Body, Fut>(
    cassette: Option<CassetteSpec>,
    workers: Option<usize>,
    body: Body,
) where
    Body: FnOnce(TestCtx) -> Fut,
    Fut: Future<Output = ()>,
{
    finish(drive_tokio_multi(cassette.as_ref(), workers, true, body));
}

#[cfg(feature = "tokio-driver")]
fn drive_tokio<Body, Fut>(cassette: Option<&CassetteSpec>, body: Body) -> Option<CapturedPanic>
where
    Body: FnOnce(TestCtx) -> Fut,
    Fut: Future<Output = ()>,
{
    drive_tokio_inner(cassette, false, body)
}

#[cfg(feature = "tokio-driver")]
fn drive_tokio_inner<Body, Fut>(
    cassette: Option<&CassetteSpec>,
    start_paused: bool,
    body: Body,
) -> Option<CapturedPanic>
where
    Body: FnOnce(TestCtx) -> Fut,
    Fut: Future<Output = ()>,
{
    install_panic_hook_once();
    raise_fd_limit_once();
    let ctx = build_test_ctx(cassette);
    let teardowns = ctx.teardowns.clone();
    let mut builder = tokio::runtime::Builder::new_current_thread();
    builder.enable_all();
    if start_paused {
        builder.start_paused(true);
    }
    let runtime = builder
        .build()
        .expect("proxima::test: build tokio current-thread runtime");
    runtime.block_on(async move {
        arm();
        let result = CatchUnwind { inner: body(ctx) }.await;
        let body_outcome = outcome_of(result);
        let teardown_panic = run_teardowns(&teardowns).await;
        disarm();
        body_outcome.or(teardown_panic)
    })
}

#[cfg(feature = "tokio-driver")]
fn drive_tokio_multi<Body, Fut>(
    cassette: Option<&CassetteSpec>,
    workers: Option<usize>,
    start_paused: bool,
    body: Body,
) -> Option<CapturedPanic>
where
    Body: FnOnce(TestCtx) -> Fut,
    Fut: Future<Output = ()>,
{
    install_panic_hook_once();
    raise_fd_limit_once();
    let ctx = build_test_ctx(cassette);
    let teardowns = ctx.teardowns.clone();
    let mut builder = tokio::runtime::Builder::new_multi_thread();
    builder.enable_all();
    if let Some(count) = workers {
        builder.worker_threads(count);
    }
    if start_paused {
        builder.start_paused(true);
    }
    let runtime = builder
        .build()
        .expect("proxima::test: build tokio multi-thread runtime");
    runtime.block_on(async move {
        arm();
        let result = CatchUnwind { inner: body(ctx) }.await;
        let body_outcome = outcome_of(result);
        let teardown_panic = run_teardowns(&teardowns).await;
        disarm();
        body_outcome.or(teardown_panic)
    })
}

#[cfg(feature = "test-prime")]
fn shared_prime_runtime() -> &'static PrimeRuntime {
    static RUNTIME: OnceLock<&'static PrimeRuntime> = OnceLock::new();
    RUNTIME.get_or_init(|| {
        #[cfg(feature = "test-prime-tokio-compat")]
        let runtime = PrimeRuntime::new_with_tokio_compat(1)
            .expect("proxima::test: build prime+tokio-compat runtime");
        #[cfg(not(feature = "test-prime-tokio-compat"))]
        let runtime = PrimeRuntime::new(1).expect("proxima::test: build prime runtime");
        // leaked on purpose: one prime runtime serves every test in the process
        // and must outlive the last of them.
        Box::leak(Box::new(runtime))
    })
}

#[cfg(feature = "test-prime")]
fn drive_prime<Body, Fut>(cassette: Option<&CassetteSpec>, body: Body) -> Option<CapturedPanic>
where
    Body: FnOnce(TestCtx) -> Fut + Send + 'static,
    Fut: Future<Output = ()> + 'static,
{
    install_panic_hook_once();
    raise_fd_limit_once();
    let ctx = build_test_ctx(cassette);
    let teardowns = ctx.teardowns.clone();
    let runtime = shared_prime_runtime();
    let (sender, receiver) = sync_channel::<BodyProgress>(2);

    // `spawn_factory_on_core`, not `spawn_on_core`: the factory (Send) crosses
    // the per-core channel and constructs the test body's future ON the
    // target core, so `Fut` itself never needs `Send` — this is what lets a
    // `#[proxima::test]` body await a `?Send` base-rung `Pipe::call`
    // (proxima-runtime/src/lib.rs:299-305 documents the same mechanism for
    // `App::run_until_signal`'s per-core listener loop).
    let started_sender = sender.clone();
    let factory: Box<dyn FnOnce() -> Pin<Box<dyn Future<Output = ()> + 'static>> + Send + 'static> =
        Box::new(move || {
            Box::pin(async move {
                // sent the instant this future is first polled — the executor
                // has genuinely begun the body, as opposed to still queueing
                // behind other work on the shared single worker.
                let _ = started_sender.send(BodyProgress::Started);
                arm();
                let result = CatchUnwind { inner: body(ctx) }.await;
                let body_outcome = outcome_of(result);
                let teardown_panic = run_teardowns(&teardowns).await;
                disarm();
                let _ = sender.send(BodyProgress::Done(body_outcome.or(teardown_panic)));
            })
        });

    match runtime.spawn_factory_on_core(CoreId(0), factory) {
        Ok(()) => {}
        Err(SpawnError::InboxFull) => return failed("prime core 0 inbox full on dispatch"),
        Err(SpawnError::Disconnected) => return failed("prime core 0 disconnected"),
    }

    // phase 1 — wait for the worker to actually begin polling the body. This
    // window absorbs OS scheduling / machine-wide contention (queueing), not
    // body work, so its budget is deliberately generous: see `dispatch_timeout`.
    match receiver.recv_timeout(dispatch_timeout()) {
        Ok(BodyProgress::Started) => {}
        Ok(BodyProgress::Done(outcome)) => return outcome,
        Err(RecvTimeoutError::Timeout) => {
            return failed("worker never began polling the test body within the dispatch timeout");
        }
        Err(RecvTimeoutError::Disconnected) => {
            return failed("prime core 0 disconnected before the body started");
        }
    }

    // phase 2 — the body is genuinely running now, so this clock measures
    // actual execution and only actual execution: a real hang still fails.
    match receiver.recv_timeout(body_timeout()) {
        Ok(BodyProgress::Done(outcome)) => outcome,
        Ok(BodyProgress::Started) => failed("worker reported a second start signal"),
        Err(RecvTimeoutError::Timeout) => failed("body did not complete within the timeout"),
        Err(RecvTimeoutError::Disconnected) => {
            failed("worker dropped the completion channel without reporting")
        }
    }
}

#[cfg(feature = "test-prime")]
enum BodyProgress {
    Started,
    Done(Option<CapturedPanic>),
}

#[cfg(feature = "test-prime")]
fn body_timeout() -> Duration {
    duration_from_env("PROXIMA_TEST_TIMEOUT_MS", Duration::from_secs(60))
}

/// Generous by design (default: 5 minutes) — this phase absorbs machine-wide
/// scheduling contention (many concurrent test processes, sibling builds)
/// that is not the test's fault, so it must not fire on contention alone.
/// `body_timeout` — not this one — is what catches a genuine hang.
#[cfg(feature = "test-prime")]
fn dispatch_timeout() -> Duration {
    duration_from_env("PROXIMA_TEST_DISPATCH_TIMEOUT_MS", Duration::from_secs(300))
}

#[cfg(feature = "test-prime")]
fn duration_from_env(key: &str, default: Duration) -> Duration {
    match std::env::var(key) {
        Ok(value) => value.parse::<u64>().map_or(default, Duration::from_millis),
        Err(_) => default,
    }
}

#[cfg(feature = "test-prime")]
fn failed(message: &str) -> Option<CapturedPanic> {
    Some(CapturedPanic {
        message: format!("proxima::test: {message}"),
        location: None,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[cfg(feature = "tokio-driver")]
    #[test]
    fn tokio_reports_passing_body() {
        assert!(drive_tokio(None, |_cx| async {}).is_none());
    }

    #[cfg(feature = "tokio-driver")]
    #[test]
    fn tokio_reports_panicking_body_with_message() {
        let captured = drive_tokio(None, |_cx| async { panic!("boom-tokio") })
            .expect("a panicking body must be reported as failed");
        assert!(captured.message.contains("boom-tokio"));
    }

    #[cfg(feature = "test-prime")]
    #[test]
    fn prime_reports_passing_body() {
        assert!(drive_prime(None, |_cx| async {}).is_none());
    }

    #[cfg(feature = "test-prime")]
    #[test]
    fn prime_reports_panicking_body_with_message() {
        let captured = drive_prime(None, |_cx| async { panic!("boom-prime") })
            .expect("a panicking body must be reported as failed");
        assert!(captured.message.contains("boom-prime"));
    }

    // mirrors `proxima_primitives::pipe::Pipe::call` (base rung, no `Send`
    // bound on the returned future — `proxima-primitives/src/pipe/primitives.rs:100-101`)
    // via the same `RefCell` idiom as `proxima_autograd::optimizer::AdamStep`
    // (`proxima-autograd/src/optimizer.rs:217-254`): `RefCell` is `!Sync`, so
    // `&RefCellPipe` is `!Send` and any future holding it — as `call`'s does —
    // is `!Send` too, which is exactly what `drive_prime` could not accept
    // before this change. The guard is dropped before the `.await` (never
    // held across it) to stay clean under `clippy::await_holding_refcell_ref`.
    #[cfg(feature = "test-prime")]
    struct RefCellPipe {
        state: RefCell<u32>,
    }

    #[cfg(feature = "test-prime")]
    impl RefCellPipe {
        async fn call(&self, delta: u32) -> u32 {
            {
                let mut guard = self.state.borrow_mut();
                *guard += delta;
            }
            std::future::ready(()).await;
            *self.state.borrow()
        }
    }

    #[cfg(feature = "test-prime")]
    #[test]
    fn prime_drives_a_non_send_refcell_backed_pipe_future() {
        let outcome = drive_prime(None, |_cx| async move {
            let pipe = RefCellPipe {
                state: RefCell::new(0),
            };
            assert_eq!(pipe.call(5).await, 5);
            assert_eq!(pipe.call(3).await, 8);
        });
        assert!(outcome.is_none(), "non-Send body must run to completion");
    }

    #[cfg(feature = "tokio-driver")]
    #[test]
    fn teardown_runs_on_pass_and_on_panic() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static RAN: AtomicUsize = AtomicUsize::new(0);

        let outcome = drive_tokio(None, |cx| async move {
            cx.defer(async {
                RAN.fetch_add(1, Ordering::SeqCst);
            });
        });
        assert!(outcome.is_none());
        assert_eq!(RAN.load(Ordering::SeqCst), 1, "teardown must run on pass");

        let outcome = drive_tokio(None, |cx| async move {
            cx.defer(async {
                RAN.fetch_add(1, Ordering::SeqCst);
            });
            panic!("boom");
        });
        assert!(outcome.is_some());
        assert_eq!(
            RAN.load(Ordering::SeqCst),
            2,
            "teardown must run even when the body panics"
        );
    }

    #[cfg(feature = "tokio-driver")]
    #[test]
    fn a_teardown_panic_fails_an_otherwise_passing_body() {
        let captured = drive_tokio(None, |cx| async move {
            cx.defer(async { panic!("boom-teardown") });
        })
        .expect("a panicking teardown must fail the test");
        assert!(captured.message.contains("boom-teardown"));
    }

    #[test]
    fn cassette_path_is_manifest_tests_cassettes_name_case() {
        let ctx = build_test_ctx(Some(&CassetteSpec {
            name: "health",
            case: "warm",
            manifest_dir: "/tmp/proxima-test-manifest",
        }));
        let cassette = ctx.cassette().expect("spec was supplied");
        assert!(
            cassette
                .path
                .ends_with("tests/cassettes/health__warm.jsonl"),
            "got: {}",
            cassette.path.display()
        );
    }

    #[test]
    fn an_absent_cassette_file_records() {
        temp_env::with_var("PROXIMA_CASSETTE", None::<&str>, || {
            let ctx = build_test_ctx(Some(&CassetteSpec {
                name: "never-recorded",
                case: "",
                manifest_dir: "/tmp/proxima-test-manifest",
            }));
            let cassette = ctx.cassette().expect("spec was supplied");
            assert!(
                cassette
                    .path
                    .ends_with("tests/cassettes/never-recorded.jsonl")
            );
            assert_eq!(cassette.mode, Mode::Record);
        });
    }

    #[test]
    fn the_env_override_forces_replay_even_without_a_recorded_file() {
        temp_env::with_var("PROXIMA_CASSETTE", Some("replay"), || {
            let ctx = build_test_ctx(Some(&CassetteSpec {
                name: "never-recorded",
                case: "",
                manifest_dir: "/tmp/proxima-test-manifest",
            }));
            let cassette = ctx.cassette().expect("spec was supplied");
            assert_eq!(cassette.mode, Mode::Replay);
        });
    }

    #[test]
    fn no_spec_means_no_cassette() {
        assert!(build_test_ctx(None).cassette().is_none());
    }
}
