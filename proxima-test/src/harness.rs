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
use std::sync::{Arc, Mutex, Once};
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

use crate::{CassetteCtx, Mode, Plan, TeardownRegistry, TestCtx};

// ---------------------------------------------------------------------------
// resolve_mode + build_test_ctx
// ---------------------------------------------------------------------------

pub fn resolve_mode(path: &Path) -> Mode {
    match std::env::var("PROXIMA_CASSETTE").as_deref() {
        Ok("record") => Mode::Record,
        Ok("replay") => Mode::Replay,
        _ if path.exists() => Mode::Replay,
        _ => Mode::Record,
    }
}

pub fn build_test_ctx(plan: &Plan) -> TestCtx {
    let cassette = plan.cassette.as_ref().map(|spec| {
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
    });
    TestCtx {
        cassette,
        teardowns: Arc::new(Mutex::new(Vec::new())),
    }
}

// ---------------------------------------------------------------------------
// teardowns
// ---------------------------------------------------------------------------

async fn run_teardowns(teardowns: &TeardownRegistry) -> Option<CapturedPanic> {
    let mut pending = {
        let mut guard = teardowns.lock().expect("teardown registry");
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

enum TestReport {
    Passed,
    Failed(CapturedPanic),
}

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

fn finish(report: TestReport) {
    if let TestReport::Failed(captured) = report {
        match captured.location {
            Some(location) => panic!("{} (at {location})", captured.message),
            None => panic!("{}", captured.message),
        }
    }
}

fn report_from(result: std::thread::Result<()>) -> TestReport {
    match result {
        Ok(()) => TestReport::Passed,
        Err(_) => TestReport::Failed(take_captured()),
    }
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
pub fn run<Body, Fut>(plan: Plan, body: Body)
where
    Body: FnOnce(TestCtx) -> Fut + Send + 'static,
    Fut: Future<Output = ()> + Send + 'static,
{
    finish(drive_prime(&plan, body));
}

#[cfg(all(not(feature = "test-prime"), feature = "tokio-driver"))]
pub fn run<Body, Fut>(plan: Plan, body: Body)
where
    Body: FnOnce(TestCtx) -> Fut,
    Fut: Future<Output = ()>,
{
    finish(drive_tokio(&plan, body));
}

#[cfg(feature = "test-prime")]
pub fn run_prime<Body, Fut>(plan: Plan, body: Body)
where
    Body: FnOnce(TestCtx) -> Fut + Send + 'static,
    Fut: Future<Output = ()> + Send + 'static,
{
    finish(drive_prime(&plan, body));
}

#[cfg(feature = "tokio-driver")]
pub fn run_tokio<Body, Fut>(plan: Plan, body: Body)
where
    Body: FnOnce(TestCtx) -> Fut,
    Fut: Future<Output = ()>,
{
    finish(drive_tokio(&plan, body));
}

#[cfg(feature = "tokio-driver")]
pub fn run_tokio_current_thread_paused<Body, Fut>(plan: Plan, body: Body)
where
    Body: FnOnce(TestCtx) -> Fut,
    Fut: Future<Output = ()>,
{
    finish(drive_tokio_paused(&plan, body));
}

#[cfg(feature = "tokio-driver")]
pub fn run_tokio_multi_thread<Body, Fut>(plan: Plan, workers: Option<usize>, body: Body)
where
    Body: FnOnce(TestCtx) -> Fut,
    Fut: Future<Output = ()>,
{
    finish(drive_tokio_multi(&plan, workers, false, body));
}

#[cfg(feature = "tokio-driver")]
pub fn run_tokio_multi_thread_paused<Body, Fut>(plan: Plan, workers: Option<usize>, body: Body)
where
    Body: FnOnce(TestCtx) -> Fut,
    Fut: Future<Output = ()>,
{
    finish(drive_tokio_multi(&plan, workers, true, body));
}

#[cfg(feature = "tokio-driver")]
fn drive_tokio<Body, Fut>(plan: &Plan, body: Body) -> TestReport
where
    Body: FnOnce(TestCtx) -> Fut,
    Fut: Future<Output = ()>,
{
    drive_tokio_inner(plan, false, body)
}

#[cfg(feature = "tokio-driver")]
fn drive_tokio_paused<Body, Fut>(plan: &Plan, body: Body) -> TestReport
where
    Body: FnOnce(TestCtx) -> Fut,
    Fut: Future<Output = ()>,
{
    drive_tokio_inner(plan, true, body)
}

#[cfg(feature = "tokio-driver")]
fn drive_tokio_inner<Body, Fut>(plan: &Plan, start_paused: bool, body: Body) -> TestReport
where
    Body: FnOnce(TestCtx) -> Fut,
    Fut: Future<Output = ()>,
{
    install_panic_hook_once();
    raise_fd_limit_once();
    let ctx = build_test_ctx(plan);
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
        let body_report = report_from(result);
        let teardown_panic = run_teardowns(&teardowns).await;
        disarm();
        combine_report(body_report, teardown_panic)
    })
}

#[cfg(feature = "tokio-driver")]
fn drive_tokio_multi<Body, Fut>(
    plan: &Plan,
    workers: Option<usize>,
    start_paused: bool,
    body: Body,
) -> TestReport
where
    Body: FnOnce(TestCtx) -> Fut,
    Fut: Future<Output = ()>,
{
    install_panic_hook_once();
    raise_fd_limit_once();
    let ctx = build_test_ctx(plan);
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
        let body_report = report_from(result);
        let teardown_panic = run_teardowns(&teardowns).await;
        disarm();
        combine_report(body_report, teardown_panic)
    })
}

fn combine_report(body: TestReport, teardown_panic: Option<CapturedPanic>) -> TestReport {
    match (body, teardown_panic) {
        (TestReport::Passed, Some(panic)) => TestReport::Failed(panic),
        (other, _) => other,
    }
}

#[cfg(feature = "test-prime")]
pub fn shared_prime_runtime() -> &'static PrimeRuntime {
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
fn drive_prime<Body, Fut>(plan: &Plan, body: Body) -> TestReport
where
    Body: FnOnce(TestCtx) -> Fut + Send + 'static,
    Fut: Future<Output = ()> + Send + 'static,
{
    install_panic_hook_once();
    raise_fd_limit_once();
    let ctx = build_test_ctx(plan);
    let teardowns = ctx.teardowns.clone();
    let runtime = shared_prime_runtime();
    let (sender, receiver) = sync_channel::<TestReport>(1);

    let task = async move {
        arm();
        let result = CatchUnwind { inner: body(ctx) }.await;
        let body_report = report_from(result);
        let teardown_panic = run_teardowns(&teardowns).await;
        disarm();
        let _ = sender.send(combine_report(body_report, teardown_panic));
    };

    match runtime.spawn_on_core(CoreId(0), Box::pin(task)) {
        Ok(()) => {}
        Err(SpawnError::InboxFull) => return failed("prime core 0 inbox full on dispatch"),
        Err(SpawnError::Disconnected) => return failed("prime core 0 disconnected"),
    }

    match receiver.recv_timeout(body_timeout()) {
        Ok(report) => report,
        Err(RecvTimeoutError::Timeout) => failed("body did not complete within the timeout"),
        Err(RecvTimeoutError::Disconnected) => {
            failed("worker dropped the completion channel without reporting")
        }
    }
}

#[cfg(feature = "test-prime")]
fn body_timeout() -> Duration {
    match std::env::var("PROXIMA_TEST_TIMEOUT_MS") {
        Ok(value) => value
            .parse::<u64>()
            .map_or_else(|_| Duration::from_secs(60), Duration::from_millis),
        Err(_) => Duration::from_secs(60),
    }
}

#[cfg(feature = "test-prime")]
fn failed(message: &str) -> TestReport {
    TestReport::Failed(CapturedPanic {
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
        assert!(matches!(
            drive_tokio(&Plan::new(), |_cx| async {}),
            TestReport::Passed
        ));
    }

    #[cfg(feature = "tokio-driver")]
    #[test]
    fn tokio_reports_panicking_body_with_message() {
        match drive_tokio(&Plan::new(), |_cx| async { panic!("boom-tokio") }) {
            TestReport::Failed(captured) => assert!(captured.message.contains("boom-tokio")),
            TestReport::Passed => panic!("expected the panicking body to be reported as failed"),
        }
    }

    #[cfg(feature = "test-prime")]
    #[test]
    fn prime_reports_passing_body() {
        assert!(matches!(
            drive_prime(&Plan::new(), |_cx| async {}),
            TestReport::Passed
        ));
    }

    #[cfg(feature = "test-prime")]
    #[test]
    fn prime_reports_panicking_body_with_message() {
        match drive_prime(&Plan::new(), |_cx| async { panic!("boom-prime") }) {
            TestReport::Failed(captured) => assert!(captured.message.contains("boom-prime")),
            TestReport::Passed => panic!("expected the panicking body to be reported as failed"),
        }
    }

    #[cfg(feature = "tokio-driver")]
    #[test]
    fn teardown_runs_on_pass_and_on_panic() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static RAN: AtomicUsize = AtomicUsize::new(0);

        let report = drive_tokio(&Plan::new(), |cx| async move {
            cx.defer(async {
                RAN.fetch_add(1, Ordering::SeqCst);
            });
        });
        assert!(matches!(report, TestReport::Passed));
        assert_eq!(RAN.load(Ordering::SeqCst), 1, "teardown must run on pass");

        let report = drive_tokio(&Plan::new(), |cx| async move {
            cx.defer(async {
                RAN.fetch_add(1, Ordering::SeqCst);
            });
            panic!("boom");
        });
        assert!(matches!(report, TestReport::Failed(_)));
        assert_eq!(
            RAN.load(Ordering::SeqCst),
            2,
            "teardown must run even when the body panics"
        );
    }
}
