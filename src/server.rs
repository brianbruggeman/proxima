//! `Server` — fluent terminal handle returned by `App::serve(...)`.
//!
//! Wraps the existing `Shutdown` (listener-loop lifecycle) plus an
//! `Arc<dyn ControlPlane>` (operations surface) into one type that:
//!
//! - is `Clone` via internal `Arc` — clones share control-plane state
//! - impls `ControlPlane` so trait method calls work uniformly
//! - impls `IntoFuture` so `let s = ...; s.await` runs until signal
//! - has explicit drive methods: `run_until_signal`, `join`, `drain`,
//!   `stop` for callers who want control over how the loop terminates
//!
//! Listener-loop ownership is "single-owner": the first call to a
//! drive method takes the `Shutdown` handle; subsequent calls (from
//! clones) become no-ops. Calling `Server::shutdown()` from any clone
//! signals the loop to exit cleanly.

use std::future::{Future, IntoFuture};
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use futures::{FutureExt, select};
use proxima_primitives::sync::Notify;

use crate::app::Shutdown;
use crate::control_plane::{ControlPlane, DynControlPlane, PipeStatus};
use crate::error::ProximaError;
use crate::telemetry::MetricsSnapshot;
use proxima_primitives::sync::shutdown::ShutdownReport;

/// Inner shared state — all `Server` clones point at the same `Arc`.
struct ServerInner {
    control: DynControlPlane,
    /// External shutdown trigger — any clone can fire this; the
    /// driving `run_until_signal` waits on it alongside the OS
    /// signal. Without this, `stop()` from a clone can't actually
    /// wake the awaiting task (the OS-signal future never fires).
    shutdown_notify: Arc<Notify>,
    /// `Some` until the first drive method consumes it. After that
    /// the listener loop has been driven by someone else; subsequent
    /// drive-method calls (from clones) become no-ops.
    ///
    /// WHY here:        single-owner listener-loop semantics require
    ///                  one caller to win the consume. Clone-able
    ///                  `Server` shares this inner state via Arc, so
    ///                  the take() must be serialized against other
    ///                  clones racing on the same drive.
    /// WHY NOT removable: alternatives are (a) make `Server` not
    ///                  Clone (rejected — clones are the operator
    ///                  surface), (b) use `AtomicPtr<Shutdown>`
    ///                  (introduces unsafe + Drop ordering hazard
    ///                  around the dropped-Shutdown auto-signal),
    ///                  (c) put the Shutdown behind an `Arc` and
    ///                  require ref-count == 1 to take (changes
    ///                  Drop semantics callers rely on). None are
    ///                  cleaner than the Mutex.
    /// WHY this is right: contention is zero in steady state — drive
    ///                  methods are called once at startup, never
    ///                  on the hot path. The lock is held for one
    ///                  `Option::take`. Poison is recovered (see
    ///                  `take_shutdown`); poison only means a prior
    ///                  panic took the value out, which is the
    ///                  state we want anyway.
    shutdown: Mutex<Option<Shutdown>>,
}

/// Fluent handle returned by `App::serve(...)`.
///
/// `Clone` produces a control-plane-only view of the same underlying
/// listener. Awaiting (`server.await`) runs until SIGTERM/SIGINT.
#[derive(Clone)]
pub struct Server {
    inner: Arc<ServerInner>,
}

impl Server {
    /// Wrap an existing `Shutdown` + control-plane reference into a
    /// fluent `Server`. Used by `App::serve(...)` and by tests.
    #[must_use]
    pub fn new(shutdown: Shutdown, control: DynControlPlane) -> Self {
        Self {
            inner: Arc::new(ServerInner {
                control,
                shutdown_notify: Arc::new(Notify::new()),
                shutdown: Mutex::new(Some(shutdown)),
            }),
        }
    }

    /// Block until SIGTERM/SIGINT (or `stop()` from any clone), then
    /// signal the listener to stop. In-flight requests continue to
    /// completion (or cancel via their own cancel Signal).
    /// Subsequent calls from clones are no-ops because the `Shutdown`
    /// was consumed.
    pub async fn run_until_signal(self) {
        let notify = self.inner.shutdown_notify.clone();
        select! {
            _ = shutdown_signal().fuse() => {}
            _ = notify.notified().fuse() => {}
        }
        if let Some(shutdown) = take_shutdown(&self.inner.shutdown) {
            shutdown.stop();
        }
    }

    /// Block until SIGTERM/SIGINT (or `stop()` from any clone), then
    /// run the full graceful drain (listener stop → wait for in-
    /// flight → broadcast per-core drop). Returns the report with
    /// per-core ack counts.
    pub async fn run_until_signal_with_drain(self) -> ShutdownReport {
        let notify = self.inner.shutdown_notify.clone();
        select! {
            _ = shutdown_signal().fuse() => {}
            _ = notify.notified().fuse() => {}
        }
        match take_shutdown(&self.inner.shutdown) {
            Some(shutdown) => shutdown.drain().await,
            None => ShutdownReport::default(),
        }
    }

    /// Stop the listener loop without waiting for a signal. Callable
    /// from any clone. Fires the shutdown-notify so a driving
    /// `run_until_signal` wakes up; also stops the Shutdown directly
    /// in case no one is driving (e.g. tests that don't await).
    pub fn stop(&self) {
        self.inner.shutdown_notify.notify_waiters();
        if let Some(shutdown) = take_shutdown(&self.inner.shutdown) {
            shutdown.stop();
        }
    }

    /// Graceful drain without waiting for a signal — useful in tests
    /// and in clients that have their own shutdown trigger (e.g. the
    /// `proxima ui` TUI's "stop the daemon" command).
    pub async fn drain(self) -> ShutdownReport {
        let shutdown = take_shutdown(&self.inner.shutdown);
        match shutdown {
            Some(s) => s.drain().await,
            None => ShutdownReport::default(),
        }
    }
}

/// Take the `Shutdown` out of the mutex, recovering from a poisoned
/// lock by accepting the inner state as-is. Poison only happens if a
/// panic interrupted a previous take; the `Option` is still valid.
fn take_shutdown(mutex: &Mutex<Option<Shutdown>>) -> Option<Shutdown> {
    match mutex.lock() {
        Ok(mut guard) => guard.take(),
        Err(poisoned) => poisoned.into_inner().take(),
    }
}

/// Awaiting a `Server` is equivalent to `.run_until_signal()`. Lets
/// callers write `App::builder()...serve()?.await?` in the common case.
impl IntoFuture for Server {
    type Output = Result<(), ProximaError>;
    type IntoFuture = Pin<Box<dyn Future<Output = Self::Output> + Send>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move {
            self.run_until_signal().await;
            Ok(())
        })
    }
}

/// Delegate every operation to the inner `ControlPlane`. Clones of
/// `Server` are first-class control-plane handles.
impl ControlPlane for Server {
    fn list_pipes<'lifetime>(
        &'lifetime self,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<PipeStatus>, ProximaError>> + Send + 'lifetime>>
    {
        self.inner.control.list_pipes()
    }

    fn status<'lifetime>(
        &'lifetime self,
        name: &'lifetime str,
    ) -> Pin<Box<dyn Future<Output = Result<PipeStatus, ProximaError>> + Send + 'lifetime>> {
        self.inner.control.status(name)
    }

    fn snapshot_metrics<'lifetime>(
        &'lifetime self,
    ) -> Pin<Box<dyn Future<Output = Result<MetricsSnapshot, ProximaError>> + Send + 'lifetime>>
    {
        self.inner.control.snapshot_metrics()
    }

    fn start<'lifetime>(
        &'lifetime self,
        name: &'lifetime str,
    ) -> Pin<Box<dyn Future<Output = Result<PipeStatus, ProximaError>> + Send + 'lifetime>> {
        self.inner.control.start(name)
    }

    fn stop<'lifetime>(
        &'lifetime self,
        name: &'lifetime str,
    ) -> Pin<Box<dyn Future<Output = Result<PipeStatus, ProximaError>> + Send + 'lifetime>> {
        self.inner.control.stop(name)
    }

    fn restart<'lifetime>(
        &'lifetime self,
        name: &'lifetime str,
    ) -> Pin<Box<dyn Future<Output = Result<PipeStatus, ProximaError>> + Send + 'lifetime>> {
        self.inner.control.restart(name)
    }

    fn reload<'lifetime>(
        &'lifetime self,
    ) -> Pin<Box<dyn Future<Output = Result<(), ProximaError>> + Send + 'lifetime>> {
        self.inner.control.reload()
    }

    fn shutdown<'lifetime>(
        &'lifetime self,
    ) -> Pin<Box<dyn Future<Output = Result<(), ProximaError>> + Send + 'lifetime>> {
        // Signal stop synchronously from any clone; the listener-owning
        // instance's drive method picks it up next iteration.
        Server::stop(self);
        Box::pin(async move { Ok(()) })
    }

    fn apply<'lifetime>(
        &'lifetime self,
        name: &'lifetime str,
        spec: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<PipeStatus, ProximaError>> + Send + 'lifetime>> {
        self.inner.control.apply(name, spec)
    }

    fn logs<'lifetime>(
        &'lifetime self,
        name: &'lifetime str,
        max_lines: Option<usize>,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<String>, ProximaError>> + Send + 'lifetime>> {
        self.inner.control.logs(name, max_lines)
    }
}

/// Runtime-agnostic shutdown-signal seam: wait for SIGTERM or SIGINT,
/// whichever fires first. Both `run_until_signal*` methods above call only
/// this name — every runtime-specific implementation lives in its own
/// adapter arm below, selected by a disjoint `cfg`, so no runtime symbol
/// ever appears in the generic serve path itself.
///
/// Runtimes are additive (`tokio` and `runtime-prime-reactor` may both be
/// linked at once — see their feature docs in `Cargo.toml`), so the arms
/// below are mutually exclusive by construction rather than an if/else:
/// exactly one is compiled for any feature combination, never zero, never
/// two. Precedence when both are linked: `tokio` wins, unchanged from
/// this function's behavior before the prime adapter existed (P14) — the
/// tokio arm's body is untouched. The prime arm is selected exactly when
/// tokio is absent, which is the shipped DEFAULT build (`serve-prime`, no
/// tokio) — the bug this seam fixes.
#[cfg(all(unix, feature = "tokio"))]
async fn shutdown_signal() {
    use tokio::signal::unix::{SignalKind, signal};

    let mut sigterm = match signal(SignalKind::terminate()) {
        Ok(s) => s,
        Err(_) => return,
    };
    let mut sigint = match signal(SignalKind::interrupt()) {
        Ok(s) => s,
        Err(_) => return,
    };
    select! {
        _ = sigterm.recv().fuse() => {}
        _ = sigint.recv().fuse() => {}
    }
}

#[cfg(all(not(unix), feature = "tokio"))]
async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
}

/// Prime adapter: real self-pipe + reactor-backed SIGINT/SIGTERM wait
/// (`prime::os::signal::wait_for_signal`, see that module for the design
/// and why it's a self-pipe over the reactor rather than a dedicated
/// thread). Selected whenever the tokio arms above aren't compiled but the
/// prime reactor is — the default build.
#[cfg(all(unix, not(feature = "tokio"), feature = "runtime-prime-reactor"))]
async fn shutdown_signal() {
    let _ = prime::os::signal::wait_for_signal().await;
}

/// No compiled-in OS signal source (neither tokio nor the prime reactor,
/// e.g. a feature-stripped build, or non-unix without tokio) — this arm
/// never resolves, and that is the honest, minimal-scope truth for a build
/// with no signal source at all, not a silent gap in a shipped default:
/// `run_until_signal` still terminates via an explicit `Server::stop()`/
/// `Shutdown::stop()` call or process kill.
#[cfg(not(any(feature = "tokio", all(unix, feature = "runtime-prime-reactor"))))]
async fn shutdown_signal() {
    core::future::pending::<()>().await;
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    // used only by the real-signal tests below, all of which require
    // `runtime-prime-reactor` (prime-path or tokio+prime-compat-path) — a
    // pure tokio-only build (no prime features at all, newly reachable now
    // that `tcp` no longer smuggles `serve-prime` in — see the umbrella
    // Cargo.toml's `tcp` doc) compiles none of them.
    #[cfg(feature = "runtime-prime-reactor")]
    use std::task::Poll;

    use super::*;
    use crate::control_plane::{PipeState, StaticControlPlane};

    fn fixture_control() -> DynControlPlane {
        Arc::new(StaticControlPlane::new(vec![PipeStatus {
            name: "echo".into(),
            state: PipeState::Running,
            uptime_ms: Some(1234),
            restart_count: 0,
            last_message: None,
        }]))
    }

    #[proxima::test]
    async fn clone_shares_control_plane_state() {
        // Construct a Shutdown via the test scaffold path — we don't
        // need a real listener; the Shutdown's stop just sends an
        // unused signal.
        let (_tx, rx) = futures::channel::oneshot::channel();
        let shutdown = Shutdown::for_test(rx);
        let server = Server::new(shutdown, fixture_control());
        let clone = server.clone();

        let listed_a = server.list_pipes().await.expect("list a");
        let listed_b = clone.list_pipes().await.expect("list b");
        assert_eq!(listed_a.len(), listed_b.len());
        assert_eq!(listed_a[0].name, "echo");
        assert_eq!(listed_b[0].name, "echo");
    }

    #[proxima::test]
    async fn stop_from_clone_signals_listener() {
        let (tx, mut rx) = futures::channel::oneshot::channel();
        let shutdown = Shutdown::for_test_with_tx(tx);
        let server = Server::new(shutdown, fixture_control());
        let clone = server.clone();

        clone.stop();
        // Original receiver fires.
        let outcome = proxima_core::time::timeout(std::time::Duration::from_millis(100), &mut rx)
            .await
            .expect("oneshot fires within 100ms");
        assert!(outcome.is_ok(), "shutdown sender fired");
    }

    #[proxima::test]
    async fn stop_is_idempotent_across_clones() {
        let (tx, _rx) = futures::channel::oneshot::channel();
        let shutdown = Shutdown::for_test_with_tx(tx);
        let server = Server::new(shutdown, fixture_control());
        let clone = server.clone();

        server.stop();
        clone.stop(); // already consumed — no-op, shouldn't panic
    }

    // ── real signal-driven shutdown ─────────────────────────────────────
    //
    // These raise a genuine OS signal against the test process and assert
    // `run_until_signal` actually resolves — the deliverable for "the
    // prime default build has no real signal shutdown". They need
    // `cargo nextest run` (one process per test, this workspace's default
    // runner): raising SIGINT/SIGTERM in a shared `cargo test` process
    // would race whatever other test is running at delivery time.
    //
    // Both tests manually poll `run_until_signal()` once, off a no-op
    // waker, before raising. Signal-handler install happens synchronously
    // inside the adapter's first poll (before its first real await point,
    // on both the prime and tokio arms) — SIGINT/SIGTERM's default
    // disposition terminates the process, so the handler must already be
    // registered or `raise` below kills the test instead of the server.

    /// The default (prime, no tokio) build's shutdown-signal adapter is
    /// `prime::os::signal::wait_for_signal` (see that module). Before the
    /// fix this arm was `core::future::pending` and this test would time
    /// out — it is the direct proof of the bug and the fix.
    #[cfg(all(unix, feature = "runtime-prime-reactor", not(feature = "tokio")))]
    #[proxima::test]
    async fn run_until_signal_resolves_on_real_sigint_prime_path() {
        let (tx, mut rx) = futures::channel::oneshot::channel();
        let shutdown = Shutdown::for_test_with_tx(tx);
        let server = Server::new(shutdown, fixture_control());

        let mut driving = std::pin::pin!(server.run_until_signal());
        // scoped so the (`!Send`) `Context`/`Waker` are dropped before the
        // `.await` below — the generated `#[proxima::test]` future must
        // stay `Send` across every await point.
        {
            let waker = std::task::Waker::noop();
            let mut context = std::task::Context::from_waker(waker);
            assert_eq!(
                driving.as_mut().poll(&mut context),
                Poll::Pending,
                "run_until_signal must park on the OS signal, not resolve immediately"
            );
        }

        // SAFETY: raising a signal against our own process is the
        // documented use of raise(2); the handler installed above only
        // performs an async-signal-safe write(2) (prime::os::signal).
        unsafe {
            libc::raise(libc::SIGINT);
        }

        proxima_core::time::timeout(std::time::Duration::from_secs(5), driving)
            .await
            .expect("run_until_signal must resolve within 5s of a real SIGINT");

        let outcome = proxima_core::time::timeout(std::time::Duration::from_millis(100), &mut rx)
            .await
            .expect("Shutdown::stop() must fire the shutdown sender");
        assert!(outcome.is_ok(), "shutdown sender fired");
    }

    /// SIGTERM sibling of the SIGINT test above — the prime adapter must
    /// handle both.
    #[cfg(all(unix, feature = "runtime-prime-reactor", not(feature = "tokio")))]
    #[proxima::test]
    async fn run_until_signal_resolves_on_real_sigterm_prime_path() {
        let (tx, mut rx) = futures::channel::oneshot::channel();
        let shutdown = Shutdown::for_test_with_tx(tx);
        let server = Server::new(shutdown, fixture_control());

        let mut driving = std::pin::pin!(server.run_until_signal());
        // scoped so the (`!Send`) `Context`/`Waker` are dropped before the
        // `.await` below — the generated `#[proxima::test]` future must
        // stay `Send` across every await point.
        {
            let waker = std::task::Waker::noop();
            let mut context = std::task::Context::from_waker(waker);
            assert_eq!(
                driving.as_mut().poll(&mut context),
                Poll::Pending,
                "run_until_signal must park on the OS signal, not resolve immediately"
            );
        }

        // SAFETY: see the SIGINT test above.
        unsafe {
            libc::raise(libc::SIGTERM);
        }

        proxima_core::time::timeout(std::time::Duration::from_secs(5), driving)
            .await
            .expect("run_until_signal must resolve within 5s of a real SIGTERM");

        let outcome = proxima_core::time::timeout(std::time::Duration::from_millis(100), &mut rx)
            .await
            .expect("Shutdown::stop() must fire the shutdown sender");
        assert!(outcome.is_ok(), "shutdown sender fired");
    }

    /// Mirrors the prime-path test above on the tokio adapter — proves
    /// P14 parity: the tokio arm's real-signal behavior is unchanged by
    /// this fix (its body is untouched; only its name and cfg predicate
    /// moved to make room for the prime arm).
    #[cfg(all(unix, feature = "tokio", feature = "runtime-prime-reactor"))]
    #[proxima::test(runtime = "tokio")]
    async fn run_until_signal_resolves_on_real_sigint_tokio_path() {
        let (tx, mut rx) = futures::channel::oneshot::channel();
        let shutdown = Shutdown::for_test_with_tx(tx);
        let server = Server::new(shutdown, fixture_control());

        let mut driving = std::pin::pin!(server.run_until_signal());
        // scoped so the (`!Send`) `Context`/`Waker` are dropped before the
        // `.await` below — the generated `#[proxima::test]` future must
        // stay `Send` across every await point.
        {
            let waker = std::task::Waker::noop();
            let mut context = std::task::Context::from_waker(waker);
            assert_eq!(
                driving.as_mut().poll(&mut context),
                Poll::Pending,
                "run_until_signal must park on the OS signal, not resolve immediately"
            );
        }

        // SAFETY: see the prime-path test above.
        unsafe {
            libc::raise(libc::SIGINT);
        }

        proxima_core::time::timeout(std::time::Duration::from_secs(5), driving)
            .await
            .expect("run_until_signal must resolve within 5s of a real SIGINT");

        let outcome = proxima_core::time::timeout(std::time::Duration::from_millis(100), &mut rx)
            .await
            .expect("Shutdown::stop() must fire the shutdown sender");
        assert!(outcome.is_ok(), "shutdown sender fired");
    }
}
