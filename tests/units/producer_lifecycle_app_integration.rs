#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]
//! Integration test for the `App::source` / `ProducerLifecycle` surface
//! (proxima-pipe TARGET 4 — replaces `App::spawn_producer_background_tasks`).
//!
//! These tests exercise the integration point at src/app.rs where sources
//! registered via `App::source` get spawned onto a `ProducerLifecycle` (the
//! same construction `run_until_signal` does internally, driven here
//! directly so the test doesn't need a live listener). The standalone
//! module is unit-tested in proxima-pipe/src/lifecycle.rs; this file
//! validates the App-level surface that consumers actually call.

use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use proxima::app::App;
use proxima_core::signal::Signal;
use proxima_primitives::pipe::ProximaError;
use proxima_primitives::pipe::SendPipe;

/// Test source that records how many times it was called and, if
/// `observe_cancel` is set, waits for the passed-in `Signal` to fire before
/// returning.
struct CountingSource {
    ran: Arc<AtomicUsize>,
    observe_cancel: bool,
}

impl SendPipe for CountingSource {
    type In = Signal;
    type Out = ();
    type Err = ProximaError;

    fn call(&self, cancel: Signal) -> impl Future<Output = Result<(), ProximaError>> + Send {
        let ran = self.ran.clone();
        let observe_cancel = self.observe_cancel;
        async move {
            if observe_cancel {
                cancel.fired().await;
            }
            ran.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }
}

fn spawn_all(
    app: &App,
    lifecycle: &mut proxima_primitives::pipe::ProducerLifecycle,
) {
    for name in app.sources().map(str::to_string).collect::<Vec<_>>() {
        let handle = app.lookup_source(&name).expect("registered source");
        lifecycle.spawn_from_source(&name, &handle);
    }
}

#[proxima::test]
async fn empty_app_spawns_lifecycle_with_zero_tasks() {
    let app = App::new().expect("App::new");
    let mut lifecycle = proxima_primitives::pipe::ProducerLifecycle::new();
    spawn_all(&app, &mut lifecycle);
    assert_eq!(lifecycle.task_count(), 0);
    let report = lifecycle.shutdown(Duration::from_millis(100)).await;
    assert_eq!(report.total, 0);
    assert_eq!(report.drained, 0);
}

#[proxima::test]
async fn app_with_one_source_spawns_and_drains_it() {
    let ran = Arc::new(AtomicUsize::new(0));

    let mut app = App::new().expect("App::new");
    app.source(
        "heartbeat",
        proxima_primitives::pipe::into_source_handle(CountingSource {
            ran: ran.clone(),
            observe_cancel: false,
        }),
    );

    let mut lifecycle = proxima_primitives::pipe::ProducerLifecycle::new();
    spawn_all(&app, &mut lifecycle);
    assert_eq!(lifecycle.task_count(), 1);

    let names: Vec<&str> = lifecycle.spawned_task_names().collect();
    assert!(
        names.contains(&"heartbeat"),
        "expected heartbeat in {names:?}"
    );

    let report = lifecycle.shutdown(Duration::from_secs(1)).await;
    assert_eq!(report.total, 1);
    assert_eq!(report.drained, 1);
    assert_eq!(report.panics, 0);
    assert_eq!(ran.load(Ordering::SeqCst), 1);
}

#[proxima::test]
async fn parent_cancellation_token_propagates_through_app_lifecycle_spawn() {
    let parent = Signal::new();
    let ran = Arc::new(AtomicUsize::new(0));

    let mut app = App::new().expect("App::new");
    app.source(
        "watcher",
        proxima_primitives::pipe::into_source_handle(CountingSource {
            ran: ran.clone(),
            observe_cancel: true,
        }),
    );

    let mut lifecycle = proxima_primitives::pipe::ProducerLifecycle::with_parent_signal(&parent);
    spawn_all(&app, &mut lifecycle);
    assert_eq!(lifecycle.task_count(), 1);

    // cancel via the parent — the source observes it cooperatively (the
    // real signal, not a wrapper's) and returns.
    parent.fire();

    let report = lifecycle.shutdown(Duration::from_secs(1)).await;
    assert_eq!(report.total, 1);
    assert_eq!(report.drained, 1);
    assert_eq!(
        ran.load(Ordering::SeqCst),
        1,
        "watcher source should have observed cancel via the inherited signal"
    );
}

#[proxima::test]
async fn handlers_and_sources_are_disjoint_registries() {
    // A registered handler (mounted via `App::pipe`) never contributes to
    // the source lifecycle — `pipes` and `sources` are separate maps
    // (TARGET 4), unlike the old model where a served Pipe could also
    // declare `background_tasks()`.
    use proxima::load::Spec;
    use proxima::pipe::PipeHandle;
    use proxima::request::{Request, Response};

    struct SilentHandler;
    impl SendPipe for SilentHandler {
        type In = Request<bytes::Bytes>;
        type Out = Response<bytes::Bytes>;
        type Err = ProximaError;

        fn call(
            &self,
            _request: Request<bytes::Bytes>,
        ) -> impl Future<Output = Result<Response<bytes::Bytes>, ProximaError>> + Send {
            async { Ok(Response::ok("")) }
        }
    }
    let silent: PipeHandle = Arc::new(SilentHandler) as PipeHandle;

    let mut app = App::new().expect("App::new");
    app.pipe("silent", Spec::Handle(silent))
        .await
        .expect("register silent");

    let mut lifecycle = proxima_primitives::pipe::ProducerLifecycle::new();
    spawn_all(&app, &mut lifecycle);
    assert_eq!(
        lifecycle.task_count(),
        0,
        "a registered handler contributes no sources"
    );
    let report = lifecycle.shutdown(Duration::from_millis(100)).await;
    assert_eq!(report.total, 0);
}
