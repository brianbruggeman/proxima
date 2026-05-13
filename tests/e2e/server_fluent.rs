//! Phase 3a: `App::serve()` returns a `Server` that satisfies all three
//! drive shapes: awaiting (IntoFuture), explicit method, and clone-
//! and-control. Tests verify each shape works end-to-end against a
//! real HTTP listener bound to an ephemeral port.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]
#![cfg(feature = "http1")]

use std::time::Duration;

use proxima::control_plane::ControlPlane;
use proxima::{App, RunConfig};

async fn build_app() -> App {
    App::new().expect("app")
}

#[proxima::test(flavor = "multi_thread", worker_threads = 2)]
async fn server_run_until_signal_with_explicit_stop() {
    let app = build_app().await;
    let config = RunConfig::http("127.0.0.1:0".parse().unwrap());
    let server = app.serve(config).await.expect("serve");

    // Clone the server. The clone shares control-plane state; either
    // can call .stop() to break the listener loop.
    let stopper = server.clone();

    let drive = tokio::spawn(async move { server.run_until_signal().await });
    // Let the listener bind.
    tokio::time::sleep(Duration::from_millis(50)).await;
    stopper.stop();
    // Drive task returns once the loop exits.
    tokio::time::timeout(Duration::from_millis(500), drive)
        .await
        .expect("drive completes within 500ms")
        .expect("drive task succeeds");
}

#[proxima::test(flavor = "multi_thread", worker_threads = 2)]
async fn server_clone_is_control_plane() {
    let app = build_app().await;
    let config = RunConfig::http("127.0.0.1:0".parse().unwrap());
    let server = app.serve(config).await.expect("serve");

    // Clone exposes ControlPlane methods — currently a read-only stub
    // (StaticControlPlane). list_pipes returns empty by default.
    let handle = server.clone();
    let listed = handle.list_pipes().await.expect("list");
    assert!(listed.is_empty());

    server.stop();
}

#[proxima::test(flavor = "multi_thread", worker_threads = 2)]
async fn server_into_future_runs_until_stop_from_clone() {
    let app = build_app().await;
    let config = RunConfig::http("127.0.0.1:0".parse().unwrap());
    let server = app.serve(config).await.expect("serve");
    let stopper = server.clone();

    // IntoFuture: awaiting Server is .run_until_signal().
    let drive = tokio::spawn(async move {
        let _ = server.await;
    });
    tokio::time::sleep(Duration::from_millis(50)).await;
    stopper.stop();
    tokio::time::timeout(Duration::from_millis(500), drive)
        .await
        .expect("drive completes within 500ms")
        .expect("drive task succeeds");
}

#[proxima::test(flavor = "multi_thread", worker_threads = 2)]
async fn server_drain_returns_report_when_consumed() {
    let app = build_app().await;
    let config = RunConfig::http("127.0.0.1:0".parse().unwrap());
    let server = app.serve(config).await.expect("serve");

    // Drain consumes the Shutdown; the report captures per-core acks.
    // We just verify it returns without panicking — semantics are
    // covered by existing shutdown tests.
    let _report = server.drain().await;
}

#[proxima::test(flavor = "multi_thread", worker_threads = 2)]
async fn http_listener_typed_config_drives_serve() {
    use proxima::settings::HttpListener;

    let app = build_app().await;
    // Typed builder; passes through Into<RunConfig> at the boundary.
    let server = app
        .serve(HttpListener::http("127.0.0.1:0".parse().unwrap()))
        .await
        .expect("serve via typed listener");

    let stopper = server.clone();
    let drive = tokio::spawn(async move { server.run_until_signal().await });
    tokio::time::sleep(Duration::from_millis(50)).await;
    stopper.stop();
    tokio::time::timeout(Duration::from_millis(500), drive)
        .await
        .expect("drive completes")
        .expect("drive task succeeds");
}

#[proxima::test(flavor = "multi_thread", worker_threads = 2)]
async fn composed_chain_flows_through_app_pipe() {
    use proxima::settings::{BearerAuth, Composable, HttpUpstream, RateLimit};

    let mut app = App::new().expect("app");
    // Reading top-down = execution order: bearer auth fires first,
    // then rate-limit, then the http upstream handles the request.
    let chain = BearerAuth::allow_tokens(["t-1", "t-2"])
        .then(RateLimit::token_bucket(100, 50))
        .then(HttpUpstream::url("https://backend.internal"));
    let handle = app.pipe("api", chain).await.expect("register chain");
    assert!(app.pipes().contains_key("api"));
    let _ = handle;
}

#[proxima::test(flavor = "multi_thread", worker_threads = 2)]
async fn http_upstream_typed_config_flows_through_app_pipe() {
    use proxima::settings::HttpUpstream;

    let mut app = App::new().expect("app");
    // Typed builder; serializes to the `type = "http"` registry-entry
    // shape the existing load() factory dispatch already understands.
    let upstream = HttpUpstream::builder()
        .url("https://example.com")
        .timeout(Duration::from_secs(5))
        .build();
    let handle = app
        .pipe("backend", upstream)
        .await
        .expect("register backend");
    // PipeHandle round-tripped — registry got the named pipe.
    assert!(app.pipes().contains_key("backend"));
    let _ = handle;
}

#[cfg(unix)]
#[proxima::test(flavor = "multi_thread", worker_threads = 2)]
async fn http_uds_listener_typed_config_drives_serve() {
    use proxima::settings::HttpUdsListener;
    use tempfile::tempdir;

    let dir = tempdir().expect("tempdir");
    let socket = dir.path().join("proxima.test.sock");

    let app = build_app().await;
    let server = app
        .serve(HttpUdsListener::local(socket.clone()))
        .await
        .expect("serve via typed uds listener");

    // app.serve() blocks (via run_until_signal) until the uds listener is
    // actually bound, so the socket must already exist here.
    assert!(socket.exists(), "uds socket appears at {:?}", socket);

    let stopper = server.clone();
    let drive = tokio::spawn(async move { server.run_until_signal().await });
    tokio::time::sleep(Duration::from_millis(50)).await;
    stopper.stop();
    let _ = tokio::time::timeout(Duration::from_millis(500), drive).await;
}
