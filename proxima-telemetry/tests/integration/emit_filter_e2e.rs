//! End-to-end: a real `Recorder` whose terminal is an `EmitFilterPipe` over an
//! `InMemoryPipe`, proving the compiled filter actually gates emits on the live
//! drain path (not just the resolver in isolation). Gated on `emit`.
//!
//! Also excluded under `loom`: a real Recorder uses proxima-core's Ring/
//! StaticRing internally, cfg-swapped to loom's mocked primitives
//! (forwarded via proxima-core/loom), which only work inside an actual
//! loom::model(...) closure that this plain #[test] doesn't provide.
#![cfg(all(feature = "emit", not(feature = "loom")))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;

use proxima_telemetry::emit::{Coord, EnvFilter};
use proxima_telemetry::level::Level;
use proxima_telemetry::pipes::{InMemoryPipe, TelemetryPipeExt};
use proxima_telemetry::recorder::Recorder;

// a RUST_LOG-shaped filter applied to a live recorder filters logs by
// (module_path, level) exactly as the directive says — the "match the old API"
// path proven end to end.
#[test]
fn env_filter_gates_logs_on_the_live_drain_path() {
    let sink = InMemoryPipe::new();
    // "proxima::h2=debug,error" -> debug-and-above for proxima::h2; the bare
    // `error` sets the global default (errors from any target survive).
    let compiled = EnvFilter::parse("proxima::h2=debug,error");
    let pipe = sink
        .clone()
        .emit_filter(Arc::new(compiled), Coord::from(Level::INFO));

    let recorder = Recorder::builder()
        .pipe(pipe)
        .core_count(1)
        .start()
        .expect("recorder build");

    // kept: info under proxima::h2 (info >= debug floor)
    recorder
        .log()
        .level(Level::INFO)
        .message("h2 info")
        .module_path("proxima::h2::frame")
        .emit();
    // dropped: info under downstream (hits ERROR default, info < error)
    recorder
        .log()
        .level(Level::INFO)
        .message("downstream info")
        .module_path("downstream::store")
        .emit();
    // kept: error under downstream (error >= ERROR default)
    recorder
        .log()
        .level(Level::ERROR)
        .message("downstream error")
        .module_path("downstream::store")
        .emit();

    recorder.drain();

    let logs = sink.logs();
    assert_eq!(
        logs.len(),
        2,
        "exactly the h2-info and downstream-error logs survive"
    );
    assert!(
        logs.iter()
            .any(|record| record.module_path == "proxima::h2::frame")
    );
    assert!(
        logs.iter()
            .any(|record| record.module_path == "downstream::store" && record.level == Level::ERROR)
    );
    assert!(
        !logs
            .iter()
            .any(|record| record.module_path == "downstream::store" && record.level == Level::INFO),
        "the below-default downstream info log was dropped"
    );
}
