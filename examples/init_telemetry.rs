#![allow(clippy::unwrap_used, clippy::expect_used)]

//! The ordinary way to turn on telemetry: `proxima::init_telemetry()`.
//! Zero required arguments — console output, `RUST_LOG` honored, ambient
//! recorder registered, background drain already running. Both
//! `proxima_telemetry::*` macros and bridged `tracing::*` events land on the
//! same console.
//!
//! Run: `RUST_LOG=info cargo run --example init_telemetry --features tracing-init`

fn main() {
    proxima::init_telemetry().expect("install console telemetry");

    proxima_telemetry::info!("service starting");
    proxima_telemetry::error!("this always shows, even with RUST_LOG unset (default floor)");
    tracing::warn!("tracing:: events are bridged into the same recorder");

    // the drain thread is event-driven; give it a moment before exit so this
    // run's records are visible rather than racing process shutdown.
    std::thread::sleep(std::time::Duration::from_millis(200));
}
