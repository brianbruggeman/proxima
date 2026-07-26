#![allow(clippy::unwrap_used, clippy::expect_used)]

//! The ordinary way to turn on telemetry: `proxima::init_telemetry()`.
//! Zero required arguments, **zero required features** — console output,
//! `RUST_LOG` honored, ambient recorder registered, background drain
//! already running, on a plain `cargo run --example init_telemetry`.
//!
//! With `--features tracing-init` also enabled, events from the `tracing`
//! crate itself are additionally bridged into the same recorder (run with
//! `--features tracing-init` to see the extra line below).
//!
//! Run: `RUST_LOG=info cargo run --example init_telemetry`

fn main() {
    proxima::init_telemetry().expect("install console telemetry");

    proxima_telemetry::info!("service starting");
    proxima_telemetry::error!("this always shows, even with RUST_LOG unset (default floor)");

    #[cfg(feature = "tracing-init")]
    tracing::warn!("tracing:: events are bridged into the same recorder (needs tracing-init)");
    #[cfg(not(feature = "tracing-init"))]
    println!(
        "(tracing:: events are NOT bridged without --features tracing-init; \
         proxima_telemetry:: events above still work)"
    );

    // the drain thread is event-driven; give it a moment before exit so this
    // run's records are visible rather than racing process shutdown.
    std::thread::sleep(std::time::Duration::from_millis(200));
}
