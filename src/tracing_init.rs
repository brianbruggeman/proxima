//! Crate-root telemetry init. [`init_telemetry`] is the one-liner a regular
//! user reaches for: console output, `RUST_LOG` honored, ambient recorder
//! registered, drain already running, zero required arguments.
//!
//! ```no_run
//! proxima::init_telemetry().expect("install console telemetry");
//!
//! proxima_telemetry::info!("service starting");
//! tracing::warn!("tracing:: events land on the same console too");
//! ```
//!
//! [`init_telemetry_with`] is the escape hatch for choosing the format.
//! [`init_tracing`]/[`init_tracing_default`] are the original names, kept
//! working as thin delegations for source compatibility — prefer the names
//! above in new code.

use std::sync::Arc;

use proxima_telemetry::error::Error;
use proxima_telemetry::recorder::Recorder;

#[cfg(feature = "tracing-init")]
use tracing_subscriber::EnvFilter;
#[cfg(feature = "tracing-init")]
use tracing_subscriber::layer::SubscriberExt;

#[cfg(feature = "tracing-init")]
use proxima_telemetry::export::{
    Formatter, install_console_logging, install_console_logging_with, set_default_recorder,
};
#[cfg(feature = "tracing-init")]
use proxima_telemetry::tracing_bridge::TracingLayer;

/// Console text vs JSON — maps onto [`proxima_telemetry::export::Formatter`].
#[derive(Debug, Clone, Copy, Default)]
pub enum LogFormat {
    #[default]
    Human,
    Json,
}

#[cfg(feature = "tracing-init")]
impl LogFormat {
    const fn to_formatter(self) -> Formatter {
        match self {
            Self::Human => Formatter::Text,
            Self::Json => Formatter::Json,
        }
    }
}

/// The one-liner: level-routed console logging (trace/debug/info → stdout,
/// warn/error → stderr), `RUST_LOG`-filtered, registered as the process
/// default so both `proxima_telemetry::{info!, warn!, ...}` and bridged
/// `tracing::*` callsites resolve to it, with a background thread already
/// draining it. No arguments, no assembly required — delegates to
/// [`proxima_telemetry::export::install_console_logging`], which does the
/// real work (recorder + exporter + bridge + drain thread) so this crate
/// carries no parallel implementation of any of it.
///
/// The returned `Arc<Recorder>` does not need to be held for logging to keep
/// working — [`proxima_telemetry::export::set_default_recorder`] and the
/// background drain thread each hold their own strong reference. Keep it if
/// you want to call `.drain()` yourself before process exit.
///
/// # Errors
/// Propagates recorder-build or drain-thread-spawn failures — never returns
/// `Ok` without a working console recorder installed.
#[cfg(feature = "tracing-init")]
pub fn init_telemetry() -> Result<Arc<Recorder>, Error> {
    install_console_logging()
}

/// [`init_telemetry`] with an explicit [`LogFormat`] (e.g. `LogFormat::Json`
/// for structured console output).
///
/// # Errors
/// Propagates recorder-build or drain-thread-spawn failures.
#[cfg(feature = "tracing-init")]
pub fn init_telemetry_with(format: LogFormat) -> Result<Arc<Recorder>, Error> {
    install_console_logging_with(format.to_formatter())
}

#[cfg(not(feature = "tracing-init"))]
pub fn init_telemetry() -> Result<Arc<Recorder>, Error> {
    eprintln!(
        "proxima::init_telemetry called without the `tracing-init` feature; enable it (--features tracing-init) for console telemetry"
    );
    Err(Error::InvalidInput)
}

#[cfg(not(feature = "tracing-init"))]
pub fn init_telemetry_with(_format: LogFormat) -> Result<Arc<Recorder>, Error> {
    eprintln!(
        "proxima::init_telemetry_with called without the `tracing-init` feature; enable it (--features tracing-init) for console telemetry"
    );
    Err(Error::InvalidInput)
}

/// Deprecated alias for [`init_telemetry_with`] — `format` is now honored
/// (it used to be silently dropped on the floor, the same defect class as
/// the `NullPipe` bug this name was originally reported for).
#[deprecated(note = "renamed: use proxima::init_telemetry_with (or init_telemetry() for defaults)")]
pub fn init_tracing_default(format: LogFormat) -> Result<Arc<Recorder>, Error> {
    init_telemetry_with(format)
}

/// Bridge `tracing::` events into an already-built [`Recorder`] the caller
/// owns: registers it as the process-default (so `proxima_telemetry`'s emit
/// macros resolve to it too), installs the `tracing` subscriber bridge
/// behind the same `RUST_LOG`-driven filter [`init_telemetry`] uses, and
/// spawns a background drain thread so buffered records reach the
/// recorder's sink (the rings are multi-consumer, so this is safe even if
/// the caller also drains the recorder itself elsewhere).
///
/// Not deprecated — unlike [`init_tracing_default`] this takes a
/// caller-supplied recorder, a distinct capability
/// [`init_telemetry`]/[`init_telemetry_with`] cannot express (they always
/// build their own). `format` has no effect here: the recorder's sink
/// formatting was already fixed when the caller built it, before this
/// function ever saw it — kept for signature compatibility.
///
/// # Errors
/// Returns an error if a global `tracing` subscriber is already installed,
/// or if the background drain thread cannot be spawned.
#[cfg(feature = "tracing-init")]
pub fn init_tracing(recorder: Arc<Recorder>, _format: LogFormat) -> Result<(), Error> {
    set_default_recorder(Arc::clone(&recorder));

    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn,proxima=info"));
    let layer = TracingLayer::new(Arc::clone(&recorder));
    let subscriber = tracing_subscriber::registry().with(filter).with(layer);
    tracing::subscriber::set_global_default(subscriber)
        .map_err(|error| Error::GlobalSubscriberAlreadySet(error.to_string()))?;

    let pump = Arc::clone(&recorder);
    std::thread::Builder::new()
        .name("proxima-tracing-init-drain".to_string())
        .spawn(move || pump.run_drain_loop())
        .map_err(|error| Error::ThreadSpawn(error.to_string()))?;

    Ok(())
}

#[cfg(not(feature = "tracing-init"))]
pub fn init_tracing(_recorder: Arc<Recorder>, _format: LogFormat) -> Result<(), Error> {
    eprintln!(
        "proxima::init_tracing called without the `tracing-init` feature; enable it (--features tracing-init) to install the bridge"
    );
    Err(Error::InvalidInput)
}
