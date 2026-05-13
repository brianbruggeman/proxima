#[cfg(feature = "tracing-init")]
use std::sync::Arc;
#[cfg(feature = "tracing-init")]
use tracing_subscriber::EnvFilter;
#[cfg(feature = "tracing-init")]
use tracing_subscriber::layer::SubscriberExt;
#[cfg(feature = "tracing-init")]
use tracing_subscriber::util::SubscriberInitExt;

#[cfg(feature = "tracing-init")]
use proxima_telemetry::recorder::Recorder;
#[cfg(feature = "tracing-init")]
use proxima_telemetry::tracing_bridge::TracingLayer;

#[derive(Debug, Clone, Copy, Default)]
pub enum LogFormat {
    #[default]
    Human,
    Json,
}

#[cfg(feature = "tracing-init")]
pub fn init_tracing(recorder: Arc<Recorder>, _format: LogFormat) {
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn,proxima=info"));
    let layer = TracingLayer::new(recorder);
    let _ = tracing_subscriber::registry()
        .with(filter)
        .with(layer)
        .try_init();
}

#[cfg(feature = "tracing-init")]
pub fn init_tracing_default(format: LogFormat) {
    use proxima_telemetry::pipes::NullPipe;

    let recorder = Recorder::builder()
        .pipe(NullPipe::new())
        .core_count(1)
        .start();
    match recorder {
        Ok(recorder) => init_tracing(Arc::new(recorder), format),
        Err(_) => {}
    }
}

#[cfg(not(feature = "tracing-init"))]
pub fn init_tracing(_recorder: std::sync::Arc<()>, _format: LogFormat) {
    eprintln!(
        "proxima::init_tracing called without the `tracing-init` feature; enable it to install the bridge"
    );
}

#[cfg(not(feature = "tracing-init"))]
pub fn init_tracing_default(_format: LogFormat) {
    eprintln!(
        "proxima::init_tracing_default called without the `tracing-init` feature; enable it to install the bridge"
    );
}
