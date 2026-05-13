//! Composable exporter setup + the process default recorder.
//!
//! [`Exporter`] is a composable sink value — `Exporter::stdout()` / `stderr()` /
//! `file(path)` / `writer(w)` — rendered by a [`Formatter`]. The builder gains
//! [`RecorderBuilder::export`] (lowers the sink to a [`crate::pipes::FormatterPipe`])
//! and [`RecorderBuilder::install`] (build + register as the process default so
//! emit sites with no explicit recorder resolve via [`default_recorder`]).
//!
//! This is the spine: the http/OTLP sinks, `.strategy()`/`.retry()` resilience,
//! and the `[[telemetry.export]]` conflaguration mirror compose onto it as
//! further stages. Tier T2 (std — io + arc-swap).

use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::sync::Arc;
use std::io::Write;
use std::path::PathBuf;

use arc_swap::ArcSwapOption;
use bon::Builder;
use conflaguration::{Settings, Validate, ValidationMessage};
use serde::{Deserialize, Serialize};

use crate::config::{ExporterChoice, OverflowPolicy};
use crate::error::Error;
use crate::pipes::{
    FormatterPipe, LogFormat, NullPipe, TelemetryPipeHandle, into_telemetry_handle,
};
use crate::recorder::{HasPipe, NoPipe, Recorder, RecorderBuilder};

static DEFAULT_RECORDER: ArcSwapOption<Recorder> = ArcSwapOption::const_empty();

/// Register the process-wide default recorder (also done by
/// [`RecorderBuilder::install`]). Emit sites with no explicit recorder resolve
/// to this — the "zero-wiring" path.
pub fn set_default_recorder(recorder: Arc<Recorder>) {
    // the ambient recorder IS the `#[instrument]` target, and its drain exports
    // the per-name duration histograms — so installing it is subscribing a
    // span-metric consumer. Open the gate here so `#[instrument]` durations record
    // out of the box; otherwise the histogram stays empty until a manual
    // enable_span_metrics(), which every install path forgot (mirrors capture()).
    #[cfg(feature = "instrument-metrics")]
    recorder.enable_span_metrics();
    DEFAULT_RECORDER.store(Some(recorder));
}

/// The process-wide default recorder, if one has been installed.
#[must_use]
pub fn default_recorder() -> Option<Arc<Recorder>> {
    DEFAULT_RECORDER.load_full()
}

/// How records render at a sink. The OTLP slice adds `Otlp` (protobuf) and
/// `OtlpJson` (the OTLP-debug console form).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum Formatter {
    /// Pretty single-line console text.
    #[default]
    Text,
    /// One JSON object per record (proxima's own shape).
    Json,
}

impl Formatter {
    const fn to_log_format(self) -> LogFormat {
        match self {
            Self::Text => LogFormat::Human,
            Self::Json => LogFormat::Json,
        }
    }
}

enum Sink {
    Stdout,
    Stderr,
    /// Level-routed console: trace/debug/info → stdout, warn/error → stderr.
    StdSplit,
    File(PathBuf),
    /// Size-triggered rolling file: `path`, roll threshold, rolled-file retention.
    RotatingFile {
        path: PathBuf,
        max_bytes: u64,
        max_files: usize,
    },
    Writer(Box<dyn Write + Send + Sync>),
    Handle(TelemetryPipeHandle),
}

/// A file [`Write`] sink that rolls itself once the active file exceeds
/// `max_bytes`: `path` → `path.1`, `path.1` → `path.2`, ... up to `path.max_files`,
/// dropping whatever was at `path.max_files`. Synchronous, checked on every
/// write — no background thread, matching the plain [`Sink::File`] write model.
struct RotatingFileWriter {
    path: PathBuf,
    max_bytes: u64,
    max_files: usize,
    file: std::fs::File,
    written_bytes: u64,
}

impl RotatingFileWriter {
    fn open(path: PathBuf, max_bytes: u64, max_files: usize) -> std::io::Result<Self> {
        let file = std::fs::File::create(&path)?;
        Ok(Self {
            path,
            max_bytes: max_bytes.max(1),
            max_files: max_files.max(1),
            file,
            written_bytes: 0,
        })
    }

    fn rolled_path(&self, index: usize) -> PathBuf {
        let mut file_name = self.path.clone().into_os_string();
        file_name.push(format!(".{index}"));
        PathBuf::from(file_name)
    }

    fn rotate(&mut self) -> std::io::Result<()> {
        let oldest = self.rolled_path(self.max_files);
        if oldest.exists() {
            std::fs::remove_file(&oldest)?;
        }
        for index in (1..self.max_files).rev() {
            let from = self.rolled_path(index);
            if from.exists() {
                std::fs::rename(&from, self.rolled_path(index + 1))?;
            }
        }
        std::fs::rename(&self.path, self.rolled_path(1))?;
        self.file = std::fs::File::create(&self.path)?;
        self.written_bytes = 0;
        Ok(())
    }
}

impl Write for RotatingFileWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let would_overflow =
            self.written_bytes > 0 && self.written_bytes + buf.len() as u64 > self.max_bytes;
        if would_overflow {
            self.rotate()?;
        }
        self.file.write_all(buf)?;
        self.written_bytes += buf.len() as u64;
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.file.flush()
    }
}

/// A composable exporter — a sink plus a [`Formatter`]. Compose onto a recorder
/// with [`RecorderBuilder::export`]; it lowers to a [`crate::pipes::FormatterPipe`]
/// over the sink's writer (RISC — the writer pipe already exists).
pub struct Exporter {
    sink: Sink,
    format: Formatter,
    capacity_bytes: usize,
}

impl Exporter {
    fn with(sink: Sink) -> Self {
        Self {
            sink,
            format: Formatter::Text,
            capacity_bytes: crate::sized::SINK_CAPACITY_BYTES,
        }
    }

    /// Stdout (text by default).
    #[must_use]
    pub fn stdout() -> Self {
        Self::with(Sink::Stdout)
    }

    /// Stderr.
    #[must_use]
    pub fn stderr() -> Self {
        Self::with(Sink::Stderr)
    }

    /// Level-routed console logging: trace/debug/info → stdout, warn/error →
    /// stderr. The standard "just give me logs, problems to stderr" sink.
    #[must_use]
    pub fn std() -> Self {
        Self::with(Sink::StdSplit)
    }

    /// A file, created/truncated when the recorder starts.
    #[must_use]
    pub fn file(path: impl Into<PathBuf>) -> Self {
        Self::with(Sink::File(path.into()))
    }

    /// A file that rolls once it exceeds `max_bytes`: the active file becomes
    /// `path.1`, prior `path.N` becomes `path.N+1`, and whatever sits at
    /// `path.max_files` is dropped. Size-based, checked on every write.
    #[must_use]
    pub fn file_rotating(path: impl Into<PathBuf>, max_bytes: u64, max_files: usize) -> Self {
        Self::with(Sink::RotatingFile {
            path: path.into(),
            max_bytes,
            max_files,
        })
    }

    /// An arbitrary writer (tests, sockets, pipes).
    pub fn writer(writer: impl Write + Send + Sync + 'static) -> Self {
        Self::with(Sink::Writer(Box::new(writer)))
    }

    /// Compose an already-built telemetry pipe as an exporter — the escape hatch
    /// for a custom sink, the OTLP wire client, or any `impl Pipe`. `format` is
    /// ignored (the pipe is already complete). This is what keeps `.export()`
    /// open: anything that lowers to a [`TelemetryPipeHandle`] composes.
    #[must_use]
    pub fn pipe(handle: TelemetryPipeHandle) -> Self {
        Self::with(Sink::Handle(handle))
    }

    /// Choose the formatter (default [`Formatter::Text`]).
    #[must_use]
    pub fn format(mut self, format: Formatter) -> Self {
        self.format = format;
        self
    }

    /// Override the terminal's per-drain-batch buffer preallocation (default the
    /// build-time floor [`crate::sized::SINK_CAPACITY_BYTES`]).
    #[must_use]
    pub fn capacity(mut self, capacity_bytes: usize) -> Self {
        self.capacity_bytes = capacity_bytes;
        self
    }

    fn into_handle(self) -> Result<TelemetryPipeHandle, Error> {
        let format = self.format.to_log_format();
        let capacity = self.capacity_bytes;
        let handle = match self.sink {
            Sink::Stdout => into_telemetry_handle(
                FormatterPipe::new(std::io::stdout(), format).with_capacity_bytes(capacity),
            ),
            Sink::Stderr => into_telemetry_handle(
                FormatterPipe::new(std::io::stderr(), format).with_capacity_bytes(capacity),
            ),
            Sink::StdSplit => into_telemetry_handle(
                crate::pipes::StdSplitPipe::new(format).with_capacity_bytes(capacity),
            ),
            Sink::File(path) => {
                let file = std::fs::File::create(&path).map_err(|_| Error::InvalidInput)?;
                into_telemetry_handle(
                    FormatterPipe::new(file, format).with_capacity_bytes(capacity),
                )
            }
            Sink::RotatingFile {
                path,
                max_bytes,
                max_files,
            } => {
                let writer = RotatingFileWriter::open(path, max_bytes, max_files)
                    .map_err(|_| Error::InvalidInput)?;
                into_telemetry_handle(
                    FormatterPipe::new(writer, format).with_capacity_bytes(capacity),
                )
            }
            Sink::Writer(writer) => into_telemetry_handle(
                FormatterPipe::new(writer, format).with_capacity_bytes(capacity),
            ),
            Sink::Handle(handle) => handle, // already a complete pipe; format ignored
        };
        Ok(handle)
    }
}

impl RecorderBuilder<NoPipe> {
    /// Compose an [`Exporter`] as the recorder's sink. (Fan-out over multiple
    /// exporters lands with the OTLP slice's `FanOut` stage.)
    pub fn export(self, exporter: Exporter) -> Result<RecorderBuilder<HasPipe>, Error> {
        Ok(self.pipe_handle(exporter.into_handle()?))
    }
}

impl RecorderBuilder<HasPipe> {
    /// Build the recorder AND register it as the process default, so emit sites
    /// with no explicit recorder find it via [`default_recorder`].
    pub fn install(self) -> Result<Arc<Recorder>, Error> {
        let recorder = Arc::new(self.start()?);
        set_default_recorder(Arc::clone(&recorder));
        Ok(recorder)
    }
}

/// One call for the overwhelmingly common case: level-routed console logging.
///
/// Builds a recorder whose sink is the level-routed [`Exporter::std`]
/// (trace/debug/info → stdout, warn/error → stderr), registers it as the
/// process-default recorder, bridges `tracing` events into it (so existing
/// `tracing::warn!`/`debug!` callsites surface), and spawns a background drain
/// thread so records reach the console. Returns the recorder (lives for the
/// process). This is the "just give me logs" fast path — no builder dance.
///
/// # Errors
///
/// Propagates recorder-build failures from [`RecorderBuilder::install`].
#[cfg(feature = "tracing-init")]
pub fn install_console_logging() -> Result<Arc<Recorder>, Error> {
    install_console_logging_with(Formatter::Text)
}

/// [`install_console_logging`] with an explicit [`Formatter`] (e.g.
/// [`Formatter::Json`] for structured console output).
///
/// # Errors
///
/// Propagates recorder-build failures.
#[cfg(feature = "tracing-init")]
pub fn install_console_logging_with(format: Formatter) -> Result<Arc<Recorder>, Error> {
    use tracing_subscriber::layer::SubscriberExt;

    let recorder = Recorder::builder()
        .export(Exporter::std().format(format))?
        .install()?;

    // Bridge `tracing` callsites into the recorder. Ignore the error if a
    // global subscriber is already installed — console logging is best-effort.
    let layer = crate::tracing_bridge::TracingLayer::new(Arc::clone(&recorder));
    let subscriber = tracing_subscriber::registry().with(layer);
    let _ = tracing::subscriber::set_global_default(subscriber);

    // Background drain so buffered records reach the console. MUST be a plain OS
    // thread: `drain()` block_on's the terminal pipe and would deadlock on a
    // prime executor thread. Event-driven (parks on the size trigger), not a
    // fixed poll — see `Recorder::run_drain_loop`.
    //
    // a caller that needs to know the final flush landed (e.g. before process
    // exit) calls `Recorder::drain()` directly: the rings are multi-consumer,
    // so it is safe alongside this background pump and returns only once its
    // own pass has actually written out.
    let pump = Arc::clone(&recorder);
    std::thread::Builder::new()
        .name("telemetry-console-drain".to_string())
        .spawn(move || pump.run_drain_loop())
        .map_err(|error| Error::ThreadSpawn(error.to_string()))?;

    Ok(recorder)
}

fn default_to() -> String {
    "stdout".to_string()
}
fn default_format() -> String {
    "text".to_string()
}
fn default_sink_capacity() -> usize {
    crate::sized::SINK_CAPACITY_BYTES
}

/// The single exporter-config surface — conflaguration mirror of an [`Exporter`].
///
/// Covers every sink: `to = "stdout" | "stderr" | "file" | "noop" | "otlp-http"
/// | "otlp-grpc"`. Local sinks lower directly via [`into_exporter`](Self::into_exporter);
/// the OTLP/noop cases lower to the [`ExporterChoice`] IR via
/// [`to_exporter_choice`](Self::to_exporter_choice), which the umbrella's
/// `exporter_pipe`/`recorder_from_config` resolve to a real network client — so
/// the OTLP wiring is reused, not reinvented. Same Settings + Validate + builder
/// pattern as the rest of the telemetry config surface.
#[derive(Debug, Clone, PartialEq, Eq, Builder, Deserialize, Serialize, Settings)]
#[settings(prefix = "TELEMETRY_EXPORT")]
#[builder(derive(Clone, Debug))]
pub struct ExportConfig {
    /// `"stdout"` | `"stderr"` | `"file"` | `"noop"` | `"otlp-http"` | `"otlp-grpc"`.
    #[setting(default = "stdout")]
    #[serde(default = "default_to")]
    #[builder(default = default_to())]
    pub to: String,

    /// `"text"` | `"json"`.
    #[setting(default = "text")]
    #[serde(default = "default_format")]
    #[builder(default = default_format())]
    pub format: String,

    /// File path (only for `to = "file"`).
    #[setting(skip)]
    #[serde(default)]
    pub path: Option<String>,

    /// Endpoint URL (only for `to = "otlp-http" | "otlp-grpc"`).
    #[setting(skip)]
    #[serde(default)]
    pub url: Option<String>,

    /// Ring overflow policy: `block` (lossless via producer-assist, default) or
    /// `drop` (shed + count). Std-config only — below std a sink bakes one policy.
    #[setting(skip)]
    #[serde(default)]
    #[builder(default)]
    pub overflow: OverflowPolicy,

    /// Per-drain-batch terminal buffer preallocation (bytes). Seeded from the
    /// build-time sized floor (`crate::sized::SINK_CAPACITY_BYTES`); override
    /// per-process via config file / builder.
    #[setting(skip)]
    #[serde(default = "default_sink_capacity")]
    #[builder(default = default_sink_capacity())]
    pub capacity_bytes: usize,
}

impl ExportConfig {
    /// Lower to a typed [`Exporter`]. Errors on an unknown `to`/`format`.
    pub fn into_exporter(&self) -> Result<Exporter, Error> {
        let format = match self.format.trim().to_ascii_lowercase().as_str() {
            "text" => Formatter::Text,
            "json" => Formatter::Json,
            _ => return Err(Error::InvalidInput),
        };
        let exporter = match self.to.trim().to_ascii_lowercase().as_str() {
            "stdout" => Exporter::stdout(),
            "stderr" => Exporter::stderr(),
            "split" => Exporter::std(),
            "file" => Exporter::file(self.path.as_ref().ok_or(Error::InvalidInput)?.clone()),
            "noop" | "none" => Exporter::pipe(into_telemetry_handle(NullPipe::new())),
            // OTLP sinks resolve to a network client in the umbrella — see
            // `to_exporter_choice`; they are not leaf-resolvable.
            _ => return Err(Error::InvalidInput),
        };
        Ok(exporter.format(format).capacity(self.capacity_bytes))
    }

    /// Lower the OTLP/noop cases to the [`ExporterChoice`] IR that the umbrella's
    /// `exporter_pipe`/`recorder_from_config` already resolve to a real client —
    /// so the OTLP transport wiring is reused, not duplicated. `None` for local
    /// sinks (use [`into_exporter`](Self::into_exporter) for those).
    #[must_use]
    pub fn to_exporter_choice(&self) -> Option<ExporterChoice> {
        let _url = self.url.clone().unwrap_or_default();
        match self.to.trim().to_ascii_lowercase().as_str() {
            "noop" | "none" => Some(ExporterChoice::Noop),
            #[cfg(feature = "otlp-http")]
            "otlp-http" => Some(ExporterChoice::OtlpHttp { endpoint: _url }),
            #[cfg(feature = "otlp-grpc")]
            "otlp-grpc" => Some(ExporterChoice::OtlpGrpc { endpoint: _url }),
            _ => None,
        }
    }

    /// Load from a TOML/JSON file (validated via [`Validate`]).
    pub fn from_path(path: impl AsRef<std::path::Path>) -> Result<Self, conflaguration::Error> {
        conflaguration::from_file(path.as_ref())
    }

    // ---- named degenerate forms (config-as-composition) -------------------
    // Each is a mostly-defaulted composition of the compiled terminal + policy
    // primitives — adding one is a constructor + a `preset` arm, never a new
    // sink impl. The same form is expressible purely in config (`to = "split"`).

    /// `console` — severity-split stdout/stderr, text, lossless. The
    /// `install_console_logging` shape as a named, config-buildable form.
    #[must_use]
    pub fn console() -> Self {
        Self::builder().to("split".into()).build()
    }

    /// `file(path)` — a single file sink, text, lossless.
    #[must_use]
    pub fn file(path: impl Into<String>) -> Self {
        Self::builder().to("file".into()).path(path.into()).build()
    }

    /// `audit(path)` — a file sink that never drops (every record matters):
    /// lossless `block` overflow, no shed stage.
    #[must_use]
    pub fn audit(path: impl Into<String>) -> Self {
        Self::builder()
            .to("file".into())
            .path(path.into())
            .overflow(OverflowPolicy::Block)
            .build()
    }

    /// `null` — a no-op sink (drops everything), for benches / disabled telemetry.
    #[must_use]
    pub fn null() -> Self {
        Self::builder().to("noop".into()).build()
    }

    /// Build a named degenerate form by name: `console | file | audit | null`.
    /// `file`/`audit` require `path`; unknown names are rejected.
    pub fn preset(name: &str, path: Option<&str>) -> Result<Self, Error> {
        match name.trim().to_ascii_lowercase().as_str() {
            "console" => Ok(Self::console()),
            "file" => Ok(Self::file(path.ok_or(Error::InvalidInput)?)),
            "audit" => Ok(Self::audit(path.ok_or(Error::InvalidInput)?)),
            "null" | "noop" | "none" => Ok(Self::null()),
            _ => Err(Error::InvalidInput),
        }
    }
}

impl Validate for ExportConfig {
    fn validate(&self) -> conflaguration::Result<()> {
        // valid if it lowers to a local sink OR to an OTLP/noop choice.
        if self.into_exporter().is_ok() || self.to_exporter_choice().is_some() {
            Ok(())
        } else {
            Err(conflaguration::Error::Validation {
                errors: alloc::vec![ValidationMessage::new(
                    "export",
                    "expected to = stdout|stderr|file|noop|otlp-http|otlp-grpc (otlp needs its feature) and format = text|json",
                )],
            })
        }
    }
}

/// The dead-simple conflaguration path: load an export config from a file, build
/// the recorder, and register it as the process default — one call.
pub fn install_from_path(
    path: impl AsRef<std::path::Path>,
) -> Result<Arc<Recorder>, conflaguration::Error> {
    let config = ExportConfig::from_path(path)?;
    install_from_config(&config)
}

/// Build + install a recorder from a loaded [`ExportConfig`].
pub fn install_from_config(config: &ExportConfig) -> Result<Arc<Recorder>, conflaguration::Error> {
    let exporter = config
        .into_exporter()
        .map_err(|_| conflaguration::Error::Validation {
            errors: alloc::vec![ValidationMessage::new("export", "could not build exporter")],
        })?;
    Recorder::builder()
        .export(exporter)
        .map(|builder| builder.overflow(config.overflow))
        .and_then(RecorderBuilder::install)
        .map_err(|_| conflaguration::Error::Validation {
            errors: alloc::vec![ValidationMessage::new("export", "could not start recorder")],
        })
}

// these tests build a real Recorder, which uses proxima-core's Ring/
// StaticRing internally -- cfg-swapped to loom under `--features loom`
// (forwarded via proxima-core/loom), only usable inside an actual
// loom::model(...) closure, which these plain #[test] functions don't
// provide.
#[cfg(all(test, not(feature = "loom")))]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::field_reassign_with_default,
        clippy::type_complexity,
        clippy::useless_vec,
        clippy::needless_range_loop,
        clippy::default_constructed_unit_structs
    )]

    use alloc::string::String;
    use alloc::sync::Arc;
    use std::io::{self, Write};
    use std::sync::Mutex;

    use super::{Exporter, default_recorder};
    use crate::level::Level;
    use crate::recorder::Recorder;

    // a shared in-memory writer so the test can assert the formatted bytes.
    #[derive(Clone)]
    struct SharedBuf(Arc<Mutex<alloc::vec::Vec<u8>>>);

    impl Write for SharedBuf {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    // the dead-simple path: one composable `.export()`, `.install()` registers
    // the global, and a log reaches the sink formatted.
    #[test]
    fn export_writer_installs_global_and_formats() {
        let buf = Arc::new(Mutex::new(alloc::vec::Vec::new()));
        let recorder = Recorder::builder()
            .export(Exporter::writer(SharedBuf(buf.clone())))
            .unwrap()
            .core_count(1)
            .install()
            .unwrap();

        assert!(
            default_recorder().is_some(),
            "install registered the global default"
        );

        recorder
            .log()
            .level(Level::INFO)
            .message("hello-export")
            .module_path("proxima::test")
            .emit();
        recorder.drain();

        let out = String::from_utf8(buf.lock().unwrap().clone()).unwrap();
        assert!(
            out.contains("hello-export"),
            "log reached the writer formatted: {out}"
        );
    }

    // the DEAD-SIMPLE conflaguration path: a TOML file → install → a record
    // lands at the configured sink, through the real conflaguration loader.
    #[test]
    fn dead_simple_conflag_installs_from_toml() {
        use super::install_from_path;
        let dir = tempfile::TempDir::new().unwrap();
        let out = dir.path().join("telemetry.log");
        let cfg_path = dir.path().join("telemetry.toml");
        std::fs::write(
            &cfg_path,
            std::format!(
                "to = \"file\"\nformat = \"text\"\npath = \"{}\"\n",
                out.display()
            ),
        )
        .unwrap();

        let recorder = install_from_path(&cfg_path).unwrap();
        recorder
            .log()
            .level(Level::INFO)
            .message("from-conflag")
            .module_path("proxima::test")
            .emit();
        recorder.drain();

        let written = std::fs::read_to_string(&out).unwrap();
        assert!(
            written.contains("from-conflag"),
            "config-driven export wrote it: {written}"
        );
    }

    // the `to = "stdout"` one-liner parses + lowers.
    #[test]
    fn stdout_config_lowers_to_exporter() {
        use super::ExportConfig;
        let cfg = ExportConfig::builder().to(String::from("stdout")).build();
        assert!(cfg.into_exporter().is_ok());
    }

    // the composability escape hatch: any pre-built pipe composes via .export().
    #[test]
    fn export_pipe_composes_any_handle() {
        use crate::pipes::{InMemoryPipe, into_telemetry_handle};
        let sink = InMemoryPipe::new();
        let recorder = Recorder::builder()
            .export(Exporter::pipe(into_telemetry_handle(sink.clone())))
            .unwrap()
            .core_count(1)
            .install()
            .unwrap();
        recorder
            .log()
            .level(Level::INFO)
            .message("via-pipe")
            .module_path("proxima::test")
            .emit();
        recorder.drain();
        assert_eq!(sink.logs().len(), 1, "the composed pipe received the log");
    }

    // consolidation: ExportConfig covers noop (a leaf NullPipe) AND lowers it to
    // the same ExporterChoice IR the umbrella resolver uses.
    #[test]
    fn config_covers_noop_and_lowers_to_choice() {
        use super::ExportConfig;
        use crate::config::ExporterChoice;
        let noop = ExportConfig::builder().to(String::from("noop")).build();
        assert!(noop.into_exporter().is_ok(), "noop is a leaf NullPipe sink");
        assert!(matches!(
            noop.to_exporter_choice(),
            Some(ExporterChoice::Noop)
        ));
    }

    // OTLP isn't a leaf sink — it lowers to the ExporterChoice IR the umbrella
    // resolves to a real client (reused, not reinvented).
    #[cfg(feature = "otlp-http")]
    #[test]
    fn otlp_http_config_lowers_to_choice_not_leaf() {
        use super::ExportConfig;
        use crate::config::ExporterChoice;
        let otlp = ExportConfig::builder()
            .to(String::from("otlp-http"))
            .url(String::from("http://collector:4318"))
            .build();
        assert!(otlp.into_exporter().is_err(), "otlp is not leaf-resolvable");
        assert!(matches!(
            otlp.to_exporter_choice(),
            Some(ExporterChoice::OtlpHttp { .. })
        ));
    }

    // the named degenerate forms compose terminal + policy from the compiled
    // primitives, and `preset` builds them by name — adding a form is a
    // constructor, not a new sink impl.
    #[test]
    fn presets_compose_terminal_and_policy() {
        use super::{ExportConfig, OverflowPolicy};

        let console = ExportConfig::console();
        assert_eq!(console.to, "split");
        assert_eq!(
            console.overflow,
            OverflowPolicy::Block,
            "console is lossless"
        );
        assert!(
            console.into_exporter().is_ok(),
            "console lowers to a real exporter"
        );

        let audit = ExportConfig::audit("/var/log/audit.log");
        assert_eq!(audit.to, "file");
        assert_eq!(audit.overflow, OverflowPolicy::Block, "audit never drops");

        assert_eq!(ExportConfig::preset("console", None).unwrap(), console);
        assert!(
            ExportConfig::preset("file", None).is_err(),
            "file needs a path"
        );
        assert!(ExportConfig::preset("bogus", None).is_err());
    }

    // config-as-composition: a sink variant (split terminal + a different overflow
    // policy) composed PURELY from a two-line TOML, lowering to a real exporter —
    // zero new Rust for the variant.
    #[test]
    fn sink_form_composed_from_toml() {
        use super::{ExportConfig, OverflowPolicy};

        let dir = tempfile::TempDir::new().unwrap();
        let cfg_path = dir.path().join("sink.toml");
        std::fs::write(&cfg_path, "to = \"split\"\noverflow = \"drop\"\n").unwrap();

        let cfg = ExportConfig::from_path(&cfg_path).unwrap();
        assert_eq!(cfg.to, "split");
        assert_eq!(
            cfg.overflow,
            OverflowPolicy::Drop,
            "policy composed from config"
        );
        assert!(cfg.into_exporter().is_ok());
    }

    // bidirectional interop (principle 4): the builder form and the config-loaded
    // form are the same value.
    #[test]
    fn builder_matches_config_loaded_from_toml() {
        use super::ExportConfig;

        let dir = tempfile::TempDir::new().unwrap();
        let cfg_path = dir.path().join("file.toml");
        std::fs::write(
            &cfg_path,
            "to = \"file\"\nformat = \"text\"\npath = \"/tmp/x.log\"\noverflow = \"block\"\n",
        )
        .unwrap();

        let loaded = ExportConfig::from_path(&cfg_path).unwrap();
        let built = ExportConfig::file("/tmp/x.log");
        assert_eq!(loaded, built, "config-loaded form == builder form");
    }

    // the runtime sink-buffer default is SEEDED from the build-time sized floor —
    // one source of truth across the std/no_std tiers (the conflag bridge).
    #[test]
    fn sink_capacity_default_tracks_sized_floor() {
        use super::ExportConfig;
        let cfg = ExportConfig::builder().build();
        assert_eq!(
            cfg.capacity_bytes,
            crate::sized::SINK_CAPACITY_BYTES,
            "runtime default seeded from the sized floor",
        );
    }

    // reads every rolled file (`path.1..=path.max_files`) plus the active file,
    // in oldest-to-newest order, so a test can assert on the retained window.
    fn read_all_rotated(path: &std::path::Path, max_files: usize) -> alloc::vec::Vec<String> {
        let mut lines = alloc::vec::Vec::new();
        for index in (1..=max_files).rev() {
            let mut file_name = path.as_os_str().to_owned();
            file_name.push(std::format!(".{index}"));
            let rolled = std::path::PathBuf::from(file_name);
            if let Ok(contents) = std::fs::read_to_string(&rolled) {
                lines.extend(contents.lines().map(str::to_string));
            }
        }
        if let Ok(contents) = std::fs::read_to_string(path) {
            lines.extend(contents.lines().map(str::to_string));
        }
        lines
    }

    // phase 1: fill exactly active + max_files rolled files (zero drops yet) —
    // proves a roll never loses a record while under retention. Phase 2: push
    // one record further to force a fifth rotation past capacity — proves
    // exactly max_files rolled files remain and the oldest was dropped, not the
    // newest.
    #[test]
    fn rotating_writer_rolls_at_max_bytes_and_retains_max_files() {
        use super::RotatingFileWriter;

        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("app.log");
        let max_bytes = 10_u64;
        let max_files = 3_usize;
        let mut writer = RotatingFileWriter::open(path.clone(), max_bytes, max_files).unwrap();

        // fixed 5-byte records ("0000\n"): two fit per file before a roll.
        let phase_one_records = (max_files as u32 + 1) * 2;
        for index in 0..phase_one_records {
            writer
                .write_all(std::format!("{index:04}\n").as_bytes())
                .unwrap();
        }
        writer.flush().unwrap();

        let active_size = std::fs::metadata(&path).unwrap().len();
        assert!(
            active_size <= max_bytes,
            "active file stays under max_bytes after a roll: {active_size}"
        );

        let retained = read_all_rotated(&path, max_files);
        assert_eq!(
            retained.len(),
            phase_one_records as usize,
            "no record lost while still under retention capacity"
        );

        // phase 2: one more record forces a fifth roll, past the max_files
        // retention window — the oldest rolled file must now be dropped.
        writer.write_all(b"9999\n").unwrap();
        writer.flush().unwrap();

        assert!(
            !writer.rolled_path(max_files + 1).exists(),
            "never more than max_files rolled files on disk"
        );
        assert!(
            writer.rolled_path(max_files).exists(),
            "max_files rolled files retained"
        );

        let after_drop = read_all_rotated(&path, max_files);
        assert_eq!(
            after_drop.len(),
            phase_one_records as usize - 1,
            "exactly one record dropped: the oldest, past retention"
        );
        assert!(
            !after_drop.contains(&"0000".to_string()),
            "the oldest record was dropped, not retained"
        );
        assert!(
            after_drop.contains(&"9999".to_string()),
            "the newest record is present"
        );
    }

    // proves `Exporter::file_rotating` composes into the same Recorder/pipe
    // machinery as `Exporter::file` — an install-and-emit path, not just the
    // bare `Write` impl — and that rotation actually happens end to end.
    #[test]
    fn exporter_file_rotating_composes_into_recorder_and_rotates() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("recorder.log");
        let max_files = 3_usize;

        let recorder = Recorder::builder()
            .export(Exporter::file_rotating(path.clone(), 1, max_files))
            .unwrap()
            .core_count(1)
            .install()
            .unwrap();

        let markers = [
            "rotate-marker-0",
            "rotate-marker-1",
            "rotate-marker-2",
            "rotate-marker-3",
            "rotate-marker-4",
            "rotate-marker-5",
        ];
        let total_records = markers.len();
        for marker in markers {
            recorder
                .log()
                .level(Level::INFO)
                .message(marker)
                .module_path("proxima::test")
                .emit();
            recorder.drain();
        }

        let retained = read_all_rotated(&path, max_files);
        let rotated_files = (1..=max_files)
            .filter(|index| {
                let mut file_name = path.as_os_str().to_owned();
                file_name.push(std::format!(".{index}"));
                std::path::PathBuf::from(file_name).exists()
            })
            .count();
        assert!(
            rotated_files >= 2,
            "at least two rotations happened: {rotated_files} rolled files present"
        );
        assert!(
            !retained.iter().any(|line| line.contains("rotate-marker-0")),
            "the oldest record was dropped past retention"
        );
        assert!(
            retained
                .iter()
                .any(|line| line.contains(&std::format!("rotate-marker-{}", total_records - 1))),
            "the newest record is present: {retained:?}"
        );
    }
}
