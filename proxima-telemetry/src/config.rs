//! `TelemetryConfig` + `Recorder::from_config` / `to_config` bridge.
//!
//! `TelemetryLayerBuilder` supports call-order precedence:
//!
//! - **Operator config wins**: put `.with_*` BEFORE `.from_path` / `.from_env`.
//! - **Code overrides win**: put `.with_*` AFTER `.from_path` / `.from_env`.

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use std::collections::BTreeSet;

use bon::Builder;
use conflaguration::{Settings, Validate, ValidationMessage};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::config_merge::{MergeMode, apply_layer, insert_if_env_set};
use crate::level::Level;
use crate::pipes::{NullPipe, TelemetryPipeHandle, fan_exporters, into_telemetry_handle};
use crate::recorder::{HasPipe, Recorder, RecorderBuilder, RingCapacities};
use crate::tag::{ScalarValue, Tag};

/// Telemetry configuration surface.
#[derive(Debug, Clone, Builder, Deserialize, Serialize, Settings)]
#[settings(prefix = "TELEMETRY")]
#[builder(derive(Clone, Debug))]
pub struct TelemetryConfig {
    #[setting(default = 4096)]
    #[serde(default = "default_ring_spans")]
    #[builder(default = default_ring_spans())]
    pub ring_spans: usize,

    #[setting(default = 4096)]
    #[serde(default = "default_ring_events")]
    #[builder(default = default_ring_events())]
    pub ring_events: usize,

    #[setting(default = 4096)]
    #[serde(default = "default_ring_logs")]
    #[builder(default = default_ring_logs())]
    pub ring_logs: usize,

    #[setting(default = 8192)]
    #[serde(default = "default_ring_metrics")]
    #[builder(default = default_ring_metrics())]
    pub ring_metrics: usize,

    #[setting(default = 1024)]
    #[serde(default = "default_ring_links")]
    #[builder(default = default_ring_links())]
    pub ring_links: usize,

    #[setting(default = 2048)]
    #[serde(default = "default_ring_overflow_attrs")]
    #[builder(default = default_ring_overflow_attrs())]
    pub ring_overflow_attrs: usize,

    #[setting(default = 0)]
    #[serde(default)]
    #[builder(default)]
    pub core_count: usize,

    #[setting(skip)]
    #[serde(default)]
    #[builder(default)]
    pub exporter: ExporterChoice,

    /// Fan each record to these exporters. Non-empty makes the recorder's
    /// terminal pipe a `FanExporter` over the list (each lowered like
    /// `exporter`) and the single `exporter` is ignored; empty (default) keeps
    /// the single-`exporter` path byte-identical. P4 config-as-composition: a
    /// multi-sink deployment is a TOML list, not new Rust.
    ///
    /// The fan clones the drained record per secondary exporter; with >=2
    /// exporters `from_config` automatically selects `record_sharing = Arc` so
    /// that clone is a refcount bump on the `*BatchArc` form, not a deep copy —
    /// the fan-out intent drives the sharing mechanism, no manual pairing needed.
    #[setting(skip)]
    #[serde(default)]
    #[builder(default)]
    pub exporters: Vec<ExporterChoice>,

    #[setting(skip)]
    #[serde(default)]
    #[builder(default)]
    pub resource: Vec<ResourceTag>,

    #[setting(skip)]
    #[serde(default)]
    pub sampler: Option<SamplerSpec>,

    /// Error-elevation policy. `None` (default) is the **simple form**: today's
    /// record-time gate, no per-trace buffer stage installed — genuinely
    /// zero-cost. `Some` installs the tail-sampled replay: a floor level always
    /// emits; a sampled fraction of traces, on an error trigger, replay their
    /// full tree down to `elevated` to a separate exporter. See [`Elevation`].
    #[setting(skip)]
    #[serde(default)]
    pub elevation: Option<Elevation>,

    #[setting(skip)]
    #[serde(default)]
    #[builder(default)]
    pub record_sharing: RecordSharing,

    #[setting(skip)]
    #[serde(default)]
    #[builder(default)]
    pub overflow: OverflowPolicy,

    /// Spawn an owned background drainer thread (default false). Off keeps the
    /// recorder bring-your-own-pump; on gives steady state a dedicated pump so
    /// producers rarely self-drain under `Block`. Shutdown is lossless either way.
    #[setting(default = false)]
    #[serde(default)]
    #[builder(default)]
    pub managed_drainer: bool,

    /// Records a background drain pass moves+exports per ring per pass. Off the
    /// emit hot path — sized for drain throughput. Build-time default lives in
    /// `proxima-telemetry.toml` (`crate::sized::DRAIN_BATCH`, override at build
    /// via `PROXIMA_TELEMETRY_DRAIN_BATCH`); this overrides it per-process.
    #[setting(default = 512)]
    #[serde(default = "default_drain_batch")]
    #[builder(default = default_drain_batch())]
    pub drain_batch: usize,

    /// Records a producer drains+exports on a full ring (elastic producer-assist)
    /// before retrying its push — runs on the producer/request thread, so it
    /// bounds the worst-case emit tail (stall ≈ this × per-record sink latency).
    /// Keep small, well below `drain_batch`. Build-time default in
    /// `proxima-telemetry.toml` (`crate::sized::DRAIN_ASSIST_BATCH`, override at
    /// build via `PROXIMA_TELEMETRY_DRAIN_ASSIST_BATCH`); this overrides per-process.
    #[setting(default = 64)]
    #[serde(default = "default_assist_batch")]
    #[builder(default = default_assist_batch())]
    pub assist_batch: usize,

    /// How often the managed pump flushes when no ring fills — the time trigger
    /// of the size-or-time pump (a full ring is the size trigger). The pump sleeps
    /// between flushes, so this also sets the idle wake cadence: smaller bounds
    /// export latency, larger spends less CPU at idle. Microseconds.
    #[setting(default = 1000)]
    #[serde(default = "default_flush_interval_micros")]
    #[builder(default = default_flush_interval_micros())]
    pub flush_interval_micros: u64,
}

/// What `emit` does when a per-core ring is full (producer outrunning the drain).
///
/// `Block` (default) is **lossless AND deadlock-free** via elastic
/// producer-assist: on a full ring the blocked producer becomes a consumer
/// momentarily — it drains+exports a batch itself, freeing slots, then pushes.
/// Progress is guaranteed by the producer's own action, so it never hangs even
/// with no separate drainer running; under genuine overload it throttles the
/// producer to real sink throughput (correct backpressure). It only ever waits
/// on actual downstream work, never on "nothing".
///
/// `Drop` discards the record and counts it (`recorder.dropped()`) — lossy but
/// never throttles the emitter; the explicit opt-out for expendable / sampled
/// signals where shedding under overload beats slowing the app.
///
/// The policy is read only on the cold `Full` path, so steady-state emit is
/// byte-identical across policies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum OverflowPolicy {
    #[default]
    Block,
    Drop,
}

/// Pre-allocation sampling gate spec.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SamplerSpec {
    AlwaysOn,
    AlwaysOff,
    TraceIdRatioBased {
        p: f64,
    },
    ParentBased {
        root: alloc::boxed::Box<SamplerSpec>,
        sampled: alloc::boxed::Box<SamplerSpec>,
        not_sampled: alloc::boxed::Box<SamplerSpec>,
    },
}

/// Error-elevation policy — the `Some` arm of [`TelemetryConfig::elevation`].
///
/// A floor level always emits to the normal exporter (unchanged). A
/// `sample_ratio` fraction of traces are admitted to verbose-buffered mode
/// (tail-sampling); for those, records down to `elevated` are buffered per-trace,
/// and a `trigger_level` record replays that trace's full ordered tree to a
/// separate exporter. Below-floor records for non-sampled traces never exist —
/// the record-time gate still drops them — so the healthy-path cost is bounded to
/// the sampled fraction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Elevation {
    /// The always-emitted floor. It SHOULD equal the operator's effective
    /// emit gate floor (`RUST_LOG` / the `CallsiteGate`); the two are not
    /// auto-synced because `RUST_LOG` can be per-module with no single floor
    /// to sync to — aligning them is the operator's responsibility. A
    /// mismatch makes the normal sink inconsistent (verbose-sampled traces
    /// would carry different levels than the rest). Defaults to `info`.
    #[serde(default = "default_elevation_floor")]
    pub floor: Level,
    /// The replay depth for a triggered trace; `None` (default) means
    /// identical to `floor` (no extra depth — opt into detail by lowering
    /// it). Must not be coarser than `floor`.
    #[serde(default)]
    pub elevated: Option<Level>,
    /// Fraction of traces admitted to verbose-buffered mode, `0.0..=1.0`. The
    /// tail-sampling knob that bounds the buffering cost.
    pub sample_ratio: f64,
    /// A record at or above this level fires the replay for its buffered trace.
    /// Default `error`. A bare [`Level`] IS the trigger — a level comparison
    /// expresses "fire on error" without a bespoke predicate type.
    #[serde(default = "default_trigger_level")]
    pub trigger_level: Level,
    /// Where a triggered replay is sent — a SEPARATE forensic sink, distinct from
    /// the normal `exporter`. Default `Noop` (replays discarded): set this to a
    /// real sink or the elevated tree goes nowhere. Resolved transport-free like
    /// the normal exporter (compose transport at the umbrella for the wire).
    #[serde(default)]
    pub exporter: ExporterChoice,
    /// Per-trace buffer lifetime + memory bounds.
    #[serde(default)]
    pub retention: Retention,
}

fn default_trigger_level() -> Level {
    Level::ERROR
}

fn default_elevation_floor() -> Level {
    Level::INFO
}

impl Elevation {
    /// Resolve the replay depth: `elevated` if set, else `floor` — unset
    /// `elevated` means "no extra depth," not "no depth."
    #[must_use]
    pub fn resolved_elevated(&self) -> Level {
        self.elevated.unwrap_or(self.floor)
    }
}

/// Per-trace replay-buffer lifetime and memory bounds — the layered eviction
/// policy. Root-close is the semantic completion signal; TTL reclaims traces
/// whose root-close was never observed; the count-cap is the hard OOM backstop.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Retention {
    /// Reclaim a trace's buffer when its root span closes (the precise
    /// completion signal). Default `true`.
    #[serde(default = "default_true")]
    pub drain_on_root_close: bool,
    /// Reclaim a trace whose root-close was never seen after this idle time
    /// (crash / lost span), in milliseconds. `0` disables the TTL sweep.
    #[serde(default = "default_ttl_millis")]
    pub ttl_millis: u64,
    /// Hard cap on concurrently-buffered traces (OOM backstop); the
    /// least-recently-touched trace is evicted past it. `0` = the build-time
    /// default `sized::ELEVATION_MAX_TRACES`.
    #[serde(default)]
    pub max_traces: usize,
    /// Per-trace replay ring capacity (records); DropOldest past it. `0` = the
    /// build-time default `sized::ELEVATION_PER_TRACE_RING`.
    #[serde(default)]
    pub per_trace_ring: usize,
}

fn default_true() -> bool {
    true
}
fn default_ttl_millis() -> u64 {
    60_000
}

impl Default for Retention {
    fn default() -> Self {
        Self {
            drain_on_root_close: default_true(),
            ttl_millis: default_ttl_millis(),
            max_traces: 0,
            per_trace_ring: 0,
        }
    }
}

/// How drained records are shared with downstream Pipes.
///
/// `Inline` hands the drainer's owned `Vec<Record>` straight to the pipe — no
/// per-record allocation. `Arc` wraps every drained record in an `Arc` so a
/// fan-out pipe can clone it cheaply to N sinks.
///
/// Default is `Inline`: the common case is a single terminal sink, where the
/// per-record `Arc` is pure waste — load testing measured it **halving** the
/// single-drainer throughput (1.12 → 2.37 M spans/s, drops 63% → 23%). Set
/// `Arc` explicitly only when the pipe fans out to multiple downstream sinks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum RecordSharing {
    #[default]
    Inline,
    Arc,
}

fn default_ring_spans() -> usize {
    4096
}
fn default_ring_events() -> usize {
    4096
}
fn default_ring_logs() -> usize {
    4096
}
fn default_ring_metrics() -> usize {
    8192
}
fn default_ring_links() -> usize {
    1024
}
fn default_ring_overflow_attrs() -> usize {
    2048
}
// drain/assist batch defaults trace to the build-time sizing TOML
// (proxima-telemetry.toml -> crate::sized) so there is one source of truth; the
// `#[setting]` literal mirrors it for the env-overlay path and is drift-guarded
// by `setting_defaults_match_sized` in tests.
fn default_drain_batch() -> usize {
    crate::sized::DRAIN_BATCH
}
fn default_assist_batch() -> usize {
    crate::sized::DRAIN_ASSIST_BATCH
}
fn default_flush_interval_micros() -> u64 {
    crate::sized::FLUSH_INTERVAL_MICROS
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self::builder().build()
    }
}

/// Exporter choice.
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ExporterChoice {
    #[default]
    Noop,
    #[cfg(feature = "otlp-http")]
    OtlpHttp { endpoint: String },
    #[cfg(feature = "otlp-grpc")]
    OtlpGrpc { endpoint: String },
}

/// Resource attribute.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct ResourceTag {
    pub key: String,
    pub value: String,
}

impl Validate for TelemetryConfig {
    fn validate(&self) -> conflaguration::Result<()> {
        let mut errors = Vec::new();
        for (name, value) in [
            ("ring_spans", self.ring_spans),
            ("ring_events", self.ring_events),
            ("ring_logs", self.ring_logs),
            ("ring_metrics", self.ring_metrics),
            ("ring_links", self.ring_links),
            ("ring_overflow_attrs", self.ring_overflow_attrs),
        ] {
            if value == 0 {
                errors.push(ValidationMessage::new(name, "must be > 0"));
            } else if !value.is_power_of_two() {
                errors.push(ValidationMessage::new(
                    name,
                    "must be a power of two (the ring caps require it)",
                ));
            }
        }
        push_exporter_endpoint_errors(&self.exporter, "exporter.endpoint", &mut errors);
        for (index, choice) in self.exporters.iter().enumerate() {
            push_exporter_endpoint_errors(
                choice,
                &alloc::format!("exporters[{index}].endpoint"),
                &mut errors,
            );
        }
        if self.flush_interval_micros == 0 {
            // a zero interval makes the pump's wait return immediately — a hot-spin.
            errors.push(ValidationMessage::new(
                "flush_interval_micros",
                "must be > 0 (0 would hot-spin the pump)",
            ));
        }
        if let Some(elevation) = &self.elevation {
            if !(0.0..=1.0).contains(&elevation.sample_ratio) {
                errors.push(ValidationMessage::new(
                    "elevation.sample_ratio",
                    "must be in 0.0..=1.0",
                ));
            }
            if elevation.resolved_elevated().severity() > elevation.floor.severity() {
                errors.push(ValidationMessage::new(
                    "elevation.elevated",
                    "must not be coarser than floor (elevated severity <= floor severity)",
                ));
            }
            // fail loud rather than silently ignore a policy the build can't honour.
            #[cfg(not(feature = "elevation"))]
            errors.push(ValidationMessage::new(
                "elevation",
                "set but the `elevation` feature is not compiled in",
            ));
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(conflaguration::Error::Validation { errors })
        }
    }
}

impl Recorder {
    /// Build a `HasPipe`-state recorder builder from a config, resolving the
    /// terminal pipe from `cfg.exporter` via [`pipe_from_choice`].
    ///
    /// This crate is transport-free, so a transport-requiring exporter
    /// (`OtlpHttp`/`OtlpGrpc`) resolves to its codec only — it encodes but does
    /// not send. To actually send over the wire, compose the transport at a
    /// layer that has an HTTP client and inject it via [`from_config_with_pipe`].
    pub fn from_config(cfg: &TelemetryConfig) -> RecorderBuilder<HasPipe> {
        Self::from_config_with_pipe(cfg, pipe_from_config(cfg))
    }

    /// Build a `HasPipe`-state recorder builder from a config with an
    /// externally-composed terminal `pipe`, ignoring `cfg.exporter`.
    ///
    /// This is the seam the umbrella's config builder uses: it composes
    /// `OtlpHttpCodec -> transport` (the leaf can't, being transport-free) and
    /// injects the result here, so everything else (rings, sampler, batches,
    /// overflow) still comes from `cfg`.
    pub fn from_config_with_pipe(
        cfg: &TelemetryConfig,
        pipe: TelemetryPipeHandle,
    ) -> RecorderBuilder<HasPipe> {
        let core_count = if cfg.core_count == 0 {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        } else {
            cfg.core_count
        };
        // elevation (when configured) rewraps the terminal pipe as a fan-out over
        // [FloorFilter -> terminal, ElevationSink] and arms the verbose sampler.
        // None / feature-off leaves the terminal pipe exactly as passed in.
        #[cfg(feature = "elevation")]
        let pipe = install_elevation(cfg, pipe);
        let mut builder = Recorder::builder()
            .ring_capacities(RingCapacities {
                spans: cfg.ring_spans,
                events: cfg.ring_events,
                logs: cfg.ring_logs,
                metrics: cfg.ring_metrics,
                links: cfg.ring_links,
                overflow_attrs: cfg.ring_overflow_attrs,
                #[cfg(feature = "deferred-metric-fold")]
                span_obs: cfg.ring_metrics,
            })
            .core_count(core_count)
            .pipe_handle(pipe);
        for tag in &cfg.resource {
            builder =
                builder.resource_tag(leak_str(&tag.key), ScalarValue::Str(leak_str(&tag.value)));
        }
        if let Some(spec) = &cfg.sampler {
            let sampler_box = crate::sampler::spec_to_box(spec);
            builder = builder.sampler_boxed(sampler_box);
        }
        // the fan-out intent (>=2 exporters) DRIVES the sharing mechanism: a fan
        // needs Arc so the FanExporter clones each record with a refcount bump
        // instead of a deep copy per sink. A single sink stays Inline (the cheap
        // default — Arc there is pure waste, halving the single-drainer throughput).
        // elevation installs a fan-out too, so it wants Arc sharing for the same
        // reason (cheap per-arm clone of the *BatchArc drain form).
        #[cfg(feature = "elevation")]
        let elevation_fans = cfg.elevation.is_some();
        #[cfg(not(feature = "elevation"))]
        let elevation_fans = false;
        let sharing = if cfg.exporters.len() >= 2 || elevation_fans {
            RecordSharing::Arc
        } else {
            cfg.record_sharing
        };
        builder = builder.record_sharing(sharing);
        builder = builder.overflow(cfg.overflow);
        builder = builder.managed_drainer(cfg.managed_drainer);
        builder = builder.drain_batch(cfg.drain_batch);
        builder = builder.assist_batch(cfg.assist_batch);
        builder =
            builder.flush_interval(core::time::Duration::from_micros(cfg.flush_interval_micros));
        builder
    }

    /// Inverse of `from_config`.
    pub fn to_config(&self) -> TelemetryConfig {
        TelemetryConfig {
            ring_spans: self.ring_caps().spans,
            ring_events: self.ring_caps().events,
            ring_logs: self.ring_caps().logs,
            ring_metrics: self.ring_caps().metrics,
            ring_links: self.ring_caps().links,
            ring_overflow_attrs: self.ring_caps().overflow_attrs,
            core_count: self.core_count(),
            exporter: ExporterChoice::Noop,
            exporters: Vec::new(),
            resource: self
                .resource()
                .tags()
                .iter()
                .filter_map(|tag| match tag {
                    Tag::Scalar {
                        key,
                        value: ScalarValue::Str(value),
                    } => Some(ResourceTag {
                        key: (*key).to_string(),
                        value: (*value).to_string(),
                    }),
                    _ => None,
                })
                .collect(),
            sampler: None,
            elevation: None,
            record_sharing: RecordSharing::default(),
            overflow: self.overflow(),
            managed_drainer: self.is_managed_drainer(),
            drain_batch: self.drain_batch(),
            assist_batch: self.assist_batch(),
            flush_interval_micros: self.flush_interval().as_micros() as u64,
        }
    }
}

/// Rewrap a terminal pipe with the error-elevation fan-out and arm the verbose
/// sampler. `None` returns the terminal pipe untouched (the simple form).
///
/// The fan is `[FloorFilter -> terminal, ElevationSink]`: arm A (primary) exports
/// floor+ exactly as before; arm B buffers verbose-sampled traces and replays a
/// triggered tree to the separate elevated exporter. `0` retention caps resolve
/// to the build-time `sized` defaults (principle 12: one source of truth).
#[cfg(feature = "elevation")]
fn install_elevation(cfg: &TelemetryConfig, terminal: TelemetryPipeHandle) -> TelemetryPipeHandle {
    let Some(elevation) = &cfg.elevation else {
        return terminal;
    };
    crate::current::set_verbose_ratio(elevation.sample_ratio);
    crate::current::set_verbose_admit_floor(elevation.resolved_elevated());
    let floor_arm =
        into_telemetry_handle(crate::pipes::FloorFilter::new(elevation.floor, terminal));
    let elevated = pipe_from_choice(&elevation.exporter);
    let max_traces = if elevation.retention.max_traces == 0 {
        crate::sized::ELEVATION_MAX_TRACES
    } else {
        elevation.retention.max_traces
    };
    let per_trace_ring = if elevation.retention.per_trace_ring == 0 {
        crate::sized::ELEVATION_PER_TRACE_RING
    } else {
        elevation.retention.per_trace_ring
    };
    let ttl_ns = elevation.retention.ttl_millis.saturating_mul(1_000_000);
    let sink = into_telemetry_handle(crate::pipes::ElevationSink::new(
        elevated,
        elevation.trigger_level,
        per_trace_ring,
        max_traces,
        ttl_ns,
        elevation.retention.drain_on_root_close,
    ));
    fan_exporters(alloc::vec![floor_arm, sink])
}

/// Resolve the recorder's terminal pipe from a config: the single `exporter`
/// when `exporters` is empty (byte-identical to the legacy path), else a
/// `FanExporter` over the `exporters` list.
fn pipe_from_config(cfg: &TelemetryConfig) -> TelemetryPipeHandle {
    if cfg.exporters.is_empty() {
        return pipe_from_choice(&cfg.exporter);
    }
    fan_exporters(cfg.exporters.iter().map(pipe_from_choice).collect())
}

// endpoint validation only exists for the transport exporters, which are
// feature-gated; with neither otlp feature the params are genuinely unused and
// the Vec is never pushed to (so ptr_arg would suggest a slice).
#[cfg_attr(
    not(any(feature = "otlp-http", feature = "otlp-grpc")),
    allow(unused_variables, clippy::ptr_arg)
)]
fn push_exporter_endpoint_errors(
    choice: &ExporterChoice,
    label: &str,
    errors: &mut Vec<ValidationMessage>,
) {
    match choice {
        #[cfg(feature = "otlp-http")]
        ExporterChoice::OtlpHttp { endpoint } if endpoint.is_empty() => {
            errors.push(ValidationMessage::new(label, "must be non-empty"));
        }
        #[cfg(feature = "otlp-grpc")]
        ExporterChoice::OtlpGrpc { endpoint } if endpoint.is_empty() => {
            errors.push(ValidationMessage::new(label, "must be non-empty"));
        }
        _ => {}
    }
}

fn pipe_from_choice(choice: &ExporterChoice) -> TelemetryPipeHandle {
    match choice {
        ExporterChoice::Noop => into_telemetry_handle(NullPipe::new()),
        #[cfg(feature = "otlp-http")]
        ExporterChoice::OtlpHttp { endpoint } => {
            into_telemetry_handle(crate::pipes::OtlpHttpPipe::new(endpoint.clone()))
        }
        #[cfg(feature = "otlp-grpc")]
        ExporterChoice::OtlpGrpc { endpoint } => {
            into_telemetry_handle(crate::pipes::OtlpGrpcPipe::new(endpoint.clone()))
        }
    }
}

fn leak_str(value: &str) -> &'static str {
    alloc::boxed::Box::leak(value.to_string().into_boxed_str())
}

/// Fluent builder for [`TelemetryConfig`]. Every source (`.from_path`,
/// `.from_env`, `.underlay_path`, `.underlay_env`, `.with_*`) contributes
/// only the fields it actually specifies, merged onto the accumulated
/// config — a field a source doesn't touch falls through to whatever prior
/// layers set. `.from_path`/`.from_env` override (last writer wins per
/// field); `.underlay_path`/`.underlay_env` fill only fields still unset;
/// `.with_*` always acts as an override at its call position.
pub struct TelemetryLayerBuilder {
    inner: TelemetryConfig,
    touched: BTreeSet<String>,
}

impl TelemetryConfig {
    pub fn layered() -> TelemetryLayerBuilder {
        TelemetryLayerBuilder {
            inner: TelemetryConfig::default(),
            touched: BTreeSet::new(),
        }
    }
}

impl TelemetryLayerBuilder {
    /// Merge a file's fields onto the accumulated config; the file wins for
    /// every field it specifies.
    pub fn from_path<P: AsRef<std::path::Path>>(
        mut self,
        path: P,
    ) -> Result<Self, conflaguration::Error> {
        let incoming: Value = conflaguration::from_file(path.as_ref())?;
        apply_layer(
            &mut self.inner,
            &mut self.touched,
            incoming,
            MergeMode::Override,
            &[],
        )?;
        Ok(self)
    }

    /// Fill any still-unset fields from a file; already-set fields are left
    /// untouched.
    pub fn underlay_path<P: AsRef<std::path::Path>>(
        mut self,
        path: P,
    ) -> Result<Self, conflaguration::Error> {
        let incoming: Value = conflaguration::from_file(path.as_ref())?;
        apply_layer(
            &mut self.inner,
            &mut self.touched,
            incoming,
            MergeMode::Underlay,
            &[],
        )?;
        Ok(self)
    }

    /// Merge env-set fields onto the accumulated config; env wins for every
    /// field it sets. Unset env vars leave the current value untouched.
    pub fn from_env(mut self) -> Result<Self, conflaguration::Error> {
        let incoming = telemetry_env_partial()?;
        apply_layer(
            &mut self.inner,
            &mut self.touched,
            incoming,
            MergeMode::Override,
            &[],
        )?;
        Ok(self)
    }

    /// Fill any still-unset fields from env vars; already-set fields are left
    /// untouched even if the matching env var is set.
    pub fn underlay_env(mut self) -> Result<Self, conflaguration::Error> {
        let incoming = telemetry_env_partial()?;
        apply_layer(
            &mut self.inner,
            &mut self.touched,
            incoming,
            MergeMode::Underlay,
            &[],
        )?;
        Ok(self)
    }

    pub fn with_sampler(mut self, sampler: SamplerSpec) -> Self {
        self.inner.sampler = Some(sampler);
        self.touched.insert("sampler".to_string());
        self
    }

    pub fn with_no_sampler(mut self) -> Self {
        self.inner.sampler = None;
        self.touched.insert("sampler".to_string());
        self
    }

    pub fn with_record_sharing(mut self, sharing: RecordSharing) -> Self {
        self.inner.record_sharing = sharing;
        self.touched.insert("record_sharing".to_string());
        self
    }

    /// Install the error-elevation policy (the `Some` arm). Off by default; see
    /// [`Elevation`].
    pub fn with_elevation(mut self, elevation: Elevation) -> Self {
        self.inner.elevation = Some(elevation);
        self.touched.insert("elevation".to_string());
        self
    }

    /// Collapse back to the simple form — no per-trace buffer stage.
    pub fn with_no_elevation(mut self) -> Self {
        self.inner.elevation = None;
        self.touched.insert("elevation".to_string());
        self
    }

    /// Opt the ring overflow policy in or out of losslessness. `Block` (default)
    /// never drops; `Drop` sheds and counts under overload. See [`OverflowPolicy`].
    pub fn with_overflow(mut self, overflow: OverflowPolicy) -> Self {
        self.inner.overflow = overflow;
        self.touched.insert("overflow".to_string());
        self
    }

    pub fn with_ring_spans(mut self, capacity: usize) -> Self {
        self.inner.ring_spans = capacity;
        self.touched.insert("ring_spans".to_string());
        self
    }

    pub fn with_ring_events(mut self, capacity: usize) -> Self {
        self.inner.ring_events = capacity;
        self.touched.insert("ring_events".to_string());
        self
    }

    pub fn with_ring_logs(mut self, capacity: usize) -> Self {
        self.inner.ring_logs = capacity;
        self.touched.insert("ring_logs".to_string());
        self
    }

    pub fn with_ring_metrics(mut self, capacity: usize) -> Self {
        self.inner.ring_metrics = capacity;
        self.touched.insert("ring_metrics".to_string());
        self
    }

    pub fn with_ring_links(mut self, capacity: usize) -> Self {
        self.inner.ring_links = capacity;
        self.touched.insert("ring_links".to_string());
        self
    }

    pub fn with_ring_overflow_attrs(mut self, capacity: usize) -> Self {
        self.inner.ring_overflow_attrs = capacity;
        self.touched.insert("ring_overflow_attrs".to_string());
        self
    }

    pub fn with_core_count(mut self, count: usize) -> Self {
        self.inner.core_count = count;
        self.touched.insert("core_count".to_string());
        self
    }

    pub fn with_exporter(mut self, exporter: ExporterChoice) -> Self {
        self.inner.exporter = exporter;
        self.touched.insert("exporter".to_string());
        self
    }

    pub fn with_exporters(mut self, exporters: Vec<ExporterChoice>) -> Self {
        self.inner.exporters = exporters;
        self.touched.insert("exporters".to_string());
        self
    }

    pub fn with_resource(mut self, resource: Vec<ResourceTag>) -> Self {
        self.inner.resource = resource;
        self.touched.insert("resource".to_string());
        self
    }

    pub fn build(self) -> TelemetryConfig {
        self.inner
    }
}

/// The env-set subset of [`TelemetryConfig`]'s fields, as a partial JSON
/// object containing only the fields whose env var is actually present —
/// never the ones `Settings::from_env` filled with a default.
fn telemetry_env_partial() -> Result<Value, conflaguration::Error> {
    let resolved = TelemetryConfig::from_env()?;
    let mut partial = Map::new();
    insert_if_env_set(
        &mut partial,
        "ring_spans",
        &["TELEMETRY_RING_SPANS"],
        &resolved.ring_spans,
    )?;
    insert_if_env_set(
        &mut partial,
        "ring_events",
        &["TELEMETRY_RING_EVENTS"],
        &resolved.ring_events,
    )?;
    insert_if_env_set(
        &mut partial,
        "ring_logs",
        &["TELEMETRY_RING_LOGS"],
        &resolved.ring_logs,
    )?;
    insert_if_env_set(
        &mut partial,
        "ring_metrics",
        &["TELEMETRY_RING_METRICS"],
        &resolved.ring_metrics,
    )?;
    insert_if_env_set(
        &mut partial,
        "ring_links",
        &["TELEMETRY_RING_LINKS"],
        &resolved.ring_links,
    )?;
    insert_if_env_set(
        &mut partial,
        "ring_overflow_attrs",
        &["TELEMETRY_RING_OVERFLOW_ATTRS"],
        &resolved.ring_overflow_attrs,
    )?;
    insert_if_env_set(
        &mut partial,
        "core_count",
        &["TELEMETRY_CORE_COUNT"],
        &resolved.core_count,
    )?;
    insert_if_env_set(
        &mut partial,
        "managed_drainer",
        &["TELEMETRY_MANAGED_DRAINER"],
        &resolved.managed_drainer,
    )?;
    insert_if_env_set(
        &mut partial,
        "drain_batch",
        &["TELEMETRY_DRAIN_BATCH"],
        &resolved.drain_batch,
    )?;
    insert_if_env_set(
        &mut partial,
        "assist_batch",
        &["TELEMETRY_ASSIST_BATCH"],
        &resolved.assist_batch,
    )?;
    insert_if_env_set(
        &mut partial,
        "flush_interval_micros",
        &["TELEMETRY_FLUSH_INTERVAL_MICROS"],
        &resolved.flush_interval_micros,
    )?;
    Ok(Value::Object(partial))
}

// these tests build a real Recorder from config, which uses proxima-core's
// Ring/StaticRing internally -- cfg-swapped to loom under `--features
// loom` (forwarded via proxima-core/loom), only usable inside an actual
// loom::model(...) closure, which these plain #[test] functions don't
// provide.
#[cfg(all(test, not(feature = "loom")))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn default_config_validates() {
        let cfg = TelemetryConfig::default();
        assert!(cfg.validate().is_ok(), "default config should validate");
        assert_eq!(cfg.ring_spans, 4096);
        assert_eq!(cfg.core_count, 0);
        assert!(matches!(cfg.exporter, ExporterChoice::Noop));
        assert_eq!(
            cfg.overflow,
            OverflowPolicy::Block,
            "Block (lossless elastic) is the default overflow policy"
        );
        assert!(!cfg.managed_drainer, "managed drainer is off by default");
    }

    #[cfg(feature = "lossless-backpressure")]
    #[test]
    fn managed_drainer_round_trips_through_config() {
        let cfg = TelemetryConfig::builder().managed_drainer(true).build();
        let recorder = Recorder::from_config(&cfg).start().expect("start");
        assert!(recorder.is_managed_drainer(), "config opted the pump on");
        assert!(
            recorder.to_config().managed_drainer,
            "built recorder round-trips the managed_drainer flag back to config"
        );
    }

    #[test]
    fn overflow_policy_round_trips_through_config() {
        let cfg = TelemetryConfig::builder()
            .overflow(OverflowPolicy::Drop)
            .build();
        let recorder = Recorder::from_config(&cfg).start().expect("start");
        assert_eq!(recorder.overflow(), OverflowPolicy::Drop);
        assert_eq!(recorder.to_config().overflow, OverflowPolicy::Drop);
    }

    // build-time sizing TOML (crate::sized) is the single source of truth for the
    // batch defaults; the serde/builder defaults reference it, the `#[setting]`
    // env-overlay literal mirrors it. This guards against the literal drifting
    // from the const (e.g. after a PROXIMA_TELEMETRY_* build-time override).
    #[test]
    fn setting_defaults_match_sized() {
        let cfg = TelemetryConfig::default();
        assert_eq!(cfg.drain_batch, crate::sized::DRAIN_BATCH);
        assert_eq!(cfg.assist_batch, crate::sized::DRAIN_ASSIST_BATCH);
        assert_eq!(
            cfg.flush_interval_micros,
            crate::sized::FLUSH_INTERVAL_MICROS
        );
        // the env-overlay (from_env, no vars set) must agree with the const too.
        temp_env::with_vars::<&str, &str, _, _>([], || {
            let from_env = TelemetryConfig::from_env().expect("from_env");
            assert_eq!(
                from_env.drain_batch,
                crate::sized::DRAIN_BATCH,
                "#[setting] drain_batch literal drifted from sized::DRAIN_BATCH"
            );
            assert_eq!(
                from_env.assist_batch,
                crate::sized::DRAIN_ASSIST_BATCH,
                "#[setting] assist_batch literal drifted from sized::DRAIN_ASSIST_BATCH"
            );
            assert_eq!(
                from_env.flush_interval_micros,
                crate::sized::FLUSH_INTERVAL_MICROS,
                "#[setting] flush_interval_micros literal drifted from sized::FLUSH_INTERVAL_MICROS"
            );
        });
    }

    #[test]
    fn batch_sizes_round_trip_through_config() {
        let cfg = TelemetryConfig::builder()
            .drain_batch(1024)
            .assist_batch(32)
            .flush_interval_micros(2000)
            .build();
        let recorder = Recorder::from_config(&cfg).start().expect("start");
        assert_eq!(recorder.drain_batch(), 1024);
        assert_eq!(recorder.assist_batch(), 32);
        assert_eq!(
            recorder.flush_interval(),
            core::time::Duration::from_micros(2000)
        );
        let restored = recorder.to_config();
        assert_eq!(restored.drain_batch, 1024);
        assert_eq!(restored.assist_batch, 32);
        assert_eq!(restored.flush_interval_micros, 2000);
    }

    #[test]
    fn zero_flush_interval_rejected() {
        let cfg = TelemetryConfig::builder().flush_interval_micros(0).build();
        let err = cfg
            .validate()
            .expect_err("validate must reject 0 flush interval");
        assert!(format!("{err:?}").contains("flush_interval_micros"));
    }

    #[test]
    fn non_power_of_two_ring_rejected() {
        let cfg = TelemetryConfig::builder().ring_spans(3000).build();
        let err = cfg.validate().expect_err("validate must reject 3000");
        let message = format!("{err:?}");
        assert!(message.contains("ring_spans"), "got: {message}");
    }

    #[test]
    fn zero_ring_rejected() {
        let cfg = TelemetryConfig::builder().ring_events(0).build();
        assert!(cfg.validate().is_err(), "validate must reject 0");
    }

    #[test]
    fn round_trip_default_matches() {
        let cfg = TelemetryConfig::default();
        let recorder = Recorder::from_config(&cfg).start().expect("start");
        let restored = recorder.to_config();
        assert_eq!(restored.ring_spans, cfg.ring_spans);
        assert_eq!(restored.ring_events, cfg.ring_events);
        assert_eq!(restored.ring_logs, cfg.ring_logs);
        assert_eq!(restored.ring_metrics, cfg.ring_metrics);
        assert_eq!(restored.ring_links, cfg.ring_links);
        assert_eq!(restored.ring_overflow_attrs, cfg.ring_overflow_attrs);
        assert!(
            restored.core_count >= 1,
            "core_count auto-resolves to at least 1"
        );
    }

    #[test]
    fn round_trip_resource_attrs() {
        let cfg = TelemetryConfig::builder()
            .resource(alloc::vec![
                ResourceTag {
                    key: "service.name".to_string(),
                    value: "judi-api".to_string()
                },
                ResourceTag {
                    key: "service.version".to_string(),
                    value: "1.2.0".to_string()
                },
            ])
            .build();
        let recorder = Recorder::from_config(&cfg).start().expect("start");
        let restored = recorder.to_config();
        assert_eq!(restored.resource.len(), 2);
    }

    #[test]
    fn explicit_core_count_round_trips() {
        let cfg = TelemetryConfig::builder().core_count(3).build();
        let recorder = Recorder::from_config(&cfg).start().expect("start");
        assert_eq!(recorder.to_config().core_count, 3);
    }

    #[test]
    fn builder_starts_at_default() {
        let from_layered = TelemetryConfig::layered().build();
        let from_default = TelemetryConfig::default();
        assert_eq!(from_layered.ring_spans, from_default.ring_spans);
        assert_eq!(from_layered.sampler, from_default.sampler);
        assert_eq!(from_layered.record_sharing, from_default.record_sharing);
    }

    #[test]
    fn with_overrides_default() {
        // record_sharing is the shared-vs-per-sink fan-out axis (Arc shares the
        // batch across sinks; Inline gives each its own); the builder overrides it.
        let cfg = TelemetryConfig::layered()
            .with_record_sharing(RecordSharing::Arc)
            .build();
        assert_eq!(cfg.record_sharing, RecordSharing::Arc);
        assert_eq!(
            TelemetryConfig::default().record_sharing,
            RecordSharing::Inline
        );
    }

    #[test]
    fn from_path_overrides_default() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let path = dir.path().join("telemetry.toml");
        std::fs::write(
            &path,
            "ring_spans = 8192
",
        )
        .expect("write toml");
        let cfg = TelemetryConfig::layered()
            .from_path(&path)
            .expect("from_path")
            .build();
        assert_eq!(cfg.ring_spans, 8192);
        assert_eq!(cfg.ring_events, 4096);
    }

    #[test]
    fn from_env_overlays_via_conflaguration() {
        temp_env::with_vars([("TELEMETRY_RING_SPANS", Some("16384"))], || {
            let cfg = TelemetryConfig::layered()
                .from_env()
                .expect("from_env")
                .build();
            assert_eq!(cfg.ring_spans, 16384);
        });
    }

    // the exact seam-#3 case: a file sets TWO fields, env sets only ONE —
    // the file's other field must survive `.from_path().from_env()`.
    #[test]
    fn seam_3_from_path_then_from_env_preserves_files_untouched_field() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let path = dir.path().join("telemetry.toml");
        std::fs::write(&path, "ring_spans = 8192\nring_events = 2048\n").expect("write toml");
        temp_env::with_vars([("TELEMETRY_RING_SPANS", Some("16384"))], || {
            let cfg = TelemetryConfig::layered()
                .from_path(&path)
                .expect("from_path")
                .from_env()
                .expect("from_env")
                .build();
            assert_eq!(cfg.ring_spans, 16384, "env wins the field it sets");
            assert_eq!(cfg.ring_events, 2048, "the file's field must survive");
        });
    }

    // order-independence: the same two sources, built both orders, resolve
    // correctly per field for that order.
    #[test]
    fn order_independence_file_then_env_vs_env_then_file() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let path = dir.path().join("telemetry.toml");
        std::fs::write(&path, "ring_events = 2048\n").expect("write toml");
        temp_env::with_vars([("TELEMETRY_RING_SPANS", Some("16384"))], || {
            let file_then_env = TelemetryConfig::layered()
                .from_path(&path)
                .expect("from_path")
                .from_env()
                .expect("from_env")
                .build();
            assert_eq!(file_then_env.ring_events, 2048, "file's field survives");
            assert_eq!(file_then_env.ring_spans, 16384, "env's field applies");

            let env_then_file = TelemetryConfig::layered()
                .from_env()
                .expect("from_env")
                .from_path(&path)
                .expect("from_path")
                .build();
            assert_eq!(env_then_file.ring_spans, 16384, "env's field survives");
            assert_eq!(env_then_file.ring_events, 2048, "file's field applies");
        });
    }

    // full stack: defaults < file < env < with_*, each overriding only the
    // field it sets.
    #[test]
    fn full_stack_defaults_file_env_with_override_each_field() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let path = dir.path().join("telemetry.toml");
        std::fs::write(&path, "ring_logs = 512\nring_events = 2048\n").expect("write toml");
        temp_env::with_vars([("TELEMETRY_RING_SPANS", Some("16384"))], || {
            let cfg = TelemetryConfig::layered()
                .from_path(&path)
                .expect("from_path")
                .from_env()
                .expect("from_env")
                .with_core_count(3)
                .build();
            assert_eq!(cfg.ring_spans, 16384, "env layer");
            assert_eq!(cfg.ring_logs, 512, "file layer");
            assert_eq!(cfg.ring_events, 2048, "file layer");
            assert_eq!(cfg.core_count, 3, "with_* layer");
            assert_eq!(
                cfg.ring_metrics, 8192,
                "untouched — falls through to the default"
            );
        });
    }

    // underlay never clobbers an already-set field; it DOES fill an unset one.
    #[test]
    fn underlay_path_fills_only_unset_fields() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let path = dir.path().join("telemetry.toml");
        std::fs::write(&path, "ring_spans = 1024\nring_events = 1024\n").expect("write toml");
        let cfg = TelemetryConfig::layered()
            .with_ring_spans(64)
            .underlay_path(&path)
            .expect("underlay_path")
            .build();
        assert_eq!(
            cfg.ring_spans, 64,
            "already set by with_*; the file's value is dropped"
        );
        assert_eq!(
            cfg.ring_events, 1024,
            "unset before underlay; the file fills it"
        );
    }

    #[test]
    fn underlay_env_fills_only_unset_fields() {
        temp_env::with_vars(
            [
                ("TELEMETRY_RING_SPANS", Some("1024")),
                ("TELEMETRY_RING_EVENTS", Some("2048")),
            ],
            || {
                let cfg = TelemetryConfig::layered()
                    .with_ring_spans(64)
                    .underlay_env()
                    .expect("underlay_env")
                    .build();
                assert_eq!(cfg.ring_spans, 64, "already set; env's value is dropped");
                assert_eq!(
                    cfg.ring_events, 2048,
                    "unset before underlay_env; env fills it"
                );
            },
        );
    }

    // order-independence for underlay: the first-applied source wins for a
    // field both specify.
    #[test]
    fn order_independence_underlay_flavor_first_setter_wins_either_direction() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let path = dir.path().join("telemetry.toml");
        std::fs::write(&path, "ring_spans = 1024\n").expect("write toml");
        temp_env::with_vars([("TELEMETRY_RING_SPANS", Some("16384"))], || {
            let file_first = TelemetryConfig::layered()
                .underlay_path(&path)
                .expect("underlay_path")
                .underlay_env()
                .expect("underlay_env")
                .build();
            assert_eq!(file_first.ring_spans, 1024, "file applied first, wins");

            let env_first = TelemetryConfig::layered()
                .underlay_env()
                .expect("underlay_env")
                .underlay_path(&path)
                .expect("underlay_path")
                .build();
            assert_eq!(env_first.ring_spans, 16384, "env applied first, wins");
        });
    }

    // combined: defaults -> underlay(file) -> override(env) -> override(with_*).
    #[test]
    fn combined_underlay_file_then_override_env_then_with() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let path = dir.path().join("telemetry.toml");
        std::fs::write(&path, "ring_spans = 1024\nring_logs = 256\n").expect("write toml");
        temp_env::with_vars([("TELEMETRY_RING_SPANS", Some("16384"))], || {
            let cfg = TelemetryConfig::layered()
                .underlay_path(&path)
                .expect("underlay_path")
                .from_env()
                .expect("from_env")
                .with_ring_logs(999)
                .build();
            assert_eq!(
                cfg.ring_spans, 16384,
                "override(env) wins over underlay(file)"
            );
            assert_eq!(
                cfg.ring_logs, 999,
                "the later with_* overrides underlay(file)"
            );
            assert_eq!(
                cfg.ring_metrics, 8192,
                "untouched — falls through to the default"
            );
        });
    }

    // collection (Vec) replace-if-present, and underlay never element-merges.
    #[test]
    fn resource_collection_replaces_wholesale_and_underlay_never_element_merges() {
        let first = alloc::vec![ResourceTag {
            key: "service.name".to_string(),
            value: "a".to_string(),
        }];
        let second = alloc::vec![
            ResourceTag {
                key: "k1".to_string(),
                value: "v1".to_string()
            },
            ResourceTag {
                key: "k2".to_string(),
                value: "v2".to_string()
            },
        ];

        let overridden = TelemetryConfig::layered()
            .with_resource(first.clone())
            .with_resource(second.clone())
            .build();
        assert_eq!(
            overridden.resource, second,
            "second with_* replaces wholesale, not unions"
        );

        let underlaid = TelemetryConfig::layered()
            .with_resource(first.clone())
            .build();
        // underlay has no `.underlay_resource()` (resource is set only via
        // `.with_*`/file), but the same rule applies at the file layer: an
        // already-set collection is never merged, only ever replaced by a
        // LATER override. Assert the already-set value survives an
        // unrelated underlay_env() call that doesn't touch `resource`.
        temp_env::with_vars::<&str, &str, _, _>([], || {
            let after_underlay_env = TelemetryConfig::layered()
                .with_resource(first.clone())
                .underlay_env()
                .expect("underlay_env")
                .build();
            assert_eq!(after_underlay_env.resource, first);
        });
        assert_eq!(underlaid.resource, first);
    }

    // Env-isolation contract: `from_env()`/`layered().from_env()` are the ONLY
    // process-env readers in this crate. A recorder built from explicit values
    // (fluent builder OR `from_config`) ignores TELEMETRY_* / PROXIMA_TELEMETRY_*
    // entirely. This is the guarantee a re-branding downstream relies on: it
    // builds explicitly from its OWN config so a proxima/telemetry env can never
    // silently "work" in that downstream. A future change that taught the builder
    // to read env would break this test (by design).
    #[test]
    fn explicit_config_ignores_telemetry_env() {
        temp_env::with_vars(
            [
                ("TELEMETRY_DRAIN_BATCH", Some("9999")),
                ("TELEMETRY_RING_SPANS", Some("9999")),
                ("PROXIMA_TELEMETRY_DRAIN_BATCH", Some("9999")),
            ],
            || {
                let recorder = Recorder::builder()
                    .pipe(crate::pipes::NullPipe::new())
                    .drain_batch(77)
                    .start()
                    .expect("recorder");
                assert_eq!(
                    recorder.drain_batch(),
                    77,
                    "builder value wins; env ignored"
                );
                assert_ne!(
                    recorder.drain_batch(),
                    9999,
                    "TELEMETRY_/PROXIMA_ env must NOT leak in"
                );

                let cfg = TelemetryConfig::builder().drain_batch(88).build();
                assert_eq!(cfg.drain_batch, 88, "builder build() does not read env");
                let from_cfg = Recorder::from_config(&cfg).start().expect("from_config");
                assert_eq!(
                    from_cfg.drain_batch(),
                    88,
                    "from_config value wins; env ignored"
                );
            },
        );
    }

    #[test]
    fn builder_complete_axis_coverage() {
        let cfg = TelemetryConfig::layered()
            .with_ring_spans(4096)
            .with_ring_events(4096)
            .with_ring_logs(4096)
            .with_ring_metrics(8192)
            .with_ring_links(1024)
            .with_ring_overflow_attrs(2048)
            .with_core_count(0)
            .with_exporter(ExporterChoice::Noop)
            .with_resource(alloc::vec![])
            .with_sampler(SamplerSpec::AlwaysOn)
            .with_no_sampler()
            .with_record_sharing(RecordSharing::Arc)
            .build();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn multi_exporter_config_fans_and_runs() {
        let cfg = TelemetryConfig::builder()
            .exporters(alloc::vec![ExporterChoice::Noop, ExporterChoice::Noop])
            .record_sharing(RecordSharing::Arc)
            .core_count(1)
            .build();
        assert!(cfg.validate().is_ok(), "two Noop exporters validate");
        assert_eq!(cfg.exporters.len(), 2);

        let recorder = Recorder::from_config(&cfg).start().expect("start");
        for _ in 0..16 {
            drop(recorder.span("op").start());
        }
        recorder.drain();
        assert_eq!(recorder.dropped(), 0, "the fan accepted every record");
    }

    #[test]
    fn fan_out_intent_drives_arc_sharing() {
        // 2 exporters = fan-out, with record_sharing left at the default Inline:
        // from_config must DERIVE Arc so the fan clones cheaply, not deep-copy.
        let cfg = TelemetryConfig::builder()
            .exporters(alloc::vec![ExporterChoice::Noop, ExporterChoice::Noop])
            .core_count(1)
            .build();
        assert_eq!(
            cfg.record_sharing,
            RecordSharing::Inline,
            "config field is the untouched default"
        );
        let recorder = Recorder::from_config(&cfg).start().expect("start");
        assert_eq!(
            recorder.sharing(),
            RecordSharing::Arc,
            "fan-out drives Arc sharing"
        );

        // a single sink stays Inline (Arc there is pure waste).
        let single = TelemetryConfig::builder().core_count(1).build();
        let recorder = Recorder::from_config(&single).start().expect("start");
        assert_eq!(
            recorder.sharing(),
            RecordSharing::Inline,
            "single sink stays Inline"
        );
    }

    #[test]
    fn empty_exporters_uses_single_exporter_path() {
        let cfg = TelemetryConfig::default();
        assert!(cfg.exporters.is_empty(), "default fans nothing");
        let recorder = Recorder::from_config(&cfg).start().expect("start");
        drop(recorder.span("op").start());
        recorder.drain();
    }

    #[test]
    fn exporters_round_trip_through_serde() {
        let cfg = TelemetryConfig::builder()
            .exporters(alloc::vec![ExporterChoice::Noop, ExporterChoice::Noop])
            .build();
        let json = serde_json::to_string(&cfg).expect("serialize");
        let back: TelemetryConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(
            back.exporters, cfg.exporters,
            "exporters list survives serde"
        );
        assert_eq!(back.exporters.len(), 2);
    }

    fn info_floor_trace_elevated(sample_ratio: f64) -> Elevation {
        Elevation {
            floor: Level::INFO,
            elevated: Some(Level::TRACE),
            sample_ratio,
            trigger_level: Level::ERROR,
            exporter: ExporterChoice::Noop,
            retention: Retention::default(),
        }
    }

    #[test]
    fn default_config_is_the_simple_form() {
        // the collapse-to-simple-form guarantee: no elevation by default.
        assert!(
            TelemetryConfig::default().elevation.is_none(),
            "None (simple form) is the default — no per-trace buffer stage"
        );
    }

    #[test]
    fn elevation_round_trips_through_serde() {
        let cfg = TelemetryConfig::builder()
            .elevation(info_floor_trace_elevated(0.01))
            .build();
        let toml_ish = serde_json::to_string(&cfg).expect("serialize");
        let back: TelemetryConfig = serde_json::from_str(&toml_ish).expect("deserialize");
        let elevation = back.elevation.expect("elevation survives serde");
        assert_eq!(elevation.floor, Level::INFO);
        assert_eq!(
            elevation.elevated,
            Some(Level::TRACE),
            "levels serialize by name"
        );
        assert_eq!(elevation.trigger_level, Level::ERROR, "trigger default is error");
        assert!((elevation.sample_ratio - 0.01).abs() < f64::EPSILON);
    }

    // the "configurable in a file" requirement: an [elevation] table naming
    // only sample_ratio must load, with every other Elevation field falling
    // through to its serde default (floor -> info, elevated -> None ->
    // resolves to floor) — the per-field #[serde(default)] annotations, not a
    // wholesale struct default, are what make a partial table loadable.
    #[test]
    fn elevation_loads_from_toml_with_defaults() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let path = dir.path().join("telemetry.toml");
        std::fs::write(
            &path,
            "[elevation]
sample_ratio = 0.02
",
        )
        .expect("write toml");
        let cfg = TelemetryConfig::layered()
            .from_path(&path)
            .expect("from_path")
            .build();
        let elevation = cfg.elevation.expect("elevation table loaded");
        assert_eq!(elevation.floor, Level::INFO, "floor defaults to info");
        assert_eq!(
            elevation.resolved_elevated(),
            Level::INFO,
            "unset elevated defaults to floor"
        );
        assert!((elevation.sample_ratio - 0.02).abs() < f64::EPSILON);
    }

    #[test]
    fn elevated_defaults_to_floor() {
        let elevation = Elevation {
            floor: Level::WARN,
            elevated: None,
            sample_ratio: 0.01,
            trigger_level: Level::ERROR,
            exporter: ExporterChoice::Noop,
            retention: Retention::default(),
        };
        assert_eq!(
            elevation.resolved_elevated(),
            Level::WARN,
            "unset elevated resolves to floor — no extra depth by default"
        );
    }

    #[cfg(feature = "elevation")]
    #[test]
    fn sane_elevation_validates_under_feature() {
        let cfg = TelemetryConfig::builder()
            .elevation(info_floor_trace_elevated(0.01))
            .build();
        assert!(cfg.validate().is_ok(), "a sane elevation validates when compiled in");
    }

    #[cfg(not(feature = "elevation"))]
    #[test]
    fn elevation_set_without_feature_is_rejected() {
        // the compile-time collapse: a policy the build can't honour fails loud.
        let cfg = TelemetryConfig::builder()
            .elevation(info_floor_trace_elevated(0.01))
            .build();
        let err = cfg
            .validate()
            .expect_err("elevation set on a feature-off build must be rejected");
        assert!(format!("{err:?}").contains("elevation"), "got: {err:?}");
    }

    #[test]
    fn elevation_level_names_are_the_wire_form() {
        // floor/elevated read as level names in a config file, not integers.
        let json = serde_json::to_string(&info_floor_trace_elevated(0.05)).expect("serialize");
        assert!(json.contains("\"info\""), "floor serialized by name: {json}");
        assert!(json.contains("\"trace\""), "elevated serialized by name: {json}");
    }

    #[test]
    fn validate_rejects_elevated_coarser_than_floor() {
        // elevated=warn is COARSER than floor=info — nothing extra would buffer.
        let cfg = TelemetryConfig::builder()
            .elevation(Elevation {
                floor: Level::INFO,
                elevated: Some(Level::WARN),
                sample_ratio: 0.1,
                trigger_level: Level::ERROR,
                exporter: ExporterChoice::Noop,
                retention: Retention::default(),
            })
            .build();
        let err = cfg.validate().expect_err("coarser elevated must be rejected");
        assert!(format!("{err:?}").contains("elevated"), "got: {err:?}");
    }

    #[test]
    fn validate_rejects_out_of_range_sample_ratio() {
        let cfg = TelemetryConfig::builder()
            .elevation(info_floor_trace_elevated(1.5))
            .build();
        let err = cfg.validate().expect_err("ratio > 1 must be rejected");
        assert!(format!("{err:?}").contains("sample_ratio"), "got: {err:?}");
    }

    #[test]
    fn layered_with_elevation_then_no_elevation_collapses() {
        let cfg = TelemetryConfig::layered()
            .with_elevation(info_floor_trace_elevated(0.02))
            .with_no_elevation()
            .build();
        assert!(cfg.elevation.is_none(), "later with_no_elevation wins");
    }

    #[cfg(feature = "otlp-http")]
    #[test]
    fn validate_rejects_empty_endpoint_in_exporters_list() {
        let cfg = TelemetryConfig::builder()
            .exporters(alloc::vec![ExporterChoice::OtlpHttp {
                endpoint: alloc::string::String::new(),
            }])
            .build();
        assert!(
            cfg.validate().is_err(),
            "an empty endpoint inside the exporters list is rejected like the single exporter"
        );
    }
}
