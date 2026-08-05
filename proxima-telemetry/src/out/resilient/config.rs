//! First-class config for the resilient OTLP sink: buffer sizing, backoff
//! bounds, and the per-severity retention horizon table.
//!
//! Mirrors `proxima_listen::config::ListenTuningConfig`'s shape (a flat,
//! standalone-usable tunable config): `#[derive(Builder, Deserialize,
//! Serialize, Settings)]` + [`Validate`], a `ResilientOtlpConfigLayerBuilder`
//! with call-order precedence (`.with_*` before `.from_path`/`.from_env` ==
//! operator config wins; after == code override wins), and a private
//! all-`Option` `*Partial` struct so a file/env layer only touches the
//! fields it actually specifies. `TelemetryConfig`'s heavier
//! `BTreeSet<String>`-tracked `TelemetryLayerBuilder` was NOT the model here
//! — that machinery exists to partial-merge `Option`/`Vec`-shaped fields
//! (exporters list, elevation policy); this config has none of that, so the
//! per-field-bool-flag shape `ListenTuningConfig` already uses is the right
//! fit (P1: mirror the closest structural sibling, not the biggest one).

use alloc::vec::Vec;
use core::time::Duration;
use std::path::Path;

use bon::Builder;
use conflaguration::{Settings, Validate, ValidationMessage};
use serde::{Deserialize, Serialize};

// Illustrative defaults picked so a transient blip (seconds) never sheds
// anything, and a multi-hour outage sheds low-value signal long before it
// touches error records. An operator tunes these per how long a maintenance
// window they need to survive at each severity.
const DEFAULT_TRACE_HORIZON_SECS: u64 = 10 * 60;
const DEFAULT_DEBUG_HORIZON_SECS: u64 = 20 * 60;
const DEFAULT_INFO_HORIZON_SECS: u64 = 30 * 60;
const DEFAULT_WARN_HORIZON_SECS: u64 = 35 * 60;
const DEFAULT_ERROR_HORIZON_SECS: u64 = 40 * 60;

// A collector restarting must be noticed promptly: 30s is low enough that an
// "always up" service recovers within about one interval of the collector
// coming back, but high enough not to hammer a still-down collector.
const DEFAULT_BACKOFF_CAP_MS: u64 = 30_000;
const DEFAULT_BACKOFF_BASE_MS: u64 = 200;

// Leaves headroom under the OTel Collector's `otlpreceiver` default
// `max_recv_msg_size_mib: 4` (4 MiB) so a re-batched flush chunk clears the
// collector's own limit with margin for framing overhead.
const DEFAULT_MAX_BATCH_BYTES: usize = 3 * 1024 * 1024;

const DEFAULT_BUFFER_CAPACITY: usize = 65_536;
const DEFAULT_DROP_ANNOUNCE_INTERVAL_MS: u64 = 5_000;
const DEFAULT_IDLE_POLL_MS: u64 = 1_000;

/// Per-severity max retained age. A record older than its own bucket's
/// horizon is evicted regardless of remaining space — the table an operator
/// reads to know exactly what survives an outage of a given length.
///
/// The ladder MUST be non-decreasing (`trace <= debug <= info <= warn <=
/// error`) and every horizon MUST be non-zero — see [`Validate`]. An
/// inverted or zeroed table silently defeats the entire graceful-degradation
/// design (a shorter error horizon than debug means debug outlives errors),
/// so this is checked, not just documented.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Builder, Deserialize, Serialize, Settings)]
#[settings(prefix = "PROXIMA_OTLP_RESILIENT_HORIZONS")]
#[builder(derive(Clone, Debug))]
pub struct RetentionHorizons {
    #[setting(default = 600)]
    #[serde(default = "default_trace_horizon_secs")]
    #[builder(default = default_trace_horizon_secs())]
    pub trace_secs: u64,
    #[setting(default = 1200)]
    #[serde(default = "default_debug_horizon_secs")]
    #[builder(default = default_debug_horizon_secs())]
    pub debug_secs: u64,
    #[setting(default = 1800)]
    #[serde(default = "default_info_horizon_secs")]
    #[builder(default = default_info_horizon_secs())]
    pub info_secs: u64,
    #[setting(default = 2100)]
    #[serde(default = "default_warn_horizon_secs")]
    #[builder(default = default_warn_horizon_secs())]
    pub warn_secs: u64,
    #[setting(default = 2400)]
    #[serde(default = "default_error_horizon_secs")]
    #[builder(default = default_error_horizon_secs())]
    pub error_secs: u64,
}

impl Default for RetentionHorizons {
    fn default() -> Self {
        Self {
            trace_secs: default_trace_horizon_secs(),
            debug_secs: default_debug_horizon_secs(),
            info_secs: default_info_horizon_secs(),
            warn_secs: default_warn_horizon_secs(),
            error_secs: default_error_horizon_secs(),
        }
    }
}

impl RetentionHorizons {
    /// Horizon for bucket index `0..=4` (trace, debug, info, warn, error).
    /// Out-of-range indices saturate to the error horizon (fail toward
    /// keeping data, not discarding it).
    #[must_use]
    pub fn for_bucket(&self, bucket: usize) -> Duration {
        let secs = match bucket {
            0 => self.trace_secs,
            1 => self.debug_secs,
            2 => self.info_secs,
            3 => self.warn_secs,
            _ => self.error_secs,
        };
        Duration::from_secs(secs)
    }
}

impl Validate for RetentionHorizons {
    fn validate(&self) -> conflaguration::Result<()> {
        let mut errors = Vec::new();
        let ladder: [(&str, u64); 5] = [
            ("trace_secs", self.trace_secs),
            ("debug_secs", self.debug_secs),
            ("info_secs", self.info_secs),
            ("warn_secs", self.warn_secs),
            ("error_secs", self.error_secs),
        ];
        for (name, value) in ladder {
            if value == 0 {
                errors.push(ValidationMessage::new(
                    name,
                    "must be >= 1 (0 evicts this severity on the very first sweep)",
                ));
            }
        }
        for pair in ladder.windows(2) {
            let (lower_name, lower_value) = pair[0];
            let (upper_name, upper_value) = pair[1];
            if lower_value > upper_value {
                errors.push(ValidationMessage::new(
                    upper_name,
                    alloc::format!(
                        "must be >= {lower_name} ({lower_value}s) — the ladder must be \
                         non-decreasing by severity, or a less-severe record would \
                         outlive a more-severe one"
                    ),
                ));
            }
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(conflaguration::Error::Validation { errors })
        }
    }
}

fn default_trace_horizon_secs() -> u64 {
    DEFAULT_TRACE_HORIZON_SECS
}
fn default_debug_horizon_secs() -> u64 {
    DEFAULT_DEBUG_HORIZON_SECS
}
fn default_info_horizon_secs() -> u64 {
    DEFAULT_INFO_HORIZON_SECS
}
fn default_warn_horizon_secs() -> u64 {
    DEFAULT_WARN_HORIZON_SECS
}
fn default_error_horizon_secs() -> u64 {
    DEFAULT_ERROR_HORIZON_SECS
}

/// Config for [`super::ResilientSink`] — buffer sizing, backoff bounds, and
/// the retention-horizon table. Fluent (`Builder`) and config-value
/// (`Settings`/`serde`) surfaces stay in parity, same as the rest of the
/// crate's config structs (see `RetryConfig`).
#[derive(Debug, Clone, PartialEq, Builder, Deserialize, Serialize, Settings)]
#[settings(prefix = "PROXIMA_OTLP_RESILIENT")]
#[builder(derive(Clone, Debug))]
pub struct ResilientOtlpConfig {
    /// Total items retained across all severities before space pressure
    /// forces an eviction (shortest-horizon-severity-first).
    #[setting(default = 65536)]
    #[serde(default = "default_buffer_capacity")]
    #[builder(default = default_buffer_capacity())]
    pub buffer_capacity: usize,

    /// Ceiling on one outgoing OTLP batch's encoded proto size. A backlog
    /// larger than this is chunked into multiple batches on flush.
    #[setting(default = 3145728)]
    #[serde(default = "default_max_batch_bytes")]
    #[builder(default = default_max_batch_bytes())]
    pub max_batch_bytes: usize,

    /// Items per outgoing batch, independent of the byte ceiling — whichever
    /// limit is hit first ends the batch.
    #[setting(default = 512)]
    #[serde(default = "default_max_batch_items")]
    #[builder(default = default_max_batch_items())]
    pub max_batch_items: usize,

    /// First retry delay before jitter.
    #[setting(default = 200)]
    #[serde(default = "default_backoff_base_ms")]
    #[builder(default = default_backoff_base_ms())]
    pub backoff_base_ms: u64,

    /// Backoff ceiling — retries never wait longer than this, so a recovered
    /// collector is noticed within about one interval.
    #[setting(default = 30000)]
    #[serde(default = "default_backoff_cap_ms")]
    #[builder(default = default_backoff_cap_ms())]
    pub backoff_cap_ms: u64,

    /// How often accumulated drops are announced to the floor sink as one
    /// aggregated summary line (never one line per drop).
    #[setting(default = 5000)]
    #[serde(default = "default_drop_announce_interval_ms")]
    #[builder(default = default_drop_announce_interval_ms())]
    pub drop_announce_interval_ms: u64,

    /// Safety-net poll interval the background worker falls back to when no
    /// notify wakes it (bounds staleness of the horizon sweep / metrics tick).
    #[setting(default = 1000)]
    #[serde(default = "default_idle_poll_ms")]
    #[builder(default = default_idle_poll_ms())]
    pub idle_poll_ms: u64,

    /// Per-severity max retained age. `nested` composes this field's env
    /// keys as `PROXIMA_OTLP_RESILIENT_HORIZONS_*` (parent prefix + field
    /// name + the child's own field names) — see [`RetentionHorizons`].
    #[setting(nested)]
    #[serde(default)]
    #[builder(default)]
    pub horizons: RetentionHorizons,
}

fn default_buffer_capacity() -> usize {
    DEFAULT_BUFFER_CAPACITY
}
fn default_max_batch_bytes() -> usize {
    DEFAULT_MAX_BATCH_BYTES
}
fn default_max_batch_items() -> usize {
    512
}
fn default_backoff_base_ms() -> u64 {
    DEFAULT_BACKOFF_BASE_MS
}
fn default_backoff_cap_ms() -> u64 {
    DEFAULT_BACKOFF_CAP_MS
}
fn default_drop_announce_interval_ms() -> u64 {
    DEFAULT_DROP_ANNOUNCE_INTERVAL_MS
}
fn default_idle_poll_ms() -> u64 {
    DEFAULT_IDLE_POLL_MS
}

impl Default for ResilientOtlpConfig {
    fn default() -> Self {
        Self {
            buffer_capacity: default_buffer_capacity(),
            max_batch_bytes: default_max_batch_bytes(),
            max_batch_items: default_max_batch_items(),
            backoff_base_ms: default_backoff_base_ms(),
            backoff_cap_ms: default_backoff_cap_ms(),
            drop_announce_interval_ms: default_drop_announce_interval_ms(),
            idle_poll_ms: default_idle_poll_ms(),
            horizons: RetentionHorizons::default(),
        }
    }
}

impl Validate for ResilientOtlpConfig {
    fn validate(&self) -> conflaguration::Result<()> {
        let mut errors = Vec::new();
        if self.buffer_capacity == 0 {
            errors.push(ValidationMessage::new("buffer_capacity", "must be >= 1"));
        }
        if self.max_batch_bytes == 0 {
            errors.push(ValidationMessage::new("max_batch_bytes", "must be >= 1"));
        }
        if self.max_batch_items == 0 {
            errors.push(ValidationMessage::new("max_batch_items", "must be >= 1"));
        }
        if self.backoff_cap_ms < self.backoff_base_ms {
            errors.push(ValidationMessage::new(
                "backoff_cap_ms",
                "must be >= backoff_base_ms",
            ));
        }
        // cascade into the nested strategy table — the whole point of the
        // design is the graceful-degradation ladder, so a caller must see
        // it get validated too, not just the flat scalar knobs above.
        if let Err(error) = self.horizons.validate() {
            collect_nested_errors(&mut errors, "horizons", error);
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(conflaguration::Error::Validation { errors })
        }
    }
}

/// Fold a nested `Validate::validate()` failure into the parent's error set,
/// prefixing each message's path with `section` — the same composition
/// `ProximaSettings::validate` uses for its own `nested` sub-configs
/// (`src/settings/mod.rs`), so a caller gets every problem across the whole
/// tree in one `Err`, not one nested error per validate() call.
fn collect_nested_errors(
    target: &mut Vec<ValidationMessage>,
    section: &str,
    error: conflaguration::Error,
) {
    if let conflaguration::Error::Validation { errors } = error {
        for mut message in errors {
            message.prepend_path(section);
            target.push(message);
        }
    }
}

impl ResilientOtlpConfig {
    #[must_use]
    pub fn backoff_base(&self) -> Duration {
        Duration::from_millis(self.backoff_base_ms)
    }

    #[must_use]
    pub fn backoff_cap(&self) -> Duration {
        Duration::from_millis(self.backoff_cap_ms)
    }

    #[must_use]
    pub fn drop_announce_interval(&self) -> Duration {
        Duration::from_millis(self.drop_announce_interval_ms)
    }

    #[must_use]
    pub fn idle_poll(&self) -> Duration {
        Duration::from_millis(self.idle_poll_ms)
    }

    /// Start a layered builder from the defaults above.
    #[must_use]
    pub fn layered() -> ResilientOtlpConfigLayerBuilder {
        ResilientOtlpConfigLayerBuilder {
            inner: ResilientOtlpConfig::default(),
            buffer_capacity_set: false,
            max_batch_bytes_set: false,
            max_batch_items_set: false,
            backoff_base_ms_set: false,
            backoff_cap_ms_set: false,
            drop_announce_interval_ms_set: false,
            idle_poll_ms_set: false,
            horizons_set: false,
        }
    }
}

/// Partial view of [`ResilientOtlpConfig`] used by `.from_path`/`.underlay_path`
/// — only fields actually present in the file are applied, so a file setting
/// one field never clobbers the others with re-resolved defaults.
///
/// `horizons` is whole-or-nothing here (a file that sets it must set the
/// entire table) — the same granularity `TelemetryConfig` gives its own
/// nested `elevation: Option<Elevation>` policy; sub-field partial-merge
/// inside the ladder isn't warranted for a 5-field table nobody's asked to
/// override piecemeal.
#[derive(Debug, Default, Deserialize)]
struct ResilientOtlpConfigPartial {
    buffer_capacity: Option<usize>,
    max_batch_bytes: Option<usize>,
    max_batch_items: Option<usize>,
    backoff_base_ms: Option<u64>,
    backoff_cap_ms: Option<u64>,
    drop_announce_interval_ms: Option<u64>,
    idle_poll_ms: Option<u64>,
    horizons: Option<RetentionHorizons>,
}

/// Fluent builder for [`ResilientOtlpConfig`]. Every source (`.from_path`,
/// `.from_env`, `.underlay_path`, `.underlay_env`, `.with_*`) contributes
/// only the fields it actually specifies, merged onto the accumulated
/// config — a field a source doesn't touch falls through to whatever prior
/// layers set. `.from_path`/`.from_env` override (last writer wins per
/// field); `.underlay_path`/`.underlay_env` fill only fields still unset;
/// `.with_*` always acts as an override at its call position.
pub struct ResilientOtlpConfigLayerBuilder {
    inner: ResilientOtlpConfig,
    buffer_capacity_set: bool,
    max_batch_bytes_set: bool,
    max_batch_items_set: bool,
    backoff_base_ms_set: bool,
    backoff_cap_ms_set: bool,
    drop_announce_interval_ms_set: bool,
    idle_poll_ms_set: bool,
    horizons_set: bool,
}

impl ResilientOtlpConfigLayerBuilder {
    /// Merge a TOML/JSON file's fields onto the accumulated config; the file
    /// wins for every field it specifies.
    // matches the established `from_path`/`from_env` naming on every other
    // layered-config builder in the workspace (`ListenTuningLayerBuilder`,
    // `TelemetryLayerBuilder`) — clippy's `from_*` convention has no
    // exception for a consuming-self builder chain.
    #[allow(clippy::wrong_self_convention)]
    pub fn from_path<P: AsRef<Path>>(mut self, path: P) -> Result<Self, conflaguration::Error> {
        let partial: ResilientOtlpConfigPartial = conflaguration::from_file(path.as_ref())?;
        if let Some(buffer_capacity) = partial.buffer_capacity {
            self.inner.buffer_capacity = buffer_capacity;
            self.buffer_capacity_set = true;
        }
        if let Some(max_batch_bytes) = partial.max_batch_bytes {
            self.inner.max_batch_bytes = max_batch_bytes;
            self.max_batch_bytes_set = true;
        }
        if let Some(max_batch_items) = partial.max_batch_items {
            self.inner.max_batch_items = max_batch_items;
            self.max_batch_items_set = true;
        }
        if let Some(backoff_base_ms) = partial.backoff_base_ms {
            self.inner.backoff_base_ms = backoff_base_ms;
            self.backoff_base_ms_set = true;
        }
        if let Some(backoff_cap_ms) = partial.backoff_cap_ms {
            self.inner.backoff_cap_ms = backoff_cap_ms;
            self.backoff_cap_ms_set = true;
        }
        if let Some(drop_announce_interval_ms) = partial.drop_announce_interval_ms {
            self.inner.drop_announce_interval_ms = drop_announce_interval_ms;
            self.drop_announce_interval_ms_set = true;
        }
        if let Some(idle_poll_ms) = partial.idle_poll_ms {
            self.inner.idle_poll_ms = idle_poll_ms;
            self.idle_poll_ms_set = true;
        }
        if let Some(horizons) = partial.horizons {
            self.inner.horizons = horizons;
            self.horizons_set = true;
        }
        Ok(self)
    }

    /// Fill any still-unset fields from a TOML/JSON file; already-set fields
    /// are left untouched.
    pub fn underlay_path<P: AsRef<Path>>(mut self, path: P) -> Result<Self, conflaguration::Error> {
        let partial: ResilientOtlpConfigPartial = conflaguration::from_file(path.as_ref())?;
        if !self.buffer_capacity_set
            && let Some(buffer_capacity) = partial.buffer_capacity
        {
            self.inner.buffer_capacity = buffer_capacity;
            self.buffer_capacity_set = true;
        }
        if !self.max_batch_bytes_set
            && let Some(max_batch_bytes) = partial.max_batch_bytes
        {
            self.inner.max_batch_bytes = max_batch_bytes;
            self.max_batch_bytes_set = true;
        }
        if !self.max_batch_items_set
            && let Some(max_batch_items) = partial.max_batch_items
        {
            self.inner.max_batch_items = max_batch_items;
            self.max_batch_items_set = true;
        }
        if !self.backoff_base_ms_set
            && let Some(backoff_base_ms) = partial.backoff_base_ms
        {
            self.inner.backoff_base_ms = backoff_base_ms;
            self.backoff_base_ms_set = true;
        }
        if !self.backoff_cap_ms_set
            && let Some(backoff_cap_ms) = partial.backoff_cap_ms
        {
            self.inner.backoff_cap_ms = backoff_cap_ms;
            self.backoff_cap_ms_set = true;
        }
        if !self.drop_announce_interval_ms_set
            && let Some(drop_announce_interval_ms) = partial.drop_announce_interval_ms
        {
            self.inner.drop_announce_interval_ms = drop_announce_interval_ms;
            self.drop_announce_interval_ms_set = true;
        }
        if !self.idle_poll_ms_set
            && let Some(idle_poll_ms) = partial.idle_poll_ms
        {
            self.inner.idle_poll_ms = idle_poll_ms;
            self.idle_poll_ms_set = true;
        }
        if !self.horizons_set
            && let Some(horizons) = partial.horizons
        {
            self.inner.horizons = horizons;
            self.horizons_set = true;
        }
        Ok(self)
    }

    /// Merge env-set fields onto the accumulated config; env wins for every
    /// field it sets. Unset env vars leave the current value untouched.
    /// `horizons` is considered "set" if ANY of its five constituent env
    /// keys is present (the nested table resolves as a whole via the
    /// `Settings` derive's own `nested` composition).
    #[allow(clippy::wrong_self_convention)]
    pub fn from_env(mut self) -> Result<Self, conflaguration::Error> {
        let resolved = ResilientOtlpConfig::from_env()?;
        if env_is_set("PROXIMA_OTLP_RESILIENT_BUFFER_CAPACITY") {
            self.inner.buffer_capacity = resolved.buffer_capacity;
            self.buffer_capacity_set = true;
        }
        if env_is_set("PROXIMA_OTLP_RESILIENT_MAX_BATCH_BYTES") {
            self.inner.max_batch_bytes = resolved.max_batch_bytes;
            self.max_batch_bytes_set = true;
        }
        if env_is_set("PROXIMA_OTLP_RESILIENT_MAX_BATCH_ITEMS") {
            self.inner.max_batch_items = resolved.max_batch_items;
            self.max_batch_items_set = true;
        }
        if env_is_set("PROXIMA_OTLP_RESILIENT_BACKOFF_BASE_MS") {
            self.inner.backoff_base_ms = resolved.backoff_base_ms;
            self.backoff_base_ms_set = true;
        }
        if env_is_set("PROXIMA_OTLP_RESILIENT_BACKOFF_CAP_MS") {
            self.inner.backoff_cap_ms = resolved.backoff_cap_ms;
            self.backoff_cap_ms_set = true;
        }
        if env_is_set("PROXIMA_OTLP_RESILIENT_DROP_ANNOUNCE_INTERVAL_MS") {
            self.inner.drop_announce_interval_ms = resolved.drop_announce_interval_ms;
            self.drop_announce_interval_ms_set = true;
        }
        if env_is_set("PROXIMA_OTLP_RESILIENT_IDLE_POLL_MS") {
            self.inner.idle_poll_ms = resolved.idle_poll_ms;
            self.idle_poll_ms_set = true;
        }
        if any_horizon_env_set() {
            self.inner.horizons = resolved.horizons;
            self.horizons_set = true;
        }
        Ok(self)
    }

    /// Fill any still-unset fields from env vars; already-set fields are
    /// left untouched even if the matching env var is set.
    pub fn underlay_env(mut self) -> Result<Self, conflaguration::Error> {
        let resolved = ResilientOtlpConfig::from_env()?;
        if !self.buffer_capacity_set && env_is_set("PROXIMA_OTLP_RESILIENT_BUFFER_CAPACITY") {
            self.inner.buffer_capacity = resolved.buffer_capacity;
            self.buffer_capacity_set = true;
        }
        if !self.max_batch_bytes_set && env_is_set("PROXIMA_OTLP_RESILIENT_MAX_BATCH_BYTES") {
            self.inner.max_batch_bytes = resolved.max_batch_bytes;
            self.max_batch_bytes_set = true;
        }
        if !self.max_batch_items_set && env_is_set("PROXIMA_OTLP_RESILIENT_MAX_BATCH_ITEMS") {
            self.inner.max_batch_items = resolved.max_batch_items;
            self.max_batch_items_set = true;
        }
        if !self.backoff_base_ms_set && env_is_set("PROXIMA_OTLP_RESILIENT_BACKOFF_BASE_MS") {
            self.inner.backoff_base_ms = resolved.backoff_base_ms;
            self.backoff_base_ms_set = true;
        }
        if !self.backoff_cap_ms_set && env_is_set("PROXIMA_OTLP_RESILIENT_BACKOFF_CAP_MS") {
            self.inner.backoff_cap_ms = resolved.backoff_cap_ms;
            self.backoff_cap_ms_set = true;
        }
        if !self.drop_announce_interval_ms_set
            && env_is_set("PROXIMA_OTLP_RESILIENT_DROP_ANNOUNCE_INTERVAL_MS")
        {
            self.inner.drop_announce_interval_ms = resolved.drop_announce_interval_ms;
            self.drop_announce_interval_ms_set = true;
        }
        if !self.idle_poll_ms_set && env_is_set("PROXIMA_OTLP_RESILIENT_IDLE_POLL_MS") {
            self.inner.idle_poll_ms = resolved.idle_poll_ms;
            self.idle_poll_ms_set = true;
        }
        if !self.horizons_set && any_horizon_env_set() {
            self.inner.horizons = resolved.horizons;
            self.horizons_set = true;
        }
        Ok(self)
    }

    #[must_use]
    pub fn with_buffer_capacity(mut self, buffer_capacity: usize) -> Self {
        self.inner.buffer_capacity = buffer_capacity;
        self.buffer_capacity_set = true;
        self
    }

    #[must_use]
    pub fn with_max_batch_bytes(mut self, max_batch_bytes: usize) -> Self {
        self.inner.max_batch_bytes = max_batch_bytes;
        self.max_batch_bytes_set = true;
        self
    }

    #[must_use]
    pub fn with_max_batch_items(mut self, max_batch_items: usize) -> Self {
        self.inner.max_batch_items = max_batch_items;
        self.max_batch_items_set = true;
        self
    }

    #[must_use]
    pub fn with_backoff_base_ms(mut self, backoff_base_ms: u64) -> Self {
        self.inner.backoff_base_ms = backoff_base_ms;
        self.backoff_base_ms_set = true;
        self
    }

    #[must_use]
    pub fn with_backoff_cap_ms(mut self, backoff_cap_ms: u64) -> Self {
        self.inner.backoff_cap_ms = backoff_cap_ms;
        self.backoff_cap_ms_set = true;
        self
    }

    #[must_use]
    pub fn with_drop_announce_interval_ms(mut self, drop_announce_interval_ms: u64) -> Self {
        self.inner.drop_announce_interval_ms = drop_announce_interval_ms;
        self.drop_announce_interval_ms_set = true;
        self
    }

    #[must_use]
    pub fn with_idle_poll_ms(mut self, idle_poll_ms: u64) -> Self {
        self.inner.idle_poll_ms = idle_poll_ms;
        self.idle_poll_ms_set = true;
        self
    }

    #[must_use]
    pub fn with_horizons(mut self, horizons: RetentionHorizons) -> Self {
        self.inner.horizons = horizons;
        self.horizons_set = true;
        self
    }

    /// The built config. Does NOT validate — call `.validate()` on the
    /// result explicitly (same contract as `ListenTuningConfig::build` /
    /// `RetryConfig`), so a caller decides whether an invalid layered
    /// config is a hard error or something to report and fall back from.
    #[must_use]
    pub fn build(self) -> ResilientOtlpConfig {
        self.inner
    }
}

fn env_is_set(name: &str) -> bool {
    std::env::var(name).is_ok()
}

fn any_horizon_env_set() -> bool {
    [
        "PROXIMA_OTLP_RESILIENT_HORIZONS_TRACE_SECS",
        "PROXIMA_OTLP_RESILIENT_HORIZONS_DEBUG_SECS",
        "PROXIMA_OTLP_RESILIENT_HORIZONS_INFO_SECS",
        "PROXIMA_OTLP_RESILIENT_HORIZONS_WARN_SECS",
        "PROXIMA_OTLP_RESILIENT_HORIZONS_ERROR_SECS",
    ]
    .into_iter()
    .any(env_is_set)
}
