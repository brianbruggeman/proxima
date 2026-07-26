//! First-class config for the resilient OTLP sink: buffer sizing, backoff
//! bounds, and the per-severity retention horizon table.

use alloc::vec::Vec;
use core::time::Duration;

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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Builder, Deserialize, Serialize)]
#[builder(derive(Clone, Debug))]
pub struct RetentionHorizons {
    #[serde(default = "default_trace_horizon_secs")]
    #[builder(default = default_trace_horizon_secs())]
    pub trace_secs: u64,
    #[serde(default = "default_debug_horizon_secs")]
    #[builder(default = default_debug_horizon_secs())]
    pub debug_secs: u64,
    #[serde(default = "default_info_horizon_secs")]
    #[builder(default = default_info_horizon_secs())]
    pub info_secs: u64,
    #[serde(default = "default_warn_horizon_secs")]
    #[builder(default = default_warn_horizon_secs())]
    pub warn_secs: u64,
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
#[derive(Debug, Clone, Builder, Deserialize, Serialize, Settings)]
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

    /// Per-severity max retained age.
    #[setting(skip)]
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
        if errors.is_empty() {
            Ok(())
        } else {
            Err(conflaguration::Error::Validation { errors })
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
}
