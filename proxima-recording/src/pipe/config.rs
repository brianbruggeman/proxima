//! `conflaguration`-derived [`RecordingConfig`] + factory entry point.
//!
//! Gated behind the `config` feature so the raw `BoundedRecordingSink::new` /
//! `LiveCaptureContext` paths stay dependency-light. With the feature on
//! callers can drive recording-pipe assembly entirely from env vars or a TOML
//! block — matching the [`crate::pipe::cap::FailMode`] taxonomy.
//!
//! ```text
//! PROXIMA_RECORDING_ENABLED=true
//! PROXIMA_RECORDING_SINK_CAPACITY=4096
//! PROXIMA_RECORDING_FAIL_MODE=drop_oldest
//! ```

use std::sync::Arc;

use bon::Builder;
use conflaguration::{Settings, Validate, ValidationMessage};
use proxima_core::ProximaError;
use proxima_runtime::Runtime;
use serde::{Deserialize, Serialize};

use crate::pipe::cap::FailMode;
use crate::pipe::dest::{FormatKind, SinkSpec};
use crate::pipe::fanout::FanOut;
use crate::pipe::log_pipe::AppendLog;

/// One durable sink in a fan-out: a destination + its format. The fluent /
/// serde config surface that lowers to a std-level [`SinkSpec`].
#[derive(Debug, Clone, PartialEq, Eq, Builder, Serialize, Deserialize)]
#[builder(on(String, into))]
pub struct SinkConfig {
    /// Filesystem path the durable log appends to.
    /// example: `"/var/log/audit.bin"`
    pub path: String,
    /// Which format codec this sink uses.
    /// example: `"bin"`
    #[serde(default)]
    #[builder(default)]
    pub format: FormatKind,
}

impl SinkConfig {
    /// Lower to the builder-free descriptor the durable terminals open.
    #[must_use]
    pub fn to_spec(&self) -> SinkSpec {
        SinkSpec::new(self.path.clone(), self.format)
    }
}

/// The recording topology: a fan-out to N durable sinks, each with its own
/// format. Composed interchangeably via config (TOML/JSON `sinks = [...]`,
/// zero new Rust to add a sink) OR the fluent builder, proven identical by the
/// round-trip parity test. Mirrors how proxima composes pipes (file-driven
/// composition + fluent surface; principle 4 config-as-composition).
#[derive(Debug, Clone, PartialEq, Eq, Default, Builder, Serialize, Deserialize)]
pub struct RecorderConfig {
    /// The durable sinks the recorder fans every event out to.
    #[serde(default)]
    #[builder(default)]
    pub sinks: Vec<SinkConfig>,
}

impl RecorderConfig {
    /// Build the fan-out Pipe tree: one [`AppendLog`] terminal per sink (format
    /// → codec), broadcast by a [`FanOut`]. The runtime drives every sink's
    /// blocking I/O off-core.
    pub fn build(&self, runtime: Arc<dyn Runtime>) -> Result<FanOut, ProximaError> {
        let mut sinks = Vec::with_capacity(self.sinks.len());
        for sink in &self.sinks {
            let log = AppendLog::open(&sink.path, sink.format.codec()?, Arc::clone(&runtime))?;
            sinks.push(log);
        }
        Ok(FanOut::new(sinks))
    }

    /// Lower the topology to the builder-free descriptors a spigot terminal
    /// ([`crate::pipe::lazy::LazyFanOut`]) opens once armed.
    #[must_use]
    pub fn specs(&self) -> Vec<SinkSpec> {
        self.sinks.iter().map(SinkConfig::to_spec).collect()
    }
}

#[derive(Debug, Clone, Builder, Deserialize, Serialize, Settings)]
#[settings(prefix = "PROXIMA_RECORDING")]
#[builder(derive(Clone, Debug), on(String, into))]
pub struct RecordingConfig {
    /// Master switch. When false, the factory returns a no-op
    /// sink-equivalent so the surrounding Pipe graph still composes
    /// without writing to disk. Default off so adding the config
    /// surface to a binary doesn't change behavior until opt-in.
    /// example: `false`
    #[setting(default = false)]
    #[serde(default)]
    #[builder(default)]
    pub enabled: bool,

    /// `BoundedRecordingSink` queue capacity in events (NOT bytes).
    /// Matches the existing constructor's `capacity` argument: items
    /// queued for the worker to drain to the backend. Sized for a
    /// typical chunked streaming turn (~40 events) with margin.
    /// example: `4096`
    #[setting(default = 4096)]
    #[serde(default = "default_sink_capacity")]
    #[builder(default = default_sink_capacity())]
    pub sink_capacity: usize,

    /// What the sink does when the queue is full. Wire shape mirrors
    /// the [`FailMode`] enum; the typed accessor [`Self::fail_mode`]
    /// parses the string.
    /// example: `"drop_oldest"`
    #[setting(default_str = "drop_oldest")]
    #[serde(default = "default_fail_mode")]
    #[builder(default = default_fail_mode())]
    pub fail_mode: String,
}

fn default_sink_capacity() -> usize {
    4096
}

fn default_fail_mode() -> String {
    "drop_oldest".to_string()
}

impl RecordingConfig {
    /// Typed view of the configured [`FailMode`], or `None` if the
    /// wire string is not a known shape. `validate` rejects unknown
    /// shapes so a successful build guarantees this returns `Some`.
    #[must_use]
    pub fn fail_mode(&self) -> Option<FailMode> {
        match self.fail_mode.as_str() {
            "drop_oldest" => Some(FailMode::DropOldest),
            "drop_newest" => Some(FailMode::DropNewest),
            "fail_closed" => Some(FailMode::FailClosed),
            _ => None,
        }
    }
}

impl Validate for RecordingConfig {
    fn validate(&self) -> conflaguration::Result<()> {
        let mut errors: Vec<ValidationMessage> = Vec::new();
        if self.sink_capacity == 0 {
            errors.push(ValidationMessage::new(
                "sink_capacity",
                "sink_capacity must be > 0",
            ));
        }
        if self.fail_mode().is_none() {
            errors.push(ValidationMessage::new(
                "fail_mode",
                "fail_mode must be one of: drop_oldest, drop_newest, fail_closed",
            ));
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(conflaguration::Error::Validation { errors })
        }
    }
}

impl RecordingConfig {
    /// Run `Validate` and convert any error into a `ProximaError::Config`
    /// so callers building a graph from a single error chain can `?` it.
    pub fn validated(&self) -> Result<&Self, ProximaError> {
        self.validate()
            .map_err(|err| ProximaError::Config(err.to_string()))?;
        Ok(self)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_disabled_drop_oldest_4096() {
        let config = RecordingConfig::builder().build();
        assert!(!config.enabled);
        assert_eq!(config.sink_capacity, 4096);
        assert_eq!(config.fail_mode(), Some(FailMode::DropOldest));
    }

    #[test]
    fn validation_rejects_zero_capacity() {
        let config = RecordingConfig::builder().sink_capacity(0).build();
        assert!(config.validate().is_err());
    }

    #[test]
    fn validation_rejects_unknown_fail_mode() {
        let config = RecordingConfig::builder().fail_mode("explode").build();
        assert!(config.validate().is_err());
    }

    #[test]
    fn validation_accepts_each_known_fail_mode() {
        for mode in ["drop_oldest", "drop_newest", "fail_closed"] {
            let config = RecordingConfig::builder().fail_mode(mode).build();
            assert!(config.validate().is_ok(), "mode `{mode}` must validate");
        }
    }

    #[test]
    fn validated_returns_self_on_ok() {
        let config = RecordingConfig::builder().build();
        let view = config.validated().expect("default validates");
        assert_eq!(view.sink_capacity, config.sink_capacity);
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod recorder_config_tests {
    use super::*;

    // a two-sink fan-out topology built BY CONFIG (the serde shape a TOML
    // `[[sinks]]` block produces) must equal the SAME topology built fluently.
    // This is the config-first ≡ fluent-first guarantee (principle 4).
    #[test]
    fn config_and_builder_reach_identical_topology() {
        let from_config: RecorderConfig = serde_json::from_value(serde_json::json!({
            "sinks": [
                { "path": "/var/log/audit.bin", "format": "bin" },
                { "path": "/var/log/audit.jsonl", "format": "json" },
            ]
        }))
        .unwrap();

        let from_builder = RecorderConfig::builder()
            .sinks(vec![
                SinkConfig::builder()
                    .path("/var/log/audit.bin")
                    .format(FormatKind::Bin)
                    .build(),
                SinkConfig::builder()
                    .path("/var/log/audit.jsonl")
                    .format(FormatKind::Json)
                    .build(),
            ])
            .build();

        assert_eq!(
            from_config, from_builder,
            "config-built ≡ fluent-built topology"
        );
    }

    // adding a sink is config, zero new Rust — and the serde shape round-trips.
    #[test]
    fn topology_round_trips_through_serde() {
        let topology = RecorderConfig::builder()
            .sinks(vec![
                SinkConfig::builder().path("/a.bin").build(), // format defaults to bin
                SinkConfig::builder()
                    .path("/b.jsonl")
                    .format(FormatKind::Json)
                    .build(),
            ])
            .build();
        let json = serde_json::to_string(&topology).unwrap();
        let parsed: RecorderConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, topology, "serde round-trip preserves the topology");
        assert_eq!(
            parsed.sinks[0].format,
            FormatKind::Bin,
            "omitted format defaults to bin"
        );
    }

    #[test]
    fn default_is_empty_and_equals_builder() {
        assert_eq!(RecorderConfig::default(), RecorderConfig::builder().build());
        assert!(RecorderConfig::default().sinks.is_empty());
    }

    // the config actually BUILDS the fan-out: a 2-sink topology yields a FanOut
    // over 2 durable terminals, and every event lands in both.
    #[test]
    fn config_builds_a_working_two_sink_fanout() {
        use crate::event::{
            FrameMetadata, HttpEvent, InteractionId, ProtocolEvent, RecordingEvent,
        };
        use crate::pipe::log_pipe::ReplayLog;
        use bytes::Bytes;
        use proxima_primitives::pipe::SendPipe;

        let dir = tempfile::tempdir().unwrap();
        let bin_path = dir.path().join("audit.bin");
        let json_path = dir.path().join("audit.jsonl");
        let topology: RecorderConfig = serde_json::from_value(serde_json::json!({
            "sinks": [
                { "path": bin_path.to_str().unwrap(), "format": "bin" },
                { "path": json_path.to_str().unwrap(), "format": "json" },
            ]
        }))
        .unwrap();

        let runtime: Arc<dyn Runtime> =
            Arc::new(prime::os::runtime::PrimeRuntime::new(1).expect("prime"));
        let recorder = topology.build(Arc::clone(&runtime)).unwrap();
        assert_eq!(recorder.sink_count(), 2);

        let id = InteractionId::new();
        let events = vec![RecordingEvent {
            id,
            ts_ms: 7,
            parent: None,
            event: ProtocolEvent::Http(HttpEvent::ResponseChunk {
                data: Bytes::from_static(b"data: hello\n\n"),
                metadata: FrameMetadata::new(),
            }),
        }];

        let drain = |path: &std::path::Path, fmt: FormatKind| {
            let reader = ReplayLog::open(path, fmt.codec().unwrap(), Arc::clone(&runtime)).unwrap();
            futures::executor::block_on(async {
                let mut got = Vec::new();
                let mut offset = 0_u64;
                loop {
                    let chunk = reader.call(offset).await.unwrap();
                    if chunk.done {
                        break;
                    }
                    got.extend(chunk.events);
                    offset = chunk.next_offset;
                }
                got
            })
        };

        futures::executor::block_on(async {
            recorder.call(events.clone()).await.unwrap();
            recorder.flush().await.unwrap();
        });

        assert_eq!(
            drain(&bin_path, FormatKind::Bin),
            events,
            "bin sink got the event"
        );
        assert_eq!(
            drain(&json_path, FormatKind::Json),
            events,
            "json sink got the event"
        );
    }
}
