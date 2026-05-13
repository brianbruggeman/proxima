//! `conflaguration`-derived [`ReplayConfig`] + factory entry point.
//!
//! Gated behind the `config` feature so consumers that only need the
//! raw `ReplayUpstream::from_jsonl` / `from_source` constructors stay
//! dependency-light. With the feature on, callers can drive a Replay
//! pipe entirely from env vars or a TOML block:
//!
//! ```text
//! PROXIMA_REPLAY_SOURCE_PATH=/var/lib/proxima/recording.jsonl
//! PROXIMA_REPLAY_LABEL=replay
//! ```
//!
//! plus
//!
//! ```toml
//! [replay]
//! source_path = "/var/lib/proxima/recording.jsonl"
//! label = "replay"
//! format = "jsonl"   # or "bin"
//! ```

use std::path::PathBuf;

use bon::Builder;
use conflaguration::{Settings, Validate, ValidationMessage};
use proxima_core::ProximaError;
use serde::{Deserialize, Serialize};

use crate::replay::ReplayUpstream;

#[derive(Debug, Clone, Builder, Deserialize, Serialize, Settings)]
#[settings(prefix = "PROXIMA_REPLAY")]
#[builder(derive(Clone, Debug), on(String, into), on(PathBuf, into))]
pub struct ReplayConfig {
    /// Path to the recording file the replay reads from. Must exist
    /// and be readable. The format is inferred from `format` (below)
    /// rather than the extension because operators sometimes name the
    /// same payload `.log` or `.recording` regardless of encoding.
    /// example: `/var/lib/proxima/recording.jsonl`
    pub source_path: PathBuf,

    /// Friendly name for the replay Pipe, surfaced through
    /// `Pipe::name()` and logged on every replay hit / miss.
    /// example: `replay`
    #[setting(default_str = "replay")]
    #[serde(default = "default_label")]
    #[builder(default = default_label())]
    pub label: String,

    /// On-disk format of the source file. Today the supported values
    /// are `jsonl` (default — `JsonlSource`) and `bin` (`BinSource`).
    /// Unknown values fail validation.
    /// example: `jsonl`
    #[setting(default_str = "jsonl")]
    #[serde(default = "default_format")]
    #[builder(default = default_format())]
    pub format: String,
}

fn default_label() -> String {
    "replay".to_string()
}

fn default_format() -> String {
    "jsonl".to_string()
}

impl Validate for ReplayConfig {
    fn validate(&self) -> conflaguration::Result<()> {
        let mut errors: Vec<ValidationMessage> = Vec::new();
        if self.source_path.as_os_str().is_empty() {
            errors.push(ValidationMessage::new(
                "source_path",
                "source_path must be non-empty",
            ));
        }
        if !matches!(self.format.as_str(), "jsonl" | "bin") {
            errors.push(ValidationMessage::new(
                "format",
                "format must be 'jsonl' or 'bin'",
            ));
        }
        if self.label.is_empty() {
            errors.push(ValidationMessage::new("label", "label must be non-empty"));
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(conflaguration::Error::Validation { errors })
        }
    }
}

impl ReplayConfig {
    /// Async factory: instantiate a `ReplayUpstream` from a validated
    /// `ReplayConfig`. Today only `jsonl` is wired through this
    /// fast-path because `BinSource` requires a `RecordingSourceRegistry`
    /// to compose with custom format extensions — callers wanting `bin`
    /// pass through `ReplayPipeFactory::new` and provide the registry.
    pub async fn build(
        &self,
        runtime: std::sync::Arc<dyn proxima_runtime::Runtime>,
    ) -> Result<ReplayUpstream, ProximaError> {
        self.validate()
            .map_err(|err| ProximaError::Config(err.to_string()))?;
        match self.format.as_str() {
            "jsonl" => {
                ReplayUpstream::from_jsonl(self.source_path.clone(), self.label.clone(), runtime)
                    .await
            }
            other => Err(ProximaError::Config(format!(
                "ReplayConfig::build: format `{other}` not yet wired through the fast path; use ReplayPipeFactory + RecordingSourceRegistry"
            ))),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn defaults_use_jsonl_label_and_format() {
        let config = ReplayConfig::builder()
            .source_path("/tmp/recording.jsonl")
            .build();
        assert_eq!(config.label, "replay");
        assert_eq!(config.format, "jsonl");
    }

    #[test]
    fn validation_rejects_empty_source_path() {
        let config = ReplayConfig::builder().source_path("").build();
        assert!(config.validate().is_err());
    }

    #[test]
    fn validation_rejects_unknown_format() {
        let config = ReplayConfig::builder()
            .source_path("/tmp/r.bin")
            .format("toml")
            .build();
        assert!(config.validate().is_err());
    }

    #[test]
    fn validation_accepts_jsonl_or_bin_format() {
        let jsonl = ReplayConfig::builder().source_path("/tmp/r.jsonl").build();
        let bin = ReplayConfig::builder()
            .source_path("/tmp/r.bin")
            .format("bin")
            .build();
        assert!(jsonl.validate().is_ok());
        assert!(bin.validate().is_ok());
    }

    #[test]
    fn toml_round_trip_preserves_state() {
        let original = ReplayConfig::builder()
            .source_path("/var/lib/proxima/r.jsonl")
            .label("test-replay")
            .build();
        let toml_text = toml::to_string(&original).expect("serialize");
        let restored: ReplayConfig = toml::from_str(&toml_text).expect("deserialize");
        assert_eq!(restored.source_path, original.source_path);
        assert_eq!(restored.label, original.label);
        assert_eq!(restored.format, original.format);
    }

    #[proxima::test(runtime = "tokio")]
    async fn build_fast_path_rejects_bin_format() {
        let config = ReplayConfig::builder()
            .source_path("/tmp/nonexistent.bin")
            .format("bin")
            .build();
        let runtime: std::sync::Arc<dyn proxima_runtime::Runtime> =
            std::sync::Arc::new(prime::os::runtime::PrimeRuntime::new(1).expect("prime"));
        let outcome = config.build(runtime).await;
        assert!(matches!(outcome, Err(ProximaError::Config(_))));
    }
}
