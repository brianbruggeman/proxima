//! Runtime configuration surface for [`crate::cpu`]'s execution policy
//! (`config`-feature-gated, mirroring `proxima-gguf/src/config.rs`'s "one
//! type = builder result = config" shape). `conflaguration` is std-only
//! (see `~/.claude/rules/rust.md`'s conflag skill and
//! `proxima-telemetry/src/config.rs`, the canonical example), so this
//! module lives behind the `config` feature (which already implies
//! `std`) -- the no_std+alloc floor never sees it and [`crate::cpu`]
//! always uses [`crate::sized`]'s constants directly.
//!
//! The bridge: every default here seeds from `crate::sized`, never
//! re-declaring the value, so a build-time floor and its std-tier runtime
//! default cannot silently drift apart
//! (`defaults_track_the_sized_floor` below pins the invariant).
//!
//! # Not yet wired into `cpu.rs`
//!
//! Unlike `proxima-gguf::config::GgufParserConfig` (consumed by
//! `GgufParser::with_config`) or
//! `proxima-safetensors::config::SafetensorsParserConfig` (consumed by
//! `SafetensorsParser::with_config`), nothing in [`crate::cpu`] reads
//! [`TensorExecutionConfig`] yet. `parallel_threshold`/`oversubscribe`/
//! `split_alignment` are read as bare [`crate::sized`] constants inside
//! several private `cpu.rs` helpers (`evaluate_node_parallel`,
//! `run_chunks_threaded`, `BoundOp::split_aligned`'s caller); threading a
//! config value through those call sites is real surgery on the same file
//! another agent is concurrently restructuring (its tile operands and
//! quantized paths) this session, so it is deliberately not attempted
//! here. This struct is the declared, validated, tested runtime surface a
//! future `cpu::evaluate_parallel_with_config`-style entry point would
//! consume -- landing it now (rather than leaving the policy values as
//! unreachable bare consts) is the `sized`/`config` pair the rest of the
//! workspace's format crates already have; the wiring is the follow-up.

use bon::Builder;
use conflaguration::{Settings, Validate, ValidationMessage};
use serde::{Deserialize, Serialize};

/// Runtime policy for [`crate::cpu`]'s parallel-execution heuristics.
/// Every field defaults from [`crate::sized`]'s build-time floor and may
/// be overridden per process (env `TENSOR_EXECUTION_*`, a config file, or
/// the fluent builder) once a `cpu.rs` entry point consumes it -- see this
/// module's doc for why that wiring is not landed yet.
#[derive(Debug, Clone, PartialEq, Eq, Builder, Deserialize, Serialize, Settings)]
#[settings(prefix = "TENSOR_EXECUTION")]
#[builder(derive(Clone, Debug))]
pub struct TensorExecutionConfig {
    /// Below this many iteration-space elements, a nest runs the plain
    /// sequential path even when `workers > 1`. Build-time default:
    /// `crate::sized::PARALLEL_THRESHOLD`.
    #[setting(default = 4096)]
    #[serde(default = "default_parallel_threshold")]
    #[builder(default = default_parallel_threshold())]
    pub parallel_threshold: u64,

    /// Chunk-count multiplier over `workers`. Build-time default:
    /// `crate::sized::OVERSUBSCRIBE`.
    #[setting(default = 1)]
    #[serde(default = "default_oversubscribe")]
    #[builder(default = default_oversubscribe())]
    pub oversubscribe: u64,

    /// Row-alignment applied to every non-final chunk boundary.
    /// Build-time default: `crate::sized::SPLIT_ALIGNMENT`.
    #[setting(default = 1)]
    #[serde(default = "default_split_alignment")]
    #[builder(default = default_split_alignment())]
    pub split_alignment: u64,
}

// `config` (this module's own gate) implies `std`, so `crate::sized`'s
// std-gated constants are always present here -- no fallback branch
// needed, and none should exist: a second literal here is exactly the
// duplicated-default drift this module's whole job is to prevent.
fn default_parallel_threshold() -> u64 {
    crate::sized::PARALLEL_THRESHOLD as u64
}

fn default_oversubscribe() -> u64 {
    crate::sized::OVERSUBSCRIBE as u64
}

fn default_split_alignment() -> u64 {
    crate::sized::SPLIT_ALIGNMENT
}

impl Default for TensorExecutionConfig {
    fn default() -> Self {
        Self::builder().build()
    }
}

impl Validate for TensorExecutionConfig {
    fn validate(&self) -> conflaguration::Result<()> {
        let mut errors = Vec::new();
        if self.parallel_threshold == 0 {
            errors.push(ValidationMessage::new(
                "parallel_threshold",
                "must be > 0",
            ));
        }
        if self.oversubscribe == 0 {
            errors.push(ValidationMessage::new("oversubscribe", "must be > 0"));
        }
        if self.split_alignment == 0 {
            errors.push(ValidationMessage::new("split_alignment", "must be > 0"));
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(conflaguration::Error::Validation { errors })
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn default_config_validates() {
        let config = TensorExecutionConfig::default();
        assert!(config.validate().is_ok(), "default config should validate");
    }

    // the whole bridge: the runtime default must be SEEDED from the
    // build-time sized constant, never duplicated, so the two can never
    // silently drift apart.
    #[test]
    fn defaults_track_the_sized_floor() {
        let config = TensorExecutionConfig::default();
        assert_eq!(
            config.parallel_threshold,
            crate::sized::PARALLEL_THRESHOLD as u64
        );
        assert_eq!(config.oversubscribe, crate::sized::OVERSUBSCRIBE as u64);
        assert_eq!(config.split_alignment, crate::sized::SPLIT_ALIGNMENT);

        // the env-overlay path (from_env, no vars set) must agree too --
        // guards against the #[setting] literal drifting from the const.
        temp_env::with_vars::<&str, &str, _, _>([], || {
            let from_env = TensorExecutionConfig::from_env().expect("from_env");
            assert_eq!(
                from_env.parallel_threshold,
                crate::sized::PARALLEL_THRESHOLD as u64,
                "#[setting] parallel_threshold literal drifted from sized::PARALLEL_THRESHOLD"
            );
            assert_eq!(
                from_env.oversubscribe,
                crate::sized::OVERSUBSCRIBE as u64,
                "#[setting] oversubscribe literal drifted from sized::OVERSUBSCRIBE"
            );
            assert_eq!(
                from_env.split_alignment,
                crate::sized::SPLIT_ALIGNMENT,
                "#[setting] split_alignment literal drifted from sized::SPLIT_ALIGNMENT"
            );
        });
    }

    #[test]
    fn env_override_takes_effect_and_does_not_leak() {
        temp_env::with_vars(
            [
                ("TENSOR_EXECUTION_PARALLEL_THRESHOLD", Some("8192")),
                ("TENSOR_EXECUTION_OVERSUBSCRIBE", Some("2")),
                ("TENSOR_EXECUTION_SPLIT_ALIGNMENT", Some("6")),
            ],
            || {
                let config = TensorExecutionConfig::from_env().expect("from_env");
                assert_eq!(config.parallel_threshold, 8192);
                assert_eq!(config.oversubscribe, 2);
                assert_eq!(config.split_alignment, 6);
            },
        );

        // outside the scoped block, the override must not leak into a
        // fresh read.
        let config = TensorExecutionConfig::from_env().expect("from_env");
        assert_eq!(
            config.parallel_threshold,
            crate::sized::PARALLEL_THRESHOLD as u64,
            "env override leaked past its scope"
        );
    }

    #[test]
    fn zero_fields_rejected() {
        let config = TensorExecutionConfig::builder().parallel_threshold(0).build();
        let err = config
            .validate()
            .expect_err("validate must reject parallel_threshold = 0");
        assert!(format!("{err:?}").contains("parallel_threshold"));
    }
}
