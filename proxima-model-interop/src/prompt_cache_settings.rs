//! `PromptCacheSettings`: the conflaguration-facing (env/TOML) loader and
//! `bon` fluent builder for [`crate::PromptCacheConfig`]'s data, the same
//! split and pattern as [`crate::SpeculativeSettings`] (see that module's doc
//! for why the `Copy`, no_std-safe config lives in `serving.rs` and the
//! owned builder/loader shape lives here, `std`-gated).

use bon::Builder;
use conflaguration::{Settings, Validate};
use serde::{Deserialize, Serialize};

use crate::serving::PromptCacheConfig;

/// The conflaguration/`bon` mirror of [`crate::PromptCacheConfig`]; the
/// builder, a TOML file and `PROXIMA_PROMPT_CACHE_*` env vars all produce the
/// same value, and [`Self::as_prompt_cache_config`] lowers it into the
/// `Copy` shape [`crate::ServingConfig::prompt_cache`] reads at decode time.
#[derive(Debug, Clone, PartialEq, Eq, Builder, Deserialize, Serialize, Settings, Validate)]
#[settings(prefix = "PROXIMA_PROMPT_CACHE")]
#[builder(derive(Clone, Debug))]
pub struct PromptCacheSettings {
    /// See [`crate::PromptCacheConfig::byte_budget`].
    #[setting(default = 0)]
    #[builder(default = 0)]
    pub byte_budget: u64,
    /// See [`crate::PromptCacheConfig::max_entries`].
    #[setting(default = 4)]
    #[builder(default = 4)]
    pub max_entries: u32,
    /// See [`crate::PromptCacheConfig::ring_rewind_slack`].
    #[setting(default = 256)]
    #[builder(default = 256)]
    pub ring_rewind_slack: u32,
    /// See [`crate::PromptCacheConfig::checkpoint_interval`].
    #[setting(default = 0)]
    #[builder(default = 0)]
    pub checkpoint_interval: u32,
    /// See [`crate::PromptCacheConfig::max_checkpoints`].
    #[setting(default = 0)]
    #[builder(default = 0)]
    pub max_checkpoints: u32,
    /// See [`crate::PromptCacheConfig::cache_reuse_min`].
    #[setting(default = 0)]
    #[builder(default = 0)]
    pub cache_reuse_min: u32,
}

impl PromptCacheSettings {
    /// The `Copy` shape `ServingConfig::prompt_cache` reads.
    #[must_use]
    pub const fn as_prompt_cache_config(&self) -> PromptCacheConfig {
        PromptCacheConfig {
            byte_budget: self.byte_budget,
            max_entries: self.max_entries,
            ring_rewind_slack: self.ring_rewind_slack,
            checkpoint_interval: self.checkpoint_interval,
            max_checkpoints: self.max_checkpoints,
            cache_reuse_min: self.cache_reuse_min,
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::io::Write;

    use tempfile::NamedTempFile;

    use super::*;

    const PROMPT_CACHE_ENV_KEYS: [&str; 6] = [
        "PROXIMA_PROMPT_CACHE_BYTE_BUDGET",
        "PROXIMA_PROMPT_CACHE_MAX_ENTRIES",
        "PROXIMA_PROMPT_CACHE_RING_REWIND_SLACK",
        "PROXIMA_PROMPT_CACHE_CHECKPOINT_INTERVAL",
        "PROXIMA_PROMPT_CACHE_MAX_CHECKPOINTS",
        "PROXIMA_PROMPT_CACHE_CACHE_REUSE_MIN",
    ];

    fn cleared_env() -> Vec<(&'static str, Option<&'static str>)> {
        PROMPT_CACHE_ENV_KEYS
            .iter()
            .map(|key| (*key, None))
            .collect()
    }

    /// AC8: the fluent builder, the TOML loader and the env loader produce
    /// one identical config for the same values, and that config lowers into
    /// the `Copy` shape `ServingConfig` carries.
    #[test]
    fn prompt_cache_builder_matches_toml_and_env_loaders() {
        let via_builder = PromptCacheSettings::builder()
            .byte_budget(1_073_741_824)
            .max_entries(8)
            .ring_rewind_slack(512)
            .checkpoint_interval(1024)
            .max_checkpoints(4)
            .cache_reuse_min(64)
            .build();

        let mut toml_file = NamedTempFile::with_suffix(".toml").expect("create temp toml file");
        writeln!(
            toml_file,
            "byte_budget = 1073741824\nmax_entries = 8\nring_rewind_slack = 512\n\
             checkpoint_interval = 1024\nmax_checkpoints = 4\ncache_reuse_min = 64"
        )
        .expect("write temp toml file");
        let via_file: PromptCacheSettings = conflaguration::from_file(toml_file.path())
            .unwrap_or_else(|err| panic!("from_file failed: {err}"));
        assert_eq!(via_builder, via_file, "builder must match the TOML loader");

        let mut env = cleared_env();
        env.iter_mut().for_each(|(key, value)| {
            *value = match *key {
                "PROXIMA_PROMPT_CACHE_BYTE_BUDGET" => Some("1073741824"),
                "PROXIMA_PROMPT_CACHE_MAX_ENTRIES" => Some("8"),
                "PROXIMA_PROMPT_CACHE_RING_REWIND_SLACK" => Some("512"),
                "PROXIMA_PROMPT_CACHE_CHECKPOINT_INTERVAL" => Some("1024"),
                "PROXIMA_PROMPT_CACHE_MAX_CHECKPOINTS" => Some("4"),
                _ => Some("64"),
            };
        });
        temp_env::with_vars(env, || {
            let via_env = PromptCacheSettings::from_env()
                .unwrap_or_else(|err| panic!("from_env failed: {err}"));
            assert_eq!(via_builder, via_env, "builder must match the env loader");
        });

        let lowered = via_builder.as_prompt_cache_config();
        assert_eq!(lowered.byte_budget, 1_073_741_824);
        assert_eq!(lowered.ring_rewind_slack, 512);
        assert!(lowered.is_enabled());
    }

    /// With nothing set, every loader lands on the off switch.
    #[test]
    fn default_settings_lower_to_the_disabled_config() {
        temp_env::with_vars(cleared_env(), || {
            let from_env = PromptCacheSettings::from_env()
                .unwrap_or_else(|err| panic!("from_env failed: {err}"));
            assert_eq!(from_env, PromptCacheSettings::builder().build());
            assert_eq!(from_env.as_prompt_cache_config(), PromptCacheConfig::off());
            assert!(!from_env.as_prompt_cache_config().is_enabled());
        });
    }
}
