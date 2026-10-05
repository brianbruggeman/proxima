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
    #[setting(default = 2147483648)]
    #[builder(default = 2147483648)]
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
    #[setting(default = 2048)]
    #[builder(default = 2048)]
    pub checkpoint_interval: u32,
    /// See [`crate::PromptCacheConfig::max_checkpoints`].
    #[setting(default = 4)]
    #[builder(default = 4)]
    pub max_checkpoints: u32,
    /// See [`crate::PromptCacheConfig::cache_reuse_min`].
    #[setting(default = 0)]
    #[builder(default = 0)]
    pub cache_reuse_min: u32,
    /// See [`crate::PromptCacheConfig::prewarm_chunk_tokens`].
    #[setting(default = 256)]
    #[builder(default = 256)]
    pub prewarm_chunk_tokens: u32,
    /// See [`crate::PromptCacheConfig::follow_up_branches`].
    #[setting(default = 0)]
    #[builder(default = 0)]
    pub follow_up_branches: u32,
    /// See [`crate::PromptCacheConfig::follow_up_max_tokens`].
    #[setting(default = 48)]
    #[builder(default = 48)]
    pub follow_up_max_tokens: u32,
    /// See [`crate::PromptCacheConfig::follow_up_temperature_milli`].
    #[setting(default = 800)]
    #[builder(default = 800)]
    pub follow_up_temperature_milli: u32,
    /// See [`crate::PromptCacheConfig::min_similarity_milli`].
    #[setting(default = 100)]
    #[builder(default = 100)]
    pub min_similarity_milli: u32,
    /// See [`crate::PromptCacheConfig::block_tokens`]. `None` takes the
    /// owning block size (`kv.block_tokens` under `ServingSettings`, 64
    /// standalone); `Some` is an explicit choice and wins.
    pub block_tokens: Option<u32>,
    /// See [`crate::PromptCacheConfig::seal_horizon_rows`].
    #[setting(default = 256)]
    #[builder(default = 256)]
    pub seal_horizon_rows: u32,
    /// See [`crate::PromptCacheConfig::bloom_bits_per_entry`].
    #[setting(default = 4096)]
    #[builder(default = 4096)]
    pub bloom_bits_per_entry: u32,
    /// See [`crate::PromptCacheConfig::bloom_hashes`].
    #[setting(default = 4)]
    #[builder(default = 4)]
    pub bloom_hashes: u32,
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
            prewarm_chunk_tokens: self.prewarm_chunk_tokens,
            follow_up_branches: self.follow_up_branches,
            follow_up_max_tokens: self.follow_up_max_tokens,
            follow_up_temperature_milli: self.follow_up_temperature_milli,
            min_similarity_milli: self.min_similarity_milli,
            block_tokens: match self.block_tokens {
                Some(value) => value,
                None => PromptCacheConfig::standard().block_tokens,
            },
            seal_horizon_rows: self.seal_horizon_rows,
            bloom_bits_per_entry: self.bloom_bits_per_entry,
            bloom_hashes: self.bloom_hashes,
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::io::Write;

    use tempfile::NamedTempFile;

    use super::*;

    const PROMPT_CACHE_ENV_KEYS: [&str; 15] = [
        "PROXIMA_PROMPT_CACHE_BYTE_BUDGET",
        "PROXIMA_PROMPT_CACHE_MAX_ENTRIES",
        "PROXIMA_PROMPT_CACHE_RING_REWIND_SLACK",
        "PROXIMA_PROMPT_CACHE_CHECKPOINT_INTERVAL",
        "PROXIMA_PROMPT_CACHE_MAX_CHECKPOINTS",
        "PROXIMA_PROMPT_CACHE_CACHE_REUSE_MIN",
        "PROXIMA_PROMPT_CACHE_PREWARM_CHUNK_TOKENS",
        "PROXIMA_PROMPT_CACHE_FOLLOW_UP_BRANCHES",
        "PROXIMA_PROMPT_CACHE_FOLLOW_UP_MAX_TOKENS",
        "PROXIMA_PROMPT_CACHE_FOLLOW_UP_TEMPERATURE_MILLI",
        "PROXIMA_PROMPT_CACHE_MIN_SIMILARITY_MILLI",
        "PROXIMA_PROMPT_CACHE_BLOCK_TOKENS",
        "PROXIMA_PROMPT_CACHE_BLOOM_BITS_PER_ENTRY",
        "PROXIMA_PROMPT_CACHE_BLOOM_HASHES",
        "PROXIMA_PROMPT_CACHE_SEAL_HORIZON_ROWS",
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
            .prewarm_chunk_tokens(128)
            .follow_up_branches(3)
            .follow_up_max_tokens(64)
            .follow_up_temperature_milli(700)
            .min_similarity_milli(250)
            .block_tokens(32)
            .seal_horizon_rows(512)
            .bloom_bits_per_entry(8192)
            .bloom_hashes(6)
            .build();

        let mut toml_file = NamedTempFile::with_suffix(".toml").expect("create temp toml file");
        writeln!(
            toml_file,
            "byte_budget = 1073741824\nmax_entries = 8\nring_rewind_slack = 512\n\
             checkpoint_interval = 1024\nmax_checkpoints = 4\ncache_reuse_min = 64\n\
             prewarm_chunk_tokens = 128\nfollow_up_branches = 3\nfollow_up_max_tokens = 64\n\
             follow_up_temperature_milli = 700\nmin_similarity_milli = 250\nblock_tokens = 32\nseal_horizon_rows = 512\nbloom_bits_per_entry = 8192\nbloom_hashes = 6"
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
                "PROXIMA_PROMPT_CACHE_CACHE_REUSE_MIN" => Some("64"),
                "PROXIMA_PROMPT_CACHE_PREWARM_CHUNK_TOKENS" => Some("128"),
                "PROXIMA_PROMPT_CACHE_FOLLOW_UP_BRANCHES" => Some("3"),
                "PROXIMA_PROMPT_CACHE_FOLLOW_UP_MAX_TOKENS" => Some("64"),
                "PROXIMA_PROMPT_CACHE_MIN_SIMILARITY_MILLI" => Some("250"),
                "PROXIMA_PROMPT_CACHE_BLOCK_TOKENS" => Some("32"),
                "PROXIMA_PROMPT_CACHE_SEAL_HORIZON_ROWS" => Some("512"),
                "PROXIMA_PROMPT_CACHE_BLOOM_BITS_PER_ENTRY" => Some("8192"),
                "PROXIMA_PROMPT_CACHE_BLOOM_HASHES" => Some("6"),
                _ => Some("700"),
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
        assert_eq!(lowered.prewarm_chunk_tokens, 128);
        assert_eq!(lowered.follow_up_branches, 3);
        assert_eq!(lowered.follow_up_max_tokens, 64);
        assert_eq!(lowered.follow_up_temperature_milli, 700);
        assert_eq!(lowered.min_similarity_milli, 250);
        assert_eq!(lowered.block_tokens, 32);
        assert_eq!(lowered.seal_horizon_rows, 512);
        assert_eq!(lowered.bloom_bits_per_entry, 8192);
        assert_eq!(lowered.bloom_hashes, 6);
        assert!(lowered.is_enabled());
    }

    #[test]
    fn seal_horizon_defaults_to_256_rows() {
        let built = PromptCacheSettings::builder().build();

        assert_eq!(PromptCacheConfig::standard().seal_horizon_rows, 256);
        assert_eq!(PromptCacheConfig::off().seal_horizon_rows, 256);
        assert_eq!(built.seal_horizon_rows, 256);
        assert_eq!(built.as_prompt_cache_config(), PromptCacheConfig::standard());
    }

    /// With nothing set, every loader lands on the shipped default: the
    /// cache is on.
    #[test]
    fn default_settings_lower_to_the_standard_config() {
        temp_env::with_vars(cleared_env(), || {
            let from_env = PromptCacheSettings::from_env()
                .unwrap_or_else(|err| panic!("from_env failed: {err}"));
            assert_eq!(from_env, PromptCacheSettings::builder().build());
            assert_eq!(
                from_env.as_prompt_cache_config(),
                PromptCacheConfig::standard()
            );
            assert_eq!(PromptCacheConfig::default(), PromptCacheConfig::standard());
            assert!(from_env.as_prompt_cache_config().is_enabled());
        });
    }

    /// `PROXIMA_PROMPT_CACHE_BYTE_BUDGET=0` is the off switch and lowers to
    /// the same config as [`PromptCacheConfig::off`].
    #[test]
    fn a_zero_byte_budget_from_the_environment_turns_the_cache_off() {
        let mut env = cleared_env();
        env[0].1 = Some("0");
        temp_env::with_vars(env, || {
            let from_env = PromptCacheSettings::from_env()
                .unwrap_or_else(|err| panic!("from_env failed: {err}"));
            assert_eq!(from_env.as_prompt_cache_config(), PromptCacheConfig::off());
            assert!(!from_env.as_prompt_cache_config().is_enabled());
        });
    }

    #[test]
    fn serving_scalars_prompt_cache_block_tokens_is_optional() {
        temp_env::with_vars(cleared_env(), || {
            let unset = PromptCacheSettings::from_env()
                .unwrap_or_else(|err| panic!("from_env failed: {err}"));
            assert_eq!(unset.block_tokens, None);
            assert_eq!(unset.as_prompt_cache_config().block_tokens, 64);
        });

        let mut env = cleared_env();
        env.iter_mut().for_each(|(key, value)| {
            if *key == "PROXIMA_PROMPT_CACHE_BLOCK_TOKENS" {
                *value = Some("128");
            }
        });
        temp_env::with_vars(env, || {
            let set = PromptCacheSettings::from_env()
                .unwrap_or_else(|err| panic!("from_env failed: {err}"));
            assert_eq!(set.block_tokens, Some(128));
            assert_eq!(set.as_prompt_cache_config().block_tokens, 128);
        });
        let defaults_toml = "byte_budget = 2147483648\nmax_entries = 4\nring_rewind_slack = 256\n\
             checkpoint_interval = 2048\nmax_checkpoints = 4\ncache_reuse_min = 0\n\
             prewarm_chunk_tokens = 256\nfollow_up_branches = 0\nfollow_up_max_tokens = 48\n\
             follow_up_temperature_milli = 800\nmin_similarity_milli = 100\n\
             bloom_bits_per_entry = 4096\nbloom_hashes = 4\n";

        let mut with_key = NamedTempFile::with_suffix(".toml").expect("create temp toml file");
        writeln!(with_key, "{defaults_toml}block_tokens = 128").expect("write temp toml file");
        let from_toml: PromptCacheSettings = conflaguration::from_file(with_key.path())
            .unwrap_or_else(|err| panic!("from_file failed: {err}"));
        assert_eq!(from_toml.block_tokens, Some(128));

        let mut without_key = NamedTempFile::with_suffix(".toml").expect("create temp toml file");
        writeln!(without_key, "{defaults_toml}").expect("write temp toml file");
        let from_toml_unset: PromptCacheSettings = conflaguration::from_file(without_key.path())
            .unwrap_or_else(|err| panic!("from_file failed: {err}"));
        assert_eq!(from_toml_unset.block_tokens, None);

        let via_builder = PromptCacheSettings::builder().block_tokens(128).build();
        assert_eq!(via_builder, from_toml);
    }
}
