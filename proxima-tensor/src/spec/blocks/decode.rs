use serde::{Deserialize, Serialize};

use crate::spec::{CacheMask, CacheStrategy};

/// How a model is executed, as opposed to what math it computes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "config", derive(bon::Builder, conflaguration::Settings))]
#[cfg_attr(feature = "config", settings(prefix = "PROXIMA_DECODE"))]
#[serde(deny_unknown_fields, default)]
pub struct Decode {
    /// Lower a second program that keeps every new row's logits, for checking drafted tokens.
    #[cfg_attr(feature = "config", setting(default = false))]
    pub speculative_verify: bool,
    /// Command-buffer chunks one decode step is submitted in; more chunks bound per-submission work.
    #[cfg_attr(feature = "config", setting(default = 1))]
    pub command_buffer_chunks: u32,
    /// Whether the decode program keeps a key/value cache across steps.
    #[cfg_attr(feature = "config", setting(skip))]
    pub cache_strategy: CacheStrategy,
    /// Which mask the cached block carries; read only when `cache_strategy` is `Cached`.
    #[cfg_attr(feature = "config", setting(skip))]
    pub cache_mask: CacheMask,
    /// Positions per prefill program; None prefills one position per program.
    pub prefill_width: Option<u32>,
    /// Project only the last position's row to logits; a verify program keeps every row.
    #[cfg_attr(feature = "config", setting(default = true))]
    pub last_row_only: bool,
    /// Windowed layers keep their keys in one sliding ring slot instead of the full cache.
    #[cfg_attr(feature = "config", setting(default = false))]
    pub sliding_kv_ring: bool,
}

impl Default for Decode {
    fn default() -> Self {
        Self {
            speculative_verify: false,
            command_buffer_chunks: 1,
            cache_strategy: CacheStrategy::default(),
            cache_mask: CacheMask::default(),
            prefill_width: None,
            last_row_only: true,
            sliding_kv_ring: false,
        }
    }
}

#[cfg(feature = "config")]
impl conflaguration::Validate for Decode {
    fn validate(&self) -> conflaguration::Result<()> {
        if self.command_buffer_chunks == 0 {
            return Err(conflaguration::Error::Validation {
                errors: alloc::vec![conflaguration::ValidationMessage::new(
                    "command_buffer_chunks",
                    "must be positive",
                )],
            });
        }
        Ok(())
    }
}
