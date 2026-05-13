//! Std-level recording destination primitives: the format codec selector and
//! the `path + format` descriptor the durable terminals open.
//!
//! These live below the `config` feature (no `bon`/`conflaguration`) so the
//! spigot terminal ([`crate::pipe::lazy::LazyFanOut`]) is available to any `std`
//! consumer. The config surface ([`crate::pipe::config::SinkConfig`]) is the
//! fluent/serde wrapper that lowers to [`SinkSpec`].

use crate::{BinFormat, Format, JsonFormat};
use proxima_core::ProximaError;
use serde::{Deserialize, Serialize};

/// On-disk recording format — the config-selected CODEC axis (not a peer
/// type). Adding a format is one [`Format`] impl + one variant here; choosing
/// one per sink is config.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FormatKind {
    /// `[u32 len|flag]` + zstd block of postcard frames (compact, default).
    #[default]
    Bin,
    /// One JSON line per event (human-readable).
    Json,
}

impl FormatKind {
    /// Instantiate the codec for this format.
    pub fn codec(self) -> Result<Box<dyn Format>, ProximaError> {
        match self {
            Self::Bin => Ok(Box::new(BinFormat::new()?)),
            Self::Json => Ok(Box::new(JsonFormat::new())),
        }
    }
}

/// One durable destination: where events go + which codec writes them. The
/// builder-free descriptor [`crate::pipe::lazy::LazyFanOut`] opens.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SinkSpec {
    /// Filesystem path the durable log appends to.
    pub path: String,
    /// Which format codec this destination uses.
    pub format: FormatKind,
    /// zstd block-compressor level for `Bin` (CPU/ratio lever). `None` uses
    /// the codec default; ignored by `Json`.
    pub zstd_level: Option<i32>,
}

impl SinkSpec {
    #[must_use]
    pub fn new(path: impl Into<String>, format: FormatKind) -> Self {
        Self {
            path: path.into(),
            format,
            zstd_level: None,
        }
    }

    /// Set the `Bin` block-compressor level (no effect on `Json`).
    #[must_use]
    pub fn with_zstd_level(mut self, zstd_level: i32) -> Self {
        self.zstd_level = Some(zstd_level);
        self
    }

    /// Instantiate the codec for this destination, honoring `zstd_level`.
    pub fn codec(&self) -> Result<Box<dyn Format>, ProximaError> {
        match (self.format, self.zstd_level) {
            (FormatKind::Bin, Some(level)) => Ok(Box::new(BinFormat::with_level(level)?)),
            (FormatKind::Bin, None) => Ok(Box::new(BinFormat::new()?)),
            (FormatKind::Json, _) => Ok(Box::new(JsonFormat::new())),
        }
    }
}
