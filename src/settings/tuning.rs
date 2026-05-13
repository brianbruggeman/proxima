//! Tunable knobs. Each struct derives `conflaguration::Settings` for
//! env-var overrides (`PROXIMA_HTTP_BUFFER_BYTES=32768`) and
//! `bon::Builder` for fluent construction. Defaults match the values
//! that used to live as `const`s in source.

use bon::Builder;
use conflaguration::{Settings, Validate, ValidationMessage};
use serde::{Deserialize, Serialize};

/// HTTP framing-layer tunables. These were `const`s in `listeners::
/// http` and `h2::*` before Stage B.
#[derive(Debug, Clone, Builder, Deserialize, Serialize, Settings)]
#[settings(prefix = "HTTP")]
#[builder(derive(Clone, Debug))]
pub struct HttpTuning {
    /// Per-connection response output buffer initial capacity (bytes).
    /// Reused across responses on the same connection; grows to the
    /// largest response head + chunk seen and stays there.
    #[setting(default = 8192)]
    #[serde(default = "default_response_buffer_bytes")]
    #[builder(default = default_response_buffer_bytes())]
    pub response_buffer_bytes: usize,

    /// Stack-allocated per-connection read slot (bytes). 16 KiB fits
    /// a typical request head + small body in one syscall.
    #[setting(default = 16384)]
    #[serde(default = "default_read_buffer_bytes")]
    #[builder(default = default_read_buffer_bytes())]
    pub read_buffer_bytes: usize,

    /// HeaderList initial capacity (entries) per request.
    #[setting(default = 32)]
    #[serde(default = "default_headers_capacity")]
    #[builder(default = default_headers_capacity())]
    pub headers_capacity: usize,

    /// HTTP/1.1 maximum header lines per request.
    #[setting(default = 100)]
    #[serde(default = "default_h1_max_headers")]
    #[builder(default = default_h1_max_headers())]
    pub h1_max_headers: usize,

    /// HTTP/2 maximum frame size we announce to peers in SETTINGS.
    /// RFC 7540 §6.5.2 — minimum 16384, maximum 16777215.
    #[setting(default = 16384)]
    #[serde(default = "default_h2_max_frame_size")]
    #[builder(default = default_h2_max_frame_size())]
    pub h2_max_frame_size: u32,

    /// HTTP/2 maximum concurrent streams per connection.
    #[setting(default = 100)]
    #[serde(default = "default_h2_max_concurrent_streams")]
    #[builder(default = default_h2_max_concurrent_streams())]
    pub h2_max_concurrent_streams: u32,

    /// HTTP/2 HPACK dynamic table size we announce.
    #[setting(default = 4096)]
    #[serde(default = "default_h2_header_table_size")]
    #[builder(default = default_h2_header_table_size())]
    pub h2_header_table_size: u32,

    /// Default max body bytes — `None` means unbounded. CLI / config
    /// can override per listener.
    #[setting(skip)]
    #[serde(default)]
    pub max_body_bytes: Option<usize>,
}

impl Default for HttpTuning {
    fn default() -> Self {
        Self::builder().build()
    }
}

impl Validate for HttpTuning {
    fn validate(&self) -> conflaguration::Result<()> {
        let mut errors = Vec::new();
        if self.response_buffer_bytes == 0 {
            errors.push(ValidationMessage::new(
                "response_buffer_bytes",
                "must be > 0",
            ));
        }
        if self.read_buffer_bytes == 0 {
            errors.push(ValidationMessage::new("read_buffer_bytes", "must be > 0"));
        }
        if !(16_384..=16_777_215).contains(&self.h2_max_frame_size) {
            errors.push(ValidationMessage::new(
                "h2_max_frame_size",
                "must be in [16384, 16777215] per RFC 7540 §6.5.2",
            ));
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(conflaguration::Error::Validation { errors })
        }
    }
}

fn default_response_buffer_bytes() -> usize {
    8192
}
fn default_read_buffer_bytes() -> usize {
    16_384
}
fn default_headers_capacity() -> usize {
    32
}
fn default_h1_max_headers() -> usize {
    100
}
fn default_h2_max_frame_size() -> u32 {
    16_384
}
fn default_h2_max_concurrent_streams() -> u32 {
    100
}
fn default_h2_header_table_size() -> u32 {
    4_096
}

/// zstd compression tunables.
#[derive(Debug, Clone, Builder, Deserialize, Serialize, Settings)]
#[settings(prefix = "ZSTD")]
#[builder(derive(Clone, Debug))]
pub struct ZstdTuning {
    /// Compression level for recording sinks. Range: 1 (fastest) to
    /// 22 (densest). Default 3 matches zstd's "balanced" preset.
    #[setting(default = 3)]
    #[serde(default = "default_zstd_level")]
    #[builder(default = default_zstd_level())]
    pub compression_level: i32,
}

impl Default for ZstdTuning {
    fn default() -> Self {
        Self::builder().build()
    }
}

impl Validate for ZstdTuning {
    fn validate(&self) -> conflaguration::Result<()> {
        if !(1..=22).contains(&self.compression_level) {
            return Err(conflaguration::Error::Validation {
                errors: vec![ValidationMessage::new(
                    "compression_level",
                    "must be in [1, 22] per zstd",
                )],
            });
        }
        Ok(())
    }
}

fn default_zstd_level() -> i32 {
    3
}

/// Per-worker pooled-buffer tunables.
#[derive(Debug, Clone, Builder, Deserialize, Serialize, Settings)]
#[settings(prefix = "BUFFER_POOL")]
#[builder(derive(Clone, Debug))]
pub struct BufferPoolTuning {
    /// Pool depth per worker thread. Buffers are returned on drop
    /// when the pool isn't full; allocated fresh otherwise.
    #[setting(default = 256)]
    #[serde(default = "default_pool_depth")]
    #[builder(default = default_pool_depth())]
    pub depth_per_worker: usize,

    /// Per-buffer capacity (bytes).
    #[setting(default = 16384)]
    #[serde(default = "default_buffer_bytes")]
    #[builder(default = default_buffer_bytes())]
    pub buffer_bytes: usize,
}

impl Default for BufferPoolTuning {
    fn default() -> Self {
        Self::builder().build()
    }
}

impl Validate for BufferPoolTuning {
    fn validate(&self) -> conflaguration::Result<()> {
        let mut errors = Vec::new();
        if self.depth_per_worker == 0 {
            errors.push(ValidationMessage::new("depth_per_worker", "must be > 0"));
        }
        if self.buffer_bytes == 0 {
            errors.push(ValidationMessage::new("buffer_bytes", "must be > 0"));
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(conflaguration::Error::Validation { errors })
        }
    }
}

fn default_pool_depth() -> usize {
    256
}
fn default_buffer_bytes() -> usize {
    16_384
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_valid() {
        HttpTuning::default()
            .validate()
            .expect("http defaults valid");
        ZstdTuning::default()
            .validate()
            .expect("zstd defaults valid");
        BufferPoolTuning::default()
            .validate()
            .expect("pool defaults valid");
    }

    #[test]
    fn builder_constructs_with_defaults_and_overrides() {
        let custom = HttpTuning::builder().response_buffer_bytes(32_768).build();
        assert_eq!(custom.response_buffer_bytes, 32_768);
        // Untouched fields keep their defaults.
        assert_eq!(custom.read_buffer_bytes, default_read_buffer_bytes());
        assert_eq!(custom.h2_max_frame_size, default_h2_max_frame_size());
    }

    #[test]
    fn h2_max_frame_size_under_minimum_fails_validation() {
        let bad = HttpTuning::builder().h2_max_frame_size(1024).build();
        let outcome = bad.validate();
        assert!(outcome.is_err());
    }

    #[test]
    fn zstd_level_above_22_fails_validation() {
        let bad = ZstdTuning::builder().compression_level(50).build();
        let outcome = bad.validate();
        assert!(outcome.is_err());
    }
}
