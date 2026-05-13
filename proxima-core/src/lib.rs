//! Foundation error type for the proxima workspace.
//!
//! Two surfaces today:
//!
//! 1. **Legacy `String`-payload variants** (`Decode(String)`, `Config(String)`, etc.) —
//!    used by 51 source files across the workspace. Kept as-is, deprecated.
//! 2. **Typed-payload variants** (`DecodeKind(DecodeError)`, `ConfigKind(ConfigError)`,
//!    etc.) — added alongside the legacy variants. New code uses these; legacy
//!    sites migrate at their own pace.
//!
//! Sub-enum surface is informed by a workspace-wide audit of the 14 String-payload
//! call-site patterns (see `docs/runtime-prime-nostd/discipline.md`, DC2 row). The
//! 6 most-common construction shapes per variant get a named typed kind; the rest
//! fall back to a generic `Other(&'static str)` constructor for the long tail.
//!
//! Sources are erased through `Box<dyn core::error::Error + Send + Sync + 'static>`
//! so the sub-enum surface stays uniform regardless of whether the underlying
//! source is `std::io::Error`, `serde_json::Error`, `FromUtf8Error`, etc.
//! `core::error::Error` is stable since Rust 1.81; the workspace is edition 2024
//! which requires 1.85+, so the trait is always available.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod arch;
#[cfg(feature = "alloc")]
pub mod arena;
pub mod ring;
// per-worker BytesMut reservoir; std-only (thread_local has no no_std
// analog). folded in from the former proxima-io satellite crate.
#[cfg(feature = "std")]
pub mod buffer;
// histogram accumulates count/sum/buckets in AtomicU64; targets without
// 64-bit atomics (cortex-m thumbv7m/em) can't back it lock-free, so gate it out
// there — no_std consumers with 64-bit atomics still get it.
#[cfg(feature = "alloc")]
pub mod batch;
pub mod datagram_batch;
pub mod factory;
#[cfg(target_has_atomic = "64")]
pub mod histogram;
#[cfg(feature = "io-async")]
pub mod io;
pub mod markers;
#[cfg(feature = "park")]
pub mod park;
pub mod per_core;
// std tier only for now: the waiter store is a std Mutex. the no_std
// tier needs a bounded waiter table over SlotPark — deliberate follow-on.
#[cfg(feature = "std")]
pub mod signal;
// runtime-agnostic timer primitives (folded from the former proxima-time
// crate). Unconditional: compiles at every tier, falling back to a
// panics-on-use driver when no `time-driver-*` feature/profile is active.
pub mod time;
// lock-free live-swappable cell (arc-swap); std-tier, opt-in via `live`.
#[cfg(feature = "live")]
pub mod live;

/// Build-time sizing constants emitted by `build.rs` from `proxima-core.toml`
/// (principle 12). Only the `no_std + no-alloc` tier is bounded by these; the
/// alloc tier grows dynamically.
pub mod sized {
    include!(concat!(env!("OUT_DIR"), "/proxima_core_sized.rs"));
}

#[cfg(feature = "registry")]
pub use factory::FactoryRegistry;
pub use factory::Named;
#[cfg(feature = "config")]
pub use factory::{Composition, Factory, FactorySpec};

#[cfg(feature = "alloc")]
use alloc::boxed::Box;
#[cfg(feature = "alloc")]
use alloc::string::String;

use core::time::Duration;

use thiserror::Error;

/// Erased source for typed sub-enum variants. Concrete error types
/// (`std::io::Error`, `serde_json::Error`, `FromUtf8Error`, `ParseIntError`,
/// etc.) all impl `core::error::Error` and box cleanly through this alias.
#[cfg(feature = "alloc")]
pub type ErrorSource = Box<dyn core::error::Error + Send + Sync + 'static>;

// ---------------------------------------------------------------------------
// Typed sub-enums (alongside the legacy String variants)
// ---------------------------------------------------------------------------

/// Decode failure during wire / format parsing.
#[derive(Debug, Error)]
#[cfg(feature = "alloc")]
pub enum DecodeError {
    #[error("missing header terminator")]
    MissingHeaderTerminator,
    #[error("empty status line")]
    EmptyStatusLine,
    #[error("malformed status line: {raw}")]
    MalformedStatusLine { raw: String },
    #[error("chunked: size CRLF missing")]
    ChunkedSizeMissing,
    #[error("chunked: payload short")]
    ChunkedPayloadShort,
    #[error("not utf-8 ({context}): {source}")]
    NotUtf8 {
        context: &'static str,
        source: ErrorSource,
    },
    #[error("parse {context}: {source}")]
    Parse {
        context: &'static str,
        source: ErrorSource,
    },
    #[error("missing required field: {field}")]
    MissingField { field: &'static str },
    #[error("decode ({context}): {source}")]
    Other {
        context: &'static str,
        source: ErrorSource,
    },
}

/// Encode failure during wire / format serialization.
#[derive(Debug, Error)]
#[cfg(feature = "alloc")]
pub enum EncodeError {
    #[error("json encode: {source}")]
    Json { source: ErrorSource },
    #[error("format convert ({context}): {source}")]
    FormatConvert {
        context: &'static str,
        source: ErrorSource,
    },
    #[error("render ({context}): {source}")]
    Render {
        context: &'static str,
        source: ErrorSource,
    },
    #[error("base64: {source}")]
    Base64 { source: ErrorSource },
    #[error("requires base64 string input")]
    RequiresBase64Input,
    #[error("encode ({context}): {source}")]
    Other {
        context: &'static str,
        source: ErrorSource,
    },
}

/// Upstream / I/O-side failure.
#[derive(Debug, Error)]
#[cfg(feature = "alloc")]
pub enum UpstreamError {
    #[error("io ({operation}): {source}")]
    Io {
        operation: &'static str,
        source: ErrorSource,
    },
    #[error("upstream returned status {status}")]
    BadStatus { status: u16 },
    #[error("upstream cancelled")]
    Cancelled,
    #[error("protocol ({context}): {source}")]
    Protocol {
        context: &'static str,
        source: ErrorSource,
    },
    #[error("connect ({context}): {source}")]
    Connect {
        context: &'static str,
        source: ErrorSource,
    },
    #[error("upstream ({context}): {source}")]
    Other {
        context: &'static str,
        source: ErrorSource,
    },
}

/// Configuration / spec error.
#[derive(Debug, Error)]
#[cfg(feature = "alloc")]
pub enum ConfigError {
    #[error("{component}: requires field '{field}'")]
    RequiredField {
        component: &'static str,
        field: &'static str,
    },
    #[error("{component}: requires an inner pipe")]
    MissingInnerPipe { component: &'static str },
    #[error("no upstreams configured")]
    NoUpstreams,
    #[error("parse ({context}): {source}")]
    Parse {
        context: &'static str,
        source: ErrorSource,
    },
    #[error("spawn failed ({context}): {source}")]
    SpawnFailed {
        context: &'static str,
        source: ErrorSource,
    },
    #[error("config ({context}): {source}")]
    Other {
        context: &'static str,
        source: ErrorSource,
    },
}

/// Body-stream / worker-side failure.
#[derive(Debug, Error)]
#[cfg(feature = "alloc")]
pub enum BodyError {
    #[error("body cancelled")]
    Cancelled,
    #[error("worker dropped sender ({context})")]
    WorkerDropped { context: &'static str },
    #[error("worker panicked ({context})")]
    WorkerPanic { context: &'static str },
    #[error("serialize ({context}): {source}")]
    Serialize {
        context: &'static str,
        source: ErrorSource,
    },
    #[error("body ({context}): {source}")]
    Other {
        context: &'static str,
        source: ErrorSource,
    },
}

/// Resource-not-found error.
#[derive(Debug, Error)]
#[cfg(feature = "alloc")]
pub enum NotFoundError {
    #[error("no pipe registered as '{name}'")]
    Pipe { name: String },
    #[error("{kind} '{name}' not found")]
    Resource { kind: &'static str, name: String },
}

/// Registry-level error (factories, listen protocols, codecs).
#[derive(Debug, Error)]
#[cfg(feature = "alloc")]
pub enum RegistryError {
    #[error("no {kind} named '{name}'")]
    NotRegistered { kind: &'static str, name: String },
    #[error("{kind} '{name}' already registered")]
    AlreadyRegistered { kind: &'static str, name: String },
}

/// Recording / replay sink-side error.
#[derive(Debug, Error)]
#[cfg(feature = "alloc")]
pub enum RecordError {
    #[error("recording io ({operation}): {source}")]
    Io {
        operation: &'static str,
        source: ErrorSource,
    },
    #[error("recording codec ({operation}): {source}")]
    Codec {
        operation: &'static str,
        source: ErrorSource,
    },
    #[error("recording frame exceeds u32::MAX bytes")]
    FrameTooLarge,
    #[error("recording version mismatch: expected {expected}, got {got}")]
    VersionMismatch { expected: u8, got: u8 },
    #[error("recording buffer full at capacity {capacity}")]
    BufferFull { capacity: usize },
}

// ---------------------------------------------------------------------------
// ProximaError — legacy String-payload variants + new typed variants
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum ProximaError {
    /// Legacy String-payload variants. Construct via `ProximaError::Decode(...)`
    /// today; migrate to `ProximaError::DecodeKind(DecodeError::...)` over time.
    #[error("upstream error: {0}")]
    #[cfg(feature = "alloc")]
    Upstream(String),

    #[error("timeout after {0:?}")]
    Timeout(Duration),

    #[error("rate limited")]
    RateLimited,

    #[error("no upstream had data")]
    NoData,

    #[error("config: {0}")]
    #[cfg(feature = "alloc")]
    Config(String),

    #[error("io: {0}")]
    #[cfg(feature = "std")]
    Io(#[from] std::io::Error),

    #[error("decode: {0}")]
    #[cfg(feature = "alloc")]
    Decode(String),

    #[error("encode: {0}")]
    #[cfg(feature = "alloc")]
    Encode(String),

    #[error("body stream: {0}")]
    #[cfg(feature = "alloc")]
    Body(String),

    #[error("not found: {0}")]
    #[cfg(feature = "alloc")]
    NotFound(String),

    /// A deliberate refusal, not a failure — the payload is the response
    /// body verbatim (e.g. a `filter` predicate's `RejectMode::Drop`, which
    /// renders as 403 with this payload as the body).
    #[error("forbidden: {0}")]
    #[cfg(feature = "alloc")]
    Forbidden(String),

    #[error("registry: {0}")]
    #[cfg(feature = "alloc")]
    Registry(String),

    #[error("retries exhausted after {attempts} attempts: {last}")]
    #[cfg(feature = "alloc")]
    RetriesExhausted { attempts: usize, last: String },

    #[error("recording: {0}")]
    #[cfg(feature = "alloc")]
    Record(String),

    #[error("replay miss: {fingerprint}")]
    #[cfg(feature = "alloc")]
    ReplayMiss { fingerprint: String },

    /// Typed payload variants. Carry sub-enum kinds with concrete fields
    /// (e.g. `Decode(format!("missing field 'name'"))` becomes
    /// `DecodeKind(DecodeError::MissingField { field: "name" })`). Match
    /// callers can now inspect the kind without parsing strings.
    #[error("decode: {0}")]
    #[cfg(feature = "alloc")]
    DecodeKind(#[from] DecodeError),

    #[error("encode: {0}")]
    #[cfg(feature = "alloc")]
    EncodeKind(#[from] EncodeError),

    #[error("upstream: {0}")]
    #[cfg(feature = "alloc")]
    UpstreamKind(#[from] UpstreamError),

    #[error("config: {0}")]
    #[cfg(feature = "alloc")]
    ConfigKind(#[from] ConfigError),

    #[error("body: {0}")]
    #[cfg(feature = "alloc")]
    BodyKind(#[from] BodyError),

    #[error("not found: {0}")]
    #[cfg(feature = "alloc")]
    NotFoundKind(#[from] NotFoundError),

    #[error("registry: {0}")]
    #[cfg(feature = "alloc")]
    RegistryKind(#[from] RegistryError),

    #[error("recording: {0}")]
    #[cfg(feature = "alloc")]
    RecordKind(#[from] RecordError),
}

impl ProximaError {
    /// Whether this error class is worth retrying. Covers both legacy and
    /// typed variants — the heuristic matches `Upstream*` (transient I/O),
    /// `Timeout`, `Body*` (worker churn), and (under std) `Io`.
    #[must_use]
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::Timeout(_) | Self::RateLimited => matches!(self, Self::Timeout(_)),
            #[cfg(feature = "alloc")]
            Self::Upstream(_) | Self::Body(_) => true,
            #[cfg(feature = "std")]
            Self::Io(_) => true,
            #[cfg(feature = "alloc")]
            Self::UpstreamKind(kind) => !matches!(kind, UpstreamError::BadStatus { .. }),
            #[cfg(feature = "alloc")]
            Self::BodyKind(kind) => matches!(
                kind,
                BodyError::WorkerDropped { .. } | BodyError::WorkerPanic { .. }
            ),
            _ => false,
        }
    }
}

pub type ProximaResult<T> = Result<T, ProximaError>;

#[cfg(test)]
#[cfg(feature = "alloc")]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use alloc::format;

    #[test]
    fn timeout_is_retryable() {
        assert!(ProximaError::Timeout(Duration::from_secs(1)).is_retryable());
    }

    #[test]
    fn rate_limited_is_not_retryable() {
        assert!(!ProximaError::RateLimited.is_retryable());
    }

    #[test]
    fn legacy_upstream_is_retryable() {
        assert!(ProximaError::Upstream("transient".into()).is_retryable());
    }

    #[test]
    fn typed_upstream_io_is_retryable_but_bad_status_is_not() {
        let io = UpstreamError::Io {
            operation: "read",
            source: Box::new(LightweightTestError("transient")),
        };
        assert!(ProximaError::from(io).is_retryable());

        let bad = UpstreamError::BadStatus { status: 500 };
        assert!(!ProximaError::from(bad).is_retryable());
    }

    #[test]
    fn decode_kind_carries_typed_payload() {
        let err = DecodeError::MissingField {
            field: "pipeline_id",
        };
        let display = format!("{}", err);
        assert!(display.contains("pipeline_id"));
    }

    #[test]
    fn from_decode_error_to_proxima_error_via_from_impl() {
        let err: ProximaError = DecodeError::EmptyStatusLine.into();
        assert!(matches!(
            err,
            ProximaError::DecodeKind(DecodeError::EmptyStatusLine)
        ));
    }

    #[test]
    fn config_required_field_renders_component_and_field() {
        let err = ConfigError::RequiredField {
            component: "auth",
            field: "allow",
        };
        let display = format!("{}", err);
        assert!(display.contains("auth"));
        assert!(display.contains("allow"));
    }

    #[test]
    fn registry_not_registered_uses_kind_and_name() {
        let err = RegistryError::NotRegistered {
            kind: "codec factory",
            name: "json".into(),
        };
        let display = format!("{}", err);
        assert!(display.contains("codec factory"));
        assert!(display.contains("json"));
    }

    #[test]
    fn record_version_mismatch_carries_numbers() {
        let err = RecordError::VersionMismatch {
            expected: 3,
            got: 2,
        };
        let display = format!("{}", err);
        assert!(display.contains('3'));
        assert!(display.contains('2'));
    }

    #[derive(Debug)]
    struct LightweightTestError(&'static str);
    impl core::fmt::Display for LightweightTestError {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.write_str(self.0)
        }
    }
    impl core::error::Error for LightweightTestError {}
}
