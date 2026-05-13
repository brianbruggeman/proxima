//! Wide grounds — dispatch units that touch host state (real
//! filesystem, real entropy, real clock).
//!
//! These exercise G1 (effect tracking) and G2 (capability tokens):
//! - They do NOT impl the `Without*` markers for the effects they
//!   have. A chain that includes one of these is provably NOT
//!   `WithoutFilesystem` / `WithoutRandom` / `WithoutTime`.
//! - Privileged constructors require the matching capability token
//!   (`CapFilesystem`, etc.), so only the trust boundary can build
//!   them.
//!
//! # Determinism via type discrimination
//!
//! Entropy and clock split into two TYPES, not one enum:
//! - [`OsEntropy`] — reads OS entropy; non-`Deterministic`.
//! - [`SeededEntropy`] — deterministic PRNG from a seed; impls
//!   `Deterministic` + `Reproducible`.
//! - [`RealClock`] — reads wall-clock time; non-`Deterministic`.
//! - [`FixedClock`] — emits a fixed epoch; impls `Deterministic` +
//!   `Reproducible`.
//!
//! # Stubs vs real I/O
//!
//! The current implementations are STUBS — they return placeholder
//! bytes / canned errors. Real I/O (`std::fs::read`, `getrandom`,
//! `SystemTime::now`) lands when the architectural proof (C8e)
//! needs it. The structural shape (types + markers + cap-gating)
//! is what's load-bearing for G1+G2.

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;
use core::future::Future;
use std::path::PathBuf;

use proxima_primitives::pipe::ProximaError;
use proxima_primitives::pipe::SendPipe;

use super::capabilities::CapFilesystem;
use super::markers::{
    AllocFree, Deterministic, IdempotentSideEffectFree, NoStd, Reproducible, WithoutFilesystem,
    WithoutNetwork, WithoutRandom, WithoutSpawn, WithoutTime,
};
use super::protocol::{ChildRequest, ChildResponse, ReadResponse, WriteResponse};

// HostRead — reads from a real host filesystem path

/// Reads bytes from a real filesystem path on the host. Requires
/// [`CapFilesystem`] at construction.
#[derive(Debug, Clone)]
pub struct HostRead {
    host_path: PathBuf,
}

impl HostRead {
    /// Construct with a host path. The `_cap` reference is a
    /// compile-time gate — without a `&CapFilesystem` in scope,
    /// this constructor is uncallable.
    #[must_use]
    pub fn new(host_path: impl Into<PathBuf>, _cap: &CapFilesystem) -> Self {
        Self {
            host_path: host_path.into(),
        }
    }

    /// Borrow the configured host path.
    #[must_use]
    pub fn host_path(&self) -> &PathBuf {
        &self.host_path
    }
}

impl SendPipe for HostRead {
    type In = ChildRequest;
    type Out = ChildResponse;
    type Err = ProximaError;
    fn call(
        &self,
        request: Self::In,
    ) -> impl Future<Output = Result<Self::Out, ProximaError>> + Send {
        // Stub: real implementation would `std::fs::read(&self.host_path)`.
        let response = match request {
            ChildRequest::Read { .. } | ChildRequest::Stat { .. } => {
                ChildResponse::Error { errno: 38 } // ENOSYS
            }
            _ => ChildResponse::Error { errno: 30 }, // EROFS for writes
        };
        async move { Ok(response) }
    }
}

impl WithoutNetwork for HostRead {}
impl WithoutSpawn for HostRead {}
impl WithoutTime for HostRead {}
impl WithoutRandom for HostRead {}

// HostWrite — writes to a real host filesystem path

/// Writes bytes to a real filesystem path on the host. Requires
/// [`CapFilesystem`] at construction.
#[derive(Debug, Clone)]
pub struct HostWrite {
    host_path: PathBuf,
}

impl HostWrite {
    #[must_use]
    pub fn new(host_path: impl Into<PathBuf>, _cap: &CapFilesystem) -> Self {
        Self {
            host_path: host_path.into(),
        }
    }

    #[must_use]
    pub fn host_path(&self) -> &PathBuf {
        &self.host_path
    }
}

impl SendPipe for HostWrite {
    type In = ChildRequest;
    type Out = ChildResponse;
    type Err = ProximaError;
    fn call(
        &self,
        request: Self::In,
    ) -> impl Future<Output = Result<Self::Out, ProximaError>> + Send {
        let response = match request {
            ChildRequest::Write { bytes, .. } => ChildResponse::Write(WriteResponse {
                bytes_written: bytes.len() as u32,
            }),
            _ => ChildResponse::Error { errno: 38 }, // ENOSYS for non-write
        };
        async move { Ok(response) }
    }
}

impl WithoutNetwork for HostWrite {}
impl WithoutSpawn for HostWrite {}
impl WithoutTime for HostWrite {}
impl WithoutRandom for HostWrite {}

// OsEntropy — reads OS entropy (non-deterministic)

/// Reads bytes from OS entropy (`/dev/urandom`-shaped source).
///
/// Markers: NOT `WithoutRandom`, NOT `Deterministic`. The kernel
/// returns different bytes every call.
#[derive(Debug, Clone, Copy, Default)]
pub struct OsEntropy;

impl OsEntropy {
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl SendPipe for OsEntropy {
    type In = ChildRequest;
    type Out = ChildResponse;
    type Err = ProximaError;
    fn call(
        &self,
        request: Self::In,
    ) -> impl Future<Output = Result<Self::Out, ProximaError>> + Send {
        let response = match request {
            ChildRequest::Read { max_bytes, .. } => {
                let bytes: Vec<u8> = alloc::vec![0u8; max_bytes as usize];
                ChildResponse::Read(ReadResponse { bytes, eof: false })
            }
            _ => ChildResponse::Error { errno: 30 },
        };
        async move { Ok(response) }
    }
}

impl NoStd for OsEntropy {}
impl WithoutFilesystem for OsEntropy {}
impl WithoutNetwork for OsEntropy {}
impl WithoutSpawn for OsEntropy {}
impl WithoutTime for OsEntropy {}

// SeededEntropy — deterministic PRNG from a seed

/// Deterministic PRNG from a fixed seed.
#[derive(Debug, Clone, Copy)]
pub struct SeededEntropy {
    seed: u64,
}

impl SeededEntropy {
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self { seed }
    }

    #[must_use]
    pub const fn seed(&self) -> u64 {
        self.seed
    }
}

impl SendPipe for SeededEntropy {
    type In = ChildRequest;
    type Out = ChildResponse;
    type Err = ProximaError;
    fn call(
        &self,
        request: Self::In,
    ) -> impl Future<Output = Result<Self::Out, ProximaError>> + Send {
        let seed = self.seed;
        let response = match request {
            ChildRequest::Read {
                max_bytes, offset, ..
            } => {
                let mut bytes: Vec<u8> = Vec::with_capacity(max_bytes as usize);
                let mut state = seed.wrapping_add(offset);
                for _ in 0..max_bytes {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005_u64)
                        .wrapping_add(1_442_695_040_888_963_407_u64);
                    bytes.push((state >> 33) as u8);
                }
                ChildResponse::Read(ReadResponse { bytes, eof: false })
            }
            _ => ChildResponse::Error { errno: 30 },
        };
        async move { Ok(response) }
    }
}

impl NoStd for SeededEntropy {}
impl Deterministic for SeededEntropy {}
impl Reproducible for SeededEntropy {}
impl IdempotentSideEffectFree for SeededEntropy {}
impl WithoutFilesystem for SeededEntropy {}
impl WithoutNetwork for SeededEntropy {}
impl WithoutSpawn for SeededEntropy {}
impl WithoutTime for SeededEntropy {}

// RealClock — reads the system clock

/// Reads the wall-clock time. NOT `Deterministic`.
#[derive(Debug, Clone, Copy, Default)]
pub struct RealClock;

impl RealClock {
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl SendPipe for RealClock {
    type In = ChildRequest;
    type Out = ChildResponse;
    type Err = ProximaError;
    fn call(
        &self,
        request: Self::In,
    ) -> impl Future<Output = Result<Self::Out, ProximaError>> + Send {
        let response = match request {
            ChildRequest::Read { .. } => ChildResponse::Read(ReadResponse {
                bytes: String::from("0").into_bytes(),
                eof: true,
            }),
            _ => ChildResponse::Error { errno: 30 },
        };
        async move { Ok(response) }
    }
}

impl WithoutFilesystem for RealClock {}
impl WithoutNetwork for RealClock {}
impl WithoutSpawn for RealClock {}
impl WithoutRandom for RealClock {}

// FixedClock — emits a fixed epoch

/// Emits a fixed epoch every call. Deterministic — replay-safe.
#[derive(Debug, Clone, Copy)]
pub struct FixedClock {
    epoch_seconds: u64,
}

impl FixedClock {
    #[must_use]
    pub const fn new(epoch_seconds: u64) -> Self {
        Self { epoch_seconds }
    }

    #[must_use]
    pub const fn epoch_seconds(&self) -> u64 {
        self.epoch_seconds
    }
}

impl SendPipe for FixedClock {
    type In = ChildRequest;
    type Out = ChildResponse;
    type Err = ProximaError;
    fn call(
        &self,
        request: Self::In,
    ) -> impl Future<Output = Result<Self::Out, ProximaError>> + Send {
        let epoch = self.epoch_seconds;
        let response = match request {
            ChildRequest::Read { .. } => {
                let bytes = format_u64(epoch);
                ChildResponse::Read(ReadResponse { bytes, eof: true })
            }
            _ => ChildResponse::Error { errno: 30 },
        };
        async move { Ok(response) }
    }
}

impl NoStd for FixedClock {}
impl AllocFree for FixedClock {}
impl Deterministic for FixedClock {}
impl Reproducible for FixedClock {}
impl IdempotentSideEffectFree for FixedClock {}
impl WithoutFilesystem for FixedClock {}
impl WithoutNetwork for FixedClock {}
impl WithoutSpawn for FixedClock {}
impl WithoutRandom for FixedClock {}

// Helpers

/// Format a u64 as a decimal byte string. Used by FixedClock stub.
fn format_u64(value: u64) -> Vec<u8> {
    if value == 0 {
        return Vec::from(b"0".as_slice());
    }
    let mut buffer = Vec::new();
    let mut remaining = value;
    while remaining > 0 {
        buffer.push(b'0' + (remaining % 10) as u8);
        remaining /= 10;
    }
    buffer.reverse();
    buffer
}
