//! Ground dispatch units — leaves of the dispatch tree.
//!
//! Grounds are concrete types that handle [`ChildRequest`] →
//! [`ChildResponse`] without delegating to inner dispatch units.
//! Each ground impls the [`super::markers`] it qualifies for; the
//! compiler propagates the markers through operators.
//!
//! This module ships the trivial grounds (`Canned`, `Empty`, `Deny`)
//! that don't require capability tokens or I/O. Capability-gated
//! grounds (`HostRead`, `HostWrite`, etc.) land in a later step
//! against [`super::capabilities`].

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;
use bytes::Bytes;

use proxima_primitives::pipe::ProximaError;
use proxima_primitives::pipe::SendPipe;

use super::markers::{
    AllocFree, Commutative, Deterministic, IdempotentSideEffectFree, IsPure, NoStd, Reproducible,
    WithoutFilesystem, WithoutNetwork, WithoutRandom, WithoutSpawn, WithoutTime,
};
use super::protocol::{ChildRequest, ChildResponse, ReadResponse, WriteResponse};

/// Canned bytes source: answers `Read` requests with a fixed byte
/// payload, writes succeed but bytes are dropped (write-through), and
/// other verbs are rejected with `EROFS`-like semantics.
///
/// Cheap to clone — the underlying [`Bytes`] is reference-counted,
/// so cloning is an atomic increment.
#[derive(Debug, Clone)]
pub struct Canned {
    bytes: Bytes,
}

impl Canned {
    /// Construct from owned bytes. The payload becomes the canonical
    /// `Read` response.
    #[must_use]
    pub fn new(bytes: impl Into<Bytes>) -> Self {
        Self {
            bytes: bytes.into(),
        }
    }
}

impl SendPipe for Canned {
    type In = ChildRequest;
    type Out = ChildResponse;
    type Err = ProximaError;
    fn call(
        &self,
        request: Self::In,
    ) -> impl core::future::Future<Output = Result<Self::Out, ProximaError>> + Send {
        let response = match request {
            ChildRequest::Read {
                max_bytes, offset, ..
            } => {
                let total_len = self.bytes.len() as u64;
                if offset >= total_len {
                    ChildResponse::Read(ReadResponse {
                        bytes: Vec::new(),
                        eof: true,
                    })
                } else {
                    let start = offset as usize;
                    let remaining = self.bytes.len() - start;
                    let chunk_len = core::cmp::min(remaining, max_bytes as usize);
                    let bytes = self.bytes.slice(start..start + chunk_len).to_vec();
                    let eof = (offset + chunk_len as u64) >= total_len;
                    ChildResponse::Read(ReadResponse { bytes, eof })
                }
            }
            ChildRequest::Write { bytes, .. } => ChildResponse::Write(WriteResponse {
                bytes_written: bytes.len() as u32,
            }),
            ChildRequest::Stat { .. } => ChildResponse::Stat {
                size: self.bytes.len() as u64,
                mode: 0o444,
                is_directory: false,
            },
            ChildRequest::Open { .. } => ChildResponse::Open { handle: 0 },
            ChildRequest::Close { .. } => ChildResponse::Close,
        };
        async move { Ok(response) }
    }
}

impl NoStd for Canned {}
// AllocFree is conditional: Read with non-zero output allocates a
// Vec for the response bytes. Mark Canned as AllocFree only for the
// no-output paths (zero-length response, write-ack, etc.); strictly
// we cannot claim it. Leave AllocFree unimpl'd until we have a
// zero-copy Body variant that holds a Bytes slice directly.
impl IsPure for Canned {}
impl Deterministic for Canned {}
impl Reproducible for Canned {}
impl IdempotentSideEffectFree for Canned {}
impl WithoutFilesystem for Canned {}
impl WithoutNetwork for Canned {}
impl WithoutSpawn for Canned {}
impl WithoutTime for Canned {}
impl WithoutRandom for Canned {}

/// Empty source — answers `Read` with EOF immediately, writes
/// succeed but are dropped, stat reports zero size. Useful for
/// `/dev/null`-style endpoints where you want a present-but-empty
/// resource.
#[derive(Debug, Clone, Copy, Default)]
pub struct Empty;

impl Empty {
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl SendPipe for Empty {
    type In = ChildRequest;
    type Out = ChildResponse;
    type Err = ProximaError;
    fn call(
        &self,
        request: Self::In,
    ) -> impl core::future::Future<Output = Result<Self::Out, ProximaError>> + Send {
        let response = match request {
            ChildRequest::Read { .. } => ChildResponse::Read(ReadResponse {
                bytes: Vec::new(),
                eof: true,
            }),
            ChildRequest::Write { bytes, .. } => ChildResponse::Write(WriteResponse {
                bytes_written: bytes.len() as u32,
            }),
            ChildRequest::Stat { .. } => ChildResponse::Stat {
                size: 0,
                mode: 0o666,
                is_directory: false,
            },
            ChildRequest::Open { .. } => ChildResponse::Open { handle: 0 },
            ChildRequest::Close { .. } => ChildResponse::Close,
        };
        async move { Ok(response) }
    }
}

impl NoStd for Empty {}
impl AllocFree for Empty {}
impl IsPure for Empty {}
impl Deterministic for Empty {}
impl Reproducible for Empty {}
impl IdempotentSideEffectFree for Empty {}
impl Commutative for Empty {}
impl WithoutFilesystem for Empty {}
impl WithoutNetwork for Empty {}
impl WithoutSpawn for Empty {}
impl WithoutTime for Empty {}
impl WithoutRandom for Empty {}

/// Deny source — answers every request with the configured errno.
/// Useful for "this resource exists but you can't touch it" or
/// blanket-denying writes to a read-only mount.
#[derive(Debug, Clone, Copy)]
pub struct Deny {
    errno: i32,
}

impl Deny {
    /// Construct with a specific errno (e.g. `libc::EROFS`,
    /// `libc::EACCES`, `libc::ENOENT`).
    #[must_use]
    pub const fn new(errno: i32) -> Self {
        Self { errno }
    }
}

impl SendPipe for Deny {
    type In = ChildRequest;
    type Out = ChildResponse;
    type Err = ProximaError;
    fn call(
        &self,
        _request: Self::In,
    ) -> impl core::future::Future<Output = Result<Self::Out, ProximaError>> + Send {
        let errno = self.errno;
        async move { Ok(ChildResponse::Error { errno }) }
    }
}

impl NoStd for Deny {}
impl AllocFree for Deny {}
impl IsPure for Deny {}
impl Deterministic for Deny {}
impl Reproducible for Deny {}
impl IdempotentSideEffectFree for Deny {}
impl Commutative for Deny {}
impl WithoutFilesystem for Deny {}
impl WithoutNetwork for Deny {}
impl WithoutSpawn for Deny {}
impl WithoutTime for Deny {}
impl WithoutRandom for Deny {}

/// Construct a Canned source from a byte slice. The slice is copied
/// into a refcounted [`Bytes`] at construction; subsequent dispatches
/// borrow from that one allocation.
#[must_use]
pub fn canned(bytes: impl Into<Bytes>) -> Canned {
    Canned::new(bytes)
}

/// Construct an Empty source.
#[must_use]
pub const fn empty() -> Empty {
    Empty::new()
}

/// Construct a Deny source returning the given errno.
#[must_use]
pub const fn deny(errno: i32) -> Deny {
    Deny::new(errno)
}

/// Construct a Deny source returning `EROFS` (read-only filesystem).
#[must_use]
pub const fn deny_writes() -> Deny {
    // libc::EROFS = 30 on macOS and Linux. Hardcoded to keep the
    // constructor const and platform-independent for now.
    Deny::new(30)
}

// `vec!` macro is used inside dispatch implementations through
// alloc::vec — re-exported here for clippy's benefit on the
// `use alloc::vec` line if a path stays unused.
#[allow(unused_imports)]
use vec as _vec_marker;
