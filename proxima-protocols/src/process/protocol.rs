//! Typed envelope for dispatch: [`ChildRequest`] /
//! [`ChildResponse`].
//!
//! `Open` carries the path key the routing operator dispatches on;
//! every other variant is fd-keyed, carrying the `handle` a prior
//! `Open` returned. Paths use Linux-style canonical namespaces
//! (`/proc/sys/kernel/*`, `/proc/self/*`, `/dev/*`, `/etc/*`) even on
//! macOS — the shim's libc hook is responsible for translating
//! `uname(2)` / `gethostname(2)` / etc. into the canonical-path form
//! before calling `Open`, then addressing the resulting handle for
//! the actual `Read`/`Write`/`Stat`/`Close`.
//!
//! # Parity contract — load-bearing
//!
//! This protocol is the **shared dispatch contract** between two
//! containment mechanisms:
//!
//! 1. The in-process **libc-interpose shim** (this crate's
//!    [`super::interpose`](../interpose/index.html) C source,
//!    compiled by `build.rs` into a `.dylib`/`.so`). Scoped to
//!    owned children per `proxima.failure.hardened_dyld_interpose`.
//! 2. **`proxima-vm`** — the OS-level VM boundary that handles
//!    hardened-runtime / opaque third-party binaries. Per
//!    `proxima.decision.contained_discovery_boundary`.
//!
//! The two mechanisms MUST present the same dispatch surface so a
//! single `DispatchChoice` config / typed chain works against
//! either. If the libc-shim grows a new `ChildRequest` variant,
//! `proxima-vm` must also handle it (or vice versa); adding one
//! without the other is a parity regression and breaks the
//! "configure once, route through either mechanism" principle.
//!
//! Parity tests live at the workspace level (TBD — currently
//! enforced by convention + the discipline log's C8c row).
//!
//! # Tier
//!
//! This module is `no_std + alloc`. Lives in the standalone
//! `proxima-process-protocol` crate (extracted 2026-05-23) so
//! both consumers — proxima-process's libc-shim AND proxima-vm —
//! depend on the protocol-only surface without inheriting the
//! larger proxima-process dep tree (nix, bon, conflaguration,
//! etc.). proxima-process re-exports
//! `proxima_process::protocol::*` for backward compatibility.
//!
//! # Wire format (load-bearing — parity-locked)
//!
//! Both consumers (libc-shim's C side, `proxima-vm`'s host-side
//! dispatcher) MUST speak the same bytes on the wire. The parent
//! side is already postcard-based via `super::ipc` and
//! `super::framing`; per the
//! `proxima.decision.libc_shim_vm_parity` invariant + RISC
//! principle 1, both consumers reuse the parent's format:
//!
//! ```text
//! ┌──────────────┬──────────────────────────────────────────┐
//! │ length (u32) │ postcard-serialised ChildRequest or      │
//! │   big-endian │ ChildResponse                            │
//! └──────────────┴──────────────────────────────────────────┘
//! ```
//!
//! Postcard's binary spec
//! (<https://postcard.jamesmunns.com/wire-format>) is the
//! authoritative encoding. The fields below are the minimum the
//! C-side encoder/decoder must handle to satisfy the smoke set
//! (`gethostname`, `uname`, `getpid`, basic `read`):
//!
//! - `varint(u32)` — LEB128, max 5 bytes
//! - `varint(u64)` — LEB128, max 10 bytes
//! - `varint(i32)` — zigzag(LEB128), max 5 bytes
//! - `String` — `varint(len)` then raw UTF-8 bytes
//! - `Vec<u8>` — `varint(len)` then raw bytes
//! - `bool` — single byte (0 or 1)
//! - enum discriminant — `varint(u32)` indexed in `derive` order
//!
//! ## Variant discriminant index (LOCKED — do not reorder)
//!
//! Reordering breaks both consumers' decoders. The discriminant is
//! the `derive`-source-order index, starting from 0.
//!
//! - `ChildRequest::Read` = 0
//! - `ChildRequest::Write` = 1
//! - `ChildRequest::Open` = 2
//! - `ChildRequest::Close` = 3
//! - `ChildRequest::Stat` = 4
//!
//! - `ChildResponse::Read(ReadResponse)` = 0
//! - `ChildResponse::Write(WriteResponse)` = 1
//! - `ChildResponse::Open { handle }` = 2
//! - `ChildResponse::Close` = 3
//! - `ChildResponse::Stat { … }` = 4
//! - `ChildResponse::Error { errno }` = 5
//!
//! ## Parity invariant
//!
//! Adding a new `ChildRequest` / `ChildResponse` variant requires
//! parallel updates to BOTH consumers' decoders (or pre-staging
//! the decoder side first). See
//! `proxima.decision.libc_shim_vm_parity`. Variants land at the
//! END of the enum (appending preserves discriminant indices); do
//! not insert mid-enum.

use alloc::string::String;
use alloc::vec::Vec;

use serde::{Deserialize, Serialize};

use proxima_core::markers::{AllocFree, Deterministic, IsPure, NoStd};

/// One IPC call from the child, as routed through the IPC fd.
///
/// `Open` is the sole path-keyed variant — it is the only call that
/// names a resource that does not exist yet. Every other variant is
/// **fd-keyed**, matching WASI and Linux convention: a prior `Open`
/// hands back a `handle`, and `Read` / `Write` / `Close` / `Stat`
/// address the resource by that handle, not by re-sending the path.
/// The routing operator dispatches `Open` by path prefix; handle
/// resolution for the fd-keyed verbs is the dispatcher's job, not
/// the router's (per `tools/proxima-vm/ROADMAP.md` P0).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ChildRequest {
    /// Read N bytes from a resource. The shim translates `uname` /
    /// `gethostname` / `read(fd)` etc. into this variant.
    Read {
        /// Handle returned by a prior `Open` of the resource.
        handle: i32,
        /// Maximum bytes to read (caller's buffer size).
        max_bytes: u32,
        /// Byte offset for sequential reads (cursor position).
        offset: u64,
    },
    /// Write bytes to a resource.
    Write {
        /// Handle returned by a prior `Open` of the resource.
        handle: i32,
        /// Bytes the child is writing.
        bytes: Vec<u8>,
    },
    /// Open a resource for subsequent read/write. Allocates a handle
    /// the dispatcher tracks internally.
    Open {
        /// Path being opened.
        path: String,
        /// Open flags (mirrors `open(2)` flags).
        flags: u32,
    },
    /// Release a handle. The dispatcher reclaims state for it.
    Close {
        /// Handle being released.
        handle: i32,
    },
    /// Retrieve metadata for a resource (`fstat(2)`-shaped).
    Stat {
        /// Handle returned by a prior `Open` of the resource.
        handle: i32,
    },
}

impl ChildRequest {
    /// The path being addressed by this request. Only `Open` names a
    /// path — every other variant is fd-keyed. Used by the routing
    /// operator to dispatch `Open` calls by path prefix.
    #[must_use]
    pub fn path(&self) -> Option<&str> {
        match self {
            Self::Open { path, .. } => Some(path.as_str()),
            Self::Read { .. } | Self::Write { .. } | Self::Close { .. } | Self::Stat { .. } => None,
        }
    }

    /// The handle being addressed by this request. Every variant
    /// except `Open` is fd-keyed and carries one.
    #[must_use]
    pub fn handle(&self) -> Option<i32> {
        match self {
            Self::Read { handle, .. }
            | Self::Write { handle, .. }
            | Self::Close { handle }
            | Self::Stat { handle } => Some(*handle),
            Self::Open { .. } => None,
        }
    }
}

/// One dispatch result, returned to the shim across the IPC
/// fd and decoded into the original libc-call's return value.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ChildResponse {
    /// Bytes read from a resource.
    Read(ReadResponse),
    /// Acknowledgement of a write, with byte count consumed.
    Write(WriteResponse),
    /// Handle allocated for an open call.
    Open {
        /// Allocated handle number (mirrors libc fd convention).
        handle: i32,
    },
    /// Acknowledgement of a close.
    Close,
    /// Stat metadata.
    Stat {
        /// File size in bytes.
        size: u64,
        /// Unix mode bits.
        mode: u32,
        /// Whether the entry is a directory.
        is_directory: bool,
    },
    /// Error response. Carries the errno the shim should surface to
    /// the child's libc call.
    Error {
        /// errno value (e.g. `libc::EROFS`, `libc::ENOENT`).
        errno: i32,
    },
}

/// Read response payload. Separate struct so [`ChildResponse::Read`]
/// can carry a Body-shaped value without inflating the enum size for
/// other variants.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ReadResponse {
    /// Bytes returned to the child's read buffer.
    pub bytes: Vec<u8>,
    /// `true` if this read consumed all available data and the next
    /// read would return EOF.
    pub eof: bool,
}

/// Write response payload — number of bytes the dispatcher accepted.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WriteResponse {
    /// Bytes accepted (may be less than requested for partial writes).
    pub bytes_written: u32,
}

impl NoStd for ChildRequest {}
impl NoStd for ChildResponse {}
impl NoStd for ReadResponse {}
impl NoStd for WriteResponse {}

// ChildResponse::Error is a pure-data variant — emitting an Error
// response is deterministic and effect-free given the errno input.
// The marker impls apply to the TYPE, not specific variants; the
// dispatcher inspects the variant at runtime to decide flow.
impl IsPure for ChildResponse {}
impl Deterministic for ChildResponse {}

// The types themselves can be constructed without per-call heap
// allocation when their Vec/String fields are empty. The grounds
// that produce them (Canned/Empty/Deny in grounds.rs) may or may
// not allocate per call depending on the variant — AllocFree
// applies to specific grounds, not blanket to the protocol types.
impl AllocFree for ChildResponse {}
