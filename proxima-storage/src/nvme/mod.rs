//! Prime-native NVMe queue-pair engine.
//!
//! Drives the sans-IO [`proxima_protocols::nvme`] (SQE/CQE codec + phase/doorbell
//! ring FSM) over a pluggable [`QueueBackend`] and exposes "submit a command,
//! await its completion" as a proxima [`Pipe`](proxima_primitives::pipe::Pipe) — the
//! per-core, `!Send` root form that matches NVMe's core-affine queue pairs —
//! and a [`SendPipe`](proxima_primitives::pipe::SendPipe) for when the handle crosses
//! cores.
//!
//! Tier-3: the engine is `#![no_std]` + no-alloc. It only touches the codec, the
//! ring cursors (held in atomics so the pair is `Sync`), and the backend trait.
//! The std half — device acquisition via VFIO/uio mmap of the controller BAR,
//! hugepage queue allocation, real doorbell MMIO — is a separate backend impl
//! behind the `std` boundary, exactly as `proxima-net-dpdk` sits over the
//! `no_std` `proxima-inet-codec`. Zero C, mirroring net-dpdk's pure-Rust FFI.
pub mod backend;
pub mod engine;
pub mod error;

#[cfg(feature = "nvme-uio")]
pub mod uio;

pub use backend::QueueBackend;
pub use engine::{Completion, QueuePair};
pub use error::NvmeError;
