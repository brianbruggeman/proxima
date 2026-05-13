//! Sans-IO NVMe command-set wire codec for a userspace storage stack.
//!
//! Tier-3: compiles under `#![no_std]` with no allocator. Decode borrows views
//! over caller-owned queue memory; encode writes into caller-owned 64-byte SQE
//! and 16-byte CQE slots. No I/O, no memory ownership — the codec parses and
//! builds queue entries, and the ring state machine is pure index + phase
//! arithmetic. The I/O facade (VFIO/mmap of the controller BAR, hugepage
//! queues, doorbell MMIO) lives one layer up — pure Rust, zero C linked,
//! mirroring how `proxima-net-dpdk` does pure-Rust dpdk FFI over
//! `proxima-inet-codec`. SPDK is a parity reference, not a dependency.
//!
//! NVMe queue entries live in host DRAM, so every multi-byte field is
//! little-endian — the opposite of the big-endian network codec next door.

pub mod command;
pub mod completion;
pub mod error;
pub mod queue;
mod raw;

pub use command::{CommandBuilder, DataTransfer, SubmissionEntry};
pub use completion::{CompletionEntry, StatusField, write_completion};
pub use error::DecodeError;
pub use queue::{CompletionRing, SubmissionRing};
