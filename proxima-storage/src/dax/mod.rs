//! std storage facade for [`crate::pmem`] — two backends of one crash-consistent
//! cell interface (`create` / `commit` / `read` / `recover`, a read after a crash
//! always sees the complete old or new value, never torn):
//!
//! - [`FileCell`] (the portable store-backed FLOOR, any OS incl. macOS): a
//!   conventional durable store — fsync + atomic rename. No mmap, no pmem
//!   hardware, unbounded value size ("big stuff"); slower (rewrites the value per
//!   commit). This is the "always works, just slower" tier.
//! - [`PmemCowStore`] (the pmem-native FAST tier, Linux): composes
//!   [`crate::pmem::CowRoot`] over an `mmap`'d region (a `/dev/dax` device, or a
//!   regular file via `msync`). Byte-addressable in-place atomic-root-swap; the
//!   8-byte root relies on SNIA/Intel ADR power-fail atomicity, so it needs the
//!   mapped path. `mmap`/`munmap`/`msync` go through `rustix`'s `linux_raw`
//!   backend (no libc) — pure Rust, zero C, like the leaf's `core::arch` flush.
//!
//! Same guarantee, two mechanisms (atomic rename vs in-place root swap), perf
//! tiers. The store-backed floor means "pmem" is never unavailable — it degrades
//! to a store rather than requiring hardware.

pub mod cell;
pub mod config;
pub mod error;

#[cfg(target_os = "linux")]
pub mod region;
#[cfg(target_os = "linux")]
pub mod store;

pub use cell::FileCell;
pub use config::DaxConfig;
pub use config::PersistMode;
pub use error::DaxError;

#[cfg(target_os = "linux")]
pub use region::MappedRegion;
#[cfg(target_os = "linux")]
pub use store::PmemCowStore;
