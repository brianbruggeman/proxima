//! Re-export of the compile-time effect markers.
//!
//! The trait definitions and their module-level documentation live in
//! `proxima_core::markers`; the shared dispatch contract names them at
//! [`proxima_protocols::process`] so `proxima-vm` and other downstream
//! sandbox consumers reach them without inheriting proxima-process's
//! larger dep tree. This re-export preserves the
//! `proxima_process::markers::*` import path.
//!
//! See `proxima.decision.libc_shim_vm_parity` in
//! `proxima/ai_docs/invariants.jsonl` for the rationale.

pub use proxima_protocols::process::{
    AllocFree, Commutative, Deterministic, DropSafe, IdempotentSideEffectFree, IsPure, NoStd,
    Reproducible, WithoutFilesystem, WithoutNetwork, WithoutRandom, WithoutSpawn, WithoutTime,
};
