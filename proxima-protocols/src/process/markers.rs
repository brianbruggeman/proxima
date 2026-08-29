//! Re-export of the marker traits from `proxima-core::markers`.
//!
//! The 12 marker traits were promoted to `proxima-core` in Phase C.1
//! of the cliff-extension plan so the whole workspace can use them
//! without depending on `proxima-process-protocol`. This shim
//! preserves the `proxima_process_protocol::markers::*` import path
//! for backward compatibility with existing callers
//! (proxima-process, proxima-vm).
//!
//! See `proxima.decision.libc_shim_vm_parity` in
//! `proxima/ai_docs/invariants.jsonl` for the rationale behind the
//! protocol-extract crate.

pub use proxima_core::markers::*;
