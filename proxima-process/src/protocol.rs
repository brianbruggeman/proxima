//! Re-export of the dispatch protocol types from
//! `proxima-process-protocol`.
//!
//! `ChildRequest` / `ChildResponse` / `ReadResponse` /
//! `WriteResponse` + the wire-format spec + the parity contract
//! live in [`proxima_protocols::process::protocol`]. This re-export
//! preserves the `proxima_process::protocol::*` import path for
//! backward compatibility while letting `proxima-vm` and other
//! downstream sandbox consumers depend on the protocol-only
//! crate without inheriting proxima-process's larger dep tree.
//!
//! See `proxima.decision.libc_shim_vm_parity` in
//! `proxima/ai_docs/invariants.jsonl` for the rationale.

pub use proxima_protocols::process::protocol::*;
