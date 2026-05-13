//! Re-export of the marker traits from `proxima-process-protocol`.
//!
//! These were extracted into a sibling no_std crate so `proxima-vm`
//! (and other downstream sandbox consumers) can depend on the
//! shared dispatch contract without inheriting proxima-process's
//! larger dep tree. The original module-level documentation +
//! trait definitions now live at
//! [`proxima_protocols::process::markers`]; this re-export
//! preserves the `proxima_process::markers::*` import path for
//! backward compatibility.
//!
//! See `proxima.decision.libc_shim_vm_parity` in
//! `proxima/ai_docs/invariants.jsonl` for the rationale.

pub use proxima_protocols::process::markers::*;
