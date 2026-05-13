//! Parity-side handler shape for the
//! `proxima_protocols::process::{ChildRequest, ChildResponse}`
//! dispatch contract.
//!
//! Per `proxima.decision.libc_shim_vm_parity` in
//! `proxima/ai_docs/invariants.jsonl`, proxima-vm and the
//! proxima-process libc-shim must consume the same protocol
//! variants. This module defines the trait shape proxima-vm
//! exposes for that consumption.
//!
//! # Current state — scaffolding only
//!
//! [`ScratchVm`](super::ScratchVm) doesn't yet have guests that
//! issue `ChildRequest`s — it's a bare-metal "emit bytes and
//! halt" guest with no OS layer making syscalls. The trait shape
//! lands now so that:
//!
//! 1. The parity-contract surface is **present in code**, not
//!    just in docs. Future VM work (mirror VM, full OS guest)
//!    impls this trait, and downstream sandbox plumbing wires
//!    against the trait — no per-VM-implementation churn.
//! 2. The wire-format parity test
//!    ([`wire_format_round_trips_for_parity`] below) proves the
//!    proxima-process-protocol crate exposes the same bytes
//!    proxima-vm will consume.
//!
//! # When the trait grows real implementations
//!
//! A real `MirrorVm` (per `proxima.decision.mirror_is_pipe`) will
//! impl `VmDispatchHandler` and route guest syscalls through it.
//! The libc-shim's C side already emits the same wire bytes;
//! same dispatch-chain config drives both. See the libc-shim
//! component's C8c discipline-log row at
//! `pty-tester/docs/proxima-pty/discipline.md`.

extern crate alloc;

use core::future::Future;

use proxima_core::ProximaError;
use proxima_protocols::process::{ChildRequest, ChildResponse};

/// VM-side parity handler — the proxima-vm equivalent of the
/// dispatch chain the libc-shim talks to. Implementations
/// receive `ChildRequest`s from inside the VM's guest and
/// return `ChildResponse`s that the guest's syscall returns.
///
/// The shape mirrors `proxima_primitives::pipe::SendPipe`
/// with `In = ChildRequest, Out = ChildResponse` so existing
/// dispatch chains (grounds, AndThen, Match, etc.) compose as-is
/// when proxima-vm grows protocol-emitting guests.
pub trait VmDispatchHandler {
    /// Handle one `ChildRequest`, producing one `ChildResponse`.
    /// `async fn` shape matches the proxima-process dispatch
    /// chain so a single config can drive either consumer.
    fn handle(
        &self,
        request: ChildRequest,
    ) -> impl Future<Output = Result<ChildResponse, ProximaError>> + Send;

    /// Identifier for diagnostics.
    fn name(&self) -> &'static str;
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::field_reassign_with_default,
        clippy::type_complexity,
        clippy::useless_vec,
        clippy::needless_range_loop,
        clippy::default_constructed_unit_structs
    )]

    use super::*;
    use proxima_protocols::process::ReadResponse;

    /// A stub handler that returns canned bytes for every Read
    /// — useful for testing the trait shape without booting a
    /// real VM. NOT the production path.
    struct CannedReadHandler {
        bytes: alloc::vec::Vec<u8>,
    }

    impl VmDispatchHandler for CannedReadHandler {
        fn handle(
            &self,
            _request: ChildRequest,
        ) -> impl Future<Output = Result<ChildResponse, ProximaError>> + Send {
            let bytes = self.bytes.clone();
            async move { Ok(ChildResponse::Read(ReadResponse { bytes, eof: true })) }
        }
        fn name(&self) -> &'static str {
            "canned-read-test"
        }
    }

    #[test]
    fn handler_trait_compiles_with_canned_implementation() {
        let handler = CannedReadHandler {
            bytes: alloc::vec::Vec::from(b"vm-side-canned" as &[u8]),
        };
        let request = ChildRequest::Read {
            path: alloc::string::String::from("/proc/sys/kernel/hostname"),
            max_bytes: 256,
            offset: 0,
        };
        let response = futures::executor::block_on(handler.handle(request)).expect("handler runs");
        match response {
            ChildResponse::Read(read) => {
                assert_eq!(read.bytes, b"vm-side-canned");
                assert!(read.eof);
            }
            _ => panic!("unexpected variant"),
        }
    }

    #[test]
    fn handler_name_is_static_str() {
        let handler = CannedReadHandler {
            bytes: alloc::vec::Vec::new(),
        };
        assert_eq!(handler.name(), "canned-read-test");
    }
}
