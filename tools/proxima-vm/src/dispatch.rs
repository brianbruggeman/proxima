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

#[cfg(feature = "std")]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use bytes::Bytes;
#[cfg(feature = "std")]
use core::ffi::c_void;
use proxima_core::ProximaError;
#[cfg(feature = "std")]
use proxima_primitives::pipe::SendPipe;
#[cfg(feature = "std")]
use proxima_process::framing::{FrameDecoder, FrameEncoder};
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

/// Concrete adapter type behind the FFI boundary in
/// [`proxima_vm_dispatch_hypercall`]. `extern "C"` functions cannot be
/// generic, so this one Rust type exists purely to give the C ABI's
/// `void *dispatcher` argument something concrete to cast back to on the
/// Rust side. It answers every request with one configured
/// [`ChildResponse`] and records call order.
///
/// This is `pub(crate)`, not library API: only this module's own
/// `#[cfg(test)]` tests and [`run_dispatch_loop`] construct one — the real
/// M1 driver, which passes a live `&FfiRecordingDispatcher` across the FFI
/// boundary into the C exit loop
/// (`src/backend_macos.c`'s / `src/backend_linux.c`'s
/// `proxima_vm_run_dispatch_loop`), which calls back into
/// [`proxima_vm_dispatch_hypercall`] per capability hypercall. NOT a
/// general-purpose production dispatcher — it answers every request
/// variant with the same configured response rather than routing by
/// variant, which is sufficient to prove request-side and response-side
/// bidirectionality (`tools/proxima-vm/ROADMAP.md`'s M1 exit criterion) but
/// not a real syscall router.
#[cfg(feature = "std")]
#[derive(Debug)]
pub(crate) struct FfiRecordingDispatcher {
    configured_response: ChildResponse,
    // held only across the synchronous body of `call`, never across an
    // `.await` — same discipline as `src/upstreams/record.rs`'s `rng` field.
    recorded_requests: std::sync::Mutex<Vec<ChildRequest>>,
}

#[cfg(feature = "std")]
impl FfiRecordingDispatcher {
    /// Build a dispatcher that answers every request with `configured_response`.
    /// Constructed by this module's own `#[cfg(test)]` tests directly, and
    /// by [`run_dispatch_loop`] as the one dispatcher a real VM-exit run
    /// drives.
    pub(crate) fn new(configured_response: ChildResponse) -> Self {
        Self {
            configured_response,
            recorded_requests: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// The requests this dispatcher has been called with, in call order.
    pub(crate) fn requests(&self) -> Vec<ChildRequest> {
        self.recorded_requests
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }
}

#[cfg(feature = "std")]
impl SendPipe for FfiRecordingDispatcher {
    type In = ChildRequest;
    type Out = ChildResponse;
    type Err = ProximaError;

    fn call(
        &self,
        request: ChildRequest,
    ) -> impl Future<Output = Result<ChildResponse, ProximaError>> + Send {
        self.recorded_requests
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(request);
        let response = self.configured_response.clone();
        async move { Ok(response) }
    }
}

/// Decode one hypercall's payload through
/// [`proxima_process::framing::FrameDecoder`], drive the resulting
/// [`ChildRequest`] through `dispatcher`'s [`SendPipe::call`], and encode the
/// resulting [`ChildResponse`] through
/// [`proxima_process::framing::FrameEncoder`] into `response_out`. Shared by
/// [`proxima_vm_dispatch_hypercall`] (the `extern "C"` trampoline entry,
/// necessarily monomorphized to [`FfiRecordingDispatcher`] because
/// `extern "C"` cannot be generic; this crate's own [`run_dispatch_loop`]
/// calls it that way too, through the C exit loop).
///
/// The three stages are called in sequence rather than composed as one
/// `AndThen(FrameDecoder, dispatcher, FrameEncoder)` value (the shape
/// `framing.rs`'s own module doc names) because `AndThen` stores every
/// stage by value, and `dispatcher` here is a borrow reconstructed from the
/// FFI boundary's `*const c_void` — never an owned `D` this function could
/// move in. `SendPipe` itself requires `'static`, so no blanket `&P:
/// SendPipe` bridge could paper over that borrow either. Calling
/// `FrameDecoder`/`FrameEncoder` directly still reuses the exact codec
/// logic `AndThen` would have run; only the wrapper value is absent.
///
/// This is not hand-rolled the way `proxima_process::ipc::run_dispatch_loop`
/// (`proxima-process/src/ipc.rs:111`, a different crate and a different
/// function from this module's own [`run_dispatch_loop`] despite the name
/// collision) would require: that helper loops a length-prefixed
/// `[u32_be length][postcard payload]` byte STREAM to EOF over
/// `Read`/`Write`. A hypercall is one-shot — its payload length is already
/// known from the vCPU's `length` register rather than an in-band length
/// prefix, and the source is a raw guest-memory pointer/length view
/// recovered from a single VM exit, not an `io::Read` a loop can drain to
/// EOF. `FrameDecoder`/`FrameEncoder` operate on the prefix-stripped
/// payload directly, which is exactly this situation.
///
/// Returns the encoded response length on success, or a negative sentinel:
/// -1 payload pointer/length out of range, -2 `FrameDecoder` reported a
/// decode failure (unreachable today — malformed payloads decode to a
/// fallback `Read` request per `FrameDecoder`'s own contract, so this arm
/// exists for forward compatibility if that contract ever changes), -3 the
/// dispatcher call failed, -4 `FrameEncoder` reported an encode failure
/// (unreachable for well-formed `ChildResponse`s, same as `FrameEncoder`'s
/// own doc), -5 the encoded response exceeds `response_out`'s length.
#[cfg(feature = "std")]
fn dispatch_hypercall_bytes<D>(
    dispatcher: &D,
    guest_memory: &[u8],
    verb: u16,
    pointer: u64,
    length: u64,
    response_out: &mut [u8],
) -> i64
where
    D: SendPipe<In = ChildRequest, Out = ChildResponse, Err = ProximaError>,
{
    let view = match crate::abi::decode_hypercall(verb, pointer, length, guest_memory) {
        Ok(view) => view,
        Err(_) => return -1,
    };
    let payload = Bytes::copy_from_slice(view.payload());
    let request = match futures::executor::block_on(SendPipe::call(&FrameDecoder::new(), payload)) {
        Ok(request) => request,
        Err(_) => return -2,
    };
    let response = match futures::executor::block_on(SendPipe::call(dispatcher, request)) {
        Ok(response) => response,
        Err(_) => return -3,
    };
    let encoded = match futures::executor::block_on(SendPipe::call(&FrameEncoder::new(), response))
    {
        Ok(bytes) => bytes,
        Err(_) => return -4,
    };
    if encoded.len() > response_out.len() {
        return -5;
    }
    response_out[..encoded.len()].copy_from_slice(&encoded);
    encoded.len() as i64
}

/// The `extern "C"` trampoline entry the C-side hypercall exit loop
/// (`src/dispatch_trampoline.h`) calls back into once a real hypercall
/// exit recovers `verb`/`pointer`/`length` from the vCPU
/// (`src/backend_macos.c`'s and `src/backend_linux.c`'s
/// `proxima_vm_run_dispatch_loop`, which [`run_dispatch_loop`] drives).
/// `#[unsafe(no_mangle)]` is load-bearing: without it the C loop's
/// `proxima_vm_dispatch_hypercall(...)` call sees no matching symbol at
/// link time, since a bare `extern "C" fn` without it still gets a mangled
/// Rust symbol name. `dispatcher` is typed `*const c_void` to match the C
/// header's `const void *dispatcher` exactly; the cast to
/// [`FfiRecordingDispatcher`] happens on the Rust side only, since
/// `extern "C"` functions cannot be generic over the caller's dispatcher
/// type.
///
/// # Safety
///
/// `dispatcher` must be a valid, live pointer to an
/// [`FfiRecordingDispatcher`] for the duration of the call. `guest_memory`
/// must be valid for `guest_memory_length` bytes. `response_out` must be
/// valid for `response_capacity` bytes and not alias `guest_memory`.
#[cfg(feature = "std")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn proxima_vm_dispatch_hypercall(
    dispatcher: *const c_void,
    guest_memory: *const u8,
    guest_memory_length: usize,
    verb: u16,
    pointer: u64,
    length: u64,
    response_out: *mut u8,
    response_capacity: usize,
) -> i64 {
    // SAFETY: forwarded verbatim from this function's own safety contract.
    let dispatcher = unsafe { &*dispatcher.cast::<FfiRecordingDispatcher>() };
    // SAFETY: forwarded verbatim from this function's own safety contract.
    let guest_memory = unsafe { core::slice::from_raw_parts(guest_memory, guest_memory_length) };
    // SAFETY: forwarded verbatim from this function's own safety contract.
    let response_out = unsafe { core::slice::from_raw_parts_mut(response_out, response_capacity) };

    dispatch_hypercall_bytes(
        dispatcher,
        guest_memory,
        verb,
        pointer,
        length,
        response_out,
    )
}

/// The M1 production path: loads `segments` (already validated by
/// [`crate::elf::parse_elf`]) into a real hypervisor guest address space at
/// address 0, sets `PC = entry`, and drives every `hvc #0` / `out dx, al`
/// trap through `dispatcher` via a real VM exit
/// (`src/backend_macos.c`'s / `src/backend_linux.c`'s
/// `proxima_vm_run_dispatch_loop`, calling back into
/// [`proxima_vm_dispatch_hypercall`] per capability hypercall) — as opposed
/// to [`dispatch_hypercall_direct_for_tests`], which never boots a guest at
/// all.
///
/// `configured_response` is the one [`ChildResponse`] every capability
/// hypercall this run receives is answered with (the same
/// canned-per-instance shape [`FfiRecordingDispatcher`] already carries);
/// two runs with two different `configured_response` values, driven against
/// the same guest, are what prove the host's response — not a value the
/// guest compiled in — decides the bytes the guest emits.
///
/// Returns every [`ChildRequest`] the guest issued, in call order, and
/// every byte the guest emitted via its dedicated emit verb — the M1 exit
/// proof's two observables (`tools/proxima-vm/ROADMAP.md`'s M1 section: "the
/// guest issues ≥2 distinct `ChildRequest` verbs, and the host's responses
/// change the bytes the guest emits").
///
/// `max_hypercalls` bounds the exit loop so a guest that never issues the
/// halt verb cannot hang the host.
///
/// Every `segment` is mapped at its own real permissions
/// (`crate::loader::RawSegment::from_segment` — the exact marshaling
/// [`crate::loader::GuestMemory::map`] uses), never the single RWX blob the
/// scratch guest's `hv_vm_map` call uses at `backend_macos.c:87` — that
/// blob is a deliberately minimal proof surface for a guest with no OS
/// layer at all, not the shape a real ELF-loaded guest should run under.
/// One additional writable, non-executable segment covers the range between
/// the last ELF segment and `GUEST_MEMORY_SIZE`
/// ([`crate::loader::RawSegment::stack`]) — the stack reservation
/// `guests/lambda/link.ld`'s `__stack_top` implies but no `PT_LOAD` entry
/// declares.
///
/// # Errors
///
/// Returns [`ProximaError::Upstream`] naming the failing hypervisor call,
/// an exceeded hypercall budget, or a hypercall dispatch failure.
#[cfg(all(
    feature = "std",
    any(
        all(target_os = "linux", target_arch = "x86_64"),
        all(target_os = "macos", target_arch = "aarch64")
    )
))]
pub fn run_dispatch_loop(
    entry: u64,
    segments: &[crate::elf::Segment<'_>],
    configured_response: ChildResponse,
    max_hypercalls: usize,
    emitted_capacity: usize,
) -> Result<(Vec<ChildRequest>, Vec<u8>), ProximaError> {
    use std::ffi::CStr;
    use std::os::raw::c_char;

    use crate::loader::RawSegment;

    const ERROR_CAPACITY: usize = 512;
    /// The lambda guest's linker script (`guests/lambda/link.ld`) reserves
    /// the entire 64 MiB `RAM` region for `__stack_top`; the run loop must
    /// map at least that much guest memory or the guest's own stack pointer
    /// lands outside the mapped region.
    const GUEST_MEMORY_SIZE: u64 = 64 * 1024 * 1024;

    unsafe extern "C" {
        fn proxima_vm_run_dispatch_loop(
            segments: *const RawSegment,
            segment_count: usize,
            guest_memory_size: u64,
            entry: u64,
            dispatcher: *const core::ffi::c_void,
            max_hypercalls: usize,
            emitted_out: *mut u8,
            emitted_capacity: usize,
            emitted_length_out: *mut usize,
            error_buffer: *mut c_char,
            error_capacity: usize,
        ) -> i32;
    }

    /// Both host mapping APIs this crate drives (`hv_vm_map`,
    /// `KVM_SET_USER_MEMORY_REGION`) only accept page-aligned ranges — the
    /// base page size on both `aarch64` and `x86_64` (hugepages aside).
    /// Starting the stack region on a page boundary keeps it out of
    /// whichever page the last real ELF segment's tail shares, so the C
    /// side's window-merge (`build_mapped_windows` in each backend) never
    /// has to fold the writable stack into the same page as an executable
    /// segment.
    const GUEST_PAGE_SIZE: u64 = 4096;

    let extent = segments
        .iter()
        .map(|segment| segment.virtual_address() + segment.memory_size())
        .max()
        .unwrap_or(0);
    let stack_start = extent.div_ceil(GUEST_PAGE_SIZE) * GUEST_PAGE_SIZE;
    let mut raw_segments: Vec<RawSegment> = segments.iter().map(RawSegment::from_segment).collect();
    let stack_size = GUEST_MEMORY_SIZE.saturating_sub(stack_start);
    if stack_size > 0 {
        raw_segments.push(RawSegment::stack(stack_start, stack_size));
    }

    let dispatcher = FfiRecordingDispatcher::new(configured_response);
    let mut emitted = alloc::vec![0_u8; emitted_capacity];
    let mut emitted_length: usize = 0;
    let mut error_buffer = [0_i8; ERROR_CAPACITY];

    let status = unsafe {
        proxima_vm_run_dispatch_loop(
            raw_segments.as_ptr(),
            raw_segments.len(),
            GUEST_MEMORY_SIZE,
            entry,
            (&raw const dispatcher).cast(),
            max_hypercalls,
            emitted.as_mut_ptr(),
            emitted.len(),
            &raw mut emitted_length,
            error_buffer.as_mut_ptr(),
            error_buffer.len(),
        )
    };
    if status != 0 {
        let message = unsafe { CStr::from_ptr(error_buffer.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        return Err(ProximaError::Upstream(message));
    }
    emitted.truncate(emitted_length);
    Ok((dispatcher.requests(), emitted))
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

    /// Postcard variant discriminant for `ChildRequest::Read` — matches
    /// `guests/lambda/src/main.rs`'s constant of the same name, reused as
    /// the hypercall verb.
    const READ_VERB: u16 = 0x00;

    /// Postcard variant discriminant for `ChildRequest::Close` (source
    /// order: Read=0, Write=1, Open=2, Close=3, Stat=4 —
    /// `proxima-protocols/src/process/protocol.rs:72-77`).
    const CLOSE_VERB: u16 = 0x03;

    const FFI_RESPONSE_CAPACITY: usize = 256;

    /// Drives one hypercall through [`proxima_vm_dispatch_hypercall`]
    /// against a synthetic guest-memory buffer holding `payload` at a
    /// nonzero offset (exercising the pointer arithmetic, not just
    /// `pointer == 0`, per `src/abi.rs`'s own test convention), and
    /// returns the raw response bytes the trampoline wrote back.
    fn dispatch_one(dispatcher: &FfiRecordingDispatcher, verb: u16, payload: &[u8]) -> Vec<u8> {
        let pointer = 4_usize;
        let mut guest_memory = alloc::vec![0xaa_u8; pointer + payload.len()];
        guest_memory[pointer..pointer + payload.len()].copy_from_slice(payload);
        let mut response_out = alloc::vec![0_u8; FFI_RESPONSE_CAPACITY];

        // SAFETY: `dispatcher` is a live `&FfiRecordingDispatcher` cast to
        // the `void *` the extern fn expects; `guest_memory` and
        // `response_out` are live, non-aliasing, correctly-sized buffers.
        let written = unsafe {
            proxima_vm_dispatch_hypercall(
                (dispatcher as *const FfiRecordingDispatcher).cast(),
                guest_memory.as_ptr(),
                guest_memory.len(),
                verb,
                pointer as u64,
                payload.len() as u64,
                response_out.as_mut_ptr(),
                response_out.len(),
            )
        };
        assert!(written >= 0, "trampoline reported failure: {written}");
        response_out.truncate(written as usize);
        response_out
    }

    /// [`two_distinct_child_request_verbs_are_recorded_in_call_order`] and
    /// [`two_differently_canned_responses_produce_different_emitted_bytes_for_the_same_request`]
    /// together are the M1 exit proof from `tools/proxima-vm/ROADMAP.md`:
    /// driving two distinct `ChildRequest` verbs through one dispatcher
    /// proves the request side decodes correctly per call (not stuck on
    /// the first payload); driving the *same* request through two
    /// differently-configured dispatchers and observing different emitted
    /// bytes proves the response side actually carries the host's
    /// configured answer through the channel, not some fixed echo baked
    /// into the trampoline.
    #[test]
    fn two_distinct_child_request_verbs_are_recorded_in_call_order() {
        let configured = ChildResponse::Read(ReadResponse {
            bytes: b"canned-response".to_vec(),
            eof: true,
        });
        let dispatcher = FfiRecordingDispatcher::new(configured.clone());

        let read_request = ChildRequest::Read {
            path: "/etc/hostname".to_string(),
            max_bytes: 256,
            offset: 0,
        };
        let close_request = ChildRequest::Close {
            path: "/tmp/lambda-guest".to_string(),
        };
        let read_wire_bytes =
            postcard::to_allocvec(&read_request).expect("postcard encode read request");
        let close_wire_bytes =
            postcard::to_allocvec(&close_request).expect("postcard encode close request");

        let read_response = dispatch_one(&dispatcher, READ_VERB, &read_wire_bytes);
        let close_response = dispatch_one(&dispatcher, CLOSE_VERB, &close_wire_bytes);

        assert_eq!(
            dispatcher.requests(),
            vec![read_request, close_request],
            "the dispatcher must record exactly the two distinct requests it decoded, in call order"
        );

        let expected_bytes =
            postcard::to_allocvec(&configured).expect("postcard encode configured response");
        assert_eq!(read_response, expected_bytes);
        assert_eq!(close_response, expected_bytes);
    }

    #[test]
    fn two_differently_canned_responses_produce_different_emitted_bytes_for_the_same_request() {
        let read_request = ChildRequest::Read {
            path: "/etc/hostname".to_string(),
            max_bytes: 256,
            offset: 0,
        };
        let read_wire_bytes =
            postcard::to_allocvec(&read_request).expect("postcard encode read request");

        let response_alpha = ChildResponse::Read(ReadResponse {
            bytes: b"alpha-canned-response".to_vec(),
            eof: true,
        });
        let dispatcher_alpha = FfiRecordingDispatcher::new(response_alpha.clone());
        let bytes_alpha = dispatch_one(&dispatcher_alpha, READ_VERB, &read_wire_bytes);

        let response_beta = ChildResponse::Close;
        let dispatcher_beta = FfiRecordingDispatcher::new(response_beta.clone());
        let bytes_beta = dispatch_one(&dispatcher_beta, READ_VERB, &read_wire_bytes);

        assert_eq!(dispatcher_alpha.requests(), vec![read_request.clone()]);
        assert_eq!(dispatcher_beta.requests(), vec![read_request]);

        assert_eq!(
            bytes_alpha,
            postcard::to_allocvec(&response_alpha).expect("postcard encode response_alpha")
        );
        assert_eq!(
            bytes_beta,
            postcard::to_allocvec(&response_beta).expect("postcard encode response_beta")
        );
        assert_ne!(
            bytes_alpha, bytes_beta,
            "the same request driven through two differently-configured dispatchers must not replay the same bytes"
        );
    }
}
