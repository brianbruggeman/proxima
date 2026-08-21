//! The M1 exit gate, as tests.
//!
//! Every case here drives production code paths that are host-portable and
//! do not touch a hypervisor: [`elf::parse_elf`] (tier-3, pure) and
//! [`dispatch::proxima_vm_dispatch_hypercall`] (the exact trampoline
//! `src/backend_macos.c`'s and `src/backend_linux.c`'s `proxima_vm_dispatch_run`
//! call back into once a real hypercall exit recovers `verb`/`pointer`/`length`
//! from the vCPU — same decode/dispatch/encode logic, driven here with a
//! synthetic guest-memory buffer instead of a live vCPU's registers). That
//! scope keeps every case in this file green on any host, unlike
//! `tests/boot.rs`/`tests/dispatch_hypercall.rs`, which need a signed
//! subprocess and only run their assertions on the two hypervisor lanes.
//!
//! # Bidirectional, not a replay
//!
//! [`two_distinct_child_request_verbs_are_recorded_in_call_order`] and
//! [`two_differently_canned_responses_produce_different_emitted_bytes_for_the_same_request`]
//! together are the M1 exit proof from `tools/proxima-vm/ROADMAP.md`:
//! driving two distinct `ChildRequest` verbs through one dispatcher proves the
//! request side decodes correctly per call (not stuck on the first payload);
//! driving the *same* request through two differently-configured dispatchers
//! and observing different emitted bytes proves the response side actually
//! carries the host's configured answer through the channel, not some fixed
//! echo baked into the trampoline.
//!
//! # Sad paths
//!
//! [`malformed_elf_is_rejected_with_a_named_loader_error`] and
//! [`out_of_range_p_vaddr_is_rejected_with_the_bad_address_visible_in_the_error`]
//! assert `elf::parse_elf` rejects bad input with a named [`LoaderError`]
//! variant — never a panic, never a silent empty accept.
//!
//! # The M0 degenerate control
//!
//! `tests/boot.rs`'s `empty_guest_emits_zero_bytes` is the M0 degenerate
//! control (a guest that emits nothing must report zero bytes, not a false
//! success). It is untouched by this file and keeps running as part of
//! `cargo nextest run -p proxima-vm`.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use proxima_protocols::process::{ChildRequest, ChildResponse, ReadResponse};
use proxima_vm::dispatch::{self, RecordingDispatcher};
use proxima_vm::elf::{self, LoaderError};

const RESPONSE_CAPACITY: usize = 256;

/// Postcard variant discriminant for `ChildRequest::Read` — matches
/// `guests/lambda/src/main.rs`'s and `src/dispatch.rs`'s constant of the
/// same name, reused as the hypercall verb.
const READ_VERB: u16 = 0x00;

/// Postcard variant discriminant for `ChildRequest::Close` (source order:
/// Read=0, Write=1, Open=2, Close=3, Stat=4 —
/// `proxima-protocols/src/process/protocol.rs:72-77`).
const CLOSE_VERB: u16 = 0x03;

/// Drives one hypercall through `dispatch::proxima_vm_dispatch_hypercall`
/// against a synthetic guest-memory buffer holding `payload` at a nonzero
/// offset (exercising the pointer arithmetic, not just `pointer == 0`, per
/// `src/abi.rs`'s own test convention), and returns the raw response bytes
/// the trampoline wrote back.
fn dispatch_one(dispatcher: &RecordingDispatcher, verb: u16, payload: &[u8]) -> Vec<u8> {
    let pointer = 4_usize;
    let mut guest_memory = vec![0xaa_u8; pointer + payload.len()];
    guest_memory[pointer..pointer + payload.len()].copy_from_slice(payload);
    let mut response_out = vec![0_u8; RESPONSE_CAPACITY];

    let written = unsafe {
        dispatch::proxima_vm_dispatch_hypercall(
            dispatcher,
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

#[proxima::test]
async fn two_distinct_child_request_verbs_are_recorded_in_call_order() {
    let configured = ChildResponse::Read(ReadResponse {
        bytes: b"canned-response".to_vec(),
        eof: true,
    });
    let dispatcher = RecordingDispatcher::new(configured.clone());

    let read_request = ChildRequest::Read {
        path: "/etc/hostname".to_string(),
        max_bytes: 256,
        offset: 0,
    };
    let close_request = ChildRequest::Close {
        path: "/tmp/lambda-guest".to_string(),
    };
    let read_wire_bytes = postcard::to_allocvec(&read_request).expect("postcard encode read request");
    let close_wire_bytes =
        postcard::to_allocvec(&close_request).expect("postcard encode close request");

    let read_response = dispatch_one(&dispatcher, READ_VERB, &read_wire_bytes);
    let close_response = dispatch_one(&dispatcher, CLOSE_VERB, &close_wire_bytes);

    assert_eq!(
        dispatcher.requests(),
        vec![read_request, close_request],
        "the dispatcher must record exactly the two distinct requests it decoded, in call order"
    );

    let expected_bytes = postcard::to_allocvec(&configured).expect("postcard encode configured response");
    assert_eq!(read_response, expected_bytes);
    assert_eq!(close_response, expected_bytes);
}

#[proxima::test]
async fn two_differently_canned_responses_produce_different_emitted_bytes_for_the_same_request() {
    let read_request = ChildRequest::Read {
        path: "/etc/hostname".to_string(),
        max_bytes: 256,
        offset: 0,
    };
    let read_wire_bytes = postcard::to_allocvec(&read_request).expect("postcard encode read request");

    let response_alpha = ChildResponse::Read(ReadResponse {
        bytes: b"alpha-canned-response".to_vec(),
        eof: true,
    });
    let dispatcher_alpha = RecordingDispatcher::new(response_alpha.clone());
    let bytes_alpha = dispatch_one(&dispatcher_alpha, READ_VERB, &read_wire_bytes);

    let response_beta = ChildResponse::Close;
    let dispatcher_beta = RecordingDispatcher::new(response_beta.clone());
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

/// e_phoff this fixture declares — the ELF64 spec's own `Elf64_Ehdr` size
/// (64 bytes), not a proxima-invented value.
const PROGRAM_HEADER_OFFSET: usize = 64;
const SEGMENT_CONTENT: [u8; 4] = [0xd4, 0x20, 0x00, 0x00];

/// A minimal, valid ELF64 `ET_EXEC` image carrying exactly one readable and
/// executable `PT_LOAD` segment at `p_vaddr = 0`. Built directly here
/// (mirroring the gABI field layout `elf.rs`'s own module doc cites) rather
/// than reusing `proxima_vm::elf::test_support`, which is `pub(crate)` and
/// not visible from this external integration-test crate.
fn valid_single_segment_elf() -> Vec<u8> {
    let file_offset = (PROGRAM_HEADER_OFFSET + 56) as u64;
    let mut image = vec![0_u8; file_offset as usize + SEGMENT_CONTENT.len()];

    image[0..4].copy_from_slice(&[0x7f, b'E', b'L', b'F']);
    image[4] = 2; // ELFCLASS64
    image[5] = 1; // ELFDATA2LSB
    image[6] = 1; // EI_VERSION
    image[16..18].copy_from_slice(&2_u16.to_le_bytes()); // e_type = ET_EXEC
    image[20..24].copy_from_slice(&1_u32.to_le_bytes()); // e_version
    image[24..32].copy_from_slice(&0_u64.to_le_bytes()); // e_entry
    image[32..40].copy_from_slice(&(PROGRAM_HEADER_OFFSET as u64).to_le_bytes()); // e_phoff
    image[52..54].copy_from_slice(&64_u16.to_le_bytes()); // e_ehsize
    image[54..56].copy_from_slice(&56_u16.to_le_bytes()); // e_phentsize
    image[56..58].copy_from_slice(&1_u16.to_le_bytes()); // e_phnum

    let phdr = PROGRAM_HEADER_OFFSET;
    image[phdr..phdr + 4].copy_from_slice(&1_u32.to_le_bytes()); // p_type = PT_LOAD
    image[phdr + 4..phdr + 8].copy_from_slice(&5_u32.to_le_bytes()); // p_flags = PF_R | PF_X
    image[phdr + 8..phdr + 16].copy_from_slice(&file_offset.to_le_bytes()); // p_offset
    image[phdr + 16..phdr + 24].copy_from_slice(&0_u64.to_le_bytes()); // p_vaddr
    image[phdr + 24..phdr + 32].copy_from_slice(&0_u64.to_le_bytes()); // p_paddr
    image[phdr + 32..phdr + 40].copy_from_slice(&(SEGMENT_CONTENT.len() as u64).to_le_bytes()); // p_filesz
    image[phdr + 40..phdr + 48].copy_from_slice(&(SEGMENT_CONTENT.len() as u64).to_le_bytes()); // p_memsz
    // align 1 means "no alignment constraint" (gABI) — this fixture's
    // p_vaddr (0) and p_offset (120) are not congruent modulo a real page
    // size, and alignment congruence is not what these two tests exercise.
    image[phdr + 48..phdr + 56].copy_from_slice(&1_u64.to_le_bytes()); // p_align

    image[file_offset as usize..file_offset as usize + SEGMENT_CONTENT.len()]
        .copy_from_slice(&SEGMENT_CONTENT);
    image
}

#[proxima::test]
#[case::random_bytes_are_not_an_elf_at_all(
    b"#!/bin/sh\necho hello\n".repeat(4),
    LoaderError::BadMagic
)]
#[case::truncated_before_the_header_ends(
    valid_single_segment_elf()[..40].to_vec(),
    LoaderError::Truncated { need: 64, got: 40 }
)]
async fn malformed_elf_is_rejected_with_a_named_loader_error(
    #[case] image: Vec<u8>,
    #[case] expected: LoaderError,
) {
    let error = elf::parse_elf::<4>(&image).expect_err("malformed image must be rejected, not accepted");
    assert_eq!(error, expected);
}

#[proxima::test]
async fn out_of_range_p_vaddr_is_rejected_with_the_bad_address_visible_in_the_error() {
    let baseline = valid_single_segment_elf();
    elf::parse_elf::<4>(&baseline).expect("fixture must be a valid baseline before mutation");

    let mut image = baseline;
    let bad_p_vaddr = u64::MAX - 1;
    let phdr = PROGRAM_HEADER_OFFSET;
    image[phdr + 16..phdr + 24].copy_from_slice(&bad_p_vaddr.to_le_bytes());

    let error =
        elf::parse_elf::<4>(&image).expect_err("a p_vaddr that overflows the address space must be rejected");

    assert_eq!(
        error,
        LoaderError::SegmentAddressOverflow {
            header_index: 0,
            virtual_address: bad_p_vaddr,
            memory_size: SEGMENT_CONTENT.len() as u64,
        }
    );
}
