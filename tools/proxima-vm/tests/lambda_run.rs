//! The M1 exit gate, as tests.
//!
//! Every case here drives production code paths that are host-portable and
//! do not touch a hypervisor: [`elf::parse_elf`] (tier-3, pure). That scope
//! keeps every case in this file green on any host, unlike
//! `tests/boot.rs`/`tests/dispatch_hypercall.rs`, which need a signed
//! subprocess and only run their assertions on the two hypervisor lanes.
//!
//! The real-VM-exit M1 exit proof — a guest ELF issuing ≥2 distinct
//! `ChildRequest` verbs through a real `hvc #0` trap, with the host's
//! configured response provably changing the guest's emitted bytes — lives
//! in `tests/dispatch_hypercall.rs`, which drives
//! `dispatch::run_dispatch_loop` against the real `proxima-vm-guest-lambda`
//! ELF. This file stays scoped to host-portable, hypervisor-free coverage:
//! `elf::parse_elf`'s sad paths.
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

use proxima_vm::elf::{self, LoaderError};

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
    let error =
        elf::parse_elf::<4>(&image).expect_err("malformed image must be rejected, not accepted");
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

    let error = elf::parse_elf::<4>(&image)
        .expect_err("a p_vaddr that overflows the address space must be rejected");

    assert_eq!(
        error,
        LoaderError::SegmentAddressOverflow {
            header_index: 0,
            virtual_address: bad_p_vaddr,
            memory_size: SEGMENT_CONTENT.len() as u64,
        }
    );
}
