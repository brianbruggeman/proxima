//! M11's differential exit clause: "a differential test against QEMU's
//! `info tlb` on the same page tables" (`ROADMAP.md` M11 section). Three
//! formats have a real QEMU-captured fixture and differential test in
//! this file: x86-64 4-level (below), x86-64 5-level/LA57, and AArch64
//! stage-1 (see each test's own doc comment for its capture recipe).
//! AArch64 stage-2 is not covered here — see `BENCH_LOG.md`'s M11 section
//! for why, and the exact monitor commands tried.
//!
//! The x86-64 4-level fixture is a real captured guest-physical-memory
//! dump
//! (`fixtures/qemu_9_2_2_x86_64_page_tables.bin`, principle 9: real data,
//! never hand-rolled literals): a minimal multiboot stub built with
//! `x86_64-unknown-linux-gnu-gcc`'s cross `as`/`ld` boots under
//! `qemu-system-x86_64 9.2.2` (`-M pc`), sets up one identity 2MB huge
//! page (so its own code stays mapped across the `CR0.PG` transition)
//! plus one explicit 4KB mapping — `PML4[0] -> PDPT[0] -> PD[2] (table,
//! not huge) -> PT[0]` encoding vaddr `0x400000 -> paddr 0x500000` — then
//! touches that page in 64-bit long mode and halts.
//!
//! QEMU's own monitor, queried live over `info tlb` against that exact
//! run, reported:
//!
//! ```text
//! 0000000000400000: 0000000000500000 ----A---W
//! ```
//!
//! (`W` = writable, `A` = accessed; no `X` shown because this build's
//! monitor output does not surface NX/execute state for this entry — this
//! walker's own execute-permission path is exercised separately by the
//! unit-test sad paths in `src/page_table.rs`, not by this fixture).
//! `pmemsave` captured the guest's first 6 pages of physical memory
//! (0x0..0x6000) to disk immediately after; `dd` trimmed that to the
//! committed fixture. This test re-derives the same `0x400000 ->
//! 0x500000` mapping from those exact bytes through
//! [`proxima_vm::page_table::walk`] and asserts parity with what QEMU's
//! `info tlb` reported, never re-deriving the fixture itself.

#![allow(clippy::expect_used)]

use proxima_vm::page_table::{Access, Format, walk};

const FIXTURE: &[u8] = include_bytes!("fixtures/qemu_9_2_2_x86_64_page_tables.bin");

#[test]
fn walk_matches_qemus_own_info_tlb_output_on_the_same_page_tables() {
    let resolved = walk(
        FIXTURE,
        0x1000, // CR3 the guest loaded: PML4 base
        0x400000,
        Access {
            write: true,
            execute: false,
        },
        Format::X86_64FourLevel,
    )
    .expect("QEMU's own MMU resolved this exact mapping; ours must too");

    assert_eq!(
        resolved, 0x500000,
        "must match qemu-system-x86_64 9.2.2's `info tlb`: `0000000000400000: 0000000000500000 ----A---W`"
    );
}

/// x86-64 5-level (LA57) differential — same mechanism as the 4-level
/// test above, extended by one table level (BENCH_LOG.md M11 section,
/// "not built" item now closed). The guest stub sets `CR4.LA57` and
/// `CR4.PAE`, `EFER.LME`, and `CR0.PG`, walks a 5-level tree (PML5 ->
/// PML4 -> PDPT -> PD -> PT), and maps the same `vaddr 0x400000 -> paddr
/// 0x500000` shape as the 4-level fixture, one table level deeper.
///
/// Booted under `qemu-system-x86_64 9.2.2 -M pc -cpu qemu64,+la57
/// -kernel stub.elf` (a 32-bit-ELF, Xen-PVH-noted stub — this qemu
/// build's `-kernel` loader for x86_64 has no `multiboot.bin` option ROM
/// wired for `-M pc`, so a plain multiboot header alone was refused with
/// `Error loading uncompressed kernel without PVH ELF Note`; adding an
/// `XEN_ELFNOTE_PHYS32_ENTRY` note, per docs.xenproject.org/misc/pvh.html,
/// let the same loader direct-boot the 32-bit ELF at its entry point).
/// Live monitor query over that run:
///
/// ```text
/// (qemu) info tlb
/// 0000000000000000: 0000000000000000 --PDA---W
/// 0000000000400000: 0000000000500000 ---DA---W
/// (qemu) info registers
/// CR3=0000000000001000 CR4=00001020 EFER=0000000000000500
/// ```
///
/// (`CR4` bit 12 = `LA57`, bit 5 = `PAE`; `EFER` bit 8 = `LME`, bit 10 =
/// `LMA`, confirming the vCPU was actually in 5-level long mode when
/// queried, not merely configured for it.) `pmemsave 0x0 0x7000` captured
/// the guest's first 7 pages of physical memory immediately after —
/// `tests/fixtures/qemu_9_2_2_x86_64_five_level_page_tables.bin`.
const FIVE_LEVEL_FIXTURE: &[u8] =
    include_bytes!("fixtures/qemu_9_2_2_x86_64_five_level_page_tables.bin");

#[test]
fn walk_matches_qemus_own_info_tlb_output_on_a_five_level_la57_page_table() {
    let resolved = walk(
        FIVE_LEVEL_FIXTURE,
        0x1000, // CR3 the guest loaded: PML5 base
        0x400000,
        Access {
            write: true,
            execute: false,
        },
        Format::X86_64FiveLevel,
    )
    .expect("QEMU's own LA57 MMU resolved this exact mapping; ours must too");

    assert_eq!(
        resolved, 0x500000,
        "must match qemu-system-x86_64 9.2.2's LA57 `info tlb`: `0000000000400000: 0000000000500000 ---DA---W`"
    );
}

/// AArch64 VMSAv8-64 stage-1 differential. The `qemu-system-aarch64`
/// monitor has no `info tlb` command at all (`help info` on a live `-M
/// virt` session lists every `info` subcommand and `tlb` is not among
/// them — an x86-only HMP command, not a cross-target one); the
/// per-address equivalent the monitor does expose is `gva2gpa`, which
/// this test's fixture was captured against instead, per the task's own
/// documented fallback.
///
/// A minimal EL1 guest stub (dropping EL2 -> EL1 first when the CPU
/// resets into EL2, the default for `cortex-a72` under `-M virt`) builds
/// a 4-level, 4KB-granule stage-1 tree with the same shape as the x86-64
/// fixture: `L2[0]` is a 2MB identity block covering the stub's own
/// running code (`0x40000000`, the `virt` machine's fixed RAM base — the
/// low physical range below it is unbacked MMIO/GIC space on this
/// machine, not RAM, so the tables cannot live below it), and `L2[2] ->
/// L3[0]` is an explicit 4KB page mapping `vaddr 0x40400000 -> paddr
/// 0x40500000` (RAM-base + 4MB -> RAM-base + 5MB, the same "+4MB/+5MB"
/// shape as the x86-64 fixture, offset by the machine's RAM base). Sets
/// `MAIR_EL1`/`TCR_EL1` (`T0SZ=16` for a 48-bit input address, `TG0=4KB`,
/// `IPS=36-bit`), `TTBR0_EL1`, then `SCTLR_EL1.M`, touches the mapped
/// page, and parks in `wfi`.
///
/// Booted under `qemu-system-aarch64 9.2.2 -M virt -cpu cortex-a72
/// -kernel stub.elf` (a plain ELF entry — this target's `-kernel` loader
/// needed no PVH-note workaround). Live monitor query over that run:
///
/// ```text
/// (qemu) gva2gpa 0x40400000
/// gpa: 0x40500000
/// ```
///
/// `pmemsave 0x40000000 0x6000` captured the guest's first 6 pages of
/// physical RAM (RAM base through the fourth table page) immediately
/// after —
/// `tests/fixtures/qemu_9_2_2_aarch64_stage1_page_tables.bin`. Because
/// this fixture's byte 0 is guest physical address `0x40000000` (the
/// `virt` machine's RAM base), not guest physical address `0` (unlike
/// the x86-64 fixtures, where PC-platform RAM legitimately starts at
/// `0`), this test pads [`walk`]'s `memory` argument with
/// `0x40000000` leading zero bytes so the fixture's own captured
/// descriptor content — which encodes true absolute physical addresses,
/// because that is what the real hardware MMU requires — indexes
/// correctly; the leading zeros are runtime scaffolding representing
/// "unbacked below RAM base", never fixture bytes themselves (principle
/// 9: only the tail is captured data).
const AARCH64_STAGE1_FIXTURE: &[u8] =
    include_bytes!("fixtures/qemu_9_2_2_aarch64_stage1_page_tables.bin");
const AARCH64_VIRT_RAM_BASE: usize = 0x4000_0000;

#[test]
fn walk_matches_qemus_own_gva2gpa_output_on_an_aarch64_stage1_page_table() {
    let mut memory = vec![0_u8; AARCH64_VIRT_RAM_BASE];
    memory.extend_from_slice(AARCH64_STAGE1_FIXTURE);

    let resolved = walk(
        &memory,
        0x4000_1000, // TTBR0_EL1 the guest loaded: L0 base
        0x4040_0000,
        Access {
            write: true,
            execute: false,
        },
        Format::Aarch64Stage1,
    )
    .expect("QEMU's own AArch64 MMU resolved this exact mapping; ours must too");

    assert_eq!(
        resolved, 0x4050_0000,
        "must match qemu-system-aarch64 9.2.2's `gva2gpa 0x40400000`: `gpa: 0x40500000`"
    );
}
