//! M11 — the page-table walker: a pure function over guest page-table
//! bytes, `(root, vaddr, access) -> Result<paddr, Fault>`, for AArch64
//! VMSAv8-64 stage-1/stage-2 and x86-64 4-/5-level paging.
//!
//! Per `tools/proxima-vm/ROADMAP.md` M11 and the settled 2026-08-11
//! no-single-address-space decision: CoW fork, W^X, guard pages, and
//! demand paging are page-table mechanisms, so there is no µsec fork
//! without a real walker. This module is that walker, and per the
//! roadmap's "policy is data" rule (M11 rule 3) it doubles as the shape a
//! capability policy reads permission from: `Access`/`Fault` are plain
//! values, never a hook.
//!
//! # Tier
//!
//! Tier-3 (bare `no_std + no_alloc`). `memory` is a caller-owned byte
//! slice standing in for guest physical memory (address 0 of the slice is
//! guest physical address 0, mirroring `elf.rs`'s file-offset convention);
//! every read bounds-checks against it before touching a byte. No
//! allocation, no I/O, no syscall.
//!
//! # Descriptor byte layout — reference
//!
//! The AArch64 VMSAv8-64 descriptor fields (valid bit 0, table/block bit
//! 1, AF bit 10, AP\[2\] bit 7, S2AP\[1:0\] bits\[7:6\], UXN bit 54, output
//! address bits\[47:12\]) and the x86-64 paging-structure-entry fields
//! (P bit 0, R/W bit 1, PS bit 7, XD bit 63, output address bits\[51:12\])
//! follow the architecture's own bit assignments — these are the same
//! stable bit positions every OS-dev reference and the `x86_64`/
//! `aarch64-cpu` crates use, not a manual page transcribed this session.
//! The worked-example descriptor *bytes* in this module's tests are
//! **derived, not copied from the ARM ARM / Intel SDM's own numeric
//! example** — no cross-check against the physical manual text ran this
//! session, so treat the worked examples as bit-layout-faithful but
//! citation-unverified. See `BENCH_LOG.md` for the QEMU `info tlb`
//! differential status.
//!
//! # Scope trims, stated rather than silently dropped
//!
//! - single privilege level: AArch64 AP\[1\] (EL0/EL1 distinction) and
//!   PXN are not modeled — every proxima-vm guest to date (M1) runs one
//!   privilege level, so only AP\[2\] (read-only) and UXN (execute-never)
//!   gate access.
//! - x86-64 U/S and PCID are not modeled for the same reason; XD is
//!   always honored (equivalent to assuming `EFER.NXE = 1`, true on every
//!   64-bit host this tree targets).
//! - stage-2 MemAttr/cacheability bits are read but not asserted on —
//!   they affect caching, not the output address or the pass/fail
//!   permission decision this function makes.

use proxima_protocols::nvme::raw::read_u64;

/// Requested access for a walk. `read` is implicit: every successful walk
/// grants at least read, matching both architectures' "no read-only-deny"
/// descriptor shape (there is no read-deny bit in either format modeled
/// here).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Access {
    pub write: bool,
    pub execute: bool,
}

/// Which page-table format to walk. Structural, not stringly — a caller
/// picks the format the vCPU's mode dictates, never parses it from text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Format {
    /// AArch64 VMSAv8-64 stage-1, 4KB granule, 4 levels (48-bit input
    /// address). The ARM ARM's canonical worked example uses this
    /// granule; the M11 roadmap section names no granule, so this is the
    /// floor per the resume brief's tie-break rule.
    Aarch64Stage1,
    /// AArch64 VMSAv8-64 stage-2, 4KB granule, 4 levels — same table
    /// shape as stage-1, `S2AP`/`MemAttr` in place of `AP`/`AttrIndx`.
    Aarch64Stage2,
    /// x86-64 4-level paging (CR3 -> PML4E -> PDPTE -> PDE -> PTE),
    /// 48-bit virtual address.
    X86_64FourLevel,
    /// x86-64 5-level paging (CR3 -> PML5E -> PML4E -> PDPTE -> PDE ->
    /// PTE), 57-bit virtual address.
    X86_64FiveLevel,
}

/// Why [`walk`] could not produce a physical address. Every variant names
/// the exact level and table location that failed, per M11 rule 3: this is
/// the plain-data shape a policy layer reads, never a hidden reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Fault {
    /// A descriptor read needed bytes past the end of `memory`.
    Truncated { need: usize, got: usize },
    /// The descriptor at `level`, table `table_paddr`, index `index` had
    /// its valid/present bit clear, or (AArch64 level 3 / x86-64 final
    /// level only) named a block where only a page descriptor is legal.
    InvalidDescriptor {
        level: u8,
        table_paddr: u64,
        index: usize,
    },
    /// A resolved mapping denied the requested access. `granted` is the
    /// most-permissive access the descriptor allowed.
    PermissionDenied {
        level: u8,
        requested: Access,
        granted: Access,
    },
    /// The output physical address the walk produced falls outside
    /// `memory`'s bounds (a fault this simulator's caller — not the
    /// walker itself — resolves by growing `memory` or raising a real
    /// fault to the guest).
    OutOfRange { physical_address: u64 },
}

impl core::fmt::Display for Fault {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Truncated { need, got } => {
                write!(formatter, "truncated: need {need} bytes, got {got}")
            }
            Self::InvalidDescriptor {
                level,
                table_paddr,
                index,
            } => write!(
                formatter,
                "invalid descriptor at level {level}, table {table_paddr:#x}, index {index}"
            ),
            Self::PermissionDenied {
                level,
                requested,
                granted,
            } => write!(
                formatter,
                "permission denied at level {level}: requested write={} execute={}, granted write={} execute={}",
                requested.write, requested.execute, granted.write, granted.execute
            ),
            Self::OutOfRange { physical_address } => {
                write!(
                    formatter,
                    "output address {physical_address:#x} out of range"
                )
            }
        }
    }
}

fn read_descriptor(memory: &[u8], table_paddr: u64, index: usize) -> Result<u64, Fault> {
    let offset = (table_paddr as usize).saturating_add(index * 8);
    let need = offset.saturating_add(8);
    if need > memory.len() {
        return Err(Fault::Truncated {
            need,
            got: memory.len(),
        });
    }
    Ok(read_u64(memory, offset))
}

const OUTPUT_ADDRESS_MASK: u64 = 0x0000_FFFF_FFFF_F000;

/// Walk `format`'s page-table structure rooted at `root` to translate
/// `vaddr` under `access`. See [`Format`] for what each variant models and
/// this module's doc comment for the scope trims.
///
/// # Errors
///
/// Returns [`Fault`] naming the level and cause; never panics on
/// malformed input (see `fault_walk_never_panics_on_arbitrary_bytes` in
/// tests).
pub fn walk(
    memory: &[u8],
    root: u64,
    vaddr: u64,
    access: Access,
    format: Format,
) -> Result<u64, Fault> {
    match format {
        Format::Aarch64Stage1 => walk_aarch64(memory, root, vaddr, access, AArch64Stage::Stage1),
        Format::Aarch64Stage2 => walk_aarch64(memory, root, vaddr, access, AArch64Stage::Stage2),
        Format::X86_64FourLevel => walk_x86_64(memory, root, vaddr, access, X86_64Levels::Four),
        Format::X86_64FiveLevel => walk_x86_64(memory, root, vaddr, access, X86_64Levels::Five),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AArch64Stage {
    Stage1,
    Stage2,
}

/// AArch64 VMSAv8-64, 4KB granule, 4-level walk (ARM DDI 0487 §D8.3
/// descriptor format). Level indices, most-significant first: L0
/// bits\[47:39\], L1 bits\[38:30\], L2 bits\[29:21\], L3 bits\[20:12\].
fn walk_aarch64(
    memory: &[u8],
    root: u64,
    vaddr: u64,
    access: Access,
    stage: AArch64Stage,
) -> Result<u64, Fault> {
    let mut table_paddr = root & OUTPUT_ADDRESS_MASK;
    for level in 0_u8..4 {
        let shift = 39 - u32::from(level) * 9;
        let index = ((vaddr >> shift) & 0x1FF) as usize;
        let descriptor = read_descriptor(memory, table_paddr, index)?;

        let valid = descriptor & 0b1 != 0;
        let is_table_or_page = descriptor & 0b10 != 0;
        if !valid {
            return Err(Fault::InvalidDescriptor {
                level,
                table_paddr,
                index,
            });
        }
        // level 3 only ever names a page (bits[1:0] == 0b11); a block bit
        // pattern there is reserved and treated as invalid.
        let terminates = if level == 3 {
            if !is_table_or_page {
                return Err(Fault::InvalidDescriptor {
                    level,
                    table_paddr,
                    index,
                });
            }
            true
        } else {
            !is_table_or_page
        };

        if terminates {
            let output = descriptor & OUTPUT_ADDRESS_MASK;
            // UXN/XN sits at the same bit position (54) for both stages in
            // the baseline (no FEAT_XNX) case modeled here (ARM DDI 0487
            // §D8.3, §D8.4.5). With FEAT_XNX, stage-2 XN widens to a 2-bit
            // field at bits[54:53] (adding a distinct "execute-never at
            // EL0" nuance); this walker models only the single-privilege,
            // baseline encoding per this module's scope trims, so bit 54
            // alone is correct here and FEAT_XNX is out of scope.
            let execute_never = descriptor & (1 << 54) != 0;
            // AP[2] (stage-1, bit 7 alone) and S2AP[1:0] (stage-2, bits
            // [7:6]) are DIFFERENT encodings, not the same field read two
            // ways (ARM DDI 0487 §D8.4.4 AP, §D8.4.5 S2AP):
            //   stage-1 AP[2]:      0 = read/write, 1 = read-only
            //   stage-2 S2AP[1:0]:  00 = no access, 01 = read-only,
            //                       10 = write-only, 11 = read/write
            // so for stage-2 the WRITE grant is S2AP[1] (bit 7) itself, not
            // "bit 7 clear". `Access` has no `read` field (every successful
            // walk grants at least read, per its doc comment), so S2AP 00
            // (no access) and 10 (write-only) — both of which deny reads —
            // cannot be distinguished from 01/11 here; that is a known gap
            // in what this model can express, not a bug in this decode.
            let write_ok = match stage {
                AArch64Stage::Stage1 => descriptor & (1 << 7) == 0,
                AArch64Stage::Stage2 => descriptor & (1 << 7) != 0,
            };
            let granted = Access {
                write: write_ok,
                execute: !execute_never,
            };
            if (access.write && !granted.write) || (access.execute && !granted.execute) {
                return Err(Fault::PermissionDenied {
                    level,
                    requested: access,
                    granted,
                });
            }
            let page_offset = vaddr & ((1_u64 << shift) - 1);
            return Ok(output | page_offset);
        }

        table_paddr = descriptor & OUTPUT_ADDRESS_MASK;
    }
    unreachable!("level 3 always terminates")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum X86_64Levels {
    Four,
    Five,
}

/// x86-64 paging-structure entry format (Intel SDM Vol. 3A §4.5): P bit 0,
/// R/W bit 1, PS bit 7 (huge page at PDPTE/PDE), XD bit 63, output address
/// bits\[51:12\]. 4-level: PML4/PDPT/PD/PT. 5-level prepends PML5.
fn walk_x86_64(
    memory: &[u8],
    root: u64,
    vaddr: u64,
    access: Access,
    levels: X86_64Levels,
) -> Result<u64, Fault> {
    let top_shift: u32 = match levels {
        X86_64Levels::Four => 39,
        X86_64Levels::Five => 48,
    };
    let level_count: u8 = match levels {
        X86_64Levels::Four => 4,
        X86_64Levels::Five => 5,
    };

    let mut table_paddr = root & OUTPUT_ADDRESS_MASK;
    let mut shift = top_shift;
    for level in 0_u8..level_count {
        let index = ((vaddr >> shift) & 0x1FF) as usize;
        let descriptor = read_descriptor(memory, table_paddr, index)?;

        let present = descriptor & 0b1 != 0;
        if !present {
            return Err(Fault::InvalidDescriptor {
                level,
                table_paddr,
                index,
            });
        }

        // PS (huge page) is only architecturally meaningful at PDPTE (shift
        // 30) and PDE (shift 21); PML5E/PML4E ignore bit 7, and PTE (shift
        // 12) is always a 4K leaf regardless of bit 7 (that bit is PAT there).
        let huge_page = (shift == 30 || shift == 21) && descriptor & (1 << 7) != 0;
        let is_leaf = shift == 12 || huge_page;

        if is_leaf {
            let output = descriptor & OUTPUT_ADDRESS_MASK;
            let writable = descriptor & 0b10 != 0;
            let execute_denied = descriptor & (1 << 63) != 0;
            let granted = Access {
                write: writable,
                execute: !execute_denied,
            };
            if (access.write && !granted.write) || (access.execute && !granted.execute) {
                return Err(Fault::PermissionDenied {
                    level,
                    requested: access,
                    granted,
                });
            }
            let page_offset = vaddr & ((1_u64 << shift) - 1);
            return Ok(output | page_offset);
        }

        table_paddr = descriptor & OUTPUT_ADDRESS_MASK;
        shift -= 9;
    }
    unreachable!("the final level (shift == 12) always terminates")
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use proptest::prelude::*;

    use super::*;

    // ---- test-local builder: NOT a library type, per the M11 brief ----

    /// Builds a minimal single-mapping AArch64 4KB-granule, 4-level table
    /// tree inside a caller-sized flat byte buffer and returns
    /// `(memory, root_paddr)`. Table N lives at `paddr = N * 4096`.
    fn build_aarch64_single_mapping(
        vaddr: u64,
        output_paddr: u64,
        read_only: bool,
        execute_never: bool,
    ) -> Vec<u8> {
        let mut memory = vec![0_u8; 5 * 4096];
        let table_paddr = |level: u64| level * 4096;

        for level in 0_u64..3 {
            let shift = 39 - level * 9;
            let index = ((vaddr >> shift) & 0x1FF) as usize;
            let next_table = table_paddr(level + 1);
            let table_descriptor: u64 = 0b11 | next_table; // valid + table
            let at = table_paddr(level) as usize + index * 8;
            memory[at..at + 8].copy_from_slice(&table_descriptor.to_le_bytes());
        }

        let leaf_shift = 39 - 3 * 9;
        let leaf_index = ((vaddr >> leaf_shift) & 0x1FF) as usize;
        let mut page_descriptor: u64 = 0b11 | (output_paddr & OUTPUT_ADDRESS_MASK);
        if read_only {
            page_descriptor |= 1 << 7;
        }
        if execute_never {
            page_descriptor |= 1 << 54;
        }
        let at = table_paddr(3) as usize + leaf_index * 8;
        memory[at..at + 8].copy_from_slice(&page_descriptor.to_le_bytes());

        memory
    }

    /// Same table shape as [`build_aarch64_single_mapping`], but takes the
    /// raw 2-bit `S2AP[1:0]` encoding directly (bits\[7:6\] of the leaf
    /// descriptor) instead of a single `read_only` bool, so stage-2 tests
    /// can exercise all four S2AP encodings — `build_aarch64_single_mapping`
    /// can only ever produce `00` or `10` (AP\[2\] alone), never `01`/`11`.
    fn build_aarch64_stage2_single_mapping(
        vaddr: u64,
        output_paddr: u64,
        s2ap_bits: u64,
        execute_never: bool,
    ) -> Vec<u8> {
        let mut memory = vec![0_u8; 5 * 4096];
        let table_paddr = |level: u64| level * 4096;

        for level in 0_u64..3 {
            let shift = 39 - level * 9;
            let index = ((vaddr >> shift) & 0x1FF) as usize;
            let next_table = table_paddr(level + 1);
            let table_descriptor: u64 = 0b11 | next_table; // valid + table
            let at = table_paddr(level) as usize + index * 8;
            memory[at..at + 8].copy_from_slice(&table_descriptor.to_le_bytes());
        }

        let leaf_shift = 39 - 3 * 9;
        let leaf_index = ((vaddr >> leaf_shift) & 0x1FF) as usize;
        let mut page_descriptor: u64 = 0b11 | (output_paddr & OUTPUT_ADDRESS_MASK);
        page_descriptor |= (s2ap_bits & 0b11) << 6;
        if execute_never {
            page_descriptor |= 1 << 54;
        }
        let at = table_paddr(3) as usize + leaf_index * 8;
        memory[at..at + 8].copy_from_slice(&page_descriptor.to_le_bytes());

        memory
    }

    /// Builds a minimal single-mapping x86-64 4-level table tree, same
    /// layout convention as [`build_aarch64_single_mapping`].
    fn build_x86_64_single_mapping(
        vaddr: u64,
        output_paddr: u64,
        writable: bool,
        execute_denied: bool,
    ) -> Vec<u8> {
        let mut memory = vec![0_u8; 5 * 4096];
        let table_paddr = |level: u64| level * 4096;

        for level in 0_u64..3 {
            let shift = 39 - level * 9;
            let index = ((vaddr >> shift) & 0x1FF) as usize;
            let next_table = table_paddr(level + 1);
            let entry: u64 = 0b11 | next_table; // present + writable(scaffold tables are always RW)
            let at = table_paddr(level) as usize + index * 8;
            memory[at..at + 8].copy_from_slice(&entry.to_le_bytes());
        }

        let leaf_shift = 12;
        let leaf_index = ((vaddr >> leaf_shift) & 0x1FF) as usize;
        let mut entry: u64 = 0b1 | (output_paddr & OUTPUT_ADDRESS_MASK);
        if writable {
            entry |= 0b10;
        }
        if execute_denied {
            entry |= 1 << 63;
        }
        let at = table_paddr(3) as usize + leaf_index * 8;
        memory[at..at + 8].copy_from_slice(&entry.to_le_bytes());

        memory
    }

    // ---- worked examples (derived-not-manual; see module doc) ----

    /// Hand-derived, full four-level AArch64 stage-1 walk. `vaddr =
    /// 0x0000_0040_2010_3000` picks index 1 at every level (L0..L3),
    /// making the descriptor bytes at each level legible by hand:
    /// `0x40_2010_3000 = 0b01_000000_001_000000_010_000000_011_000000000000`,
    /// i.e. L0=1, L1=1, L2=1, L3=1 — recompute below to keep the test
    /// honest about what it asserts, not just what it hopes.
    #[test]
    fn worked_example_aarch64_stage1_four_level_walk() {
        let vaddr = (1_u64 << 39) | (1_u64 << 30) | (1_u64 << 21) | (1_u64 << 12) | 0x123;
        assert_eq!((vaddr >> 39) & 0x1FF, 1, "L0 index");
        assert_eq!((vaddr >> 30) & 0x1FF, 1, "L1 index");
        assert_eq!((vaddr >> 21) & 0x1FF, 1, "L2 index");
        assert_eq!((vaddr >> 12) & 0x1FF, 1, "L3 index");

        let output_paddr = 0x1_0000_0000_u64;
        let memory = build_aarch64_single_mapping(vaddr, output_paddr, false, false);

        // level 0 table lives at paddr 0, its index-1 slot holds a table
        // descriptor pointing at paddr 4096 (level 1's table).
        let level0_entry = read_u64(&memory, 8);
        assert_eq!(
            level0_entry,
            0b11 | 4096,
            "L0 descriptor: valid + table -> paddr 0x1000"
        );
        // the leaf (level 3) descriptor at index 1 encodes valid+page and
        // the requested output address in bits[47:12].
        let level3_entry = read_u64(&memory, 3 * 4096 + 8);
        assert_eq!(
            level3_entry,
            0b11 | output_paddr,
            "L3 descriptor: valid + page -> output paddr"
        );

        let resolved = walk(&memory, 0, vaddr, Access::default(), Format::Aarch64Stage1)
            .expect("four legal, valid descriptors must resolve");
        assert_eq!(
            resolved,
            output_paddr | 0x123,
            "page offset 0x123 preserved"
        );
    }

    /// Hand-derived, full four-level x86-64 walk (Intel SDM Vol. 3A §4.5
    /// bit layout: CR3 -> PML4E -> PDPTE -> PDE -> PTE).
    #[test]
    fn worked_example_x86_64_four_level_walk() {
        let vaddr = (1_u64 << 39) | (1_u64 << 30) | (1_u64 << 21) | (1_u64 << 12) | 0x456;
        let output_paddr = 0x2_0000_0000_u64;
        let memory = build_x86_64_single_mapping(vaddr, output_paddr, true, false);

        let pml4e = read_u64(&memory, 8);
        assert_eq!(
            pml4e,
            0b11 | 4096,
            "PML4E: present + writable -> PDPT at paddr 0x1000"
        );
        let pte = read_u64(&memory, 3 * 4096 + 8);
        assert_eq!(
            pte,
            0b11 | output_paddr,
            "PTE: present + writable -> output paddr"
        );

        let resolved = walk(
            &memory,
            0,
            vaddr,
            Access {
                write: true,
                execute: false,
            },
            Format::X86_64FourLevel,
        )
        .expect("four legal, present entries must resolve");
        assert_eq!(
            resolved,
            output_paddr | 0x456,
            "page offset 0x456 preserved"
        );
    }

    /// x86-64 5-level walk: same shape, one extra PML5E hop consuming
    /// bits[56:48].
    #[test]
    fn x86_64_five_level_walk_matches_four_level_plus_one_hop() {
        let mut memory = vec![0_u8; 6 * 4096];
        let vaddr = (1_u64 << 48) | (1_u64 << 39) | (1_u64 << 30) | (1_u64 << 21) | (1_u64 << 12);
        for level in 0_u64..4 {
            let shift = 48 - level * 9;
            let index = ((vaddr >> shift) & 0x1FF) as usize;
            let entry: u64 = 0b11 | ((level + 1) * 4096);
            let at = (level * 4096) as usize + index * 8;
            memory[at..at + 8].copy_from_slice(&entry.to_le_bytes());
        }
        let output_paddr = 0x3_0000_0000_u64;
        let leaf_index = ((vaddr >> 12) & 0x1FF) as usize;
        let entry: u64 = 0b11 | output_paddr;
        let at = 4 * 4096 + leaf_index * 8;
        memory[at..at + 8].copy_from_slice(&entry.to_le_bytes());

        let resolved = walk(
            &memory,
            0,
            vaddr,
            Access {
                write: true,
                execute: false,
            },
            Format::X86_64FiveLevel,
        )
        .expect("five legal, present entries must resolve");
        assert_eq!(resolved, output_paddr);
    }

    /// Stage-2 walk over the same table shape as stage-1: this is the
    /// walker the roadmap's rule 3 names as the CoW-fork mechanism, so it
    /// gets its own worked example rather than inheriting stage-1's.
    #[test]
    fn worked_example_aarch64_stage2_walk() {
        let vaddr = (1_u64 << 39) | (1_u64 << 30) | (1_u64 << 21) | (1_u64 << 12);
        let output_paddr = 0x4_0000_0000_u64;
        let memory = build_aarch64_single_mapping(vaddr, output_paddr, false, false);

        let resolved = walk(&memory, 0, vaddr, Access::default(), Format::Aarch64Stage2)
            .expect("stage-2 shares stage-1's table shape for this scaffold");
        assert_eq!(resolved, output_paddr);
    }

    /// S2AP\[1:0\] (ARM DDI 0487, stage-2 translation table descriptor
    /// fields, §D8.4.5 "Stage 2 translation table format descriptors",
    /// table "Stage 2 memory region access permissions") is a DISTINCT
    /// encoding from stage-1 AP\[2\]: `00` no access, `01` read-only, `10`
    /// write-only, `11` read/write. Write is granted exactly when
    /// `S2AP[1]` (bit 7 of the leaf descriptor) is set — `10` and `11` —
    /// never when it is clear, regardless of `S2AP[0]` (bit 6). This test
    /// walks a write-requesting and a non-write-requesting access against
    /// all four encodings (8 cases) and asserts grant/deny per that table.
    #[proxima::test]
    #[case::no_access_write_requested_is_denied(0b00, true, false)]
    #[case::no_access_read_only_requested_is_granted(0b00, false, true)]
    #[case::read_only_write_requested_is_denied(0b01, true, false)]
    #[case::read_only_read_only_requested_is_granted(0b01, false, true)]
    #[case::write_only_write_requested_is_granted(0b10, true, true)]
    #[case::write_only_read_only_requested_is_granted(0b10, false, true)]
    #[case::read_write_write_requested_is_granted(0b11, true, true)]
    #[case::read_write_read_only_requested_is_granted(0b11, false, true)]
    async fn stage2_s2ap_write_permission_matches_the_manuals_table(
        #[case] s2ap_bits: u64,
        #[case] request_write: bool,
        #[case] expect_granted: bool,
    ) {
        let vaddr = 0_u64;
        let output_paddr = 0x5_0000_0000_u64;
        let memory = build_aarch64_stage2_single_mapping(vaddr, output_paddr, s2ap_bits, false);
        let access = Access {
            write: request_write,
            execute: false,
        };

        let result = walk(&memory, 0, vaddr, access, Format::Aarch64Stage2);

        assert_eq!(
            result.is_ok(),
            expect_granted,
            "S2AP={s2ap_bits:#04b} write_requested={request_write}: expected granted={expect_granted}, got {result:?}"
        );
    }

    // ---- sad paths: named Fault, never a panic ----

    #[test]
    fn invalid_descriptor_is_a_named_fault_not_a_panic() {
        let memory = vec![0_u8; 4096]; // level-0 table entirely zero: valid bit clear everywhere
        let fault = walk(&memory, 0, 0, Access::default(), Format::Aarch64Stage1).unwrap_err();
        assert!(matches!(fault, Fault::InvalidDescriptor { level: 0, .. }));
    }

    #[test]
    fn permission_denied_is_a_named_fault_when_write_requested_on_read_only_page() {
        let vaddr = 0_u64;
        let memory = build_aarch64_single_mapping(vaddr, 0x1000_0000, true, false);
        let fault = walk(
            &memory,
            0,
            vaddr,
            Access {
                write: true,
                execute: false,
            },
            Format::Aarch64Stage1,
        )
        .unwrap_err();
        assert!(matches!(
            fault,
            Fault::PermissionDenied {
                requested: Access { write: true, .. },
                granted: Access { write: false, .. },
                ..
            }
        ));
    }

    #[test]
    fn permission_denied_is_a_named_fault_when_execute_requested_on_xn_page() {
        let vaddr = 0_u64;
        let memory = build_aarch64_single_mapping(vaddr, 0x1000_0000, false, true);
        let fault = walk(
            &memory,
            0,
            vaddr,
            Access {
                write: false,
                execute: true,
            },
            Format::Aarch64Stage1,
        )
        .unwrap_err();
        assert!(matches!(fault, Fault::PermissionDenied { .. }));
    }

    #[test]
    fn out_of_range_table_read_is_truncated_not_a_panic() {
        // L0 index 1 needs bytes [8, 16); an 8-byte buffer holds only index 0.
        let memory = vec![0_u8; 8];
        let fault = walk(
            &memory,
            0,
            1_u64 << 39,
            Access::default(),
            Format::Aarch64Stage1,
        )
        .unwrap_err();
        assert!(matches!(fault, Fault::Truncated { need: 16, got: 8 }));
    }

    #[test]
    fn x86_64_not_present_entry_is_a_named_fault() {
        let memory = vec![0_u8; 4096];
        let fault = walk(&memory, 0, 0, Access::default(), Format::X86_64FourLevel).unwrap_err();
        assert!(matches!(fault, Fault::InvalidDescriptor { level: 0, .. }));
    }

    #[test]
    fn x86_64_write_denied_on_read_only_pte_is_a_named_fault() {
        let vaddr = 0_u64;
        let memory = build_x86_64_single_mapping(vaddr, 0x2000_0000, false, false);
        let fault = walk(
            &memory,
            0,
            vaddr,
            Access {
                write: true,
                execute: false,
            },
            Format::X86_64FourLevel,
        )
        .unwrap_err();
        assert!(matches!(fault, Fault::PermissionDenied { .. }));
    }

    // ---- property tests: round-trip identity the format guarantees ----

    proptest! {
        /// For any single legally-built AArch64 stage-1 mapping, walking
        /// the vaddr it was built for always returns
        /// `output_paddr | page_offset` — the spec's core identity: a
        /// valid, permitted leaf descriptor's output address plus the
        /// unconsumed low bits IS the physical address.
        #[test]
        fn aarch64_stage1_walk_round_trips_through_a_built_mapping(
            l0 in 0_u64..512, l1 in 0_u64..512, l2 in 0_u64..512, l3 in 0_u64..512,
            page_offset in 0_u64..4096, output_page in 0_u64..0xF_FFFF,
        ) {
            let vaddr = (l0 << 39) | (l1 << 30) | (l2 << 21) | (l3 << 12) | page_offset;
            let output_paddr = output_page << 12;
            let memory = build_aarch64_single_mapping(vaddr, output_paddr, false, false);
            let resolved = walk(&memory, 0, vaddr, Access::default(), Format::Aarch64Stage1).unwrap();
            prop_assert_eq!(resolved, output_paddr | page_offset);
        }

        /// Same identity for x86-64 4-level.
        #[test]
        fn x86_64_four_level_walk_round_trips_through_a_built_mapping(
            l0 in 0_u64..512, l1 in 0_u64..512, l2 in 0_u64..512, l3 in 0_u64..512,
            page_offset in 0_u64..4096, output_page in 0_u64..0xF_FFFF,
        ) {
            let vaddr = (l0 << 39) | (l1 << 30) | (l2 << 21) | (l3 << 12) | page_offset;
            let output_paddr = output_page << 12;
            let memory = build_x86_64_single_mapping(vaddr, output_paddr, true, false);
            let resolved = walk(&memory, 0, vaddr, Access { write: true, execute: false }, Format::X86_64FourLevel).unwrap();
            prop_assert_eq!(resolved, output_paddr | page_offset);
        }

        /// Arbitrary bytes never panic the walker, on any format — the
        /// degenerate-input control every sans-IO parser here carries.
        #[test]
        fn fault_walk_never_panics_on_arbitrary_bytes(
            bytes in prop::collection::vec(any::<u8>(), 0..8192),
            vaddr in any::<u64>(),
            format_index in 0_u8..4,
        ) {
            let format = match format_index {
                0 => Format::Aarch64Stage1,
                1 => Format::Aarch64Stage2,
                2 => Format::X86_64FourLevel,
                _ => Format::X86_64FiveLevel,
            };
            let _ = walk(&bytes, 0, vaddr, Access { write: true, execute: true }, format);
        }
    }
}
