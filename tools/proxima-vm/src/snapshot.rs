//! M7 — snapshot (`tools/proxima-vm/ROADMAP.md`'s M7 section): serialize
//! vCPU registers, device state, and memory; restore into a fresh VM.
//!
//! Built on the scratch-guest path (`platform::run` in `crate`'s own
//! `ScratchVm`, the same guest program [`capture`] and [`restore`] drive
//! through the new `proxima_vm_scratch_snapshot`/`proxima_vm_scratch_restore`
//! FFI leaves) rather than the full ELF+virtio dispatch loop, because M7's
//! exit criterion is restore wall time and fault count, not a spec-complete
//! device model — device-state serialization is proven separately:
//! [`crate::virtio_console::ConsoleTransport`],
//! [`crate::virtio_net::NetTransport`], and [`crate::virtio_blk::BlkTransport`]
//! now derive `Clone`, so a device-state snapshot is `.clone()` and a
//! restore is substituting the clone in place of a freshly constructed
//! device — the direct payoff M6's `MmioDevice`/`RingCursor` state-machine
//! shape promised, not a new serialization path.
//!
//! [`VmSnapshot`] is the one new type this milestone adds — the section's own
//! design ("Serialize vCPU registers, device state, and memory") names a
//! bundle of captured state, which is data, not a transform: nothing pipes
//! through it. [`capture`] and [`restore`] are plain functions over that
//! data, the same shape [`crate::named_memory`]'s `create`/`map_shared_view`
//! already use for M4 — captured guest memory is `Vec<u8>` here rather than a
//! second [`crate::named_memory::GuestMemoryRegion`] view, because a
//! snapshot is inert bytes at rest, not a second live mapper of the source
//! VM's own backing object; [`restore`] allocates its own fresh named
//! region on the far side.
//!
//! # A fresh VM is a fresh process, on the HVF lane
//!
//! `hv_vm_create` is documented (and, empirically, enforced by hanging
//! rather than erroring on a second call) as once-per-process on
//! Hypervisor.framework: a process that has already created and destroyed
//! one `hv_vm` cannot create a second. So "restore into a fresh VM" is only
//! achievable across a process boundary on this lane — [`capture`] and
//! [`restore`] must run in separate `Command::new(..)` invocations, never
//! sequentially in the same process, which is exactly why [`VmSnapshot`]
//! round-trips through [`VmSnapshot::to_postcard_bytes`] /
//! [`VmSnapshot::from_postcard_bytes`] rather than only living in memory —
//! `tests/vm_snapshot.rs` drives `snapshot_capture_probe` and
//! `snapshot_restore_probe` (`src/bin/`) as two codesigned child processes
//! connected by exactly those bytes on disk. The KVM lane has no such
//! restriction (`KVM_CREATE_VM` is unlimited per process), but the two-
//! binary shape is kept uniform across both lanes rather than branching the
//! test harness by platform.
//!
//! # Tier
//!
//! Std-only (`tools/proxima-vm` is a std-tier host binary), same as
//! [`crate::named_memory`] and [`crate::dispatch::run_dispatch_loop`].

#![cfg(feature = "std")]

use proxima_core::ProximaError;
use serde::{Deserialize, Serialize};

/// FFI mirror of `proxima_vm_registers_t` (`ffi_segment.h`). Field order and
/// width must match the C struct exactly.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
struct RawVcpuRegisters {
    gpr: [u64; 31],
    pc: u64,
    flags: u64,
}

/// One vCPU's captured general-register file — `x0..x30`/`pc`/`cpsr` on the
/// HVF/aarch64 lane, `rax..r15`/`rip`/`rflags` on the KVM/x86_64 lane (see
/// `ffi_segment.h`'s own doc on `proxima_vm_registers_t` for the exact
/// per-lane field mapping). Never portable across lanes, only within one.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct VcpuRegisters {
    raw: RawVcpuRegisters,
}

/// Captured vCPU registers, guest memory, and the bytes the guest had
/// already emitted before the snapshot was taken — everything [`restore`]
/// needs to reproduce the exact guest state in a brand-new VM. Device state
/// (virtio transports) is not carried here: a live device snapshot is that
/// device's own `.clone()` (`ConsoleTransport`/`NetTransport`/`BlkTransport`
/// all derive `Clone` for this reason), composed by the caller alongside
/// this type rather than folded into it.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VmSnapshot {
    registers: VcpuRegisters,
    guest_memory: Vec<u8>,
    emitted: Vec<u8>,
}

/// Deterministic byte pattern the warm-restore size sweep uses to seed
/// [`VmSnapshot::with_padded_memory`]'s padding region and later verify
/// [`WarmVm::restore_oracle_full_copy`] actually copied it — a lazy no-op restore would still
/// re-trap correctly (the halting trap never touches padding past the
/// scratch guest's own tiny code blob), so the correctness gate checks
/// content, not only the trap.
#[must_use]
pub fn pattern_byte(offset: usize, seed: u64) -> u8 {
    let value = (offset as u64 ^ seed).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (value >> 56) as u8
}

/// The scratch guest's own halting sentinel (`TERMINAL_VALUE` in
/// `backend_macos.c`) — [`dirty_probe_guest_code`]'s emitted loop halts by
/// loading this value into `x0` and trapping via `hvc #0`, the identical
/// convention [`capture`]'s own guest uses.
const DIRTY_PROBE_TERMINAL: u16 = 256;

/// Word count of [`dirty_probe_guest_code`]'s emitted program — also the
/// minimum legal `data_offset` a caller may pass it, since the code itself
/// occupies `DIRTY_PROBE_CODE_WORDS * 4` bytes starting at guest address 0.
const DIRTY_PROBE_CODE_WORDS: usize = 13;

/// aarch64 `MOVZ <reg>, #imm16, LSL #(16*shift_word)` (64-bit form) —
/// `shift_word` is `hw` in ARM DDI 0487's own field name, 0..=3. Verified
/// against `backend_macos.c`'s own hand-encoded `code[0] = 0xd2800541u; /*
/// movz x1, #0x2a */`: `movz(1, 0x2a, 0) == 0xd280_0541`, this module's own
/// `dirty_probe_guest_code_matches_hand_encoded_movz_reference` test.
fn movz(reg: u8, imm16: u16, shift_word: u8) -> u32 {
    0xD280_0000_u32 | (u32::from(shift_word) << 21) | (u32::from(imm16) << 5) | u32::from(reg)
}

/// aarch64 `MOVK <reg>, #imm16, LSL #(16*shift_word)` (64-bit form) — the
/// same opcode family as [`movz`] with `opc` widened from `10` to `11`
/// (ARM DDI 0487's own encoding table), which is exactly the `1 << 29` bit
/// difference between `0xD280_0000` and `0xF280_0000` below.
fn movk(reg: u8, imm16: u16, shift_word: u8) -> u32 {
    0xF280_0000_u32 | (u32::from(shift_word) << 21) | (u32::from(imm16) << 5) | u32::from(reg)
}

/// aarch64 `MOVZ <32-bit reg>, #imm16` — [`movz`]'s 32-bit sibling (`sf = 0`
/// instead of `1`, ARM DDI 0487), used once to load the dirty-probe write
/// byte into `w0`.
fn movz32(reg: u8, imm16: u16) -> u32 {
    0x5280_0000_u32 | (u32::from(imm16) << 5) | u32::from(reg)
}

/// aarch64 `STRB <Wt>, [<Xn>]` (unsigned immediate offset, `imm12 = 0`).
fn strb_unsigned_offset(transfer_reg: u8, base_reg: u8) -> u32 {
    0x3900_0000_u32 | (u32::from(base_reg) << 5) | u32::from(transfer_reg)
}

/// aarch64 `ADD <Xd>, <Xn>, <Xm>` (shifted register, no shift).
fn add_register(dest_reg: u8, base_reg: u8, offset_reg: u8) -> u32 {
    0x8B00_0000_u32 | (u32::from(offset_reg) << 16) | (u32::from(base_reg) << 5) | u32::from(dest_reg)
}

/// aarch64 `SUBS <Xd>, <Xn>, #imm12` (sets flags — [`dirty_probe_guest_code`]'s
/// loop counter decrement, feeding the `CBNZ` [`cbnz`] emits next).
fn subs_immediate(dest_reg: u8, base_reg: u8, imm12: u16) -> u32 {
    0xF100_0000_u32 | (u32::from(imm12) << 10) | (u32::from(base_reg) << 5) | u32::from(dest_reg)
}

/// aarch64 `CBNZ <Xt>, #<word_offset>` — `word_offset` is signed, relative
/// to this instruction's own address, in units of one instruction (4
/// bytes), matching the architected `imm19` field exactly.
fn cbnz(transfer_reg: u8, word_offset: i32) -> u32 {
    let encoded = (word_offset & 0x0007_ffff) as u32;
    0xB500_0000_u32 | (encoded << 5) | u32::from(transfer_reg)
}

/// Builds the 13-instruction aarch64 program [`WarmVm::dirty_write_run`]
/// resumes the vCPU into: write `byte_value` to `page_count` guest
/// addresses starting at `data_offset` and striding `stride` bytes between
/// writes (one write per host page, when `stride` equals
/// [`WarmVm::dirty_page_granule`]), then halt via the same `hvc #0` terminal
/// trap [`capture`]'s own guest uses. Registers `x1` (address accumulator),
/// `x2` (stride), `x3` (remaining count), and `w0` (byte value, then the
/// terminal sentinel) are scratch — [`WarmVm::dirty_write_run`] resets every
/// gpr before resuming, so no caller-visible register survives a previous
/// run.
///
/// Pure data, no platform call — [`dirty_probe_snapshot`] embeds this
/// program's bytes directly into a [`VmSnapshot`]'s `guest_memory` rather
/// than have the C trampoline synthesize it at run time: an earlier design
/// synthesized guest code from the C side on every call, which is a
/// HOST-side write to the mapped region and therefore never trips the
/// EC-0x24 fault path [`WarmVm::dirty_write_run`] tracks dirty pages
/// through — it would silently corrupt the byte-identical-twin oracle at
/// the code page (never recorded dirty, yet mutated). Embedding the program
/// as ordinary snapshot bytes means [`WarmVm::restore`]/[`WarmVm::restore_dirty`]
/// carry it exactly the same way they carry every other guest-memory byte.
fn dirty_probe_guest_code(data_offset: u64, stride: u16, page_count: u16, byte_value: u8) -> [u32; DIRTY_PROBE_CODE_WORDS] {
    [
        movz(1, (data_offset & 0xffff) as u16, 0),
        movk(1, ((data_offset >> 16) & 0xffff) as u16, 1),
        movk(1, ((data_offset >> 32) & 0xffff) as u16, 2),
        movk(1, ((data_offset >> 48) & 0xffff) as u16, 3),
        movz(2, stride, 0),
        movz(3, page_count, 0),
        movz32(0, u16::from(byte_value)),
        strb_unsigned_offset(0, 1),
        add_register(1, 1, 2),
        subs_immediate(3, 3, 1),
        cbnz(3, -3),
        movz(0, DIRTY_PROBE_TERMINAL, 0),
        0xD400_0002_u32, // hvc #0
    ]
}

/// Builds a [`VmSnapshot`] whose `guest_memory` is `dirty_probe_guest_code`'s
/// program at offset 0, followed by [`pattern_byte`]-seeded padding from
/// `data_offset` to `target_size` — the µsec-campaign slice 4 dirty-tracking
/// measurement's own fixture: [`WarmVm::restore_oracle_full_copy`] establishes this as the
/// clean baseline, then repeated
/// [`WarmVm::run_dirty_write`] calls dirty exactly `page_count` of its pages,
/// and [`WarmVm::restore_layered`] must reproduce it byte-for-byte from only
/// those dirtied pages — the strong oracle a plain sampled-padding check
/// (`with_padded_memory`'s own doc) cannot catch a missed dirty page with.
///
/// The registers embedded in the returned snapshot are all-zero:
/// [`WarmVm::run_dirty_write`] resets the vCPU's own register file before
/// every run, so this snapshot's `registers` field only matters as the
/// value [`WarmVm::restore_oracle_full_copy`]/[`WarmVm::restore_layered`] write back — zero is
/// as good as any other fixed value for that.
///
/// # Panics
///
/// Panics if `data_offset` is smaller than the code program's own byte
/// length, or if the highest touched address (`data_offset + stride *
/// (page_count - 1)`) does not fit within `target_size`.
#[must_use]
pub fn dirty_probe_snapshot(
    target_size: usize,
    data_offset: usize,
    stride: u16,
    page_count: u16,
    byte_value: u8,
    seed: u64,
) -> VmSnapshot {
    let code_bytes = DIRTY_PROBE_CODE_WORDS * 4;
    assert!(
        data_offset >= code_bytes,
        "data_offset {data_offset} overlaps the {code_bytes}-byte dirty-probe guest code"
    );
    let highest_touched = data_offset + usize::from(stride) * usize::from(page_count.saturating_sub(1));
    assert!(
        highest_touched < target_size,
        "highest touched address {highest_touched} does not fit within target_size {target_size}"
    );

    let mut guest_memory = vec![0_u8; target_size];
    let code = dirty_probe_guest_code(data_offset as u64, stride, page_count, byte_value);
    for (index, word) in code.iter().enumerate() {
        guest_memory[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
    }
    for (offset, byte) in guest_memory.iter_mut().enumerate().skip(data_offset) {
        *byte = pattern_byte(offset, seed);
    }

    VmSnapshot { registers: VcpuRegisters::default(), guest_memory, emitted: Vec::new() }
}

impl VmSnapshot {
    /// The captured register file.
    #[must_use]
    pub fn registers(&self) -> VcpuRegisters {
        self.registers
    }

    /// Returns a copy of this snapshot with `guest_memory` grown to
    /// `target_size` bytes, every byte beyond the original length filled
    /// with [`pattern_byte`] seeded by `seed` — the µsec-campaign warm-restore
    /// size sweep (`tools/proxima-vm/BENCH_LOG.md` slice 2) needs to scale
    /// the memcpy term [`WarmVm::restore_oracle_full_copy`] pays independent of the scratch
    /// guest's own tiny code blob, since the guest never executes the
    /// padding it resumes past.
    ///
    /// # Panics
    ///
    /// Panics if `target_size` is smaller than this snapshot's own
    /// `guest_memory` length — shrinking would truncate the code blob the
    /// captured register file's `pc` points into.
    #[must_use]
    pub fn with_padded_memory(&self, target_size: usize, seed: u64) -> VmSnapshot {
        let original_length = self.guest_memory.len();
        assert!(
            target_size >= original_length,
            "target_size {target_size} is smaller than the existing guest_memory length {original_length}"
        );
        let mut guest_memory = self.guest_memory.clone();
        guest_memory.resize(target_size, 0);
        for (offset, byte) in guest_memory.iter_mut().enumerate().skip(original_length) {
            *byte = pattern_byte(offset, seed);
        }
        VmSnapshot {
            registers: self.registers,
            guest_memory,
            emitted: self.emitted.clone(),
        }
    }

    /// The captured guest-memory bytes, page-rounded to the size the
    /// scratch guest's code blob mapped.
    #[must_use]
    pub fn guest_memory(&self) -> &[u8] {
        &self.guest_memory
    }

    /// The bytes the guest had emitted before halting into this snapshot.
    #[must_use]
    pub fn emitted(&self) -> &[u8] {
        &self.emitted
    }

    /// Postcard-encode this snapshot — the workspace's standing
    /// serialization discipline (`ChildRequest`/`ChildResponse` round-trip
    /// the same way, `src/dispatch.rs`'s own tests), used here to carry a
    /// snapshot across the process boundary [`restore`]'s module doc
    /// explains is mandatory on the HVF lane.
    ///
    /// # Errors
    ///
    /// Returns [`ProximaError::Upstream`] naming the encode failure.
    pub fn to_postcard_bytes(&self) -> Result<Vec<u8>, ProximaError> {
        postcard::to_allocvec(self).map_err(|error| ProximaError::Upstream(error.to_string()))
    }

    /// Decode a snapshot produced by [`VmSnapshot::to_postcard_bytes`].
    ///
    /// # Errors
    ///
    /// Returns [`ProximaError::Upstream`] naming the decode failure.
    pub fn from_postcard_bytes(bytes: &[u8]) -> Result<Self, ProximaError> {
        postcard::from_bytes(bytes).map_err(|error| ProximaError::Upstream(error.to_string()))
    }
}

/// Wall time and fault-count evidence [`restore`] measured while rebuilding
/// a fresh VM from a [`VmSnapshot`] — the M7 exit criterion's own numbers,
/// named exactly as `tools/proxima-vm/ROADMAP.md`'s M7 section and the M3
/// instrument it points at do: restore wall time and fault count at the
/// `page_size` stride [`restore`] was called with.
#[derive(Clone, Copy, Debug)]
pub struct RestoreReport {
    /// Total wall time to rebuild the fresh VM: named-region creation, the
    /// `page_size`-strided memory copy, vCPU creation, and register
    /// restoration.
    pub restore_wall_nanos: u64,
    /// Wall time for the `page_size`-strided memory copy alone — the same
    /// axis `proxima_vm_run_dispatch_loop`'s own M3 `touch_all_pages_nanos`
    /// measures, mirrored here over restored bytes instead of zeros.
    pub touch_all_pages_nanos: u64,
    /// Count of data-abort/`KVM_EXIT_MMIO` exits observed during the one
    /// resumed step — legitimately 0 for this guest, which touches no mmio
    /// window; the same auxiliary-count caveat
    /// `proxima_vm_run_dispatch_loop`'s own `mmio_trap_count` carries.
    pub fault_count: u64,
    /// The `x0` value the resumed vCPU's re-trapped exit read back.
    pub resumed_x0: u64,
    /// Whether the resumed step re-trapped at all — `false` means the
    /// restored register file did not reproduce a halting trap, and
    /// `resumed_x0` is meaningless.
    pub resumed_matched_trap: bool,
    /// Per-phase breakdown of `restore_wall_nanos` (µsec campaign).
    pub phases: RestorePhases,
}

/// Per-phase breakdown of one [`restore`] call's own [`RestoreReport::restore_wall_nanos`]
/// (µsec campaign, restore-path decomposition) plus the resumed re-trap
/// [`RestoreReport`] already timed separately as `touch_all_pages_nanos`'s
/// sibling. Every field is a phase of the cold path only — [`WarmVm::restore_oracle_full_copy`]
/// reports its own, smaller [`WarmRestorePhases`] with the creation phases
/// removed, since a warm restore never runs them.
#[derive(Clone, Copy, Debug, Default)]
pub struct RestorePhases {
    /// `proxima_vm_create_named_region` — the M4 named-region object.
    pub region_create_nanos: u64,
    /// `hv_vm_create` / `KVM_CREATE_VM`.
    pub vm_create_nanos: u64,
    /// `hv_vm_map` / `KVM_SET_USER_MEMORY_REGION`.
    pub vm_map_nanos: u64,
    /// `hv_vcpu_create` / `KVM_CREATE_VCPU`.
    pub vcpu_create_nanos: u64,
    /// Writing the captured register file into the fresh vCPU.
    pub register_restore_nanos: u64,
    /// The one resumed `hv_vcpu_run` / `KVM_RUN` step that re-traps.
    pub first_retrap_nanos: u64,
}

/// The same six-phase breakdown [`RestorePhases`] carries, minus the three
/// creation phases a warm restore never runs (region/vm/vcpu creation) —
/// [`WarmVm::restore_oracle_full_copy`]'s own doc explains why they are gone, not zeroed.
#[derive(Clone, Copy, Debug, Default)]
pub struct WarmRestorePhases {
    /// Writing the captured register file into the reused vCPU.
    pub register_restore_nanos: u64,
    /// The one resumed `hv_vcpu_run` / `KVM_RUN` step that re-traps.
    pub first_retrap_nanos: u64,
}

/// Run `message` through the scratch guest to its halting trap and capture
/// the vCPU register file plus the full guest-memory contents before
/// tearing the VM down.
///
/// # Errors
///
/// Returns [`ProximaError::Upstream`] naming the failing platform call, or
/// [`ProximaError::Config`] on platforms with no scratch-guest lane.
pub fn capture(message: &[u8]) -> Result<VmSnapshot, ProximaError> {
    platform::capture(message)
}

/// Restore `snapshot` into a brand-new named guest-memory region and a
/// brand-new vCPU — never the ones [`capture`] used — copying the memory in
/// `page_size`-strided chunks, then resume the vCPU exactly once.
///
/// Because the snapshot was captured at the guest's own halting trap (whose
/// faulting instruction had not yet retired), the one resumed step re-traps
/// at the identical instruction: [`RestoreReport::resumed_matched_trap`] is
/// `true` and [`RestoreReport::resumed_x0`] reads back the same value the
/// guest emitted right before the snapshot, iff restore reproduced the
/// exact guest state.
///
/// # Errors
///
/// Returns [`ProximaError::Upstream`] naming the failing platform call, or
/// [`ProximaError::Config`] on platforms with no scratch-guest lane.
pub fn restore(snapshot: &VmSnapshot, page_size: usize) -> Result<RestoreReport, ProximaError> {
    platform::restore(snapshot, page_size)
}

/// This host's HVF stage-2 mapping granule (`getpagesize()`, 16KiB on Apple
/// silicon — M3's own measured value, `tools/proxima-vm/BENCH_LOG.md`). Every
/// [`WarmVm`] layered call indexes its dirty bitmap and per-page remaps by
/// this granule; a caller building a dirty-write guest program
/// ([`dirty_probe_snapshot`]) should stride by exactly this value to dirty
/// one page per write.
#[must_use]
pub fn host_page_size() -> usize {
    platform::host_page_size()
}

/// A fresh, unnamed, zero-filled named memory region a [`WarmVm`]'s layered
/// base maps read-only+exec — the owner's own design (slot-0 vault,
/// 2026-08-11 execution-boundary note): "Map the image read-only shared...
/// snapshot stops being an operation and restore becomes a mapping, not a
/// copy." Composes [`crate::named_memory::GuestMemoryRegion`] directly (M4's
/// own named-region primitive) rather than a bespoke allocation — the only
/// thing this type adds is the "never written after
/// [`WarmVm::adopt_base`]" discipline the design depends on, which
/// `GuestMemoryRegion` itself has no opinion about.
///
/// [`LayeredBase::share`] is the M4 exit criterion's own second-mapper case
/// applied to this design: two [`WarmVm`]s (necessarily two vCPUs inside
/// this process's ONE `hv_vm` on the HVF lane — see [`WarmVm::new_layered_over`]'s
/// own doc on why) can each construct a layered context over the SAME base
/// object, one owning it, the other holding a [`LayeredBaseView`].
pub struct LayeredBase {
    region: crate::named_memory::GuestMemoryRegion,
}

impl LayeredBase {
    /// Reserves `capacity` bytes for the base's memory image. Write the
    /// image with [`WarmVm::adopt_base`] before constructing any [`WarmVm`]
    /// over it — a base's content is establish-once, per this design's own
    /// "never written after creation" invariant.
    ///
    /// # Errors
    ///
    /// Returns [`ProximaError::Upstream`] naming the failing platform call.
    pub fn new(capacity: usize) -> Result<Self, ProximaError> {
        Ok(Self { region: crate::named_memory::GuestMemoryRegion::create(capacity)? })
    }

    /// Number of bytes this base reserves.
    #[must_use]
    pub fn len(&self) -> usize {
        self.region.len()
    }

    /// Whether this base reserves zero bytes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.region.is_empty()
    }

    /// Reads `length` bytes at `offset` directly from this base's own host
    /// mapping — a plain host-side read, valid regardless of which guest IPA
    /// range any [`WarmVm`] currently has this base mapped into (or whether
    /// any does at all). The correctness oracle for the whole design: this
    /// base's bytes never change after [`WarmVm::adopt_base`] wrote them, so
    /// this is what "restore reproduced the base" is checked against.
    #[must_use]
    pub fn read(&self, offset: usize, length: usize) -> Vec<u8> {
        self.region.primary_slice()[offset..offset + length].to_vec()
    }

    /// Maps a second, independent host view of this base's named object
    /// (`GuestMemoryRegion::map_shared_view`) — the handle
    /// [`WarmVm::new_layered_over`] maps into a second vCPU's own guest IPA
    /// range, proving two [`WarmVm`]s can share one base.
    ///
    /// # Errors
    ///
    /// Returns [`ProximaError::Upstream`] naming the failing platform call.
    pub fn share(&self) -> Result<LayeredBaseView, ProximaError> {
        Ok(LayeredBaseView { view: self.region.map_shared_view()? })
    }
}

/// A second mapper of a [`LayeredBase`]'s backing object, from
/// [`LayeredBase::share`] — never written directly; a sharing [`WarmVm`]
/// only ever reads through it (mapping it read-only+exec into its own guest
/// IPA range) or [`WarmVm::layered_base_bytes`] reads it back host-side, the
/// same "never mutated" discipline [`LayeredBase`] itself carries.
pub struct LayeredBaseView {
    view: crate::named_memory::RegionView,
}

impl LayeredBaseView {
    /// Reads `length` bytes at `offset` directly from this view's own host
    /// mapping — identical in shape to [`LayeredBase::read`], since both
    /// name views of the same, never-mutated backing object.
    #[must_use]
    pub fn read(&self, offset: usize, length: usize) -> Vec<u8> {
        self.view.as_slice()[offset..offset + length].to_vec()
    }
}

/// Evidence one [`WarmVm::adopt_base`]/[`WarmVm::adopt_shared_base`] call
/// measured — the layered design's own "establish the base" step, paid
/// once per base, never repeated by [`WarmVm::restore_layered`].
#[derive(Clone, Copy, Debug)]
pub struct LayeredAdoptReport {
    /// Wall time for the `hv_vm_map` call that maps the whole base
    /// read-only+exec into guest IPA (plus an `hv_vm_unmap` first, if a
    /// prior adopt/run/restore cycle already had this range mapped).
    pub map_nanos: u64,
    /// Wall time to reset every gpr/`pc`/`cpsr` to this design's fixed
    /// entry state.
    pub register_reset_nanos: u64,
}

/// Evidence one [`WarmVm::run_dirty_write`] call measured — the design's
/// "Run" step: every write fault this call services costs exactly one
/// granule-sized host `memcpy` plus one `hv_vm_unmap`/`hv_vm_map` pair, the
/// per-fault round-trip cost the µsec campaign's slice 3 never isolated
/// (its own residual: "The exit-handling round-trip cost for a
/// GUEST-INITIATED write fault... was never measured").
#[derive(Clone, Copy, Debug)]
pub struct DirtyRunReport {
    /// Wall time for the whole resumed run, first resume to halt.
    pub run_wall_nanos: u64,
    /// Count of EC-0x24 write-fault exits serviced (new pages plus any
    /// already-dirty re-fault, though the design does not expect the
    /// latter).
    pub fault_count: u64,
    /// Count of pages newly marked dirty (remapped base -> delta) by this
    /// call — the `K` in "K-page run wall vs unprotected run / K".
    pub newly_dirty_page_count: u64,
    /// Whether the guest halted via its own terminal sentinel rather than
    /// exhausting the fault budget.
    pub halted_ok: bool,
}

/// Evidence one [`WarmVm::restore_layered`] call measured — the design's
/// own "Restore" step, reported separately from [`WarmRestoreReport`]
/// (the full-copy oracle's own report shape) because this restore has no
/// memory-copy phase to report at all, only a remap phase — the literal
/// "restore becomes a mapping, not a copy" the owner's design note names.
#[derive(Clone, Copy, Debug)]
pub struct LayeredRestoreReport {
    /// Wall time for the whole call: remap plus register reset.
    pub restore_wall_nanos: u64,
    /// Wall time for the coalesced `hv_vm_unmap`/`hv_vm_map` remap pairs
    /// alone — the axis this design claims scales with dirty page COUNT,
    /// not with base size or copied bytes.
    pub remap_nanos: u64,
    /// Wall time to reset every gpr/`pc`/`cpsr`.
    pub register_reset_nanos: u64,
    /// Count of pages this call remapped back to the base (coalesced runs
    /// summed page-by-page, not call-by-call) — 0 for a restore with
    /// nothing dirty.
    pub remapped_page_count: u64,
}

/// Wall time and fault-count evidence one [`WarmVm::restore_oracle_full_copy`] call
/// measured — [`RestoreReport`]'s warm-path mirror, minus the fields a warm
/// restore has nothing to report (there is no region/vm/vcpu creation
/// phase to roll into a `restore_wall_nanos` total the same way).
#[derive(Clone, Copy, Debug)]
pub struct WarmRestoreReport {
    /// Wall time for register restore plus the memory copy — everything
    /// this restore paid before resuming the vCPU.
    pub restore_wall_nanos: u64,
    /// Wall time for the `page_size`-strided memory copy alone.
    pub touch_all_pages_nanos: u64,
    /// Count of data-abort/`KVM_EXIT_MMIO` exits observed during the one
    /// resumed step.
    pub fault_count: u64,
    /// The `x0` value the resumed vCPU's re-trapped exit read back.
    pub resumed_x0: u64,
    /// Whether the resumed step re-trapped at all.
    pub resumed_matched_trap: bool,
    /// Per-phase breakdown (µsec campaign).
    pub phases: WarmRestorePhases,
}

/// A live vm/vcpu/mapped-region triple held open across repeated restores —
/// the µsec campaign's first slice (`tools/proxima-vm/ROADMAP.md`): cold
/// [`restore`] recreates all three every call, and `hv_vm_create` hangs on a
/// second call in one process (this module's own "A fresh VM is a fresh
/// process" doc above), so no existing primitive in this module can express
/// "restore again without recreating" — [`restore`] itself cannot be called
/// twice in one process on the HVF lane at all. `WarmVm` is the type that
/// capability needs: construct once via [`WarmVm::new`], call
/// [`WarmVm::restore_oracle_full_copy`] any number of times, drop to tear the context down.
///
/// # Tier
///
/// Std-only, same as the rest of this module — this type's own constructor
/// and destructor cross the same `extern "C"` boundary [`capture`]/[`restore`]
/// do.
pub struct WarmVm {
    inner: Option<platform::WarmVmHandle>,
    layered: Option<platform::LayeredHandle>,
}

impl WarmVm {
    /// Creates the named region, vm, and vcpu once. `guest_memory_capacity`
    /// bounds every later [`WarmVm::restore_oracle_full_copy`] call's
    /// `snapshot.guest_memory()` length against this context's own mapped
    /// region size.
    ///
    /// # Errors
    ///
    /// Returns [`ProximaError::Upstream`] naming the failing platform call, or
    /// [`ProximaError::Config`] on platforms with no scratch-guest lane.
    pub fn new(guest_memory_capacity: usize) -> Result<Self, ProximaError> {
        Ok(Self {
            inner: Some(platform::WarmVmHandle::new(guest_memory_capacity)?),
            layered: None,
        })
    }

    /// The full-copy restore [`WarmVm`] carried before the layered rework —
    /// kept ONLY as the correctness oracle every layered test compares
    /// against (`tests/vm_snapshot.rs`'s byte-identical-twin cases), never
    /// as a design this crate still recommends: resets the live vCPU's
    /// registers, `memcpy`s `snapshot`'s ENTIRE guest-memory image into the
    /// already-mapped region every single call, then resumes once. This is
    /// the µsec campaign's own slice-2 finding in one call — cost scales
    /// with the full snapshot size, not with what actually changed
    /// (`BENCH_LOG.md`'s own 331µs @ 16MiB / 5.7ms @ 256MiB rows).
    ///
    /// # Errors
    ///
    /// Returns [`ProximaError::Upstream`] naming the failing platform call.
    pub fn restore_oracle_full_copy(
        &mut self,
        snapshot: &VmSnapshot,
        page_size: usize,
    ) -> Result<WarmRestoreReport, ProximaError> {
        self.inner
            .as_mut()
            .ok_or_else(|| ProximaError::Config("this WarmVm was not constructed via WarmVm::new".into()))?
            .restore(snapshot, page_size)
    }

    /// Reads back `length` bytes at `offset` from the oracle region this
    /// handle keeps mapped — already host-addressable in this process
    /// ([`WarmVm::new`]'s own doc on the mapping), so this is a plain read,
    /// not a new platform call. The µsec-campaign size sweep's correctness
    /// gate: [`WarmRestoreReport::resumed_matched_trap`]/`resumed_x0` only
    /// prove the code blob's own few bytes and register file restored
    /// correctly, never the padding [`VmSnapshot::with_padded_memory`] adds.
    #[must_use]
    pub fn sample_guest_memory(&self, offset: usize, length: usize) -> Vec<u8> {
        self.inner
            .as_ref()
            .map(|handle| handle.sample_guest_memory(offset, length))
            .unwrap_or_default()
    }

    /// Constructs the layered design's own vCPU over a [`LayeredBase`] this
    /// [`WarmVm`] owns outright — the common single-VM path every bench and
    /// correctness test below uses. `delta_capacity` sizes the per-VM delta
    /// region (worst case: every page of `base` dirtied at once).
    ///
    /// # Errors
    ///
    /// Returns [`ProximaError::Upstream`] naming the failing platform call, or
    /// [`ProximaError::Config`] on platforms with no scratch-guest lane.
    pub fn new_layered(base: LayeredBase, delta_capacity: usize) -> Result<Self, ProximaError> {
        Ok(Self {
            inner: None,
            layered: Some(platform::LayeredHandle::new_owned(base, delta_capacity, 0)?),
        })
    }

    /// Constructs the layered design's own vCPU over a SECOND, independent
    /// mapper ([`LayeredBaseView`], from [`LayeredBase::share`]) of an
    /// EXISTING base another [`WarmVm`] may already be using — the sharing
    /// proof this design's own M4 lineage promised ("two VMs map the same
    /// region and observe each other's writes"). `ipa_base` must not
    /// overlap any other layered [`WarmVm`] sharing this process's one
    /// `hv_vm`: HVF creates exactly one `hv_vm` per process (this module's
    /// own "A fresh VM is a fresh process" doc), so two concurrent
    /// [`WarmVm`]s in one process are necessarily two vCPUs inside that one
    /// `hv_vm`'s single, flat stage-2 IPA space, not two separate `hv_vm`s —
    /// disjoint `ipa_base` ranges are what keeps their guest-memory views
    /// from colliding, the same way two real guests never share a physical
    /// address range on real hardware.
    ///
    /// # Errors
    ///
    /// Returns [`ProximaError::Upstream`] naming the failing platform call, or
    /// [`ProximaError::Config`] on platforms with no scratch-guest lane.
    pub fn new_layered_over(
        base_view: LayeredBaseView,
        delta_capacity: usize,
        ipa_base: u64,
    ) -> Result<Self, ProximaError> {
        Ok(Self {
            inner: None,
            layered: Some(platform::LayeredHandle::new_shared(base_view, delta_capacity, ipa_base)?),
        })
    }

    fn layered_mut(&mut self) -> Result<&mut platform::LayeredHandle, ProximaError> {
        self.layered
            .as_mut()
            .ok_or_else(|| ProximaError::Config("this WarmVm was not constructed via WarmVm::new_layered(_over)".into()))
    }

    /// Writes `guest_memory` into this [`WarmVm`]'s OWN [`LayeredBase`] (the
    /// design's one full-copy cost, paid exactly once) and maps it
    /// read-only+exec into guest IPA. Errors if this [`WarmVm`] was
    /// constructed via [`WarmVm::new_layered_over`] — a sharer never writes
    /// the base it does not own; see [`WarmVm::adopt_shared_base`].
    ///
    /// # Errors
    ///
    /// Returns [`ProximaError::Config`] if this [`WarmVm`] shares its base, or
    /// [`ProximaError::Upstream`] naming the failing platform call.
    pub fn adopt_base(&mut self, guest_memory: &[u8]) -> Result<LayeredAdoptReport, ProximaError> {
        self.layered_mut()?.write_base_and_adopt(guest_memory)
    }

    /// Maps an already-populated, shared base read-only+exec into this
    /// [`WarmVm`]'s own guest IPA range — the sharer's half of
    /// [`WarmVm::adopt_base`], never writing a byte.
    ///
    /// # Errors
    ///
    /// Returns [`ProximaError::Upstream`] naming the failing platform call.
    pub fn adopt_shared_base(&mut self) -> Result<LayeredAdoptReport, ProximaError> {
        self.layered_mut()?.adopt()
    }

    /// Maps a second, independent view of this [`WarmVm`]'s OWN base
    /// ([`LayeredBase::share`]) — the handle [`WarmVm::new_layered_over`]
    /// needs to construct a second, sharing [`WarmVm`]. Errors if this
    /// [`WarmVm`] does not own its base (it is itself already a sharer).
    ///
    /// # Errors
    ///
    /// Returns [`ProximaError::Config`] if this [`WarmVm`] does not own its
    /// base, or [`ProximaError::Upstream`] naming the failing platform call.
    pub fn layered_base_view(&self) -> Result<LayeredBaseView, ProximaError> {
        self.layered
            .as_ref()
            .ok_or_else(|| ProximaError::Config("this WarmVm was not constructed via WarmVm::new_layered(_over)".into()))?
            .share_base()
    }

    /// Resumes the vCPU, servicing every write fault by remapping exactly
    /// the one faulting page base -> delta (module doc). Runs until the
    /// scratch guest halts or `expected_page_count + 64` faults fire.
    ///
    /// # Errors
    ///
    /// Returns [`ProximaError::Upstream`] naming the failing platform call.
    pub fn run_dirty_write(&mut self, expected_page_count: u64) -> Result<DirtyRunReport, ProximaError> {
        self.layered_mut()?.run(expected_page_count)
    }

    /// The layered design's own restore: unmaps every delta-mapped IPA and
    /// remaps it back to the base, read-only. Never copies a guest-memory
    /// byte.
    ///
    /// # Errors
    ///
    /// Returns [`ProximaError::Upstream`] naming the failing platform call.
    pub fn restore_layered(&mut self) -> Result<LayeredRestoreReport, ProximaError> {
        self.layered_mut()?.restore()
    }

    /// Reads `length` bytes at `offset` directly from this [`WarmVm`]'s own
    /// base — [`LayeredBase::read`]/[`LayeredBaseView::read`] under the
    /// hood, so this always reads the base's original, never-mutated
    /// content regardless of which pages are currently delta-mapped.
    #[must_use]
    pub fn layered_base_bytes(&self, offset: usize, length: usize) -> Vec<u8> {
        self.layered
            .as_ref()
            .map(|handle| handle.base_bytes(offset, length))
            .unwrap_or_default()
    }

    /// Reads `length` bytes at `offset` directly from this [`WarmVm`]'s own,
    /// private delta region — never shared with any other [`WarmVm`], even
    /// one sharing this instance's base ([`WarmVm::new_layered_over`]'s own
    /// doc). The sharing proof's own check: a write through one [`WarmVm`]'s
    /// delta must never appear in another's.
    #[must_use]
    pub fn layered_delta_bytes(&self, offset: usize, length: usize) -> Vec<u8> {
        self.layered
            .as_ref()
            .map(|handle| handle.delta_bytes(offset, length))
            .unwrap_or_default()
    }
}

#[cfg(any(
    all(target_os = "linux", target_arch = "x86_64"),
    all(target_os = "macos", target_arch = "aarch64")
))]
mod platform {
    use std::ffi::CStr;
    use std::os::raw::{c_char, c_void};

    use proxima_core::ProximaError;

    use super::{
        DirtyRunReport, LayeredAdoptReport, LayeredBase, LayeredBaseView, LayeredRestoreReport, RawVcpuRegisters,
        RestorePhases, RestoreReport, VcpuRegisters, VmSnapshot, WarmRestorePhases, WarmRestoreReport,
    };

    const ERROR_CAPACITY: usize = 512;
    const OUTPUT_CAPACITY: usize = 4096;

    /// FFI mirror of `proxima_vm_named_region_t` (`ffi_segment.h`). Field
    /// order and width must match the C struct exactly.
    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct RawNamedRegion {
        handle: i32,
        primary_address: *mut c_void,
        mapped_size: usize,
    }

    /// FFI mirror of `proxima_vm_warm_restore_context_t`
    /// (`dispatch_trampoline.h`). Field order and width must match the C
    /// struct exactly. Opaque to this module beyond its own construction —
    /// every field after `new` only ever round-trips through the three
    /// warm-restore trampolines, never read directly.
    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct RawWarmRestoreContext {
        region: RawNamedRegion,
        guest_memory_capacity: usize,
        vcpu: u64,
        exit_data: *mut c_void,
    }

    /// FFI mirror of `proxima_vm_layered_context_t` (`dispatch_trampoline.h`).
    /// Field order and width must match the C struct exactly. Opaque to this
    /// module beyond its own construction — every field after
    /// [`LayeredHandle::new_owned`]/[`LayeredHandle::new_shared`] only ever
    /// round-trips through the four layered trampolines, never read
    /// directly.
    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct RawLayeredContext {
        base_host_address: *mut c_void,
        base_size: usize,
        delta_host_address: *mut c_void,
        delta_size: usize,
        ipa_base: u64,
        vcpu: u64,
        exit_data: *mut c_void,
        mapped: i32,
    }

    unsafe extern "C" {
        fn proxima_vm_scratch_guest_memory_size(message_length: usize) -> usize;

        fn proxima_vm_scratch_snapshot(
            message: *const u8,
            message_length: usize,
            output: *mut u8,
            output_capacity: usize,
            registers_out: *mut RawVcpuRegisters,
            guest_memory_out: *mut u8,
            guest_memory_capacity: usize,
            error_buffer: *mut c_char,
            error_capacity: usize,
        ) -> i32;

        fn proxima_vm_scratch_restore(
            registers_in: *const RawVcpuRegisters,
            guest_memory_in: *const u8,
            guest_memory_length: usize,
            page_size: usize,
            restore_wall_nanos_out: *mut u64,
            touch_all_pages_nanos_out: *mut u64,
            fault_count_out: *mut u64,
            resumed_x0_out: *mut u64,
            resumed_ok_out: *mut i32,
            region_create_nanos_out: *mut u64,
            vm_create_nanos_out: *mut u64,
            vm_map_nanos_out: *mut u64,
            vcpu_create_nanos_out: *mut u64,
            register_restore_nanos_out: *mut u64,
            first_retrap_nanos_out: *mut u64,
            error_buffer: *mut c_char,
            error_capacity: usize,
        ) -> i32;

        fn proxima_vm_scratch_warm_vm_create(
            guest_memory_capacity: usize,
            context_out: *mut RawWarmRestoreContext,
            error_buffer: *mut c_char,
            error_capacity: usize,
        ) -> i32;

        fn proxima_vm_scratch_warm_restore(
            context: *mut RawWarmRestoreContext,
            registers_in: *const RawVcpuRegisters,
            guest_memory_in: *const u8,
            guest_memory_length: usize,
            page_size: usize,
            restore_wall_nanos_out: *mut u64,
            touch_all_pages_nanos_out: *mut u64,
            fault_count_out: *mut u64,
            resumed_x0_out: *mut u64,
            resumed_ok_out: *mut i32,
            register_restore_nanos_out: *mut u64,
            first_retrap_nanos_out: *mut u64,
            error_buffer: *mut c_char,
            error_capacity: usize,
        ) -> i32;

        fn proxima_vm_scratch_warm_vm_destroy(context: *mut RawWarmRestoreContext);

        fn proxima_vm_layered_vcpu_create(
            base_host_address: *mut c_void,
            base_size: usize,
            delta_host_address: *mut c_void,
            delta_size: usize,
            ipa_base: u64,
            context_out: *mut RawLayeredContext,
            error_buffer: *mut c_char,
            error_capacity: usize,
        ) -> i32;

        fn proxima_vm_layered_adopt_base(
            context: *mut RawLayeredContext,
            dirty_bitmap: *mut u8,
            dirty_bitmap_capacity: usize,
            map_nanos_out: *mut u64,
            register_reset_nanos_out: *mut u64,
            error_buffer: *mut c_char,
            error_capacity: usize,
        ) -> i32;

        fn proxima_vm_layered_run(
            context: *mut RawLayeredContext,
            expected_page_count: u64,
            dirty_bitmap: *mut u8,
            dirty_bitmap_capacity: usize,
            dirty_page_indices: *mut u32,
            dirty_page_indices_capacity: usize,
            dirty_page_index_count: *mut u64,
            run_wall_nanos_out: *mut u64,
            fault_count_out: *mut u64,
            newly_dirty_page_count_out: *mut u64,
            halted_ok_out: *mut i32,
            error_buffer: *mut c_char,
            error_capacity: usize,
        ) -> i32;

        fn proxima_vm_layered_restore(
            context: *mut RawLayeredContext,
            dirty_bitmap: *mut u8,
            dirty_bitmap_capacity: usize,
            dirty_page_indices: *mut u32,
            dirty_page_index_count: *mut u64,
            restore_wall_nanos_out: *mut u64,
            remap_nanos_out: *mut u64,
            register_reset_nanos_out: *mut u64,
            remapped_page_count_out: *mut u64,
            error_buffer: *mut c_char,
            error_capacity: usize,
        ) -> i32;

        fn proxima_vm_layered_vcpu_destroy(context: *mut RawLayeredContext);

        fn proxima_vm_host_page_size() -> usize;
    }

    fn read_error(error_buffer: &[c_char]) -> String {
        unsafe { CStr::from_ptr(error_buffer.as_ptr()) }
            .to_string_lossy()
            .into_owned()
    }

    pub(super) fn capture(message: &[u8]) -> Result<VmSnapshot, ProximaError> {
        let guest_memory_capacity = unsafe { proxima_vm_scratch_guest_memory_size(message.len()) };
        let mut output = vec![0_u8; OUTPUT_CAPACITY.max(message.len())];
        let mut registers_out = RawVcpuRegisters::default();
        let mut guest_memory_out = vec![0_u8; guest_memory_capacity];
        let mut error_buffer = [0 as c_char; ERROR_CAPACITY];

        let status = unsafe {
            proxima_vm_scratch_snapshot(
                message.as_ptr(),
                message.len(),
                output.as_mut_ptr(),
                output.len(),
                &raw mut registers_out,
                guest_memory_out.as_mut_ptr(),
                guest_memory_out.len(),
                error_buffer.as_mut_ptr(),
                ERROR_CAPACITY,
            )
        };
        if status != 0 {
            return Err(ProximaError::Upstream(read_error(&error_buffer)));
        }
        output.truncate(message.len());
        Ok(VmSnapshot {
            registers: VcpuRegisters { raw: registers_out },
            guest_memory: guest_memory_out,
            emitted: output,
        })
    }

    pub(super) fn restore(
        snapshot: &VmSnapshot,
        page_size: usize,
    ) -> Result<RestoreReport, ProximaError> {
        let mut restore_wall_nanos: u64 = 0;
        let mut touch_all_pages_nanos: u64 = 0;
        let mut fault_count: u64 = 0;
        let mut resumed_x0: u64 = 0;
        let mut resumed_ok: i32 = 0;
        let mut region_create_nanos: u64 = 0;
        let mut vm_create_nanos: u64 = 0;
        let mut vm_map_nanos: u64 = 0;
        let mut vcpu_create_nanos: u64 = 0;
        let mut register_restore_nanos: u64 = 0;
        let mut first_retrap_nanos: u64 = 0;
        let mut error_buffer = [0 as c_char; ERROR_CAPACITY];

        let status = unsafe {
            proxima_vm_scratch_restore(
                &raw const snapshot.registers.raw,
                snapshot.guest_memory.as_ptr(),
                snapshot.guest_memory.len(),
                page_size,
                &raw mut restore_wall_nanos,
                &raw mut touch_all_pages_nanos,
                &raw mut fault_count,
                &raw mut resumed_x0,
                &raw mut resumed_ok,
                &raw mut region_create_nanos,
                &raw mut vm_create_nanos,
                &raw mut vm_map_nanos,
                &raw mut vcpu_create_nanos,
                &raw mut register_restore_nanos,
                &raw mut first_retrap_nanos,
                error_buffer.as_mut_ptr(),
                ERROR_CAPACITY,
            )
        };
        if status != 0 {
            return Err(ProximaError::Upstream(read_error(&error_buffer)));
        }
        Ok(RestoreReport {
            restore_wall_nanos,
            touch_all_pages_nanos,
            fault_count,
            resumed_x0,
            resumed_matched_trap: resumed_ok != 0,
            phases: RestorePhases {
                region_create_nanos,
                vm_create_nanos,
                vm_map_nanos,
                vcpu_create_nanos,
                register_restore_nanos,
                first_retrap_nanos,
            },
        })
    }

    /// Platform half of [`super::WarmVm`] — owns the raw context this
    /// module's three warm-restore trampolines round-trip, and calls
    /// `proxima_vm_scratch_warm_vm_destroy` exactly once, on drop.
    pub(super) struct WarmVmHandle {
        context: RawWarmRestoreContext,
    }

    impl WarmVmHandle {
        pub(super) fn new(guest_memory_capacity: usize) -> Result<Self, ProximaError> {
            let mut context = RawWarmRestoreContext::default();
            let mut error_buffer = [0 as c_char; ERROR_CAPACITY];

            let status = unsafe {
                proxima_vm_scratch_warm_vm_create(
                    guest_memory_capacity,
                    &raw mut context,
                    error_buffer.as_mut_ptr(),
                    ERROR_CAPACITY,
                )
            };
            if status != 0 {
                return Err(ProximaError::Upstream(read_error(&error_buffer)));
            }
            Ok(Self { context })
        }

        pub(super) fn restore(
            &mut self,
            snapshot: &VmSnapshot,
            page_size: usize,
        ) -> Result<WarmRestoreReport, ProximaError> {
            let mut restore_wall_nanos: u64 = 0;
            let mut touch_all_pages_nanos: u64 = 0;
            let mut fault_count: u64 = 0;
            let mut resumed_x0: u64 = 0;
            let mut resumed_ok: i32 = 0;
            let mut register_restore_nanos: u64 = 0;
            let mut first_retrap_nanos: u64 = 0;
            let mut error_buffer = [0 as c_char; ERROR_CAPACITY];

            let status = unsafe {
                proxima_vm_scratch_warm_restore(
                    &raw mut self.context,
                    &raw const snapshot.registers.raw,
                    snapshot.guest_memory.as_ptr(),
                    snapshot.guest_memory.len(),
                    page_size,
                    &raw mut restore_wall_nanos,
                    &raw mut touch_all_pages_nanos,
                    &raw mut fault_count,
                    &raw mut resumed_x0,
                    &raw mut resumed_ok,
                    &raw mut register_restore_nanos,
                    &raw mut first_retrap_nanos,
                    error_buffer.as_mut_ptr(),
                    ERROR_CAPACITY,
                )
            };
            if status != 0 {
                return Err(ProximaError::Upstream(read_error(&error_buffer)));
            }
            Ok(WarmRestoreReport {
                restore_wall_nanos,
                touch_all_pages_nanos,
                fault_count,
                resumed_x0,
                resumed_matched_trap: resumed_ok != 0,
                phases: WarmRestorePhases { register_restore_nanos, first_retrap_nanos },
            })
        }

        /// Reads `length` bytes at `offset` directly from the mapped named
        /// region — `mach_vm_map`/`proxima_vm_create_named_region` already
        /// made `context.region.primary_address` valid host memory in this
        /// process (`WarmVmHandle::new`'s own call), so no FFI round trip is
        /// needed to read it back.
        pub(super) fn sample_guest_memory(&self, offset: usize, length: usize) -> Vec<u8> {
            assert!(
                offset.saturating_add(length) <= self.context.guest_memory_capacity,
                "sample range [{offset}, {}) exceeds mapped capacity {}",
                offset + length,
                self.context.guest_memory_capacity
            );
            let base = self.context.region.primary_address.cast::<u8>();
            unsafe { std::slice::from_raw_parts(base.add(offset), length) }.to_vec()
        }
    }

    impl Drop for WarmVmHandle {
        fn drop(&mut self) {
            unsafe {
                proxima_vm_scratch_warm_vm_destroy(&raw mut self.context);
            }
        }
    }

    /// Either half of a [`super::LayeredBase`]/[`super::LayeredBaseView`] —
    /// the only two host-address-space sources a layered vCPU's base can
    /// come from, matching the M4 `GuestMemoryRegion`/`RegionView` split
    /// exactly (an owner and a second mapper of the same object).
    enum BaseStorage {
        Owned(LayeredBase),
        Shared(LayeredBaseView),
    }

    impl BaseStorage {
        fn raw_parts_mut(&mut self) -> (*mut c_void, usize) {
            match self {
                Self::Owned(base) => (
                    base.region.primary_slice_mut().as_mut_ptr().cast::<c_void>(),
                    base.region.len(),
                ),
                Self::Shared(view) => (
                    view.view.as_slice_mut().as_mut_ptr().cast::<c_void>(),
                    view.view.as_slice().len(),
                ),
            }
        }

        fn as_slice(&self) -> &[u8] {
            match self {
                Self::Owned(base) => base.region.primary_slice(),
                Self::Shared(view) => view.view.as_slice(),
            }
        }
    }

    /// Platform half of [`super::WarmVm`]'s layered path — owns the base
    /// storage ([`BaseStorage`]), a private delta region, the caller-owned
    /// dirty bitmap, and the raw context the four layered trampolines
    /// round-trip. Calls `proxima_vm_layered_vcpu_destroy` exactly once, on
    /// drop — never `hv_vm_destroy` (`dispatch_trampoline.h`'s own doc on
    /// why: this process's one `hv_vm` may still be serving another
    /// [`LayeredHandle`] sharing the same base).
    pub(super) struct LayeredHandle {
        base: BaseStorage,
        delta: crate::named_memory::GuestMemoryRegion,
        dirty_bitmap: Vec<u8>,
        /// Ordered-append twin of `dirty_bitmap` (µsec campaign, layered
        /// restore O(working-set) slice) — `proxima_vm_layered_run` pushes a
        /// page index here the one time it transitions bitmap-clear ->
        /// bitmap-set, and `proxima_vm_layered_restore` sorts+coalesces this
        /// list instead of scanning every bit of `dirty_bitmap`. Sized once,
        /// at construction, to the same worst case (every page dirty) that
        /// already governs `dirty_bitmap`'s own capacity.
        dirty_page_indices: Vec<u32>,
        /// IN/OUT fill count for `dirty_page_indices`, round-tripped through
        /// both `proxima_vm_layered_run` (accumulates) and
        /// `proxima_vm_layered_restore` (consumes, then resets to 0).
        dirty_page_index_count: u64,
        context: RawLayeredContext,
    }

    fn bitmap_capacity(base_size: usize, granule: usize) -> usize {
        base_size.div_ceil(granule).div_ceil(8).max(1)
    }

    fn page_count(base_size: usize, granule: usize) -> usize {
        base_size.div_ceil(granule).max(1)
    }

    impl LayeredHandle {
        fn construct(mut base: BaseStorage, delta_capacity: usize, ipa_base: u64) -> Result<Self, ProximaError> {
            let mut delta = crate::named_memory::GuestMemoryRegion::create(delta_capacity)?;
            let (base_ptr, base_size) = base.raw_parts_mut();
            let delta_ptr = delta.primary_slice_mut().as_mut_ptr().cast::<c_void>();
            let mut context = RawLayeredContext::default();
            let mut error_buffer = [0 as c_char; ERROR_CAPACITY];

            let status = unsafe {
                proxima_vm_layered_vcpu_create(
                    base_ptr,
                    base_size,
                    delta_ptr,
                    delta_capacity,
                    ipa_base,
                    &raw mut context,
                    error_buffer.as_mut_ptr(),
                    ERROR_CAPACITY,
                )
            };
            if status != 0 {
                return Err(ProximaError::Upstream(read_error(&error_buffer)));
            }

            let granule = unsafe { proxima_vm_host_page_size() };
            let dirty_bitmap = vec![0_u8; bitmap_capacity(base_size, granule)];
            let dirty_page_indices = vec![0_u32; page_count(base_size, granule)];
            Ok(Self { base, delta, dirty_bitmap, dirty_page_indices, dirty_page_index_count: 0, context })
        }

        pub(super) fn new_owned(base: LayeredBase, delta_capacity: usize, ipa_base: u64) -> Result<Self, ProximaError> {
            Self::construct(BaseStorage::Owned(base), delta_capacity, ipa_base)
        }

        pub(super) fn new_shared(
            base_view: LayeredBaseView,
            delta_capacity: usize,
            ipa_base: u64,
        ) -> Result<Self, ProximaError> {
            Self::construct(BaseStorage::Shared(base_view), delta_capacity, ipa_base)
        }

        pub(super) fn write_base_and_adopt(&mut self, guest_memory: &[u8]) -> Result<LayeredAdoptReport, ProximaError> {
            match &mut self.base {
                BaseStorage::Owned(base) => {
                    base.region.primary_slice_mut()[..guest_memory.len()].copy_from_slice(guest_memory);
                }
                BaseStorage::Shared(_) => {
                    return Err(ProximaError::Config(
                        "a WarmVm sharing another WarmVm's layered base cannot write it -- use adopt_shared_base"
                            .into(),
                    ));
                }
            }
            self.adopt()
        }

        pub(super) fn adopt(&mut self) -> Result<LayeredAdoptReport, ProximaError> {
            let mut map_nanos: u64 = 0;
            let mut register_reset_nanos: u64 = 0;
            let mut error_buffer = [0 as c_char; ERROR_CAPACITY];

            let status = unsafe {
                proxima_vm_layered_adopt_base(
                    &raw mut self.context,
                    self.dirty_bitmap.as_mut_ptr(),
                    self.dirty_bitmap.len(),
                    &raw mut map_nanos,
                    &raw mut register_reset_nanos,
                    error_buffer.as_mut_ptr(),
                    ERROR_CAPACITY,
                )
            };
            if status != 0 {
                return Err(ProximaError::Upstream(read_error(&error_buffer)));
            }
            Ok(LayeredAdoptReport { map_nanos, register_reset_nanos })
        }

        pub(super) fn run(&mut self, expected_page_count: u64) -> Result<DirtyRunReport, ProximaError> {
            let mut run_wall_nanos: u64 = 0;
            let mut fault_count: u64 = 0;
            let mut newly_dirty_page_count: u64 = 0;
            let mut halted_ok: i32 = 0;
            let mut error_buffer = [0 as c_char; ERROR_CAPACITY];

            let status = unsafe {
                proxima_vm_layered_run(
                    &raw mut self.context,
                    expected_page_count,
                    self.dirty_bitmap.as_mut_ptr(),
                    self.dirty_bitmap.len(),
                    self.dirty_page_indices.as_mut_ptr(),
                    self.dirty_page_indices.len(),
                    &raw mut self.dirty_page_index_count,
                    &raw mut run_wall_nanos,
                    &raw mut fault_count,
                    &raw mut newly_dirty_page_count,
                    &raw mut halted_ok,
                    error_buffer.as_mut_ptr(),
                    ERROR_CAPACITY,
                )
            };
            if status != 0 {
                return Err(ProximaError::Upstream(read_error(&error_buffer)));
            }
            Ok(DirtyRunReport {
                run_wall_nanos,
                fault_count,
                newly_dirty_page_count,
                halted_ok: halted_ok != 0,
            })
        }

        pub(super) fn restore(&mut self) -> Result<LayeredRestoreReport, ProximaError> {
            let mut restore_wall_nanos: u64 = 0;
            let mut remap_nanos: u64 = 0;
            let mut register_reset_nanos: u64 = 0;
            let mut remapped_page_count: u64 = 0;
            let mut error_buffer = [0 as c_char; ERROR_CAPACITY];

            let status = unsafe {
                proxima_vm_layered_restore(
                    &raw mut self.context,
                    self.dirty_bitmap.as_mut_ptr(),
                    self.dirty_bitmap.len(),
                    self.dirty_page_indices.as_mut_ptr(),
                    &raw mut self.dirty_page_index_count,
                    &raw mut restore_wall_nanos,
                    &raw mut remap_nanos,
                    &raw mut register_reset_nanos,
                    &raw mut remapped_page_count,
                    error_buffer.as_mut_ptr(),
                    ERROR_CAPACITY,
                )
            };
            if status != 0 {
                return Err(ProximaError::Upstream(read_error(&error_buffer)));
            }
            Ok(LayeredRestoreReport {
                restore_wall_nanos,
                remap_nanos,
                register_reset_nanos,
                remapped_page_count,
            })
        }

        pub(super) fn share_base(&self) -> Result<LayeredBaseView, ProximaError> {
            match &self.base {
                BaseStorage::Owned(base) => base.share(),
                BaseStorage::Shared(_) => Err(ProximaError::Config(
                    "a WarmVm that already shares its layered base cannot re-share it".into(),
                )),
            }
        }

        pub(super) fn base_bytes(&self, offset: usize, length: usize) -> Vec<u8> {
            self.base.as_slice()[offset..offset + length].to_vec()
        }

        pub(super) fn delta_bytes(&self, offset: usize, length: usize) -> Vec<u8> {
            self.delta.primary_slice()[offset..offset + length].to_vec()
        }
    }

    impl Drop for LayeredHandle {
        fn drop(&mut self) {
            unsafe {
                proxima_vm_layered_vcpu_destroy(&raw mut self.context);
            }
        }
    }

    pub(super) fn host_page_size() -> usize {
        unsafe { proxima_vm_host_page_size() }
    }
}

#[cfg(not(any(
    all(target_os = "linux", target_arch = "x86_64"),
    all(target_os = "macos", target_arch = "aarch64")
)))]
mod platform {
    use proxima_core::ProximaError;

    use super::{
        DirtyRunReport, LayeredAdoptReport, LayeredBase, LayeredBaseView, LayeredRestoreReport, RestoreReport,
        VmSnapshot, WarmRestoreReport,
    };

    const UNSUPPORTED: &str =
        "vm snapshot capture/restore supports linux/x86_64 KVM and macos/aarch64 Hypervisor.framework only";

    pub(super) fn capture(_message: &[u8]) -> Result<VmSnapshot, ProximaError> {
        Err(ProximaError::Config(UNSUPPORTED.into()))
    }

    pub(super) fn restore(
        _snapshot: &VmSnapshot,
        _page_size: usize,
    ) -> Result<RestoreReport, ProximaError> {
        Err(ProximaError::Config(UNSUPPORTED.into()))
    }

    pub(super) struct WarmVmHandle;

    impl WarmVmHandle {
        pub(super) fn new(_guest_memory_capacity: usize) -> Result<Self, ProximaError> {
            Err(ProximaError::Config(UNSUPPORTED.into()))
        }

        pub(super) fn restore(
            &mut self,
            _snapshot: &VmSnapshot,
            _page_size: usize,
        ) -> Result<WarmRestoreReport, ProximaError> {
            Err(ProximaError::Config(UNSUPPORTED.into()))
        }

        pub(super) fn sample_guest_memory(&self, _offset: usize, _length: usize) -> Vec<u8> {
            Vec::new()
        }
    }

    pub(super) struct LayeredHandle;

    impl LayeredHandle {
        pub(super) fn new_owned(_base: LayeredBase, _delta_capacity: usize, _ipa_base: u64) -> Result<Self, ProximaError> {
            Err(ProximaError::Config(UNSUPPORTED.into()))
        }

        pub(super) fn new_shared(
            _base_view: LayeredBaseView,
            _delta_capacity: usize,
            _ipa_base: u64,
        ) -> Result<Self, ProximaError> {
            Err(ProximaError::Config(UNSUPPORTED.into()))
        }

        pub(super) fn write_base_and_adopt(&mut self, _guest_memory: &[u8]) -> Result<LayeredAdoptReport, ProximaError> {
            Err(ProximaError::Config(UNSUPPORTED.into()))
        }

        pub(super) fn adopt(&mut self) -> Result<LayeredAdoptReport, ProximaError> {
            Err(ProximaError::Config(UNSUPPORTED.into()))
        }

        pub(super) fn run(&mut self, _expected_page_count: u64) -> Result<DirtyRunReport, ProximaError> {
            Err(ProximaError::Config(UNSUPPORTED.into()))
        }

        pub(super) fn restore(&mut self) -> Result<LayeredRestoreReport, ProximaError> {
            Err(ProximaError::Config(UNSUPPORTED.into()))
        }

        pub(super) fn share_base(&self) -> Result<LayeredBaseView, ProximaError> {
            Err(ProximaError::Config(UNSUPPORTED.into()))
        }

        pub(super) fn base_bytes(&self, _offset: usize, _length: usize) -> Vec<u8> {
            Vec::new()
        }

        pub(super) fn delta_bytes(&self, _offset: usize, _length: usize) -> Vec<u8> {
            Vec::new()
        }
    }

    pub(super) fn host_page_size() -> usize {
        4096
    }
}

// `capture`/`restore` call real `hv_vm_create`/`KVM_CREATE_VM`, which
// answers `HV_DENIED`/`EPERM` for a process lacking the
// `com.apple.security.hypervisor` entitlement (`tests/boot.rs`'s own doc) --
// a plain `#[cfg(test)] mod tests` here runs inside the unsigned nextest
// test binary and would fail on the very first call. The worked-example
// tests live in `tests/vm_snapshot.rs`, driven through the codesigned
// `snapshot_probe` binary (`src/bin/snapshot_probe.rs`), exactly the
// `SignedGuest` pattern `tests/boot.rs` already established for `hello`.
#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::VcpuRegisters;

    /// `VcpuRegisters`/`VmSnapshot`/`RestoreReport` are plain data with no
    /// platform call in their own construction -- this much is exercisable
    /// without an entitlement, and it is what guards their `Default`/`Copy`
    /// derives never silently regressing to something non-`Copy`.
    #[test]
    fn a_default_register_file_is_all_zero() {
        let registers = VcpuRegisters::default();
        assert_eq!(registers.raw.gpr, [0_u64; 31]);
        assert_eq!(registers.raw.pc, 0);
        assert_eq!(registers.raw.flags, 0);
    }
}
