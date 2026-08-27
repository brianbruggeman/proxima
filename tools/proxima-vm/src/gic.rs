//! GICv3 Distributor (GICD) and Redistributor (GICR) register blocks — M5b
//! slices 1 and 2 of the GIC model `ROADMAP.md` (:324-333) names alongside
//! PSCI (`src/psci.rs`) and the DTB (`src/dtb.rs`). `src/dtb.rs`'s
//! `write_gic` advertises `compatible = "arm,gic-v3"` with `reg =
//! <gicd_base gicd_size gicr_base gicr_size>` (`src/dtb.rs:181-201`) — this
//! module is the pure decode/state machine for both of those windows. The
//! CPU interface is HVF-trap territory investigated later (`ROADMAP.md`'s
//! own GIC ordering); neither window is wired to an MMIO trap yet (that
//! composes in the final GIC slice).
//!
//! # ID banking: GICD owns 32-255, GICR owns 0-31, no overlap and no gap
//!
//! GICv3 affinity routing splits `NUM_INTERRUPT_IDS` (256) architected IDs
//! across the two register blocks with no ID modeled twice and none
//! unmodeled: [`GicDistributor`] holds real per-ID state for SPIs
//! (`SPI_BASE`..`NUM_INTERRUPT_IDS`, i.e. 32-255) and RAZ/WI's every
//! register's "word 0" (the sub-range that would otherwise cover 0-31 —
//! see `GicDistributor::apply_group_bitmap`'s doc). [`GicRedistributor`]
//! holds real per-ID state for exactly that banked-away range, SGIs (0-15)
//! and PPIs (16-31), in its SGI_base frame. A guest probing ID 40 reads live
//! state from the Distributor; a guest probing ID 5 reads live state from
//! this vCPU's Redistributor; neither block claims the other's IDs, and
//! together they cover 0..256 exactly once.
//!
//! Mirrors `proxima-protocols/src/virtio/mmio.rs`'s shape exactly: one raw
//! `(offset, is_write, value)` access in, one typed effect or error out, a
//! single match over offset per register block — no cursor, no I/O. Both
//! blocks share [`GicAccess`] (the same raw access triple, since a trapped
//! MMIO load/store looks identical regardless of which window it landed in)
//! but keep distinct effect/error enums — the state each block owns, and so
//! what a write can report changing, differs.
//!
//! # Spec citations
//!
//! Register names, offsets, and field layouts are ARM IHI 0069 (the GICv3
//! and GICv4 architecture specification), chapter 8 ("The GIC Distributor
//! register descriptions"). Offsets and bit layouts below are the
//! architected constants every GICv3 implementation shares; this module's
//! own citations name the register, not a specific sub-section number,
//! because no local copy of the PDF was available to re-verify exact
//! subsection numbering in this session (principle 6: do not assert a
//! precision that was not checked).
//!
//! # This GICD models a fixed 256-ID GIC
//!
//! `NUM_INTERRUPT_IDS` (256) and `SPI_COUNT` (224) are QEMU's own `virt`
//! machine constant (`hw/arm/virt.c`'s `NUM_IRQS`), the same incumbent
//! `src/dtb.rs`'s `QemuVirtLayout::CANONICAL` already anchors its GICD/GICR
//! MMIO windows against. A GICv3 `ITLinesNumber` (`GICD_TYPER` bits\[4:0\])
//! is derived from it, not hand-picked, so the two numbers can never drift
//! apart. This is a fixed array, not `MmioDevice<const MAX_QUEUES: usize>`'s
//! const-generic shape: `MmioDevice`'s queue count is a genuine per-call
//! configuration (how many virtqueues *this* device negotiates), while a
//! GICD's SPI count is fixed by which physical/virtual GIC instance is being
//! modeled — QEMU virt's own memory map, already a `pub const` sibling to
//! `SINGLE_VCPU_GICR_SIZE` in `src/dtb.rs`. Generalizing to a type parameter
//! would add an unconstrained runtime dimension this slice's one worked
//! example (QEMU virt, HVF) never needs; ROADMAP.md's own condition-3 gate
//! (extend before adding a peer) says fixed until a second GIC instance with
//! a different SPI count actually shows up.
//!
//! # Tier
//!
//! Tier-3 (bare `no_std + no_alloc`), landed unconditional like `src/psci.rs`
//! and `src/page_table.rs`: every field is a fixed-size array, no
//! allocation, no register access, no syscall.

/// Total architected interrupt IDs this GICD models: SGI (0-15) + PPI
/// (16-31) + SPI (32-255) — QEMU `virt`'s own `NUM_IRQS` (`hw/arm/virt.c`),
/// the same incumbent layout `src/dtb.rs`'s `QemuVirtLayout` anchors its
/// GICD/GICR windows against.
const NUM_INTERRUPT_IDS: u32 = 256;

/// First Shared Peripheral Interrupt ID; IDs below this are SGIs/PPIs, which
/// GICv3 affinity routing banks per-PE in the Redistributor rather than the
/// Distributor (spec: GICD_ISENABLER0 and its siblings' "n=0 word" note).
const SPI_BASE: u32 = 32;

/// SPIs this GICD implements: `NUM_INTERRUPT_IDS` minus the SGI/PPI range.
pub const SPI_COUNT: usize = (NUM_INTERRUPT_IDS - SPI_BASE) as usize;

/// `GICD_TYPER.ITLinesNumber`: `N` such that the highest implemented SPI ID
/// is `32*(N+1) - 1`. Derived from [`NUM_INTERRUPT_IDS`] so it can never
/// disagree with the array sizes above.
const IT_LINES_NUMBER: u32 = NUM_INTERRUPT_IDS / 32 - 1;

/// Number of 32-bit words `GICD_IGROUPR`/`GICD_ISENABLER`/`GICD_ICENABLER`/
/// `GICD_ISPENDR`/`GICD_ICPENDR` each implement: one word per 32 interrupt
/// IDs, `ITLinesNumber + 1` words total (word 0 covers SGIs/PPIs and is
/// RAZ/WI at the Distributor).
const BITMAP_WORD_COUNT: u32 = IT_LINES_NUMBER + 1;

/// Number of 32-bit words `GICD_ICFGR` implements: 2 bits per interrupt ID,
/// 16 IDs per word, so `NUM_INTERRUPT_IDS / 16` words total (double
/// [`BITMAP_WORD_COUNT`]'s one-bit-per-ID word count). Words 0-1 (IDs 0-31)
/// are RAZ/WI at the Distributor — the same SGI/PPI banking every other
/// bitmap register's word 0 follows (this module's own "ID banking" doc
/// section) — since GICv3 affinity routing owns SGI/PPI trigger config in
/// the Redistributor's `GICR_ICFGR0`/`GICR_ICFGR1`, not the Distributor.
const ICFGR_WORD_COUNT: u32 = NUM_INTERRUPT_IDS / 16;

/// Number of 32-bit words `GICD_IPRIORITYR` implements: one byte per
/// interrupt ID, four IDs per word.
const PRIORITY_WORD_COUNT: u32 = NUM_INTERRUPT_IDS / 4;

const REG_CTLR: u64 = 0x0000;
/// `GICD_STATUSR` (offset 0x0010, RW). ARM IHI 0069 ch.8: a latched
/// error-status word (`Wr_D`/`Wr_D_A`/`Rd_D`/etc), write-1-to-clear.
const REG_STATUSR: u64 = 0x0010;
const REG_TYPER: u64 = 0x0004;
const REG_IIDR: u64 = 0x0008;
/// `GICD_TYPER2` (GICv3.1+, offset 0x000C, RO). Linux's `gic-v3` driver
/// (`drivers/irqchip/irq-gic-v3.c`'s `gic_of_init`/`gic_acpi_init` probe
/// path) reads this unconditionally to detect extended-SPI/extended-PPI and
/// virtual-LPI support; this GICD implements none of those extensions, so
/// every bit reads architecturally zero (RAZ/WI, the reserved-field
/// default), matching a GICv3.0 distributor that predates this register.
const REG_TYPER2: u64 = 0x000c;
const REG_IGROUPR_BASE: u64 = 0x0080;
const REG_ISENABLER_BASE: u64 = 0x0100;
const REG_ICENABLER_BASE: u64 = 0x0180;
const REG_ISPENDR_BASE: u64 = 0x0200;
const REG_ICPENDR_BASE: u64 = 0x0280;
/// `GICD_ISACTIVER<n>` (offset 0x0300 + 4n, RW1S): the active-state twin of
/// [`REG_ISENABLER_BASE`], set on the same one-bit-per-ID layout.
const REG_ISACTIVER_BASE: u64 = 0x0300;
/// `GICD_ICACTIVER<n>` (offset 0x0380 + 4n, RW1C): clears the same active
/// bit [`REG_ISACTIVER_BASE`] sets.
const REG_ICACTIVER_BASE: u64 = 0x0380;
const REG_IPRIORITYR_BASE: u64 = 0x0400;
const REG_IROUTER_BASE: u64 = 0x6000;
/// `GICD_ICFGR<n>` (offset 0x0c00 + 4n, RW): trigger-mode config, 2 bits per
/// interrupt ID, 16 IDs per word.
const REG_ICFGR_BASE: u64 = 0x0c00;
/// `GICD_IGRPMODR<n>` (offset 0x0d00 + 4n, RW): the Group 1 security-state
/// modifier bitmap, one bit per interrupt ID, same word layout as
/// [`REG_IGROUPR_BASE`].
const REG_IGRPMODR_BASE: u64 = 0x0d00;
const REG_PIDR2: u64 = 0xffe8;

/// GICv3 architecture revision `GICD_PIDR2.ArchRev` reports (bits\[7:4\]) —
/// the field Linux's `gic-v3` driver reads at probe to confirm it is
/// talking to a GICv3, not a GICv2, distributor.
const GICD_ARCH_REV_GICV3: u32 = 3;

/// One raw register access recovered from a trapped guest load/store,
/// mirroring `proxima_protocols::virtio::MmioAccess` exactly: a byte offset
/// from the containing window's own base ([`GicDistributor::apply`]'s GICD
/// window, or [`GicRedistributor::apply`]'s per-vCPU GICR window), whether
/// it was a write, and — for a write — the 32-bit value the guest stored. A
/// read carries `value: 0`, ignored by every read arm below.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GicAccess {
    pub offset: u64,
    pub is_write: bool,
    pub value: u32,
}

/// What the caller must do in response to one applied [`GicAccess`] —
/// mirrors `MmioEffect`'s shape: a read names the word to write back, a
/// state-changing write names exactly what changed so the caller never has
/// to re-read [`GicDistributor`] state to find out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GicdEffect {
    /// A read register access: the value the caller writes back into the
    /// guest's destination register.
    ReadValue(u32),
    /// A write register access accepted with no further consequence — the
    /// register's new value has already been recorded in [`GicDistributor`].
    Applied,
    /// A `GICD_CTLR` write changed the group-enable state.
    ControlUpdated {
        group0_enabled: bool,
        group1_enabled: bool,
    },
    /// An `ISENABLER`/`ICENABLER` write changed one SPI's enable state.
    SpiEnableChanged { spi: u32, enabled: bool },
    /// An `ISPENDR`/`ICPENDR` write changed one SPI's pending state.
    SpiPendingChanged { spi: u32, pending: bool },
    /// An `ISACTIVER`/`ICACTIVER` write changed one SPI's active state.
    SpiActiveChanged { spi: u32, active: bool },
    /// A `GICD_STATUSR` write cleared one or more latched error bits — the
    /// value carried is the register's new (post-clear) state.
    StatusrUpdated { value: u32 },
}

/// Why [`GicDistributor::apply`] rejected an access.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GicdError {
    /// No register this GICD implements exists at this offset — either a
    /// genuinely undefined offset, or an offset that would extend past
    /// [`SPI_COUNT`] SPIs / `IT_LINES_NUMBER` words for this fixed-size
    /// GIC. Never silently RAZ/WI'd: an out-of-range probe is a bug worth
    /// surfacing, not hiding.
    UnknownRegister { offset: u64 },
    /// The access offset was not naturally aligned to the register's
    /// 4-byte word size.
    UnalignedAccess { offset: u64 },
    /// The register at this offset is read-only; the guest attempted a
    /// write.
    ReadOnlyRegister { offset: u64 },
}

impl core::fmt::Display for GicdError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::UnknownRegister { offset } => {
                write!(formatter, "no gicd register defined at offset {offset:#x}")
            }
            Self::UnalignedAccess { offset } => {
                write!(
                    formatter,
                    "gicd access at offset {offset:#x} is not 4-byte aligned"
                )
            }
            Self::ReadOnlyRegister { offset } => {
                write!(
                    formatter,
                    "gicd register at offset {offset:#x} is read-only"
                )
            }
        }
    }
}

impl core::error::Error for GicdError {}

/// One GICv3 Distributor's register-block state: the group-enable flags
/// `GICD_CTLR` owns, plus per-SPI group/enable/pending/priority/routing
/// state for every SPI this fixed-size GICD implements ([`SPI_COUNT`]).
/// SGI/PPI (IDs 0-31) state is deliberately absent — GICv3 affinity routing
/// banks it per-PE in the Redistributor, the next slice.
#[derive(Debug, Clone)]
pub struct GicDistributor {
    group0_enabled: bool,
    group1_enabled: bool,
    statusr: u32,
    spi_group1: [bool; SPI_COUNT],
    spi_group_mod: [bool; SPI_COUNT],
    spi_enabled: [bool; SPI_COUNT],
    spi_pending: [bool; SPI_COUNT],
    spi_active: [bool; SPI_COUNT],
    spi_edge_triggered: [bool; SPI_COUNT],
    spi_priority: [u8; SPI_COUNT],
    spi_route: [u64; SPI_COUNT],
}

impl Default for GicDistributor {
    fn default() -> Self {
        Self::new()
    }
}

impl GicDistributor {
    /// A freshly reset distributor: both interrupt groups disabled
    /// (`GICD_CTLR.EnableGrp0`/`EnableGrp1` reset to 0 per spec), no latched
    /// `STATUSR` error bits, every SPI in group 0, no group-1 security
    /// modifier, disabled, not pending, not active, level-triggered (the
    /// architected `GICD_ICFGR` reset value), priority 0 (highest), and
    /// routed to affinity 0 — the architected power-on reset state.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            group0_enabled: false,
            group1_enabled: false,
            statusr: 0,
            spi_group1: [false; SPI_COUNT],
            spi_group_mod: [false; SPI_COUNT],
            spi_enabled: [false; SPI_COUNT],
            spi_pending: [false; SPI_COUNT],
            spi_active: [false; SPI_COUNT],
            spi_edge_triggered: [false; SPI_COUNT],
            spi_priority: [0; SPI_COUNT],
            spi_route: [0; SPI_COUNT],
        }
    }

    #[must_use]
    pub fn group0_enabled(&self) -> bool {
        self.group0_enabled
    }

    #[must_use]
    pub fn group1_enabled(&self) -> bool {
        self.group1_enabled
    }

    #[must_use]
    pub fn spi_enabled(&self, spi: u32) -> Option<bool> {
        self.spi_enabled.get(spi_index(spi)?).copied()
    }

    #[must_use]
    pub fn spi_pending(&self, spi: u32) -> Option<bool> {
        self.spi_pending.get(spi_index(spi)?).copied()
    }

    #[must_use]
    pub fn spi_priority(&self, spi: u32) -> Option<u8> {
        self.spi_priority.get(spi_index(spi)?).copied()
    }

    #[must_use]
    pub fn spi_route(&self, spi: u32) -> Option<u64> {
        self.spi_route.get(spi_index(spi)?).copied()
    }

    #[must_use]
    pub fn spi_active(&self, spi: u32) -> Option<bool> {
        self.spi_active.get(spi_index(spi)?).copied()
    }

    #[must_use]
    pub fn spi_edge_triggered(&self, spi: u32) -> Option<bool> {
        self.spi_edge_triggered.get(spi_index(spi)?).copied()
    }

    #[must_use]
    pub fn statusr(&self) -> u32 {
        self.statusr
    }

    /// Apply one register access, returning the effect the caller must carry
    /// out. Mirrors `MmioDevice::apply`'s single match over `(offset,
    /// is_write)` — the whole GICD register map lives in this one function
    /// because every register here is a fixed 32-bit word at a fixed
    /// offset, never a variable-length or streamed field.
    pub fn apply(&mut self, access: GicAccess) -> Result<GicdEffect, GicdError> {
        if !access.offset.is_multiple_of(4) {
            return Err(GicdError::UnalignedAccess {
                offset: access.offset,
            });
        }

        match access.offset {
            REG_CTLR => self.apply_ctlr(access),
            REG_STATUSR => self.apply_statusr(access),
            REG_TYPER => read_only(access, self.read_typer()),
            REG_IIDR => read_only(access, 0),
            REG_TYPER2 => read_only(access, 0),
            REG_PIDR2 => read_only(access, GICD_ARCH_REV_GICV3 << 4),
            offset if in_bitmap_range(offset, REG_IGROUPR_BASE) => {
                self.apply_group_bitmap(offset, access)
            }
            offset if in_bitmap_range(offset, REG_ISENABLER_BASE) => {
                self.apply_enable_bitmap(offset, access, true)
            }
            offset if in_bitmap_range(offset, REG_ICENABLER_BASE) => {
                self.apply_enable_bitmap(offset, access, false)
            }
            offset if in_bitmap_range(offset, REG_ISPENDR_BASE) => {
                self.apply_pending_bitmap(offset, access, true)
            }
            offset if in_bitmap_range(offset, REG_ICPENDR_BASE) => {
                self.apply_pending_bitmap(offset, access, false)
            }
            offset if in_bitmap_range(offset, REG_ISACTIVER_BASE) => {
                self.apply_active_bitmap(offset, access, true)
            }
            offset if in_bitmap_range(offset, REG_ICACTIVER_BASE) => {
                self.apply_active_bitmap(offset, access, false)
            }
            offset if in_priority_range(offset) => self.apply_priority(offset, access),
            offset if in_router_range(offset) => self.apply_router(offset, access),
            offset if in_icfgr_range(offset) => self.apply_icfgr(offset, access),
            offset if in_bitmap_range(offset, REG_IGRPMODR_BASE) => {
                self.apply_group_mod_bitmap(offset, access)
            }
            offset => Err(GicdError::UnknownRegister { offset }),
        }
    }

    /// `GICD_CTLR` (offset 0x0000, RW). Modeled for a GICv3 in single
    /// security state: bit 0 `EnableGrp0`, bit 1 `EnableGrp1`, bit 4 `ARE`
    /// (Affinity Routing Enable) forced Read-As-One/Write-Ignored because
    /// this GICD never implements the legacy non-affinity-routed mode, bit
    /// 31 `RWP` (Register Write Pending) always 0 because every write here
    /// completes synchronously. Unrecognized bits are RES0/write-ignored,
    /// the architected default for reserved control-register fields.
    fn apply_ctlr(&mut self, access: GicAccess) -> Result<GicdEffect, GicdError> {
        const ARE_BIT: u32 = 1 << 4;
        if !access.is_write {
            let value =
                u32::from(self.group0_enabled) | (u32::from(self.group1_enabled) << 1) | ARE_BIT;
            return Ok(GicdEffect::ReadValue(value));
        }
        self.group0_enabled = access.value & 0b1 != 0;
        self.group1_enabled = access.value & 0b10 != 0;
        Ok(GicdEffect::ControlUpdated {
            group0_enabled: self.group0_enabled,
            group1_enabled: self.group1_enabled,
        })
    }

    /// `GICD_STATUSR` (offset 0x0010, RW). Latched error-status bits
    /// (`Wr_D`/`Wr_D_A`/`Rd_D`/etc — this decode-only model never sets any
    /// of them, since every access this GICD rejects is surfaced as
    /// [`GicdError`] to the caller rather than latched here) are cleared by
    /// writing 1 to the bit, the architecture's RWC (write-1-to-clear)
    /// contract; a write of 0 to any bit leaves it unchanged.
    fn apply_statusr(&mut self, access: GicAccess) -> Result<GicdEffect, GicdError> {
        if !access.is_write {
            return Ok(GicdEffect::ReadValue(self.statusr));
        }
        self.statusr &= !access.value;
        Ok(GicdEffect::StatusrUpdated {
            value: self.statusr,
        })
    }

    /// `GICD_TYPER` (offset 0x0004, RO). Only `ITLinesNumber` (bits\[4:0\])
    /// is populated — `CPUNumber`/`SecurityExtn`/every other field reports 0
    /// for this single-vCPU, single-security-state GICD.
    fn read_typer(&self) -> u32 {
        IT_LINES_NUMBER
    }

    /// `GICD_IGROUPR<n>` (offset 0x0080 + 4n, RW): one bit per interrupt ID,
    /// 1 = Group 1, 0 = Group 0. Word 0 (IDs 0-31, SGIs/PPIs) is RAZ/WI —
    /// GICv3 affinity routing banks SGI/PPI group membership per-PE in the
    /// Redistributor, never the Distributor.
    fn apply_group_bitmap(
        &mut self,
        offset: u64,
        access: GicAccess,
    ) -> Result<GicdEffect, GicdError> {
        let word = bitmap_word_index(offset, REG_IGROUPR_BASE)?;
        if word == 0 {
            return raz_wi(access);
        }
        let base_spi = 32 * word;
        if access.is_write {
            for bit in 0..32 {
                let Some(index) = spi_index(base_spi + bit) else {
                    break;
                };
                self.spi_group1[index] = access.value & (1 << bit) != 0;
            }
            return Ok(GicdEffect::Applied);
        }
        let mut value = 0u32;
        for bit in 0..32 {
            let Some(index) = spi_index(base_spi + bit) else {
                break;
            };
            if self.spi_group1[index] {
                value |= 1 << bit;
            }
        }
        Ok(GicdEffect::ReadValue(value))
    }

    /// `GICD_ISENABLER<n>`/`GICD_ICENABLER<n>` (offsets 0x0100/0x0180 + 4n):
    /// aliased RW1S/RW1C views over one shared per-SPI enable bit — a write
    /// of 1 to a given bit in `ISENABLER` sets that SPI's enable state, the
    /// same bit in `ICENABLER` clears it; a 0 bit is a no-op in both,
    /// matching the spec's "write 1 to set/clear, write 0 has no effect"
    /// contract. Reading either alias reports the same current state. Word
    /// 0 is RAZ/WI (see [`Self::apply_group_bitmap`]'s SGI/PPI note).
    fn apply_enable_bitmap(
        &mut self,
        offset: u64,
        access: GicAccess,
        set_bits: bool,
    ) -> Result<GicdEffect, GicdError> {
        let base = if set_bits {
            REG_ISENABLER_BASE
        } else {
            REG_ICENABLER_BASE
        };
        let word = bitmap_word_index(offset, base)?;
        if word == 0 {
            return raz_wi(access);
        }
        let base_spi = 32 * word;
        if access.is_write {
            let mut last_changed = None;
            for bit in 0..32 {
                if access.value & (1 << bit) == 0 {
                    continue;
                }
                let Some(index) = spi_index(base_spi + bit) else {
                    break;
                };
                self.spi_enabled[index] = set_bits;
                last_changed = Some(base_spi + bit);
            }
            return match last_changed {
                Some(spi) => Ok(GicdEffect::SpiEnableChanged {
                    spi,
                    enabled: set_bits,
                }),
                None => Ok(GicdEffect::Applied),
            };
        }
        let mut value = 0u32;
        for bit in 0..32 {
            let Some(index) = spi_index(base_spi + bit) else {
                break;
            };
            if self.spi_enabled[index] {
                value |= 1 << bit;
            }
        }
        Ok(GicdEffect::ReadValue(value))
    }

    /// `GICD_ISPENDR<n>`/`GICD_ICPENDR<n>` (offsets 0x0200/0x0280 + 4n): the
    /// pending-state twin of [`Self::apply_enable_bitmap`], same RW1S/RW1C
    /// contract over one shared per-SPI pending bit.
    fn apply_pending_bitmap(
        &mut self,
        offset: u64,
        access: GicAccess,
        set_bits: bool,
    ) -> Result<GicdEffect, GicdError> {
        let base = if set_bits {
            REG_ISPENDR_BASE
        } else {
            REG_ICPENDR_BASE
        };
        let word = bitmap_word_index(offset, base)?;
        if word == 0 {
            return raz_wi(access);
        }
        let base_spi = 32 * word;
        if access.is_write {
            let mut last_changed = None;
            for bit in 0..32 {
                if access.value & (1 << bit) == 0 {
                    continue;
                }
                let Some(index) = spi_index(base_spi + bit) else {
                    break;
                };
                self.spi_pending[index] = set_bits;
                last_changed = Some(base_spi + bit);
            }
            return match last_changed {
                Some(spi) => Ok(GicdEffect::SpiPendingChanged {
                    spi,
                    pending: set_bits,
                }),
                None => Ok(GicdEffect::Applied),
            };
        }
        let mut value = 0u32;
        for bit in 0..32 {
            let Some(index) = spi_index(base_spi + bit) else {
                break;
            };
            if self.spi_pending[index] {
                value |= 1 << bit;
            }
        }
        Ok(GicdEffect::ReadValue(value))
    }

    /// `GICD_ISACTIVER<n>`/`GICD_ICACTIVER<n>` (offsets 0x0300/0x0380 + 4n):
    /// the active-state twin of [`Self::apply_enable_bitmap`], same
    /// RW1S/RW1C contract over one shared per-SPI active bit — Linux's
    /// `gic-v3` SPI-init path probes this pair right alongside
    /// enable/pending during its save/restore walk. Word 0 is RAZ/WI, the
    /// same SGI/PPI banking [`Self::apply_enable_bitmap`] documents.
    fn apply_active_bitmap(
        &mut self,
        offset: u64,
        access: GicAccess,
        set_bits: bool,
    ) -> Result<GicdEffect, GicdError> {
        let base = if set_bits {
            REG_ISACTIVER_BASE
        } else {
            REG_ICACTIVER_BASE
        };
        let word = bitmap_word_index(offset, base)?;
        if word == 0 {
            return raz_wi(access);
        }
        let base_spi = 32 * word;
        if access.is_write {
            let mut last_changed = None;
            for bit in 0..32 {
                if access.value & (1 << bit) == 0 {
                    continue;
                }
                let Some(index) = spi_index(base_spi + bit) else {
                    break;
                };
                self.spi_active[index] = set_bits;
                last_changed = Some(base_spi + bit);
            }
            return match last_changed {
                Some(spi) => Ok(GicdEffect::SpiActiveChanged {
                    spi,
                    active: set_bits,
                }),
                None => Ok(GicdEffect::Applied),
            };
        }
        let mut value = 0u32;
        for bit in 0..32 {
            let Some(index) = spi_index(base_spi + bit) else {
                break;
            };
            if self.spi_active[index] {
                value |= 1 << bit;
            }
        }
        Ok(GicdEffect::ReadValue(value))
    }

    /// `GICD_IPRIORITYR<n>` (offset 0x0400 + 4n, RW): one byte per interrupt
    /// ID, four IDs packed little-endian per word. Words covering only
    /// SGI/PPI IDs (n < 8) are RAZ/WI for the same reason the bitmap
    /// registers' word 0 is.
    fn apply_priority(&mut self, offset: u64, access: GicAccess) -> Result<GicdEffect, GicdError> {
        let word = ((offset - REG_IPRIORITYR_BASE) / 4) as u32;
        if word >= PRIORITY_WORD_COUNT {
            return Err(GicdError::UnknownRegister { offset });
        }
        let base_id = 4 * word;
        if base_id + 3 < SPI_BASE {
            return raz_wi(access);
        }
        if access.is_write {
            let bytes = access.value.to_le_bytes();
            for (byte_offset, byte) in bytes.iter().enumerate() {
                let Some(index) = spi_index(base_id + byte_offset as u32) else {
                    continue;
                };
                self.spi_priority[index] = *byte;
            }
            return Ok(GicdEffect::Applied);
        }
        let mut bytes = [0u8; 4];
        for (byte_offset, byte) in bytes.iter_mut().enumerate() {
            if let Some(index) = spi_index(base_id + byte_offset as u32) {
                *byte = self.spi_priority[index];
            }
        }
        Ok(GicdEffect::ReadValue(u32::from_le_bytes(bytes)))
    }

    /// `GICD_IROUTER<n>` (offset 0x6000 + 8n, RW, one 64-bit register per
    /// interrupt ID): the affinity-routing target, split into a low/high
    /// 32-bit pair the same way `proxima_protocols::virtio::mmio`'s
    /// `QueueDescLow`/`QueueDescHigh` split a 64-bit guest-physical address
    /// — every access this crate's decode layer recovers from a trap is one
    /// 32-bit word, never a genuine 64-bit load/store. IDs below
    /// [`SPI_BASE`] are reserved (RES0): the architecture does not define
    /// per-SGI/PPI routing at the Distributor at all.
    fn apply_router(&mut self, offset: u64, access: GicAccess) -> Result<GicdEffect, GicdError> {
        let rel = offset - REG_IROUTER_BASE;
        let id = (rel / 8) as u32;
        let is_high_half = rel % 8 == 4;
        if id >= NUM_INTERRUPT_IDS {
            return Err(GicdError::UnknownRegister { offset });
        }
        if id < SPI_BASE {
            return raz_wi(access);
        }
        let Some(index) = spi_index(id) else {
            return Err(GicdError::UnknownRegister { offset });
        };
        if access.is_write {
            let route = self.spi_route[index];
            self.spi_route[index] = if is_high_half {
                (route & 0xffff_ffff) | (u64::from(access.value) << 32)
            } else {
                (route & !0xffff_ffff) | u64::from(access.value)
            };
            return Ok(GicdEffect::Applied);
        }
        let route = self.spi_route[index];
        let value = if is_high_half {
            (route >> 32) as u32
        } else {
            route as u32
        };
        Ok(GicdEffect::ReadValue(value))
    }

    /// `GICD_ICFGR<n>` (offset 0x0c00 + 4n, RW): 2 bits per interrupt ID, 16
    /// IDs per word — `Int_config` bit `2i+1` set = edge-triggered, clear =
    /// level (bit `2i` is architecturally RES0, the same convention
    /// [`GicRedistributor::apply_icfgr1`] follows for PPIs). Words 0-1 (IDs
    /// 0-31) are RAZ/WI, the SGI/PPI banking this module's own "ID banking"
    /// doc section names — the observed boot wall's word (word 2, SPIs
    /// 32-47, offset 0x0c08) is the first real word.
    fn apply_icfgr(&mut self, offset: u64, access: GicAccess) -> Result<GicdEffect, GicdError> {
        let word = ((offset - REG_ICFGR_BASE) / 4) as u32;
        if word >= ICFGR_WORD_COUNT {
            return Err(GicdError::UnknownRegister { offset });
        }
        if word < 2 {
            return raz_wi(access);
        }
        let base_spi = 16 * word;
        if access.is_write {
            for id_in_word in 0..16 {
                let Some(index) = spi_index(base_spi + id_in_word) else {
                    break;
                };
                self.spi_edge_triggered[index] = access.value & (1 << (2 * id_in_word + 1)) != 0;
            }
            return Ok(GicdEffect::Applied);
        }
        let mut value = 0u32;
        for id_in_word in 0..16 {
            let Some(index) = spi_index(base_spi + id_in_word) else {
                break;
            };
            if self.spi_edge_triggered[index] {
                value |= 1 << (2 * id_in_word + 1);
            }
        }
        Ok(GicdEffect::ReadValue(value))
    }

    /// `GICD_IGRPMODR<n>` (offset 0x0d00 + 4n, RW): the Group 1
    /// secure/non-secure modifier bitmap — store/echo like every other
    /// per-SPI membership bitmap ([`Self::apply_group_bitmap`]'s own
    /// pattern), since this GICD models a single security state and nothing
    /// downstream reacts to the modifier bit yet. Word 0 is RAZ/WI, the same
    /// SGI/PPI banking [`Self::apply_group_bitmap`] documents.
    fn apply_group_mod_bitmap(
        &mut self,
        offset: u64,
        access: GicAccess,
    ) -> Result<GicdEffect, GicdError> {
        let word = bitmap_word_index(offset, REG_IGRPMODR_BASE)?;
        if word == 0 {
            return raz_wi(access);
        }
        let base_spi = 32 * word;
        if access.is_write {
            for bit in 0..32 {
                let Some(index) = spi_index(base_spi + bit) else {
                    break;
                };
                self.spi_group_mod[index] = access.value & (1 << bit) != 0;
            }
            return Ok(GicdEffect::Applied);
        }
        let mut value = 0u32;
        for bit in 0..32 {
            let Some(index) = spi_index(base_spi + bit) else {
                break;
            };
            if self.spi_group_mod[index] {
                value |= 1 << bit;
            }
        }
        Ok(GicdEffect::ReadValue(value))
    }
}

/// Total architected IDs this GICR's SGI_base frame owns: SGIs (0-15) plus
/// PPIs (16-31), the exact range [`GicDistributor`] RAZ/WI's away — see this
/// module's own "ID banking" doc section above.
const REDISTRIBUTOR_ID_COUNT: usize = SPI_BASE as usize;

/// Number of 32-bit words `GICR_IPRIORITYR` implements for the 32 IDs this
/// Redistributor owns: one byte per ID, four IDs per word.
const REDISTRIBUTOR_PRIORITY_WORD_COUNT: usize = REDISTRIBUTOR_ID_COUNT / 4;

/// First PPI ID; SGIs occupy IDs below this. `GICR_ICFGR0` (SGI config) is
/// architecturally fixed edge-triggered and read-only; `GICR_ICFGR1` (PPI
/// config) is the guest-writable half.
const PPI_BASE: u32 = 16;

/// Size of one GICv3 Redistributor's RD_base frame — a fixed 64KiB per the
/// architecture, matching `src/dtb.rs`'s `SINGLE_VCPU_GICR_SIZE` (0x2_0000)
/// being exactly two of these frames (RD_base then SGI_base) for a
/// single-vCPU guest.
const RD_BASE_FRAME_SIZE: u64 = 0x0001_0000;

/// Offset of the SGI_base frame within one Redistributor's combined window
/// — immediately after RD_base, per the architecture's fixed frame pairing.
const SGI_BASE_OFFSET: u64 = RD_BASE_FRAME_SIZE;

const REG_GICR_CTLR: u64 = 0x0000;
const REG_GICR_IIDR: u64 = 0x0004;
const REG_GICR_TYPER_LOW: u64 = 0x0008;
const REG_GICR_TYPER_HIGH: u64 = 0x000c;
const REG_GICR_WAKER: u64 = 0x0014;
const REG_GICR_PIDR2: u64 = 0xffe8;

const REG_GICR_IGROUPR0: u64 = SGI_BASE_OFFSET + 0x0080;
const REG_GICR_ISENABLER0: u64 = SGI_BASE_OFFSET + 0x0100;
const REG_GICR_ICENABLER0: u64 = SGI_BASE_OFFSET + 0x0180;
const REG_GICR_ISPENDR0: u64 = SGI_BASE_OFFSET + 0x0200;
const REG_GICR_ICPENDR0: u64 = SGI_BASE_OFFSET + 0x0280;
const REG_GICR_IPRIORITYR_BASE: u64 = SGI_BASE_OFFSET + 0x0400;
/// `GICR_ISACTIVER0` (SGI_base offset 0x0300, RW1S): the active-state twin
/// of [`REG_GICR_ISENABLER0`], banked per the same SGI/PPI split
/// [`GicDistributor`]'s own `GICD_ISACTIVER<n>` word 0 RAZ/WI's away — the
/// observed boot wall's write (offset 0x10380 = `SGI_BASE_OFFSET + 0x0380`,
/// the `ICACTIVER0` alias below) is this pair's clear side.
const REG_GICR_ISACTIVER0: u64 = SGI_BASE_OFFSET + 0x0300;
/// `GICR_ICACTIVER0` (SGI_base offset 0x0380, RW1C): clears the same active
/// bit [`REG_GICR_ISACTIVER0`] sets.
const REG_GICR_ICACTIVER0: u64 = SGI_BASE_OFFSET + 0x0380;
const REG_GICR_ICFGR0: u64 = SGI_BASE_OFFSET + 0x0c00;
const REG_GICR_ICFGR1: u64 = SGI_BASE_OFFSET + 0x0c04;
/// `GICR_IGRPMODR0` (SGI_base offset 0x0d00, RW): the banked twin of
/// [`REG_IGRPMODR_BASE`]'s Group 1 security-state modifier bitmap, same
/// store/echo contract as [`GicRedistributor::apply_group_bitmap`] since
/// this GICR models a single security state and nothing downstream reacts
/// to the modifier bit yet.
const REG_GICR_IGRPMODR0: u64 = SGI_BASE_OFFSET + 0x0d00;

/// `GICR_WAKER.ProcessorSleep` (bit 1): the bit Linux's `gic-v3` driver
/// clears at probe to bring this redistributor's PE out of the sleep state
/// the architecture resets it into.
const WAKER_PROCESSOR_SLEEP: u32 = 1 << 1;

/// `GICR_WAKER.ChildrenAsleep` (bit 2): the bit Linux polls (read-only from
/// software's perspective — the architecture defines it as hardware-set)
/// after clearing `ProcessorSleep`, until it reads back clear.
const WAKER_CHILDREN_ASLEEP: u32 = 1 << 2;

/// `GICR_TYPER.Last` (bit 4 of the low word): set when this is the
/// highest-numbered Redistributor in a contiguous series — always set here,
/// since [`GicRedistributor`] models exactly one Redistributor for the
/// single-vCPU guest `src/dtb.rs`'s `QemuVirtLayout::single_vcpu` describes.
const TYPER_LAST: u32 = 1 << 4;

/// What the caller must do in response to one applied [`GicAccess`] against
/// a [`GicRedistributor`] — mirrors [`GicdEffect`]'s shape: a read names the
/// value to write back, a state-changing write names exactly what changed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GicrEffect {
    /// A read register access: the value the caller writes back into the
    /// guest's destination register.
    ReadValue(u32),
    /// A write register access accepted with no further consequence.
    Applied,
    /// A `GICR_CTLR` write changed `EnableLPIs`.
    ControlUpdated { lpis_enabled: bool },
    /// A `GICR_WAKER` write changed the processor-sleep/children-asleep
    /// state — see `GicRedistributor::apply_waker`'s worked-example doc
    /// for the wake-up dance this models.
    WakerUpdated {
        processor_sleep: bool,
        children_asleep: bool,
    },
    /// An `ISENABLER0`/`ICENABLER0` write changed one SGI/PPI's enable
    /// state.
    IdEnableChanged { id: u32, enabled: bool },
    /// An `ISPENDR0`/`ICPENDR0` write changed one SGI/PPI's pending state.
    IdPendingChanged { id: u32, pending: bool },
    /// An `ISACTIVER0`/`ICACTIVER0` write changed one SGI/PPI's active
    /// state — the banked twin of [`GicdEffect::SpiActiveChanged`], same
    /// shape as this enum's own `IdEnableChanged`/`IdPendingChanged`.
    IdActiveChanged { id: u32, active: bool },
}

/// Why [`GicRedistributor::apply`] rejected an access — mirrors
/// [`GicdError`]'s three cases exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GicrError {
    /// No register this GICR implements exists at this offset.
    UnknownRegister { offset: u64 },
    /// The access offset was not naturally aligned to the register's 4-byte
    /// word size.
    UnalignedAccess { offset: u64 },
    /// The register at this offset is read-only; the guest attempted a
    /// write.
    ReadOnlyRegister { offset: u64 },
}

impl core::fmt::Display for GicrError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::UnknownRegister { offset } => {
                write!(formatter, "no gicr register defined at offset {offset:#x}")
            }
            Self::UnalignedAccess { offset } => {
                write!(
                    formatter,
                    "gicr access at offset {offset:#x} is not 4-byte aligned"
                )
            }
            Self::ReadOnlyRegister { offset } => {
                write!(
                    formatter,
                    "gicr register at offset {offset:#x} is read-only"
                )
            }
        }
    }
}

impl core::error::Error for GicrError {}

/// One GICv3 Redistributor's register-block state for a single vCPU: the
/// RD_base frame's `GICR_CTLR`/`GICR_WAKER` bits, plus SGI_base's real
/// per-ID group/enable/pending/priority/config state for the 32 IDs
/// [`GicDistributor`] deliberately RAZ/WI's away (this module's own "ID
/// banking" doc section names the split). LPI state (`GICR_PROPBASER` and
/// friends) is out of scope for this slice — this guest never enables LPIs.
#[derive(Debug, Clone)]
pub struct GicRedistributor {
    lpis_enabled: bool,
    processor_sleep: bool,
    children_asleep: bool,
    id_group1: [bool; REDISTRIBUTOR_ID_COUNT],
    id_group_mod: [bool; REDISTRIBUTOR_ID_COUNT],
    id_enabled: [bool; REDISTRIBUTOR_ID_COUNT],
    id_pending: [bool; REDISTRIBUTOR_ID_COUNT],
    id_active: [bool; REDISTRIBUTOR_ID_COUNT],
    id_priority: [u8; REDISTRIBUTOR_ID_COUNT],
    ppi_edge_triggered: [bool; PPI_BASE as usize],
}

impl Default for GicRedistributor {
    fn default() -> Self {
        Self::new()
    }
}

impl GicRedistributor {
    /// A freshly reset redistributor: `EnableLPIs` clear, `ProcessorSleep`
    /// and `ChildrenAsleep` both set (the architected `GICR_WAKER` reset
    /// state — every PE resets asleep until software wakes it), every SGI
    /// and PPI in group 0 with no group-1 security modifier, disabled, not
    /// pending, not active, priority 0 (highest), and every PPI
    /// level-triggered (the architected `GICR_ICFGR1` reset value).
    #[must_use]
    pub const fn new() -> Self {
        Self {
            lpis_enabled: false,
            processor_sleep: true,
            children_asleep: true,
            id_group1: [false; REDISTRIBUTOR_ID_COUNT],
            id_group_mod: [false; REDISTRIBUTOR_ID_COUNT],
            id_enabled: [false; REDISTRIBUTOR_ID_COUNT],
            id_pending: [false; REDISTRIBUTOR_ID_COUNT],
            id_active: [false; REDISTRIBUTOR_ID_COUNT],
            id_priority: [0; REDISTRIBUTOR_ID_COUNT],
            ppi_edge_triggered: [false; PPI_BASE as usize],
        }
    }

    #[must_use]
    pub fn lpis_enabled(&self) -> bool {
        self.lpis_enabled
    }

    #[must_use]
    pub fn processor_sleep(&self) -> bool {
        self.processor_sleep
    }

    #[must_use]
    pub fn children_asleep(&self) -> bool {
        self.children_asleep
    }

    #[must_use]
    pub fn id_enabled(&self, id: u32) -> Option<bool> {
        self.id_enabled.get(id as usize).copied()
    }

    #[must_use]
    pub fn id_pending(&self, id: u32) -> Option<bool> {
        self.id_pending.get(id as usize).copied()
    }

    #[must_use]
    pub fn id_active(&self, id: u32) -> Option<bool> {
        self.id_active.get(id as usize).copied()
    }

    #[must_use]
    pub fn id_priority(&self, id: u32) -> Option<u8> {
        self.id_priority.get(id as usize).copied()
    }

    /// Apply one register access, returning the effect the caller must carry
    /// out. `access.offset` spans this Redistributor's whole combined
    /// window (RD_base then SGI_base, `RD_BASE_FRAME_SIZE` apart) — the
    /// same flattened addressing [`GicDistributor::apply`] uses for its own
    /// single window.
    pub fn apply(&mut self, access: GicAccess) -> Result<GicrEffect, GicrError> {
        if !access.offset.is_multiple_of(4) {
            return Err(GicrError::UnalignedAccess {
                offset: access.offset,
            });
        }

        match access.offset {
            REG_GICR_CTLR => self.apply_ctlr(access),
            REG_GICR_IIDR => gicr_read_only(access, 0),
            REG_GICR_TYPER_LOW => gicr_read_only(access, self.read_typer_low()),
            REG_GICR_TYPER_HIGH => gicr_read_only(access, self.read_typer_high()),
            REG_GICR_WAKER => self.apply_waker(access),
            REG_GICR_PIDR2 => gicr_read_only(access, GICD_ARCH_REV_GICV3 << 4),
            REG_GICR_IGROUPR0 => self.apply_group_bitmap(access),
            REG_GICR_ISENABLER0 => self.apply_enable_bitmap(access, true),
            REG_GICR_ICENABLER0 => self.apply_enable_bitmap(access, false),
            REG_GICR_ISPENDR0 => self.apply_pending_bitmap(access, true),
            REG_GICR_ICPENDR0 => self.apply_pending_bitmap(access, false),
            REG_GICR_ISACTIVER0 => self.apply_active_bitmap(access, true),
            REG_GICR_ICACTIVER0 => self.apply_active_bitmap(access, false),
            REG_GICR_ICFGR0 => gicr_read_only(access, sgi_icfgr0_reset_value()),
            REG_GICR_ICFGR1 => self.apply_icfgr1(access),
            REG_GICR_IGRPMODR0 => self.apply_group_mod_bitmap(access),
            offset if in_redistributor_priority_range(offset) => {
                self.apply_priority(offset, access)
            }
            offset => Err(GicrError::UnknownRegister { offset }),
        }
    }

    /// `GICR_CTLR` (offset 0x0000, RW). Only bit 0 `EnableLPIs` is modeled —
    /// this guest never enables LPIs, so this bit is tracked and echoed back
    /// but nothing downstream reacts to it yet.
    fn apply_ctlr(&mut self, access: GicAccess) -> Result<GicrEffect, GicrError> {
        if !access.is_write {
            return Ok(GicrEffect::ReadValue(u32::from(self.lpis_enabled)));
        }
        self.lpis_enabled = access.value & 0b1 != 0;
        Ok(GicrEffect::ControlUpdated {
            lpis_enabled: self.lpis_enabled,
        })
    }

    /// `GICR_TYPER` low word (offset 0x0008, RO). [`TYPER_LAST`] is always
    /// set (single Redistributor); `Affinity_Value` (this register's high
    /// word) matches [`crate::psci::RESIDENT_MPIDR_AFFINITY`] (0) — the same
    /// single core PSCI, the DTB's `cpu@0`, and this Redistributor all name
    /// as the one resident PE. `DirectLPI`/`Dirty`/`VLPIS`/`PLPIS` (bits
    /// 3-0) report 0: this slice models no LPI support.
    fn read_typer_low(&self) -> u32 {
        TYPER_LAST
    }

    /// `GICR_TYPER` high word (offset 0x000c, RO): `Affinity_Value` in bits
    /// \[31:0\] of this word (`GICR_TYPER`'s bits \[63:32\] overall).
    /// [`crate::psci::RESIDENT_MPIDR_AFFINITY`] is 0, so this word is 0 —
    /// the same affinity `cpu@0`'s MPIDR and PSCI's resident-core check
    /// both use.
    fn read_typer_high(&self) -> u32 {
        crate::psci::RESIDENT_MPIDR_AFFINITY as u32
    }

    /// `GICR_WAKER` (offset 0x0014, RW). Models the architected wake-up
    /// dance: [`WAKER_PROCESSOR_SLEEP`] and [`WAKER_CHILDREN_ASLEEP`] both
    /// reset set (every PE starts asleep). Linux's `gic-v3` driver clears
    /// `ProcessorSleep` and polls `ChildrenAsleep` until hardware clears it;
    /// this software model has no power domain to poll away, so clearing
    /// `ProcessorSleep` on a write clears `ChildrenAsleep` synchronously in
    /// the same write, and re-asserting `ProcessorSleep` re-asserts
    /// `ChildrenAsleep` — a real GIC's `ChildrenAsleep` is hardware-owned
    /// and would lag by some implementation-defined latency a guest must
    /// poll for; this model has none to simulate.
    fn apply_waker(&mut self, access: GicAccess) -> Result<GicrEffect, GicrError> {
        if !access.is_write {
            let mut value = 0u32;
            if self.processor_sleep {
                value |= WAKER_PROCESSOR_SLEEP;
            }
            if self.children_asleep {
                value |= WAKER_CHILDREN_ASLEEP;
            }
            return Ok(GicrEffect::ReadValue(value));
        }
        self.processor_sleep = access.value & WAKER_PROCESSOR_SLEEP != 0;
        self.children_asleep = self.processor_sleep;
        Ok(GicrEffect::WakerUpdated {
            processor_sleep: self.processor_sleep,
            children_asleep: self.children_asleep,
        })
    }

    /// `GICR_IGROUPR0` (SGI_base offset 0x0080, RW): one bit per SGI/PPI ID,
    /// 1 = Group 1, 0 = Group 0 — the real state [`GicDistributor`]'s own
    /// `IGROUPR<n>` word 0 RAZ/WI's away.
    fn apply_group_bitmap(&mut self, access: GicAccess) -> Result<GicrEffect, GicrError> {
        if access.is_write {
            for (id, group1) in self.id_group1.iter_mut().enumerate() {
                *group1 = access.value & (1 << id) != 0;
            }
            return Ok(GicrEffect::Applied);
        }
        Ok(GicrEffect::ReadValue(pack_bits(&self.id_group1)))
    }

    /// `GICR_ISENABLER0`/`GICR_ICENABLER0` (SGI_base offsets 0x0100/0x0180,
    /// RW1S/RW1C): the enable-state twin of [`GicDistributor`]'s own
    /// `ISENABLER<n>`/`ICENABLER<n>`, for the SGI/PPI IDs the Distributor
    /// RAZ/WI's at word 0.
    fn apply_enable_bitmap(
        &mut self,
        access: GicAccess,
        set_bits: bool,
    ) -> Result<GicrEffect, GicrError> {
        if access.is_write {
            let mut last_changed = None;
            for (id, enabled) in self.id_enabled.iter_mut().enumerate() {
                if access.value & (1 << id) == 0 {
                    continue;
                }
                *enabled = set_bits;
                last_changed = Some(id as u32);
            }
            return match last_changed {
                Some(id) => Ok(GicrEffect::IdEnableChanged {
                    id,
                    enabled: set_bits,
                }),
                None => Ok(GicrEffect::Applied),
            };
        }
        Ok(GicrEffect::ReadValue(pack_bits(&self.id_enabled)))
    }

    /// `GICR_ISPENDR0`/`GICR_ICPENDR0` (SGI_base offsets 0x0200/0x0280,
    /// RW1S/RW1C): the pending-state twin of [`Self::apply_enable_bitmap`].
    fn apply_pending_bitmap(
        &mut self,
        access: GicAccess,
        set_bits: bool,
    ) -> Result<GicrEffect, GicrError> {
        if access.is_write {
            let mut last_changed = None;
            for (id, pending) in self.id_pending.iter_mut().enumerate() {
                if access.value & (1 << id) == 0 {
                    continue;
                }
                *pending = set_bits;
                last_changed = Some(id as u32);
            }
            return match last_changed {
                Some(id) => Ok(GicrEffect::IdPendingChanged {
                    id,
                    pending: set_bits,
                }),
                None => Ok(GicrEffect::Applied),
            };
        }
        Ok(GicrEffect::ReadValue(pack_bits(&self.id_pending)))
    }

    /// `GICR_ISACTIVER0`/`GICR_ICACTIVER0` (SGI_base offsets 0x0300/0x0380,
    /// RW1S/RW1C): the active-state twin of [`Self::apply_enable_bitmap`],
    /// same shared-bit contract — the observed boot wall
    /// (`offset=0x10380`, `SGI_BASE_OFFSET + 0x0380`) is this pair's clear
    /// side.
    fn apply_active_bitmap(
        &mut self,
        access: GicAccess,
        set_bits: bool,
    ) -> Result<GicrEffect, GicrError> {
        if access.is_write {
            let mut last_changed = None;
            for (id, active) in self.id_active.iter_mut().enumerate() {
                if access.value & (1 << id) == 0 {
                    continue;
                }
                *active = set_bits;
                last_changed = Some(id as u32);
            }
            return match last_changed {
                Some(id) => Ok(GicrEffect::IdActiveChanged {
                    id,
                    active: set_bits,
                }),
                None => Ok(GicrEffect::Applied),
            };
        }
        Ok(GicrEffect::ReadValue(pack_bits(&self.id_active)))
    }

    /// `GICR_IGRPMODR0` (SGI_base offset 0x0d00, RW): the banked Group 1
    /// security-state modifier bitmap — store/echo like
    /// [`Self::apply_group_bitmap`], since this GICR models a single
    /// security state and nothing downstream reacts to the modifier bit
    /// yet.
    fn apply_group_mod_bitmap(&mut self, access: GicAccess) -> Result<GicrEffect, GicrError> {
        if access.is_write {
            for (id, group_mod) in self.id_group_mod.iter_mut().enumerate() {
                *group_mod = access.value & (1 << id) != 0;
            }
            return Ok(GicrEffect::Applied);
        }
        Ok(GicrEffect::ReadValue(pack_bits(&self.id_group_mod)))
    }

    /// `GICR_IPRIORITYR<n>` (SGI_base offset 0x0400 + 4n, n=0..7, RW): one
    /// byte per SGI/PPI ID, four IDs packed little-endian per word — the
    /// priority-state twin of [`GicDistributor::apply_priority`] for the IDs
    /// the Distributor RAZ/WI's.
    fn apply_priority(&mut self, offset: u64, access: GicAccess) -> Result<GicrEffect, GicrError> {
        let word = ((offset - REG_GICR_IPRIORITYR_BASE) / 4) as usize;
        let base_id = 4 * word;
        if access.is_write {
            let bytes = access.value.to_le_bytes();
            for (byte_offset, byte) in bytes.iter().enumerate() {
                self.id_priority[base_id + byte_offset] = *byte;
            }
            return Ok(GicrEffect::Applied);
        }
        let mut bytes = [0u8; 4];
        for (byte_offset, byte) in bytes.iter_mut().enumerate() {
            *byte = self.id_priority[base_id + byte_offset];
        }
        Ok(GicrEffect::ReadValue(u32::from_le_bytes(bytes)))
    }

    /// `GICR_ICFGR1` (SGI_base offset 0x0c04, RW): 2 bits per PPI
    /// (`Int_config`, bit `2n+1` set = edge-triggered, clear = level —
    /// `2n` is architecturally RES0). `GICR_ICFGR0` (the SGI half) is fixed
    /// edge-triggered and read-only; SGIs have no configurable trigger.
    fn apply_icfgr1(&mut self, access: GicAccess) -> Result<GicrEffect, GicrError> {
        if access.is_write {
            for (ppi, edge) in self.ppi_edge_triggered.iter_mut().enumerate() {
                *edge = access.value & (1 << (2 * ppi + 1)) != 0;
            }
            return Ok(GicrEffect::Applied);
        }
        let mut value = 0u32;
        for (ppi, edge) in self.ppi_edge_triggered.iter().enumerate() {
            if *edge {
                value |= 1 << (2 * ppi + 1);
            }
        }
        Ok(GicrEffect::ReadValue(value))
    }
}

/// GICv3 CPU interface — the third register block this module names in its
/// own doc's "M5b GIC ordering" ("the CPU interface is HVF-trap territory
/// investigated later"). Unlike [`GicDistributor`]/[`GicRedistributor`],
/// which the guest reaches through ordinary MMIO loads/stores against a
/// mapped window, this block is reached through `MRS`/`MSR` against system
/// registers (ARM DDI 0487 D17.6, "AArch64 System register access
/// instructions") — the trap the exit loop reports as exception class
/// `0x18` rather than `0x24`, and this module's own `is_narrow`/`offset`
/// vocabulary does not apply: an [`IccAccess`] is keyed by the same
/// `(op0, op1, CRn, CRm, op2)` tuple the architecture uses to name a system
/// register, not a byte offset. [`IccCpuInterface::apply`] mirrors
/// [`GicDistributor::apply`]'s and [`GicRedistributor::apply`]'s shape
/// exactly — one raw access in, one typed effect or error out, a single
/// match — for the same "no cursor, no I/O" reason.
///
/// EC 0x18 is not GICv3-specific — it is ARM's single trap class for every
/// trapped `MSR`/`MRS`, and HVF routes debug-register traps through the
/// identical path as GICv3 CPU-interface traps (no second exit reason, no
/// second dispatch site). This VM's own second EC 0x18 wall proved it: the
/// tuple `S2_0_C1_C3_4` (`op0=2`) is architecturally outside the ICC space
/// entirely (every ICC register in this module uses `op0=3`) — ARM DDI 0487
/// D17.2's external-debug system-register encoding table names `op0=2,
/// op1=0, CRn=1, CRm=3, op2=4` as `OSDLR_EL1` (OS Double Lock Register),
/// which Linux's `arch/arm64/kernel/debug-monitors.c` CPU-bringup path
/// (`reset_os_lock`) writes at every CPU's online transition. Rather than
/// mint a second tuple-keyed dispatch site for "debug register traps" next
/// to this one, [`IccCpuInterface::apply`] models `OSDLR_EL1` alongside the
/// ICC registers it already owns — the dispatch mechanism (raw tuple in,
/// typed effect or error out) is identical either way, and the exit loop
/// has exactly one EC 0x18 handler to route both families through.
///
/// # Minimum register set
///
/// The set below is exactly what Linux's `gic_cpu_sys_reg_init`
/// (`drivers/irqchip/irq-gic-v3.c`) touches at CPU-interface bring-up:
/// `ICC_SRE_EL1` (probed first — Linux writes then reads back to confirm
/// sysreg mode, so this bit must read 1 and stick, never clear on
/// readback), `ICC_PMR_EL1` (priority mask), `ICC_BPR1_EL1` (binary point),
/// `ICC_CTLR_EL1` (EOI mode and friends), `ICC_IGRPEN1_EL1` (group-1
/// enable), `ICC_IAR1_EL1` (interrupt acknowledge — this VM injects no
/// interrupts yet, so every read reports [`ICC_IAR1_SPURIOUS`]),
/// `ICC_EOIR1_EL1` (end-of-interrupt, write-only, accepted with no further
/// effect since no real interrupt was ever acknowledged), and
/// `ICC_AP1R0_EL1` (active-priorities register, probed by some kernels'
/// save/restore paths even with no active interrupt). `OSDLR_EL1`
/// (`S2_0_C1_C3_4`) joins this set for a different reason: it is not a
/// GICv3 register at all, but Linux's `reset_os_lock` CPU-bringup path
/// writes it over the identical EC 0x18 trap, so this VM's boot cannot get
/// past CPU 0 online without it modeled (see the struct doc above).
/// `ICC_SGI1R_EL1` (`S3_0_C12_C11_5`) joins this set for the next wall past
/// CPU 0 online: `gic_raise_softirq`
/// (the SMP IPI path, `arch_send_call_function_single_ipi` and friends)
/// writes it over this exact trap the moment the guest issues its first
/// inter-processor interrupt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IccAccess {
    pub op0: u8,
    pub op1: u8,
    pub crn: u8,
    pub crm: u8,
    pub op2: u8,
    pub is_write: bool,
    pub value: u64,
}

/// What the caller must do in response to one applied [`IccAccess`] —
/// mirrors [`GicdEffect`]/[`GicrEffect`]'s shape: a read names the value to
/// write back into the guest's destination register, a write is either
/// accepted with no further consequence or names what changed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IccEffect {
    /// A read access: the value the caller writes back into the guest's
    /// destination register (`Rt`).
    ReadValue(u64),
    /// A write access accepted with no further consequence.
    Applied,
    /// An `ICC_EOIR1_EL1` write that matched this interface's one active
    /// interrupt (the INTID named). The caller owes HVF two calls this
    /// effect alone signals are now due: `hv_vcpu_set_pending_interrupt`
    /// (false) and `hv_vcpu_set_vtimer_mask` (false) when the deactivated
    /// INTID is the vtimer's — the architected "servicing this interrupt is
    /// complete" contract `hv_vcpu_set_vtimer_mask`'s own SDK doc names.
    InterruptDeactivated(u32),
}

/// Why [`IccCpuInterface::apply`] rejected an access. Unlike
/// [`GicdError`]/[`GicrError`] (byte-offset space), the unknown case here
/// names the full architected register encoding — the next wall this
/// module cannot yet decode must decode itself the same way this one did
/// (this module's own doc: "the next wall must decode itself").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IccError {
    /// No ICC system register this interface implements exists at this
    /// `(op0, op1, CRn, CRm, op2)` encoding.
    UnknownRegister {
        op0: u8,
        op1: u8,
        crn: u8,
        crm: u8,
        op2: u8,
    },
    /// The register at this encoding is read-only; the guest attempted a
    /// write (`ICC_IAR1_EL1`).
    ReadOnlyRegister {
        op0: u8,
        op1: u8,
        crn: u8,
        crm: u8,
        op2: u8,
    },
    /// The register at this encoding is write-only; the guest attempted a
    /// read (`ICC_EOIR1_EL1`).
    WriteOnlyRegister {
        op0: u8,
        op1: u8,
        crn: u8,
        crm: u8,
        op2: u8,
    },
}

impl core::fmt::Display for IccError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::UnknownRegister {
                op0,
                op1,
                crn,
                crm,
                op2,
            } => write!(
                formatter,
                "no icc register defined at S{op0}_{op1}_C{crn}_C{crm}_{op2}"
            ),
            Self::ReadOnlyRegister {
                op0,
                op1,
                crn,
                crm,
                op2,
            } => write!(
                formatter,
                "icc register S{op0}_{op1}_C{crn}_C{crm}_{op2} is read-only"
            ),
            Self::WriteOnlyRegister {
                op0,
                op1,
                crn,
                crm,
                op2,
            } => write!(
                formatter,
                "icc register S{op0}_{op1}_C{crn}_C{crm}_{op2} is write-only"
            ),
        }
    }
}

impl core::error::Error for IccError {}

/// `ICC_IAR1_EL1`'s spurious-interrupt sentinel (ARM IHI 0069, "1023
/// Spurious"): the value Linux's GIC driver treats as "no interrupt
/// pending" when the acknowledge register is read with nothing asserted —
/// exactly this VM's own state, since it injects no interrupts.
pub const ICC_IAR1_SPURIOUS: u64 = 1023;

/// This interface's one-deep interrupt-pending slot (M5b-beyond scope: one
/// vCPU, one interrupt source — the virtual timer — so a single slot is the
/// honest model, not a full GICv3 priority-ordered list-register array).
/// `Idle` — nothing pending or active. `Pending` — `set_pending` recorded an
/// INTID the guest has not yet acknowledged via `ICC_IAR1_EL1`. `Active` —
/// an `ICC_IAR1_EL1` read acknowledged it (ARM IHI 0069 4.1.1's pending →
/// active transition); it stays active until a matching `ICC_EOIR1_EL1`
/// write retires it back to `Idle`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InterruptState {
    Idle,
    Pending(u32),
    Active(u32),
}

/// One vCPU's GICv3 CPU-interface system-register file — the minimum set
/// [`IccCpuInterface`]'s own module doc names.
#[derive(Debug, Clone)]
pub struct IccCpuInterface {
    pmr: u8,
    bpr1: u8,
    ctlr: u32,
    group1_enabled: bool,
    ap1r0: u32,
    /// `OSDLR_EL1.DLK` (bit 0), RES0 elsewhere — the debug-architecture OS
    /// double-lock, not a GICv3 register at all (see [`IccCpuInterface`]'s
    /// own module doc for the EC-0x18 evidence). Reset 0 (unlocked): the
    /// architected reset value is UNKNOWN, and 0 is the value Linux's own
    /// `reset_os_lock` writes at every CPU bring-up, so a guest reading it
    /// back before its own first write sees the value it is about to set
    /// anyway.
    osdlr: u32,
    /// [`InterruptState`]'s own doc names the scope this field holds to.
    pending_interrupt: InterruptState,
}

impl Default for IccCpuInterface {
    fn default() -> Self {
        Self::new()
    }
}

impl IccCpuInterface {
    /// A freshly reset CPU interface: `ICC_PMR_EL1` at reset value 0 (mask
    /// nothing, per the architected power-on reset — a guest must
    /// explicitly raise the mask to filter), `ICC_BPR1_EL1` at its
    /// architected reset value 2 (fewest group priority bits, most subgroup
    /// bits — the minimum-capability reset a guest must widen if it wants
    /// more), `ICC_CTLR_EL1` and `ICC_AP1R0_EL1` both 0, group 1 disabled.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            pmr: 0,
            bpr1: 2,
            ctlr: 0,
            group1_enabled: false,
            ap1r0: 0,
            osdlr: 0,
            pending_interrupt: InterruptState::Idle,
        }
    }

    #[must_use]
    pub fn priority_mask(&self) -> u8 {
        self.pmr
    }

    #[must_use]
    pub fn group1_enabled(&self) -> bool {
        self.group1_enabled
    }

    /// Records `intid` pending in this interface's one-deep slot
    /// (`InterruptState`'s own doc names the scope). The HVF exit loop
    /// calls this the instant `HV_EXIT_REASON_VTIMER_ACTIVATED` fires
    /// (`backend_macos.c`), before telling HVF the guest's IRQ line is
    /// asserted via `hv_vcpu_set_pending_interrupt`. Overwrites whatever was
    /// recorded before — out of scope for this slice to arbitrate between
    /// two pending sources, and the only source this model has (the vtimer)
    /// cannot re-fire until its own EOI clears the mask, so a genuine
    /// overwrite never happens on the boot path this slice measured.
    pub fn set_pending(&mut self, intid: u32) {
        self.pending_interrupt = InterruptState::Pending(intid);
    }

    /// Apply one register access, returning the effect the caller must
    /// carry out. Mirrors [`GicDistributor::apply`]'s single match over the
    /// whole register map — every register here is a fixed value at a fixed
    /// encoding, never a variable-length or streamed field.
    pub fn apply(&mut self, access: IccAccess) -> Result<IccEffect, IccError> {
        match (access.op0, access.op1, access.crn, access.crm, access.op2) {
            (3, 0, 4, 6, 0) => self.apply_pmr(access),
            (3, 0, 12, 12, 3) => self.apply_bpr1(access),
            (3, 0, 12, 12, 4) => self.apply_ctlr(access),
            (3, 0, 12, 12, 5) => self.apply_sre(access),
            (3, 0, 12, 12, 7) => self.apply_igrpen1(access),
            (3, 0, 12, 12, 0) => self.apply_iar1(access),
            (3, 0, 12, 12, 1) => self.apply_eoir1(access),
            (3, 0, 12, 9, 0) => self.apply_ap1r0(access),
            (3, 0, 12, 11, 5) => self.apply_sgi1r(access),
            (2, 0, 1, 3, 4) => self.apply_osdlr(access),
            (2, 0, 1, 0, 4) => self.apply_oslar(access),
            (op0, op1, crn, crm, op2) => Err(IccError::UnknownRegister {
                op0,
                op1,
                crn,
                crm,
                op2,
            }),
        }
    }

    /// `ICC_PMR_EL1` (`S3_0_C4_C6_0`, RW): the priority mask — this VM's own
    /// first-decoded wall (syndrome `0x6230102d`, `pc=0xffff8000807da100`,
    /// a `MRS` read of this exact register during Linux's GIC CPU-interface
    /// probe).
    fn apply_pmr(&mut self, access: IccAccess) -> Result<IccEffect, IccError> {
        if !access.is_write {
            return Ok(IccEffect::ReadValue(u64::from(self.pmr)));
        }
        self.pmr = access.value as u8;
        Ok(IccEffect::Applied)
    }

    /// `ICC_BPR1_EL1` (`S3_0_C12_C12_3`, RW): the group-1 binary point,
    /// store/echo with no further consequence since this VM raises no
    /// interrupts yet to subgroup.
    fn apply_bpr1(&mut self, access: IccAccess) -> Result<IccEffect, IccError> {
        if !access.is_write {
            return Ok(IccEffect::ReadValue(u64::from(self.bpr1)));
        }
        self.bpr1 = access.value as u8;
        Ok(IccEffect::Applied)
    }

    /// `ICC_CTLR_EL1` (`S3_0_C12_C12_4`, RW): EOI mode and the other control
    /// bits, store/echo verbatim — no field here changes how this VM
    /// services an interrupt, since it raises none yet.
    fn apply_ctlr(&mut self, access: IccAccess) -> Result<IccEffect, IccError> {
        if !access.is_write {
            return Ok(IccEffect::ReadValue(u64::from(self.ctlr)));
        }
        self.ctlr = access.value as u32;
        Ok(IccEffect::Applied)
    }

    /// `ICC_SRE_EL1` (`S3_0_C12_C12_5`, RW): system register enable. `SRE`
    /// (bit 0) MUST read 1 and stay 1 — Linux's probe writes it then reads
    /// back to confirm sysreg mode; a clear-on-readback here would send the
    /// driver down the memory-mapped CPU-interface path, which this VM does
    /// not implement at all. A write is accepted (echoed back set) rather
    /// than rejected, since the architecture defines `SRE` as sticky, not
    /// read-only — Linux's own probe performs the write unconditionally.
    fn apply_sre(&mut self, access: IccAccess) -> Result<IccEffect, IccError> {
        if access.is_write {
            return Ok(IccEffect::Applied);
        }
        Ok(IccEffect::ReadValue(1))
    }

    /// `ICC_IGRPEN1_EL1` (`S3_0_C12_C12_7`, RW): group-1 enable, the last
    /// step of Linux's CPU-interface bring-up dance.
    fn apply_igrpen1(&mut self, access: IccAccess) -> Result<IccEffect, IccError> {
        if !access.is_write {
            return Ok(IccEffect::ReadValue(u64::from(self.group1_enabled)));
        }
        self.group1_enabled = access.value & 0b1 != 0;
        Ok(IccEffect::Applied)
    }

    /// `ICC_IAR1_EL1` (`S3_0_C12_C12_0`, RO): interrupt acknowledge. A guest
    /// write here is architecturally undefined behaviour on real hardware;
    /// this model names it a decode error instead of silently accepting it.
    /// A read against [`InterruptState::Pending`] acknowledges it (ARM IHI
    /// 0069 4.1.1: pending → active) and reports the INTID. A read with
    /// nothing newly pending — [`InterruptState::Idle`], or
    /// [`InterruptState::Active`] with no second source this one-deep model
    /// could ever hold — reports [`ICC_IAR1_SPURIOUS`] (ARM IHI 0069 4.1.1:
    /// "a read of ICC_IAR1_EL1 returns 1023 ... if there is no pending
    /// interrupt with sufficient priority").
    fn apply_iar1(&mut self, access: IccAccess) -> Result<IccEffect, IccError> {
        if access.is_write {
            return Err(IccError::ReadOnlyRegister {
                op0: access.op0,
                op1: access.op1,
                crn: access.crn,
                crm: access.crm,
                op2: access.op2,
            });
        }
        match self.pending_interrupt {
            InterruptState::Pending(intid) => {
                self.pending_interrupt = InterruptState::Active(intid);
                Ok(IccEffect::ReadValue(u64::from(intid)))
            }
            InterruptState::Idle | InterruptState::Active(_) => Ok(IccEffect::ReadValue(ICC_IAR1_SPURIOUS)),
        }
    }

    /// `ICC_EOIR1_EL1` (`S3_0_C12_C12_1`, WO): end of interrupt. A write
    /// naming this interface's one active INTID retires it to
    /// [`InterruptState::Idle`] and reports
    /// [`IccEffect::InterruptDeactivated`] so the caller can service the
    /// architected re-arm contract (`hv_vcpu_set_pending_interrupt` /
    /// `hv_vcpu_set_vtimer_mask`, both false). A write naming any other
    /// value — nothing active, or an INTID that does not match — is a
    /// spurious EOI: real guest behaviour (ARM IHI 0069 4.1.4 treats it as
    /// implementation-defined, not a fault), accepted with no state change
    /// rather than surfaced as a protocol violation.
    fn apply_eoir1(&mut self, access: IccAccess) -> Result<IccEffect, IccError> {
        if !access.is_write {
            return Err(IccError::WriteOnlyRegister {
                op0: access.op0,
                op1: access.op1,
                crn: access.crn,
                crm: access.crm,
                op2: access.op2,
            });
        }
        let acknowledged_intid = access.value as u32;
        if let InterruptState::Active(intid) = self.pending_interrupt
            && intid == acknowledged_intid
        {
            self.pending_interrupt = InterruptState::Idle;
            return Ok(IccEffect::InterruptDeactivated(intid));
        }
        Ok(IccEffect::Applied)
    }

    /// `ICC_AP1R0_EL1` (`S3_0_C12_C9_0`, RW): group-1 active-priorities
    /// register, store/echo with no further consequence for the same reason
    /// [`Self::apply_bpr1`] is a plain store.
    fn apply_ap1r0(&mut self, access: IccAccess) -> Result<IccEffect, IccError> {
        if !access.is_write {
            return Ok(IccEffect::ReadValue(u64::from(self.ap1r0)));
        }
        self.ap1r0 = access.value as u32;
        Ok(IccEffect::Applied)
    }

    /// `ICC_SGI1R_EL1` (`S3_0_C12_C11_5`, WO): software-generated-interrupt
    /// distribution register — Linux's `gic_raise_softirq`
    /// (`drivers/irqchip/irq-gic-v3.c`) writes this to send an SGI to one or
    /// more PEs, ARM IHI 0069 §5.3 / DDI 0487 `ICC_SGI1R_EL1`'s encoding:
    /// `INTID` in bits\[27:24\] (an SGI number, always < 16), `IRM` (bit 40,
    /// Interrupt Routing Mode) selecting between an affinity-routed target
    /// list (`IRM` = 0: `Aff3`\[55:48\]/`Aff2`\[39:32\]/`Aff1`\[23:16\] name the
    /// cluster, `TargetList`\[15:0\] is a bitmap of `Aff0` values within it)
    /// and "every PE in the system" (`IRM` = 1). A read is architecturally
    /// UNDEFINED (mirrors [`Self::apply_eoir1`]'s write-only rejection).
    ///
    /// This VM models exactly one PE, resident at affinity 0
    /// ([`crate::psci::RESIDENT_MPIDR_AFFINITY`]). The one target this
    /// interface can ever resolve to is that PE, so `IRM` = 1 ("every other
    /// PE in the system") is treated as reaching it too — the strict
    /// architected reading excludes the requesting PE, which on a
    /// single-PE system would mean broadcast reaches nobody, but a real
    /// multi-PE system's guest kernel expects a broadcast SGI to be
    /// serviceable by every PE that receives the trap, and this model's
    /// only receiver is CPU 0. An `IRM` = 0 write resolves to CPU 0 only
    /// when `Aff1`/`Aff2`/`Aff3` are all 0 (this VM's own affinity) and
    /// `TargetList` bit 0 is set; any other encoding names a PE this VM
    /// never modeled and is accepted as a no-op, matching real hardware's
    /// own "write ignored for absent targets" behaviour rather than
    /// surfaced as an error.
    fn apply_sgi1r(&mut self, access: IccAccess) -> Result<IccEffect, IccError> {
        if !access.is_write {
            return Err(IccError::WriteOnlyRegister {
                op0: access.op0,
                op1: access.op1,
                crn: access.crn,
                crm: access.crm,
                op2: access.op2,
            });
        }
        let value = access.value;
        let intid = ((value >> 24) & 0xf) as u32;
        let is_routing_mode_broadcast = (value >> 40) & 0x1 != 0;
        let aff1 = (value >> 16) & 0xff;
        let aff2 = (value >> 32) & 0xff;
        let aff3 = (value >> 48) & 0xff;
        let target_list = value & 0xffff;
        let targets_this_cpu = is_routing_mode_broadcast
            || (aff1 == 0 && aff2 == 0 && aff3 == 0 && target_list & 0b1 != 0);
        if targets_this_cpu {
            self.set_pending(intid);
        }
        Ok(IccEffect::Applied)
    }

    /// `OSDLR_EL1` (`S2_0_C1_C3_4`, RW, debug architecture, not GICv3 —
    /// this struct's own module doc explains why it lives here): the OS
    /// double-lock this VM's own second EC 0x18 wall named
    /// (`S2_0_C1_C3_4 write`, Linux's `reset_os_lock` CPU-bringup path).
    /// Plain store/echo with no further consequence — this VM implements
    /// no external-debug functionality for the double lock to gate, so a
    /// guest setting or clearing `DLK` observes exactly the bit it wrote,
    /// same shape as [`Self::apply_bpr1`]'s plain store.
    fn apply_osdlr(&mut self, access: IccAccess) -> Result<IccEffect, IccError> {
        if !access.is_write {
            return Ok(IccEffect::ReadValue(u64::from(self.osdlr)));
        }
        self.osdlr = access.value as u32;
        Ok(IccEffect::Applied)
    }

    /// `OSLAR_EL1` (`S2_0_C1_C0_4`, WO, debug architecture): the OS Lock
    /// Access Register `reset_os_lock` writes in the same CPU-bringup call
    /// as `OSDLR_EL1` above — this VM's own third EC 0x18 wall, the next
    /// one past `OSDLR_EL1`. Architecturally write-only (an `MRS` against
    /// this encoding is UNDEFINED, mirrored the same way
    /// [`Self::apply_eoir1`] rejects a read of `ICC_EOIR1_EL1`); the write
    /// itself is accepted with no further consequence, since this VM
    /// implements no external-debug lock for `OSLK` to gate.
    fn apply_oslar(&self, access: IccAccess) -> Result<IccEffect, IccError> {
        if !access.is_write {
            return Err(IccError::WriteOnlyRegister {
                op0: access.op0,
                op1: access.op1,
                crn: access.crn,
                crm: access.crm,
                op2: access.op2,
            });
        }
        Ok(IccEffect::Applied)
    }
}

/// `GICR_ICFGR0`'s fixed reset value: every SGI (0-15) architecturally
/// edge-triggered, `Int_config` bit `2n+1` set for all 16 IDs.
fn sgi_icfgr0_reset_value() -> u32 {
    let mut value = 0u32;
    for sgi in 0..PPI_BASE {
        value |= 1 << (2 * sgi + 1);
    }
    value
}

/// Packs a per-ID boolean array into one 32-bit bitmap word, bit `n` set
/// when `bits[n]` is true — the read-side twin of every SGI/PPI bitmap
/// register's write loop above.
fn pack_bits(bits: &[bool; REDISTRIBUTOR_ID_COUNT]) -> u32 {
    let mut value = 0u32;
    for (id, bit) in bits.iter().enumerate() {
        if *bit {
            value |= 1 << id;
        }
    }
    value
}

fn in_redistributor_priority_range(offset: u64) -> bool {
    offset >= REG_GICR_IPRIORITYR_BASE
        && offset < REG_GICR_IPRIORITYR_BASE + REDISTRIBUTOR_PRIORITY_WORD_COUNT as u64 * 4
}

/// A read-only GICR register: reads return `value`, writes are rejected.
fn gicr_read_only(access: GicAccess, value: u32) -> Result<GicrEffect, GicrError> {
    if access.is_write {
        return Err(GicrError::ReadOnlyRegister {
            offset: access.offset,
        });
    }
    Ok(GicrEffect::ReadValue(value))
}

/// `spi` (an architected interrupt ID, `SPI_BASE..NUM_INTERRUPT_IDS`) to its
/// position in every `spi_*` array — `None` for any ID this fixed-size GICD
/// does not implement (below [`SPI_BASE`], or at/above [`NUM_INTERRUPT_IDS`]).
fn spi_index(spi: u32) -> Option<usize> {
    if !(SPI_BASE..NUM_INTERRUPT_IDS).contains(&spi) {
        return None;
    }
    Some((spi - SPI_BASE) as usize)
}

/// True when `offset` falls within one of the four-word-per-register
/// bitmap blocks (`IGROUPR`/`ISENABLER`/`ICENABLER`/`ISPENDR`/`ICPENDR`)
/// starting at `base`, before per-word bounds are checked.
fn in_bitmap_range(offset: u64, base: u64) -> bool {
    offset >= base && offset < base + u64::from(BITMAP_WORD_COUNT) * 4
}

fn in_priority_range(offset: u64) -> bool {
    offset >= REG_IPRIORITYR_BASE
        && offset < REG_IPRIORITYR_BASE + u64::from(PRIORITY_WORD_COUNT) * 4
}

fn in_router_range(offset: u64) -> bool {
    offset >= REG_IROUTER_BASE && offset < REG_IROUTER_BASE + u64::from(NUM_INTERRUPT_IDS) * 8
}

fn in_icfgr_range(offset: u64) -> bool {
    offset >= REG_ICFGR_BASE && offset < REG_ICFGR_BASE + u64::from(ICFGR_WORD_COUNT) * 4
}

/// `offset` to a bitmap register's word index, `Err` if it lands beyond the
/// `ITLinesNumber + 1` words this GICD implements.
fn bitmap_word_index(offset: u64, base: u64) -> Result<u32, GicdError> {
    let word = ((offset - base) / 4) as u32;
    if word >= BITMAP_WORD_COUNT {
        return Err(GicdError::UnknownRegister { offset });
    }
    Ok(word)
}

/// A read-only register: reads return `value`, writes are rejected.
fn read_only(access: GicAccess, value: u32) -> Result<GicdEffect, GicdError> {
    if access.is_write {
        return Err(GicdError::ReadOnlyRegister {
            offset: access.offset,
        });
    }
    Ok(GicdEffect::ReadValue(value))
}

/// Read-As-Zero/Write-Ignored: the spec-mandated behavior for the SGI/PPI
/// sub-range of a register this GICD does implement for SPIs (as opposed to
/// [`GicdError::UnknownRegister`], reserved for offsets belonging to a
/// register this GICD does not implement at all).
fn raz_wi(access: GicAccess) -> Result<GicdEffect, GicdError> {
    if access.is_write {
        Ok(GicdEffect::Applied)
    } else {
        Ok(GicdEffect::ReadValue(0))
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::{
        GicAccess, GicDistributor, GicRedistributor, GicdEffect, GicdError, GicrEffect, GicrError,
        REG_CTLR, REG_GICR_CTLR, REG_GICR_ICACTIVER0, REG_GICR_ICFGR0, REG_GICR_ICFGR1,
        REG_GICR_IGROUPR0, REG_GICR_IGRPMODR0, REG_GICR_IIDR, REG_GICR_IPRIORITYR_BASE,
        REG_GICR_ISACTIVER0, REG_GICR_PIDR2, REG_GICR_TYPER_HIGH, REG_GICR_TYPER_LOW,
        REG_GICR_WAKER, REG_ICFGR_BASE, REG_IGRPMODR_BASE, REG_PIDR2, REG_STATUSR, REG_TYPER,
    };

    fn read(offset: u64) -> GicAccess {
        GicAccess {
            offset,
            is_write: false,
            value: 0,
        }
    }

    fn write(offset: u64, value: u32) -> GicAccess {
        GicAccess {
            offset,
            is_write: true,
            value,
        }
    }

    /// Worked example (principle 9 / `algorithm-development`): the register
    /// sequence Linux's `drivers/irqchip/irq-gic-v3.c` performs at probe —
    /// read `GICD_PIDR2` to confirm the architecture revision is GICv3
    /// (`ArchRev` field, bits\[7:4\]), read `GICD_TYPER` to learn how many
    /// SPIs exist (`ITLinesNumber`), then the enable dance: `GICD_CTLR`
    /// starts at reset (both groups disabled), the driver writes
    /// `EnableGrp1` (bit 1) to bring the distributor up, and reads it back.
    #[test]
    fn linux_gic_v3_probe_sequence_reads_arch_rev_then_typer_then_enables_group_one() {
        let mut distributor = GicDistributor::new();

        let pidr2 = distributor
            .apply(read(REG_PIDR2))
            .expect("pidr2 is readable");
        assert_eq!(pidr2, GicdEffect::ReadValue(0x30), "ArchRev=3 in bits[7:4]");

        let typer = distributor
            .apply(read(REG_TYPER))
            .expect("typer is readable");
        assert_eq!(
            typer,
            GicdEffect::ReadValue(7),
            "ITLinesNumber=7 for a 256-ID GIC"
        );
        let max_spi_id = 32 * (7 + 1) - 1;
        assert_eq!(max_spi_id, 255, "256-ID GIC's highest architected SPI");

        let reset_ctlr = distributor.apply(read(REG_CTLR)).expect("ctlr is readable");
        assert_eq!(
            reset_ctlr,
            GicdEffect::ReadValue(1 << 4),
            "both groups disabled at reset, ARE forced to 1"
        );

        let enabled = distributor
            .apply(write(REG_CTLR, 0b10))
            .expect("ctlr write is legal");
        assert_eq!(
            enabled,
            GicdEffect::ControlUpdated {
                group0_enabled: false,
                group1_enabled: true
            }
        );

        let readback = distributor.apply(read(REG_CTLR)).expect("ctlr is readable");
        assert_eq!(readback, GicdEffect::ReadValue(0b10 | (1 << 4)));
    }

    #[test]
    fn writing_isenabler_sets_the_named_spi_and_icenabler_clears_it() {
        let mut distributor = GicDistributor::new();
        const ISENABLER1: u64 = 0x0100 + 4;
        const ICENABLER1: u64 = 0x0180 + 4;
        let spi = 34u32;
        let bit_in_word = spi - 32;

        let effect = distributor
            .apply(write(ISENABLER1, 1 << bit_in_word))
            .expect("isenabler1 is a valid word");
        assert_eq!(effect, GicdEffect::SpiEnableChanged { spi, enabled: true });
        assert_eq!(distributor.spi_enabled(spi), Some(true));

        let effect = distributor
            .apply(write(ICENABLER1, 1 << bit_in_word))
            .expect("icenabler1 is a valid word");
        assert_eq!(
            effect,
            GicdEffect::SpiEnableChanged {
                spi,
                enabled: false
            }
        );
        assert_eq!(distributor.spi_enabled(spi), Some(false));
    }

    #[test]
    fn writing_ispendr_sets_the_named_spi_pending_and_icpendr_clears_it() {
        let mut distributor = GicDistributor::new();
        const ISPENDR1: u64 = 0x0200 + 4;
        const ICPENDR1: u64 = 0x0280 + 4;
        let spi = 40u32;
        let bit_in_word = spi - 32;

        let effect = distributor
            .apply(write(ISPENDR1, 1 << bit_in_word))
            .expect("ispendr1 is a valid word");
        assert_eq!(effect, GicdEffect::SpiPendingChanged { spi, pending: true });
        assert_eq!(distributor.spi_pending(spi), Some(true));

        let effect = distributor
            .apply(write(ICPENDR1, 1 << bit_in_word))
            .expect("icpendr1 is a valid word");
        assert_eq!(
            effect,
            GicdEffect::SpiPendingChanged {
                spi,
                pending: false
            }
        );
        assert_eq!(distributor.spi_pending(spi), Some(false));
    }

    #[test]
    fn ipriorityr_write_then_read_round_trips_one_spis_priority_byte() {
        let mut distributor = GicDistributor::new();
        let spi = 34u32;
        const IPRIORITYR8: u64 = 0x0400 + 4 * 8;
        let byte_in_word = spi % 4;
        let value = 0xa0u32 << (byte_in_word * 8);

        distributor
            .apply(write(IPRIORITYR8, value))
            .expect("ipriorityr8 is valid");
        assert_eq!(distributor.spi_priority(spi), Some(0xa0));

        let read_back = distributor
            .apply(read(IPRIORITYR8))
            .expect("ipriorityr8 is readable");
        assert_eq!(read_back, GicdEffect::ReadValue(value));
    }

    #[test]
    fn irouter_low_and_high_halves_round_trip_a_64_bit_affinity_target() {
        let mut distributor = GicDistributor::new();
        let spi = 34u32;
        let low_offset = 0x6000 + 8 * u64::from(spi);
        let high_offset = low_offset + 4;

        distributor
            .apply(write(low_offset, 0x0000_0001))
            .expect("irouter low half");
        distributor
            .apply(write(high_offset, 0x0000_0002))
            .expect("irouter high half");

        assert_eq!(distributor.spi_route(spi), Some(0x0000_0002_0000_0001));
        assert_eq!(
            distributor.apply(read(low_offset)).unwrap(),
            GicdEffect::ReadValue(1)
        );
        assert_eq!(
            distributor.apply(read(high_offset)).unwrap(),
            GicdEffect::ReadValue(2)
        );
    }

    #[test]
    fn sgi_ppi_word_zero_is_raz_wi_across_every_bitmap_register() {
        let mut distributor = GicDistributor::new();
        for base in [0x0080u64, 0x0100, 0x0180, 0x0200, 0x0280] {
            let read_effect = distributor.apply(read(base)).expect("word 0 is readable");
            assert_eq!(read_effect, GicdEffect::ReadValue(0), "raz at {base:#x}");
            let write_effect = distributor
                .apply(write(base, 0xffff_ffff))
                .expect("word 0 write is legal but ignored");
            assert_eq!(write_effect, GicdEffect::Applied, "wi at {base:#x}");
        }
    }

    #[test]
    fn ipriorityr_word_zero_is_raz_wi_since_it_covers_only_sgi_ppi_ids() {
        let mut distributor = GicDistributor::new();
        let read_effect = distributor.apply(read(0x0400)).expect("word 0 readable");
        assert_eq!(read_effect, GicdEffect::ReadValue(0));
        let write_effect = distributor
            .apply(write(0x0400, 0xffff_ffff))
            .expect("word 0 write legal");
        assert_eq!(write_effect, GicdEffect::Applied);
    }

    #[test]
    fn irouter_below_spi_base_is_reserved_raz_wi() {
        let mut distributor = GicDistributor::new();
        let read_effect = distributor.apply(read(0x6000)).expect("id 0 readable");
        assert_eq!(read_effect, GicdEffect::ReadValue(0));
        let write_effect = distributor
            .apply(write(0x6000, 0xffff_ffff))
            .expect("id 0 write legal");
        assert_eq!(write_effect, GicdEffect::Applied);
    }

    #[test]
    fn ctlr_write_is_read_only_free_but_typer_write_is_rejected() {
        let mut distributor = GicDistributor::new();
        let error = distributor
            .apply(write(REG_TYPER, 0))
            .expect_err("typer is read-only");
        assert_eq!(error, GicdError::ReadOnlyRegister { offset: REG_TYPER });
    }

    #[test]
    fn an_offset_in_the_gap_between_icactiver_and_ipriorityr_is_a_named_unknown_register_error() {
        let mut distributor = GicDistributor::new();
        const GAP_OFFSET: u64 = 0x03a0;
        let error = distributor
            .apply(read(GAP_OFFSET))
            .expect_err("gap offset is unmapped");
        assert_eq!(error, GicdError::UnknownRegister { offset: GAP_OFFSET });
    }

    #[test]
    fn an_unaligned_offset_is_rejected_before_any_register_match() {
        let mut distributor = GicDistributor::new();
        const UNALIGNED_OFFSET: u64 = 0x0001;
        let error = distributor
            .apply(read(UNALIGNED_OFFSET))
            .expect_err("offset is not 4-byte aligned");
        assert_eq!(
            error,
            GicdError::UnalignedAccess {
                offset: UNALIGNED_OFFSET
            }
        );
    }

    #[test]
    fn a_word_index_beyond_the_implemented_itlinesnumber_range_is_unknown_not_raz_wi() {
        let mut distributor = GicDistributor::new();
        const ISENABLER_WORD_8: u64 = 0x0100 + 4 * 8;
        let error = distributor
            .apply(read(ISENABLER_WORD_8))
            .expect_err("word 8 exceeds ITLinesNumber=7, must not be silently RAZ'd");
        assert_eq!(
            error,
            GicdError::UnknownRegister {
                offset: ISENABLER_WORD_8
            }
        );
    }

    /// Worked example (principle 9 / `algorithm-development`): the observed
    /// boot wall — `GICD_ICFGR2` (offset 0x0c08, SPIs 32-47) — round-trips a
    /// trigger-config write and must succeed post-change.
    #[test]
    fn icfgr_word_two_round_trips_spi_32s_edge_trigger_config() {
        let mut distributor = GicDistributor::new();
        const ICFGR2: u64 = REG_ICFGR_BASE + 4 * 2;
        let spi = 32u32;
        let id_in_word = spi - 32;

        let applied = distributor
            .apply(write(ICFGR2, 1 << (2 * id_in_word + 1)))
            .expect("icfgr2 (offset 0xc08) must be accepted, the observed boot wall");
        assert_eq!(applied, GicdEffect::Applied);
        assert_eq!(distributor.spi_edge_triggered(spi), Some(true));

        let read_back = distributor.apply(read(ICFGR2)).expect("icfgr2 is readable");
        assert_eq!(read_back, GicdEffect::ReadValue(1 << (2 * id_in_word + 1)));
    }

    #[test]
    fn icfgr_words_zero_and_one_are_raz_wi_since_they_cover_sgi_ppi_ids() {
        let mut distributor = GicDistributor::new();
        for word in [0u64, 1] {
            let offset = REG_ICFGR_BASE + 4 * word;
            let read_effect = distributor.apply(read(offset)).expect("word is readable");
            assert_eq!(read_effect, GicdEffect::ReadValue(0), "raz at word {word}");
            let write_effect = distributor
                .apply(write(offset, 0xffff_ffff))
                .expect("word write is legal but ignored");
            assert_eq!(write_effect, GicdEffect::Applied, "wi at word {word}");
        }
    }

    #[test]
    fn icfgr_word_beyond_the_implemented_range_is_unknown_not_raz_wi() {
        let mut distributor = GicDistributor::new();
        const ICFGR_WORD_16: u64 = REG_ICFGR_BASE + 4 * 16;
        let error = distributor
            .apply(read(ICFGR_WORD_16))
            .expect_err("word 16 exceeds the 16 words this GICD implements");
        assert_eq!(
            error,
            GicdError::UnknownRegister {
                offset: ICFGR_WORD_16
            }
        );
    }

    #[test]
    fn igrpmodr_word_zero_is_raz_wi_and_word_one_round_trips_a_membership_bit() {
        let mut distributor = GicDistributor::new();
        let word_zero_read = distributor
            .apply(read(REG_IGRPMODR_BASE))
            .expect("word 0 readable");
        assert_eq!(word_zero_read, GicdEffect::ReadValue(0));
        let word_zero_write = distributor
            .apply(write(REG_IGRPMODR_BASE, 0xffff_ffff))
            .expect("word 0 write legal but ignored");
        assert_eq!(word_zero_write, GicdEffect::Applied);

        const IGRPMODR1: u64 = REG_IGRPMODR_BASE + 4;
        distributor
            .apply(write(IGRPMODR1, 0b1))
            .expect("igrpmodr1 is writable");
        let read_back = distributor
            .apply(read(IGRPMODR1))
            .expect("igrpmodr1 is readable");
        assert_eq!(read_back, GicdEffect::ReadValue(0b1));
    }

    #[test]
    fn isactiver_write_sets_the_named_spi_active_and_icactiver_clears_it() {
        let mut distributor = GicDistributor::new();
        const ISACTIVER1: u64 = 0x0300 + 4;
        const ICACTIVER1: u64 = 0x0380 + 4;
        let spi = 36u32;
        let bit_in_word = spi - 32;

        let effect = distributor
            .apply(write(ISACTIVER1, 1 << bit_in_word))
            .expect("isactiver1 is a valid word");
        assert_eq!(effect, GicdEffect::SpiActiveChanged { spi, active: true });
        assert_eq!(distributor.spi_active(spi), Some(true));

        let effect = distributor
            .apply(write(ICACTIVER1, 1 << bit_in_word))
            .expect("icactiver1 is a valid word");
        assert_eq!(effect, GicdEffect::SpiActiveChanged { spi, active: false });
        assert_eq!(distributor.spi_active(spi), Some(false));
    }

    #[test]
    fn isactiver_word_zero_is_raz_wi_across_both_active_bitmap_aliases() {
        let mut distributor = GicDistributor::new();
        for base in [0x0300u64, 0x0380] {
            let read_effect = distributor.apply(read(base)).expect("word 0 is readable");
            assert_eq!(read_effect, GicdEffect::ReadValue(0), "raz at {base:#x}");
            let write_effect = distributor
                .apply(write(base, 0xffff_ffff))
                .expect("word 0 write is legal but ignored");
            assert_eq!(write_effect, GicdEffect::Applied, "wi at {base:#x}");
        }
    }

    #[test]
    fn isactiver_word_beyond_the_implemented_range_is_unknown_not_raz_wi() {
        let mut distributor = GicDistributor::new();
        const ISACTIVER_WORD_8: u64 = 0x0300 + 4 * 8;
        let error = distributor
            .apply(read(ISACTIVER_WORD_8))
            .expect_err("word 8 exceeds ITLinesNumber=7, must not be silently RAZ'd");
        assert_eq!(
            error,
            GicdError::UnknownRegister {
                offset: ISACTIVER_WORD_8
            }
        );
    }

    #[test]
    fn statusr_write_one_clears_the_named_bit_and_write_zero_leaves_it_set() {
        let mut distributor = GicDistributor::new();
        let reset = distributor
            .apply(read(REG_STATUSR))
            .expect("statusr is readable");
        assert_eq!(
            reset,
            GicdEffect::ReadValue(0),
            "no latched error bits at reset"
        );

        let no_op = distributor
            .apply(write(REG_STATUSR, 0))
            .expect("statusr write is legal");
        assert_eq!(no_op, GicdEffect::StatusrUpdated { value: 0 });

        let cleared = distributor
            .apply(write(REG_STATUSR, 0b1))
            .expect("statusr write is legal");
        assert_eq!(
            cleared,
            GicdEffect::StatusrUpdated { value: 0 },
            "clearing an already-clear bit is a no-op"
        );
    }

    /// Worked example (principle 9 / `algorithm-development`): the register
    /// sequence Linux's `drivers/irqchip/irq-gic-v3.c` performs per-CPU at
    /// probe — read `GICR_PIDR2` to confirm GICv3, read `GICR_TYPER` (both
    /// words) to learn `Last` and this PE's affinity, then the
    /// `GICR_WAKER` wake-up dance: read the reset state (`ProcessorSleep`
    /// and `ChildrenAsleep` both set), clear `ProcessorSleep`, and poll
    /// until `ChildrenAsleep` reads clear.
    #[test]
    fn linux_gic_v3_redistributor_probe_reads_pidr2_then_typer_then_runs_the_waker_dance() {
        let mut redistributor = GicRedistributor::new();

        let pidr2 = redistributor
            .apply(read(REG_GICR_PIDR2))
            .expect("pidr2 is readable");
        assert_eq!(pidr2, GicrEffect::ReadValue(0x30), "ArchRev=3 in bits[7:4]");

        let typer_low = redistributor
            .apply(read(REG_GICR_TYPER_LOW))
            .expect("typer low is readable");
        assert_eq!(
            typer_low,
            GicrEffect::ReadValue(1 << 4),
            "Last is set for the sole redistributor"
        );
        let typer_high = redistributor
            .apply(read(REG_GICR_TYPER_HIGH))
            .expect("typer high is readable");
        assert_eq!(
            typer_high,
            GicrEffect::ReadValue(0),
            "Affinity_Value matches RESIDENT_MPIDR_AFFINITY"
        );

        let reset_waker = redistributor
            .apply(read(REG_GICR_WAKER))
            .expect("waker is readable");
        assert_eq!(
            reset_waker,
            GicrEffect::ReadValue((1 << 1) | (1 << 2)),
            "both ProcessorSleep and ChildrenAsleep set at reset"
        );

        let woken = redistributor
            .apply(write(REG_GICR_WAKER, 0))
            .expect("clearing ProcessorSleep is a legal write");
        assert_eq!(
            woken,
            GicrEffect::WakerUpdated {
                processor_sleep: false,
                children_asleep: false
            }
        );

        let polled = redistributor
            .apply(read(REG_GICR_WAKER))
            .expect("waker is readable");
        assert_eq!(
            polled,
            GicrEffect::ReadValue(0),
            "ChildrenAsleep now reads clear, the poll succeeds"
        );
    }

    #[test]
    fn writing_isenabler0_sets_the_named_id_and_icenabler0_clears_it() {
        let mut redistributor = GicRedistributor::new();
        const ISENABLER0: u64 = super::SGI_BASE_OFFSET + 0x0100;
        const ICENABLER0: u64 = super::SGI_BASE_OFFSET + 0x0180;
        let ppi = 20u32;

        let effect = redistributor
            .apply(write(ISENABLER0, 1 << ppi))
            .expect("isenabler0 is a valid register");
        assert_eq!(
            effect,
            GicrEffect::IdEnableChanged {
                id: ppi,
                enabled: true
            }
        );
        assert_eq!(redistributor.id_enabled(ppi), Some(true));

        let effect = redistributor
            .apply(write(ICENABLER0, 1 << ppi))
            .expect("icenabler0 is a valid register");
        assert_eq!(
            effect,
            GicrEffect::IdEnableChanged {
                id: ppi,
                enabled: false
            }
        );
        assert_eq!(redistributor.id_enabled(ppi), Some(false));
    }

    #[test]
    fn writing_ispendr0_sets_the_named_id_pending_and_icpendr0_clears_it() {
        let mut redistributor = GicRedistributor::new();
        const ISPENDR0: u64 = super::SGI_BASE_OFFSET + 0x0200;
        const ICPENDR0: u64 = super::SGI_BASE_OFFSET + 0x0280;
        let sgi = 3u32;

        let effect = redistributor
            .apply(write(ISPENDR0, 1 << sgi))
            .expect("ispendr0 is a valid register");
        assert_eq!(
            effect,
            GicrEffect::IdPendingChanged {
                id: sgi,
                pending: true
            }
        );
        assert_eq!(redistributor.id_pending(sgi), Some(true));

        let effect = redistributor
            .apply(write(ICPENDR0, 1 << sgi))
            .expect("icpendr0 is a valid register");
        assert_eq!(
            effect,
            GicrEffect::IdPendingChanged {
                id: sgi,
                pending: false
            }
        );
        assert_eq!(redistributor.id_pending(sgi), Some(false));
    }

    #[test]
    fn ipriorityr_write_then_read_round_trips_one_ids_priority_byte() {
        let mut redistributor = GicRedistributor::new();
        let ppi = 18u32;
        let byte_in_word = ppi % 4;
        let word = ppi / 4;
        let offset = REG_GICR_IPRIORITYR_BASE + 4 * u64::from(word);
        let value = 0x40u32 << (byte_in_word * 8);

        redistributor
            .apply(write(offset, value))
            .expect("ipriorityr word is valid");
        assert_eq!(redistributor.id_priority(ppi), Some(0x40));

        let read_back = redistributor
            .apply(read(offset))
            .expect("ipriorityr word is readable");
        assert_eq!(read_back, GicrEffect::ReadValue(value));
    }

    #[test]
    fn icfgr1_write_then_read_round_trips_a_ppis_edge_trigger_bit() {
        let mut redistributor = GicRedistributor::new();
        let ppi_index_within_icfgr1 = 4; // PPI 20 (PPI_BASE + 4)

        let effect = redistributor
            .apply(write(
                REG_GICR_ICFGR1,
                1 << (2 * ppi_index_within_icfgr1 + 1),
            ))
            .expect("icfgr1 is writable");
        assert_eq!(effect, GicrEffect::Applied);

        let read_back = redistributor
            .apply(read(REG_GICR_ICFGR1))
            .expect("icfgr1 is readable");
        assert_eq!(
            read_back,
            GicrEffect::ReadValue(1 << (2 * ppi_index_within_icfgr1 + 1))
        );
    }

    #[test]
    fn icfgr0_reports_every_sgi_fixed_edge_triggered_and_rejects_writes() {
        let mut redistributor = GicRedistributor::new();
        let read_effect = redistributor
            .apply(read(REG_GICR_ICFGR0))
            .expect("icfgr0 is readable");
        assert_eq!(
            read_effect,
            GicrEffect::ReadValue(0xaaaa_aaaa),
            "every sgi's Int_config bit is set"
        );

        let error = redistributor
            .apply(write(REG_GICR_ICFGR0, 0))
            .expect_err("icfgr0 is read-only");
        assert_eq!(
            error,
            GicrError::ReadOnlyRegister {
                offset: REG_GICR_ICFGR0
            }
        );
    }

    #[test]
    fn gicr_iidr_and_typer_are_read_only() {
        let mut redistributor = GicRedistributor::new();
        let error = redistributor
            .apply(write(REG_GICR_IIDR, 0))
            .expect_err("iidr is read-only");
        assert_eq!(
            error,
            GicrError::ReadOnlyRegister {
                offset: REG_GICR_IIDR
            }
        );

        let error = redistributor
            .apply(write(REG_GICR_TYPER_LOW, 0))
            .expect_err("typer low is read-only");
        assert_eq!(
            error,
            GicrError::ReadOnlyRegister {
                offset: REG_GICR_TYPER_LOW
            }
        );
    }

    #[test]
    fn gicr_ctlr_round_trips_enable_lpis() {
        let mut redistributor = GicRedistributor::new();
        let reset = redistributor
            .apply(read(REG_GICR_CTLR))
            .expect("ctlr is readable");
        assert_eq!(reset, GicrEffect::ReadValue(0), "EnableLPIs clear at reset");

        let updated = redistributor
            .apply(write(REG_GICR_CTLR, 1))
            .expect("ctlr write is legal");
        assert_eq!(updated, GicrEffect::ControlUpdated { lpis_enabled: true });
        assert!(redistributor.lpis_enabled());
    }

    #[test]
    fn an_offset_in_the_gap_between_ctlr_and_iidr_neighbors_is_unknown_at_the_redistributor() {
        let mut redistributor = GicRedistributor::new();
        const GAP_OFFSET: u64 = 0x0018; // just past GICR_WAKER (0x0014), before PIDR2
        let error = redistributor
            .apply(read(GAP_OFFSET))
            .expect_err("gap offset is unmapped");
        assert_eq!(error, GicrError::UnknownRegister { offset: GAP_OFFSET });
    }

    #[test]
    fn an_unaligned_gicr_offset_is_rejected_before_any_register_match() {
        let mut redistributor = GicRedistributor::new();
        const UNALIGNED_OFFSET: u64 = 0x0001;
        let error = redistributor
            .apply(read(UNALIGNED_OFFSET))
            .expect_err("offset is not 4-byte aligned");
        assert_eq!(
            error,
            GicrError::UnalignedAccess {
                offset: UNALIGNED_OFFSET
            }
        );
    }

    #[test]
    fn a_word_index_beyond_the_implemented_ipriorityr_range_is_unknown_not_read_as_zero() {
        let mut redistributor = GicRedistributor::new();
        const IPRIORITYR_WORD_8: u64 = super::SGI_BASE_OFFSET + 0x0400 + 4 * 8;
        let error = redistributor.apply(read(IPRIORITYR_WORD_8)).expect_err(
            "word 8 exceeds the 8 words needed for 32 ids, must not be silently accepted",
        );
        assert_eq!(
            error,
            GicrError::UnknownRegister {
                offset: IPRIORITYR_WORD_8
            }
        );
    }

    #[test]
    fn igroupr0_round_trips_a_group_membership_bit() {
        let mut redistributor = GicRedistributor::new();
        let sgi = 2u32;

        redistributor
            .apply(write(REG_GICR_IGROUPR0, 1 << sgi))
            .expect("igroupr0 is writable");
        let read_back = redistributor
            .apply(read(REG_GICR_IGROUPR0))
            .expect("igroupr0 is readable");
        assert_eq!(read_back, GicrEffect::ReadValue(1 << sgi));
    }

    /// Worked example (principle 9 / `algorithm-development`): the observed
    /// boot wall — `mmio register access rejected: window=gicr
    /// offset=0x10380 is_write=1` — decodes to `SGI_base + 0x0380`,
    /// `GICR_ICACTIVER0`. Must succeed post-change and RW1C-clear a bit
    /// `ISACTIVER0` set.
    #[test]
    fn icactiver0_write_at_the_observed_offset_0x10380_clears_a_bit_isactiver0_set() {
        let mut redistributor = GicRedistributor::new();
        assert_eq!(
            REG_GICR_ICACTIVER0, 0x10380,
            "the observed boot wall's exact offset"
        );
        let ppi = 20u32;

        let set = redistributor
            .apply(write(REG_GICR_ISACTIVER0, 1 << ppi))
            .expect("isactiver0 is a valid register");
        assert_eq!(
            set,
            GicrEffect::IdActiveChanged {
                id: ppi,
                active: true
            }
        );
        assert_eq!(redistributor.id_active(ppi), Some(true));

        let cleared = redistributor
            .apply(write(REG_GICR_ICACTIVER0, 1 << ppi))
            .expect("icactiver0 (offset 0x10380) must be accepted, the observed boot wall");
        assert_eq!(
            cleared,
            GicrEffect::IdActiveChanged {
                id: ppi,
                active: false
            }
        );
        assert_eq!(redistributor.id_active(ppi), Some(false));
    }

    #[test]
    fn isactiver0_write_sets_the_named_id_active_and_round_trips_on_read() {
        let mut redistributor = GicRedistributor::new();
        let sgi = 5u32;

        let effect = redistributor
            .apply(write(REG_GICR_ISACTIVER0, 1 << sgi))
            .expect("isactiver0 is a valid register");
        assert_eq!(
            effect,
            GicrEffect::IdActiveChanged {
                id: sgi,
                active: true
            }
        );

        let read_back = redistributor
            .apply(read(REG_GICR_ISACTIVER0))
            .expect("isactiver0 is readable");
        assert_eq!(read_back, GicrEffect::ReadValue(1 << sgi));
    }

    #[test]
    fn igrpmodr0_round_trips_a_group_modifier_bit() {
        let mut redistributor = GicRedistributor::new();
        let ppi = 18u32;

        redistributor
            .apply(write(REG_GICR_IGRPMODR0, 1 << ppi))
            .expect("igrpmodr0 is writable");
        let read_back = redistributor
            .apply(read(REG_GICR_IGRPMODR0))
            .expect("igrpmodr0 is readable");
        assert_eq!(read_back, GicrEffect::ReadValue(1 << ppi));
    }

    #[test]
    fn an_untouched_offset_between_igrpmodr0_and_pidr2_is_still_unknown_register() {
        let mut redistributor = GicRedistributor::new();
        const UNTOUCHED_OFFSET: u64 = super::SGI_BASE_OFFSET + 0x0e00;
        let error = redistributor
            .apply(read(UNTOUCHED_OFFSET))
            .expect_err("offset past igrpmodr0 with no register defined stays unmapped");
        assert_eq!(
            error,
            GicrError::UnknownRegister {
                offset: UNTOUCHED_OFFSET
            }
        );
    }
}

#[cfg(test)]
mod icc_tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::{ICC_IAR1_SPURIOUS, IccAccess, IccCpuInterface, IccEffect, IccError};

    fn read(op0: u8, op1: u8, crn: u8, crm: u8, op2: u8) -> IccAccess {
        IccAccess {
            op0,
            op1,
            crn,
            crm,
            op2,
            is_write: false,
            value: 0,
        }
    }

    fn write(op0: u8, op1: u8, crn: u8, crm: u8, op2: u8, value: u64) -> IccAccess {
        IccAccess {
            op0,
            op1,
            crn,
            crm,
            op2,
            is_write: true,
            value,
        }
    }

    /// Worked example (principle 9 / `algorithm-development`): the exact
    /// syndrome this VM's boot wall reported, `0x6230102d`
    /// (`pc=0xffff8000807da100`), decoded field by field against ARM ARM's
    /// ISS-for-trapped-MSR/MRS layout (Op0\[21:20\], Op2\[19:17\],
    /// Op1\[16:14\], CRn\[13:10\], Rt\[9:5\], CRm\[4:1\], Direction\[0\]) —
    /// the same decode this module's own [`super::IccCpuInterface::apply`]
    /// doc names as the wall this register's addition closed.
    #[test]
    fn observed_syndrome_0x6230102d_decodes_to_icc_pmr_el1_read() {
        let syndrome: u64 = 0x6230102d;
        let exception_class = (syndrome >> 26) & 0x3f;
        assert_eq!(
            exception_class, 0x18,
            "trapped MSR/MRS/system-instruction exception class"
        );

        let iss = (syndrome & 0x1ff_ffff) as u32;
        let op0 = ((iss >> 20) & 0x3) as u8;
        let op2 = ((iss >> 17) & 0x7) as u8;
        let op1 = ((iss >> 14) & 0x7) as u8;
        let crn = ((iss >> 10) & 0xf) as u8;
        let rt = (iss >> 5) & 0x1f;
        let crm = ((iss >> 1) & 0xf) as u8;
        let is_read = iss & 0x1 != 0;

        assert_eq!(
            (op0, op1, crn, crm, op2),
            (3, 0, 4, 6, 0),
            "S3_0_C4_C6_0 == ICC_PMR_EL1"
        );
        assert_eq!(rt, 1, "destination register is x1");
        assert!(is_read, "direction bit sets, an MRS read");

        let mut icc = IccCpuInterface::new();
        let effect = icc
            .apply(read(op0, op1, crn, crm, op2))
            .expect("icc_pmr_el1 is modeled");
        assert_eq!(
            effect,
            IccEffect::ReadValue(0),
            "pmr resets to 0 (mask nothing)"
        );
    }

    #[test]
    fn icc_sre_el1_reads_one_and_stays_set_after_a_guest_write() {
        let mut icc = IccCpuInterface::new();
        let reset = icc
            .apply(read(3, 0, 12, 12, 5))
            .expect("icc_sre_el1 is modeled");
        assert_eq!(
            reset,
            IccEffect::ReadValue(1),
            "SRE reads 1 even before any guest write"
        );

        let write_effect = icc
            .apply(write(3, 0, 12, 12, 5, 1))
            .expect("icc_sre_el1 write is legal");
        assert_eq!(write_effect, IccEffect::Applied);

        let readback = icc
            .apply(read(3, 0, 12, 12, 5))
            .expect("icc_sre_el1 is modeled");
        assert_eq!(
            readback,
            IccEffect::ReadValue(1),
            "SRE never clears on readback"
        );
    }

    #[test]
    fn icc_iar1_el1_always_reports_spurious_with_no_interrupts_injected() {
        let mut icc = IccCpuInterface::new();
        let effect = icc
            .apply(read(3, 0, 12, 12, 0))
            .expect("icc_iar1_el1 is modeled");
        assert_eq!(effect, IccEffect::ReadValue(ICC_IAR1_SPURIOUS));
    }

    /// Worked example: the whole pending -> IAR -> active -> EOIR -> idle
    /// cycle this slice's `set_pending` + `apply_iar1` + `apply_eoir1` exist
    /// to serve, driven exactly once end to end (ARM IHI 0069 4.1.1's
    /// acknowledge/deactivate transitions), for the vtimer's own INTID (27,
    /// PPI 11 -- `dtb.rs`'s `write_timer`).
    #[test]
    fn pending_intid_27_acknowledges_via_iar1_then_deactivates_via_matching_eoir1() {
        const VTIMER_INTID: u32 = 27;
        let mut icc = IccCpuInterface::new();

        icc.set_pending(VTIMER_INTID);

        let acknowledge = icc
            .apply(read(3, 0, 12, 12, 0))
            .expect("icc_iar1_el1 is modeled");
        assert_eq!(
            acknowledge,
            IccEffect::ReadValue(u64::from(VTIMER_INTID)),
            "a pending intid acknowledges via iar1, moving pending to active"
        );

        let idle_readback = icc
            .apply(read(3, 0, 12, 12, 0))
            .expect("icc_iar1_el1 is modeled");
        assert_eq!(
            idle_readback,
            IccEffect::ReadValue(ICC_IAR1_SPURIOUS),
            "a second iar1 read while active and nothing else pending is spurious"
        );

        let deactivate = icc
            .apply(write(3, 0, 12, 12, 1, u64::from(VTIMER_INTID)))
            .expect("icc_eoir1_el1 write is legal");
        assert_eq!(
            deactivate,
            IccEffect::InterruptDeactivated(VTIMER_INTID),
            "eoir1 naming the active intid retires it"
        );

        let after_idle = icc
            .apply(read(3, 0, 12, 12, 0))
            .expect("icc_iar1_el1 is modeled");
        assert_eq!(
            after_idle,
            IccEffect::ReadValue(ICC_IAR1_SPURIOUS),
            "nothing pending or active after a full cycle"
        );
    }

    #[test]
    fn eoir1_naming_an_intid_that_was_never_active_is_a_spurious_eoi_accepted_with_no_state_change() {
        let mut icc = IccCpuInterface::new();
        icc.set_pending(27);
        icc.apply(read(3, 0, 12, 12, 0))
            .expect("icc_iar1_el1 acknowledges the pending intid");

        let spurious_eoi = icc
            .apply(write(3, 0, 12, 12, 1, 30))
            .expect("a mismatched eoir1 write is accepted, not rejected");
        assert_eq!(
            spurious_eoi,
            IccEffect::Applied,
            "a spurious eoi (wrong intid) has no deactivation effect"
        );

        let still_active = icc
            .apply(write(3, 0, 12, 12, 1, 27))
            .expect("the genuinely active intid can still be retired afterward");
        assert_eq!(
            still_active,
            IccEffect::InterruptDeactivated(27),
            "the mismatched eoir1 above did not disturb the real active interrupt"
        );
    }

    #[test]
    fn eoir1_with_nothing_ever_pending_is_a_spurious_eoi_accepted_with_no_state_change() {
        let mut icc = IccCpuInterface::new();
        let effect = icc
            .apply(write(3, 0, 12, 12, 1, 27))
            .expect("an eoir1 write with nothing active is accepted, not rejected");
        assert_eq!(effect, IccEffect::Applied);
    }

    #[test]
    fn iar1_read_with_nothing_pending_reports_spurious() {
        let mut icc = IccCpuInterface::new();
        let effect = icc
            .apply(read(3, 0, 12, 12, 0))
            .expect("icc_iar1_el1 is modeled");
        assert_eq!(effect, IccEffect::ReadValue(ICC_IAR1_SPURIOUS));
    }

    #[test]
    fn icc_eoir1_el1_write_is_accepted_and_read_is_rejected_as_write_only() {
        let mut icc = IccCpuInterface::new();
        let effect = icc
            .apply(write(3, 0, 12, 12, 1, 0))
            .expect("icc_eoir1_el1 write is legal");
        assert_eq!(effect, IccEffect::Applied);

        let error = icc
            .apply(read(3, 0, 12, 12, 1))
            .expect_err("icc_eoir1_el1 is write-only");
        assert_eq!(
            error,
            IccError::WriteOnlyRegister {
                op0: 3,
                op1: 0,
                crn: 12,
                crm: 12,
                op2: 1
            }
        );
    }

    #[test]
    fn icc_pmr_el1_write_then_read_round_trips_the_priority_mask() {
        let mut icc = IccCpuInterface::new();
        icc.apply(write(3, 0, 4, 6, 0, 0x80))
            .expect("icc_pmr_el1 write is legal");
        assert_eq!(icc.priority_mask(), 0x80);

        let effect = icc
            .apply(read(3, 0, 4, 6, 0))
            .expect("icc_pmr_el1 is modeled");
        assert_eq!(effect, IccEffect::ReadValue(0x80));
    }

    #[test]
    fn icc_igrpen1_el1_write_then_read_round_trips_group_one_enable() {
        let mut icc = IccCpuInterface::new();
        assert!(!icc.group1_enabled(), "group 1 disabled at reset");

        icc.apply(write(3, 0, 12, 12, 7, 1))
            .expect("icc_igrpen1_el1 write is legal");
        assert!(icc.group1_enabled());

        let effect = icc
            .apply(read(3, 0, 12, 12, 7))
            .expect("icc_igrpen1_el1 is modeled");
        assert_eq!(effect, IccEffect::ReadValue(1));
    }

    /// Worked example: this VM's own second EC 0x18 wall, captured verbatim
    /// from `backend_macos.c`'s now-self-documenting error message before
    /// this register existed — `"icc sysreg access rejected: S2_0_C1_C3_4
    /// write (no icc register modeled at this encoding)"`. `op0=2` is
    /// outside the ICC space (every ICC register in this module is
    /// `op0=3`); ARM DDI 0487 D17.2's external-debug encoding table names
    /// `S2_0_C1_C3_4` as `OSDLR_EL1`, and this is exactly the register
    /// Linux's `reset_os_lock` writes at CPU bring-up.
    #[test]
    fn observed_s2_0_c1_c3_4_write_decodes_to_osdlr_el1() {
        let (op0, op1, crn, crm, op2) = (2u8, 0u8, 1u8, 3u8, 4u8);

        // Before this register existed, this exact tuple was the wall:
        // `IccError::UnknownRegister`. Assert the OLD behaviour is gone,
        // not merely that a new behaviour exists, so a regression that
        // silently reintroduced the wall would fail this test too.
        let mut icc = IccCpuInterface::new();
        let effect = icc
            .apply(write(op0, op1, crn, crm, op2, 1))
            .expect("osdlr_el1 is modeled after the M5b follow-through");
        assert_eq!(effect, IccEffect::Applied);
    }

    #[test]
    fn osdlr_el1_write_then_read_round_trips_the_double_lock_bit() {
        let mut icc = IccCpuInterface::new();
        let reset = icc
            .apply(read(2, 0, 1, 3, 4))
            .expect("osdlr_el1 is modeled");
        assert_eq!(reset, IccEffect::ReadValue(0), "double lock clear at reset");

        icc.apply(write(2, 0, 1, 3, 4, 1))
            .expect("osdlr_el1 write is legal");
        let readback = icc
            .apply(read(2, 0, 1, 3, 4))
            .expect("osdlr_el1 is modeled");
        assert_eq!(
            readback,
            IccEffect::ReadValue(1),
            "write of DLK=1 round-trips on read"
        );
    }

    /// Worked example: this VM's own third EC 0x18 wall, the register
    /// `OSDLR_EL1`'s own modeling exposed immediately behind it —
    /// `"icc sysreg access rejected: S2_0_C1_C0_4 write (no icc register
    /// modeled at this encoding)"`. ARM DDI 0487 D17.2 names
    /// `S2_0_C1_C0_4` as `OSLAR_EL1`, the OS Lock Access Register
    /// `reset_os_lock` writes in the same call as `OSDLR_EL1`.
    #[test]
    fn observed_s2_0_c1_c0_4_write_decodes_to_oslar_el1() {
        let mut icc = IccCpuInterface::new();
        let effect = icc
            .apply(write(2, 0, 1, 0, 4, 1))
            .expect("oslar_el1 is modeled after the M5b follow-through");
        assert_eq!(effect, IccEffect::Applied);
    }

    #[test]
    fn oslar_el1_read_is_rejected_as_write_only() {
        let mut icc = IccCpuInterface::new();
        let error = icc.apply(read(2, 0, 1, 0, 4)).expect_err("oslar_el1 is write-only");
        assert_eq!(
            error,
            IccError::WriteOnlyRegister { op0: 2, op1: 0, crn: 1, crm: 0, op2: 4 }
        );
    }

    #[test]
    fn an_unmodeled_icc_encoding_is_a_named_unknown_register_error() {
        let mut icc = IccCpuInterface::new();
        // ICC_DIR_EL1 (S3_0_C12_C11_1) -- a real architected register, but
        // outside this module's minimum set -- must decode itself the way
        // this wall's own PMR trap did, not silently RAZ/WI.
        let error = icc
            .apply(read(3, 0, 12, 11, 1))
            .expect_err("icc_dir_el1 is not modeled");
        assert_eq!(
            error,
            IccError::UnknownRegister {
                op0: 3,
                op1: 0,
                crn: 12,
                crm: 11,
                op2: 1
            }
        );
    }

    /// Worked example (task's own routing-around-SGI1R spec): an `IRM` = 1
    /// broadcast SGI0 write lands pending on this VM's one modeled PE,
    /// observable the same way [`pending_intid_27_acknowledges_via_iar1_then_deactivates_via_matching_eoir1`]
    /// observes any pending intid — an `ICC_IAR1_EL1` read reports it.
    #[test]
    fn broadcast_sgi1r_write_of_sgi0_lands_pending_and_acknowledges_via_iar1() {
        let mut icc = IccCpuInterface::new();
        const IRM_BROADCAST: u64 = 1 << 40;
        const SGI0_INTID: u64 = 0;

        let applied = icc
            .apply(write(3, 0, 12, 11, 5, IRM_BROADCAST | (SGI0_INTID << 24)))
            .expect("icc_sgi1r_el1 write is legal");
        assert_eq!(applied, IccEffect::Applied);

        let acknowledge = icc
            .apply(read(3, 0, 12, 12, 0))
            .expect("icc_iar1_el1 is modeled");
        assert_eq!(
            acknowledge,
            IccEffect::ReadValue(SGI0_INTID),
            "the broadcast sgi0 write must have landed pending"
        );
    }

    /// `IRM` = 0 with `TargetList` bit 0 set and every affinity field 0
    /// resolves to this VM's own PE (affinity 0) — the non-broadcast target
    /// path the broadcast test above does not exercise.
    #[test]
    fn targeted_sgi1r_write_naming_affinity_zero_target_list_bit_zero_lands_pending() {
        let mut icc = IccCpuInterface::new();
        const SGI3_INTID: u64 = 3;
        const TARGET_LIST_BIT_ZERO: u64 = 0b1;

        icc.apply(write(3, 0, 12, 11, 5, (SGI3_INTID << 24) | TARGET_LIST_BIT_ZERO))
            .expect("icc_sgi1r_el1 write is legal");

        let acknowledge = icc
            .apply(read(3, 0, 12, 12, 0))
            .expect("icc_iar1_el1 is modeled");
        assert_eq!(acknowledge, IccEffect::ReadValue(SGI3_INTID));
    }

    /// A targeted write whose `TargetList` names only a PE this VM never
    /// modeled (bit 1, not bit 0) is accepted with no pending interrupt —
    /// the "target resolves to no existing CPU" no-op the task's own spec
    /// names, mirrored the same way an out-of-range GICD SPI write above is
    /// a no-op rather than a fault.
    #[test]
    fn targeted_sgi1r_write_naming_only_a_nonexistent_cpu_is_a_no_op() {
        let mut icc = IccCpuInterface::new();
        const SGI5_INTID: u64 = 5;
        const TARGET_LIST_BIT_ONE: u64 = 0b10;

        let applied = icc
            .apply(write(3, 0, 12, 11, 5, (SGI5_INTID << 24) | TARGET_LIST_BIT_ONE))
            .expect("icc_sgi1r_el1 write is legal even when it targets nobody modeled");
        assert_eq!(applied, IccEffect::Applied);

        let readback = icc
            .apply(read(3, 0, 12, 12, 0))
            .expect("icc_iar1_el1 is modeled");
        assert_eq!(
            readback,
            IccEffect::ReadValue(ICC_IAR1_SPURIOUS),
            "no pending interrupt landed for a target this vm never modeled"
        );
    }

    /// `ICC_SGI1R_EL1` is write-only per the architecture — a read is
    /// UNDEFINED, mirrored the same way [`Self::apply_eoir1`]'s own test
    /// rejects a read.
    #[test]
    fn icc_sgi1r_el1_read_is_rejected_as_write_only() {
        let mut icc = IccCpuInterface::new();
        let error = icc
            .apply(read(3, 0, 12, 11, 5))
            .expect_err("icc_sgi1r_el1 is write-only");
        assert_eq!(
            error,
            IccError::WriteOnlyRegister {
                op0: 3,
                op1: 0,
                crn: 12,
                crm: 11,
                op2: 5
            }
        );
    }
}
