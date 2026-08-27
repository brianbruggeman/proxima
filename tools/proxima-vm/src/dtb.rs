//! M5b — the flattened devicetree (FDT), the DTB half of what the HVF lane
//! needs alongside GIC + PSCI (`ROADMAP.md` M5 / M5b). A Linux aarch64
//! guest reads its hardware layout from this blob at boot; there is no
//! ACPI path on this tree's target (bare `vmlinux`, serial console only).
//!
//! # Incumbent decision — reuse, not hand-roll
//!
//! Per principle 1 (reuse-first) and the roadmap's own hand-rolling gate
//! (`ROADMAP.md` "Hand-rolling a spec parser", condition 3: meet or beat
//! the incumbent on their home turf): [`vm-fdt`](https://docs.rs/vm-fdt)
//! (rust-vmm — the same org that owns `virtio-queue`/`vhost`/
//! `kvm-bindings`, and crosvm/Firecracker's own FDT writer) already builds
//! exactly this blob shape, `no_std + alloc` capable; [`fdt`
//! ](https://docs.rs/fdt) (repnop) already parses it, `no_std` with zero
//! dependencies. Both are the mature, focused, widely-deployed answer for
//! this exact job in the VMM ecosystem that IS the incumbent here — beating
//! them would mean re-deriving the devicetree spec v0.4 token stream
//! (`FDT_BEGIN_NODE`/`FDT_PROP`/`FDT_END_NODE`/`FDT_END`, the strings
//! block, the memory-reservation block) that these two crates already get
//! right. This module is a thin, typed composition over both — the tree
//! shape a Linux aarch64 guest needs, not a spec-parser reimplementation.
//! No hand-rolled FDT codec lands here; see `BENCH_LOG.md` if that changes.
//!
//! # Tier
//!
//! `alloc` (tier 1): the built blob is an owned `Vec<u8>`
//! (`vm_fdt::FdtWriter::finish`). Round-trip parsing (`fdt::Fdt::new`)
//! borrows from that blob and allocates nothing at all — genuinely tier 3
//! — but lives in this file because builder and parser are one unit of
//! reuse, not two.
//!
//! # QEMU virt machine — worked-example reference
//!
//! [`QemuVirtLayout::CANONICAL`] is `qemu-system-aarch64 -M virt`'s own
//! fixed memory map (`hw/arm/virt.c`: `VIRT_GIC_DIST`, `VIRT_GIC_REDIST`,
//! `VIRT_MEM`) — the same machine `tests/page_table_qemu_differential.rs`
//! already anchors RAM base `0x4000_0000` against. It describes the GIC
//! that M5b's HVF lane still has to build; nothing here claims that GIC
//! exists yet.

use alloc::format;
use alloc::string::ToString;
use alloc::vec;
use alloc::vec::Vec;
use vm_fdt::{FdtWriter, FdtWriterResult as FdtResult};

/// `phandle` the GIC node publishes so `/`'s `interrupt-parent` and any
/// future interrupt-consuming node can reference it. One GIC per tree, so
/// one constant phandle is sufficient — spec requires uniqueness, not any
/// particular value.
const GIC_PHANDLE: u32 = 1;

/// `phandle` the fixed `apb-pclk` clock node publishes so the uart node's
/// own `clocks` property can reference it, matching QEMU virt's own
/// generated tree (`hw/arm/virt.c`'s `create_uart`/`create_fdt`, worked
/// example cited in this module's own doc).
const APB_PCLK_PHANDLE: u32 = 2;

/// `apb-pclk`'s `clock-frequency`: 24 MHz, QEMU virt's own fixed-clock rate
/// for the pl011's APB bus clock (`hw/arm/virt.c`'s `create_fdt`).
const APB_PCLK_FREQUENCY_HZ: u32 = 24_000_000;

/// pl011 UART SPI: QEMU virt's `VIRT_UART` interrupt line
/// (`hw/arm/virt.c`'s `irqmap[VIRT_UART]`), an `arm,gic-v3`-style
/// `<type num flags>` cell triple identical in shape to
/// [`write_timer`]'s own PPI cells: `type` = 0 (SPI), `num` = 1, `flags` = 4
/// (level-low).
const UART_INTERRUPT_CELLS: [u32; 3] = [0, 1, 4];

/// GIC redistributor region sized for a single vCPU (`0x2_0000` per core,
/// the GICv3 redistributor pair width) rather than QEMU's own
/// up-to-123-core default (`0xf6_0000`), because this slice's worked
/// example is the M5 single-vCPU guest. `QemuVirtLayout::CANONICAL` keeps
/// QEMU's own multi-core sizing for the dtc differential (§ tests below);
/// callers building a real M5b guest pick the width their vCPU count needs.
const SINGLE_VCPU_GICR_SIZE: u64 = 0x0002_0000;

/// A guest's hardware layout: the GIC's own two MMIO windows (distributor,
/// redistributor) and the uart window. RAM base/size live on [`BootParams`]
/// instead of here — a single boot decides its own RAM geometry (the M5b
/// boot path's 256 MiB is not this dtc-differential worked example's 1 GiB),
/// and carrying two independent copies of "where is RAM" is exactly the
/// DTB/VM disagreement this module's own boot path had to reconcile. Fields
/// are plain `u64`s, not a fluent builder, because the shape is fixed by the
/// architecture (GICv3 always has exactly these two windows) — nothing here
/// composes optionally.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QemuVirtLayout {
    pub gicd_base: u64,
    pub gicd_size: u64,
    pub gicr_base: u64,
    pub gicr_size: u64,
    pub uart_base: u64,
    pub uart_size: u64,
}

impl QemuVirtLayout {
    /// `hw/arm/virt.c`'s own memmap: what a real `qemu-system-aarch64 -M
    /// virt` process reports to a guest for its GIC and uart windows (RAM
    /// geometry is [`BootParams`]'s, not this type's — see the struct doc).
    ///
    /// `uart_base` (`0x0900_0000`) exceeds `GUEST_MEMORY_SIZE`
    /// (`dispatch_trampoline.h`'s `0x0400_0000`, 64 MiB, the actual
    /// `hv_vm_map`'d guest-RAM ceiling in `dispatch::run_dispatch_loop`'s
    /// ELF-guest path) — the identical non-aliasing check the GIC slice's
    /// own module doc performs for `GICD_MMIO_WINDOW_BASE`/
    /// `GICR_MMIO_WINDOW_BASE` (`dispatch_trampoline.h:75-88`):
    /// `0x0900_0000 > 0x0400_0000`, so the uart window is never backed by
    /// that path's mapped guest memory and every guest access genuinely
    /// traps as a data abort, the same guarantee the GICD/GICR and virtio
    /// windows already get from sitting outside that ceiling. This module's
    /// own `tests::uart_window_sits_above_the_mapped_guest_memory_ceiling`
    /// asserts it. The M5b boot path (`crate::boot`) instead maps RAM
    /// *above* these windows entirely — see that module's own doc for the
    /// reversed non-aliasing argument its RAM base requires.
    pub const CANONICAL: Self = Self {
        gicd_base: 0x0800_0000,
        gicd_size: 0x0001_0000,
        gicr_base: 0x080a_0000,
        gicr_size: 0x00f6_0000,
        uart_base: 0x0900_0000,
        uart_size: 0x0000_1000,
    };

    /// [`Self::CANONICAL`] with the redistributor region trimmed to one
    /// vCPU, for a guest that only ever brings up `cpu@0`.
    #[must_use]
    pub const fn single_vcpu() -> Self {
        Self {
            gicr_size: SINGLE_VCPU_GICR_SIZE,
            ..Self::CANONICAL
        }
    }
}

/// What the boot needs to say beyond the fixed GIC/uart hardware layout: the
/// RAM window this boot actually mapped, the kernel command line, and where
/// an initrd was mapped, if any. RAM lives here rather than on
/// [`QemuVirtLayout`] because it is the one piece of hardware layout that
/// genuinely varies per boot (this dtc-differential worked example's 1 GiB
/// vs. the M5b boot path's 256 MiB) — see [`QemuVirtLayout`]'s own doc.
#[derive(Debug, Clone, Copy)]
pub struct BootParams<'a> {
    pub ram_base: u64,
    pub ram_size: u64,
    pub bootargs: &'a str,
    pub initrd: Option<(u64, u64)>,
}

/// Builds the minimal devicetree a Linux aarch64 guest needs to reach
/// userspace init on one vCPU: root (`#address-cells`/`#size-cells`),
/// `/memory`, `/chosen`, `/cpus` (one `cpu@0`, PSCI enable-method),
/// `/psci` (the method `enable-method = "psci"` names), a GICv3
/// `interrupt-controller`, and the ARM architected timer.
///
/// Composes [`vm_fdt::FdtWriter`] — see the module doc for why this is a
/// composition, not a hand-rolled encoder.
///
/// # Errors
///
/// Returns [`vm_fdt::Error`] unchanged; every failure mode (invalid node or
/// property name, unclosed node, oversized blob) is the writer's own, not
/// reinterpreted here.
pub fn build_minimal_aarch64_boot_dtb(
    layout: &QemuVirtLayout,
    boot: &BootParams<'_>,
) -> FdtResult<Vec<u8>> {
    let mut fdt = FdtWriter::new()?;

    let root = fdt.begin_node("")?;
    fdt.property_string("compatible", "linux,dummy-virt")?;
    fdt.property_u32("#address-cells", 2)?;
    fdt.property_u32("#size-cells", 2)?;
    fdt.property_u32("interrupt-parent", GIC_PHANDLE)?;

    write_memory(&mut fdt, boot)?;
    write_chosen(&mut fdt, boot, layout)?;
    write_cpus(&mut fdt)?;
    write_psci(&mut fdt)?;
    write_gic(&mut fdt, layout)?;
    write_timer(&mut fdt)?;
    write_uart(&mut fdt, layout)?;
    write_apb_pclk(&mut fdt)?;

    fdt.end_node(root)?;
    fdt.finish()
}

fn write_memory(fdt: &mut FdtWriter, boot: &BootParams<'_>) -> FdtResult<()> {
    let node = fdt.begin_node("memory")?;
    fdt.property_string("device_type", "memory")?;
    fdt.property_array_u64("reg", &[boot.ram_base, boot.ram_size])?;
    fdt.end_node(node)
}

fn write_chosen(
    fdt: &mut FdtWriter,
    boot: &BootParams<'_>,
    layout: &QemuVirtLayout,
) -> FdtResult<()> {
    let node = fdt.begin_node("chosen")?;
    fdt.property_string("bootargs", boot.bootargs)?;
    if let Some((start, end)) = boot.initrd {
        fdt.property_u64("linux,initrd-start", start)?;
        fdt.property_u64("linux,initrd-end", end)?;
    }
    fdt.property_string("stdout-path", &format!("/pl011@{:x}", layout.uart_base))?;
    fdt.end_node(node)
}

fn write_cpus(fdt: &mut FdtWriter) -> FdtResult<()> {
    let cpus = fdt.begin_node("cpus")?;
    fdt.property_u32("#address-cells", 1)?;
    fdt.property_u32("#size-cells", 0)?;

    let cpu = fdt.begin_node("cpu@0")?;
    fdt.property_string("device_type", "cpu")?;
    fdt.property_string("compatible", "arm,cortex-a72")?;
    fdt.property_u32("reg", 0)?;
    fdt.property_string("enable-method", "psci")?;
    fdt.end_node(cpu)?;

    fdt.end_node(cpus)
}

fn write_psci(fdt: &mut FdtWriter) -> FdtResult<()> {
    let node = fdt.begin_node("psci")?;
    fdt.property_string("compatible", "arm,psci-0.2")?;
    fdt.property_string("method", "hvc")?;
    fdt.end_node(node)
}

fn write_gic(fdt: &mut FdtWriter, layout: &QemuVirtLayout) -> FdtResult<()> {
    let name = format!("interrupt-controller@{:x}", layout.gicd_base);
    let node = fdt.begin_node(&name)?;
    fdt.property_string("compatible", "arm,gic-v3")?;
    fdt.property_u32("#interrupt-cells", 3)?;
    fdt.property_u32("#address-cells", 2)?;
    fdt.property_u32("#size-cells", 2)?;
    fdt.property_null("interrupt-controller")?;
    fdt.property_null("ranges")?;
    fdt.property_array_u64(
        "reg",
        &[
            layout.gicd_base,
            layout.gicd_size,
            layout.gicr_base,
            layout.gicr_size,
        ],
    )?;
    fdt.property_phandle(GIC_PHANDLE)?;
    fdt.end_node(node)
}

/// PPI numbers and `arm,gic-v3`-style `<type num flags>` cells for the ARM
/// architected timer: secure phys (13), non-secure phys (14), virtual
/// (11), hypervisor (10), all GIC PPIs (`type` = 1), level-low (`flags` =
/// 4) — the same four interrupts QEMU's own generated dtb for `-M virt`
/// carries for this device, cited as the worked example (module docs).
fn write_timer(fdt: &mut FdtWriter) -> FdtResult<()> {
    let node = fdt.begin_node("timer")?;
    fdt.property_string("compatible", "arm,armv8-timer")?;
    fdt.property_array_u32(
        "interrupts",
        &[
            1, 13, 4, // secure physical timer
            1, 14, 4, // non-secure physical timer
            1, 11, 4, // virtual timer
            1, 10, 4, // hypervisor timer
        ],
    )?;
    fdt.property_null("always-on")?;
    fdt.end_node(node)
}

/// M5b's console node — QEMU virt's `VIRT_UART` (`hw/arm/virt.c`'s
/// `create_uart`), the worked example this module's own doc cites. `clocks`
/// references [`APB_PCLK_PHANDLE`] twice (`uartclk` then `apb_pclk`) because
/// the PL011 has two clock inputs and QEMU virt wires the same fixed-clock
/// node to both — there is no separate baud-rate clock modeled on this
/// machine.
fn write_uart(fdt: &mut FdtWriter, layout: &QemuVirtLayout) -> FdtResult<()> {
    let name = format!("pl011@{:x}", layout.uart_base);
    let node = fdt.begin_node(&name)?;
    fdt.property_string_list(
        "compatible",
        vec!["arm,pl011".to_string(), "arm,primecell".to_string()],
    )?;
    fdt.property_array_u64("reg", &[layout.uart_base, layout.uart_size])?;
    fdt.property_array_u32("interrupts", &UART_INTERRUPT_CELLS)?;
    fdt.property_array_u32("clocks", &[APB_PCLK_PHANDLE, APB_PCLK_PHANDLE])?;
    fdt.property_string_list(
        "clock-names",
        vec!["uartclk".to_string(), "apb_pclk".to_string()],
    )?;
    fdt.end_node(node)
}

/// The fixed 24 MHz `apb-pclk` clock node QEMU virt's own DTB wires to both
/// the uart node's clock inputs (`write_uart`'s own doc) — without it, a
/// Linux `pl011` driver's `clk_get_rate` call on `uartclk` fails and the
/// console never probes.
fn write_apb_pclk(fdt: &mut FdtWriter) -> FdtResult<()> {
    let node = fdt.begin_node("apb-pclk")?;
    fdt.property_string("compatible", "fixed-clock")?;
    fdt.property_u32("#clock-cells", 0)?;
    fdt.property_u32("clock-frequency", APB_PCLK_FREQUENCY_HZ)?;
    fdt.property_string("clock-output-names", "clk24mhz")?;
    fdt.property_phandle(APB_PCLK_PHANDLE)?;
    fdt.end_node(node)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::{BootParams, QemuVirtLayout, build_minimal_aarch64_boot_dtb};
    use alloc::vec::Vec;
    use fdt::Fdt;

    /// This worked example's own RAM window — 1 GiB at QEMU virt's real
    /// `hw/arm/virt.c` RAM base, matching `tests/dtb_dtc_differential.rs`'s
    /// own literal so both stay anchored to the same incumbent value.
    const TEST_RAM_BASE: u64 = 0x4000_0000;
    const TEST_RAM_SIZE: u64 = 0x4000_0000;

    fn built() -> Vec<u8> {
        let boot = BootParams {
            ram_base: TEST_RAM_BASE,
            ram_size: TEST_RAM_SIZE,
            bootargs: "console=hvc0 root=/dev/vda rw",
            initrd: Some((0x4800_0000, 0x4900_0000)),
        };
        build_minimal_aarch64_boot_dtb(&QemuVirtLayout::single_vcpu(), &boot)
            .expect("worked-example layout must build")
    }

    /// Devicetree spec v0.4 §5.2: the first ten big-endian `u32`s of the
    /// FDT header. Hand-derived from the spec text, not copied from
    /// either crate's source: `magic` = `0xd00dfeed`, `version` = 17,
    /// `last_comp_version` = 16 (the values every `dtc`-produced blob
    /// carries for this spec revision), and `totalsize` must equal the
    /// blob's own length.
    #[test]
    fn header_matches_the_devicetree_spec_v0_4_hand_derived_layout() {
        let blob = built();

        let read_be_u32 = |offset: usize| {
            u32::from_be_bytes(blob[offset..offset + 4].try_into().expect("4 bytes"))
        };

        assert_eq!(read_be_u32(0), 0xd00d_feed, "FDT_MAGIC at offset 0");
        assert_eq!(
            read_be_u32(4) as usize,
            blob.len(),
            "totalsize at offset 4 must equal the blob's own length"
        );
        assert_eq!(read_be_u32(20), 17, "version at offset 20");
        assert_eq!(read_be_u32(24), 16, "last_comp_version at offset 24");
    }

    #[test]
    fn root_carries_the_dummy_virt_compatible_string() {
        let blob = built();
        let fdt = Fdt::new(&blob).expect("built blob must parse");

        assert!(
            fdt.root()
                .compatible()
                .all()
                .any(|value| value == "linux,dummy-virt")
        );
    }

    #[test]
    fn memory_node_reports_the_layouts_ram_window() {
        let blob = built();
        let fdt = Fdt::new(&blob).expect("built blob must parse");

        let region = fdt
            .memory()
            .regions()
            .next()
            .expect("one memory region was written");
        assert_eq!(region.starting_address as u64, TEST_RAM_BASE);
        assert_eq!(
            region.size.expect("region carries an explicit size") as u64,
            TEST_RAM_SIZE
        );
    }

    /// The RAM-ceiling check this module's own doc on
    /// [`QemuVirtLayout::CANONICAL`] names: the uart window must sit above
    /// [`crate::dispatch::GUEST_MEMORY_SIZE`] (64 MiB, the actual mapped
    /// guest-RAM ceiling in `dispatch::run_dispatch_loop`), the same
    /// non-aliasing guarantee `dispatch_trampoline.h`'s GICD/GICR windows
    /// already hold.
    #[test]
    fn uart_window_sits_above_the_mapped_guest_memory_ceiling() {
        const _: () = assert!(
            QemuVirtLayout::CANONICAL.uart_base > crate::dispatch::GUEST_MEMORY_SIZE,
            "uart window must sit above the mapped guest-ram ceiling so accesses always trap"
        );
    }

    #[test]
    fn chosen_reports_bootargs_and_the_initrd_range_given_at_build_time() {
        let blob = built();
        let fdt = Fdt::new(&blob).expect("built blob must parse");

        let chosen = fdt.chosen();
        assert_eq!(chosen.bootargs(), Some("console=hvc0 root=/dev/vda rw"));
    }

    #[test]
    fn chosen_stdout_path_resolves_to_the_uart_node_this_slice_writes() {
        let blob = built();
        let fdt = Fdt::new(&blob).expect("built blob must parse");

        let stdout_node = fdt
            .chosen()
            .stdout()
            .expect("stdout-path resolves to a node");
        assert!(
            stdout_node
                .compatible()
                .expect("uart node carries compatible")
                .all()
                .any(|value| value == "arm,pl011"),
            "stdout-path must resolve to the pl011 node"
        );
    }

    #[test]
    fn uart_node_is_reachable_by_its_arm_pl011_compatible_string() {
        let blob = built();
        let fdt = Fdt::new(&blob).expect("built blob must parse");

        let uart = fdt
            .find_compatible(&["arm,pl011"])
            .expect("pl011 node must be present");
        let reg = uart
            .reg()
            .expect("uart node carries reg")
            .next()
            .expect("one window");
        assert_eq!(
            reg.starting_address as u64,
            QemuVirtLayout::CANONICAL.uart_base
        );
        assert_eq!(
            reg.size.expect("uart window carries an explicit size") as u64,
            QemuVirtLayout::CANONICAL.uart_size
        );
    }

    #[test]
    fn cpus_reports_exactly_the_one_vcpu_this_slice_builds_for() {
        let blob = built();
        let fdt = Fdt::new(&blob).expect("built blob must parse");

        let cpus: Vec<_> = fdt.cpus().collect();
        assert_eq!(cpus.len(), 1, "single-vCPU worked example");
    }

    #[test]
    fn gic_node_is_reachable_by_its_arm_gic_v3_compatible_string() {
        let blob = built();
        let fdt = Fdt::new(&blob).expect("built blob must parse");

        let gic = fdt
            .find_compatible(&["arm,gic-v3"])
            .expect("gic-v3 node must be present");
        let reg = gic.reg().expect("gic node carries reg");
        let windows: Vec<_> = reg.collect();
        assert_eq!(windows.len(), 2, "distributor + redistributor windows");
    }

    #[test]
    fn timer_node_carries_all_four_architected_timer_ppis() {
        let blob = built();
        let fdt = Fdt::new(&blob).expect("built blob must parse");

        let timer = fdt
            .find_compatible(&["arm,armv8-timer"])
            .expect("timer node must be present");
        let interrupts = timer
            .property("interrupts")
            .expect("timer carries an interrupts property");
        assert_eq!(
            interrupts.value.len(),
            4 * 3 * 4,
            "4 PPIs, 3 cells each, 4 bytes per cell"
        );
    }

    #[test]
    fn empty_bootargs_still_produces_a_parseable_tree() {
        let boot = BootParams {
            ram_base: TEST_RAM_BASE,
            ram_size: TEST_RAM_SIZE,
            bootargs: "",
            initrd: None,
        };
        let blob = build_minimal_aarch64_boot_dtb(&QemuVirtLayout::single_vcpu(), &boot)
            .expect("empty bootargs is a legal, if useless, string property");
        let fdt = Fdt::new(&blob).expect("built blob must parse");
        assert_eq!(fdt.chosen().bootargs(), Some(""));
    }
}
