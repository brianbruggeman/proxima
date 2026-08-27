//! M5b — booting a real arm64 Linux `Image` (`ROADMAP.md`'s M5 section:
//! "kernel console output arrives through our byte channel; the assertion is
//! on kernel-emitted bytes and the count is asserted nonzero").
//!
//! Composes the same primitives [`crate::dispatch::run_dispatch_loop`]
//! already proves against the lambda ELF guest — one call into the C-side
//! `proxima_vm_run_dispatch_loop` exit loop, the GIC/pl011/virtio device
//! models, and [`crate::dtb::build_minimal_aarch64_boot_dtb`] — against a
//! raw (non-ELF) kernel `Image` instead of an ELF binary. This is a second
//! *caller* of the one C dispatch loop, not a second loop: the exit-handling
//! C code this module links against is byte-for-byte the same function
//! `run_dispatch_loop` calls.
//!
//! # Why RAM moves to `0x4000_0000`
//!
//! The lambda ELF guest links at 0 and `run_dispatch_loop` maps its RAM at
//! `[0, guest_memory_size)` accordingly. A real Linux aarch64 guest instead
//! expects QEMU virt's own RAM base (`crate::dtb`'s module doc,
//! `hw/arm/virt.c`'s memmap) — `0x4000_0000` — because that is the address
//! its own devicetree (this module's own [`build_boot_dtb`]) advertises for
//! `/memory`, and the arm64 boot protocol (`Documentation/arm64/booting`)
//! requires the kernel `Image` to load at `RAM_base + text_offset`.
//!
//! This RAM window, `[0x4000_0000, 0x5000_0000)` at [`RAM_SIZE`] = 256 MiB,
//! sits entirely ABOVE every fixed MMIO window this crate already traps:
//! the GICv3 distributor (`[0x0800_0000, 0x0801_0000)`) and single-vCPU
//! redistributor (`[0x080a_0000, 0x080c_0000)`,
//! `crate::dtb::QemuVirtLayout::single_vcpu`), and the pl011 uart
//! (`[0x0900_0000, 0x0900_1000)`) all end below `0x4000_0000` — the reverse
//! of the ELF-guest path's own non-aliasing argument (there, RAM sits below
//! the device windows; here, RAM sits above them). The virtio-console/net/
//! blk windows (`dispatch_trampoline.h`'s `0x1000000000+`) sit far above
//! this RAM window's own end instead. Every one of these ranges is disjoint
//! from `[RAM_BASE, RAM_BASE + RAM_SIZE)` by construction — this module's
//! own [`tests::ram_window_is_disjoint_from_every_fixed_mmio_window`] checks
//! it at compile time.
//!
//! # Tier
//!
//! `std` (tier 2): this module drives a real hypervisor VM exit loop through
//! FFI, exactly like [`crate::dispatch::run_dispatch_loop`].

#![cfg(all(
    feature = "std",
    any(
        all(target_os = "linux", target_arch = "x86_64"),
        all(target_os = "macos", target_arch = "aarch64")
    )
))]

extern crate alloc;

use alloc::vec::Vec;
use std::ffi::CStr;
use std::os::raw::c_char;

use proxima_core::ProximaError;
use proxima_protocols::process::ChildResponse;

use crate::dispatch::FfiRecordingDispatcher;
use crate::dtb::{BootParams, QemuVirtLayout};
use crate::loader::RawSegment;

/// QEMU virt's own RAM base (`crate::dtb`'s module doc, `hw/arm/virt.c`) —
/// where the arm64 boot protocol requires `Image` (`text_offset = 0`) and
/// this module's own devicetree to agree RAM starts.
pub const RAM_BASE: u64 = 0x4000_0000;

/// 256 MiB: enough headroom above a ~38 MiB `Image` (this slice's own
/// worked example) for early boot allocations, with room to spare before
/// [`DTB_OFFSET`].
pub const RAM_SIZE: u64 = 256 * 1024 * 1024;

/// Where this boot places the built DTB relative to [`RAM_BASE`] — high
/// enough that no `Image` this slice targets could ever grow into it
/// (`Documentation/arm64/booting`'s only DTB placement rule: 8-byte
/// aligned, in RAM, not overlapping the kernel).
pub const DTB_OFFSET: u64 = 240 * 1024 * 1024;

/// Where this boot places an optional initramfs (userspace's own PID 1
/// root filesystem) relative to [`RAM_BASE`] — clear of both the `Image`
/// (this slice's own worked example never exceeds ~38 MiB) and, at the
/// other end, [`DTB_OFFSET`] (this module's own
/// [`tests::initrd_window_sits_between_the_image_ceiling_and_the_dtb`]
/// checks both margins at compile time). The kernel reads `/chosen`'s
/// `linux,initrd-start`/`linux,initrd-end` (`crate::dtb::write_chosen`) to
/// find it, exactly the way it reads `bootargs` from the same node.
pub const INITRD_OFFSET: u64 = 200 * 1024 * 1024;

const ERROR_CAPACITY: usize = 512;

/// One serviced boot's instrumentation: wall nanoseconds from this call's
/// own entry to the first vCPU exit, wall nanoseconds to first-touch every
/// mapped page, and the count of MMIO-trap exits serviced during the run
/// (M3's three numbers, `crate::dispatch::run_dispatch_loop`'s own doc on
/// the same fields), plus M5b's per-window breakdown of `mmio_trap_count`
/// (`gicd`/`gicr`/`pl011`/`virtio`, `backend_macos.c::handle_mmio_data_abort`'s
/// own window resolution) and the count of
/// `HV_EXIT_REASON_VTIMER_ACTIVATED` exits serviced — the earlycon
/// investigation this struct exists for reads `pl011_trap_count`: zero means
/// earlycon never touched the device (a DTB/bootargs problem), nonzero with
/// zero emitted bytes means the pl011 model mishandled an access pattern
/// earlycon actually issued.
#[derive(Debug, Clone, Copy)]
pub struct BootTrapStatistics {
    pub create_to_first_exit_nanos: u64,
    pub touch_all_pages_nanos: u64,
    pub mmio_trap_count: u64,
    pub gicd_trap_count: u64,
    pub gicr_trap_count: u64,
    pub pl011_trap_count: u64,
    pub virtio_trap_count: u64,
    pub vtimer_activation_count: u64,
    /// Count of EC 0x1 (`WFI`/`WFE`) traps serviced — PID1's own idle park
    /// loop issues these once nothing is left to schedule
    /// (`backend_macos.c::proxima_vm_run_dispatch_loop`'s own EC-0x1 arm).
    /// Nonzero here, paired with a clean (`Ok`) [`BootOutcome`] loop
    /// outcome, is the "reached idle, ended cleanly" evidence this field
    /// exists to distinguish from the "unexpected arm exception class 0x1"
    /// failure this same trap used to produce before that arm existed.
    pub wfi_wfe_trap_count: u64,
    /// `true` only when this run both asked for EL2 entry (a nonzero
    /// `boot_cpsr` whose low nibble names EL2h/EL2t — only
    /// [`boot_edk2_firmware`] ever does) AND this host's own HVF actually
    /// honored it (`create_vm`'s own doc in `backend_macos.c` on the
    /// `HV_UNSUPPORTED` fallback this field distinguishes from). `false`
    /// for [`boot_linux_kernel`] always (it never requests EL2), and
    /// `false` for [`boot_edk2_firmware`] on any host whose HVF reports
    /// `hv_vm_config_get_el2_supported() == false` — MEASURED true on this
    /// investigation's own M1 Max / macOS 15.8 host.
    pub entered_el2: bool,
}

/// One completed setup's output: every pl011 byte captured, the trap
/// statistics, and the hypervisor loop's own outcome — see
/// [`boot_linux_kernel`]'s own doc for why the loop's outcome travels
/// inside this tuple rather than as the function's outer `Result`.
pub type BootOutcome = (Vec<u8>, BootTrapStatistics, Result<(), ProximaError>);

/// Loads `image` (a real arm64 Linux `Image`, uncompressed, `text_offset ==
/// 0`) at [`RAM_BASE`], builds and loads a minimal devicetree at
/// `RAM_BASE + `[`DTB_OFFSET`] carrying `bootargs` in `/chosen`, and drives
/// the same real hypervisor exit loop [`crate::dispatch::run_dispatch_loop`]
/// drives — MMU off, `PC = RAM_BASE`, `x0` = the DTB's physical address,
/// `x1..=x3 = 0` (`Documentation/arm64/booting`), EL1h (the same reset
/// CPSR/PSTATE every guest this crate boots enters at,
/// `backend_macos.c::create_and_start_vcpu`) — until the guest halts, the
/// hypercall budget is exceeded, or an unrecoverable exit occurs.
///
/// Returns every byte the guest wrote to the pl011's `UARTDR` register, in
/// write order — this VM's console byte channel, M5's exit criterion names
/// — alongside [`BootTrapStatistics`] and the loop's own outcome, ALWAYS,
/// even when that outcome is `Err`: `tests/kernel_boot.rs`'s own module doc
/// states the M5b contract precisely ("a kernel panicking with no initramfs
/// still counts, as long as the panic message itself crossed the pl011
/// channel first"), which means the caller needs the bytes on a
/// non-clean-halt run exactly as much as it needs them on a clean one. The
/// outer `Result` is reserved for failures where no hypervisor loop ever
/// ran at all (the DTB build itself, or a placement conflict) — there are
/// no bytes to report in that case because the guest never executed.
///
/// # Errors
///
/// The outer `Result` returns [`ProximaError::Config`] only when the DTB
/// build or segment layout fails before any vCPU runs. Once the hypervisor
/// loop starts, its own outcome travels in the returned tuple's inner
/// `Result` — [`ProximaError::Upstream`] there names the failing hypervisor
/// call or an exceeded hypercall/exit budget, alongside whatever bytes were
/// captured before that point.
#[allow(
    clippy::too_many_arguments,
    reason = "mirrors crate::dispatch::run_dispatch_loop's own per-channel-capacity \
              parameter list, the established shape for this crate's FFI-facing entry points"
)]
pub fn boot_linux_kernel(
    image: &[u8],
    bootargs: &str,
    initramfs: Option<&[u8]>,
    max_hypercalls: usize,
    pl011_capacity: usize,
) -> Result<BootOutcome, ProximaError> {
    let dtb_address = RAM_BASE + DTB_OFFSET;
    let initrd = initramfs.map(|bytes| {
        (
            RAM_BASE + INITRD_OFFSET,
            RAM_BASE + INITRD_OFFSET + bytes.len() as u64,
        )
    });
    let boot = BootParams {
        ram_base: RAM_BASE,
        ram_size: RAM_SIZE,
        bootargs,
        initrd,
    };
    let dtb = crate::dtb::build_minimal_aarch64_boot_dtb(&QemuVirtLayout::single_vcpu(), &boot)
        .map_err(|error| ProximaError::Config(alloc::format!("build boot dtb: {error:?}")))?;

    if DTB_OFFSET + dtb.len() as u64 > RAM_SIZE {
        return Err(ProximaError::Config(alloc::format!(
            "built dtb ({} bytes) does not fit before the ram window's end at offset {DTB_OFFSET}",
            dtb.len()
        )));
    }
    if image.len() as u64 > INITRD_OFFSET {
        return Err(ProximaError::Config(alloc::format!(
            "image ({} bytes) would overlap the initrd window placed at offset {INITRD_OFFSET}",
            image.len()
        )));
    }
    if let Some(bytes) = initramfs
        && INITRD_OFFSET + bytes.len() as u64 > DTB_OFFSET
    {
        return Err(ProximaError::Config(alloc::format!(
            "initramfs ({} bytes) placed at offset {INITRD_OFFSET} would overlap the dtb at \
             offset {DTB_OFFSET}",
            bytes.len()
        )));
    }

    // The kernel's own page tables, once it enables its MMU, cover every
    // byte the DTB's `/memory` node declares (`[RAM_BASE, RAM_BASE +
    // RAM_SIZE)`) -- early boot touches pages well outside the `Image`/DTB/
    // initrd byte ranges themselves (zeroing bss, the initial task's kernel
    // stack, `swapper_pg_dir`). Every gap between real segments is filled
    // with a data-free RW-non-exec region (`RawSegment::stack`, the same
    // shape `dispatch::run_dispatch_loop` already uses for the lambda
    // guest's own stack reservation) so `hv_vm_map` backs the WHOLE RAM
    // window the DTB advertises, not just the bytes this boot loaded.
    let dtb_end = DTB_OFFSET + dtb.len() as u64;
    let image_len = image.len() as u64;
    let mut raw_segments = alloc::vec![
        RawSegment::raw(0, image, image_len, true, true, true),
        RawSegment::raw(DTB_OFFSET, &dtb, dtb.len() as u64, true, true, false),
    ];
    let initrd_end = if let Some(bytes) = initramfs {
        let end = INITRD_OFFSET + bytes.len() as u64;
        raw_segments.push(RawSegment::raw(
            INITRD_OFFSET,
            bytes,
            bytes.len() as u64,
            true,
            true,
            false,
        ));
        if image_len < INITRD_OFFSET {
            raw_segments.push(RawSegment::stack(image_len, INITRD_OFFSET - image_len));
        }
        end
    } else if image_len < DTB_OFFSET {
        raw_segments.push(RawSegment::stack(image_len, DTB_OFFSET - image_len));
        DTB_OFFSET
    } else {
        image_len
    };
    if initrd_end < DTB_OFFSET {
        raw_segments.push(RawSegment::stack(initrd_end, DTB_OFFSET - initrd_end));
    }
    if dtb_end < RAM_SIZE {
        raw_segments.push(RawSegment::stack(dtb_end, RAM_SIZE - dtb_end));
    }

    unsafe extern "C" {
        fn proxima_vm_run_dispatch_loop(
            segments: *const RawSegment,
            segment_count: usize,
            guest_memory_size: u64,
            guest_memory_base: u64,
            entry: u64,
            boot_x0: u64,
            boot_cpsr: u64,
            dispatcher: *const core::ffi::c_void,
            console_transport: *mut core::ffi::c_void,
            net_transport: *mut core::ffi::c_void,
            blk_transport: *mut core::ffi::c_void,
            gicd_transport: *mut core::ffi::c_void,
            gicr_transport: *mut core::ffi::c_void,
            pl011_transport: *mut core::ffi::c_void,
            icc_transport: *mut core::ffi::c_void,
            max_hypercalls: usize,
            watchdog_millis: u64,
            emitted_out: *mut u8,
            emitted_capacity: usize,
            emitted_length_out: *mut usize,
            mmio_emitted_out: *mut u8,
            mmio_emitted_capacity: usize,
            mmio_emitted_length_out: *mut usize,
            net_emitted_out: *mut u8,
            net_emitted_capacity: usize,
            net_emitted_length_out: *mut usize,
            blk_emitted_out: *mut u8,
            blk_emitted_capacity: usize,
            blk_emitted_length_out: *mut usize,
            pl011_emitted_out: *mut u8,
            pl011_emitted_capacity: usize,
            pl011_emitted_length_out: *mut usize,
            create_to_first_exit_nanos_out: *mut u64,
            touch_all_pages_nanos_out: *mut u64,
            mmio_trap_count_out: *mut u64,
            gicd_trap_count_out: *mut u64,
            gicr_trap_count_out: *mut u64,
            pl011_trap_count_out: *mut u64,
            virtio_trap_count_out: *mut u64,
            vtimer_activation_count_out: *mut u64,
            wfi_wfe_trap_count_out: *mut u64,
            entered_el2_out: *mut u64,
            error_buffer: *mut c_char,
            error_capacity: usize,
        ) -> i32;
    }

    // this boot's guest speaks no `ChildRequest` hypercall protocol at all;
    // the dispatcher exists only to satisfy the C loop's signature, exactly
    // like `run_dispatch_loop`'s own instance -- a real kernel never issues
    // our `hvc`-encoded verbs, so it is never actually invoked.
    let dispatcher = FfiRecordingDispatcher::new(ChildResponse::Close);
    let mut console_transport =
        crate::virtio_console::ConsoleTransport::new(proxima_protocols::virtio::FEATURE_VERSION_1);
    let mut net_transport = crate::virtio_net::NetTransport::new([0_u8; 6]);
    let mut blk_transport = crate::virtio_blk::BlkTransport::new(0);
    let mut gicd_transport = crate::gic::GicDistributor::new();
    let mut gicr_transport = crate::gic::GicRedistributor::new();
    let mut pl011_transport = crate::pl011::Pl011Uart::new();
    // M5b ICC slice: the GIC's CPU-interface block, trapped via EC 0x18
    // MSR/MRS rather than MMIO -- Linux's `gic_cpu_sys_reg_init` probe path
    // is exactly this boot's own next wall until this register file lands.
    let mut icc_transport = crate::gic::IccCpuInterface::new();

    // this boot's guest never drives the hypercall/virtio-console/net/blk
    // channels; one-byte scratch buffers are enough to prove that if it's
    // ever wrong, since any write at all overruns capacity 0's implicit
    // "nothing fits" check.
    const UNUSED_CHANNEL_CAPACITY: usize = 1;
    let mut emitted = [0_u8; UNUSED_CHANNEL_CAPACITY];
    let mut emitted_length: usize = 0;
    let mut mmio_emitted = [0_u8; UNUSED_CHANNEL_CAPACITY];
    let mut mmio_emitted_length: usize = 0;
    let mut net_emitted = [0_u8; UNUSED_CHANNEL_CAPACITY];
    let mut net_emitted_length: usize = 0;
    let mut blk_emitted = [0_u8; UNUSED_CHANNEL_CAPACITY];
    let mut blk_emitted_length: usize = 0;
    let mut pl011_emitted = alloc::vec![0_u8; pl011_capacity];
    let mut pl011_emitted_length: usize = 0;
    let mut create_to_first_exit_nanos: u64 = 0;
    let mut touch_all_pages_nanos: u64 = 0;
    let mut mmio_trap_count: u64 = 0;
    let mut gicd_trap_count: u64 = 0;
    let mut gicr_trap_count: u64 = 0;
    let mut pl011_trap_count: u64 = 0;
    let mut virtio_trap_count: u64 = 0;
    let mut vtimer_activation_count: u64 = 0;
    let mut wfi_wfe_trap_count: u64 = 0;
    let mut entered_el2: u64 = 0;
    let mut error_buffer = [0_i8; ERROR_CAPACITY];

    let status = unsafe {
        proxima_vm_run_dispatch_loop(
            raw_segments.as_ptr(),
            raw_segments.len(),
            RAM_SIZE,
            RAM_BASE,
            RAM_BASE,
            dtb_address,
            // sentinel: a real Linux `Image` enters this loop's own EL1h
            // default; only `boot_edk2_firmware` passes a real CPSR.
            0,
            (&raw const dispatcher).cast(),
            (&raw mut console_transport).cast(),
            (&raw mut net_transport).cast(),
            (&raw mut blk_transport).cast(),
            (&raw mut gicd_transport).cast(),
            (&raw mut gicr_transport).cast(),
            (&raw mut pl011_transport).cast(),
            (&raw mut icc_transport).cast(),
            max_hypercalls,
            // no watchdog on the Linux-kernel boot path.
            0,
            emitted.as_mut_ptr(),
            emitted.len(),
            &raw mut emitted_length,
            mmio_emitted.as_mut_ptr(),
            mmio_emitted.len(),
            &raw mut mmio_emitted_length,
            net_emitted.as_mut_ptr(),
            net_emitted.len(),
            &raw mut net_emitted_length,
            blk_emitted.as_mut_ptr(),
            blk_emitted.len(),
            &raw mut blk_emitted_length,
            pl011_emitted.as_mut_ptr(),
            pl011_emitted.len(),
            &raw mut pl011_emitted_length,
            &raw mut create_to_first_exit_nanos,
            &raw mut touch_all_pages_nanos,
            &raw mut mmio_trap_count,
            &raw mut gicd_trap_count,
            &raw mut gicr_trap_count,
            &raw mut pl011_trap_count,
            &raw mut virtio_trap_count,
            &raw mut vtimer_activation_count,
            &raw mut wfi_wfe_trap_count,
            &raw mut entered_el2,
            error_buffer.as_mut_ptr(),
            error_buffer.len(),
        )
    };
    // `pl011_emitted_length_out` (and the other four `*_emitted_length_out`
    // params, `backend_macos.c`'s own fix note) used to be suppressed on a
    // non-zero `status`, which is exactly why M5b's own earlier
    // investigation could not tell "earlycon wrote nothing" from "earlycon
    // wrote plenty, but this boot never reached a clean halt to report it"
    // -- the length is now unconditional at the C level, so truncating here
    // is correct whether `status` is 0 or not.
    pl011_emitted.truncate(pl011_emitted_length);
    let stats = BootTrapStatistics {
        create_to_first_exit_nanos,
        touch_all_pages_nanos,
        mmio_trap_count,
        gicd_trap_count,
        gicr_trap_count,
        pl011_trap_count,
        virtio_trap_count,
        vtimer_activation_count,
        wfi_wfe_trap_count,
        entered_el2: entered_el2 != 0,
    };
    // the hypervisor loop's own outcome travels as the INNER `Result` (this
    // function's own doc explains why) -- `pl011_emitted`/`stats` are
    // returned either way, since `tests/kernel_boot.rs`'s own M5b contract
    // counts a non-clean-halt boot as a pass whenever bytes crossed first.
    let loop_outcome = if status == 0 {
        Ok(())
    } else {
        let message = unsafe { CStr::from_ptr(error_buffer.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        Err(ProximaError::Upstream(alloc::format!(
            "{message} (create_to_first_exit_nanos={create_to_first_exit_nanos} \
             touch_all_pages_nanos={touch_all_pages_nanos} mmio_trap_count={mmio_trap_count} \
             gicd_trap_count={gicd_trap_count} gicr_trap_count={gicr_trap_count} \
             pl011_trap_count={pl011_trap_count} virtio_trap_count={virtio_trap_count} \
             vtimer_activation_count={vtimer_activation_count} \
             wfi_wfe_trap_count={wfi_wfe_trap_count} \
             pl011_emitted_length={pl011_emitted_length})"
        )))
    };
    Ok((pl011_emitted, stats, loop_outcome))
}

/// QEMU virt's own flash-CODE base (`hw/arm/virt.c`'s memmap:
/// `VIRT_FLASH` split into two 64 MiB regions, CODE `[0, 0x0400_0000)` then
/// VARS `[0x0400_0000, 0x0800_0000)`) — where an edk2 `ArmVirtQemu` build's
/// own AArch64 reset vector sits, unconditionally, at the very start of the
/// CODE volume (`FLASH_BASE`'s own doc — no `text_offset` negotiation the
/// way [`RAM_BASE`]'s Linux `Image` path needs).
pub const FLASH_BASE: u64 = 0x0000_0000;

/// The CODE flash volume's own well-known size for this QEMU/AAVMF
/// convention (`/opt/homebrew/share/qemu/edk2-aarch64-code.fd`, 64 MiB —
/// MEASURED via `ls -la` this slice's own investigation). A real `.fd` file
/// shorter than this is still accepted (mapped at its own length); one
/// longer is rejected, since it would run into [`crate::dtb::QemuVirtLayout::CANONICAL`]'s
/// own `gicd_base` at `0x0800_0000` (this module's own
/// [`tests::flash_window_is_disjoint_from_the_gic`] checks the boundary at
/// compile time).
pub const FLASH_CODE_MAX_SIZE: u64 = 64 * 1024 * 1024;

/// EL2h reset `CPSR`/`PSTATE` (`M[3:0] == 0b1001`, ARM DDI 0487's own
/// `PSTATE.M` encoding) — the same `D`/`A`/`I`/`F`-all-masked shape
/// `backend_macos.c::create_and_start_vcpu`'s existing `0x3c5u` (EL1h)
/// reset value already commits every other boot this crate drives to,
/// with only the low nibble's mode field changed. Tried first for
/// edk2/`ArmVirtQemu`
/// (`ArmPlatformPkg`'s own reset code expects to run its EL2->EL1 drop
/// itself on a QEMU `virt` guest that presents EL2 — this module's own doc
/// on [`boot_edk2_firmware`] names the risk if a specific firmware build
/// disagrees).
pub const EDK2_ENTRY_CPSR_EL2H: u64 = 0x3c9;

/// Bounds this boot's own `hv_vcpu_run` call with a forced diagnostic exit
/// (`backend_macos.c::arm_watchdog`'s own doc) rather than leaving a
/// firmware that never traps hanging the caller forever: this
/// investigation's own MEASURED finding — a real edk2/AAVMF `-pflash` CODE
/// volume, entered at EL1h (this host's HVF reports `hv_vm_config_get_el2_supported()
/// == false`), executes for at least 90 continuous seconds with ZERO VM
/// exits (`sample(1)`, 1540/1540 samples inside `Hv::Vcpu::run()`) — means
/// [`boot_linux_kernel`]'s own unbounded-block shape is the wrong default
/// for this specific boot path.
pub const EDK2_WATCHDOG_MILLIS: u64 = 15_000;

/// Loads `firmware` (a real edk2/AAVMF `-pflash` CODE volume, e.g.
/// `/opt/homebrew/share/qemu/edk2-aarch64-code.fd`) at [`FLASH_BASE`],
/// read+execute, maps [`RAM_BASE`]'s window the same way
/// [`boot_linux_kernel`] does (edk2 relocates itself into DRAM once its DXE
/// phase starts, per this module's own investigation), builds and loads the
/// same minimal devicetree at `RAM_BASE + `[`DTB_OFFSET`] (QEMU's own
/// convention: `x0` carries the DTB pointer on the `-pflash` firmware path
/// exactly like the direct-kernel-boot path, and edk2's `FdtClientDxe`
/// consumes it the same way Linux's earlycon path does), and drives the
/// same real hypervisor exit loop [`boot_linux_kernel`] drives — `PC =
/// FLASH_BASE`, [`EDK2_ENTRY_CPSR_EL2H`] rather than EL1h, no VARS flash
/// (a minimal SEC/PEI/early-DXE console-output attempt does not need
/// persisted UEFI variables — this module's own investigation).
///
/// A second, dedicated function rather than a `boot_linux_kernel` mode flag
/// (principle 1's reuse-first question, answered by writing the call site
/// both ways): `boot_linux_kernel`'s own signature carries `bootargs`,
/// `initramfs`, and an `Image`-shaped single RAM-relative load address that
/// have no edk2 analogue at all (edk2 reads no kernel command line and
/// loads at flash address 0, not `RAM_BASE`) — folding both shapes behind
/// one signature would need an enum parameter plus `Option`-wrapping every
/// field only one caller ever uses, which is a worse read at every call
/// site than two functions that each say exactly what they need. Both
/// still drive the one shared C exit loop (`proxima_vm_run_dispatch_loop`'s
/// own `boot_cpsr` parameter, threaded exactly like `boot_x0` already is)
/// — this is a second *caller*, not a second loop, the identical relationship this
/// module's own doc already states between [`boot_linux_kernel`] and
/// [`crate::dispatch::run_dispatch_loop`].
///
/// Returns every pl011 byte the same way [`boot_linux_kernel`] does, on
/// EITHER loop outcome — "reached the SEC/PEI/early-DXE debug banner
/// before the next wall" is exactly as real a partial result here as "the
/// kernel panicked after the marker crossed the channel" is for
/// [`crate::boot`]'s own M5b contract.
///
/// # Errors
///
/// The outer `Result` returns [`ProximaError::Config`] when `firmware`
/// exceeds [`FLASH_CODE_MAX_SIZE`] or the DTB build fails before any vCPU
/// runs — see [`boot_linux_kernel`]'s own doc for why the loop's own
/// outcome travels in the returned tuple's inner `Result` instead.
pub fn boot_edk2_firmware(
    firmware: &[u8],
    max_hypercalls: usize,
    pl011_capacity: usize,
) -> Result<BootOutcome, ProximaError> {
    if firmware.len() as u64 > FLASH_CODE_MAX_SIZE {
        return Err(ProximaError::Config(alloc::format!(
            "firmware ({} bytes) exceeds the flash code window's own {FLASH_CODE_MAX_SIZE}-byte \
             ceiling",
            firmware.len()
        )));
    }

    let dtb_address = RAM_BASE + DTB_OFFSET;
    let boot = BootParams {
        ram_base: RAM_BASE,
        ram_size: RAM_SIZE,
        bootargs: "",
        initrd: None,
    };
    let dtb = crate::dtb::build_minimal_aarch64_boot_dtb(&QemuVirtLayout::single_vcpu(), &boot)
        .map_err(|error| ProximaError::Config(alloc::format!("build boot dtb: {error:?}")))?;
    if DTB_OFFSET + dtb.len() as u64 > RAM_SIZE {
        return Err(ProximaError::Config(alloc::format!(
            "built dtb ({} bytes) does not fit before the ram window's end at offset {DTB_OFFSET}",
            dtb.len()
        )));
    }

    // Flash CODE volume, read+execute, at FLASH_BASE -- and the RAM window
    // (data-free RW-non-exec everywhere except the DTB, matching
    // `boot_linux_kernel`'s own gap-fill strategy) at RAM_BASE, since edk2
    // relocates itself there once DXE starts (this module's own doc). The
    // two windows are disjoint by construction
    // (`tests::flash_window_is_disjoint_from_the_gic`,
    // `tests::ram_window_is_disjoint_from_every_fixed_mmio_window`).
    let firmware_len = firmware.len() as u64;
    let dtb_end = DTB_OFFSET + dtb.len() as u64;
    let mut raw_segments = alloc::vec![RawSegment::raw(
        FLASH_BASE,
        firmware,
        firmware_len,
        true,
        false,
        true
    )];
    raw_segments.push(RawSegment::raw(
        RAM_BASE + DTB_OFFSET,
        &dtb,
        dtb.len() as u64,
        true,
        true,
        false,
    ));
    if DTB_OFFSET > 0 {
        raw_segments.push(RawSegment::stack(RAM_BASE, DTB_OFFSET));
    }
    if dtb_end < RAM_SIZE {
        raw_segments.push(RawSegment::stack(RAM_BASE + dtb_end, RAM_SIZE - dtb_end));
    }

    unsafe extern "C" {
        fn proxima_vm_run_dispatch_loop(
            segments: *const RawSegment,
            segment_count: usize,
            guest_memory_size: u64,
            guest_memory_base: u64,
            entry: u64,
            boot_x0: u64,
            boot_cpsr: u64,
            dispatcher: *const core::ffi::c_void,
            console_transport: *mut core::ffi::c_void,
            net_transport: *mut core::ffi::c_void,
            blk_transport: *mut core::ffi::c_void,
            gicd_transport: *mut core::ffi::c_void,
            gicr_transport: *mut core::ffi::c_void,
            pl011_transport: *mut core::ffi::c_void,
            icc_transport: *mut core::ffi::c_void,
            max_hypercalls: usize,
            watchdog_millis: u64,
            emitted_out: *mut u8,
            emitted_capacity: usize,
            emitted_length_out: *mut usize,
            mmio_emitted_out: *mut u8,
            mmio_emitted_capacity: usize,
            mmio_emitted_length_out: *mut usize,
            net_emitted_out: *mut u8,
            net_emitted_capacity: usize,
            net_emitted_length_out: *mut usize,
            blk_emitted_out: *mut u8,
            blk_emitted_capacity: usize,
            blk_emitted_length_out: *mut usize,
            pl011_emitted_out: *mut u8,
            pl011_emitted_capacity: usize,
            pl011_emitted_length_out: *mut usize,
            create_to_first_exit_nanos_out: *mut u64,
            touch_all_pages_nanos_out: *mut u64,
            mmio_trap_count_out: *mut u64,
            gicd_trap_count_out: *mut u64,
            gicr_trap_count_out: *mut u64,
            pl011_trap_count_out: *mut u64,
            virtio_trap_count_out: *mut u64,
            vtimer_activation_count_out: *mut u64,
            wfi_wfe_trap_count_out: *mut u64,
            entered_el2_out: *mut u64,
            error_buffer: *mut c_char,
            error_capacity: usize,
        ) -> i32;
    }

    // this boot's guest speaks no `ChildRequest` hypercall protocol at all --
    // same reasoning as `boot_linux_kernel`'s own identical local.
    let dispatcher = FfiRecordingDispatcher::new(ChildResponse::Close);
    let mut console_transport =
        crate::virtio_console::ConsoleTransport::new(proxima_protocols::virtio::FEATURE_VERSION_1);
    let mut net_transport = crate::virtio_net::NetTransport::new([0_u8; 6]);
    let mut blk_transport = crate::virtio_blk::BlkTransport::new(0);
    let mut gicd_transport = crate::gic::GicDistributor::new();
    let mut gicr_transport = crate::gic::GicRedistributor::new();
    let mut pl011_transport = crate::pl011::Pl011Uart::new();
    let mut icc_transport = crate::gic::IccCpuInterface::new();

    const UNUSED_CHANNEL_CAPACITY: usize = 1;
    let mut emitted = [0_u8; UNUSED_CHANNEL_CAPACITY];
    let mut emitted_length: usize = 0;
    let mut mmio_emitted = [0_u8; UNUSED_CHANNEL_CAPACITY];
    let mut mmio_emitted_length: usize = 0;
    let mut net_emitted = [0_u8; UNUSED_CHANNEL_CAPACITY];
    let mut net_emitted_length: usize = 0;
    let mut blk_emitted = [0_u8; UNUSED_CHANNEL_CAPACITY];
    let mut blk_emitted_length: usize = 0;
    let mut pl011_emitted = alloc::vec![0_u8; pl011_capacity];
    let mut pl011_emitted_length: usize = 0;
    let mut create_to_first_exit_nanos: u64 = 0;
    let mut touch_all_pages_nanos: u64 = 0;
    let mut mmio_trap_count: u64 = 0;
    let mut gicd_trap_count: u64 = 0;
    let mut gicr_trap_count: u64 = 0;
    let mut pl011_trap_count: u64 = 0;
    let mut virtio_trap_count: u64 = 0;
    let mut vtimer_activation_count: u64 = 0;
    let mut wfi_wfe_trap_count: u64 = 0;
    let mut entered_el2: u64 = 0;
    let mut error_buffer = [0_i8; ERROR_CAPACITY];

    let status = unsafe {
        proxima_vm_run_dispatch_loop(
            raw_segments.as_ptr(),
            raw_segments.len(),
            // this loop's C side copies every segment into ONE flat
            // host-side buffer at its own `guest_address` OFFSET, then maps
            // each window at `guest_memory_base + offset` -- unlike
            // `boot_linux_kernel`, whose every segment shares one shifted
            // RAM window, this boot needs two DISJOINT absolute guest-
            // physical windows (flash at 0, ram at RAM_BASE), so every
            // segment above already carries its own absolute
            // `guest_address` and `guest_memory_base` stays 0 -- the flat
            // buffer just needs to be large enough to hold the highest
            // absolute address any segment uses (`RAM_BASE + RAM_SIZE`);
            // the unused gap between the flash volume's end and `RAM_BASE`
            // is never mapped into the guest's IPA space at all
            // (`build_mapped_windows` only ever maps segment-covered
            // ranges), only allocated (and never touched) on the host side.
            RAM_BASE + RAM_SIZE,
            0,
            FLASH_BASE,
            dtb_address,
            EDK2_ENTRY_CPSR_EL2H,
            (&raw const dispatcher).cast(),
            (&raw mut console_transport).cast(),
            (&raw mut net_transport).cast(),
            (&raw mut blk_transport).cast(),
            (&raw mut gicd_transport).cast(),
            (&raw mut gicr_transport).cast(),
            (&raw mut pl011_transport).cast(),
            (&raw mut icc_transport).cast(),
            max_hypercalls,
            EDK2_WATCHDOG_MILLIS,
            emitted.as_mut_ptr(),
            emitted.len(),
            &raw mut emitted_length,
            mmio_emitted.as_mut_ptr(),
            mmio_emitted.len(),
            &raw mut mmio_emitted_length,
            net_emitted.as_mut_ptr(),
            net_emitted.len(),
            &raw mut net_emitted_length,
            blk_emitted.as_mut_ptr(),
            blk_emitted.len(),
            &raw mut blk_emitted_length,
            pl011_emitted.as_mut_ptr(),
            pl011_emitted.len(),
            &raw mut pl011_emitted_length,
            &raw mut create_to_first_exit_nanos,
            &raw mut touch_all_pages_nanos,
            &raw mut mmio_trap_count,
            &raw mut gicd_trap_count,
            &raw mut gicr_trap_count,
            &raw mut pl011_trap_count,
            &raw mut virtio_trap_count,
            &raw mut vtimer_activation_count,
            &raw mut wfi_wfe_trap_count,
            &raw mut entered_el2,
            error_buffer.as_mut_ptr(),
            error_buffer.len(),
        )
    };
    pl011_emitted.truncate(pl011_emitted_length);
    let stats = BootTrapStatistics {
        create_to_first_exit_nanos,
        touch_all_pages_nanos,
        mmio_trap_count,
        gicd_trap_count,
        gicr_trap_count,
        pl011_trap_count,
        virtio_trap_count,
        vtimer_activation_count,
        wfi_wfe_trap_count,
        entered_el2: entered_el2 != 0,
    };
    let loop_outcome = if status == 0 {
        Ok(())
    } else {
        let message = unsafe { CStr::from_ptr(error_buffer.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        Err(ProximaError::Upstream(alloc::format!(
            "{message} (create_to_first_exit_nanos={create_to_first_exit_nanos} \
             touch_all_pages_nanos={touch_all_pages_nanos} mmio_trap_count={mmio_trap_count} \
             gicd_trap_count={gicd_trap_count} gicr_trap_count={gicr_trap_count} \
             pl011_trap_count={pl011_trap_count} virtio_trap_count={virtio_trap_count} \
             vtimer_activation_count={vtimer_activation_count} \
             wfi_wfe_trap_count={wfi_wfe_trap_count} \
             pl011_emitted_length={pl011_emitted_length})"
        )))
    };
    Ok((pl011_emitted, stats, loop_outcome))
}

#[cfg(test)]
mod tests {
    use super::{DTB_OFFSET, FLASH_BASE, FLASH_CODE_MAX_SIZE, INITRD_OFFSET, QemuVirtLayout, RAM_BASE, RAM_SIZE};

    /// [`FLASH_CODE_MAX_SIZE`]'s own doc names this check: the edk2 CODE
    /// flash window this module maps at [`FLASH_BASE`] must end at or
    /// before the GICv3 distributor's own fixed base — the scout
    /// investigation's own measured finding (`0x0000_0000..0x0400_0000`
    /// ends exactly where `0x0800_0000` begins, non-overlapping) checked at
    /// compile time rather than only asserted in prose.
    #[test]
    fn flash_window_is_disjoint_from_the_gic() {
        let layout = QemuVirtLayout::single_vcpu();
        assert!(
            FLASH_BASE + FLASH_CODE_MAX_SIZE <= layout.gicd_base,
            "the flash code window must end at or before the gicd base"
        );
    }

    /// [`INITRD_OFFSET`]'s own doc names this check: an initramfs placed at
    /// this offset must sit clear of the `Image` ceiling this slice's own
    /// worked example uses (39,845,888 bytes, well under 200 MiB) and clear
    /// of [`DTB_OFFSET`] on the other side, with room for a real initramfs
    /// (this slice's own worked example, 359,172 bytes) between the two.
    #[test]
    fn initrd_window_sits_between_the_image_ceiling_and_the_dtb() {
        const WORKED_EXAMPLE_IMAGE_BYTES: u64 = 39_845_888;
        const WORKED_EXAMPLE_INITRAMFS_BYTES: u64 = 359_172;

        const {
            assert!(
                WORKED_EXAMPLE_IMAGE_BYTES < INITRD_OFFSET,
                "the worked-example image must not reach the initrd window"
            );
        };
        const {
            assert!(
                INITRD_OFFSET + WORKED_EXAMPLE_INITRAMFS_BYTES < DTB_OFFSET,
                "the worked-example initramfs must not reach the dtb"
            );
        };
        const { assert!(INITRD_OFFSET < DTB_OFFSET, "initrd offset must precede the dtb") };
    }

    /// The non-aliasing argument this module's own doc makes, checked at
    /// compile time rather than asserted only in prose: every fixed MMIO
    /// window this crate already traps sits entirely outside
    /// `[RAM_BASE, RAM_BASE + RAM_SIZE)`.
    #[test]
    fn ram_window_is_disjoint_from_every_fixed_mmio_window() {
        let layout = QemuVirtLayout::single_vcpu();
        let ram_end = RAM_BASE + RAM_SIZE;

        assert!(
            layout.gicd_base + layout.gicd_size <= RAM_BASE,
            "gicd window must end before ram starts"
        );
        assert!(
            layout.gicr_base + layout.gicr_size <= RAM_BASE,
            "gicr window must end before ram starts"
        );
        assert!(
            layout.uart_base + layout.uart_size <= RAM_BASE,
            "pl011 window must end before ram starts"
        );
        // `dispatch_trampoline.h`'s three virtio-mmio windows all start at
        // `0x1000000000` or above -- this crate's own established headroom
        // for "always above every real guest's mapped memory."
        const VIRTIO_MMIO_WINDOW_BASE: u64 = 0x1000000000;
        assert!(
            ram_end <= VIRTIO_MMIO_WINDOW_BASE,
            "ram window must end before the virtio-mmio windows start"
        );
        const { assert!(DTB_OFFSET < RAM_SIZE, "dtb offset must land inside ram") };
    }
}
