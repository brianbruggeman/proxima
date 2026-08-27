//! Signed-subprocess probe for `boot::boot_linux_kernel` — M5's exit
//! criterion, driven from a subprocess so `tests/kernel_boot.rs` can sign
//! this small binary with the hypervisor entitlement instead of the whole
//! nextest test harness, the same shape `dispatch_probe.rs` already
//! establishes.
//!
//! argv[1]: path to a real arm64 Linux `Image`. argv[2] (optional): path to
//! an initramfs `cpio` archive — `boot::boot_linux_kernel`'s own `initramfs`
//! parameter, threaded through so a guest with no root filesystem never
//! reaches userspace on its own (`tests/kernel_boot_userspace.rs`'s own
//! module doc). Writes the guest's pl011 byte count and the M3 trap
//! statistics to stderr, and every byte the guest wrote to `UARTDR` to
//! stdout as raw bytes — `tests/kernel_boot.rs` asserts on stdout directly,
//! the same split `dispatch_probe.rs` uses to keep the assertion channel
//! unambiguous.
//!
//! Writes stdout bytes on EITHER outcome of `boot::boot_linux_kernel`'s own
//! inner `Result` (`boot.rs`'s own doc explains why: `tests/kernel_boot.rs`'s
//! own M5b contract counts a non-clean-halt boot as a pass whenever bytes
//! crossed the pl011 channel first) — only a setup failure before any vCPU
//! ran (the outer `Result`, no bytes possible) or an empty byte count
//! becomes this process's nonzero exit.

use std::env;
use std::error::Error;
use std::fs;
use std::io::{self, Write};

use proxima_vm::boot;

const MAX_HYPERCALLS: usize = 10_000;
const PL011_CAPACITY: usize = 65_536;
// `nosmp`/`nr_cpus=1` (routes around `ICC_SGI1R_EL1`'s own investigation:
// with no secondary CPU to bring up, `smp_init`'s IPI machinery never fires
// at all, the cheapest possible answer to the SMP-IPI wall this probe's own
// bootargs line names) precede the always-present earlycon/console/panic
// trio the M5/M5b investigations already anchored.
const BOOTARGS: &str =
    "nosmp nr_cpus=1 earlycon=pl011,mmio32,0x9000000 console=ttyAMA0 panic=-1";

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = env::args().skip(1);
    let image_path = arguments
        .next()
        .ok_or("usage: kernel_boot_probe <path-to-arm64-Image> [path-to-initramfs.cpio]")?;
    let initramfs_path = arguments.next();

    let image = fs::read(&image_path)?;
    let initramfs = initramfs_path.map(fs::read).transpose()?;
    let (pl011_emitted, stats, loop_outcome) = boot::boot_linux_kernel(
        &image,
        BOOTARGS,
        initramfs.as_deref(),
        MAX_HYPERCALLS,
        PL011_CAPACITY,
    )?;

    eprintln!(
        "guest drained {} pl011 byte(s) over this boot",
        pl011_emitted.len()
    );
    eprintln!(
        "m3 create_to_first_exit_nanos={} touch_all_pages_nanos={} mmio_trap_count={}",
        stats.create_to_first_exit_nanos, stats.touch_all_pages_nanos, stats.mmio_trap_count
    );
    eprintln!(
        "m5b gicd_trap_count={} gicr_trap_count={} pl011_trap_count={} virtio_trap_count={} \
         vtimer_activation_count={} wfi_wfe_trap_count={}",
        stats.gicd_trap_count,
        stats.gicr_trap_count,
        stats.pl011_trap_count,
        stats.virtio_trap_count,
        stats.vtimer_activation_count,
        stats.wfi_wfe_trap_count
    );
    if let Err(loop_error) = &loop_outcome {
        eprintln!("hypervisor loop did not reach a clean halt: {loop_error}");
    }
    // stdout carries bytes on EITHER outcome (this binary's own module doc);
    // written before the loop's own error (if any) is propagated, so a
    // caller reading stdout always sees whatever the guest emitted before
    // the wall, matching `tests/kernel_boot.rs`'s own M5b contract.
    io::stdout().write_all(&pl011_emitted)?;
    loop_outcome?;
    Ok(())
}
