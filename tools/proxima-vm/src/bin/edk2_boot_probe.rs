//! Signed-subprocess probe for `boot::boot_edk2_firmware` — the same
//! signed-binary shape `kernel_boot_probe.rs` establishes for
//! `boot::boot_linux_kernel`, driven from a subprocess so
//! `tests/edk2_boot.rs` can sign this small binary with the hypervisor
//! entitlement instead of the whole nextest test harness.
//!
//! `argv[1]`: path to a real edk2/AAVMF `-pflash` CODE volume (e.g.
//! `/opt/homebrew/share/qemu/edk2-aarch64-code.fd`). Writes the guest's
//! pl011 byte count and the trap statistics to stderr, and every byte the
//! guest wrote to `UARTDR` to stdout as raw bytes — `tests/edk2_boot.rs`
//! asserts on stdout directly, the same split `kernel_boot_probe.rs` uses.
//!
//! Writes stdout bytes on EITHER outcome of `boot::boot_edk2_firmware`'s
//! own inner `Result`, mirroring `kernel_boot_probe.rs`'s own doc on why:
//! an edk2 build that reaches its own SEC/PEI/early-DXE console banner and
//! then hits an unmodeled wall (a real, still-open possibility this task's
//! own scope names) still needs those bytes reported, not discarded.

use std::env;
use std::error::Error;
use std::fs;
use std::io::{self, Write};

use proxima_vm::boot;

const MAX_HYPERCALLS: usize = 10_000;
const PL011_CAPACITY: usize = 65_536;

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = env::args().skip(1);
    let firmware_path = arguments
        .next()
        .ok_or("usage: edk2_boot_probe <path-to-edk2-aarch64-code.fd>")?;

    let firmware = fs::read(&firmware_path)?;
    let (pl011_emitted, stats, loop_outcome) =
        boot::boot_edk2_firmware(&firmware, MAX_HYPERCALLS, PL011_CAPACITY)?;

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
         vtimer_activation_count={} wfi_wfe_trap_count={} entered_el2={}",
        stats.gicd_trap_count,
        stats.gicr_trap_count,
        stats.pl011_trap_count,
        stats.virtio_trap_count,
        stats.vtimer_activation_count,
        stats.wfi_wfe_trap_count,
        stats.entered_el2
    );
    if let Err(loop_error) = &loop_outcome {
        eprintln!("hypervisor loop did not reach a clean halt: {loop_error}");
    }
    io::stdout().write_all(&pl011_emitted)?;
    loop_outcome?;
    Ok(())
}
