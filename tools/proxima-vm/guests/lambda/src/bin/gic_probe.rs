//! Minimal `#![no_std]` guest whose only effect is reading two real GICv3
//! registers through real data-abort VM exits — `GICD_PIDR2` (distributor
//! window) and `GICR_TYPER`'s low word (redistributor window) — and emitting
//! both 32-bit values back to the host through the existing `EMIT_VERB`
//! channel one byte at a time (`main.rs`'s own convention), proving M5b's
//! GIC slice 3 wiring (`src/dispatch_trampoline.h`'s GICD/GICR windows,
//! `src/backend_macos.c`'s `handle_mmio_data_abort` routing,
//! `src/mmio_trampoline.rs`'s `proxima_vm_dispatch_mmio_gicd`/`_gicr`) is
//! reached by a real vCPU exit, not a synthesized in-memory call against
//! `src/gic.rs` directly.
//!
//! A deliberately separate `[[bin]]` from `proxima-vm-guest-lambda`
//! (`Cargo.toml`, `src/main.rs`) and from `proxima-vm-guest-lambda-psci`
//! (`src/bin/psci_probe.rs`) so this register-read sequence never perturbs
//! either binary's own pinned ELF layout
//! (`src/elf.rs::matches_readelf_on_the_real_aarch64_guest`) or pinned
//! emitted-byte contract. aarch64-only: the GIC is this VM's ARM-specific
//! interrupt controller, never x86_64's; `entry` on every other target is
//! just a park loop so the binary still compiles and links for the x86_64
//! parity build (`cargo build --target x86_64-unknown-none --bins`).

#![no_std]
#![no_main]

use core::arch::naked_asm;
use core::panic::PanicInfo;

/// Guest-physical address of `GICD_PIDR2` (GICD window base + offset
/// `0xffe8`, `src/gic.rs`'s `REG_PIDR2`) — GICD base `0x0800_0000` is
/// `src/dtb.rs`'s `QemuVirtLayout::CANONICAL.gicd_base`, the exact value
/// this VM's own DTB advertises to the guest, matching
/// `PROXIMA_VM_GICD_MMIO_WINDOW_BASE` (`src/dispatch_trampoline.h`).
#[cfg(target_arch = "aarch64")]
const GICD_PIDR2_ADDRESS: u64 = 0x0800_0000 + 0xffe8;

/// Guest-physical address of `GICR_TYPER`'s low word (GICR window base +
/// offset `0x0008`, `src/gic.rs`'s `REG_GICR_TYPER_LOW`). GICR base
/// `0x080a_0000` is `src/dtb.rs`'s `QemuVirtLayout::single_vcpu().gicr_base`
/// (identical to `QemuVirtLayout::CANONICAL.gicr_base`, only the redistributor
/// region's *size* differs between the two), matching
/// `PROXIMA_VM_GICR_MMIO_WINDOW_BASE`.
#[cfg(target_arch = "aarch64")]
const GICR_TYPER_LOW_ADDRESS: u64 = 0x080a_0000 + 0x0008;

/// Sentinel verb naming the "emit one byte from `buffer[0]`" host action —
/// matches `guests/lambda/src/main.rs`'s own `EMIT_VERB` and
/// `src/dispatch_trampoline.h`'s `PROXIMA_VM_EMIT_VERB`.
#[cfg(target_arch = "aarch64")]
const EMIT_VERB: u64 = 0xfffe;

/// Sentinel verb ending the run loop — matches `main.rs`'s `HALT_VERB` and
/// `src/dispatch_trampoline.h`'s `PROXIMA_VM_HALT_VERB`.
#[cfg(target_arch = "aarch64")]
const HALT_VERB: u64 = 0xffff;

/// Reads one 32-bit register through an ordinary volatile load — the host's
/// data-abort handler decodes a plain `ldr`/`str` from the trapped
/// instruction's own syndrome (`src/backend_macos.c`'s
/// `decode_data_abort_iss`), the same mechanism
/// `guests/lambda/src/virtio_console.rs::mmio_read32` already proves for the
/// virtio windows; no inline assembly is needed for the load itself, only
/// for the hypercall trap below.
///
/// # Safety
///
/// `address` must be a guest-physical address the host's exit loop routes to
/// a real MMIO window (never real backed guest RAM), true only when booted
/// by `dispatch::run_dispatch_loop`.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn mmio_read32(address: u64) -> u32 {
    unsafe { core::ptr::read_volatile(address as *const u32) }
}

/// Issues the existing hypercall ABI's `EMIT_VERB`, emitting exactly
/// `buffer[0]` — mirrors `psci_probe.rs::emit_one_byte` exactly.
///
/// # Safety
///
/// `buffer` must be a live, mapped one-byte guest buffer.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn emit_one_byte(buffer: &mut [u8; 1]) {
    let pointer = buffer.as_mut_ptr() as u64;
    unsafe {
        core::arch::asm!(
            "hvc #0",
            in("x0") EMIT_VERB,
            in("x1") pointer,
            in("x2") 1u64,
            options(nostack),
        );
    }
}

/// Issues `HALT_VERB`, ending the dispatch loop cleanly.
///
/// # Safety
///
/// Must run in an environment where `hvc #0` traps to
/// `dispatch::run_dispatch_loop`'s own exit handler.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn halt() {
    unsafe {
        core::arch::asm!(
            "hvc #0",
            in("x0") HALT_VERB,
            in("x1") 0u64,
            in("x2") 0u64,
            options(nostack),
        );
    }
}

#[unsafe(no_mangle)]
#[unsafe(naked)]
extern "C" fn _start() -> ! {
    #[cfg(target_arch = "aarch64")]
    naked_asm!(
        "adrp x0, __stack_top",
        "add x0, x0, :lo12:__stack_top",
        "mov sp, x0",
        "b {entry}",
        entry = sym entry,
    );
    #[cfg(target_arch = "x86_64")]
    naked_asm!(
        // `call`, not `jmp`: SysV expects rsp % 16 == 8 at function entry, as
        // if just entered via `call` — `__stack_top` itself is 16-aligned.
        "lea rsp, [rip + __stack_top]",
        "call {entry}",
        entry = sym entry,
    );
}

/// aarch64: the real GIC probe body. Other targets have no GIC meaning (see
/// module docs), so `entry` there is just a park — the binary must merely
/// compile and link, not exercise the probe.
#[cfg(target_arch = "aarch64")]
extern "C" fn entry() -> ! {
    let pidr2 = unsafe { mmio_read32(GICD_PIDR2_ADDRESS) };
    for byte in pidr2.to_le_bytes() {
        let mut single = [byte];
        unsafe { emit_one_byte(&mut single) };
    }

    let typer_low = unsafe { mmio_read32(GICR_TYPER_LOW_ADDRESS) };
    for byte in typer_low.to_le_bytes() {
        let mut single = [byte];
        unsafe { emit_one_byte(&mut single) };
    }

    unsafe { halt() };

    loop {
        core::hint::spin_loop();
    }
}

#[cfg(not(target_arch = "aarch64"))]
extern "C" fn entry() -> ! {
    loop {
        core::hint::spin_loop();
    }
}

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {
        core::hint::spin_loop();
    }
}
