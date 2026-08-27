//! Minimal `#![no_std]` guest whose only effect is writing the two bytes of
//! "OK" through the real pl011 `UARTDR` register (after polling `UARTFR` for
//! `TXFF` clear, the exact sequence a Linux `pl011` earlycon driver performs)
//! and halting — proving M5b's pl011 slice wiring
//! (`src/dispatch_trampoline.h`'s pl011 window,
//! `src/backend_macos.c`'s `handle_mmio_data_abort` routing,
//! `src/mmio_trampoline.rs`'s `proxima_vm_dispatch_mmio_pl011`) is reached by
//! a real vCPU exit, and that the emitted bytes arrive through the pl011's
//! own dedicated channel (`dispatch::run_dispatch_loop`'s `pl011_emitted`),
//! never the hypercall `EMIT_VERB` channel `gic_probe.rs` uses.
//!
//! A deliberately separate `[[bin]]` from every sibling guest, same
//! isolation reason `gic_probe.rs`'s own module doc gives: this UARTDR write
//! sequence must never perturb another binary's pinned ELF layout or pinned
//! emitted-byte contract. aarch64-only: `entry` on every other target is
//! just a park loop so the binary still compiles and links for the x86_64
//! parity build (`cargo build --target x86_64-unknown-none --bins`).

#![no_std]
#![no_main]

use core::arch::naked_asm;
use core::panic::PanicInfo;

/// Guest-physical address of the pl011 window base — `src/dtb.rs`'s
/// `QemuVirtLayout::CANONICAL.uart_base`, the exact value this VM's own DTB
/// advertises to the guest, matching `PROXIMA_VM_PL011_MMIO_WINDOW_BASE`
/// (`src/dispatch_trampoline.h`).
#[cfg(target_arch = "aarch64")]
const PL011_BASE_ADDRESS: u64 = 0x0900_0000;

/// `UARTFR` offset (`src/pl011.rs`'s `REG_UARTFR`).
#[cfg(target_arch = "aarch64")]
const UARTFR_OFFSET: u64 = 0x018;

/// `UARTDR` offset (`src/pl011.rs`'s `REG_UARTDR`).
#[cfg(target_arch = "aarch64")]
const UARTDR_OFFSET: u64 = 0x000;

/// `UARTFR.TXFF` (bit 5): transmit FIFO full.
#[cfg(target_arch = "aarch64")]
const FR_TXFF: u32 = 1 << 5;

/// Sentinel verb ending the run loop — matches `main.rs`'s `HALT_VERB` and
/// `src/dispatch_trampoline.h`'s `PROXIMA_VM_HALT_VERB`.
#[cfg(target_arch = "aarch64")]
const HALT_VERB: u64 = 0xffff;

/// Ordinary volatile MMIO load/store into the pl011 window — the host's
/// data-abort handler decodes a plain `ldr`/`str` from the trapped
/// instruction's own syndrome (`src/backend_macos.c`'s
/// `decode_data_abort_iss`), the same mechanism `gic_probe.rs::mmio_read32`
/// already proves.
///
/// # Safety
///
/// `offset` must land inside the pl011 window and this code must run under
/// `dispatch::run_dispatch_loop`, which routes it to a real trap instead of
/// backed guest RAM.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn pl011_read32(offset: u64) -> u32 {
    unsafe { core::ptr::read_volatile((PL011_BASE_ADDRESS + offset) as *const u32) }
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn pl011_write32(offset: u64, value: u32) {
    unsafe { core::ptr::write_volatile((PL011_BASE_ADDRESS + offset) as *mut u32, value) };
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

/// aarch64: the real pl011 probe body — poll `UARTFR.TXFF` clear, write one
/// byte to `UARTDR`, repeated for "OK". Other targets have no pl011 meaning
/// (see module docs), so `entry` there is just a park.
#[cfg(target_arch = "aarch64")]
extern "C" fn entry() -> ! {
    for byte in b"OK" {
        loop {
            let flags = unsafe { pl011_read32(UARTFR_OFFSET) };
            if flags & FR_TXFF == 0 {
                break;
            }
        }
        unsafe { pl011_write32(UARTDR_OFFSET, u32::from(*byte)) };
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
