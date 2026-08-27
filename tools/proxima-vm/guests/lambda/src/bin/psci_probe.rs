//! Minimal `#![no_std]` guest whose only effect is a PSCI (ARM DEN0022) call
//! sequence: `PSCI_VERSION` via `hvc #0` (routed to `proxima_vm::psci`'s
//! host-side handler because `src/dtb.rs`'s advertised conduit is `hvc`),
//! the returned `x0` value emitted back to the host through the existing
//! `EMIT_VERB` channel one byte at a time (`main.rs`'s own convention), then
//! `SYSTEM_OFF` to end the run — proving the PSCI exit path composes with
//! the existing hvc trap loop's own halt mechanism rather than needing a
//! second exit channel.
//!
//! A deliberately separate `[[bin]]` from `proxima-vm-guest-lambda`
//! (`Cargo.toml`, `src/main.rs`) so this call sequence never perturbs that
//! guest's own pinned ELF layout
//! (`src/elf.rs::matches_readelf_on_the_real_aarch64_guest`) or its pinned
//! emitted-byte contract (`tests/dispatch_hypercall.rs`). aarch64-only:
//! PSCI/HVC is this VM's ARM-specific M5b track, never x86_64's.

#![no_std]
#![no_main]

use core::arch::naked_asm;
use core::panic::PanicInfo;

/// PSCI 0.2 `PSCI_VERSION` — matches `proxima_vm::psci::PSCI_VERSION`.
/// aarch64-only: PSCI/HVC has no x86_64 meaning.
#[cfg(target_arch = "aarch64")]
const PSCI_VERSION: u32 = 0x8400_0000;

/// PSCI 0.2 `SYSTEM_OFF` — matches `proxima_vm::psci::PSCI_SYSTEM_OFF`.
#[cfg(target_arch = "aarch64")]
const PSCI_SYSTEM_OFF: u32 = 0x8400_0008;

/// Sentinel verb naming the "emit one byte from `buffer[0]`" host action —
/// matches `guests/lambda/src/main.rs`'s own `EMIT_VERB` and
/// `src/dispatch_trampoline.h`'s `PROXIMA_VM_EMIT_VERB`.
#[cfg(target_arch = "aarch64")]
const EMIT_VERB: u64 = 0xfffe;

/// Issues `hvc #0` with a full 32-bit SMCCC function ID in `x0` and three
/// arguments in `x1`/`x2`/`x3`, returning the guest's own `x0` register
/// after resume — the PSCI calling convention, distinct from
/// `guests/lambda/src/hypercall.rs::hypercall`'s 16-bit-verb-plus-buffer
/// convention (this binary does not share that module; a `[[bin]]` cannot
/// `mod` a sibling binary's private modules).
///
/// # Safety
///
/// Caller must ensure this is issued from a guest the host's trap loop is
/// actually driving; there is no memory precondition since PSCI carries no
/// buffer pointer.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn psci_call(function_id: u32, arg0: u64, arg1: u64, arg2: u64) -> u64 {
    let result: u64;
    unsafe {
        core::arch::asm!(
            "hvc #0",
            inout("x0") u64::from(function_id) => result,
            in("x1") arg0,
            in("x2") arg1,
            in("x3") arg2,
            options(nostack),
        );
    }
    result
}

/// Issues the existing hypercall ABI's `EMIT_VERB`, emitting exactly
/// `buffer[0]` — mirrors `main.rs`'s `hypercall::hypercall(EMIT_VERB, &mut
/// [byte])` call sites one for one (`backend_macos.c`'s handler only ever
/// reads a single byte from `guest_memory[pointer]` per `EMIT_VERB` call).
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

/// aarch64: the real PSCI probe body. Other targets have no PSCI/HVC meaning
/// (see module docs), so `entry` there is just a park — the binary must
/// merely compile and link, not exercise the probe.
#[cfg(target_arch = "aarch64")]
extern "C" fn entry() -> ! {
    let version = unsafe { psci_call(PSCI_VERSION, 0, 0, 0) };
    let version_bytes = (version as u32).to_le_bytes();
    for byte in version_bytes {
        let mut single = [byte];
        unsafe { emit_one_byte(&mut single) };
    }

    unsafe { psci_call(PSCI_SYSTEM_OFF, 0, 0, 0) };

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
