//! Trap-instruction wrappers for the guest side of the capability channel
//! (`tools/proxima-vm/ROADMAP.md` M1): `hvc #0` on aarch64, `out dx, al` on
//! x86_64. Both instructions cause a synchronous exit into the host, which
//! reads the full guest register file to recover the call — the same shape
//! `src/backend_macos.c` already uses (`hv_vcpu_get_reg(vcpu, HV_REG_X0,
//! ..)` after an `HVC` exception) and `src/backend_linux.c` already uses
//! (`run->io.data_offset` after a `KVM_EXIT_IO`). `dx` carries the verb
//! itself rather than a fixed immediate port, because the immediate `out
//! imm8, al` form the scratch guest uses today caps the call space at 256;
//! `out dx, al` widens that to 16 bits, matching the ARM side's `x0`.
//!
//! `main.rs`'s `entry` calls the arch-matching trap once, on a buffer holding
//! the already-pinned `ChildRequest::Read` postcard bytes (fd-keyed post-P0,
//! `tools/proxima-vm/ROADMAP.md` P0). Mapping that buffer into an actual
//! shared page, and a host that reads the exit and replies, are later steps
//! — no VM backend calls into this guest yet.

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
use core::arch::asm;

/// Issues `hvc #0`, placing `verb` in `x0`, the buffer's address in `x1`, and
/// its length in `x2` — the register slots `src/backend_macos.c`'s exit
/// handler reads after an `HV_EXIT_REASON_EXCEPTION` with exception class
/// `0x16` (HVC from AArch64 state).
///
/// # Safety
///
/// The caller must ensure `buffer` is mapped into the region the host's
/// loader placed as the guest's shared page, so the address placed in `x1`
/// is one the host can actually read or write.
#[cfg(target_arch = "aarch64")]
#[inline]
pub unsafe fn hypercall(verb: u16, buffer: &mut [u8]) -> u64 {
    let pointer = buffer.as_mut_ptr() as u64;
    let length = buffer.len() as u64;
    let result: u64;
    unsafe {
        asm!(
            "hvc #0",
            inout("x0") u64::from(verb) => result,
            in("x1") pointer,
            in("x2") length,
            options(nostack),
        );
    }
    result
}

/// Issues `out dx, al`, placing `verb` in `dx`, the buffer's address in
/// `rdi`, and its length in `rsi` — the port write `src/backend_linux.c`'s
/// exit handler reads via `KVM_EXIT_IO` today, widened from a fixed
/// immediate port to a caller-supplied one. `al` carries no payload; the
/// trap itself, not its operand, is what the host observes.
///
/// # Safety
///
/// The caller must ensure `buffer` is mapped into the region the host's
/// loader placed as the guest's shared page, so the address placed in `rdi`
/// is one the host can actually read or write.
#[cfg(target_arch = "x86_64")]
#[inline]
pub unsafe fn hypercall(verb: u16, buffer: &mut [u8]) -> u64 {
    let pointer = buffer.as_mut_ptr() as u64;
    let length = buffer.len() as u64;
    let result: u64;
    unsafe {
        asm!(
            "out dx, al",
            in("dx") verb,
            in("al") 0u8,
            in("rdi") pointer,
            in("rsi") length,
            lateout("rax") result,
            options(nostack),
        );
    }
    result
}
