//! Per-chipset CPU hint intrinsics. This is the ONE place in the workspace that
//! expresses architecture hints, so a primitive that wants one depends on this.
//!
//! These use only the **stable `core::arch` intrinsics** — no inline `asm!`, no
//! unstable features. We write the intrinsic; the COMPILER emits the instruction
//! for whatever target it is building (the asm is build output, not source we
//! maintain). Every function is a HINT: correctness never depends on it, so a
//! target we cannot express lowers to a no-op — supporting it later is one line
//! here, never a portability gate.
//!
//! Coverage today: x86_64 has a stable prefetch intrinsic (`_mm_prefetch`).
//! aarch64's (`core::arch::aarch64::_prefetch`) is still unstable
//! (`stdarch_aarch64_prefetch`), so aarch64 is a no-op until it stabilises — at
//! which point it becomes one more `#[cfg]` arm here, still no asm.

/// Prefetch the cache line at `ptr` toward L1 for an upcoming WRITE, so a
/// latency-bound store stream (e.g. a batch drain into a caller buffer that
/// exceeds L1) does not stall on read-for-ownership.
///
/// Pure hint: `ptr` is never dereferenced, never faults, and any address is
/// legal. On a target without a stable intrinsic this compiles to nothing.
///
/// - x86_64: `_mm_prefetch::<_MM_HINT_T0>` — the compiler emits `prefetcht0`
///   (baseline SSE); the line lands in L1 and the subsequent store upgrades it.
/// - other targets (incl. aarch64 until its intrinsic stabilises): no-op.
#[inline(always)]
pub fn prefetch_for_write(ptr: *const u8) {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: `_mm_prefetch` is a hint; the operand is an address it never reads.
    unsafe {
        core::arch::x86_64::_mm_prefetch::<{ core::arch::x86_64::_MM_HINT_T0 }>(ptr.cast());
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = ptr;
}

#[cfg(test)]
mod tests {
    use super::prefetch_for_write;

    // the only contract is "never faults, any address legal" — exercise a real
    // buffer, a one-past-the-end pointer, and a dangling address. (On targets
    // where this is a no-op the test still asserts the call is well-formed.)
    #[test]
    fn prefetch_is_a_safe_no_fault_hint() {
        let buffer = [0u8; 64];
        prefetch_for_write(buffer.as_ptr());
        prefetch_for_write(unsafe { buffer.as_ptr().add(buffer.len()) });
        prefetch_for_write(0x1234_usize as *const u8);
    }
}
