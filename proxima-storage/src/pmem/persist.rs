//! Persistence ordering primitives over a borrowed region.
//!
//! [`flush`] queues cache-line writebacks for a range; [`drain`] is the store
//! ordering fence; [`persist`] is the two as a unit — PMDK's `pmem_persist`.
//! The store discipline is: write, `persist`, write the commit point, `persist`
//! again. The fence between the two `persist` calls is what makes a crash see
//! old-or-new rather than a torn mix.
//!
//! Real implementations exist for x86_64 (`clflush` + `sfence`; `clwb` /
//! `clflushopt` are a faster, std-facade runtime-detected optimization) and
//! aarch64-linux (DC CVAC + DSB). Every other target — aarch64 macOS as a dev host, bare-metal
//! ARMv7-M, wasm — uses a documented **no-op fallback**. That is sound for
//! correctness work: the crash-consistency FSM is proven by the software
//! reordering oracle (see `cow`'s tests), which models persistence in pure Rust;
//! real cache maintenance only matters on a real pmem device, which is the
//! out-of-scope I/O facade's concern.
//!
//! This is the irreducible hardware boundary — the pmem analog of a NIC/NVMe
//! doorbell write. It is pure Rust with zero C linked.

/// Cache line size assumed for flush granularity. 64 bytes on x86_64 and on the
/// aarch64 cores this targets.
pub const CACHE_LINE: usize = 64;

#[cfg(target_arch = "x86_64")]
#[inline]
fn flush_line(line: *const u8) {
    // SAFETY: `line` is derived from a live borrowed slice; clflush of a mapped
    // address is a durability hint with no memory-safety effect. `clflush` is
    // SSE2-baseline, always available on x86_64. `clwb`/`clflushopt` (which avoid
    // evicting / re-fetching the line) are faster but are not cfg-queryable target
    // features and need runtime detection (`is_x86_feature_detected!`, std-only) —
    // so they belong in a std facade, not this no_std leaf.
    unsafe { core::arch::x86_64::_mm_clflush(line) }
}

/// Store fence: prior flushed stores become durable before any later store.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn drain() {
    // SAFETY: sfence has no memory-safety precondition; it only orders stores.
    unsafe { core::arch::x86_64::_mm_sfence() }
}

#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
#[inline]
fn flush_line(line: *const u8) {
    // SAFETY: DC CVAC cleans one cache line by virtual address; it is permitted
    // from EL0 on Linux aarch64 (SCTLR_EL1.UCI) and has no memory-safety effect.
    // CVAC cleans to the point of coherency; upgrading to CVAP (point of
    // persistence) is a real-hardware tuning step, gated on pmem silicon we
    // don't have — CVAC is correct but conservative until then.
    unsafe {
        core::arch::asm!("dc cvac, {addr}", addr = in(reg) line, options(nostack, preserves_flags));
    }
}

/// Store fence: prior cache-clean operations complete before any later store.
#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
#[inline]
pub fn drain() {
    // SAFETY: DSB is a barrier with no memory-safety precondition.
    unsafe {
        core::arch::asm!("dsb sy", options(nostack, preserves_flags));
    }
}

#[cfg(not(any(
    target_arch = "x86_64",
    all(target_arch = "aarch64", target_os = "linux")
)))]
#[inline]
fn flush_line(_line: *const u8) {}

/// No-op drain on targets without a real cache-maintenance path (dev hosts,
/// bare-metal). See the module docs: correctness is proven by the software
/// oracle, not by this fence.
#[cfg(not(any(
    target_arch = "x86_64",
    all(target_arch = "aarch64", target_os = "linux")
)))]
#[inline]
pub fn drain() {}

/// Queue cache-line writebacks for every line the `region` slice touches. Pairs
/// with [`drain`] to make those writebacks durable. On its own it guarantees
/// nothing — only `drain` (or [`persist`]) establishes the ordering point.
#[inline]
pub fn flush(region: &[u8]) {
    let mut offset = 0;
    while offset < region.len() {
        // SAFETY: offset < region.len(), so the pointer is in-bounds. flush_line
        // flushes the whole cache line containing the address, so stepping by
        // CACHE_LINE from the region start covers every byte's line.
        let line = unsafe { region.as_ptr().add(offset) };
        flush_line(line);
        offset += CACHE_LINE;
    }
}

/// Flush `region` and drain — make the region's stores durable as a unit. The
/// PMDK `pmem_persist` shape.
#[inline]
pub fn persist(region: &[u8]) {
    flush(region);
    drain();
}
