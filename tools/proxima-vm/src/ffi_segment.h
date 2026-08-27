#ifndef PROXIMA_VM_FFI_SEGMENT_H
#define PROXIMA_VM_FFI_SEGMENT_H

#include <stddef.h>
#include <stdint.h>

/* One PT_LOAD segment's mapping request, as the ELF loader (Rust, tier-3)
 * hands it to the host-memory driver leaf (C, tier-2). `readable` /
 * `writable` / `executable` are the gABI "Exact" interpretation of
 * `p_flags` -- never widened. */
typedef struct {
    uint64_t guest_address;
    const uint8_t *data;
    size_t data_length;
    uint64_t memory_size;
    uint8_t readable;
    uint8_t writable;
    uint8_t executable;
} proxima_vm_segment_t;

/* One segment's host-side mapping, filled in by
 * `proxima_vm_map_guest_memory` on success and consumed by
 * `proxima_vm_unmap_guest_memory` to unwind it. */
typedef struct {
    uint64_t guest_address;
    void *host_address;
    size_t mapped_size;
} proxima_vm_mapped_segment_t;

/* M4 — guest memory as a named object (`tools/proxima-vm/ROADMAP.md`'s M4
 * section): a memory region backed by a kernel/host object with an
 * identity beyond one process's virtual-address mapping of it, so a second
 * caller can map the SAME object and observe the first caller's writes --
 * the one thing `mmap(MAP_ANON)` cannot do, because an anonymous mapping
 * has no name a second mapper could ask for. `handle` is a `memfd_create`
 * file descriptor on the KVM lane (`backend_linux.c`), a mach memory-entry
 * port name on the HVF lane (`backend_macos.c`) -- opaque to the caller
 * either way, passed back into `proxima_vm_map_named_region` to create a
 * second, independent view of the same backing object. `primary_address`
 * is the first view, created and zero-filled by
 * `proxima_vm_create_named_region` itself. */
typedef struct {
    int handle;
    void *primary_address;
    size_t mapped_size;
} proxima_vm_named_region_t;

/* M7 — snapshot (`tools/proxima-vm/ROADMAP.md`'s M7 section): one vCPU's
 * general-register file, captured by `proxima_vm_scratch_snapshot` and
 * replayed by `proxima_vm_scratch_restore`. Field order and width must
 * match the Rust-side `RawVcpuRegisters` mirror exactly (`src/snapshot.rs`).
 *
 * `gpr[0..31)` is `x0..x30` on the HVF/aarch64 lane
 * (`hv_vcpu_get_reg(vcpu, HV_REG_X0 + i, ...)`, the same `X0 + offset`
 * indexing `backend_macos.c`'s own mmio-transfer-register decode already
 * uses) and `rax, rbx, rcx, rdx, rsi, rdi, rbp, rsp, r8..r15` (in that
 * `struct kvm_regs` field order, zero-padded to 31 slots) on the KVM/x86_64
 * lane -- the two architectures' register files do not correspond 1:1, so
 * this struct carries whichever one the running lane captured; a snapshot
 * is never portable across lanes, only within one. `pc` is `pc` (aarch64)
 * or `rip` (x86_64); `flags` is `cpsr` (aarch64) or `rflags` (x86_64). */
typedef struct {
    uint64_t gpr[31];
    uint64_t pc;
    uint64_t flags;
} proxima_vm_registers_t;

#endif
