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

#endif
