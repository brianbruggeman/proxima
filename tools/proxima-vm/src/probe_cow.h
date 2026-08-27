#ifndef PROXIMA_VM_PROBE_COW_H
#define PROXIMA_VM_PROBE_COW_H

#include <stddef.h>
#include <stdint.h>

/* micro-second campaign slice 3 (`tools/proxima-vm/ROADMAP.md`) -- probe-only
 * FFI leaves, never called by `WarmVm`/`restore`/`proxima_vm_run_dispatch_loop`
 * itself. Every function here exists to MEASURE a candidate no-copy warm-
 * restore mechanism before any design is chosen (this slice's own mandate:
 * "measurement first; do NOT redesign WarmVm this slice") -- driven only by
 * `src/bin/cow_primitives_probe.rs`.
 *
 * `proxima_vm_probe_vm_create` must be called exactly once per process
 * before any other function here (`hv_vm_create` is once-per-process,
 * `src/snapshot.rs`'s own module doc on the same restriction). */

/* One `hv_vm_create` per process, mirroring `create_vm` inside
 * `backend_macos.c` -- exported so the probe binary owns exactly one call
 * across every candidate section it runs. */
int proxima_vm_probe_vm_create(char *error_buffer, size_t error_capacity);

/* Creates a `size`-byte named mach memory entry (`MAP_MEM_NAMED_CREATE`,
 * the same primitive `proxima_vm_create_named_region` already uses) and
 * `memset`s it to `0xAB` once, simulating a resident snapshot's already-
 * faulted-in pages -- the CoW/vm_copy candidates below source their views
 * from this region, never from a freshly zero-filled one, so the measured
 * remap/copy cost is never conflated with a first-fault cost this source
 * itself would otherwise absorb silently. */
int proxima_vm_probe_create_source(
    size_t size,
    void **host_address_out,
    int *handle_out,
    char *error_buffer,
    size_t error_capacity
);

void proxima_vm_probe_destroy_source(void *host_address, int handle, size_t size);

/* Candidate 1 -- fresh CoW view per restore via `mach_vm_remap(..., copy=TRUE)`.
 * `previous_view_inout` is in/out: on entry, the prior iteration's guest-
 * mapped view host address (or `NULL` on the very first call); on exit, the
 * freshly created view's host address, ready to be passed back in on the
 * next call. Guest IPA is always `0`, `size` bytes -- the previous view MUST
 * be `hv_vm_unmap`ped before the new one is `hv_vm_map`ped into the same IPA
 * range, so `hv_vm_unmap_old_nanos` is measured BEFORE `hv_vm_map_nanos`,
 * not after, despite the trio's naming order in the task brief. */
int proxima_vm_probe_cow_view_trio(
    void *source_host_address,
    size_t size,
    void **previous_view_inout,
    uint64_t *remap_nanos_out,
    uint64_t *hv_vm_unmap_old_nanos_out,
    uint64_t *hv_vm_map_nanos_out,
    uint64_t *dealloc_old_nanos_out,
    char *error_buffer,
    size_t error_capacity
);

/* Touches one byte per page over `page_count` pages of `view_address`
 * (already `hv_vm_map`ped or plain host-mapped, either works since this is
 * a pure host-side write), timing the whole loop -- CoW/vm_copy fault
 * amortization, `page_count` pages at `page_size` stride. */
int proxima_vm_probe_first_touch(
    void *view_address,
    size_t page_size,
    size_t page_count,
    uint64_t *nanos_out
);

/* Candidate 2 -- `mach_make_memory_entry_64` with `MAP_MEM_VM_COPY` sourced
 * from `source_host_address`. `kern_return_out` always carries the verbatim
 * `mach_make_memory_entry_64` status; a nonzero return from this function
 * with `*kern_return_out != KERN_SUCCESS` is the rejected-primitive finding
 * the task brief asks to record verbatim, not a probe-harness bug. Same
 * in/out `previous_view_inout` and unmap-before-map ordering as
 * `proxima_vm_probe_cow_view_trio`. */
int proxima_vm_probe_vm_copy_trio(
    void *source_host_address,
    size_t size,
    void **previous_view_inout,
    int *kern_return_out,
    uint64_t *entry_create_nanos_out,
    uint64_t *map_nanos_out,
    uint64_t *hv_vm_unmap_old_nanos_out,
    uint64_t *hv_vm_map_nanos_out,
    uint64_t *dealloc_old_nanos_out,
    char *error_buffer,
    size_t error_capacity
);

/* Candidate 3a -- `hv_vm_protect` cost for one call covering the whole
 * `[guest_address, guest_address + size)` range already `hv_vm_map`ped at
 * `guest_address`. Alternates between `HV_MEMORY_READ` and
 * `HV_MEMORY_READ | HV_MEMORY_WRITE` each call so repeated calls never
 * become a no-op the kernel could short-circuit. */
int proxima_vm_probe_protect_whole(
    uint64_t guest_address,
    size_t size,
    int want_read_only,
    uint64_t *nanos_out,
    char *error_buffer,
    size_t error_capacity
);

/* Candidate 3b -- `page_count` individual `hv_vm_protect` calls, each
 * covering one `granule`-byte slice of `[guest_address, ...)`, individual
 * per-call nanosecond timings written into `nanos_out[0..page_count)`. */
int proxima_vm_probe_protect_per_page(
    uint64_t guest_address,
    size_t granule,
    size_t page_count,
    uint64_t *nanos_out,
    char *error_buffer,
    size_t error_capacity
);

/* Candidate 3c -- the load-bearing empirical check: does a guest write to
 * an `hv_vm_protect`ed read-only page exit to the host (permission fault,
 * ARM exception class `0x24`, data abort) rather than something else (a
 * silent guest-internal fault, a hang, a different exception class)? Runs
 * its own tiny two-checkpoint guest program (a code page at IPA
 * `code_guest_address` and a data page at IPA `data_guest_address`, both
 * `size` bytes, both freshly `hv_vm_map`ped and unmapped inside this one
 * call): the guest writes `0x2A` into the data page, traps via `hvc`
 * (checkpoint 1, `x0 == 1`), the host `hv_vm_protect`s the data page
 * read-only (timed into `*protect_nanos_out`), resumes the vcpu, and the
 * guest attempts a second write of `0x55` into the same page -- checkpoint
 * 2 is whatever exit `hv_vcpu_run` reports for THAT attempt, decoded into
 * `exception_class_out`/`is_data_abort_out`/`is_write_out`, never assumed. */
int proxima_vm_probe_write_protect_exit(
    uint64_t *checkpoint1_x0_out,
    uint64_t *exception_class_out,
    int *is_data_abort_out,
    int *is_write_out,
    uint8_t *data_byte_after_out,
    uint64_t *protect_nanos_out,
    char *error_buffer,
    size_t error_capacity
);

#endif
