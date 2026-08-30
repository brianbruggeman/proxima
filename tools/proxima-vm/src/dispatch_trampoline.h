#ifndef PROXIMA_VM_DISPATCH_TRAMPOLINE_H
#define PROXIMA_VM_DISPATCH_TRAMPOLINE_H

#include <stddef.h>
#include <stdint.h>

#include "ffi_segment.h"

/* Rust-side hypercall dispatcher (`src/dispatch.rs`): decodes the payload
 * at guest_memory[pointer..pointer+length) as a postcard-encoded
 * `ChildRequest`, drives it through the dispatcher's `SendPipe::call` via
 * `futures::executor::block_on` (the same pattern `hello.rs:15` already
 * uses), and postcard-encodes the resulting `ChildResponse` into
 * `response_out`. Returns the encoded response length on success, or a
 * negative sentinel: -1 payload pointer/length out of range, -2 postcard
 * decode of the payload failed, -3 the dispatcher call failed, -4 postcard
 * encode of the response failed, -5 the encoded response exceeds
 * `response_capacity`. */
int64_t proxima_vm_dispatch_hypercall(
    const void *dispatcher,
    const uint8_t *guest_memory,
    size_t guest_memory_length,
    uint16_t verb,
    uint64_t pointer,
    uint64_t length,
    uint8_t *response_out,
    size_t response_capacity
);

/* Verb sentinels the guest side (`guests/lambda/src/main.rs`) and the
 * platform run loops (`backend_macos.c`'s and `backend_linux.c`'s
 * `proxima_vm_run_dispatch_loop`) both key off, chosen outside the
 * `ChildRequest` discriminant range (0..=4,
 * `proxima-protocols/src/process/protocol.rs:72-77`) so a verb's value
 * alone tells the loop whether to route it to
 * `proxima_vm_dispatch_hypercall` or handle it locally. */
#define PROXIMA_VM_EMIT_VERB 0xfffeu
#define PROXIMA_VM_HALT_VERB 0xffffu

/* Maps every entry in `segments` into a single flat `guest_memory_size`-byte
 * guest address space starting at 0 — each at its own `guest_address`, its
 * own real `readable`/`writable`/`executable` permissions (never one RWX
 * blob; W^X per `proxima_vm_segment_t`'s own doc in `ffi_segment.h`), copying
 * `data[0..data_length)` in at that offset first (a zero-`data_length`
 * segment, e.g. a stack reservation, copies nothing) — then runs a real
 * vCPU: sets `PC = entry`, traps every `hvc #0` / `out dx, al`, and either
 * emits one byte from `guest_memory[pointer]` (`verb ==
 * PROXIMA_VM_EMIT_VERB`), ends the loop (`verb == PROXIMA_VM_HALT_VERB`),
 * or forwards the exit to `proxima_vm_dispatch_hypercall` and copies its
 * response back into `guest_memory[pointer..]` before resuming — the real
 * hypervisor-exit dispatch path `tools/proxima-vm/ROADMAP.md`'s M1 section
 * names, as opposed to `dispatch::dispatch_hypercall_bytes` driven directly
 * against a synthetic buffer with no VM in the loop.
 *
 * Returns 0 on a clean `PROXIMA_VM_HALT_VERB` exit, -1 on any mapping,
 * hypervisor, dispatch, or budget-exceeded failure (message in
 * `error_buffer`). Caps the exit count at `max_hypercalls` so a guest that
 * never halts cannot loop the host forever. */
int proxima_vm_run_dispatch_loop(
    const proxima_vm_segment_t *segments,
    size_t segment_count,
    uint64_t guest_memory_size,
    uint64_t entry,
    const void *dispatcher,
    size_t max_hypercalls,
    uint8_t *emitted_out,
    size_t emitted_capacity,
    size_t *emitted_length_out,
    char *error_buffer,
    size_t error_capacity
);

int proxima_vm_create_named_region(
    size_t size,
    proxima_vm_named_region_t *region_out,
    char *error_buffer,
    size_t error_capacity
);

int proxima_vm_map_named_region(
    const proxima_vm_named_region_t *region,
    int want_private_view,
    void **host_address_out,
    char *error_buffer,
    size_t error_capacity
);

void proxima_vm_unmap_named_region_view(void *host_address, size_t mapped_size);

void proxima_vm_destroy_named_region(proxima_vm_named_region_t *region);

/* M7 — snapshot (`tools/proxima-vm/ROADMAP.md`'s M7 section): serialize
 * vCPU registers, device state, and memory; restore into a fresh VM. Built
 * on the scratch-guest path (`proxima_vm_scratch_run`'s own guest, the
 * substrate's simplest fully proven boot shape) because M7's exit criterion
 * is restore wall time and fault count, not a spec-complete device model --
 * device-state serialization is already proven separately by
 * `ConsoleTransport`/`NetTransport`/`BlkTransport` now deriving `Clone`
 * (`src/virtio_console.rs`, `src/virtio_net.rs`, `src/virtio_blk.rs`), the
 * direct payoff M6's `MmioDevice`/`RingCursor` state-machine shape promised.
 *
 * `proxima_vm_scratch_guest_memory_size` returns the exact page-rounded byte
 * count `proxima_vm_scratch_snapshot` will map for `message_length` -- pure,
 * no side effect, callers use it to size `guest_memory_out` before the call.
 *
 * `proxima_vm_scratch_snapshot` runs `message` through the scratch guest to
 * its halting `hvc`/`out` trap (same guest program `proxima_vm_scratch_run`
 * synthesizes) over an M4 named guest-memory region
 * (`proxima_vm_create_named_region`), then -- before tearing the region and
 * vCPU down -- copies the general-register file into `registers_out` and the
 * region's full byte contents into `guest_memory_out`. Returns 0 on success,
 * -1 on failure (message in `error_buffer`). */
size_t proxima_vm_scratch_guest_memory_size(size_t message_length);

int proxima_vm_scratch_snapshot(
    const uint8_t *message,
    size_t message_length,
    uint8_t *output,
    size_t output_capacity,
    proxima_vm_registers_t *registers_out,
    uint8_t *guest_memory_out,
    size_t guest_memory_capacity,
    char *error_buffer,
    size_t error_capacity
);

/* Restores `registers_in`/`guest_memory_in` into a brand-new M4 named
 * region, a brand-new `hv_vm`/vCPU (HVF) or `KVM_CREATE_VM`/vCPU (KVM) --
 * never the region or vCPU `proxima_vm_scratch_snapshot` used -- then resumes
 * it. `hv_vm_create` is once-per-process on the HVF lane (empirically: a
 * second call in a process that already destroyed one `hv_vm` hangs rather
 * than erroring), so a caller driving both functions must do so from two
 * separate processes on that lane -- `src/snapshot.rs`'s own module doc and
 * `src/bin/snapshot_capture_probe.rs` / `src/bin/snapshot_restore_probe.rs`
 * are the two halves `tests/vm_snapshot.rs` drives as child processes.
 * the vCPU exactly once. Because the snapshot was captured at the guest's
 * halting trap (whose faulting instruction has not yet been retired), that
 * one resumed step re-traps at the identical instruction: `*resumed_ok_out`
 * is nonzero, and `*resumed_x0_out` reads back the identical register value
 * `proxima_vm_scratch_snapshot` captured, iff the restore reproduced the
 * exact guest state -- the restore-is-proven evidence M7's exit criterion
 * asks for, not a claim.
 *
 * `page_size` strides both the copy-in of `guest_memory_in` (timed as
 * `*touch_all_pages_nanos_out`, mirroring `proxima_vm_run_dispatch_loop`'s
 * own M3 first-touch loop but over restored bytes instead of zeros) and the
 * fault-count axis: `*fault_count_out` counts data-abort/`KVM_EXIT_MMIO`
 * exits observed during the one resumed step (legitimately 0 for this guest,
 * which touches no mmio window -- the same "auxiliary count, not a stage-2
 * RAM-fault index" caveat `proxima_vm_run_dispatch_loop`'s own
 * `mmio_trap_count_out` already carries, since HVF has no stage-2 fault
 * index to report instead). `*restore_wall_nanos_out` covers the whole
 * restore: region creation, the page-strided copy, vCPU creation, and
 * register restoration -- everything `proxima_vm_scratch_snapshot`'s output
 * must pass back through before the guest can run again.
 *
 * The µsec-campaign per-phase breakdown of that same `*restore_wall_nanos_out`
 * total (M7 follow-through, "restore-path decomposition"):
 * `*region_create_nanos_out` (`proxima_vm_create_named_region`),
 * `*vm_create_nanos_out` (`hv_vm_create`/`KVM_CREATE_VM`), `*vm_map_nanos_out`
 * (`hv_vm_map`/`KVM_SET_USER_MEMORY_REGION`), `*vcpu_create_nanos_out`
 * (`hv_vcpu_create`/`KVM_CREATE_VCPU`), `*register_restore_nanos_out`
 * (`restore_registers`), and `*first_retrap_nanos_out` (the one resumed
 * `hv_vcpu_run`/`KVM_RUN` step). These six plus `*touch_all_pages_nanos_out`
 * sum to (approximately -- each phase's own `now_nanos()` call pair carries
 * a few dozen ns of overhead not attributed to any phase) `*restore_wall_nanos_out`
 * minus `*first_retrap_nanos_out` (`*restore_wall_nanos_out` is stamped
 * before the resumed step, matching its own doc above). Returns 0 on
 * success, -1 on failure (message in `error_buffer`). */
int proxima_vm_scratch_restore(
    const proxima_vm_registers_t *registers_in,
    const uint8_t *guest_memory_in,
    size_t guest_memory_length,
    size_t page_size,
    uint64_t *restore_wall_nanos_out,
    uint64_t *touch_all_pages_nanos_out,
    uint64_t *fault_count_out,
    uint64_t *resumed_x0_out,
    int *resumed_ok_out,
    uint64_t *region_create_nanos_out,
    uint64_t *vm_create_nanos_out,
    uint64_t *vm_map_nanos_out,
    uint64_t *vcpu_create_nanos_out,
    uint64_t *register_restore_nanos_out,
    uint64_t *first_retrap_nanos_out,
    char *error_buffer,
    size_t error_capacity
);

/* Warm-restore trio (µsec campaign, first slice): the same restore
 * `proxima_vm_scratch_restore` performs, minus every "create a new thing"
 * phase that dominates it (region/vm/vcpu creation, per M7's own measured
 * evidence `hv_vm_create`+`hv_vcpu_create`+region-create are >99.7% of a
 * cold restore's wall time, not the memory copy) -- by never creating a
 * second time at all. `hv_vm_create` hangs on a second call in one process
 * (`proxima_vm_scratch_restore`'s own doc above); warm restore's whole
 * point is to make that call exactly once and then reuse the same vm/vcpu/
 * mapped region for every subsequent restore.
 *
 * `proxima_vm_scratch_warm_vm_create` creates the named region (sized for
 * `guest_memory_capacity`, the largest snapshot any later warm-restore call
 * against this context will pass), maps it into a freshly created vm, and
 * creates the one vCPU -- once. `*context_out` is an opaque, fixed-size,
 * caller-owned handle (no heap allocation on either side of the FFI
 * boundary) a caller passes to every subsequent warm-restore call and, at
 * the end of its lifetime, to `proxima_vm_scratch_warm_vm_destroy`. Returns
 * 0 on success, -1 on failure (message in `error_buffer`).
 *
 * `proxima_vm_scratch_warm_restore` resets the live vCPU's registers from
 * `registers_in`, copies `guest_memory_in` directly into the already-mapped
 * region (`memcpy`, not unmap/remap -- the region is host-addressable from
 * the first `warm_vm_create` call onward), and resumes the vCPU exactly
 * once, proving restore correctness the same way the cold path does: a
 * matching re-trap. Callable any number of times against the same context;
 * `guest_memory_length` must not exceed the `guest_memory_capacity` the
 * context was created with. Per-phase output params mirror the cold path's
 * new ones minus the three creation phases that no longer exist in this
 * path. Returns 0 on success, -1 on failure (message in `error_buffer`).
 *
 * `proxima_vm_scratch_warm_vm_destroy` tears the context down (vCPU, vm,
 * named region) -- call at most once, after the last warm-restore call. */
typedef struct {
    proxima_vm_named_region_t region;
    size_t guest_memory_capacity;
    uint64_t vcpu;
    void *exit_data;
} proxima_vm_warm_restore_context_t;

int proxima_vm_scratch_warm_vm_create(
    size_t guest_memory_capacity,
    proxima_vm_warm_restore_context_t *context_out,
    char *error_buffer,
    size_t error_capacity
);

int proxima_vm_scratch_warm_restore(
    proxima_vm_warm_restore_context_t *context,
    const proxima_vm_registers_t *registers_in,
    const uint8_t *guest_memory_in,
    size_t guest_memory_length,
    size_t page_size,
    uint64_t *restore_wall_nanos_out,
    uint64_t *touch_all_pages_nanos_out,
    uint64_t *fault_count_out,
    uint64_t *resumed_x0_out,
    int *resumed_ok_out,
    uint64_t *register_restore_nanos_out,
    uint64_t *first_retrap_nanos_out,
    char *error_buffer,
    size_t error_capacity
);

void proxima_vm_scratch_warm_vm_destroy(proxima_vm_warm_restore_context_t *context);

/* Layered base+delta warm restore (µsec campaign, layered rework) --
 * DELETES the rejected monolithic dirty-copyback design that used to sit
 * here (`proxima_vm_scratch_warm_vm_arm_dirty_tracking` /
 * `proxima_vm_scratch_warm_dirty_write_run` / `proxima_vm_scratch_warm_restore_dirty`:
 * one region, `hv_vm_protect` permission flips in place, restore re-copied
 * dirty pages FROM a full-size snapshot buffer back INTO that one region --
 * the "monolithic block" the owner ruled invalid). The owner's own design
 * (slot-0 vault, 2026-08-11 execution-boundary note): "Map the image
 * read-only shared with 1GB pages. Map the writable delta with 4K pages.
 * Fault count then tracks the writable working set, not the image size...
 * snapshot stops being an operation and restore becomes a mapping, not a
 * copy." This host's HVF stage-2 granule is a fixed 16KiB
 * (`proxima_vm_host_page_size`, measured, M3) -- there is no large-page/4K
 * split to make on this lane, so BOTH layers below use the host granule; the
 * LAYERING (one read-only base region, one per-VM read-write delta region,
 * page-granular remapping between them) is the portable invariant, not the
 * page-size split.
 *
 * BASE and DELTA are both ordinary caller-owned host memory (a
 * `proxima_vm::named_memory::GuestMemoryRegion` or `RegionView` on the Rust
 * side, `src/named_memory.rs` -- the M4 named-region primitive this design
 * reuses rather than reinventing) -- this header's own functions never
 * allocate either region, only map/unmap/remap them into one guest IPA
 * range and copy exactly one granule at a time between them. Because BASE is
 * an ordinary named-memory object, TWO callers can `map_shared_view` the
 * SAME base region and construct two independent `proxima_vm_layered_context_t`s
 * over it (each with its own vCPU and its own private delta, at disjoint
 * `ipa_base` ranges since HVF's stage-2 IPA space is one flat space per
 * process-wide `hv_vm`) -- this is the "two VMs map the same region" case
 * M4's own exit criterion named and deferred to this milestone. */
size_t proxima_vm_host_page_size(void);

typedef struct {
    void *base_host_address;
    size_t base_size;
    void *delta_host_address;
    size_t delta_size;
    uint64_t ipa_base;
    uint64_t vcpu;
    void *exit_data;
    int mapped;
} proxima_vm_layered_context_t;

/* Creates one vCPU (`create_and_start_vcpu(ipa_base, ...)`, reused verbatim
 * -- this design's guest code always starts at word 0 of its own base
 * region, i.e. guest address `ipa_base`) inside this process's one, shared,
 * lazily-created `hv_vm` (`create_vm` is idempotent -- see `backend_macos.c`
 * -- specifically so a second `proxima_vm_layered_vcpu_create` call in the
 * same process, the sharing proof's own shape, does not hit the documented
 * once-per-process `hv_vm_create` hang). Does not map anything into guest
 * IPA yet -- `proxima_vm_layered_adopt_base` below is the mapping step, and
 * requires `base_host_address` to already hold the base's final bytes
 * (a plain host-side `memcpy`/`copy_from_slice` the Rust caller performs
 * directly against its own `GuestMemoryRegion`/`RegionView`, before this
 * call -- never inside this FFI boundary). Returns 0 on success, -1 on
 * failure (message in `error_buffer`). */
int proxima_vm_layered_vcpu_create(
    void *base_host_address,
    size_t base_size,
    void *delta_host_address,
    size_t delta_size,
    uint64_t ipa_base,
    proxima_vm_layered_context_t *context_out,
    char *error_buffer,
    size_t error_capacity
);

/* Maps `[ipa_base, ipa_base + base_size)` read-only+exec from `base_host_address`
 * -- unmapping first if a prior adopt/run/restore cycle left any of that
 * range delta-backed. Clears `dirty_bitmap` and resets every gpr/`pc`
 * (`ipa_base`)/`cpsr` to the same fixed values every dirty-write run
 * resumes from. This is the one mapping-only "establish the base" step; it
 * never touches a guest-memory byte itself (the caller already wrote them
 * into `base_host_address`). Returns 0 on success, -1 on failure (message
 * in `error_buffer`). */
int proxima_vm_layered_adopt_base(
    proxima_vm_layered_context_t *context,
    uint8_t *dirty_bitmap,
    size_t dirty_bitmap_capacity,
    uint64_t *map_nanos_out,
    uint64_t *register_reset_nanos_out,
    char *error_buffer,
    size_t error_capacity
);

/* Resets every gpr/`pc`(`ipa_base`)/`cpsr`, then resumes the vCPU
 * repeatedly, running whatever guest code is already present in the base
 * region (`src/snapshot.rs`'s `dirty_probe_snapshot` builds that code as
 * pure data -- this function never synthesizes guest instructions itself,
 * the same non-negotiable constraint the deleted design's own doc already
 * named: a host-side code write would never trip the EC-0x24 fault path this
 * design tracks dirty pages through). Every EC-0x24 write-fault whose fault
 * address falls in `[ipa_base, ipa_base + base_size)` copies exactly ONE
 * granule from `base_host_address` to the identical offset in
 * `delta_host_address` (never the whole region), `hv_vm_unmap`s that one
 * guest-IPA page and `hv_vm_map`s it back from `delta_host_address` read
 * -write, marks it in `dirty_bitmap`, and resumes WITHOUT advancing `pc` --
 * the faulting store retries and now succeeds against the freshly writable
 * delta page (slice 3's own `3c` mechanism: a write fault never advances
 * past the trapping instruction on its own). A page already marked dirty
 * from an earlier call is already delta-mapped read-write and does not
 * fault again. `expected_page_count` only bounds the fault-count budget
 * (`expected_page_count + 64`) against a runaway guest. Returns 0 on
 * success, -1 on failure (message in `error_buffer`). */
/* `dirty_page_indices`/`dirty_page_index_count` are the O(working-set) twin
 * of `dirty_bitmap`: a page only ever transitions bitmap-clear -> bitmap-set
 * once per adopt/restore cycle (it is immediately remapped read-write, so a
 * retried store never re-faults -- `backend_macos.c`'s own dedup check on
 * this same bitmap is what makes that true), so appending its index the one
 * time that happens is a plain O(1) push, never a scan. `*dirty_page_index_count`
 * is IN/OUT and persists across repeated `proxima_vm_layered_run` calls
 * against the same context (accumulating, never reset by this function) --
 * only `proxima_vm_layered_restore` below resets it to 0. Capacity is the
 * caller's responsibility (sized to `base_size / granule`, the same worst
 * case `dirty_bitmap_capacity` already assumes); exceeding it is a genuine
 * caller bug and fails loudly rather than silently truncating the list. */
int proxima_vm_layered_run(
    proxima_vm_layered_context_t *context,
    uint64_t expected_page_count,
    uint8_t *dirty_bitmap,
    size_t dirty_bitmap_capacity,
    uint32_t *dirty_page_indices,
    size_t dirty_page_indices_capacity,
    uint64_t *dirty_page_index_count,
    uint64_t *run_wall_nanos_out,
    uint64_t *fault_count_out,
    uint64_t *newly_dirty_page_count_out,
    int *halted_ok_out,
    char *error_buffer,
    size_t error_capacity
);

/* The design's own "Restore" step: for every page named in
 * `dirty_page_indices[0..*dirty_page_index_count)` (sorted in place, then
 * adjacent runs coalesced into one `hv_vm_unmap`+`hv_vm_map` pair, not one
 * call per page -- an O(K log K) sort plus one O(K) coalescing pass over the
 * WORKING SET, never a scan of `dirty_bitmap`'s `base_size / granule` bits),
 * `hv_vm_unmap`s the delta mapping and `hv_vm_map`s that guest-IPA range
 * back to `base_host_address`, read-only+exec. Clears exactly those K bits
 * of `dirty_bitmap` (not the whole bitmap) and resets `*dirty_page_index_count`
 * to 0. Resets every gpr/`pc`/`cpsr` to the same fixed values
 * `proxima_vm_layered_run` resumes from. NEVER copies a guest-memory byte --
 * `base_host_address` was never written by a run (only `delta_host_address`
 * was), so nothing needs restoring at the byte level, only at the mapping
 * level; this is the literal "restore becomes a mapping, not a copy"
 * mechanism the owner's design note names, and the fault count tracking the
 * writable working set -- not the image size -- now holds for restore's own
 * cost too, not just `proxima_vm_layered_run`'s. Returns 0 on success, -1 on
 * failure (message in `error_buffer`). */
int proxima_vm_layered_restore(
    proxima_vm_layered_context_t *context,
    uint8_t *dirty_bitmap,
    size_t dirty_bitmap_capacity,
    uint32_t *dirty_page_indices,
    uint64_t *dirty_page_index_count,
    uint64_t *restore_wall_nanos_out,
    uint64_t *remap_nanos_out,
    uint64_t *register_reset_nanos_out,
    uint64_t *remapped_page_count_out,
    char *error_buffer,
    size_t error_capacity
);

/* Destroys this context's own vCPU only -- never the process-wide `hv_vm`
 * (`proxima_vm_layered_vcpu_create`'s own doc: the shared-base sharing proof
 * means more than one `proxima_vm_layered_context_t` can be alive over the
 * same one `hv_vm` at once, so no single context may own its teardown;
 * process exit reclaims it, the same "a fresh VM is a fresh process" shape
 * `src/snapshot.rs`'s own module doc already established for the cold
 * restore path). Does not touch `base_host_address`/`delta_host_address` --
 * those are caller-owned (`GuestMemoryRegion`/`RegionView` on the Rust
 * side) and outlive this call. */
void proxima_vm_layered_vcpu_destroy(proxima_vm_layered_context_t *context);

#endif
