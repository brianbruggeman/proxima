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

#endif
