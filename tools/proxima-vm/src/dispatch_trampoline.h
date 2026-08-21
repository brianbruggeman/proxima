#ifndef PROXIMA_VM_DISPATCH_TRAMPOLINE_H
#define PROXIMA_VM_DISPATCH_TRAMPOLINE_H

#include <stddef.h>
#include <stdint.h>

/* Rust-side hypercall dispatcher (`src/dispatch.rs`): decodes the payload
 * at guest_memory[pointer..pointer+length) as a postcard-encoded
 * `ChildRequest`, drives it through `RecordingDispatcher`'s `SendPipe::call`
 * via `futures::executor::block_on` (the same pattern `hello.rs:15` already
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

#endif
