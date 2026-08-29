#include <Hypervisor/Hypervisor.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include "dispatch_trampoline.h"
#include "ffi_segment.h"

#define TERMINAL_VALUE 256u

static int set_error(char *error_buffer, size_t error_capacity, const char *message) {
    if (error_capacity > 0) {
        snprintf(error_buffer, error_capacity, "%s", message);
    }
    return -1;
}

static int set_hv_error(
    char *error_buffer,
    size_t error_capacity,
    const char *operation,
    hv_return_t status
) {
    if (error_capacity > 0) {
        snprintf(error_buffer, error_capacity, "%s failed: 0x%x", operation, status);
    }
    return -1;
}

static uint32_t mov_x0_imm(uint16_t value) {
    return 0xd2800000u | ((uint32_t)value << 5u);
}

/* Shared vm-create step of the skeleton `proxima_vm_scratch_run` and
 * `proxima_vm_run_dispatch_loop` both otherwise repeated verbatim. Guest
 * memory mapping sits BETWEEN this and `create_and_start_vcpu` below (each
 * caller's own mapping shape — one RWX blob for the scratch guest, real
 * per-segment permissions for the dispatch-loop guest — so it cannot be
 * folded into either helper without losing that difference). */
static int create_vm(char *error_buffer, size_t error_capacity) {
    hv_return_t status = hv_vm_create(NULL);
    if (status != HV_SUCCESS) {
        return set_hv_error(error_buffer, error_capacity, "hv_vm_create", status);
    }
    return 0;
}

/* Shared vcpu-create-and-register-init step. `*vcpu_created_out` is set the
 * moment `hv_vcpu_create` succeeds — before either register write, which can
 * still fail — so a caller's cleanup always knows whether `hv_vcpu_destroy`
 * is owed, even on a partial failure inside this helper. */
static int create_and_start_vcpu(
    uint64_t entry,
    hv_vcpu_t *vcpu_out,
    hv_vcpu_exit_t **exit_data_out,
    int *vcpu_created_out,
    char *error_buffer,
    size_t error_capacity
) {
    hv_return_t status = hv_vcpu_create(vcpu_out, exit_data_out, NULL);
    if (status != HV_SUCCESS) {
        return set_hv_error(error_buffer, error_capacity, "hv_vcpu_create", status);
    }
    *vcpu_created_out = 1;

    status = hv_vcpu_set_reg(*vcpu_out, HV_REG_PC, entry);
    if (status != HV_SUCCESS) {
        return set_hv_error(error_buffer, error_capacity, "set guest pc", status);
    }
    status = hv_vcpu_set_reg(*vcpu_out, HV_REG_CPSR, 0x3c5u);
    if (status != HV_SUCCESS) {
        return set_hv_error(error_buffer, error_capacity, "set guest cpsr", status);
    }
    return 0;
}

/* Shared teardown halves, split so `proxima_vm_run_dispatch_loop`'s cleanup
 * can unmap its per-segment `hv_vm_map` calls between them (vcpu first, then
 * the segment unwind, then the vm itself) while `proxima_vm_scratch_run`
 * calls both back to back — the exact order both functions' `cleanup:`
 * labels already used. */
static void destroy_vcpu(hv_vcpu_t vcpu, int vcpu_created) {
    if (vcpu_created) {
        hv_vcpu_destroy(vcpu);
    }
}

static void destroy_vm(int vm_created) {
    if (vm_created) {
        hv_vm_destroy();
    }
}

int proxima_vm_scratch_run(
    const uint8_t *message,
    size_t message_length,
    uint8_t *output,
    size_t output_capacity,
    char *error_buffer,
    size_t error_capacity
) {
    const size_t page_size = (size_t)getpagesize();
    const size_t instruction_count = (message_length + 1u) * 2u;
    const size_t code_bytes = instruction_count * sizeof(uint32_t);
    const size_t memory_size = ((code_bytes + page_size - 1u) / page_size) * page_size;
    int result = -1;
    int vm_created = 0;
    int vcpu_created = 0;
    hv_vcpu_t vcpu = 0;
    hv_vcpu_exit_t *exit_data = NULL;
    void *guest_memory = MAP_FAILED;
    size_t output_length = 0;

    if (message_length > output_capacity) {
        return set_error(error_buffer, error_capacity, "scratch guest output capacity is too small");
    }

    guest_memory = mmap(
        NULL,
        memory_size,
        PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANON,
        -1,
        0
    );
    if (guest_memory == MAP_FAILED) {
        return set_error(error_buffer, error_capacity, "map guest memory failed");
    }

    uint32_t *code = (uint32_t *)guest_memory;
    for (size_t index = 0; index < message_length; ++index) {
        *code++ = mov_x0_imm(message[index]);
        *code++ = 0xd4000002u; /* hvc #0 */
    }
    *code++ = mov_x0_imm(TERMINAL_VALUE);
    *code = 0xd4000002u; /* hvc #0 */

    if (create_vm(error_buffer, error_capacity) != 0) {
        goto cleanup;
    }
    vm_created = 1;

    hv_return_t status = hv_vm_map(
        guest_memory,
        0,
        memory_size,
        HV_MEMORY_READ | HV_MEMORY_WRITE | HV_MEMORY_EXEC
    );
    if (status != HV_SUCCESS) {
        set_hv_error(error_buffer, error_capacity, "hv_vm_map", status);
        goto cleanup;
    }

    if (create_and_start_vcpu(0, &vcpu, &exit_data, &vcpu_created, error_buffer, error_capacity) != 0) {
        goto cleanup;
    }

    for (;;) {
        status = hv_vcpu_run(vcpu);
        if (status != HV_SUCCESS) {
            set_hv_error(error_buffer, error_capacity, "hv_vcpu_run", status);
            goto cleanup;
        }
        if (exit_data->reason != HV_EXIT_REASON_EXCEPTION) {
            if (error_capacity > 0) {
                snprintf(error_buffer, error_capacity, "unexpected Hypervisor exit reason %u", exit_data->reason);
            }
            goto cleanup;
        }
        const uint64_t exception_class = (exit_data->exception.syndrome >> 26u) & 0x3fu;
        if (exception_class != 0x16u) {
            if (error_capacity > 0) {
                snprintf(error_buffer, error_capacity, "unexpected arm exception class 0x%llx", exception_class);
            }
            goto cleanup;
        }

        uint64_t value = 0;
        status = hv_vcpu_get_reg(vcpu, HV_REG_X0, &value);
        if (status != HV_SUCCESS) {
            set_hv_error(error_buffer, error_capacity, "read guest output register", status);
            goto cleanup;
        }
        uint64_t program_counter = 0;
        status = hv_vcpu_get_reg(vcpu, HV_REG_PC, &program_counter);
        if (status != HV_SUCCESS) {
            set_hv_error(error_buffer, error_capacity, "read guest pc", status);
            goto cleanup;
        }
        if (value == TERMINAL_VALUE) {
            if (output_length != message_length) {
                set_error(error_buffer, error_capacity, "scratch guest halted before emitting declared output");
                goto cleanup;
            }
            result = 0;
            goto cleanup;
        }
        if (value > UINT8_MAX || output_length >= output_capacity) {
            if (error_capacity > 0) {
                snprintf(
                    error_buffer,
                    error_capacity,
                    "scratch guest emitted invalid byte: value=%llu pc=0x%llx syndrome=0x%llx index=%zu",
                    (unsigned long long)value,
                    (unsigned long long)program_counter,
                    (unsigned long long)exit_data->exception.syndrome,
                    output_length
                );
            }
            goto cleanup;
        }
        output[output_length++] = (uint8_t)value;
    }

cleanup:
    destroy_vcpu(vcpu, vcpu_created);
    destroy_vm(vm_created);
    if (guest_memory != MAP_FAILED) {
        munmap(guest_memory, memory_size);
    }
    return result;
}

static size_t round_up_to_page(size_t value, size_t page_size) {
    return ((value + page_size - 1u) / page_size) * page_size;
}

static hv_memory_flags_t segment_protection(const proxima_vm_segment_t *segment) {
    hv_memory_flags_t flags = 0;
    if (segment->readable) {
        flags |= HV_MEMORY_READ;
    }
    if (segment->writable) {
        flags |= HV_MEMORY_WRITE;
    }
    if (segment->executable) {
        flags |= HV_MEMORY_EXEC;
    }
    return flags;
}

/* Unmaps and unwinds every already-mapped entry in `mapped_out[0..count)` —
 * the same cleanup `proxima_vm_unmap_guest_memory` performs, factored out so
 * a partial failure inside the mapping loop below unwinds exactly what
 * succeeded before it, and a fully successful map shares one code path with
 * a later explicit unmap. */
static void unwind_mapped_segments(proxima_vm_mapped_segment_t *mapped_out, size_t count) {
    for (size_t index = 0; index < count; ++index) {
        hv_vm_unmap(mapped_out[index].guest_address, mapped_out[index].mapped_size);
        munmap(mapped_out[index].host_address, mapped_out[index].mapped_size);
    }
}

int proxima_vm_map_guest_memory(
    const proxima_vm_segment_t *segments,
    size_t segment_count,
    proxima_vm_mapped_segment_t *mapped_out,
    char *error_buffer,
    size_t error_capacity
) {
    const size_t page_size = (size_t)getpagesize();
    int vm_created = 0;

    hv_return_t status = hv_vm_create(NULL);
    if (status != HV_SUCCESS) {
        return set_hv_error(error_buffer, error_capacity, "hv_vm_create", status);
    }
    vm_created = 1;

    for (size_t index = 0; index < segment_count; ++index) {
        const proxima_vm_segment_t *segment = &segments[index];
        const size_t mapped_size = round_up_to_page(
            segment->memory_size > 0 ? segment->memory_size : 1,
            page_size
        );

        void *host_address = mmap(
            NULL,
            mapped_size,
            PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANON,
            -1,
            0
        );
        if (host_address == MAP_FAILED) {
            unwind_mapped_segments(mapped_out, index);
            hv_vm_destroy();
            return set_error(error_buffer, error_capacity, "map guest segment memory failed");
        }
        memcpy(host_address, segment->data, segment->data_length);

        status = hv_vm_map(host_address, segment->guest_address, mapped_size, segment_protection(segment));
        if (status != HV_SUCCESS) {
            munmap(host_address, mapped_size);
            unwind_mapped_segments(mapped_out, index);
            hv_vm_destroy();
            return set_hv_error(error_buffer, error_capacity, "hv_vm_map", status);
        }

        mapped_out[index].guest_address = segment->guest_address;
        mapped_out[index].host_address = host_address;
        mapped_out[index].mapped_size = mapped_size;
    }

    (void)vm_created;
    return 0;
}

void proxima_vm_unmap_guest_memory(
    const proxima_vm_mapped_segment_t *mapped,
    size_t mapped_count
) {
    for (size_t index = 0; index < mapped_count; ++index) {
        hv_vm_unmap(mapped[index].guest_address, mapped[index].mapped_size);
        munmap(mapped[index].host_address, mapped[index].mapped_size);
    }
    hv_vm_destroy();
}

#define DISPATCH_RESPONSE_SCRATCH_CAPACITY 512u
#define MAX_MAPPED_WINDOWS 16u

typedef struct {
    uint64_t start;
    uint64_t end;
    hv_memory_flags_t flags;
} mapped_window_t;

/* `hv_vm_map` only accepts page-aligned `(guest_address, size)` pairs, so
 * two segments whose BYTE ranges are disjoint (already proven by
 * `elf::parse_elf`) can still land in the same PAGE once rounded — this
 * guest's own `.text` (532 bytes) and `.rodata` (40 bytes) both sit inside
 * the first 4 KiB page. Merging same-page segments into one window and
 * OR-ing their permissions still gives real, page-granular W^X: a merged
 * window is writable only if EVERY segment sharing that page is writable,
 * so a page can never come out both writable and executable — the actual
 * property W^X protects against. `segments` is sorted by `guest_address`
 * first (insertion sort; `segment_count` is single digits, never a hot
 * path) so the merge is a single linear pass over sorted intervals.
 * Returns the window count, capped at `windows_capacity`. */
static size_t build_mapped_windows(
    const proxima_vm_segment_t *segments,
    size_t segment_count,
    size_t page_size,
    mapped_window_t *windows_out,
    size_t windows_capacity
) {
    proxima_vm_segment_t sorted[MAX_MAPPED_WINDOWS];
    const size_t count = segment_count < MAX_MAPPED_WINDOWS ? segment_count : MAX_MAPPED_WINDOWS;
    for (size_t index = 0; index < count; ++index) {
        sorted[index] = segments[index];
    }
    for (size_t index = 1; index < count; ++index) {
        const proxima_vm_segment_t key = sorted[index];
        size_t insert_at = index;
        while (insert_at > 0 && sorted[insert_at - 1].guest_address > key.guest_address) {
            sorted[insert_at] = sorted[insert_at - 1];
            --insert_at;
        }
        sorted[insert_at] = key;
    }

    size_t window_count = 0;
    for (size_t index = 0; index < count; ++index) {
        const proxima_vm_segment_t *segment = &sorted[index];
        const uint64_t start = (segment->guest_address / page_size) * page_size;
        const uint64_t size_bytes = segment->memory_size > 0 ? segment->memory_size : 1;
        const uint64_t end = start + round_up_to_page((size_t)(segment->guest_address - start + size_bytes), page_size);
        const hv_memory_flags_t flags = segment_protection(segment);

        if (window_count > 0 && start <= windows_out[window_count - 1].end) {
            mapped_window_t *last = &windows_out[window_count - 1];
            if (end > last->end) {
                last->end = end;
            }
            last->flags |= flags;
        } else if (window_count < windows_capacity) {
            windows_out[window_count].start = start;
            windows_out[window_count].end = end;
            windows_out[window_count].flags = flags;
            window_count += 1;
        }
    }
    return window_count;
}

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
) {
    const size_t page_size = (size_t)getpagesize();
    const size_t mapped_size = round_up_to_page(
        guest_memory_size > 0 ? guest_memory_size : 1,
        page_size
    );
    int result = -1;
    int vm_created = 0;
    int vcpu_created = 0;
    size_t windows_mapped = 0;
    mapped_window_t windows[MAX_MAPPED_WINDOWS];
    hv_vcpu_t vcpu = 0;
    hv_vcpu_exit_t *exit_data = NULL;
    void *guest_memory = MAP_FAILED;
    size_t emitted_length = 0;
    size_t hypercall_count = 0;
    uint8_t response_scratch[DISPATCH_RESPONSE_SCRATCH_CAPACITY];

    for (size_t index = 0; index < segment_count; ++index) {
        const proxima_vm_segment_t *segment = &segments[index];
        if (segment->guest_address > guest_memory_size
            || segment->memory_size > guest_memory_size - segment->guest_address) {
            return set_error(error_buffer, error_capacity, "guest segment exceeds guest memory reservation");
        }
    }

    guest_memory = mmap(NULL, mapped_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (guest_memory == MAP_FAILED) {
        return set_error(error_buffer, error_capacity, "map guest memory failed");
    }

    if (create_vm(error_buffer, error_capacity) != 0) {
        goto cleanup;
    }
    vm_created = 1;

    hv_return_t status;

    /* Copy every segment's file-backed bytes into its own `guest_address`
     * offset of the one flat `guest_memory` host allocation first (segments
     * never overlap — `elf::parse_elf` already proved that, so this never
     * double-writes a byte), THEN map real, page-merged permissions
     * (`build_mapped_windows`) instead of one `HV_MEMORY_READ |
     * HV_MEMORY_WRITE | HV_MEMORY_EXEC` blob covering the whole reservation.
     * Every `hv_vm_map` call below targets a disjoint sub-range of the same
     * flat `guest_memory` allocation, so pointer arithmetic elsewhere in
     * this loop (the emit verb, `proxima_vm_dispatch_hypercall`'s
     * guest-memory view) stays valid without a guest-address-to-host-pointer
     * translation table. */
    for (size_t index = 0; index < segment_count; ++index) {
        const proxima_vm_segment_t *segment = &segments[index];
        if (segment->data_length > 0) {
            memcpy((uint8_t *)guest_memory + segment->guest_address, segment->data, segment->data_length);
        }
    }

    windows_mapped = build_mapped_windows(segments, segment_count, page_size, windows, MAX_MAPPED_WINDOWS);
    for (size_t index = 0; index < windows_mapped; ++index) {
        const mapped_window_t *window = &windows[index];
        status = hv_vm_map(
            (uint8_t *)guest_memory + window->start,
            window->start,
            (size_t)(window->end - window->start),
            window->flags
        );
        if (status != HV_SUCCESS) {
            windows_mapped = index;
            set_hv_error(error_buffer, error_capacity, "hv_vm_map", status);
            goto cleanup;
        }
    }

    if (create_and_start_vcpu(entry, &vcpu, &exit_data, &vcpu_created, error_buffer, error_capacity) != 0) {
        goto cleanup;
    }

    for (;;) {
        status = hv_vcpu_run(vcpu);
        if (status != HV_SUCCESS) {
            set_hv_error(error_buffer, error_capacity, "hv_vcpu_run", status);
            goto cleanup;
        }
        if (exit_data->reason != HV_EXIT_REASON_EXCEPTION) {
            if (error_capacity > 0) {
                snprintf(error_buffer, error_capacity, "unexpected Hypervisor exit reason %u", exit_data->reason);
            }
            goto cleanup;
        }
        const uint64_t exception_class = (exit_data->exception.syndrome >> 26u) & 0x3fu;
        if (exception_class != 0x16u) {
            if (error_capacity > 0) {
                snprintf(error_buffer, error_capacity, "unexpected arm exception class 0x%llx", exception_class);
            }
            goto cleanup;
        }

        if (++hypercall_count > max_hypercalls) {
            set_error(error_buffer, error_capacity, "guest exceeded hypercall budget without halting");
            goto cleanup;
        }

        uint64_t verb = 0;
        uint64_t pointer = 0;
        uint64_t length = 0;
        status = hv_vcpu_get_reg(vcpu, HV_REG_X0, &verb);
        if (status == HV_SUCCESS) {
            status = hv_vcpu_get_reg(vcpu, HV_REG_X1, &pointer);
        }
        if (status == HV_SUCCESS) {
            status = hv_vcpu_get_reg(vcpu, HV_REG_X2, &length);
        }
        if (status != HV_SUCCESS) {
            set_hv_error(error_buffer, error_capacity, "read hypercall registers", status);
            goto cleanup;
        }

        if (verb == PROXIMA_VM_HALT_VERB) {
            result = 0;
            goto cleanup;
        }

        if (verb == PROXIMA_VM_EMIT_VERB) {
            if (pointer >= mapped_size || emitted_length >= emitted_capacity) {
                set_error(error_buffer, error_capacity, "emit hypercall pointer or output capacity out of range");
                goto cleanup;
            }
            emitted_out[emitted_length++] = ((const uint8_t *)guest_memory)[pointer];
            continue;
        }

        const int64_t dispatched = proxima_vm_dispatch_hypercall(
            dispatcher,
            (const uint8_t *)guest_memory,
            mapped_size,
            (uint16_t)verb,
            pointer,
            length,
            response_scratch,
            sizeof(response_scratch)
        );
        if (dispatched < 0) {
            if (error_capacity > 0) {
                snprintf(error_buffer, error_capacity, "hypercall dispatch failed with sentinel %lld", (long long)dispatched);
            }
            goto cleanup;
        }

        const size_t response_length = (size_t)dispatched;
        // the guest's own request buffer is the only response destination
        // the ABI names (`abi.rs`'s `pointer`/`length` describe the request,
        // not a separate reply region); write back only what the guest's
        // buffer can hold and let `x0` carry the true encoded length so a
        // truncated write is visible to the guest, not silent.
        const size_t writable = response_length < length ? response_length : (size_t)length;
        if (pointer + writable > mapped_size) {
            set_error(error_buffer, error_capacity, "dispatch response write-back would overrun guest memory");
            goto cleanup;
        }
        memcpy((uint8_t *)guest_memory + pointer, response_scratch, writable);

        status = hv_vcpu_set_reg(vcpu, HV_REG_X0, (uint64_t)response_length);
        if (status != HV_SUCCESS) {
            set_hv_error(error_buffer, error_capacity, "write hypercall result register", status);
            goto cleanup;
        }
    }

cleanup:
    destroy_vcpu(vcpu, vcpu_created);
    for (size_t index = 0; index < windows_mapped; ++index) {
        hv_vm_unmap(windows[index].start, (size_t)(windows[index].end - windows[index].start));
    }
    destroy_vm(vm_created);
    if (guest_memory != MAP_FAILED) {
        munmap(guest_memory, mapped_size);
    }
    if (result == 0 && emitted_length_out != NULL) {
        *emitted_length_out = emitted_length;
    }
    return result;
}
