#include <Hypervisor/Hypervisor.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <os/object.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include "dispatch_trampoline.h"
#include "ffi_segment.h"
#include "probe_cow.h"

#define TERMINAL_VALUE 256u

/* M3 fault-count instrument: one `CLOCK_MONOTONIC` read, nanoseconds.
 * `CLOCK_MONOTONIC` (not `_RAW`) matches what `std::time::Instant` reads on
 * this platform, so a caller comparing this number against a Rust-side
 * `Instant` timestamp is comparing the same clock. */
static uint64_t now_nanos(void) {
    struct timespec timestamp;
    clock_gettime(CLOCK_MONOTONIC, &timestamp);
    return (uint64_t)timestamp.tv_sec * 1000000000ull + (uint64_t)timestamp.tv_nsec;
}

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
/* Idempotent: `hv_vm_create` hangs (not errors) on a second call in a
 * process that already created one (`src/snapshot.rs`'s own module doc) --
 * every caller before the layered design ran in a fresh, single-use
 * process, so this guard was never exercised. The layered sharing proof
 * (`dispatch_trampoline.h`'s own doc on `proxima_vm_layered_vcpu_create`)
 * constructs a second context in the SAME process, over the same
 * process-wide `hv_vm`, so this call must become safe to make twice. */
static int g_vm_created = 0;

/* `enable_el2` is `edk2`-boot-only (`proxima_vm_run_edk2_dispatch_loop`'s
 * own doc): every other caller here passes 0, unchanged from before this
 * parameter existed. `hv_vm_config_set_el2_enabled`'s own SDK doc (read
 * directly off this host's `hv_vm_config.h`, macOS 15.8) scopes EL2-enabled
 * status to ONE concrete effect — how PMU register accesses are trapped —
 * not to "which EL the vcpu boots at"; there is no separate SDK knob for
 * that (`create_and_start_vcpu`'s own `cpsr` parameter is the only lever).
 * `hv_vm_config_t` is a once-per-process input to `hv_vm_create`, so this
 * can only ever take effect on this process's FIRST call (`g_vm_created`'s
 * own idempotence doc already establishes hv_vm_create is once-per-process
 * on this lane) -- a caller wanting EL2 enabled must be the first caller in
 * its process, which the edk2 probe binary's own single-purpose shape
 * guarantees. */
/* Return shape: 0 = created with `enable_el2` honored exactly as asked
 * (including `enable_el2 == 0`); 1 = `enable_el2` was requested but this
 * host's own HVF reported `HV_UNSUPPORTED` for
 * `hv_vm_config_set_el2_enabled` (MEASURED on this repo's own M1 Max /
 * macOS 15.8 host, `hv_vm_config_get_el2_supported` independently confirms
 * `supported=false` there) -- the VM was still created, but with EL2
 * disabled, so a caller that asked for EL2 entry must fall back to EL1h
 * before starting its vCPU; -1 = a real failure, message in
 * `error_buffer`. */
static int create_vm(int enable_el2, char *error_buffer, size_t error_capacity) {
    if (g_vm_created) {
        return 0;
    }
    hv_return_t status;
    if (enable_el2) {
        hv_vm_config_t config = hv_vm_config_create();
        if (config == NULL) {
            return set_error(error_buffer, error_capacity, "hv_vm_config_create returned NULL");
        }
        status = hv_vm_config_set_el2_enabled(config, true);
        if (status == HV_UNSUPPORTED) {
            os_release(config);
            status = hv_vm_create(NULL);
            if (status != HV_SUCCESS) {
                return set_hv_error(error_buffer, error_capacity, "hv_vm_create (el2 fallback)", status);
            }
            g_vm_created = 1;
            return 1;
        }
        if (status != HV_SUCCESS) {
            os_release(config);
            return set_hv_error(error_buffer, error_capacity, "hv_vm_config_set_el2_enabled", status);
        }
        status = hv_vm_create(config);
        os_release(config);
    } else {
        status = hv_vm_create(NULL);
    }
    if (status != HV_SUCCESS) {
        return set_hv_error(error_buffer, error_capacity, "hv_vm_create", status);
    }
    g_vm_created = 1;
    return 0;
}

/* Shared vcpu-create-and-register-init step. `*vcpu_created_out` is set the
 * moment `hv_vcpu_create` succeeds — before either register write, which can
 * still fail — so a caller's cleanup always knows whether `hv_vcpu_destroy`
 * is owed, even on a partial failure inside this helper. `cpsr` used to be
 * hard-coded `0x3c5u` (EL1h) here; every existing caller now passes that
 * same literal explicitly, so this change is a widen-not-narrow — the edk2
 * boot path (`boot::boot_edk2_firmware`'s own doc on why EL2h is tried
 * first, and why it may fall back to EL1h) is the one new caller that
 * passes something else. */
static int create_and_start_vcpu(
    uint64_t entry,
    uint64_t cpsr,
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
    status = hv_vcpu_set_reg(*vcpu_out, HV_REG_CPSR, cpsr);
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

/* Diagnostic-only, opt-in wall-clock forced exit: `proxima_vm_run_dispatch_loop`'s
 * own blocking `hv_vcpu_run` call cannot otherwise be interrupted, so a guest
 * that genuinely never traps (real execution, no VM exit at all -- this
 * crate's own edk2-boot investigation MEASURED exactly this shape via
 * `sample(1)`: every one of 1540 samples over a 2-second window sits inside
 * `Hv::Vcpu::run()`, never once returning to this file's own exit-handling
 * code) would otherwise hang the caller forever with zero diagnostic value.
 * `hv_vcpus_exit`'s own SDK doc names this exact cross-thread use: forces
 * `hv_vcpu_run` to return even while another thread is inside it. Every
 * existing caller passes `watchdog_millis == 0` (disabled, unchanged
 * behavior); [`boot::boot_edk2_firmware`]'s own doc names why it opts in. */
struct watchdog_context {
    hv_vcpu_t vcpu;
    uint64_t millis;
};

static void *watchdog_thread_main(void *argument) {
    struct watchdog_context *context = (struct watchdog_context *)argument;
    const struct timespec sleep_duration = {
        .tv_sec = (time_t)(context->millis / 1000u),
        .tv_nsec = (long)((context->millis % 1000u) * 1000000ull),
    };
    nanosleep(&sleep_duration, NULL);
    hv_vcpu_t vcpus[1] = {context->vcpu};
    // best-effort: if the loop already returned and destroyed its own vcpu
    // before this fires, `hv_vcpus_exit` targets a vcpu id that may no
    // longer be valid -- this thread is detached and its return value is
    // never observed, exactly the fire-and-forget shape a diagnostic-only
    // watchdog needs.
    (void)hv_vcpus_exit(vcpus, 1);
    free(context);
    return NULL;
}

/* Returns nonzero on failure (watchdog not armed; the caller proceeds
 * without one rather than treating this as fatal). */
static int arm_watchdog(hv_vcpu_t vcpu, uint64_t watchdog_millis) {
    if (watchdog_millis == 0) {
        return 0;
    }
    struct watchdog_context *context = malloc(sizeof(struct watchdog_context));
    if (context == NULL) {
        return -1;
    }
    context->vcpu = vcpu;
    context->millis = watchdog_millis;
    pthread_t thread;
    if (pthread_create(&thread, NULL, watchdog_thread_main, context) != 0) {
        free(context);
        return -1;
    }
    pthread_detach(thread);
    return 0;
}

static int set_mach_error(char *error_buffer, size_t error_capacity, const char *operation, kern_return_t status) {
    if (error_capacity > 0) {
        snprintf(error_buffer, error_capacity, "%s failed: %s", operation, mach_error_string(status));
    }
    return -1;
}

/* M4 — guest memory as a named object, HVF lane
 * (`tools/proxima-vm/ROADMAP.md`'s M4 section): `mach_make_memory_entry_64`
 * with `MAP_MEM_NAMED_CREATE` allocates a fresh, unbacked memory object and
 * hands back a mach port naming it -- the shareable identity `mmap(MAP_ANON)`
 * cannot produce, because an anonymous mapping's only identity is the one
 * process's virtual-address range. The first view is mapped immediately via
 * `mach_vm_map`. NOT explicitly zeroed here: a `MAP_MEM_NAMED_CREATE` entry
 * is backed by fresh anonymous zero-fill-on-demand pages, the same kernel
 * guarantee `mmap(MAP_ANON)` relied on -- an eager `memset` here would
 * first-touch (and therefore stage-2-fault) every page during region
 * creation itself, silently invalidating M3's own "wall to touch every
 * mapped page" measurement immediately below in
 * `proxima_vm_run_dispatch_loop` by measuring a second touch of
 * already-resident pages instead of the genuine first touch. */
int proxima_vm_create_named_region(
    size_t size,
    proxima_vm_named_region_t *region_out,
    char *error_buffer,
    size_t error_capacity
) {
    memory_object_size_t entry_size = (memory_object_size_t)size;
    mach_port_t handle = MACH_PORT_NULL;
    kern_return_t status = mach_make_memory_entry_64(
        mach_task_self(),
        &entry_size,
        0,
        MAP_MEM_NAMED_CREATE | VM_PROT_READ | VM_PROT_WRITE,
        &handle,
        MACH_PORT_NULL
    );
    if (status != KERN_SUCCESS) {
        return set_mach_error(error_buffer, error_capacity, "mach_make_memory_entry_64", status);
    }

    mach_vm_address_t address = 0;
    status = mach_vm_map(
        mach_task_self(),
        &address,
        (mach_vm_size_t)entry_size,
        0,
        VM_FLAGS_ANYWHERE,
        handle,
        0,
        FALSE,
        VM_PROT_READ | VM_PROT_WRITE,
        VM_PROT_READ | VM_PROT_WRITE,
        VM_INHERIT_NONE
    );
    if (status != KERN_SUCCESS) {
        mach_port_deallocate(mach_task_self(), handle);
        return set_mach_error(error_buffer, error_capacity, "mach_vm_map named region", status);
    }
    region_out->handle = (int)handle;
    region_out->primary_address = (void *)(uintptr_t)address;
    region_out->mapped_size = (size_t)entry_size;
    return 0;
}

/* `want_private_view` HVF has no simple mach-memory-entry copy-on-write
 * primitive equivalent to `mmap(MAP_PRIVATE, fd, ...)` (that shape needs a
 * `VM_FLAGS_COPY`-style submap arrangement well beyond the plain named-entry
 * API this function otherwise uses), and the M4 exit criterion's COW clause
 * is scoped to the KVM lane (`backend_linux.c`) -- so a private view request
 * here is a named, honest error rather than a silent shared-view
 * substitution. */
int proxima_vm_map_named_region(
    const proxima_vm_named_region_t *region,
    int want_private_view,
    void **host_address_out,
    char *error_buffer,
    size_t error_capacity
) {
    if (want_private_view) {
        return set_error(
            error_buffer,
            error_capacity,
            "private (copy-on-write) named-region views are not supported on the HVF lane"
        );
    }

    mach_vm_address_t address = 0;
    kern_return_t status = mach_vm_map(
        mach_task_self(),
        &address,
        (mach_vm_size_t)region->mapped_size,
        0,
        VM_FLAGS_ANYWHERE,
        (mach_port_t)region->handle,
        0,
        FALSE,
        VM_PROT_READ | VM_PROT_WRITE,
        VM_PROT_READ | VM_PROT_WRITE,
        VM_INHERIT_NONE
    );
    if (status != KERN_SUCCESS) {
        return set_mach_error(error_buffer, error_capacity, "mach_vm_map second named-region view", status);
    }
    *host_address_out = (void *)(uintptr_t)address;
    return 0;
}

void proxima_vm_unmap_named_region_view(void *host_address, size_t mapped_size) {
    mach_vm_deallocate(mach_task_self(), (mach_vm_address_t)(uintptr_t)host_address, (mach_vm_size_t)mapped_size);
}

void proxima_vm_destroy_named_region(proxima_vm_named_region_t *region) {
    if (region->primary_address != NULL) {
        proxima_vm_unmap_named_region_view(region->primary_address, region->mapped_size);
    }
    if (region->handle != 0) {
        mach_port_deallocate(mach_task_self(), (mach_port_t)region->handle);
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

    if (create_vm(0, error_buffer, error_capacity) != 0) {
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

    if (create_and_start_vcpu(0, 0x3c5u, &vcpu, &exit_data, &vcpu_created, error_buffer, error_capacity) != 0) {
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

/* Reads the full `x0..x30`/`pc`/`cpsr` register file into `registers_out` --
 * the exact set `proxima_vm_scratch_restore` below writes back. `HV_REG_X0 +
 * index` is the same offset idiom `backend_macos.c`'s own mmio
 * transfer-register decode already uses (`hv_vcpu_get_reg(vcpu, (hv_reg_t)
 * (HV_REG_X0 + iss.transfer_register), ...)`, this file's line ~610). */
static int capture_registers(
    hv_vcpu_t vcpu,
    proxima_vm_registers_t *registers_out,
    char *error_buffer,
    size_t error_capacity
) {
    for (size_t index = 0; index < 31u; ++index) {
        hv_return_t status = hv_vcpu_get_reg(vcpu, (hv_reg_t)(HV_REG_X0 + index), &registers_out->gpr[index]);
        if (status != HV_SUCCESS) {
            return set_hv_error(error_buffer, error_capacity, "capture guest gpr", status);
        }
    }
    hv_return_t status = hv_vcpu_get_reg(vcpu, HV_REG_PC, &registers_out->pc);
    if (status != HV_SUCCESS) {
        return set_hv_error(error_buffer, error_capacity, "capture guest pc", status);
    }
    status = hv_vcpu_get_reg(vcpu, HV_REG_CPSR, &registers_out->flags);
    if (status != HV_SUCCESS) {
        return set_hv_error(error_buffer, error_capacity, "capture guest cpsr", status);
    }
    return 0;
}

/* Writes `registers_in` into a freshly created vCPU -- the exact inverse of
 * `capture_registers` above, called by `proxima_vm_scratch_restore` in place
 * of `create_and_start_vcpu`'s own fixed `entry`/`0x3c5u` reset values. */
static int restore_registers(
    hv_vcpu_t vcpu,
    const proxima_vm_registers_t *registers_in,
    char *error_buffer,
    size_t error_capacity
) {
    for (size_t index = 0; index < 31u; ++index) {
        hv_return_t status = hv_vcpu_set_reg(vcpu, (hv_reg_t)(HV_REG_X0 + index), registers_in->gpr[index]);
        if (status != HV_SUCCESS) {
            return set_hv_error(error_buffer, error_capacity, "restore guest gpr", status);
        }
    }
    hv_return_t status = hv_vcpu_set_reg(vcpu, HV_REG_PC, registers_in->pc);
    if (status != HV_SUCCESS) {
        return set_hv_error(error_buffer, error_capacity, "restore guest pc", status);
    }
    status = hv_vcpu_set_reg(vcpu, HV_REG_CPSR, registers_in->flags);
    if (status != HV_SUCCESS) {
        return set_hv_error(error_buffer, error_capacity, "restore guest cpsr", status);
    }
    return 0;
}

size_t proxima_vm_scratch_guest_memory_size(size_t message_length) {
    const size_t page_size = (size_t)getpagesize();
    const size_t instruction_count = (message_length + 1u) * 2u;
    const size_t code_bytes = instruction_count * sizeof(uint32_t);
    return round_up_to_page(code_bytes, page_size);
}

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
) {
    const size_t memory_size = proxima_vm_scratch_guest_memory_size(message_length);
    int result = -1;
    int vm_created = 0;
    int vcpu_created = 0;
    int region_created = 0;
    hv_vcpu_t vcpu = 0;
    hv_vcpu_exit_t *exit_data = NULL;
    proxima_vm_named_region_t region = {0, NULL, 0};
    size_t output_length = 0;

    if (message_length > output_capacity) {
        return set_error(error_buffer, error_capacity, "scratch guest output capacity is too small");
    }
    if (memory_size > guest_memory_capacity) {
        return set_error(error_buffer, error_capacity, "guest_memory_out capacity is too small for this message");
    }

    if (proxima_vm_create_named_region(memory_size, &region, error_buffer, error_capacity) != 0) {
        return -1;
    }
    region_created = 1;

    uint32_t *code = (uint32_t *)region.primary_address;
    for (size_t index = 0; index < message_length; ++index) {
        *code++ = mov_x0_imm(message[index]);
        *code++ = 0xd4000002u; /* hvc #0 */
    }
    *code++ = mov_x0_imm(TERMINAL_VALUE);
    *code = 0xd4000002u; /* hvc #0 */

    if (create_vm(0, error_buffer, error_capacity) != 0) {
        goto cleanup;
    }
    vm_created = 1;

    hv_return_t status = hv_vm_map(
        region.primary_address,
        0,
        memory_size,
        HV_MEMORY_READ | HV_MEMORY_WRITE | HV_MEMORY_EXEC
    );
    if (status != HV_SUCCESS) {
        set_hv_error(error_buffer, error_capacity, "hv_vm_map", status);
        goto cleanup;
    }

    if (create_and_start_vcpu(0, 0x3c5u, &vcpu, &exit_data, &vcpu_created, error_buffer, error_capacity) != 0) {
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
        if (value == TERMINAL_VALUE) {
            if (output_length != message_length) {
                set_error(error_buffer, error_capacity, "scratch guest halted before emitting declared output");
                goto cleanup;
            }
            if (capture_registers(vcpu, registers_out, error_buffer, error_capacity) != 0) {
                goto cleanup;
            }
            /* HVF's synchronous `hvc` trap leaves `ELR` (this vCPU's `pc`)
             * already past the trapping instruction -- there is no more
             * guest code beyond it (this guest's halting `hvc` is its last
             * instruction), so resuming a restored vCPU from the captured
             * `pc` verbatim would execute whatever the zero-filled memory
             * past the code blob decodes to, not a repeat of the halting
             * trap. Rewinding by one instruction (4 bytes, every aarch64
             * instruction this guest emits is fixed-width) captures "about
             * to execute the halting hvc" instead of "just executed it", so
             * `proxima_vm_scratch_restore`'s one resumed step re-traps at
             * the identical instruction and reads back the identical `x0` --
             * the restore-is-proven evidence M7 asks for. */
            registers_out->pc -= 4u;
            memcpy(guest_memory_out, region.primary_address, memory_size);
            result = 0;
            goto cleanup;
        }
        if (value > UINT8_MAX || output_length >= output_capacity) {
            if (error_capacity > 0) {
                snprintf(
                    error_buffer,
                    error_capacity,
                    "scratch guest emitted invalid byte: value=%llu index=%zu",
                    (unsigned long long)value,
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
    if (region_created) {
        proxima_vm_destroy_named_region(&region);
    }
    return result;
}

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
) {
    const uint64_t restore_start_nanos = now_nanos();
    const size_t stride = page_size > 0 ? page_size : (size_t)getpagesize();
    int result = -1;
    int vm_created = 0;
    int vcpu_created = 0;
    int region_created = 0;
    hv_vcpu_t vcpu = 0;
    hv_vcpu_exit_t *exit_data = NULL;
    proxima_vm_named_region_t region = {0, NULL, 0};

    *resumed_ok_out = 0;
    *fault_count_out = 0;

    /* Phase 1: named-region creation (`mach_make_memory_entry_64` +
     * `mach_vm_map`, the M4 region-object machinery `proxima_vm_scratch_run`
     * also pays). */
    {
        const uint64_t phase_start_nanos = now_nanos();
        if (proxima_vm_create_named_region(guest_memory_length, &region, error_buffer, error_capacity) != 0) {
            return -1;
        }
        *region_create_nanos_out = now_nanos() - phase_start_nanos;
    }
    region_created = 1;

    /* Phase 2: the snapshot-bytes memcpy, `page_size`-strided -- the exact
     * mirror of `proxima_vm_run_dispatch_loop`'s own M3 "wall to touch every
     * mapped page" loop, except each stride carries the restored bytes
     * instead of a zero write. */
    {
        const uint64_t touch_start_nanos = now_nanos();
        uint8_t *destination = (uint8_t *)region.primary_address;
        for (size_t offset = 0; offset < guest_memory_length; offset += stride) {
            const size_t chunk = (guest_memory_length - offset) < stride ? (guest_memory_length - offset) : stride;
            memcpy(destination + offset, guest_memory_in + offset, chunk);
        }
        *touch_all_pages_nanos_out = now_nanos() - touch_start_nanos;
    }

    /* Phase 3: `hv_vm_create` -- the µsec campaign's own suspect (M7's
     * measured evidence: region/vm/vcpu creation, not the memory copy,
     * dominates cold restore). */
    {
        const uint64_t phase_start_nanos = now_nanos();
        if (create_vm(0, error_buffer, error_capacity) != 0) {
            goto cleanup;
        }
        *vm_create_nanos_out = now_nanos() - phase_start_nanos;
    }
    vm_created = 1;

    /* Phase 4: `hv_vm_map`. */
    hv_return_t status;
    {
        const uint64_t phase_start_nanos = now_nanos();
        status = hv_vm_map(
            region.primary_address,
            0,
            region.mapped_size,
            HV_MEMORY_READ | HV_MEMORY_WRITE | HV_MEMORY_EXEC
        );
        if (status != HV_SUCCESS) {
            set_hv_error(error_buffer, error_capacity, "hv_vm_map", status);
            goto cleanup;
        }
        *vm_map_nanos_out = now_nanos() - phase_start_nanos;
    }

    /* Phase 5: `hv_vcpu_create`. */
    {
        const uint64_t phase_start_nanos = now_nanos();
        status = hv_vcpu_create(&vcpu, &exit_data, NULL);
        if (status != HV_SUCCESS) {
            set_hv_error(error_buffer, error_capacity, "hv_vcpu_create", status);
            goto cleanup;
        }
        *vcpu_create_nanos_out = now_nanos() - phase_start_nanos;
    }
    vcpu_created = 1;

    /* Phase 6: register restore. */
    {
        const uint64_t phase_start_nanos = now_nanos();
        if (restore_registers(vcpu, registers_in, error_buffer, error_capacity) != 0) {
            goto cleanup;
        }
        *register_restore_nanos_out = now_nanos() - phase_start_nanos;
    }
    *restore_wall_nanos_out = now_nanos() - restore_start_nanos;

    /* Phase 7: the one resumed step. The snapshot was captured at the
     * guest's own halting trap, whose faulting `hvc` has not yet been
     * retired, so this resumed step re-traps at the identical instruction --
     * the proof that restore reproduced the exact guest state, not merely
     * copied bytes. */
    {
        const uint64_t phase_start_nanos = now_nanos();
        status = hv_vcpu_run(vcpu);
        *first_retrap_nanos_out = now_nanos() - phase_start_nanos;
    }
    if (status != HV_SUCCESS) {
        set_hv_error(error_buffer, error_capacity, "hv_vcpu_run resumed step", status);
        goto cleanup;
    }
    if (exit_data->reason == HV_EXIT_REASON_EXCEPTION) {
        const uint64_t exception_class = (exit_data->exception.syndrome >> 26u) & 0x3fu;
        if (exception_class == 0x16u) {
            uint64_t resumed_x0 = 0;
            status = hv_vcpu_get_reg(vcpu, HV_REG_X0, &resumed_x0);
            if (status != HV_SUCCESS) {
                set_hv_error(error_buffer, error_capacity, "read resumed output register", status);
                goto cleanup;
            }
            *resumed_x0_out = resumed_x0;
            *resumed_ok_out = 1;
        }
    }
    result = 0;

cleanup:
    destroy_vcpu(vcpu, vcpu_created);
    destroy_vm(vm_created);
    if (region_created) {
        proxima_vm_destroy_named_region(&region);
    }
    return result;
}

/* Warm-restore trio (µsec campaign, first slice) -- `dispatch_trampoline.h`'s
 * own doc on `proxima_vm_warm_restore_context_t` names the whole shape;
 * these three functions are its constructor, its per-call operation, and its
 * destructor. */
int proxima_vm_scratch_warm_vm_create(
    size_t guest_memory_capacity,
    proxima_vm_warm_restore_context_t *context_out,
    char *error_buffer,
    size_t error_capacity
) {
    int vm_created = 0;
    int vcpu_created = 0;
    int region_created = 0;
    hv_vcpu_t vcpu = 0;
    hv_vcpu_exit_t *exit_data = NULL;
    proxima_vm_named_region_t region = {0, NULL, 0};

    if (proxima_vm_create_named_region(guest_memory_capacity, &region, error_buffer, error_capacity) != 0) {
        return -1;
    }
    region_created = 1;

    if (create_vm(0, error_buffer, error_capacity) != 0) {
        goto fail;
    }
    vm_created = 1;

    hv_return_t status = hv_vm_map(
        region.primary_address,
        0,
        region.mapped_size,
        HV_MEMORY_READ | HV_MEMORY_WRITE | HV_MEMORY_EXEC
    );
    if (status != HV_SUCCESS) {
        set_hv_error(error_buffer, error_capacity, "hv_vm_map", status);
        goto fail;
    }

    status = hv_vcpu_create(&vcpu, &exit_data, NULL);
    if (status != HV_SUCCESS) {
        set_hv_error(error_buffer, error_capacity, "hv_vcpu_create", status);
        goto fail;
    }
    vcpu_created = 1;

    context_out->region = region;
    context_out->guest_memory_capacity = guest_memory_capacity;
    context_out->vcpu = vcpu;
    context_out->exit_data = (void *)exit_data;
    return 0;

fail:
    destroy_vcpu(vcpu, vcpu_created);
    destroy_vm(vm_created);
    if (region_created) {
        proxima_vm_destroy_named_region(&region);
    }
    return -1;
}

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
) {
    const uint64_t restore_start_nanos = now_nanos();
    const size_t stride = page_size > 0 ? page_size : (size_t)getpagesize();
    const hv_vcpu_t vcpu = (hv_vcpu_t)context->vcpu;
    hv_vcpu_exit_t *exit_data = (hv_vcpu_exit_t *)context->exit_data;

    *resumed_ok_out = 0;
    *fault_count_out = 0;

    if (guest_memory_length > context->guest_memory_capacity) {
        return set_error(error_buffer, error_capacity, "warm restore guest memory exceeds context capacity");
    }

    /* No region/vm/vcpu creation phase -- the whole point of this path
     * (`dispatch_trampoline.h`'s doc on this function). Memcpy straight into
     * the already-mapped region: it has been host-addressable since
     * `proxima_vm_scratch_warm_vm_create`, so there is nothing to unmap or
     * remap here. */
    {
        const uint64_t touch_start_nanos = now_nanos();
        uint8_t *destination = (uint8_t *)context->region.primary_address;
        for (size_t offset = 0; offset < guest_memory_length; offset += stride) {
            const size_t chunk = (guest_memory_length - offset) < stride ? (guest_memory_length - offset) : stride;
            memcpy(destination + offset, guest_memory_in + offset, chunk);
        }
        *touch_all_pages_nanos_out = now_nanos() - touch_start_nanos;
    }

    hv_return_t status;
    {
        const uint64_t phase_start_nanos = now_nanos();
        if (restore_registers(vcpu, registers_in, error_buffer, error_capacity) != 0) {
            return -1;
        }
        *register_restore_nanos_out = now_nanos() - phase_start_nanos;
    }
    *restore_wall_nanos_out = now_nanos() - restore_start_nanos;

    {
        const uint64_t phase_start_nanos = now_nanos();
        status = hv_vcpu_run(vcpu);
        *first_retrap_nanos_out = now_nanos() - phase_start_nanos;
    }
    if (status != HV_SUCCESS) {
        return set_hv_error(error_buffer, error_capacity, "hv_vcpu_run resumed step", status);
    }
    if (exit_data->reason == HV_EXIT_REASON_EXCEPTION) {
        const uint64_t exception_class = (exit_data->exception.syndrome >> 26u) & 0x3fu;
        if (exception_class == 0x16u) {
            uint64_t resumed_x0 = 0;
            status = hv_vcpu_get_reg(vcpu, HV_REG_X0, &resumed_x0);
            if (status != HV_SUCCESS) {
                return set_hv_error(error_buffer, error_capacity, "read resumed output register", status);
            }
            *resumed_x0_out = resumed_x0;
            *resumed_ok_out = 1;
        }
    }
    return 0;
}

void proxima_vm_scratch_warm_vm_destroy(proxima_vm_warm_restore_context_t *context) {
    hv_vcpu_destroy((hv_vcpu_t)context->vcpu);
    hv_vm_destroy();
    proxima_vm_destroy_named_region(&context->region);
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
/* M5b hang bound (`proxima_vm_run_dispatch_loop`'s exit loop): every exit
 * shape that keeps returning control to this loop (an mmio/exception storm)
 * is capped here, well above the deepest real boot this slice has measured
 * (`mmio_trap_count=13540` at the vtimer wall). A genuine `wfi` park inside
 * `hv_vcpu_run` itself is NOT bounded by this constant -- see the exit
 * loop's own comment on `HV_EXIT_REASON_VTIMER_ACTIVATED`. */
#define MAX_TOTAL_EXITS 5000000ull

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

/* Data-abort ISS decode (ARM ARM D13.2.37 "ISS encoding for an exception
 * from a Data Abort"): every field this loop needs from the low 25 bits of
 * `syndrome`. `ISV` (bit 24) gates whether the rest is even valid — HVF
 * (like KVM) only populates a syndrome-based instruction decode for a
 * single-register load/store hitting an unmapped stage-2 IPA, which is
 * exactly the shape a virtio-mmio driver's `ldr`/`str` accesses take; a
 * guest using an unsupported addressing mode (`ldp`/`stp`, exclusive
 * accesses, atomics) reports `ISV == 0` and this loop surfaces that as a
 * dispatch failure rather than attempting a fallback instruction fetch and
 * decode. */
typedef struct {
    int instruction_syndrome_valid;
    int is_write;
    int access_size_bytes;
    uint32_t transfer_register;
} data_abort_iss_t;

static data_abort_iss_t decode_data_abort_iss(uint64_t syndrome) {
    const uint32_t iss = (uint32_t)(syndrome & 0x1ffffffull);
    data_abort_iss_t decoded;
    decoded.instruction_syndrome_valid = (iss >> 24) & 0x1u;
    decoded.is_write = (iss >> 6) & 0x1u;
    const uint32_t access_size_encoding = (iss >> 22) & 0x3u;
    decoded.access_size_bytes = 1 << access_size_encoding;
    decoded.transfer_register = (iss >> 16) & 0x1fu;
    return decoded;
}

/* Layered base+delta warm restore -- `dispatch_trampoline.h`'s own doc on
 * `proxima_vm_layered_context_t` and the four functions below names the
 * whole design and what it replaced. Placed right after
 * `decode_data_abort_iss` (not beside the warm-restore trio at line ~709)
 * so `proxima_vm_layered_run` can call it without a forward declaration. */

size_t proxima_vm_host_page_size(void) {
    return (size_t)getpagesize();
}

/* Shared register-reset step every layered entry point (adopt/run/restore)
 * uses: gpr all zero, `pc = entry_pc` (this design's guest code always
 * starts at word 0 of its own base region, i.e. guest address `ipa_base`),
 * `cpsr` the same fixed value `create_and_start_vcpu` boots every vCPU
 * with. */
static int reset_layered_vcpu(hv_vcpu_t vcpu, uint64_t entry_pc, char *error_buffer, size_t error_capacity) {
    hv_return_t status = hv_vcpu_set_reg(vcpu, HV_REG_PC, entry_pc);
    for (size_t index = 0; status == HV_SUCCESS && index < 31u; ++index) {
        status = hv_vcpu_set_reg(vcpu, (hv_reg_t)(HV_REG_X0 + index), 0);
    }
    if (status == HV_SUCCESS) {
        status = hv_vcpu_set_reg(vcpu, HV_REG_CPSR, 0x3c5u);
    }
    if (status != HV_SUCCESS) {
        return set_hv_error(error_buffer, error_capacity, "reset layered vcpu registers", status);
    }
    return 0;
}

int proxima_vm_layered_vcpu_create(
    void *base_host_address,
    size_t base_size,
    void *delta_host_address,
    size_t delta_size,
    uint64_t ipa_base,
    proxima_vm_layered_context_t *context_out,
    char *error_buffer,
    size_t error_capacity
) {
    if (create_vm(0, error_buffer, error_capacity) != 0) {
        return -1;
    }

    hv_vcpu_t vcpu = 0;
    hv_vcpu_exit_t *exit_data = NULL;
    int vcpu_created = 0;
    if (create_and_start_vcpu(ipa_base, 0x3c5u, &vcpu, &exit_data, &vcpu_created, error_buffer, error_capacity) != 0) {
        destroy_vcpu(vcpu, vcpu_created);
        return -1;
    }

    context_out->base_host_address = base_host_address;
    context_out->base_size = base_size;
    context_out->delta_host_address = delta_host_address;
    context_out->delta_size = delta_size;
    context_out->ipa_base = ipa_base;
    context_out->vcpu = vcpu;
    context_out->exit_data = (void *)exit_data;
    context_out->mapped = 0;
    return 0;
}

int proxima_vm_layered_adopt_base(
    proxima_vm_layered_context_t *context,
    uint8_t *dirty_bitmap,
    size_t dirty_bitmap_capacity,
    uint64_t *map_nanos_out,
    uint64_t *register_reset_nanos_out,
    char *error_buffer,
    size_t error_capacity
) {
    const hv_vcpu_t vcpu = (hv_vcpu_t)context->vcpu;

    {
        const uint64_t map_start_nanos = now_nanos();
        if (context->mapped) {
            hv_vm_unmap(context->ipa_base, context->base_size);
        }
        hv_return_t status = hv_vm_map(
            context->base_host_address,
            context->ipa_base,
            context->base_size,
            HV_MEMORY_READ | HV_MEMORY_EXEC
        );
        *map_nanos_out = now_nanos() - map_start_nanos;
        if (status != HV_SUCCESS) {
            return set_hv_error(error_buffer, error_capacity, "hv_vm_map adopt_base", status);
        }
        context->mapped = 1;
    }

    memset(dirty_bitmap, 0, dirty_bitmap_capacity);

    {
        const uint64_t register_start_nanos = now_nanos();
        if (reset_layered_vcpu(vcpu, context->ipa_base, error_buffer, error_capacity) != 0) {
            return -1;
        }
        *register_reset_nanos_out = now_nanos() - register_start_nanos;
    }
    return 0;
}

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
) {
    const size_t granule = (size_t)getpagesize();
    const hv_vcpu_t vcpu = (hv_vcpu_t)context->vcpu;
    hv_vcpu_exit_t *exit_data = (hv_vcpu_exit_t *)context->exit_data;

    *halted_ok_out = 0;
    *fault_count_out = 0;
    *newly_dirty_page_count_out = 0;
    *run_wall_nanos_out = 0;

    if (reset_layered_vcpu(vcpu, context->ipa_base, error_buffer, error_capacity) != 0) {
        return -1;
    }

    const uint64_t run_start_nanos = now_nanos();
    const uint64_t max_exits = expected_page_count + 64u;
    uint64_t exit_index = 0;

    for (;;) {
        if (++exit_index > max_exits) {
            *run_wall_nanos_out = now_nanos() - run_start_nanos;
            return set_error(error_buffer, error_capacity, "layered run exceeded exit budget without halting");
        }
        hv_return_t status = hv_vcpu_run(vcpu);
        if (status != HV_SUCCESS) {
            *run_wall_nanos_out = now_nanos() - run_start_nanos;
            return set_hv_error(error_buffer, error_capacity, "hv_vcpu_run layered run", status);
        }
        if (exit_data->reason != HV_EXIT_REASON_EXCEPTION) {
            *run_wall_nanos_out = now_nanos() - run_start_nanos;
            return set_error(error_buffer, error_capacity, "layered run saw a non-exception exit");
        }
        const uint64_t exception_class = (exit_data->exception.syndrome >> 26u) & 0x3fu;
        if (exception_class == 0x16u) {
            uint64_t resumed_x0 = 0;
            status = hv_vcpu_get_reg(vcpu, HV_REG_X0, &resumed_x0);
            if (status != HV_SUCCESS) {
                *run_wall_nanos_out = now_nanos() - run_start_nanos;
                return set_hv_error(error_buffer, error_capacity, "read layered-run halt register", status);
            }
            if (resumed_x0 != TERMINAL_VALUE) {
                *run_wall_nanos_out = now_nanos() - run_start_nanos;
                return set_error(error_buffer, error_capacity, "layered run halted with an unexpected value");
            }
            *halted_ok_out = 1;
            break;
        }
        if (exception_class != 0x24u) {
            *run_wall_nanos_out = now_nanos() - run_start_nanos;
            if (error_capacity > 0) {
                snprintf(
                    error_buffer,
                    error_capacity,
                    "layered run saw unexpected exception class 0x%llx",
                    (unsigned long long)exception_class
                );
            }
            return -1;
        }
        const data_abort_iss_t iss = decode_data_abort_iss(exit_data->exception.syndrome);
        if (!iss.instruction_syndrome_valid || !iss.is_write) {
            *run_wall_nanos_out = now_nanos() - run_start_nanos;
            return set_error(error_buffer, error_capacity, "layered run saw a non-write data abort");
        }
        const uint64_t fault_address = exit_data->exception.physical_address;
        if (fault_address < context->ipa_base || fault_address >= context->ipa_base + (uint64_t)context->base_size) {
            *run_wall_nanos_out = now_nanos() - run_start_nanos;
            return set_error(error_buffer, error_capacity, "layered run fault address outside the base range");
        }
        const uint64_t region_offset = fault_address - context->ipa_base;
        const uint64_t page_index = region_offset / (uint64_t)granule;
        const uint64_t page_offset = page_index * (uint64_t)granule;
        const size_t byte_index = (size_t)(page_index / 8u);
        const uint8_t bit_mask = (uint8_t)(1u << (page_index % 8u));
        if (byte_index >= dirty_bitmap_capacity) {
            *run_wall_nanos_out = now_nanos() - run_start_nanos;
            return set_error(error_buffer, error_capacity, "layered run page index exceeds bitmap capacity");
        }
        ++*fault_count_out;
        if ((dirty_bitmap[byte_index] & bit_mask) != 0u) {
            /* Already delta-mapped read-write from an earlier fault in this
             * same run -- should not re-fault, but if it does (a second
             * write racing the first before it retried), there is nothing
             * further to copy or remap. */
            continue;
        }
        dirty_bitmap[byte_index] |= bit_mask;
        ++*newly_dirty_page_count_out;

        /* Appended exactly once per page, ever, between this context's last
         * adopt/restore and now -- the bitmap dedup check just above is what
         * guarantees that (a page already marked dirty `continue`s before
         * reaching here), so this list never needs its own dedup pass. */
        if (*dirty_page_index_count >= dirty_page_indices_capacity) {
            *run_wall_nanos_out = now_nanos() - run_start_nanos;
            return set_error(error_buffer, error_capacity, "layered run dirty page index list exceeded capacity");
        }
        dirty_page_indices[*dirty_page_index_count] = (uint32_t)page_index;
        ++*dirty_page_index_count;

        /* Exactly ONE granule, base -> delta -- never the whole region
         * (the deleted design's own defect). The delta page must carry the
         * base's own content before it becomes the guest's writable view,
         * since the guest's next retried store only overwrites the bytes it
         * targets, not the rest of the page. */
        memcpy(
            (uint8_t *)context->delta_host_address + page_offset,
            (const uint8_t *)context->base_host_address + page_offset,
            granule
        );
        hv_vm_unmap(context->ipa_base + page_offset, granule);
        status = hv_vm_map(
            (uint8_t *)context->delta_host_address + page_offset,
            context->ipa_base + page_offset,
            granule,
            HV_MEMORY_READ | HV_MEMORY_WRITE
        );
        if (status != HV_SUCCESS) {
            *run_wall_nanos_out = now_nanos() - run_start_nanos;
            return set_hv_error(error_buffer, error_capacity, "hv_vm_map delta page", status);
        }
        /* No PC advance -- the faulting store retries against the now-
         * writable delta page and succeeds (slice 3's `3c` mechanism). */
    }
    *run_wall_nanos_out = now_nanos() - run_start_nanos;
    return 0;
}

/* Ascending order is all `proxima_vm_layered_restore`'s coalescing pass
 * needs from the list -- `proxima_vm_layered_run` appends in FAULT order
 * (whatever order the guest's stores hit pages), not page order. */
static int compare_dirty_page_index(const void *left, const void *right) {
    const uint32_t left_value = *(const uint32_t *)left;
    const uint32_t right_value = *(const uint32_t *)right;
    if (left_value < right_value) {
        return -1;
    }
    if (left_value > right_value) {
        return 1;
    }
    return 0;
}

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
) {
    const size_t granule = (size_t)getpagesize();
    const hv_vcpu_t vcpu = (hv_vcpu_t)context->vcpu;
    *remapped_page_count_out = 0;

    const uint64_t restore_start_nanos = now_nanos();
    const size_t dirty_count = (size_t)*dirty_page_index_count;

    {
        const uint64_t remap_start_nanos = now_nanos();

        /* K log K sort (K = the working set, never `base_size / granule`),
         * then one O(K) pass coalescing adjacent page numbers into runs --
         * replaces the deleted O(region page count) bitmap scan this
         * function used to run regardless of how few pages were dirty. */
        qsort(dirty_page_indices, dirty_count, sizeof(uint32_t), compare_dirty_page_index);

        size_t run_start_index = 0;
        while (run_start_index < dirty_count) {
            size_t run_end_index = run_start_index + 1u;
            while (run_end_index < dirty_count &&
                   dirty_page_indices[run_end_index] == dirty_page_indices[run_end_index - 1u] + 1u) {
                ++run_end_index;
            }
            const uint32_t run_start_page = dirty_page_indices[run_start_index];
            const size_t run_length_pages = run_end_index - run_start_index;
            const size_t run_offset = (size_t)run_start_page * granule;
            const size_t run_end_page_bytes = (size_t)(run_start_page + run_length_pages) * granule;
            const size_t run_length =
                (run_end_page_bytes < context->base_size ? run_end_page_bytes : context->base_size) - run_offset;
            if (run_length > 0) {
                hv_vm_unmap(context->ipa_base + run_offset, run_length);
                hv_return_t status = hv_vm_map(
                    (uint8_t *)context->base_host_address + run_offset,
                    context->ipa_base + run_offset,
                    run_length,
                    HV_MEMORY_READ | HV_MEMORY_EXEC
                );
                if (status != HV_SUCCESS) {
                    *remap_nanos_out = now_nanos() - remap_start_nanos;
                    return set_hv_error(error_buffer, error_capacity, "hv_vm_map restore rearm", status);
                }
            }
            *remapped_page_count_out += (uint64_t)run_length_pages;
            run_start_index = run_end_index;
        }

        /* Clears exactly the K bits this restore just handled -- never the
         * whole `dirty_bitmap_capacity`-byte bitmap, which scales with
         * `base_size`, not with K. */
        for (size_t list_index = 0; list_index < dirty_count; ++list_index) {
            const uint32_t page_index = dirty_page_indices[list_index];
            const size_t byte_index = (size_t)(page_index / 8u);
            const uint8_t bit_mask = (uint8_t)(1u << (page_index % 8u));
            if (byte_index < dirty_bitmap_capacity) {
                dirty_bitmap[byte_index] &= (uint8_t)~bit_mask;
            }
        }
        *dirty_page_index_count = 0;

        *remap_nanos_out = now_nanos() - remap_start_nanos;
    }

    {
        const uint64_t register_start_nanos = now_nanos();
        if (reset_layered_vcpu(vcpu, context->ipa_base, error_buffer, error_capacity) != 0) {
            return -1;
        }
        *register_reset_nanos_out = now_nanos() - register_start_nanos;
    }

    *restore_wall_nanos_out = now_nanos() - restore_start_nanos;
    return 0;
}

void proxima_vm_layered_vcpu_destroy(proxima_vm_layered_context_t *context) {
    hv_vcpu_destroy((hv_vcpu_t)context->vcpu);
}

/* ISS decode for EC 0x18 (trapped `MSR`/`MRS`/system instruction), ARM DDI
 * 0487 ISS-encoding-for-trapped-sysreg layout: `Op0`\[21:20\], `Op2`\[19:17\],
 * `Op1`\[16:14\], `CRn`\[13:10\], `Rt`\[9:5\], `CRm`\[4:1\], `Direction`\[0\]
 * (1 = read/MRS, 0 = write/MSR) -- the exact fields this VM's own first
 * decoded wall (syndrome `0x6230102d`) named: `op0=3 op1=0 crn=4 crm=6
 * op2=0` is `S3_0_C4_C6_0` == `ICC_PMR_EL1`, `rt=1`, direction=read. Mirrors
 * `decode_data_abort_iss` immediately below it in spirit (one syndrome-in,
 * one typed decode-out), but this trap's ISS carries no `ISV`-style
 * validity bit -- the architecture always populates every field for a
 * trapped `MSR`/`MRS`, unlike a data abort's instruction-decode-may-be-
 * absent case. */
typedef struct {
    uint8_t op0;
    uint8_t op1;
    uint8_t crn;
    uint8_t crm;
    uint8_t op2;
    uint8_t rt;
    uint8_t is_read;
} icc_sysreg_iss_t;

static icc_sysreg_iss_t decode_icc_sysreg_iss(uint64_t syndrome) {
    const uint32_t iss = (uint32_t)(syndrome & 0x1ffffffull);
    icc_sysreg_iss_t decoded;
    decoded.op0 = (uint8_t)((iss >> 20) & 0x3u);
    decoded.op2 = (uint8_t)((iss >> 17) & 0x7u);
    decoded.op1 = (uint8_t)((iss >> 14) & 0x7u);
    decoded.crn = (uint8_t)((iss >> 10) & 0xfu);
    decoded.rt = (uint8_t)((iss >> 5) & 0x1fu);
    decoded.crm = (uint8_t)((iss >> 1) & 0xfu);
    decoded.is_read = (uint8_t)(iss & 0x1u);
    return decoded;
}

/* Handles one EC 0x18 (trapped `MSR`/`MRS`) exit whose decoded
 * `(op0, op1, crn, crm, op2)` names a GICv3 CPU-interface system register:
 * decodes the ISS, reads `Rt` for a write (`Rt == 31` is `xzr`, this
 * kernel's own probe path uses it -- see `apply_sre`'s doc for why `SRE`
 * must accept that write unconditionally), calls
 * `proxima_vm_dispatch_sysreg_icc`, writes `Rt` back for a read (`xzr`
 * discards the value, matching the architected "writes to xzr are
 * discarded" rule), and advances `PC` past the fixed-width trapping
 * instruction. Returns 0 on success, -1 on any decode, dispatch, or
 * hypervisor-register failure (message in `error_buffer`), including an
 * encoding this ICC model does not implement -- that rejection's message
 * names the full `S<op0>_<op1>_C<crn>_C<crm>_<op2>` encoding, the read/write
 * direction, and the trampoline's own reason code
 * (`ICC_DISPATCH_UNKNOWN_REGISTER`/`_READ_ONLY_REGISTER`/
 * `_WRITE_ONLY_REGISTER`, `dispatch_trampoline.h`), so the next wall this
 * model cannot yet decode names itself instead of falling back to the
 * caller's generic unknown-EC path. */
static int handle_icc_sysreg_trap(
    hv_vcpu_t vcpu,
    hv_vcpu_exit_t *exit_data,
    void *icc_transport,
    char *error_buffer,
    size_t error_capacity
) {
    const icc_sysreg_iss_t iss = decode_icc_sysreg_iss(exit_data->exception.syndrome);

    uint64_t source_value = 0;
    hv_return_t status;
    if (!iss.is_read && iss.rt != 31u) {
        status = hv_vcpu_get_reg(vcpu, (hv_reg_t)(HV_REG_X0 + iss.rt), &source_value);
        if (status != HV_SUCCESS) {
            return set_hv_error(error_buffer, error_capacity, "read icc source register", status);
        }
    }

    uint64_t read_value = 0;
    uint8_t deactivated = 0;
    uint32_t deactivated_intid = 0;
    const int32_t dispatch_status = proxima_vm_dispatch_sysreg_icc(
        icc_transport,
        iss.op0,
        iss.op1,
        iss.crn,
        iss.crm,
        iss.op2,
        iss.is_read ? 0u : 1u,
        source_value,
        &read_value,
        &deactivated,
        &deactivated_intid
    );
    if (dispatch_status != 0) {
        const char *reason = "rejected for an unrecognized reason";
        if (dispatch_status == ICC_DISPATCH_UNKNOWN_REGISTER) {
            reason = "no icc register modeled at this encoding";
        } else if (dispatch_status == ICC_DISPATCH_READ_ONLY_REGISTER) {
            reason = "register is read-only";
        } else if (dispatch_status == ICC_DISPATCH_WRITE_ONLY_REGISTER) {
            reason = "register is write-only";
        }
        if (error_capacity > 0) {
            snprintf(
                error_buffer,
                error_capacity,
                "icc sysreg access rejected: S%u_%u_C%u_C%u_%u %s (%s)",
                iss.op0,
                iss.op1,
                iss.crn,
                iss.crm,
                iss.op2,
                iss.is_read ? "read" : "write",
                reason
            );
        }
        return -1;
    }

    if (iss.is_read && iss.rt != 31u) {
        status = hv_vcpu_set_reg(vcpu, (hv_reg_t)(HV_REG_X0 + iss.rt), read_value);
        if (status != HV_SUCCESS) {
            return set_hv_error(error_buffer, error_capacity, "write icc destination register", status);
        }
    }

    uint64_t program_counter = 0;
    status = hv_vcpu_get_reg(vcpu, HV_REG_PC, &program_counter);
    if (status != HV_SUCCESS) {
        return set_hv_error(error_buffer, error_capacity, "read icc faulting pc", status);
    }
    status = hv_vcpu_set_reg(vcpu, HV_REG_PC, program_counter + 4u);
    if (status != HV_SUCCESS) {
        return set_hv_error(error_buffer, error_capacity, "advance icc faulting pc", status);
    }

    /* M5b-beyond re-arm: `deactivated` names an `ICC_EOIR1_EL1` write that
     * just retired the ICC model's one active interrupt
     * (`IccEffect::InterruptDeactivated`, `gic.rs`). When it is the
     * vtimer's own INTID, complete the architected "servicing this
     * interrupt is done" contract `hv_vcpu_set_vtimer_mask`'s own SDK doc
     * names: clear the IRQ line HVF is asserting into the guest, then clear
     * the mask so a future timeout can raise `HV_EXIT_REASON_VTIMER_
     * ACTIVATED` again. */
    if (deactivated != 0 && deactivated_intid == PROXIMA_VM_VTIMER_INTID) {
        status = hv_vcpu_set_pending_interrupt(vcpu, HV_INTERRUPT_TYPE_IRQ, false);
        if (status != HV_SUCCESS) {
            return set_hv_error(error_buffer, error_capacity, "clear vtimer pending interrupt", status);
        }
        status = hv_vcpu_set_vtimer_mask(vcpu, false);
        if (status != HV_SUCCESS) {
            return set_hv_error(error_buffer, error_capacity, "clear vtimer mask on eoi", status);
        }
    }
    return 0;
}

/* One 32-bit register-lane access, routed to whichever device's window the
 * caller already resolved `fault_address` into. Factored out so a real
 * 64-bit access (GICv3 exposes several 64-bit registers, e.g. `GICR_TYPER` —
 * already modeled here as two adjacent 32-bit halves,
 * `src/gic.rs`'s `REG_GICR_TYPER_LOW`/`REG_GICR_TYPER_HIGH`) can be serviced
 * as two lane calls at `offset` and `offset + 4` without duplicating this
 * device-selection chain. */
static int32_t dispatch_one_register_lane(
    int is_console,
    int is_net,
    int is_blk,
    int is_gicd,
    int is_gicr,
    void *console_transport,
    void *net_transport,
    void *blk_transport,
    void *gicd_transport,
    void *gicr_transport,
    void *pl011_transport,
    uint64_t lane_offset,
    uint8_t is_write,
    uint32_t value,
    uint32_t *read_value_out,
    uint16_t *notified_queue_out,
    uint8_t *pl011_tx_byte_out,
    uint8_t *pl011_tx_emitted_out
) {
    *notified_queue_out = PROXIMA_VM_MMIO_NO_QUEUE_NOTIFIED;
    *pl011_tx_emitted_out = 0;
    if (is_console) {
        return proxima_vm_dispatch_mmio(
            console_transport, lane_offset, is_write, value, read_value_out, notified_queue_out
        );
    }
    if (is_net) {
        return proxima_vm_dispatch_mmio_net(
            net_transport, lane_offset, is_write, value, read_value_out, notified_queue_out
        );
    }
    if (is_blk) {
        return proxima_vm_dispatch_mmio_blk(
            blk_transport, lane_offset, is_write, value, read_value_out, notified_queue_out
        );
    }
    if (is_gicd) {
        return proxima_vm_dispatch_mmio_gicd(gicd_transport, lane_offset, is_write, value, read_value_out);
    }
    if (is_gicr) {
        return proxima_vm_dispatch_mmio_gicr(gicr_transport, lane_offset, is_write, value, read_value_out);
    }
    return proxima_vm_dispatch_mmio_pl011(
        pl011_transport, lane_offset, is_write, value, read_value_out, pl011_tx_byte_out, pl011_tx_emitted_out
    );
}

/* Handles one data-abort exit whose fault address falls inside either
 * reserved virtio-mmio window (console or net, `dispatch_trampoline.h`):
 * decodes the ISS, reads the source register for a write (or computes the
 * value to write back for a read) via `hv_vcpu_get_reg`/`hv_vcpu_set_reg`,
 * picks the window (and therefore which transport and which drain/emit
 * channel) the fault address falls in, applies the access through that
 * transport's `proxima_vm_dispatch_mmio*`, drains the notified queue (if
 * any) via the matching `proxima_vm_mmio_drain_tx*` into that window's own
 * emitted buffer, and advances `PC` past the fixed-width A64 instruction.
 * Returns 0 on success, -1 on any decode, dispatch, or hypervisor-register
 * failure (message in `error_buffer`), including a fault address outside
 * both windows. */
static int handle_mmio_data_abort(
    hv_vcpu_t vcpu,
    hv_vcpu_exit_t *exit_data,
    void *console_transport,
    void *net_transport,
    void *blk_transport,
    void *gicd_transport,
    void *gicr_transport,
    void *pl011_transport,
    uint8_t *guest_memory,
    size_t mapped_size,
    uint8_t *mmio_emitted_out,
    size_t mmio_emitted_capacity,
    size_t *mmio_emitted_length,
    uint8_t *net_emitted_out,
    size_t net_emitted_capacity,
    size_t *net_emitted_length,
    uint8_t *blk_emitted_out,
    size_t blk_emitted_capacity,
    size_t *blk_emitted_length,
    uint8_t *pl011_emitted_out,
    size_t pl011_emitted_capacity,
    size_t *pl011_emitted_length,
    uint64_t *gicd_trap_count,
    uint64_t *gicr_trap_count,
    uint64_t *pl011_trap_count,
    uint64_t *virtio_trap_count,
    char *error_buffer,
    size_t error_capacity
) {
    const data_abort_iss_t iss = decode_data_abort_iss(exit_data->exception.syndrome);
    if (!iss.instruction_syndrome_valid) {
        return set_error(
            error_buffer,
            error_capacity,
            "mmio data abort carries no single-register instruction decode"
        );
    }
    const uint64_t fault_address = exit_data->exception.physical_address;
    const int is_console = fault_address >= PROXIMA_VM_MMIO_WINDOW_BASE
        && fault_address - PROXIMA_VM_MMIO_WINDOW_BASE < PROXIMA_VM_MMIO_WINDOW_SIZE;
    const int is_net = fault_address >= PROXIMA_VM_NET_MMIO_WINDOW_BASE
        && fault_address - PROXIMA_VM_NET_MMIO_WINDOW_BASE < PROXIMA_VM_NET_MMIO_WINDOW_SIZE;
    const int is_blk = fault_address >= PROXIMA_VM_BLK_MMIO_WINDOW_BASE
        && fault_address - PROXIMA_VM_BLK_MMIO_WINDOW_BASE < PROXIMA_VM_BLK_MMIO_WINDOW_SIZE;
    const int is_gicd = fault_address >= PROXIMA_VM_GICD_MMIO_WINDOW_BASE
        && fault_address - PROXIMA_VM_GICD_MMIO_WINDOW_BASE < PROXIMA_VM_GICD_MMIO_WINDOW_SIZE;
    const int is_gicr = fault_address >= PROXIMA_VM_GICR_MMIO_WINDOW_BASE
        && fault_address - PROXIMA_VM_GICR_MMIO_WINDOW_BASE < PROXIMA_VM_GICR_MMIO_WINDOW_SIZE;
    const int is_pl011 = fault_address >= PROXIMA_VM_PL011_MMIO_WINDOW_BASE
        && fault_address - PROXIMA_VM_PL011_MMIO_WINDOW_BASE < PROXIMA_VM_PL011_MMIO_WINDOW_SIZE;
    if (!is_console && !is_net && !is_blk && !is_gicd && !is_gicr && !is_pl011) {
        return set_error(error_buffer, error_capacity, "data abort outside any mmio window");
    }
    /* M5b per-window trap counters -- attributed by the SAME window
     * resolution the dispatch below already computed, before any dispatch
     * can fail, so a failed access still counts against the window that
     * caused it (matches `mmio_trap_count`'s own "counted before dispatch"
     * placement at the call site). */
    if (is_gicd) {
        ++*gicd_trap_count;
    } else if (is_gicr) {
        ++*gicr_trap_count;
    } else if (is_pl011) {
        ++*pl011_trap_count;
    } else {
        ++*virtio_trap_count;
    }
    const uint64_t window_base = is_console ? PROXIMA_VM_MMIO_WINDOW_BASE
        : is_net             ? PROXIMA_VM_NET_MMIO_WINDOW_BASE
        : is_blk             ? PROXIMA_VM_BLK_MMIO_WINDOW_BASE
        : is_gicd            ? PROXIMA_VM_GICD_MMIO_WINDOW_BASE
        : is_gicr            ? PROXIMA_VM_GICR_MMIO_WINDOW_BASE
                              : PROXIMA_VM_PL011_MMIO_WINDOW_BASE;
    const uint64_t offset = fault_address - window_base;

    uint64_t register_value = 0;
    hv_return_t status;
    if (iss.is_write && iss.transfer_register != 31u) {
        status = hv_vcpu_get_reg(vcpu, (hv_reg_t)(HV_REG_X0 + iss.transfer_register), &register_value);
        if (status != HV_SUCCESS) {
            return set_hv_error(error_buffer, error_capacity, "read mmio source register", status);
        }
    }

    /* A narrower-than-4-byte access (`ldrb`/`strb`/`ldrh`/`strh`) still
     * targets this register's declared byte-0 lane -- every device model
     * here is a plain register file, not a byte-addressable memory region,
     * so a narrow write carries its value in the low bytes and a narrow
     * read is answered from the low bytes of the full register value,
     * zero-extended (`hv_vcpu_set_reg` below always writes a full 64-bit
     * destination). An 8-byte access instead walks two real 32-bit register
     * lanes at `offset` and `offset + 4` -- this model's own convention for
     * every 64-bit architected register it exposes (`GICR_TYPER`'s
     * low/high split, `src/gic.rs`'s own doc). */
    const int lane_count = iss.access_size_bytes == 8 ? 2 : 1;
    const int is_narrow = iss.access_size_bytes < 4;
    uint64_t combined_read_value = 0;
    uint16_t notified_queue = PROXIMA_VM_MMIO_NO_QUEUE_NOTIFIED;
    uint8_t pl011_tx_byte = 0;
    uint8_t pl011_tx_emitted = 0;
    int32_t dispatch_status = 0;

    for (int lane = 0; lane < lane_count; ++lane) {
        const uint64_t lane_offset = offset + (uint64_t)lane * 4u;
        const uint32_t lane_write_value = (uint32_t)(register_value >> (lane * 32));
        uint32_t lane_read_value = 0;
        uint16_t lane_notified_queue = PROXIMA_VM_MMIO_NO_QUEUE_NOTIFIED;
        uint8_t lane_pl011_tx_byte = 0;
        uint8_t lane_pl011_tx_emitted = 0;

        dispatch_status = dispatch_one_register_lane(
            is_console, is_net, is_blk, is_gicd, is_gicr,
            console_transport, net_transport, blk_transport, gicd_transport, gicr_transport, pl011_transport,
            lane_offset,
            iss.is_write ? 1u : 0u,
            lane_write_value,
            &lane_read_value,
            &lane_notified_queue,
            &lane_pl011_tx_byte,
            &lane_pl011_tx_emitted
        );
        if (dispatch_status != 0) {
            break;
        }
        combined_read_value |= (uint64_t)lane_read_value << (lane * 32);
        if (lane_notified_queue != PROXIMA_VM_MMIO_NO_QUEUE_NOTIFIED) {
            notified_queue = lane_notified_queue;
        }
        if (lane_pl011_tx_emitted) {
            pl011_tx_byte = lane_pl011_tx_byte;
            pl011_tx_emitted = 1;
        }
    }
    if (dispatch_status != 0) {
        const char *window_name = is_console ? "console"
            : is_net                          ? "net"
            : is_blk                          ? "blk"
            : is_gicd                         ? "gicd"
            : is_gicr                         ? "gicr"
                                               : "pl011";
        if (error_capacity > 0) {
            snprintf(
                error_buffer,
                error_capacity,
                "mmio register access rejected: window=%s offset=0x%llx is_write=%u",
                window_name,
                (unsigned long long)offset,
                (unsigned)iss.is_write
            );
        }
        return -1;
    }
    if (is_narrow) {
        const uint64_t narrow_mask = (1ull << (iss.access_size_bytes * 8u)) - 1u;
        combined_read_value &= narrow_mask;
    }
    if (pl011_tx_emitted) {
        if (*pl011_emitted_length >= pl011_emitted_capacity) {
            return set_error(error_buffer, error_capacity, "pl011 emitted-byte channel is full");
        }
        pl011_emitted_out[*pl011_emitted_length] = pl011_tx_byte;
        *pl011_emitted_length += 1;
    }

    if (!iss.is_write && iss.transfer_register != 31u) {
        status = hv_vcpu_set_reg(vcpu, (hv_reg_t)(HV_REG_X0 + iss.transfer_register), combined_read_value);
        if (status != HV_SUCCESS) {
            return set_hv_error(error_buffer, error_capacity, "write mmio destination register", status);
        }
    }

    if (notified_queue != PROXIMA_VM_MMIO_NO_QUEUE_NOTIFIED) {
        size_t drained_length = 0;
        int32_t drain_status;
        if (is_console) {
            drain_status = proxima_vm_mmio_drain_tx(
                console_transport,
                notified_queue,
                guest_memory,
                mapped_size,
                mmio_emitted_out + *mmio_emitted_length,
                mmio_emitted_capacity - *mmio_emitted_length,
                &drained_length
            );
        } else if (is_net) {
            drain_status = proxima_vm_mmio_drain_tx_net(
                net_transport,
                notified_queue,
                guest_memory,
                mapped_size,
                net_emitted_out + *net_emitted_length,
                net_emitted_capacity - *net_emitted_length,
                &drained_length
            );
        } else {
            drain_status = proxima_vm_mmio_service_blk(
                blk_transport,
                notified_queue,
                guest_memory,
                mapped_size,
                blk_emitted_out + *blk_emitted_length,
                blk_emitted_capacity - *blk_emitted_length,
                &drained_length
            );
        }
        if (drain_status != 0) {
            return set_error(error_buffer, error_capacity, "mmio queue drain failed");
        }
        if (is_console) {
            *mmio_emitted_length += drained_length;
        } else if (is_net) {
            *net_emitted_length += drained_length;
        } else {
            *blk_emitted_length += drained_length;
        }
    }

    uint64_t program_counter = 0;
    status = hv_vcpu_get_reg(vcpu, HV_REG_PC, &program_counter);
    if (status != HV_SUCCESS) {
        return set_hv_error(error_buffer, error_capacity, "read mmio faulting pc", status);
    }
    status = hv_vcpu_set_reg(vcpu, HV_REG_PC, program_counter + 4u);
    if (status != HV_SUCCESS) {
        return set_hv_error(error_buffer, error_capacity, "advance mmio faulting pc", status);
    }
    return 0;
}

int proxima_vm_run_dispatch_loop(
    const proxima_vm_segment_t *segments,
    size_t segment_count,
    uint64_t guest_memory_size,
    uint64_t guest_memory_base,
    uint64_t entry,
    uint64_t boot_x0,
    /* 0 is a sentinel meaning "use this loop's own established default"
     * (`0x3c5u`, EL1h — every existing Linux-kernel/lambda-ELF caller's
     * unchanged behavior), not a literal CPSR value any real boot wants:
     * `0x3c5u`'s own low nibble (`0b0101`) is never 0, so no real caller
     * ever needs to pass literal 0 here. A nonzero value is written
     * verbatim (`create_and_start_vcpu`'s own `cpsr` parameter) and this
     * loop enables the HVF EL2 VM-config knob (`create_vm`'s own
     * `enable_el2` doc) whenever its low nibble names EL2h/EL2t — the
     * Rust-side `boot::boot_edk2_firmware`'s own module doc names why edk2
     * needs this and the existing kernel/lambda callers never pass it. */
    uint64_t boot_cpsr,
    const void *dispatcher,
    void *console_transport,
    void *net_transport,
    void *blk_transport,
    void *gicd_transport,
    void *gicr_transport,
    void *pl011_transport,
    void *icc_transport,
    size_t max_hypercalls,
    /* 0 = disabled (every existing caller). `arm_watchdog`'s own doc names
     * the diagnostic this exists for: a guest that genuinely never traps
     * (real, unbroken execution -- MAX_TOTAL_EXITS below only bounds exit
     * STORMS, not a guest that never exits at all) would otherwise hang
     * `hv_vcpu_run` forever with zero positional evidence. */
    uint64_t watchdog_millis,
    uint8_t *emitted_out,
    size_t emitted_capacity,
    size_t *emitted_length_out,
    uint8_t *mmio_emitted_out,
    size_t mmio_emitted_capacity,
    size_t *mmio_emitted_length_out,
    uint8_t *net_emitted_out,
    size_t net_emitted_capacity,
    size_t *net_emitted_length_out,
    uint8_t *blk_emitted_out,
    size_t blk_emitted_capacity,
    size_t *blk_emitted_length_out,
    uint8_t *pl011_emitted_out,
    size_t pl011_emitted_capacity,
    size_t *pl011_emitted_length_out,
    uint64_t *create_to_first_exit_nanos_out,
    uint64_t *touch_all_pages_nanos_out,
    uint64_t *mmio_trap_count_out,
    uint64_t *gicd_trap_count_out,
    uint64_t *gicr_trap_count_out,
    uint64_t *pl011_trap_count_out,
    uint64_t *virtio_trap_count_out,
    uint64_t *vtimer_activation_count_out,
    uint64_t *wfi_wfe_trap_count_out,
    uint64_t *entered_el2_out,
    char *error_buffer,
    size_t error_capacity
) {
    const uint64_t run_start_nanos = now_nanos();
    const size_t page_size = (size_t)getpagesize();
    const size_t mapped_size = round_up_to_page(
        guest_memory_size > 0 ? guest_memory_size : 1,
        page_size
    );
    int result = -1;
    int vm_created = 0;
    int vcpu_created = 0;
    int first_exit_seen = 0;
    /* `entered_el2_out`'s own doc: 0 by default (every existing caller's
     * `boot_cpsr == 0` sentinel never requests EL2, so this stays 0 for
     * them unchanged); set to 1 only when this run's `enable_el2` request
     * was actually honored (`create_vm`'s own return-shape doc), 0 again
     * on the HV_UNSUPPORTED fallback. */
    int entered_el2 = 0;
    size_t windows_mapped = 0;
    mapped_window_t windows[MAX_MAPPED_WINDOWS];
    hv_vcpu_t vcpu = 0;
    hv_vcpu_exit_t *exit_data = NULL;
    proxima_vm_named_region_t guest_memory_region = {0, NULL, 0};
    int guest_memory_region_created = 0;
    void *guest_memory = MAP_FAILED;
    size_t emitted_length = 0;
    size_t mmio_emitted_length = 0;
    size_t net_emitted_length = 0;
    size_t blk_emitted_length = 0;
    size_t pl011_emitted_length = 0;
    size_t hypercall_count = 0;
    uint64_t mmio_trap_count = 0;
    /* M5b per-window breakdown of `mmio_trap_count` (task's own ask: "gicd/
     * gicr/pl011/virtio each") -- the console/net/blk virtio-mmio windows
     * share one bucket since M5b's own boot never drives any of them (this
     * guest speaks no virtqueue protocol), so a three-way split there would
     * only ever show zeros in two of three counters. */
    uint64_t gicd_trap_count = 0;
    uint64_t gicr_trap_count = 0;
    uint64_t pl011_trap_count = 0;
    uint64_t virtio_trap_count = 0;
    uint64_t vtimer_activation_count = 0;
    /* EC 0x1 (trapped `WFI`/`WFE`, ARM DDI 0487 D13.2.37): PID1's own idle
     * park loop (`kernel_boot_userspace.rs`'s own wall, "unexpected arm
     * exception class 0x1") issues this once it has nothing left to
     * schedule. HVF traps it rather than actually parking the host thread
     * because this loop never told HVF interrupts are pending -- the guest
     * re-issues the same `wfi` the instant it is resumed, so counting these
     * (rather than silently swallowing them) is the only way a caller can
     * tell "idle-spinning as expected" apart from a real hang. */
    uint64_t wfi_wfe_trap_count = 0;
    uint64_t total_exit_count = 0;
    uint64_t create_to_first_exit_nanos = 0;
    uint64_t touch_all_pages_nanos = 0;
    uint8_t response_scratch[DISPATCH_RESPONSE_SCRATCH_CAPACITY];

    for (size_t index = 0; index < segment_count; ++index) {
        const proxima_vm_segment_t *segment = &segments[index];
        if (segment->guest_address > guest_memory_size
            || segment->memory_size > guest_memory_size - segment->guest_address) {
            return set_error(error_buffer, error_capacity, "guest segment exceeds guest memory reservation");
        }
    }

    /* M4: guest memory is a named mach memory object, not `mmap(MAP_ANON)` --
     * a second caller holding `guest_memory_region.handle` can map its own
     * view of the same backing object and observe writes made through this
     * one (`proxima_vm_map_named_region`), which an anonymous mapping could
     * never offer a snapshot/fork consumer. */
    if (proxima_vm_create_named_region(mapped_size, &guest_memory_region, error_buffer, error_capacity) != 0) {
        return -1;
    }
    guest_memory_region_created = 1;
    guest_memory = guest_memory_region.primary_address;

    /* M3's "wall to touch every mapped page": a first-touch write per
     * `page_size` stride of the named region's primary view, timed BEFORE
     * `hv_vm_map` so the number reflects only the host-side first-touch
     * cost of this mapping (post-M4: a `mach_vm_map`'d named-entry view
     * rather than an anonymous mapping), not any hypervisor-side mapping
     * work measured elsewhere. This is the HVF lane's whole deliverable on
     * this axis — see the header doc on why HVF has no per-page stage-2
     * fault index. */
    {
        const uint64_t touch_start_nanos = now_nanos();
        for (size_t offset = 0; offset < mapped_size; offset += page_size) {
            ((volatile uint8_t *)guest_memory)[offset] = 0u;
        }
        touch_all_pages_nanos = now_nanos() - touch_start_nanos;
    }

    /* `boot_cpsr`'s own parameter doc names the sentinel: 0 means "this
     * loop's established default", anything else is the literal CPSR to
     * enter at. EL2h is `M[3:0] == 0b1001` (`0x9u`), EL2t is `0b1000`
     * (`0x8u`) — ARM DDI 0487's own `PSTATE.M` encoding, the same field
     * `0x3c5u`'s own low nibble (`0b0101`, EL1h) already commits this loop
     * to for every other caller. */
    uint64_t resolved_cpsr = boot_cpsr != 0 ? boot_cpsr : 0x3c5u;
    const uint64_t cpsr_el_mode = resolved_cpsr & 0xfu;
    const int enable_el2 = (cpsr_el_mode == 0x9u || cpsr_el_mode == 0x8u) ? 1 : 0;

    const int create_vm_status = create_vm(enable_el2, error_buffer, error_capacity);
    if (create_vm_status < 0) {
        goto cleanup;
    }
    vm_created = 1;
    if (create_vm_status == 1) {
        /* `create_vm`'s own doc on this return value: this host's HVF
         * reported EL2 unsupported, so the vm this loop just created has
         * EL2 disabled -- the vcpu MUST enter at EL1h regardless of what
         * `boot_cpsr` asked for, or `create_and_start_vcpu`'s own
         * `hv_vcpu_set_reg(..., HV_REG_CPSR, ...)` would try to put the
         * vcpu in a mode the vm was never configured to support. */
        resolved_cpsr = 0x3c5u;
        entered_el2 = 0;
    } else if (enable_el2) {
        entered_el2 = 1;
    }

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

    /* `guest_memory_base` shifts the guest-physical address each window maps
     * at (a real boot's RAM sits at 0x4000_0000, `dtb.rs`'s own
     * `QemuVirtLayout` doc) without touching the host-side `guest_memory`
     * buffer, which always starts its own flat allocation at offset 0 — the
     * base only ever appears on the `hv_vm_map`/`hv_vm_unmap` side of this
     * split. Existing ELF-guest callers pass 0, so `window->start` alone is
     * still the mapped IPA for them, unchanged from before this parameter
     * existed. */
    windows_mapped = build_mapped_windows(segments, segment_count, page_size, windows, MAX_MAPPED_WINDOWS);
    for (size_t index = 0; index < windows_mapped; ++index) {
        const mapped_window_t *window = &windows[index];
        status = hv_vm_map(
            (uint8_t *)guest_memory + window->start,
            guest_memory_base + window->start,
            (size_t)(window->end - window->start),
            window->flags
        );
        if (status != HV_SUCCESS) {
            windows_mapped = index;
            set_hv_error(error_buffer, error_capacity, "hv_vm_map", status);
            goto cleanup;
        }
    }

    if (create_and_start_vcpu(entry, resolved_cpsr, &vcpu, &exit_data, &vcpu_created, error_buffer, error_capacity)
        != 0) {
        goto cleanup;
    }
    // best-effort: `arm_watchdog`'s own doc on why a failure to arm is
    // never fatal here -- the loop still runs correctly without one, just
    // without the forced-exit diagnostic if this host cannot spawn the
    // thread.
    (void)arm_watchdog(vcpu, watchdog_millis);
    /* arm64 boot protocol (Documentation/arm64/booting): x0 carries the DTB
     * physical address, x1-x3 are reserved and must be zero. Existing
     * ELF-guest callers pass `boot_x0 == 0`, which is inert for them (the
     * lambda guest's raw asm entry reads no incoming register). */
    status = hv_vcpu_set_reg(vcpu, HV_REG_X0, boot_x0);
    if (status == HV_SUCCESS) {
        status = hv_vcpu_set_reg(vcpu, HV_REG_X1, 0);
    }
    if (status == HV_SUCCESS) {
        status = hv_vcpu_set_reg(vcpu, HV_REG_X2, 0);
    }
    if (status == HV_SUCCESS) {
        status = hv_vcpu_set_reg(vcpu, HV_REG_X3, 0);
    }
    if (status != HV_SUCCESS) {
        set_hv_error(error_buffer, error_capacity, "set boot argument registers", status);
        goto cleanup;
    }

    for (;;) {
        status = hv_vcpu_run(vcpu);
        if (status != HV_SUCCESS) {
            set_hv_error(error_buffer, error_capacity, "hv_vcpu_run", status);
            goto cleanup;
        }
        if (!first_exit_seen) {
            first_exit_seen = 1;
            create_to_first_exit_nanos = now_nanos() - run_start_nanos;
        }
        /* M5b-beyond: `HV_EXIT_REASON_VTIMER_ACTIVATED` is exit reason 2, not
         * an EC-0x24/0x18/0x16 exception (`hv_vcpu_types.h`'s own doc, read
         * directly off this host's SDK) -- it fires once, HVF auto-masks the
         * vtimer for us before this exit is even delivered, and the vCPU
         * will not exit with this reason again until
         * `hv_vcpu_set_vtimer_mask(vcpu, false)` is called, which the SDK's
         * own documented contract ties to servicing the guest's EOI of the
         * vtimer's GIC interrupt (the virtual timer PPI, `dtb.rs`'s
         * `write_timer` triple `1 11 4` -- PPI 11 -> INTID 27,
         * `PROXIMA_VM_VTIMER_INTID`). The explicit
         * `hv_vcpu_set_vtimer_mask(vcpu, true)` below is redundant with
         * HVF's own auto-mask -- kept for readability of the contract, not
         * because it changes behavior. The injection this comment used to
         * say never happens now does: record INTID 27 pending in the ICC
         * model's one-deep slot (`proxima_vm_icc_set_vtimer_pending`,
         * `gic.rs`'s `IccCpuInterface::set_pending`) and tell HVF the
         * guest's IRQ line is asserted (`hv_vcpu_set_pending_interrupt`), so
         * the guest takes the IRQ exception once its `PSTATE.I` unmasks --
         * the re-arm half (mask/pending both cleared again) lives in
         * `handle_icc_sysreg_trap`'s own EOIR1 handling below, the
         * documented contract's other end. A guest that never unmasks IRQs
         * or never reads `ICC_IAR1_EL1` still leaves this exit's own wait
         * bounded only by `total_exit_count` below and whichever outer
         * wall-clock bound the caller wraps this call in
         * (`tests/kernel_boot.rs`'s subprocess) -- a genuine parked `wfi`
         * inside `hv_vcpu_run` itself is not preemptable by any counter
         * here. */
        if (exit_data->reason == HV_EXIT_REASON_VTIMER_ACTIVATED) {
            ++vtimer_activation_count;
            status = hv_vcpu_set_vtimer_mask(vcpu, true);
            if (status != HV_SUCCESS) {
                set_hv_error(error_buffer, error_capacity, "hv_vcpu_set_vtimer_mask", status);
                goto cleanup;
            }
            proxima_vm_icc_set_vtimer_pending(icc_transport, PROXIMA_VM_VTIMER_INTID);
            status = hv_vcpu_set_pending_interrupt(vcpu, HV_INTERRUPT_TYPE_IRQ, true);
            if (status != HV_SUCCESS) {
                set_hv_error(error_buffer, error_capacity, "hv_vcpu_set_pending_interrupt", status);
                goto cleanup;
            }
            continue;
        }
        if (exit_data->reason == HV_EXIT_REASON_CANCELED) {
            /* `arm_watchdog`'s own doc: this is that forced exit firing --
             * the guest never trapped on its own for `watchdog_millis`, so
             * report exactly where it was still executing (`PC`/`CPSR`)
             * instead of the bare "unexpected Hypervisor exit reason"
             * message every other unmodeled reason gets, since that message
             * carries no positional evidence at all for this one. */
            uint64_t stuck_pc = 0;
            uint64_t stuck_cpsr = 0;
            hv_vcpu_get_reg(vcpu, HV_REG_PC, &stuck_pc);
            hv_vcpu_get_reg(vcpu, HV_REG_CPSR, &stuck_cpsr);
            if (error_capacity > 0) {
                snprintf(
                    error_buffer,
                    error_capacity,
                    "watchdog forced exit: guest never trapped within the watchdog window, still \
                     executing at pc=0x%llx cpsr=0x%llx",
                    (unsigned long long)stuck_pc,
                    (unsigned long long)stuck_cpsr
                );
            }
            goto cleanup;
        }
        if (exit_data->reason != HV_EXIT_REASON_EXCEPTION) {
            if (error_capacity > 0) {
                snprintf(error_buffer, error_capacity, "unexpected Hypervisor exit reason %u", exit_data->reason);
            }
            goto cleanup;
        }
        /* Bounds every exit shape that DOES keep returning control to this
         * loop (an mmio/exception storm, a psci call that never halts) --
         * the existing `max_hypercalls` check below only counts EC 0x16
         * exits, so a storm on any other exception class was previously
         * unbounded. Sized generously above the deepest real boot this
         * slice has observed (`mmio_trap_count=13540` at the vtimer wall,
         * M5b's own measured evidence) so a real boot never trips it. */
        if (++total_exit_count > MAX_TOTAL_EXITS) {
            if (error_capacity > 0) {
                snprintf(
                    error_buffer,
                    error_capacity,
                    "exceeded total exit budget (%llu) without halting",
                    (unsigned long long)MAX_TOTAL_EXITS
                );
            }
            goto cleanup;
        }
        const uint64_t exception_class = (exit_data->exception.syndrome >> 26u) & 0x3fu;
        if (exception_class == 0x24u) {
            ++mmio_trap_count;
            if (handle_mmio_data_abort(
                    vcpu,
                    exit_data,
                    console_transport,
                    net_transport,
                    blk_transport,
                    gicd_transport,
                    gicr_transport,
                    pl011_transport,
                    (uint8_t *)guest_memory,
                    mapped_size,
                    mmio_emitted_out,
                    mmio_emitted_capacity,
                    &mmio_emitted_length,
                    net_emitted_out,
                    net_emitted_capacity,
                    &net_emitted_length,
                    blk_emitted_out,
                    blk_emitted_capacity,
                    &blk_emitted_length,
                    pl011_emitted_out,
                    pl011_emitted_capacity,
                    &pl011_emitted_length,
                    &gicd_trap_count,
                    &gicr_trap_count,
                    &pl011_trap_count,
                    &virtio_trap_count,
                    error_buffer,
                    error_capacity
                )
                != 0) {
                goto cleanup;
            }
            continue;
        }
        if (exception_class == 0x18u) {
            if (handle_icc_sysreg_trap(vcpu, exit_data, icc_transport, error_buffer, error_capacity) != 0) {
                goto cleanup;
            }
            continue;
        }
        if (exception_class == 0x1u) {
            /* `WFI`/`WFE` is a scheduling hint, not a fault (ARM DDI 0487
             * D13.2.37) -- the correct host action is to treat the trap as
             * a yield and resume the guest, exactly the "advance past the
             * faulting instruction" shape `handle_icc_sysreg_trap` and
             * `handle_mmio_data_abort` already use for their own traps.
             * HVF offers no "block this vCPU until an interrupt is pending"
             * primitive for a WFI-class exit the way it does for
             * `HV_EXIT_REASON_VTIMER_ACTIVATED` above (that path already
             * injects a real IRQ and lets `hv_vcpu_run` itself block); here
             * the guest re-issues the same `wfi` the instant it resumes,
             * which is the expected idle spin, not a bug. `total_exit_count`
             * above already bounds how many times this loop will do that
             * before reporting the budget exceeded, so no separate idle cap
             * is needed. */
            ++wfi_wfe_trap_count;
            uint64_t program_counter = 0;
            status = hv_vcpu_get_reg(vcpu, HV_REG_PC, &program_counter);
            if (status != HV_SUCCESS) {
                set_hv_error(error_buffer, error_capacity, "read wfi/wfe faulting pc", status);
                goto cleanup;
            }
            status = hv_vcpu_set_reg(vcpu, HV_REG_PC, program_counter + 4u);
            if (status != HV_SUCCESS) {
                set_hv_error(error_buffer, error_capacity, "advance wfi/wfe faulting pc", status);
                goto cleanup;
            }
            continue;
        }
        if (exception_class != 0x16u) {
            /* `pc`/`syndrome` ride in the error message itself (not a
             * separate stderr print) so a caller driving a real kernel
             * boot -- which can legitimately trap an unmodeled exception
             * class -- gets the exact faulting address and syndrome back
             * through `ProximaError::Upstream` rather than only the bare
             * exception-class number: the same "next wall must decode
             * itself" evidence this crate's own EC 0x18 handler above was
             * built from. */
            uint64_t faulting_pc = 0;
            hv_vcpu_get_reg(vcpu, HV_REG_PC, &faulting_pc);
            if (error_capacity > 0) {
                snprintf(
                    error_buffer,
                    error_capacity,
                    "unexpected arm exception class 0x%llx at pc=0x%llx syndrome=0x%llx",
                    (unsigned long long)exception_class,
                    (unsigned long long)faulting_pc,
                    (unsigned long long)exit_data->exception.syndrome
                );
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

        /* M5b PSCI (`src/psci.rs`): a raw `hvc` with the SMCCC function ID
         * in x0, args in x1/x2/x3. Disjoint by value from every existing
         * verb (see `dispatch_trampoline.h`'s own doc on this check), so
         * this test runs before the guest's request ever reaches
         * `proxima_vm_dispatch_hypercall`. */
        const int is_psci_32 = verb >= PROXIMA_VM_PSCI_FAST_CALL_32_BASE
            && verb < PROXIMA_VM_PSCI_FAST_CALL_32_BASE + PROXIMA_VM_PSCI_FAST_CALL_RANGE_WIDTH;
        const int is_psci_64 = verb >= PROXIMA_VM_PSCI_FAST_CALL_64_BASE
            && verb < PROXIMA_VM_PSCI_FAST_CALL_64_BASE + PROXIMA_VM_PSCI_FAST_CALL_RANGE_WIDTH;
        if (is_psci_32 || is_psci_64) {
            uint64_t arg3 = 0;
            status = hv_vcpu_get_reg(vcpu, HV_REG_X3, &arg3);
            if (status != HV_SUCCESS) {
                set_hv_error(error_buffer, error_capacity, "read psci arg register", status);
                goto cleanup;
            }
            int64_t psci_return_value = 0;
            uint8_t psci_action = 0;
            (void)proxima_vm_dispatch_psci(
                (uint32_t)verb,
                pointer,
                length,
                arg3,
                &psci_return_value,
                &psci_action
            );
            if (psci_action == 1 || psci_action == 2) {
                result = 0;
                goto cleanup;
            }
            status = hv_vcpu_set_reg(vcpu, HV_REG_X0, (uint64_t)psci_return_value);
            if (status != HV_SUCCESS) {
                set_hv_error(error_buffer, error_capacity, "write psci result register", status);
                goto cleanup;
            }
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
        hv_vm_unmap(guest_memory_base + windows[index].start, (size_t)(windows[index].end - windows[index].start));
    }
    destroy_vm(vm_created);
    if (guest_memory_region_created) {
        proxima_vm_destroy_named_region(&guest_memory_region);
    }
    /* M5b: these five length out-params used to be gated on `result == 0`,
     * which threw away real evidence -- every byte already landed in its
     * `*_emitted_out` buffer regardless of how the loop ended (each channel
     * is filled incrementally as traps are serviced, not staged and
     * committed at a clean halt), so a boot that fails mid-flight still
     * carries whatever bytes it emitted before the failure. Suppressing the
     * length was exactly the gap that made "did earlycon write anything
     * before this boot hit its next wall" unanswerable from the Rust side.
     * The buffer itself was always safe to read up to its true length; only
     * the caller's ability to KNOW that length was gated. */
    if (emitted_length_out != NULL) {
        *emitted_length_out = emitted_length;
    }
    if (mmio_emitted_length_out != NULL) {
        *mmio_emitted_length_out = mmio_emitted_length;
    }
    if (net_emitted_length_out != NULL) {
        *net_emitted_length_out = net_emitted_length;
    }
    if (blk_emitted_length_out != NULL) {
        *blk_emitted_length_out = blk_emitted_length;
    }
    if (pl011_emitted_length_out != NULL) {
        *pl011_emitted_length_out = pl011_emitted_length;
    }
    if (create_to_first_exit_nanos_out != NULL) {
        *create_to_first_exit_nanos_out = create_to_first_exit_nanos;
    }
    if (touch_all_pages_nanos_out != NULL) {
        *touch_all_pages_nanos_out = touch_all_pages_nanos;
    }
    if (mmio_trap_count_out != NULL) {
        *mmio_trap_count_out = mmio_trap_count;
    }
    if (gicd_trap_count_out != NULL) {
        *gicd_trap_count_out = gicd_trap_count;
    }
    if (gicr_trap_count_out != NULL) {
        *gicr_trap_count_out = gicr_trap_count;
    }
    if (pl011_trap_count_out != NULL) {
        *pl011_trap_count_out = pl011_trap_count;
    }
    if (virtio_trap_count_out != NULL) {
        *virtio_trap_count_out = virtio_trap_count;
    }
    if (vtimer_activation_count_out != NULL) {
        *vtimer_activation_count_out = vtimer_activation_count;
    }
    if (wfi_wfe_trap_count_out != NULL) {
        *wfi_wfe_trap_count_out = wfi_wfe_trap_count;
    }
    if (entered_el2_out != NULL) {
        *entered_el2_out = (uint64_t)entered_el2;
    }
    return result;
}

/* ---------------------------------------------------------------------
 * micro-second campaign slice 3 -- CoW-primitives feasibility probe
 * (`probe_cow.h`'s own doc). Every function below is probe-only: never
 * called by `WarmVm`/`restore`/`proxima_vm_run_dispatch_loop`.
 * --------------------------------------------------------------------- */

int proxima_vm_probe_vm_create(char *error_buffer, size_t error_capacity) {
    return create_vm(0, error_buffer, error_capacity);
}

int proxima_vm_probe_create_source(
    size_t size,
    void **host_address_out,
    int *handle_out,
    char *error_buffer,
    size_t error_capacity
) {
    proxima_vm_named_region_t region = {0, NULL, 0};
    if (proxima_vm_create_named_region(size, &region, error_buffer, error_capacity) != 0) {
        return -1;
    }
    memset(region.primary_address, 0xAB, size);
    *host_address_out = region.primary_address;
    *handle_out = region.handle;
    return 0;
}

void proxima_vm_probe_destroy_source(void *host_address, int handle, size_t size) {
    proxima_vm_named_region_t region;
    region.handle = handle;
    region.primary_address = host_address;
    region.mapped_size = size;
    proxima_vm_destroy_named_region(&region);
}

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
) {
    void *previous_view = *previous_view_inout;

    mach_vm_address_t new_address = 0;
    vm_prot_t current_protection = VM_PROT_NONE;
    vm_prot_t max_protection = VM_PROT_NONE;
    uint64_t start = now_nanos();
    kern_return_t status = mach_vm_remap(
        mach_task_self(),
        &new_address,
        (mach_vm_size_t)size,
        0,
        VM_FLAGS_ANYWHERE,
        mach_task_self(),
        (mach_vm_address_t)(uintptr_t)source_host_address,
        TRUE,
        &current_protection,
        &max_protection,
        VM_INHERIT_NONE
    );
    *remap_nanos_out = now_nanos() - start;
    if (status != KERN_SUCCESS) {
        return set_mach_error(error_buffer, error_capacity, "mach_vm_remap", status);
    }

    *hv_vm_unmap_old_nanos_out = 0;
    if (previous_view != NULL) {
        start = now_nanos();
        hv_vm_unmap(0, size);
        *hv_vm_unmap_old_nanos_out = now_nanos() - start;
    }

    start = now_nanos();
    hv_return_t hv_status = hv_vm_map(
        (void *)(uintptr_t)new_address,
        0,
        size,
        HV_MEMORY_READ | HV_MEMORY_WRITE
    );
    *hv_vm_map_nanos_out = now_nanos() - start;
    if (hv_status != HV_SUCCESS) {
        mach_vm_deallocate(mach_task_self(), new_address, (mach_vm_size_t)size);
        return set_hv_error(error_buffer, error_capacity, "hv_vm_map cow view", hv_status);
    }

    *dealloc_old_nanos_out = 0;
    if (previous_view != NULL) {
        start = now_nanos();
        mach_vm_deallocate(mach_task_self(), (mach_vm_address_t)(uintptr_t)previous_view, (mach_vm_size_t)size);
        *dealloc_old_nanos_out = now_nanos() - start;
    }

    *previous_view_inout = (void *)(uintptr_t)new_address;
    return 0;
}

int proxima_vm_probe_first_touch(
    void *view_address,
    size_t page_size,
    size_t page_count,
    uint64_t *nanos_out
) {
    volatile uint8_t *base = (volatile uint8_t *)view_address;
    uint64_t start = now_nanos();
    for (size_t index = 0; index < page_count; ++index) {
        base[index * page_size] = (uint8_t)(index & 0xffu);
    }
    *nanos_out = now_nanos() - start;
    return 0;
}

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
) {
    void *previous_view = *previous_view_inout;

    memory_object_size_t entry_size = (memory_object_size_t)size;
    mach_port_t handle = MACH_PORT_NULL;
    uint64_t start = now_nanos();
    kern_return_t status = mach_make_memory_entry_64(
        mach_task_self(),
        &entry_size,
        (memory_object_offset_t)(uintptr_t)source_host_address,
        MAP_MEM_VM_COPY | VM_PROT_READ | VM_PROT_WRITE,
        &handle,
        MACH_PORT_NULL
    );
    *entry_create_nanos_out = now_nanos() - start;
    *kern_return_out = (int)status;
    if (status != KERN_SUCCESS) {
        return set_mach_error(error_buffer, error_capacity, "mach_make_memory_entry_64 MAP_MEM_VM_COPY", status);
    }

    mach_vm_address_t new_address = 0;
    start = now_nanos();
    status = mach_vm_map(
        mach_task_self(),
        &new_address,
        (mach_vm_size_t)entry_size,
        0,
        VM_FLAGS_ANYWHERE,
        handle,
        0,
        FALSE,
        VM_PROT_READ | VM_PROT_WRITE,
        VM_PROT_READ | VM_PROT_WRITE,
        VM_INHERIT_NONE
    );
    *map_nanos_out = now_nanos() - start;
    mach_port_deallocate(mach_task_self(), handle);
    if (status != KERN_SUCCESS) {
        return set_mach_error(error_buffer, error_capacity, "mach_vm_map vm_copy entry", status);
    }

    *hv_vm_unmap_old_nanos_out = 0;
    if (previous_view != NULL) {
        start = now_nanos();
        hv_vm_unmap(0, size);
        *hv_vm_unmap_old_nanos_out = now_nanos() - start;
    }

    start = now_nanos();
    hv_return_t hv_status = hv_vm_map(
        (void *)(uintptr_t)new_address,
        0,
        size,
        HV_MEMORY_READ | HV_MEMORY_WRITE
    );
    *hv_vm_map_nanos_out = now_nanos() - start;
    if (hv_status != HV_SUCCESS) {
        mach_vm_deallocate(mach_task_self(), new_address, (mach_vm_size_t)size);
        return set_hv_error(error_buffer, error_capacity, "hv_vm_map vm_copy view", hv_status);
    }

    *dealloc_old_nanos_out = 0;
    if (previous_view != NULL) {
        start = now_nanos();
        mach_vm_deallocate(mach_task_self(), (mach_vm_address_t)(uintptr_t)previous_view, (mach_vm_size_t)size);
        *dealloc_old_nanos_out = now_nanos() - start;
    }

    *previous_view_inout = (void *)(uintptr_t)new_address;
    return 0;
}

int proxima_vm_probe_protect_whole(
    uint64_t guest_address,
    size_t size,
    int want_read_only,
    uint64_t *nanos_out,
    char *error_buffer,
    size_t error_capacity
) {
    hv_memory_flags_t flags = want_read_only
        ? HV_MEMORY_READ
        : (HV_MEMORY_READ | HV_MEMORY_WRITE);
    uint64_t start = now_nanos();
    hv_return_t status = hv_vm_protect(guest_address, size, flags);
    *nanos_out = now_nanos() - start;
    if (status != HV_SUCCESS) {
        return set_hv_error(error_buffer, error_capacity, "hv_vm_protect whole region", status);
    }
    return 0;
}

int proxima_vm_probe_protect_per_page(
    uint64_t guest_address,
    size_t granule,
    size_t page_count,
    uint64_t *nanos_out,
    char *error_buffer,
    size_t error_capacity
) {
    for (size_t index = 0; index < page_count; ++index) {
        hv_memory_flags_t flags = (index % 2u == 0u)
            ? HV_MEMORY_READ
            : (HV_MEMORY_READ | HV_MEMORY_WRITE);
        uint64_t start = now_nanos();
        hv_return_t status = hv_vm_protect(guest_address + (uint64_t)(index * granule), granule, flags);
        nanos_out[index] = now_nanos() - start;
        if (status != HV_SUCCESS) {
            return set_hv_error(error_buffer, error_capacity, "hv_vm_protect per-page", status);
        }
    }
    return 0;
}

#define PROXIMA_VM_PROBE_CODE_GUEST_ADDRESS 0x0ull
#define PROXIMA_VM_PROBE_DATA_GUEST_ADDRESS 0x4000ull

int proxima_vm_probe_write_protect_exit(
    uint64_t *checkpoint1_x0_out,
    uint64_t *exception_class_out,
    int *is_data_abort_out,
    int *is_write_out,
    uint8_t *data_byte_after_out,
    uint64_t *protect_nanos_out,
    char *error_buffer,
    size_t error_capacity
) {
    const size_t page_size = (size_t)getpagesize();
    int result = -1;
    int vcpu_created = 0;
    int code_mapped = 0;
    int data_mapped = 0;
    hv_vcpu_t vcpu = 0;
    hv_vcpu_exit_t *exit_data = NULL;
    void *code_memory = MAP_FAILED;
    void *data_memory = MAP_FAILED;

    code_memory = mmap(NULL, page_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (code_memory == MAP_FAILED) {
        return set_error(error_buffer, error_capacity, "map probe code page failed");
    }
    data_memory = mmap(NULL, page_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (data_memory == MAP_FAILED) {
        munmap(code_memory, page_size);
        return set_error(error_buffer, error_capacity, "map probe data page failed");
    }
    memset(data_memory, 0, page_size);

    /* movz x1, #0x2a ; movz x2, #(DATA_GUEST_ADDRESS & 0xffff) ; str x1,[x2] ;
     * movz x0, #1 ; hvc #0 ; movz x1, #0x55 ; str x1,[x2] ; movz x0, #2 ;
     * hvc #0 -- checkpoint 2's `hvc` is only reachable if the second `str`
     * (into the by-then read-only data page) silently succeeded instead of
     * exiting to the host as a data abort. */
    uint32_t *code = (uint32_t *)code_memory;
    code[0] = 0xd2800541u; /* movz x1, #0x2a */
    code[1] = 0xd2800002u | ((uint32_t)(PROXIMA_VM_PROBE_DATA_GUEST_ADDRESS & 0xffffu) << 5u); /* movz x2, #imm16 */
    code[2] = 0xf9000041u; /* str x1, [x2] */
    code[3] = 0xd2800020u; /* movz x0, #1 */
    code[4] = 0xd4000002u; /* hvc #0 */
    code[5] = 0xd2800aa1u; /* movz x1, #0x55 */
    code[6] = 0xf9000041u; /* str x1, [x2] */
    code[7] = 0xd2800040u; /* movz x0, #2 */
    code[8] = 0xd4000002u; /* hvc #0 */

    hv_return_t status = hv_vm_map(
        code_memory,
        PROXIMA_VM_PROBE_CODE_GUEST_ADDRESS,
        page_size,
        HV_MEMORY_READ | HV_MEMORY_WRITE | HV_MEMORY_EXEC
    );
    if (status != HV_SUCCESS) {
        set_hv_error(error_buffer, error_capacity, "hv_vm_map probe code page", status);
        goto cleanup;
    }
    code_mapped = 1;

    status = hv_vm_map(
        data_memory,
        PROXIMA_VM_PROBE_DATA_GUEST_ADDRESS,
        page_size,
        HV_MEMORY_READ | HV_MEMORY_WRITE
    );
    if (status != HV_SUCCESS) {
        set_hv_error(error_buffer, error_capacity, "hv_vm_map probe data page", status);
        goto cleanup;
    }
    data_mapped = 1;

    if (create_and_start_vcpu(
            PROXIMA_VM_PROBE_CODE_GUEST_ADDRESS, 0x3c5u, &vcpu, &exit_data, &vcpu_created, error_buffer, error_capacity
        ) != 0) {
        goto cleanup;
    }

    status = hv_vcpu_run(vcpu);
    if (status != HV_SUCCESS) {
        set_hv_error(error_buffer, error_capacity, "hv_vcpu_run checkpoint 1", status);
        goto cleanup;
    }
    if (exit_data->reason != HV_EXIT_REASON_EXCEPTION) {
        set_error(error_buffer, error_capacity, "checkpoint 1 exit was not an exception");
        goto cleanup;
    }
    {
        uint64_t checkpoint1_class = (exit_data->exception.syndrome >> 26u) & 0x3fu;
        if (checkpoint1_class != 0x16u) {
            set_error(error_buffer, error_capacity, "checkpoint 1 was not the expected hvc trap");
            goto cleanup;
        }
    }
    status = hv_vcpu_get_reg(vcpu, HV_REG_X0, checkpoint1_x0_out);
    if (status != HV_SUCCESS) {
        set_hv_error(error_buffer, error_capacity, "read x0 at checkpoint 1", status);
        goto cleanup;
    }

    {
        uint64_t protect_start = now_nanos();
        status = hv_vm_protect(PROXIMA_VM_PROBE_DATA_GUEST_ADDRESS, page_size, HV_MEMORY_READ);
        *protect_nanos_out = now_nanos() - protect_start;
    }
    if (status != HV_SUCCESS) {
        set_hv_error(error_buffer, error_capacity, "hv_vm_protect probe data page", status);
        goto cleanup;
    }

    status = hv_vcpu_run(vcpu);
    if (status != HV_SUCCESS) {
        set_hv_error(error_buffer, error_capacity, "hv_vcpu_run checkpoint 2", status);
        goto cleanup;
    }
    if (exit_data->reason != HV_EXIT_REASON_EXCEPTION) {
        set_error(error_buffer, error_capacity, "checkpoint 2 exit was not an exception");
        goto cleanup;
    }
    {
        uint64_t checkpoint2_class = (exit_data->exception.syndrome >> 26u) & 0x3fu;
        *exception_class_out = checkpoint2_class;
        *is_data_abort_out = (checkpoint2_class == 0x24u) ? 1 : 0;
        if (*is_data_abort_out) {
            data_abort_iss_t iss = decode_data_abort_iss(exit_data->exception.syndrome);
            *is_write_out = iss.is_write;
        } else {
            *is_write_out = -1;
        }
    }

    *data_byte_after_out = ((uint8_t *)data_memory)[0];
    result = 0;

cleanup:
    destroy_vcpu(vcpu, vcpu_created);
    if (data_mapped) {
        hv_vm_unmap(PROXIMA_VM_PROBE_DATA_GUEST_ADDRESS, page_size);
    }
    if (code_mapped) {
        hv_vm_unmap(PROXIMA_VM_PROBE_CODE_GUEST_ADDRESS, page_size);
    }
    if (code_memory != MAP_FAILED) {
        munmap(code_memory, page_size);
    }
    if (data_memory != MAP_FAILED) {
        munmap(data_memory, page_size);
    }
    return result;
}
