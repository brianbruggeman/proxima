#include <errno.h>
#include <fcntl.h>
#include <linux/kvm.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

/* This host's cross-compile kernel header set (zig's bundled headers, same
 * "predates" precedent as this file's own `KVM_MEM_READONLY` fallback
 * above) has neither `linux/memfd.h` nor glibc's `memfd_create` wrapper
 * declaration. `MFD_CLOEXEC`'s bit position and `__NR_memfd_create`'s
 * syscall number are both stable kernel/x86_64 UAPI since Linux 3.17
 * (`include/uapi/linux/memfd.h`, `arch/x86/entry/syscalls/syscall_64.tbl`),
 * so defining them here and calling the syscall directly is safe when the
 * libc surface is missing them. */
#ifndef MFD_CLOEXEC
#define MFD_CLOEXEC 0x0001U
#endif
#ifndef SYS_memfd_create
#define SYS_memfd_create 319
#endif
static int proxima_memfd_create(const char *name, unsigned int flags) {
    return (int)syscall(SYS_memfd_create, name, flags);
}
#define memfd_create proxima_memfd_create

#include "dispatch_trampoline.h"
#include "ffi_segment.h"

#define GUEST_MEMORY_SIZE (2u * 1024u * 1024u)
#define GUEST_CODE_ADDRESS 0x1000u

/* This host's cross-compile `linux/kvm.h` (zig's bundled kernel headers)
 * predates `KVM_MEM_READONLY` — `KVM_MEM_LOG_DIRTY_PAGES` is the only flag
 * it defines. The bit position is stable kernel UAPI
 * (`include/uapi/linux/kvm.h`: `#define KVM_MEM_READONLY (1UL << 3)`, since
 * v3.10), so it is safe to supply here when the header does not. */
#ifndef KVM_MEM_READONLY
#define KVM_MEM_READONLY (1UL << 3)
#endif
#define OUTPUT_PORT 0xe9u

static int set_error(char *error_buffer, size_t error_capacity, const char *message) {
    if (error_capacity > 0) {
        snprintf(error_buffer, error_capacity, "%s", message);
    }
    return -1;
}

static int set_errno_error(char *error_buffer, size_t error_capacity, const char *operation) {
    if (error_capacity > 0) {
        snprintf(error_buffer, error_capacity, "%s: %s", operation, strerror(errno));
    }
    return -1;
}

/* M3 fault-count instrument, KVM mirror of `backend_macos.c`'s helper of
 * the same name. */
static uint64_t now_nanos(void) {
    struct timespec timestamp;
    clock_gettime(CLOCK_MONOTONIC, &timestamp);
    return (uint64_t)timestamp.tv_sec * 1000000000ull + (uint64_t)timestamp.tv_nsec;
}

/* Shared vm-create step of the skeleton `proxima_vm_scratch_run` and
 * `proxima_vm_run_dispatch_loop` both otherwise repeated verbatim: open
 * `/dev/kvm`, check the API version, `KVM_CREATE_VM`. `*kvm_fd_out` is set
 * as soon as `open` succeeds (independent of the later steps) so a caller's
 * cleanup always knows whether to `close` it. Guest memory mapping sits
 * between this and `create_vcpu_and_run_mapping` below — each caller's own
 * shape, so it cannot be folded into either helper. */
static int create_vm(int *kvm_fd_out, int *vm_fd_out, char *error_buffer, size_t error_capacity) {
    *kvm_fd_out = open("/dev/kvm", O_RDWR | O_CLOEXEC);
    if (*kvm_fd_out < 0) {
        return set_errno_error(error_buffer, error_capacity, "open /dev/kvm");
    }
    if (ioctl(*kvm_fd_out, KVM_GET_API_VERSION, 0) != KVM_API_VERSION) {
        return set_error(error_buffer, error_capacity, "unexpected KVM API version");
    }
    *vm_fd_out = ioctl(*kvm_fd_out, KVM_CREATE_VM, 0);
    if (*vm_fd_out < 0) {
        return set_errno_error(error_buffer, error_capacity, "KVM_CREATE_VM");
    }
    return 0;
}

/* Shared vcpu-create-plus-kvm_run-mapping step. `*vcpu_fd_out` is set the
 * moment `KVM_CREATE_VCPU` succeeds — before `KVM_GET_VCPU_MMAP_SIZE` or the
 * `mmap`, which can still fail — so a caller's cleanup always knows whether
 * `close(vcpu_fd)` is owed, even on a partial failure inside this helper. */
static int create_vcpu_and_run_mapping(
    int kvm_fd,
    int vm_fd,
    int *vcpu_fd_out,
    void **run_mapping_out,
    size_t *run_mapping_size_out,
    char *error_buffer,
    size_t error_capacity
) {
    *vcpu_fd_out = ioctl(vm_fd, KVM_CREATE_VCPU, 0);
    if (*vcpu_fd_out < 0) {
        return set_errno_error(error_buffer, error_capacity, "KVM_CREATE_VCPU");
    }

    *run_mapping_size_out = (size_t)ioctl(kvm_fd, KVM_GET_VCPU_MMAP_SIZE, 0);
    if (*run_mapping_size_out == 0 || *run_mapping_size_out == (size_t)-1) {
        return set_errno_error(error_buffer, error_capacity, "KVM_GET_VCPU_MMAP_SIZE");
    }
    *run_mapping_out = mmap(NULL, *run_mapping_size_out, PROT_READ | PROT_WRITE, MAP_SHARED, *vcpu_fd_out, 0);
    if (*run_mapping_out == MAP_FAILED) {
        return set_errno_error(error_buffer, error_capacity, "map kvm_run");
    }
    return 0;
}

/* Shared vcpu register-init step: real-mode-style flat `cs` (base 0,
 * selector 0) plus `rip = entry`, `rflags = 2` (the reserved bit x86 always
 * requires set). Identical between the scratch guest's synthesized code
 * blob and a real ELF entry point — only `entry` differs per caller. */
static int start_vcpu_registers(int vcpu_fd, uint64_t entry, char *error_buffer, size_t error_capacity) {
    struct kvm_sregs special_registers;
    if (ioctl(vcpu_fd, KVM_GET_SREGS, &special_registers) < 0) {
        return set_errno_error(error_buffer, error_capacity, "KVM_GET_SREGS");
    }
    special_registers.cs.base = 0;
    special_registers.cs.selector = 0;
    if (ioctl(vcpu_fd, KVM_SET_SREGS, &special_registers) < 0) {
        return set_errno_error(error_buffer, error_capacity, "KVM_SET_SREGS");
    }

    struct kvm_regs registers = {
        .rip = entry,
        .rflags = 2,
    };
    if (ioctl(vcpu_fd, KVM_SET_REGS, &registers) < 0) {
        return set_errno_error(error_buffer, error_capacity, "KVM_SET_REGS");
    }
    return 0;
}

/* Shared teardown halves, split so `proxima_vm_run_dispatch_loop`'s cleanup
 * order (vcpu resources, then the flat guest-memory unmap, then the vm
 * resources) matches `proxima_vm_scratch_run`'s without forcing an unrelated
 * unmap in between them. */
static void destroy_vcpu(int vcpu_fd, void *run_mapping, size_t run_mapping_size) {
    if (run_mapping != MAP_FAILED) {
        munmap(run_mapping, run_mapping_size);
    }
    if (vcpu_fd >= 0) {
        close(vcpu_fd);
    }
}

static void destroy_vm(int vm_fd, int kvm_fd) {
    if (vm_fd >= 0) {
        close(vm_fd);
    }
    if (kvm_fd >= 0) {
        close(kvm_fd);
    }
}

/* M4 — guest memory as a named object, KVM lane
 * (`tools/proxima-vm/ROADMAP.md`'s M4 section): `memfd_create` allocates an
 * anonymous file with no path -- but, unlike `mmap(MAP_ANON)`, it has a file
 * descriptor identity that CAN be handed to a second mapper (dup'd across a
 * fork, sent over a unix socket, or -- as `proxima_vm_map_named_region`
 * below does in-process -- mapped a second time), so two mappers observe the
 * same physical pages through `MAP_SHARED`. `ftruncate` sizes the anonymous
 * file before the first `mmap`; a fresh `memfd` reads as all-zero, matching
 * `mmap(MAP_ANON)`'s zero-fill contract without an explicit `memset`. */
int proxima_vm_create_named_region(
    size_t size,
    proxima_vm_named_region_t *region_out,
    char *error_buffer,
    size_t error_capacity
) {
    int handle = memfd_create("proxima-vm-guest-memory", MFD_CLOEXEC);
    if (handle < 0) {
        return set_errno_error(error_buffer, error_capacity, "memfd_create");
    }
    if (ftruncate(handle, (off_t)size) != 0) {
        set_errno_error(error_buffer, error_capacity, "ftruncate named region");
        close(handle);
        return -1;
    }

    void *address = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, handle, 0);
    if (address == MAP_FAILED) {
        set_errno_error(error_buffer, error_capacity, "mmap named region primary view");
        close(handle);
        return -1;
    }

    region_out->handle = handle;
    region_out->primary_address = address;
    region_out->mapped_size = size;
    return 0;
}

/* `want_private_view == 0` maps a second `MAP_SHARED` view of the same
 * `memfd`: a write through either view is visible through the other,
 * because both views back onto the same page cache pages. `want_private_view
 * == 1` maps `MAP_PRIVATE`: the kernel gives this view copy-on-write pages
 * against the `memfd`'s current contents at map time -- a write through this
 * view triggers a private copy of just that page, so the write never
 * becomes visible through `region->primary_address` or any other view, and
 * a read before that view's own first write still observes whatever
 * `primary_address` held at map time. This is the M4 exit criterion's
 * "MAP_PRIVATE child ... observes copy-on-write" case. */
int proxima_vm_map_named_region(
    const proxima_vm_named_region_t *region,
    int want_private_view,
    void **host_address_out,
    char *error_buffer,
    size_t error_capacity
) {
    const int mapping_flags = want_private_view ? MAP_PRIVATE : MAP_SHARED;
    void *address = mmap(NULL, region->mapped_size, PROT_READ | PROT_WRITE, mapping_flags, region->handle, 0);
    if (address == MAP_FAILED) {
        return set_errno_error(error_buffer, error_capacity, "mmap second named-region view");
    }
    *host_address_out = address;
    return 0;
}

void proxima_vm_unmap_named_region_view(void *host_address, size_t mapped_size) {
    munmap(host_address, mapped_size);
}

void proxima_vm_destroy_named_region(proxima_vm_named_region_t *region) {
    if (region->primary_address != NULL) {
        proxima_vm_unmap_named_region_view(region->primary_address, region->mapped_size);
    }
    if (region->handle >= 0) {
        close(region->handle);
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
    int result = -1;
    int kvm_fd = -1;
    int vm_fd = -1;
    int vcpu_fd = -1;
    void *guest_memory = MAP_FAILED;
    void *run_mapping = MAP_FAILED;
    size_t run_mapping_size = 0;
    size_t output_length = 0;

    if (message_length > output_capacity) {
        return set_error(error_buffer, error_capacity, "scratch guest output capacity is too small");
    }
    if ((message_length * 4u) + 1u > (GUEST_MEMORY_SIZE - GUEST_CODE_ADDRESS)) {
        return set_error(error_buffer, error_capacity, "scratch guest message does not fit guest memory");
    }

    if (create_vm(&kvm_fd, &vm_fd, error_buffer, error_capacity) != 0) {
        goto cleanup;
    }

    guest_memory = mmap(
        NULL,
        GUEST_MEMORY_SIZE,
        PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANONYMOUS,
        -1,
        0
    );
    if (guest_memory == MAP_FAILED) {
        set_errno_error(error_buffer, error_capacity, "map guest memory");
        goto cleanup;
    }

    uint8_t *code = (uint8_t *)guest_memory + GUEST_CODE_ADDRESS;
    for (size_t index = 0; index < message_length; ++index) {
        *code++ = 0xb0; /* mov al, imm8 */
        *code++ = message[index];
        *code++ = 0xe6; /* out imm8, al */
        *code++ = OUTPUT_PORT;
    }
    *code = 0xf4; /* hlt */

    struct kvm_userspace_memory_region region = {
        .slot = 0,
        .flags = 0,
        .guest_phys_addr = 0,
        .memory_size = GUEST_MEMORY_SIZE,
        .userspace_addr = (uint64_t)(uintptr_t)guest_memory,
    };
    if (ioctl(vm_fd, KVM_SET_USER_MEMORY_REGION, &region) < 0) {
        set_errno_error(error_buffer, error_capacity, "KVM_SET_USER_MEMORY_REGION");
        goto cleanup;
    }

    if (create_vcpu_and_run_mapping(kvm_fd, vm_fd, &vcpu_fd, &run_mapping, &run_mapping_size, error_buffer, error_capacity) != 0) {
        goto cleanup;
    }

    if (start_vcpu_registers(vcpu_fd, GUEST_CODE_ADDRESS, error_buffer, error_capacity) != 0) {
        goto cleanup;
    }

    for (;;) {
        if (ioctl(vcpu_fd, KVM_RUN, 0) < 0) {
            set_errno_error(error_buffer, error_capacity, "KVM_RUN");
            goto cleanup;
        }
        struct kvm_run *run = (struct kvm_run *)run_mapping;
        if (run->exit_reason == KVM_EXIT_HLT) {
            if (output_length != message_length) {
                set_error(error_buffer, error_capacity, "scratch guest halted before emitting declared output");
                goto cleanup;
            }
            result = 0;
            goto cleanup;
        }
        if (run->exit_reason != KVM_EXIT_IO
            || run->io.direction != KVM_EXIT_IO_OUT
            || run->io.size != 1
            || run->io.port != OUTPUT_PORT) {
            if (error_capacity > 0) {
                snprintf(error_buffer, error_capacity, "unexpected KVM exit reason %u", run->exit_reason);
            }
            goto cleanup;
        }

        const uint8_t *emitted = (const uint8_t *)run_mapping + run->io.data_offset;
        for (uint32_t index = 0; index < run->io.count; ++index) {
            if (output_length >= output_capacity) {
                set_error(error_buffer, error_capacity, "scratch guest emitted more bytes than declared");
                goto cleanup;
            }
            output[output_length++] = emitted[index];
        }
    }

cleanup:
    destroy_vcpu(vcpu_fd, run_mapping, run_mapping_size);
    if (guest_memory != MAP_FAILED) {
        munmap(guest_memory, GUEST_MEMORY_SIZE);
    }
    destroy_vm(vm_fd, kvm_fd);
    return result;
}

static size_t round_up_to_page(size_t value, size_t page_size) {
    return ((value + page_size - 1u) / page_size) * page_size;
}

/* Reads `rax, rbx, rcx, rdx, rsi, rdi, rsp, rbp, r8..r15` (in `struct
 * kvm_regs`'s own field order) into `registers_out->gpr[0..16)`, zero-
 * padding `gpr[16..31)` -- the aarch64 lane's register file is 31 GPRs wide,
 * x86_64's is 16, and `proxima_vm_registers_t` is sized for the wider one
 * (`ffi_segment.h`'s own doc on the struct). `pc`/`flags` carry `rip`/
 * `rflags`. */
static int capture_registers(int vcpu_fd, proxima_vm_registers_t *registers_out, char *error_buffer, size_t error_capacity) {
    struct kvm_regs registers;
    if (ioctl(vcpu_fd, KVM_GET_REGS, &registers) < 0) {
        return set_errno_error(error_buffer, error_capacity, "KVM_GET_REGS");
    }
    memset(registers_out->gpr, 0, sizeof(registers_out->gpr));
    registers_out->gpr[0] = registers.rax;
    registers_out->gpr[1] = registers.rbx;
    registers_out->gpr[2] = registers.rcx;
    registers_out->gpr[3] = registers.rdx;
    registers_out->gpr[4] = registers.rsi;
    registers_out->gpr[5] = registers.rdi;
    registers_out->gpr[6] = registers.rsp;
    registers_out->gpr[7] = registers.rbp;
    registers_out->gpr[8] = registers.r8;
    registers_out->gpr[9] = registers.r9;
    registers_out->gpr[10] = registers.r10;
    registers_out->gpr[11] = registers.r11;
    registers_out->gpr[12] = registers.r12;
    registers_out->gpr[13] = registers.r13;
    registers_out->gpr[14] = registers.r14;
    registers_out->gpr[15] = registers.r15;
    registers_out->pc = registers.rip;
    registers_out->flags = registers.rflags;
    return 0;
}

/* The exact inverse of `capture_registers` above, called by
 * `proxima_vm_scratch_restore` in place of `start_vcpu_registers`'s own
 * fixed `entry`/`rflags = 2` reset values. */
static int restore_registers(int vcpu_fd, const proxima_vm_registers_t *registers_in, char *error_buffer, size_t error_capacity) {
    struct kvm_sregs special_registers;
    if (ioctl(vcpu_fd, KVM_GET_SREGS, &special_registers) < 0) {
        return set_errno_error(error_buffer, error_capacity, "KVM_GET_SREGS");
    }
    special_registers.cs.base = 0;
    special_registers.cs.selector = 0;
    if (ioctl(vcpu_fd, KVM_SET_SREGS, &special_registers) < 0) {
        return set_errno_error(error_buffer, error_capacity, "KVM_SET_SREGS");
    }

    struct kvm_regs registers = {
        .rax = registers_in->gpr[0],
        .rbx = registers_in->gpr[1],
        .rcx = registers_in->gpr[2],
        .rdx = registers_in->gpr[3],
        .rsi = registers_in->gpr[4],
        .rdi = registers_in->gpr[5],
        .rsp = registers_in->gpr[6],
        .rbp = registers_in->gpr[7],
        .r8 = registers_in->gpr[8],
        .r9 = registers_in->gpr[9],
        .r10 = registers_in->gpr[10],
        .r11 = registers_in->gpr[11],
        .r12 = registers_in->gpr[12],
        .r13 = registers_in->gpr[13],
        .r14 = registers_in->gpr[14],
        .r15 = registers_in->gpr[15],
        .rip = registers_in->pc,
        .rflags = registers_in->flags,
    };
    if (ioctl(vcpu_fd, KVM_SET_REGS, &registers) < 0) {
        return set_errno_error(error_buffer, error_capacity, "KVM_SET_REGS");
    }
    return 0;
}

/* Fixed, unlike the HVF lane's own message-length-dependent sizing: this
 * lane's scratch guest always reserves the full `GUEST_MEMORY_SIZE`
 * (`proxima_vm_scratch_run`'s own fixed allocation, never message-sized). */
size_t proxima_vm_scratch_guest_memory_size(size_t message_length) {
    (void)message_length;
    return GUEST_MEMORY_SIZE;
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
    int result = -1;
    int kvm_fd = -1;
    int vm_fd = -1;
    int vcpu_fd = -1;
    void *run_mapping = MAP_FAILED;
    size_t run_mapping_size = 0;
    size_t output_length = 0;
    proxima_vm_named_region_t region = {0, NULL, 0};
    int region_created = 0;

    if (message_length > output_capacity) {
        return set_error(error_buffer, error_capacity, "scratch guest output capacity is too small");
    }
    if ((message_length * 4u) + 1u > (GUEST_MEMORY_SIZE - GUEST_CODE_ADDRESS)) {
        return set_error(error_buffer, error_capacity, "scratch guest message does not fit guest memory");
    }
    if (GUEST_MEMORY_SIZE > guest_memory_capacity) {
        return set_error(error_buffer, error_capacity, "guest_memory_out capacity is too small");
    }

    if (create_vm(&kvm_fd, &vm_fd, error_buffer, error_capacity) != 0) {
        goto cleanup;
    }

    if (proxima_vm_create_named_region(GUEST_MEMORY_SIZE, &region, error_buffer, error_capacity) != 0) {
        goto cleanup;
    }
    region_created = 1;

    uint8_t *code = (uint8_t *)region.primary_address + GUEST_CODE_ADDRESS;
    for (size_t index = 0; index < message_length; ++index) {
        *code++ = 0xb0; /* mov al, imm8 */
        *code++ = message[index];
        *code++ = 0xe6; /* out imm8, al */
        *code++ = OUTPUT_PORT;
    }
    *code = 0xf4; /* hlt */

    struct kvm_userspace_memory_region kvm_region = {
        .slot = 0,
        .flags = 0,
        .guest_phys_addr = 0,
        .memory_size = GUEST_MEMORY_SIZE,
        .userspace_addr = (uint64_t)(uintptr_t)region.primary_address,
    };
    if (ioctl(vm_fd, KVM_SET_USER_MEMORY_REGION, &kvm_region) < 0) {
        set_errno_error(error_buffer, error_capacity, "KVM_SET_USER_MEMORY_REGION");
        goto cleanup;
    }

    if (create_vcpu_and_run_mapping(kvm_fd, vm_fd, &vcpu_fd, &run_mapping, &run_mapping_size, error_buffer, error_capacity) != 0) {
        goto cleanup;
    }

    if (start_vcpu_registers(vcpu_fd, GUEST_CODE_ADDRESS, error_buffer, error_capacity) != 0) {
        goto cleanup;
    }

    for (;;) {
        if (ioctl(vcpu_fd, KVM_RUN, 0) < 0) {
            set_errno_error(error_buffer, error_capacity, "KVM_RUN");
            goto cleanup;
        }
        struct kvm_run *run = (struct kvm_run *)run_mapping;
        if (run->exit_reason == KVM_EXIT_HLT) {
            if (output_length != message_length) {
                set_error(error_buffer, error_capacity, "scratch guest halted before emitting declared output");
                goto cleanup;
            }
            if (capture_registers(vcpu_fd, registers_out, error_buffer, error_capacity) != 0) {
                goto cleanup;
            }
            /* `hlt` retires before the exit, leaving `rip` past the one-byte
             * opcode -- the same "nothing legal to resume into" gap the HVF
             * lane's own `capture_registers` call site rewinds, and for the
             * identical reason: this guest has no code beyond its halting
             * instruction. */
            registers_out->pc -= 1u;
            memcpy(guest_memory_out, region.primary_address, GUEST_MEMORY_SIZE);
            result = 0;
            goto cleanup;
        }
        if (run->exit_reason != KVM_EXIT_IO
            || run->io.direction != KVM_EXIT_IO_OUT
            || run->io.size != 1
            || run->io.port != OUTPUT_PORT) {
            if (error_capacity > 0) {
                snprintf(error_buffer, error_capacity, "unexpected KVM exit reason %u", run->exit_reason);
            }
            goto cleanup;
        }

        const uint8_t *emitted = (const uint8_t *)run_mapping + run->io.data_offset;
        for (uint32_t index = 0; index < run->io.count; ++index) {
            if (output_length >= output_capacity) {
                set_error(error_buffer, error_capacity, "scratch guest emitted more bytes than declared");
                goto cleanup;
            }
            output[output_length++] = emitted[index];
        }
    }

cleanup:
    destroy_vcpu(vcpu_fd, run_mapping, run_mapping_size);
    if (region_created) {
        proxima_vm_destroy_named_region(&region);
    }
    destroy_vm(vm_fd, kvm_fd);
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
    char *error_buffer,
    size_t error_capacity
) {
    const uint64_t restore_start_nanos = now_nanos();
    const size_t stride = page_size > 0 ? page_size : (size_t)sysconf(_SC_PAGESIZE);
    int result = -1;
    int kvm_fd = -1;
    int vm_fd = -1;
    int vcpu_fd = -1;
    void *run_mapping = MAP_FAILED;
    size_t run_mapping_size = 0;
    proxima_vm_named_region_t region = {0, NULL, 0};
    int region_created = 0;

    *resumed_ok_out = 0;
    *fault_count_out = 0;

    if (create_vm(&kvm_fd, &vm_fd, error_buffer, error_capacity) != 0) {
        return -1;
    }

    if (proxima_vm_create_named_region(guest_memory_length, &region, error_buffer, error_capacity) != 0) {
        goto cleanup;
    }
    region_created = 1;

    /* `page_size`-strided restore copy -- the KVM lane's mirror of the M3
     * "wall to touch every mapped page" measurement, over restored bytes
     * instead of zeros (identical shape to the HVF lane's own
     * `proxima_vm_scratch_restore`). */
    {
        const uint64_t touch_start_nanos = now_nanos();
        uint8_t *destination = (uint8_t *)region.primary_address;
        for (size_t offset = 0; offset < guest_memory_length; offset += stride) {
            const size_t chunk = (guest_memory_length - offset) < stride ? (guest_memory_length - offset) : stride;
            memcpy(destination + offset, guest_memory_in + offset, chunk);
        }
        *touch_all_pages_nanos_out = now_nanos() - touch_start_nanos;
    }

    struct kvm_userspace_memory_region kvm_region = {
        .slot = 0,
        .flags = 0,
        .guest_phys_addr = 0,
        .memory_size = (uint64_t)region.mapped_size,
        .userspace_addr = (uint64_t)(uintptr_t)region.primary_address,
    };
    if (ioctl(vm_fd, KVM_SET_USER_MEMORY_REGION, &kvm_region) < 0) {
        set_errno_error(error_buffer, error_capacity, "KVM_SET_USER_MEMORY_REGION");
        goto cleanup;
    }

    if (create_vcpu_and_run_mapping(kvm_fd, vm_fd, &vcpu_fd, &run_mapping, &run_mapping_size, error_buffer, error_capacity) != 0) {
        goto cleanup;
    }

    if (restore_registers(vcpu_fd, registers_in, error_buffer, error_capacity) != 0) {
        goto cleanup;
    }
    *restore_wall_nanos_out = now_nanos() - restore_start_nanos;

    /* Resume exactly once: the snapshot was captured at the guest's own
     * halting trap, rewound to point at the `hlt` byte itself, so this
     * resumed step re-executes the identical instruction and re-exits
     * `KVM_EXIT_HLT` -- the proof that restore reproduced the exact guest
     * state, not merely copied bytes. */
    if (ioctl(vcpu_fd, KVM_RUN, 0) < 0) {
        set_errno_error(error_buffer, error_capacity, "KVM_RUN resumed step");
        goto cleanup;
    }
    {
        struct kvm_run *run = (struct kvm_run *)run_mapping;
        if (run->exit_reason == KVM_EXIT_HLT) {
            struct kvm_regs resumed_registers;
            if (ioctl(vcpu_fd, KVM_GET_REGS, &resumed_registers) < 0) {
                set_errno_error(error_buffer, error_capacity, "read resumed output register");
                goto cleanup;
            }
            *resumed_x0_out = resumed_registers.rax;
            *resumed_ok_out = 1;
        }
    }
    result = 0;

cleanup:
    destroy_vcpu(vcpu_fd, run_mapping, run_mapping_size);
    if (region_created) {
        proxima_vm_destroy_named_region(&region);
    }
    destroy_vm(vm_fd, kvm_fd);
    return result;
}

/* Unwinds every already-mapped entry in `mapped_out[0..count)` -- the same
 * cleanup `proxima_vm_unmap_guest_memory` performs, factored out so a
 * partial failure inside the mapping loop unwinds exactly what succeeded
 * before it. KVM has no per-slot "unmap" ioctl; a zero-sized
 * `KVM_SET_USER_MEMORY_REGION` for the same slot retires it, which is what
 * releases the guest_phys_addr range back to the VM. */
static void unwind_mapped_segments(int vm_fd, proxima_vm_mapped_segment_t *mapped_out, size_t count) {
    for (size_t index = 0; index < count; ++index) {
        struct kvm_userspace_memory_region retire = {
            .slot = (uint32_t)index,
            .flags = 0,
            .guest_phys_addr = mapped_out[index].guest_address,
            .memory_size = 0,
            .userspace_addr = (uint64_t)(uintptr_t)mapped_out[index].host_address,
        };
        ioctl(vm_fd, KVM_SET_USER_MEMORY_REGION, &retire);
        munmap(mapped_out[index].host_address, mapped_out[index].mapped_size);
    }
}

int proxima_vm_map_guest_memory(
    const proxima_vm_segment_t *segments,
    size_t segment_count,
    proxima_vm_mapped_segment_t *mapped_out,
    int *kvm_fd_out,
    int *vm_fd_out,
    char *error_buffer,
    size_t error_capacity
) {
    const size_t page_size = (size_t)getpagesize();
    int kvm_fd = open("/dev/kvm", O_RDWR | O_CLOEXEC);
    if (kvm_fd < 0) {
        return set_errno_error(error_buffer, error_capacity, "open /dev/kvm");
    }
    if (ioctl(kvm_fd, KVM_GET_API_VERSION, 0) != KVM_API_VERSION) {
        set_error(error_buffer, error_capacity, "unexpected KVM API version");
        close(kvm_fd);
        return -1;
    }

    int vm_fd = ioctl(kvm_fd, KVM_CREATE_VM, 0);
    if (vm_fd < 0) {
        set_errno_error(error_buffer, error_capacity, "KVM_CREATE_VM");
        close(kvm_fd);
        return -1;
    }

    for (size_t index = 0; index < segment_count; ++index) {
        const proxima_vm_segment_t *segment = &segments[index];
        const size_t mapped_size = round_up_to_page(
            segment->memory_size > 0 ? segment->memory_size : 1,
            page_size
        );

        void *host_address = mmap(NULL, mapped_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (host_address == MAP_FAILED) {
            unwind_mapped_segments(vm_fd, mapped_out, index);
            close(vm_fd);
            close(kvm_fd);
            return set_errno_error(error_buffer, error_capacity, "map guest segment memory failed");
        }
        memcpy(host_address, segment->data, segment->data_length);

        struct kvm_userspace_memory_region region = {
            .slot = (uint32_t)index,
            .flags = 0,
            .guest_phys_addr = segment->guest_address,
            .memory_size = mapped_size,
            .userspace_addr = (uint64_t)(uintptr_t)host_address,
        };
        if (ioctl(vm_fd, KVM_SET_USER_MEMORY_REGION, &region) < 0) {
            munmap(host_address, mapped_size);
            unwind_mapped_segments(vm_fd, mapped_out, index);
            close(vm_fd);
            close(kvm_fd);
            return set_errno_error(error_buffer, error_capacity, "KVM_SET_USER_MEMORY_REGION");
        }

        mapped_out[index].guest_address = segment->guest_address;
        mapped_out[index].host_address = host_address;
        mapped_out[index].mapped_size = mapped_size;
    }

    *kvm_fd_out = kvm_fd;
    *vm_fd_out = vm_fd;
    return 0;
}

void proxima_vm_unmap_guest_memory(
    const proxima_vm_mapped_segment_t *mapped,
    size_t mapped_count,
    int kvm_fd,
    int vm_fd
) {
    for (size_t index = 0; index < mapped_count; ++index) {
        munmap(mapped[index].host_address, mapped[index].mapped_size);
    }
    if (vm_fd >= 0) {
        close(vm_fd);
    }
    if (kvm_fd >= 0) {
        close(kvm_fd);
    }
}

#define DISPATCH_RESPONSE_SCRATCH_CAPACITY 512u
#define MAX_MAPPED_WINDOWS 16u

typedef struct {
    uint64_t start;
    uint64_t end;
    int writable;
} mapped_window_t;

/* `KVM_SET_USER_MEMORY_REGION` only accepts page-aligned `(guest_phys_addr,
 * memory_size)` pairs, so two segments whose BYTE ranges are disjoint
 * (already proven by `elf::parse_elf`) can still land in the same PAGE once
 * rounded — this guest's own `.text` (291 bytes on x86_64) and `.rodata`
 * (12 bytes) both sit inside the first 4 KiB page. Merging same-page
 * segments into one window — `writable` true if ANY contributing segment is
 * writable — mirrors `backend_macos.c`'s `build_mapped_windows`, restricted
 * to the one permission axis this API can express per region
 * (`KVM_MEM_READONLY`; see `proxima_vm_run_dispatch_loop`'s own doc for why
 * a per-window exec bit has no equivalent here). `segments` is sorted by
 * `guest_address` first (insertion sort; `segment_count` is single digits,
 * never a hot path) so the merge is a single linear pass over sorted
 * intervals. Returns the window count, capped at `windows_capacity`. */
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
        const int writable = segment->writable ? 1 : 0;

        if (window_count > 0 && start <= windows_out[window_count - 1].end) {
            mapped_window_t *last = &windows_out[window_count - 1];
            if (end > last->end) {
                last->end = end;
            }
            last->writable = last->writable || writable;
        } else if (window_count < windows_capacity) {
            windows_out[window_count].start = start;
            windows_out[window_count].end = end;
            windows_out[window_count].writable = writable;
            window_count += 1;
        }
    }
    return window_count;
}

/* KVM mirror of `backend_macos.c`'s `proxima_vm_run_dispatch_loop`: the
 * `out dx, al` trap (`guests/lambda/src/hypercall.rs`'s x86_64 arm) surfaces
 * as `KVM_EXIT_IO` with the verb in `run->io.port` (`dx` IS the port
 * register for `out dx, al`) -- but `KVM_EXIT_IO` carries only the I/O
 * event itself, not the general-purpose registers the ABI's `pointer`/
 * `length` live in (`rdi`/`rsi`), so each exit needs an explicit
 * `KVM_GET_REGS` before it can call `proxima_vm_dispatch_hypercall`.
 *
 * UNTESTED ON REAL HARDWARE: this host has no `/dev/kvm` (darwin/aarch64);
 * `open("/dev/kvm", ...)` fails immediately and the function returns -1 by
 * the same path `proxima_vm_scratch_run` already takes on this host. Only
 * compilation is proven here, not execution.
 *
 * `console_transport`, `net_transport`, `blk_transport`, `gicd_transport`,
 * `gicr_transport`, `pl011_transport`, and `icc_transport` mirror
 * `backend_macos.c`'s parameters of the same names, but this x86_64 mirror
 * does not yet decode `KVM_EXIT_MMIO` into `proxima_vm_dispatch_mmio`/
 * `proxima_vm_dispatch_mmio_net`/`proxima_vm_dispatch_mmio_blk`/
 * `proxima_vm_dispatch_mmio_gicd`/`proxima_vm_dispatch_mmio_gicr`/
 * `proxima_vm_dispatch_mmio_pl011` calls, nor the EC-0x18-equivalent trap KVM
 * would report for a trapped `MSR`/`MRS` into `proxima_vm_dispatch_sysreg_icc`
 * (M6 slices 3, 5, and 6, M5b's GIC slice, M5b's pl011 slice, and M5b's ICC
 * slice all land the aarch64/HVF path only, per each task's own scope; the
 * GIC, pl011, and ICC models are ARM-specific and have no x86_64 meaning at
 * all besides keeping this signature in lockstep with the shared header);
 * the parameters exist so this function's signature matches the shared
 * header (`dispatch_trampoline.h`) `dispatch.rs`'s FFI declaration links
 * against, and are otherwise unused here. */
int proxima_vm_run_dispatch_loop(
    const proxima_vm_segment_t *segments,
    size_t segment_count,
    uint64_t guest_memory_size,
    uint64_t guest_memory_base,
    uint64_t entry,
    uint64_t boot_x0,
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
    /* x86_64/KVM has no arm64 boot protocol (no DTB register argument), so
     * `boot_x0`/`boot_cpsr` are inert on this lane -- kept only to match
     * the shared header every other lockstep parameter here already
     * follows. Same for the M5b per-window/vtimer/wfi/el2 counters below:
     * x86_64 has no GICv3/pl011/ARM-vtimer/ARM-WFI-trap/ARM-EL2 concept at
     * all (x86's `hlt` is a different KVM exit shape entirely, and x86_64
     * has no ARM exception-level model), so this lane always reports these
     * as 0 rather than decode a KVM exit shape that has no ARM analogue. */
    (void)boot_cpsr;
    // no cross-thread forced-exit diagnostic on this lane yet -- `KVM_RUN`'s
    // own signal-based interruption shape differs from HVF's `hv_vcpus_exit`
    // and this lane has no live guest to hang against on this host anyway
    // (`proxima_vm_run_dispatch_loop`'s own doc: `open("/dev/kvm", ...)`
    // fails immediately here).
    (void)watchdog_millis;
    (void)boot_x0;
    (void)console_transport;
    (void)net_transport;
    (void)blk_transport;
    (void)gicd_transport;
    (void)gicr_transport;
    (void)pl011_transport;
    (void)icc_transport;
    if (gicd_trap_count_out != NULL) {
        *gicd_trap_count_out = 0;
    }
    if (gicr_trap_count_out != NULL) {
        *gicr_trap_count_out = 0;
    }
    if (pl011_trap_count_out != NULL) {
        *pl011_trap_count_out = 0;
    }
    if (virtio_trap_count_out != NULL) {
        *virtio_trap_count_out = 0;
    }
    if (vtimer_activation_count_out != NULL) {
        *vtimer_activation_count_out = 0;
    }
    if (wfi_wfe_trap_count_out != NULL) {
        *wfi_wfe_trap_count_out = 0;
    }
    if (entered_el2_out != NULL) {
        *entered_el2_out = 0;
    }
    (void)mmio_emitted_out;
    (void)mmio_emitted_capacity;
    (void)mmio_emitted_length_out;
    (void)net_emitted_out;
    (void)net_emitted_capacity;
    (void)net_emitted_length_out;
    (void)blk_emitted_out;
    (void)blk_emitted_capacity;
    (void)blk_emitted_length_out;
    (void)pl011_emitted_out;
    (void)pl011_emitted_capacity;
    (void)pl011_emitted_length_out;
    const uint64_t run_start_nanos = now_nanos();
    int result = -1;
    int kvm_fd = -1;
    int vm_fd = -1;
    int vcpu_fd = -1;
    int first_exit_seen = 0;
    size_t windows_mapped = 0;
    mapped_window_t windows[MAX_MAPPED_WINDOWS];
    proxima_vm_named_region_t guest_memory_region = {-1, NULL, 0};
    int guest_memory_region_created = 0;
    void *guest_memory = MAP_FAILED;
    void *run_mapping = MAP_FAILED;
    size_t run_mapping_size = 0;
    size_t emitted_length = 0;
    size_t hypercall_count = 0;
    /* KVM_EXIT_MMIO is not yet decoded on this lane (see this function's
     * own doc), so this stays 0 for every run rather than a fabricated
     * count — the M3 fault-count column belongs to the pagemap soft-dirty
     * index (`ROADMAP.md`'s M3 section), unimplemented here. */
    uint64_t mmio_trap_count = 0;
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

    if (create_vm(&kvm_fd, &vm_fd, error_buffer, error_capacity) != 0) {
        goto cleanup;
    }

    /* M4: guest memory is a named `memfd`-backed object, not
     * `mmap(MAP_ANON)` -- a second caller holding `guest_memory_region.handle`
     * can map its own view of the same backing file and observe writes made
     * through this one, or a `MAP_PRIVATE` copy-on-write view
     * (`proxima_vm_map_named_region`), neither of which an anonymous mapping
     * could ever offer a snapshot/fork consumer. */
    if (proxima_vm_create_named_region(guest_memory_size, &guest_memory_region, error_buffer, error_capacity) != 0) {
        goto cleanup;
    }
    guest_memory_region_created = 1;
    guest_memory = guest_memory_region.primary_address;

    /* M3's "wall to touch every mapped page" — see `backend_macos.c`'s
     * mirror of this block for the full rationale (timed before the
     * `KVM_SET_USER_MEMORY_REGION` loop, host-mmap first-touch cost only). */
    {
        const size_t page_size = (size_t)getpagesize();
        const uint64_t touch_start_nanos = now_nanos();
        for (size_t offset = 0; offset < guest_memory_size; offset += page_size) {
            ((volatile uint8_t *)guest_memory)[offset] = 0u;
        }
        touch_all_pages_nanos = now_nanos() - touch_start_nanos;
    }

    /* Copy every segment's file-backed bytes into its own `guest_address`
     * offset of the one flat `guest_memory` host allocation first (segments
     * never overlap — `elf::parse_elf` already proved that), THEN map real,
     * page-merged permissions (`build_mapped_windows`) — one
     * `KVM_SET_USER_MEMORY_REGION` slot per page-aligned window, flagged
     * `KVM_MEM_READONLY` only when no segment sharing that window is
     * writable. This is the one axis `struct
     * kvm_userspace_memory_region.flags` can express (`linux/kvm.h`
     * carries only `KVM_MEM_LOG_DIRTY_PAGES` and `KVM_MEM_READONLY` — no
     * execute bit); a bare-metal guest with paging disabled has no page
     * tables of its own for a hardware NX bit to key off, so per-window
     * execute permission stays real KVM/ISA territory this loader cannot
     * reach, unlike `backend_macos.c`'s `hv_vm_map`. */
    for (size_t index = 0; index < segment_count; ++index) {
        const proxima_vm_segment_t *segment = &segments[index];
        if (segment->data_length > 0) {
            memcpy((uint8_t *)guest_memory + segment->guest_address, segment->data, segment->data_length);
        }
    }

    windows_mapped = build_mapped_windows(segments, segment_count, (size_t)getpagesize(), windows, MAX_MAPPED_WINDOWS);
    for (size_t index = 0; index < windows_mapped; ++index) {
        const mapped_window_t *window = &windows[index];
        struct kvm_userspace_memory_region region = {
            .slot = (uint32_t)index,
            .flags = window->writable ? 0u : (uint32_t)KVM_MEM_READONLY,
            .guest_phys_addr = guest_memory_base + window->start,
            .memory_size = window->end - window->start,
            .userspace_addr = (uint64_t)(uintptr_t)guest_memory + window->start,
        };
        if (ioctl(vm_fd, KVM_SET_USER_MEMORY_REGION, &region) < 0) {
            windows_mapped = index;
            set_errno_error(error_buffer, error_capacity, "KVM_SET_USER_MEMORY_REGION");
            goto cleanup;
        }
    }

    if (create_vcpu_and_run_mapping(kvm_fd, vm_fd, &vcpu_fd, &run_mapping, &run_mapping_size, error_buffer, error_capacity) != 0) {
        goto cleanup;
    }

    if (start_vcpu_registers(vcpu_fd, entry, error_buffer, error_capacity) != 0) {
        goto cleanup;
    }

    for (;;) {
        if (ioctl(vcpu_fd, KVM_RUN, 0) < 0) {
            set_errno_error(error_buffer, error_capacity, "KVM_RUN");
            goto cleanup;
        }
        if (!first_exit_seen) {
            first_exit_seen = 1;
            create_to_first_exit_nanos = now_nanos() - run_start_nanos;
        }
        struct kvm_run *run = (struct kvm_run *)run_mapping;
        if (run->exit_reason != KVM_EXIT_IO || run->io.direction != KVM_EXIT_IO_OUT) {
            if (error_capacity > 0) {
                snprintf(error_buffer, error_capacity, "unexpected KVM exit reason %u", run->exit_reason);
            }
            goto cleanup;
        }

        if (++hypercall_count > max_hypercalls) {
            set_error(error_buffer, error_capacity, "guest exceeded hypercall budget without halting");
            goto cleanup;
        }

        const uint64_t verb = (uint64_t)run->io.port;

        struct kvm_regs call_registers;
        if (ioctl(vcpu_fd, KVM_GET_REGS, &call_registers) < 0) {
            set_errno_error(error_buffer, error_capacity, "read hypercall registers");
            goto cleanup;
        }
        const uint64_t pointer = call_registers.rdi;
        const uint64_t length = call_registers.rsi;

        if (verb == PROXIMA_VM_HALT_VERB) {
            result = 0;
            goto cleanup;
        }

        if (verb == PROXIMA_VM_EMIT_VERB) {
            if (pointer >= guest_memory_size || emitted_length >= emitted_capacity) {
                set_error(error_buffer, error_capacity, "emit hypercall pointer or output capacity out of range");
                goto cleanup;
            }
            emitted_out[emitted_length++] = ((const uint8_t *)guest_memory)[pointer];
            continue;
        }

        const int64_t dispatched = proxima_vm_dispatch_hypercall(
            dispatcher,
            (const uint8_t *)guest_memory,
            guest_memory_size,
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
        const size_t writable = response_length < length ? response_length : (size_t)length;
        if (pointer + writable > guest_memory_size) {
            set_error(error_buffer, error_capacity, "dispatch response write-back would overrun guest memory");
            goto cleanup;
        }
        memcpy((uint8_t *)guest_memory + pointer, response_scratch, writable);

        call_registers.rax = (uint64_t)response_length;
        if (ioctl(vcpu_fd, KVM_SET_REGS, &call_registers) < 0) {
            set_errno_error(error_buffer, error_capacity, "write hypercall result register");
            goto cleanup;
        }
    }

cleanup:
    destroy_vcpu(vcpu_fd, run_mapping, run_mapping_size);
    if (guest_memory_region_created) {
        proxima_vm_destroy_named_region(&guest_memory_region);
    }
    destroy_vm(vm_fd, kvm_fd);
    if (result == 0 && emitted_length_out != NULL) {
        *emitted_length_out = emitted_length;
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
    return result;
}
