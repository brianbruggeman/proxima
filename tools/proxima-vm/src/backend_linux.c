#include <errno.h>
#include <fcntl.h>
#include <linux/kvm.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#define GUEST_MEMORY_SIZE (2u * 1024u * 1024u)
#define GUEST_CODE_ADDRESS 0x1000u
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

    kvm_fd = open("/dev/kvm", O_RDWR | O_CLOEXEC);
    if (kvm_fd < 0) {
        return set_errno_error(error_buffer, error_capacity, "open /dev/kvm");
    }
    if (ioctl(kvm_fd, KVM_GET_API_VERSION, 0) != KVM_API_VERSION) {
        set_error(error_buffer, error_capacity, "unexpected KVM API version");
        goto cleanup;
    }

    vm_fd = ioctl(kvm_fd, KVM_CREATE_VM, 0);
    if (vm_fd < 0) {
        set_errno_error(error_buffer, error_capacity, "KVM_CREATE_VM");
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

    vcpu_fd = ioctl(vm_fd, KVM_CREATE_VCPU, 0);
    if (vcpu_fd < 0) {
        set_errno_error(error_buffer, error_capacity, "KVM_CREATE_VCPU");
        goto cleanup;
    }

    run_mapping_size = (size_t)ioctl(kvm_fd, KVM_GET_VCPU_MMAP_SIZE, 0);
    if (run_mapping_size == 0 || run_mapping_size == (size_t)-1) {
        set_errno_error(error_buffer, error_capacity, "KVM_GET_VCPU_MMAP_SIZE");
        goto cleanup;
    }
    run_mapping = mmap(NULL, run_mapping_size, PROT_READ | PROT_WRITE, MAP_SHARED, vcpu_fd, 0);
    if (run_mapping == MAP_FAILED) {
        set_errno_error(error_buffer, error_capacity, "map kvm_run");
        goto cleanup;
    }

    struct kvm_sregs special_registers;
    if (ioctl(vcpu_fd, KVM_GET_SREGS, &special_registers) < 0) {
        set_errno_error(error_buffer, error_capacity, "KVM_GET_SREGS");
        goto cleanup;
    }
    special_registers.cs.base = 0;
    special_registers.cs.selector = 0;
    if (ioctl(vcpu_fd, KVM_SET_SREGS, &special_registers) < 0) {
        set_errno_error(error_buffer, error_capacity, "KVM_SET_SREGS");
        goto cleanup;
    }

    struct kvm_regs registers = {
        .rip = GUEST_CODE_ADDRESS,
        .rflags = 2,
    };
    if (ioctl(vcpu_fd, KVM_SET_REGS, &registers) < 0) {
        set_errno_error(error_buffer, error_capacity, "KVM_SET_REGS");
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
    if (run_mapping != MAP_FAILED) {
        munmap(run_mapping, run_mapping_size);
    }
    if (vcpu_fd >= 0) {
        close(vcpu_fd);
    }
    if (guest_memory != MAP_FAILED) {
        munmap(guest_memory, GUEST_MEMORY_SIZE);
    }
    if (vm_fd >= 0) {
        close(vm_fd);
    }
    if (kvm_fd >= 0) {
        close(kvm_fd);
    }
    return result;
}
