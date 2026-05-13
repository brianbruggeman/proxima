#include <Hypervisor/Hypervisor.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/mman.h>
#include <unistd.h>

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

    hv_return_t status = hv_vm_create(NULL);
    if (status != HV_SUCCESS) {
        set_hv_error(error_buffer, error_capacity, "hv_vm_create", status);
        goto cleanup;
    }
    vm_created = 1;

    status = hv_vm_map(
        guest_memory,
        0,
        memory_size,
        HV_MEMORY_READ | HV_MEMORY_WRITE | HV_MEMORY_EXEC
    );
    if (status != HV_SUCCESS) {
        set_hv_error(error_buffer, error_capacity, "hv_vm_map", status);
        goto cleanup;
    }

    status = hv_vcpu_create(&vcpu, &exit_data, NULL);
    if (status != HV_SUCCESS) {
        set_hv_error(error_buffer, error_capacity, "hv_vcpu_create", status);
        goto cleanup;
    }
    vcpu_created = 1;

    status = hv_vcpu_set_reg(vcpu, HV_REG_PC, 0);
    if (status != HV_SUCCESS) {
        set_hv_error(error_buffer, error_capacity, "set guest pc", status);
        goto cleanup;
    }
    status = hv_vcpu_set_reg(vcpu, HV_REG_CPSR, 0x3c5u);
    if (status != HV_SUCCESS) {
        set_hv_error(error_buffer, error_capacity, "set guest cpsr", status);
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
    if (vcpu_created) {
        hv_vcpu_destroy(vcpu);
    }
    if (vm_created) {
        hv_vm_destroy();
    }
    if (guest_memory != MAP_FAILED) {
        munmap(guest_memory, memory_size);
    }
    return result;
}
