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

/* virtio-mmio device window (VIRTIO 1.2 spec §4.2.2): reserved guest-
 * physical range, deliberately left unmapped by both `hv_vm_map` windows
 * (`backend_macos.c`) and `KVM_SET_USER_MEMORY_REGION` slots
 * (`backend_linux.c`) so any guest load/store into it traps to the host as
 * a data-abort exit instead of touching real memory. Chosen well above the
 * 64 MiB guest-memory ceiling either dispatch loop maps so it can never
 * overlap a real ELF segment or the stack reservation. Window size covers
 * every register through `ConfigGeneration` (0x0fc) plus headroom for
 * device-specific config space (0x100+), never itself dereferenced by this
 * transport slice. */
#define PROXIMA_VM_MMIO_WINDOW_BASE 0x1000000000ull
#define PROXIMA_VM_MMIO_WINDOW_SIZE 0x1000ull

/* Second, non-overlapping virtio-mmio device window for the net device
 * (M6 slice 5, mirroring the console window immediately above): placed
 * directly after the console window's reserved range so the two windows
 * are adjacent but never overlap, and — same as the console window — well
 * above real guest RAM so it can never alias it. Sized identically for the
 * same headroom reason (register block through `ConfigGeneration` plus the
 * net device's own config-space fields,
 * `proxima_protocols::virtio::net::CONFIG_SPACE_BASE` onward, both fit
 * comfortably inside one 4 KiB window). */
#define PROXIMA_VM_NET_MMIO_WINDOW_BASE 0x1000001000ull
#define PROXIMA_VM_NET_MMIO_WINDOW_SIZE 0x1000ull

/* Third, non-overlapping virtio-mmio device window for the block device
 * (M6 slice 6, mirroring the net window immediately above): placed directly
 * after the net window's reserved range for the same non-aliasing reason.
 * Sized identically for the same headroom reason (register block through
 * `ConfigGeneration` plus the blk device's own config-space fields,
 * `proxima_protocols::virtio::blk::CONFIG_SPACE_BASE` onward, both fit
 * comfortably inside one 4 KiB window). */
#define PROXIMA_VM_BLK_MMIO_WINDOW_BASE 0x1000002000ull
#define PROXIMA_VM_BLK_MMIO_WINDOW_SIZE 0x1000ull

/* GICv3 Distributor and Redistributor windows (M5b GIC slice 3): the real
 * guest-physical addresses `src/dtb.rs`'s `QemuVirtLayout::CANONICAL` (GICD)
 * and `QemuVirtLayout::single_vcpu` (GICR) advertise to the guest in the DTB
 * this VM builds, not invented addresses — the guest reads its GIC location
 * from the same tree the DTB writes, and this exit loop must trap the exact
 * range the guest was told to expect. Both sit far above the 64 MiB guest
 * RAM region a device-model boot maps (`0x0400_0000`): `0x0800_0000` (GICD)
 * and `0x080a_0000` (GICR) both exceed that ceiling, so neither window is
 * ever backed by `hv_vm_map`'d guest memory and every guest access
 * genuinely traps as a data abort, the same non-aliasing guarantee the
 * virtio windows get from sitting at `0x1000000000+` instead. */
#define PROXIMA_VM_GICD_MMIO_WINDOW_BASE 0x08000000ull
#define PROXIMA_VM_GICD_MMIO_WINDOW_SIZE 0x00010000ull
#define PROXIMA_VM_GICR_MMIO_WINDOW_BASE 0x080a0000ull
#define PROXIMA_VM_GICR_MMIO_WINDOW_SIZE 0x00020000ull

/* PL011 UART window (M5b pl011 slice): the real guest-physical address
 * `src/dtb.rs`'s `QemuVirtLayout::CANONICAL.uart_base` advertises to the
 * guest in this VM's own DTB (QEMU virt's `VIRT_UART`, `hw/arm/virt.c`), not
 * an invented address, the same "the guest reads its device location from
 * the tree this VM built" argument the GICD/GICR windows above make.
 * `0x0900_0000` exceeds the 64 MiB / `0x0400_0000` guest-RAM ceiling, so
 * this window is never backed by `hv_vm_map`'d guest memory either, and
 * every guest access genuinely traps as a data abort. */
#define PROXIMA_VM_PL011_MMIO_WINDOW_BASE 0x09000000ull
#define PROXIMA_VM_PL011_MMIO_WINDOW_SIZE 0x00001000ull

/* Rust-side mmio register-access dispatcher (`src/mmio_trampoline.rs`):
 * applies one decoded `(offset, is_write, value)` access to a
 * `ConsoleTransport`'s `MmioDevice` FSM. `*read_value_out` carries the
 * value a read access must write back into the guest's destination
 * register (undefined for a write). `*notified_queue_out` carries the
 * notified queue index when the access was a `QueueNotify` write, or
 * `PROXIMA_VM_MMIO_NO_QUEUE_NOTIFIED` otherwise. Returns 0 on success, -1
 * if the register-block FSM rejected the access. */
#define PROXIMA_VM_MMIO_NO_QUEUE_NOTIFIED 0xffffu

int32_t proxima_vm_dispatch_mmio(
    void *console_transport,
    uint64_t offset,
    uint8_t is_write,
    uint32_t value,
    uint32_t *read_value_out,
    uint16_t *notified_queue_out
);

/* M5b PSCI (ARM DEN0022) call handler (`src/psci.rs`): a pure decision over
 * `(function_id, args)`, no transport state, so this entry takes no `void *`
 * — the stateless counterpart to `proxima_vm_dispatch_mmio` above. The hvc
 * trap loop (`proxima_vm_run_device_dispatch_loop` below) range-tests `x0`
 * against `PROXIMA_VM_PSCI_FAST_CALL_32_BASE`/`_64_BASE` (each a 0x20-wide
 * SMCCC fast-call range) BEFORE its `PROXIMA_VM_HALT_VERB`/
 * `PROXIMA_VM_EMIT_VERB` checks, since a compliant PSCI function ID is six
 * orders of magnitude above either sentinel and never collides with them
 * (`src/psci.rs`'s own module doc walks the disjointness argument).
 * `*return_value_out` carries the signed 32-bit value (sign-extended to 64
 * bits) to write into the guest's `x0` before resuming, valid only when
 * `*action_out == 0`. `*action_out`: `0` resume the guest with
 * `*return_value_out` in `x0`, `1` `SYSTEM_OFF` — end the dispatch loop
 * exactly like `PROXIMA_VM_HALT_VERB` (never a second exit channel), `2`
 * `SYSTEM_RESET` — this handler has no reset capability, so it ends the
 * loop identically to `SYSTEM_OFF` rather than returning success into a
 * guest that would then run past it. Returns 0 always; failure is not a
 * concept `src/psci.rs`'s pure match produces. */
#define PROXIMA_VM_PSCI_FAST_CALL_32_BASE 0x84000000u
#define PROXIMA_VM_PSCI_FAST_CALL_64_BASE 0xc4000000u
#define PROXIMA_VM_PSCI_FAST_CALL_RANGE_WIDTH 0x20u

int32_t proxima_vm_dispatch_psci(
    uint32_t function_id,
    uint64_t arg0,
    uint64_t arg1,
    uint64_t arg2,
    int64_t *return_value_out,
    uint8_t *action_out
);

/* Rust-side ring-codec drain (`src/mmio_trampoline.rs`): walks every
 * avail-ring entry `queue` has published since the last drain against real
 * `guest_memory`, copies the concatenated device-readable bytes into
 * `emitted_out`, and publishes one used-ring completion per chain. Returns
 * 0 on success (`*emitted_length_out` set), -1 on decode/bounds failure,
 * -2 if `emitted_out` was too small. */
int32_t proxima_vm_mmio_drain_tx(
    void *console_transport,
    uint16_t queue,
    uint8_t *guest_memory,
    size_t guest_memory_length,
    uint8_t *emitted_out,
    size_t emitted_capacity,
    size_t *emitted_length_out
);

/* Net-device mirrors of `proxima_vm_dispatch_mmio` / `proxima_vm_mmio_drain_tx`
 * above, monomorphized to `crate::virtio_net::NetTransport`
 * (`src/mmio_trampoline.rs`) for the same `extern "C"`-cannot-be-generic
 * reason the console pair is monomorphized to `ConsoleTransport`. Drained
 * bytes are the concatenated Ethernet frame bytes with each chain's
 * `virtio_net_hdr` already stripped (`NetTransport::drain_tx`'s own
 * `FrameSink` contract). */
int32_t proxima_vm_dispatch_mmio_net(
    void *net_transport,
    uint64_t offset,
    uint8_t is_write,
    uint32_t value,
    uint32_t *read_value_out,
    uint16_t *notified_queue_out
);

int32_t proxima_vm_mmio_drain_tx_net(
    void *net_transport,
    uint16_t queue,
    uint8_t *guest_memory,
    size_t guest_memory_length,
    uint8_t *emitted_out,
    size_t emitted_capacity,
    size_t *emitted_length_out
);

/* Blk-device mirrors of `proxima_vm_dispatch_mmio` /
 * `proxima_vm_mmio_drain_tx`, monomorphized to
 * `crate::virtio_blk::BlkTransport` (`src/mmio_trampoline.rs`). Unlike the
 * console/net pair, the "drain" (`proxima_vm_mmio_service_blk`) both reads
 * AND writes real guest memory (an `IN` request's data descriptor is
 * device-writable, spec §5.2.6), and `emitted_out` carries, per serviced
 * request, an 8-byte little-endian sector, a 1-byte status, then the data
 * bytes actually moved (empty for `FLUSH`/unsupported) — the proof a caller
 * one layer up needs that the bytes crossing the ring matched what the
 * local store held, without keeping the transport alive past this call. */
int32_t proxima_vm_dispatch_mmio_blk(
    void *blk_transport,
    uint64_t offset,
    uint8_t is_write,
    uint32_t value,
    uint32_t *read_value_out,
    uint16_t *notified_queue_out
);

int32_t proxima_vm_mmio_service_blk(
    void *blk_transport,
    uint16_t queue,
    uint8_t *guest_memory,
    size_t guest_memory_length,
    uint8_t *emitted_out,
    size_t emitted_capacity,
    size_t *emitted_length_out
);

/* GICD/GICR register-access trampolines (`src/mmio_trampoline.rs`), each
 * monomorphized to its own state struct in `src/gic.rs`
 * (`GicDistributor`/`GicRedistributor`) for the same `extern "C"`-cannot-be-
 * generic reason the console/net/blk pair are monomorphized to their own
 * transports -- one struct per window here, mirroring that precedent, rather
 * than a wrapper spanning both blocks, since `GicDistributor` and
 * `GicRedistributor` are already the two independently-addressed state
 * machines `src/gic.rs`'s own module doc describes. Neither GIC register
 * block owns a virtqueue, so there is no drain/service counterpart to
 * `proxima_vm_mmio_drain_tx` here -- a read's value reaches the guest through
 * `*read_value_out` exactly like every other window, and a write is applied
 * with no further host-visible effect to report. Returns 0 on success, -1 if
 * the register-block FSM rejected the access. */
int32_t proxima_vm_dispatch_mmio_gicd(
    void *gicd_transport,
    uint64_t offset,
    uint8_t is_write,
    uint32_t value,
    uint32_t *read_value_out
);

int32_t proxima_vm_dispatch_mmio_gicr(
    void *gicr_transport,
    uint64_t offset,
    uint8_t is_write,
    uint32_t value,
    uint32_t *read_value_out
);

/* GICv3 CPU-interface system-register trampoline (`src/mmio_trampoline.rs`),
 * monomorphized to `crate::gic::IccCpuInterface` for the same reason the
 * GICD/GICR pair above are monomorphized to their own state structs. Unlike
 * every MMIO trampoline in this header, this access is not offset-keyed --
 * `op0`/`op1`/`crn`/`crm`/`op2` name the trapped `MRS`/`MSR` system register
 * directly, the same tuple ARM's ISS-for-trapped-sysreg encoding recovers
 * from EC 0x18 (`decode_icc_sysreg_iss` in `backend_macos.c`). Returns 0 on
 * success (`*read_value_out` set for a read, ignored for a write), or one of
 * `ICC_DISPATCH_UNKNOWN_REGISTER` (1, no register modeled at this
 * encoding), `ICC_DISPATCH_READ_ONLY_REGISTER` (2, a write against a
 * read-only register), `ICC_DISPATCH_WRITE_ONLY_REGISTER` (3, a read of a
 * write-only register) -- the caller already holds `op0`/`op1`/`crn`/`crm`/
 * `op2` from its own ISS decode, so a distinct code per rejection reason is
 * enough to build a self-documenting error string without this trampoline
 * reaching back through the FFI boundary to format one. */
#define ICC_DISPATCH_UNKNOWN_REGISTER 1
#define ICC_DISPATCH_READ_ONLY_REGISTER 2
#define ICC_DISPATCH_WRITE_ONLY_REGISTER 3

/* M5b-beyond: the virtual timer's INTID (`dtb.rs`'s `write_timer` PPI
 * triple `1 11 4` -- PPI number 11, GICv3's PPI encoding is INTID = 16 +
 * PPI number, so 16 + 11 = 27). The one interrupt source this VM's ICC
 * model's one-deep pending slot (`IccCpuInterface`, `gic.rs`) ever holds. */
#define PROXIMA_VM_VTIMER_INTID 27u

/* `*deactivated_out` is nonzero, with `*deactivated_intid_out` set, exactly
 * when this access was an `ICC_EOIR1_EL1` write that matched the ICC
 * model's one active interrupt (`IccEffect::InterruptDeactivated`,
 * `gic.rs`) -- the caller's cue to service the architected re-arm contract
 * (`hv_vcpu_set_pending_interrupt` false, then `hv_vcpu_set_vtimer_mask`
 * false) when the deactivated INTID is `PROXIMA_VM_VTIMER_INTID`. */
int32_t proxima_vm_dispatch_sysreg_icc(
    void *icc_transport,
    uint8_t op0,
    uint8_t op1,
    uint8_t crn,
    uint8_t crm,
    uint8_t op2,
    uint8_t is_write,
    uint64_t value,
    uint64_t *read_value_out,
    uint8_t *deactivated_out,
    uint32_t *deactivated_intid_out
);

/* Records `intid` pending in `icc_transport`'s one-deep slot
 * (`IccCpuInterface::set_pending`, `gic.rs`). The HVF exit loop calls this
 * the instant `HV_EXIT_REASON_VTIMER_ACTIVATED` fires, before telling HVF
 * the guest's IRQ line is asserted. */
void proxima_vm_icc_set_vtimer_pending(void *icc_transport, uint32_t intid);

/* PL011 register-access trampoline (`src/mmio_trampoline.rs`), monomorphized
 * to `crate::pl011::Pl011Uart` for the same `extern "C"`-cannot-be-generic
 * reason every other device trampoline above is monomorphized to its own
 * state struct. The pl011 owns no virtqueue, so there is no drain/service
 * counterpart here either -- but unlike the GICD/GICR pair, a `UARTDR` write
 * IS a host-visible effect (the guest emitting one console byte), so this
 * trampoline reports it directly through `*tx_byte_out`/`*tx_emitted_out`
 * rather than routing it through a queue-notify drain: `*tx_emitted_out` is
 * nonzero iff this access was a `UARTDR` write, in which case `*tx_byte_out`
 * carries the emitted byte. Returns 0 on success, -1 if the register-block
 * FSM rejected the access. */
int32_t proxima_vm_dispatch_mmio_pl011(
    void *pl011_transport,
    uint64_t offset,
    uint8_t is_write,
    uint32_t value,
    uint32_t *read_value_out,
    uint8_t *tx_byte_out,
    uint8_t *tx_emitted_out
);

/* Second, device-model exit loop (`src/boot.rs`'s two callers,
 * `boot_linux_kernel`/`boot_edk2_firmware`): a distinct C symbol from
 * `proxima_vm_run_dispatch_loop` above, not an overload of it -- the two
 * loops' own signatures never converged (this one threads a
 * `guest_memory_base`/`boot_x0`/`boot_cpsr` triple no ELF-guest caller of
 * `dispatch::run_dispatch_loop` needs, plus the whole console/net/blk/
 * gicd/gicr/pl011/icc device-model transport set and their five extra
 * emitted-byte channels), and C has no overloading to fold them under one
 * name safely. Maps every entry in `segments` into a single flat
 * `guest_memory_size`-byte guest address space, whose guest-physical base is
 * `guest_memory_base` (`dtb.rs`'s RAM base for a real kernel boot) — each at
 * its own `guest_address` relative to that base, its own real
 * `readable`/`writable`/`executable` permissions (never one RWX blob; W^X
 * per `proxima_vm_segment_t`'s own doc in `ffi_segment.h`), copying
 * `data[0..data_length)` in at that offset first (a zero-`data_length`
 * segment, e.g. a stack reservation, copies nothing) — then runs a real
 * vCPU: sets `PC = entry`, traps every `hvc #0` / `out dx, al`, MMIO data
 * aborts against the console/net/blk/gicd/gicr/pl011 windows above, EC 0x18
 * ICC system-register traps, EC 0x1 `WFI`/`WFE`, and PSCI fast calls, and
 * either emits one byte from `guest_memory[pointer]` (`verb ==
 * PROXIMA_VM_EMIT_VERB`), ends the loop (`verb == PROXIMA_VM_HALT_VERB`, or
 * an SMCCC `SYSTEM_OFF`/`SYSTEM_RESET`), or forwards the exit to
 * `proxima_vm_dispatch_hypercall` and copies its response back into
 * `guest_memory[pointer..]` before resuming.
 *
 * `console_transport`/`net_transport`/`blk_transport`/`gicd_transport`/
 * `gicr_transport`/`pl011_transport`/`icc_transport` are each a live
 * instance of their matching Rust-side transport; a `QueueNotify` effect
 * immediately drains/services the named queue via that transport's own
 * function, appending the bytes into `mmio_emitted_out` (console),
 * `net_emitted_out` (net), or `blk_emitted_out` (blk) — kept deliberately
 * separate from each other and from `emitted_out` (the
 * `PROXIMA_VM_EMIT_VERB` hypercall path) and from `pl011_emitted_out` (the
 * pl011 `UARTDR` byte channel M5's exit criterion names) so a guest
 * exercising every path in one run never has its observable byte streams
 * silently interleaved into one buffer a test would then have to
 * disentangle.
 *
 * `create_to_first_exit_nanos_out`/`touch_all_pages_nanos_out`/
 * `mmio_trap_count_out`/`gicd_trap_count_out`/`gicr_trap_count_out`/
 * `pl011_trap_count_out`/`virtio_trap_count_out`/
 * `vtimer_activation_count_out`/`wfi_wfe_trap_count_out`/`entered_el2_out`
 * are the M3/M5b diagnostic instrument: unconditional out-params, populated
 * in `cleanup` regardless of `result`, so a run that does NOT reach a clean
 * halt still reports exactly how far it got.
 *
 * Returns 0 on a clean halt, -1 on any mapping, hypervisor, dispatch, or
 * budget-exceeded failure (message in `error_buffer`). Caps the exit count
 * at `max_hypercalls` (hypercall exits) and an internal total-exit budget
 * (every exit reason) so a guest that never halts cannot loop the host
 * forever. */
int proxima_vm_run_device_dispatch_loop(
    const proxima_vm_segment_t *segments,
    size_t segment_count,
    uint64_t guest_memory_size,
    uint64_t guest_memory_base,
    uint64_t entry,
    uint64_t boot_x0,
    /* 0 sentinel = this loop's default (`0x3c5u`, EL1h); nonzero = literal
     * CPSR (`backend_macos.c::proxima_vm_run_device_dispatch_loop`'s own doc
     * on this parameter, and `boot::boot_edk2_firmware`'s doc on why edk2
     * needs EL2h instead). */
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
    /* 0 sentinel = no watchdog (every kernel/lambda caller); nonzero =
     * milliseconds before a forced `hv_vcpus_exit` diagnostic fires
     * (`backend_macos.c::arm_watchdog`'s own doc, `boot::boot_edk2_firmware`'s
     * doc on why it opts in). */
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
    /* Written 1 only when `boot_cpsr` asked for EL2 entry AND this host's
     * own HVF actually honored it; 0 otherwise (never requested, or
     * requested and this host fell back to EL1h -- `device_create_vm`'s own
     * doc in `backend_macos.c` on why HV_UNSUPPORTED triggers a same-call
     * fallback rather than a hard error). Always 0 on the KVM/x86_64 lane
     * (`backend_linux.c`'s own doc: no ARM exception-level model there). */
    uint64_t *entered_el2_out,
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
