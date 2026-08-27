//! `extern "C"` trampolines the MMIO exit path
//! (`src/backend_macos.c`'s `proxima_vm_run_dispatch_loop`) calls back into
//! once a data-abort exit decodes to a register access inside the reserved
//! virtio-mmio window — the MMIO-transport mirror of
//! [`crate::dispatch::proxima_vm_dispatch_hypercall`], monomorphized to
//! [`crate::virtio_console::ConsoleTransport`] for the same reason that
//! function is monomorphized to `FfiRecordingDispatcher`: `extern "C"`
//! functions cannot be generic, so the C side's `void *console_transport`
//! needs one concrete Rust type to cast back to.

use core::ffi::c_void;

use crate::gic::{
    GicAccess, GicDistributor, GicRedistributor, IccAccess, IccCpuInterface, IccEffect, IccError,
};
use crate::pl011::{Pl011Access, Pl011Uart};
use crate::virtio_blk::BlkTransport;
use crate::virtio_console::ConsoleTransport;
use crate::virtio_net::{DrainedFrame, NetTransport};
use proxima_protocols::virtio::MmioAccess;

/// Sentinel `notified_queue_out` value meaning "no queue was notified by
/// this access" — outside the legal `u16` queue-index range this
/// transport's `MAX_QUEUES` (2) ever produces.
pub const NO_QUEUE_NOTIFIED: u16 = 0xffff;

/// Applies one register access to `console_transport`. On success, writes
/// the read value (or 0 for a write) into `*read_value_out` and, if the
/// access was a `QueueNotify` write, the notified queue index into
/// `*notified_queue_out` (otherwise [`NO_QUEUE_NOTIFIED`]). Returns 0 on
/// success, -1 if [`proxima_protocols::virtio::MmioDevice::apply`]
/// rejected the access (register-block protocol violation, not a crash).
///
/// # Safety
///
/// `console_transport` must be a valid, live pointer to a
/// [`ConsoleTransport`] for the duration of the call. `read_value_out` and
/// `notified_queue_out` must be valid, non-aliasing, writable pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn proxima_vm_dispatch_mmio(
    console_transport: *mut c_void,
    offset: u64,
    is_write: u8,
    value: u32,
    read_value_out: *mut u32,
    notified_queue_out: *mut u16,
) -> i32 {
    // SAFETY: forwarded verbatim from this function's own safety contract.
    let transport = unsafe { &mut *console_transport.cast::<ConsoleTransport>() };
    let access = MmioAccess {
        offset,
        is_write: is_write != 0,
        value,
    };
    match transport.apply(access) {
        Ok(effect) => {
            let (read_value, notified_queue) = match effect {
                proxima_protocols::virtio::MmioEffect::ReadValue(word) => (word, NO_QUEUE_NOTIFIED),
                proxima_protocols::virtio::MmioEffect::QueueNotify { queue } => (0, queue),
                _ => (0, NO_QUEUE_NOTIFIED),
            };
            // SAFETY: forwarded verbatim from this function's own safety contract.
            unsafe {
                *read_value_out = read_value;
                *notified_queue_out = notified_queue;
            }
            0
        }
        Err(_) => -1,
    }
}

/// Drains every avail-ring entry queue `queue` has published since the last
/// drain, copying the concatenated device-readable bytes into
/// `emitted_out` and publishing one used-ring completion per chain — the
/// real ring-codec walk (`proxima_protocols::virtio::{avail,descriptor,
/// used}`) `crate::virtio_console::ConsoleTransport::drain_tx` performs
/// against real guest memory. Returns 0 on success (with `*emitted_length_out`
/// set), -1 on any decode or bounds failure, -2 if `emitted_out` was too
/// small for the drained bytes.
///
/// # Safety
///
/// `console_transport` must be a valid, live pointer to a
/// [`ConsoleTransport`]. `guest_memory` must be valid and writable for
/// `guest_memory_length` bytes. `emitted_out` must be valid for
/// `emitted_capacity` bytes and not alias `guest_memory`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn proxima_vm_mmio_drain_tx(
    console_transport: *mut c_void,
    queue: u16,
    guest_memory: *mut u8,
    guest_memory_length: usize,
    emitted_out: *mut u8,
    emitted_capacity: usize,
    emitted_length_out: *mut usize,
) -> i32 {
    // SAFETY: forwarded verbatim from this function's own safety contract.
    let transport = unsafe { &mut *console_transport.cast::<ConsoleTransport>() };
    // SAFETY: forwarded verbatim from this function's own safety contract.
    let guest_memory =
        unsafe { core::slice::from_raw_parts_mut(guest_memory, guest_memory_length) };

    let drained = match transport.drain_tx(queue, guest_memory) {
        Ok(bytes) => bytes,
        Err(_) => return -1,
    };
    if drained.len() > emitted_capacity {
        return -2;
    }
    // SAFETY: forwarded verbatim from this function's own safety contract;
    // `drained.len() <= emitted_capacity` just checked above.
    unsafe {
        core::slice::from_raw_parts_mut(emitted_out, drained.len()).copy_from_slice(&drained);
        *emitted_length_out = drained.len();
    }
    0
}

/// Net-device mirror of [`proxima_vm_dispatch_mmio`], monomorphized to
/// [`NetTransport`] for the same `extern "C"`-cannot-be-generic reason.
///
/// # Safety
///
/// `net_transport` must be a valid, live pointer to a [`NetTransport`] for
/// the duration of the call. `read_value_out` and `notified_queue_out` must
/// be valid, non-aliasing, writable pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn proxima_vm_dispatch_mmio_net(
    net_transport: *mut c_void,
    offset: u64,
    is_write: u8,
    value: u32,
    read_value_out: *mut u32,
    notified_queue_out: *mut u16,
) -> i32 {
    // SAFETY: forwarded verbatim from this function's own safety contract.
    let transport = unsafe { &mut *net_transport.cast::<NetTransport>() };
    let access = MmioAccess {
        offset,
        is_write: is_write != 0,
        value,
    };
    match transport.apply(access) {
        Ok(effect) => {
            let (read_value, notified_queue) = match effect {
                proxima_protocols::virtio::MmioEffect::ReadValue(word) => (word, NO_QUEUE_NOTIFIED),
                proxima_protocols::virtio::MmioEffect::QueueNotify { queue } => (0, queue),
                _ => (0, NO_QUEUE_NOTIFIED),
            };
            // SAFETY: forwarded verbatim from this function's own safety contract.
            unsafe {
                *read_value_out = read_value;
                *notified_queue_out = notified_queue;
            }
            0
        }
        Err(_) => -1,
    }
}

/// Net-device mirror of [`proxima_vm_mmio_drain_tx`], monomorphized to
/// [`NetTransport`]. Each drained chain's `virtio_net_hdr` is stripped by
/// [`NetTransport::drain_tx`] before delivery, so `emitted_out` carries only
/// the concatenated raw Ethernet frame bytes, in publish order — the same
/// "concatenate what the transport handed back" shape
/// `proxima_vm_mmio_drain_tx` uses for the console's plain byte stream, with
/// the frame-boundary bookkeeping (`DrainedFrame::num_buffers`) intentionally
/// discarded here since this trampoline's caller (the C exit loop) only
/// needs the byte stream, not per-frame metadata.
///
/// # Safety
///
/// `net_transport` must be a valid, live pointer to a [`NetTransport`].
/// `guest_memory` must be valid and writable for `guest_memory_length`
/// bytes. `emitted_out` must be valid for `emitted_capacity` bytes and not
/// alias `guest_memory`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn proxima_vm_mmio_drain_tx_net(
    net_transport: *mut c_void,
    queue: u16,
    guest_memory: *mut u8,
    guest_memory_length: usize,
    emitted_out: *mut u8,
    emitted_capacity: usize,
    emitted_length_out: *mut usize,
) -> i32 {
    // SAFETY: forwarded verbatim from this function's own safety contract.
    let transport = unsafe { &mut *net_transport.cast::<NetTransport>() };
    // SAFETY: forwarded verbatim from this function's own safety contract.
    let guest_memory =
        unsafe { core::slice::from_raw_parts_mut(guest_memory, guest_memory_length) };

    let mut frames = std::vec::Vec::new();
    let mut sink = |drained: DrainedFrame| frames.extend_from_slice(&drained.frame);
    if transport.drain_tx(queue, guest_memory, &mut sink).is_err() {
        return -1;
    }
    if frames.len() > emitted_capacity {
        return -2;
    }
    // SAFETY: forwarded verbatim from this function's own safety contract;
    // `frames.len() <= emitted_capacity` just checked above.
    unsafe {
        core::slice::from_raw_parts_mut(emitted_out, frames.len()).copy_from_slice(&frames);
        *emitted_length_out = frames.len();
    }
    0
}

/// Blk-device mirror of [`proxima_vm_dispatch_mmio`], monomorphized to
/// [`BlkTransport`] for the same `extern "C"`-cannot-be-generic reason.
///
/// # Safety
///
/// `blk_transport` must be a valid, live pointer to a [`BlkTransport`] for
/// the duration of the call. `read_value_out` and `notified_queue_out` must
/// be valid, non-aliasing, writable pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn proxima_vm_dispatch_mmio_blk(
    blk_transport: *mut c_void,
    offset: u64,
    is_write: u8,
    value: u32,
    read_value_out: *mut u32,
    notified_queue_out: *mut u16,
) -> i32 {
    // SAFETY: forwarded verbatim from this function's own safety contract.
    let transport = unsafe { &mut *blk_transport.cast::<BlkTransport>() };
    let access = MmioAccess {
        offset,
        is_write: is_write != 0,
        value,
    };
    match transport.apply(access) {
        Ok(effect) => {
            let (read_value, notified_queue) = match effect {
                proxima_protocols::virtio::MmioEffect::ReadValue(word) => (word, NO_QUEUE_NOTIFIED),
                proxima_protocols::virtio::MmioEffect::QueueNotify { queue } => (0, queue),
                _ => (0, NO_QUEUE_NOTIFIED),
            };
            // SAFETY: forwarded verbatim from this function's own safety contract.
            unsafe {
                *read_value_out = read_value;
                *notified_queue_out = notified_queue;
            }
            0
        }
        Err(_) => -1,
    }
}

/// Blk-device mirror of [`proxima_vm_mmio_drain_tx`], monomorphized to
/// [`BlkTransport`]. Unlike the console/net drain (transmit-only: the device
/// only reads guest memory), this one both reads and writes real guest
/// memory (`BlkTransport::service_queue`'s own `IN`-request data write) —
/// `emitted_out` carries, per serviced request, an 8-byte little-endian
/// sector, a 1-byte status, then the data bytes actually moved, letting a
/// caller two layers up (a VM-exit test with no access to the transport
/// itself) prove the bytes that crossed the ring matched a host-seeded
/// pattern.
///
/// # Safety
///
/// `blk_transport` must be a valid, live pointer to a [`BlkTransport`].
/// `guest_memory` must be valid and writable for `guest_memory_length`
/// bytes. `emitted_out` must be valid for `emitted_capacity` bytes and not
/// alias `guest_memory`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn proxima_vm_mmio_service_blk(
    blk_transport: *mut c_void,
    queue: u16,
    guest_memory: *mut u8,
    guest_memory_length: usize,
    emitted_out: *mut u8,
    emitted_capacity: usize,
    emitted_length_out: *mut usize,
) -> i32 {
    // SAFETY: forwarded verbatim from this function's own safety contract.
    let transport = unsafe { &mut *blk_transport.cast::<BlkTransport>() };
    // SAFETY: forwarded verbatim from this function's own safety contract.
    let guest_memory =
        unsafe { core::slice::from_raw_parts_mut(guest_memory, guest_memory_length) };

    let serviced = match transport.service_queue(queue, guest_memory) {
        Ok(serviced) => serviced,
        Err(_) => return -1,
    };
    let mut encoded = std::vec::Vec::new();
    for request in serviced {
        encoded.extend_from_slice(&request.sector.to_le_bytes());
        encoded.push(request.status);
        encoded.extend_from_slice(&request.data);
    }
    if encoded.len() > emitted_capacity {
        return -2;
    }
    // SAFETY: forwarded verbatim from this function's own safety contract;
    // `encoded.len() <= emitted_capacity` just checked above.
    unsafe {
        core::slice::from_raw_parts_mut(emitted_out, encoded.len()).copy_from_slice(&encoded);
        *emitted_length_out = encoded.len();
    }
    0
}

/// GICD register-access trampoline (M5b GIC slice 3), monomorphized to
/// [`GicDistributor`] for the same `extern "C"`-cannot-be-generic reason the
/// console/net/blk trampolines are monomorphized to their own transports.
/// Neither GIC register block owns a virtqueue, so there is no
/// `notified_queue_out`/drain counterpart here: a read's value reaches the
/// guest through `*read_value_out`, a write is applied with no further
/// host-visible effect to report.
///
/// # Safety
///
/// `gicd_transport` must be a valid, live pointer to a [`GicDistributor`] for
/// the duration of the call. `read_value_out` must be a valid, writable
/// pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn proxima_vm_dispatch_mmio_gicd(
    gicd_transport: *mut c_void,
    offset: u64,
    is_write: u8,
    value: u32,
    read_value_out: *mut u32,
) -> i32 {
    // SAFETY: forwarded verbatim from this function's own safety contract.
    let distributor = unsafe { &mut *gicd_transport.cast::<GicDistributor>() };
    let access = GicAccess {
        offset,
        is_write: is_write != 0,
        value,
    };
    match distributor.apply(access) {
        Ok(effect) => {
            let read_value = match effect {
                crate::gic::GicdEffect::ReadValue(word) => word,
                _ => 0,
            };
            // SAFETY: forwarded verbatim from this function's own safety contract.
            unsafe {
                *read_value_out = read_value;
            }
            0
        }
        Err(_) => -1,
    }
}

/// GICR register-access trampoline, monomorphized to [`GicRedistributor`] —
/// the redistributor mirror of [`proxima_vm_dispatch_mmio_gicd`].
///
/// # Safety
///
/// `gicr_transport` must be a valid, live pointer to a [`GicRedistributor`]
/// for the duration of the call. `read_value_out` must be a valid, writable
/// pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn proxima_vm_dispatch_mmio_gicr(
    gicr_transport: *mut c_void,
    offset: u64,
    is_write: u8,
    value: u32,
    read_value_out: *mut u32,
) -> i32 {
    // SAFETY: forwarded verbatim from this function's own safety contract.
    let redistributor = unsafe { &mut *gicr_transport.cast::<GicRedistributor>() };
    let access = GicAccess {
        offset,
        is_write: is_write != 0,
        value,
    };
    match redistributor.apply(access) {
        Ok(effect) => {
            let read_value = match effect {
                crate::gic::GicrEffect::ReadValue(word) => word,
                _ => 0,
            };
            // SAFETY: forwarded verbatim from this function's own safety contract.
            unsafe {
                *read_value_out = read_value;
            }
            0
        }
        Err(_) => -1,
    }
}

/// pl011 register-access trampoline (M5b pl011 slice), monomorphized to
/// [`Pl011Uart`] for the same `extern "C"`-cannot-be-generic reason every
/// other device trampoline above is monomorphized to its own transport. The
/// pl011 owns no virtqueue, so unlike the console/net/blk pair there is no
/// separate drain/service call — a `UARTDR` write's effect
/// ([`crate::pl011::Pl011Effect::TxByte`]) is reported directly through
/// `tx_byte_out`/`tx_emitted_out` in this same call.
///
/// # Safety
///
/// `pl011_transport` must be a valid, live pointer to a [`Pl011Uart`] for the
/// duration of the call. `read_value_out`, `tx_byte_out`, and
/// `tx_emitted_out` must each be a valid, writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn proxima_vm_dispatch_mmio_pl011(
    pl011_transport: *mut c_void,
    offset: u64,
    is_write: u8,
    value: u32,
    read_value_out: *mut u32,
    tx_byte_out: *mut u8,
    tx_emitted_out: *mut u8,
) -> i32 {
    // SAFETY: forwarded verbatim from this function's own safety contract.
    let uart = unsafe { &mut *pl011_transport.cast::<Pl011Uart>() };
    let access = Pl011Access {
        offset,
        is_write: is_write != 0,
        value,
    };
    match uart.apply(access) {
        Ok(effect) => {
            let (read_value, tx_byte, tx_emitted) = match effect {
                crate::pl011::Pl011Effect::ReadValue(word) => (word, 0, 0),
                crate::pl011::Pl011Effect::Applied => (0, 0, 0),
                crate::pl011::Pl011Effect::TxByte(byte) => (0, byte, 1),
            };
            // SAFETY: forwarded verbatim from this function's own safety contract.
            unsafe {
                *read_value_out = read_value;
                *tx_byte_out = tx_byte;
                *tx_emitted_out = tx_emitted;
            }
            0
        }
        Err(_) => -1,
    }
}

/// GICv3 CPU-interface system-register trampoline (EC 0x18 trapped
/// `MSR`/`MRS`), monomorphized to [`IccCpuInterface`] for the same
/// `extern "C"`-cannot-be-generic reason every other device trampoline
/// above is monomorphized to its own state struct. Unlike every MMIO
/// trampoline above, this access is not offset-keyed: `op0`/`op1`/`crn`/
/// `crm`/`op2` name the trapped system register directly, the same tuple
/// the exit loop's ISS decode recovers from the trapping `MSR`/`MRS`
/// instruction (`backend_macos.c`'s `decode_icc_sysreg_iss`).
///
/// Return codes [`proxima_vm_dispatch_sysreg_icc`] reports on rejection —
/// the C caller already holds `op0`/`op1`/`crn`/`crm`/`op2` from its own ISS
/// decode (`decode_icc_sysreg_iss` in `backend_macos.c`), so the only fact
/// this trampoline loses by collapsing [`IccError`] to a bare `-1` is *why*
/// — unknown encoding vs. a direction mismatch against a register this
/// module does model. A distinct nonzero code per [`IccError`] variant lets
/// the caller's error string name the reason next to the tuple it already
/// has, instead of the generic "icc sysreg access rejected" that used to
/// carry no register-specific information at all.
pub const ICC_DISPATCH_UNKNOWN_REGISTER: i32 = 1;
pub const ICC_DISPATCH_READ_ONLY_REGISTER: i32 = 2;
pub const ICC_DISPATCH_WRITE_ONLY_REGISTER: i32 = 3;

/// # Safety
///
/// `icc_transport` must be a valid, live pointer to an [`IccCpuInterface`]
/// for the duration of the call. `read_value_out`, `deactivated_out`, and
/// `deactivated_intid_out` must be valid, writable, non-aliasing pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn proxima_vm_dispatch_sysreg_icc(
    icc_transport: *mut c_void,
    op0: u8,
    op1: u8,
    crn: u8,
    crm: u8,
    op2: u8,
    is_write: u8,
    value: u64,
    read_value_out: *mut u64,
    deactivated_out: *mut u8,
    deactivated_intid_out: *mut u32,
) -> i32 {
    // SAFETY: forwarded verbatim from this function's own safety contract.
    let icc = unsafe { &mut *icc_transport.cast::<IccCpuInterface>() };
    let access = IccAccess {
        op0,
        op1,
        crn,
        crm,
        op2,
        is_write: is_write != 0,
        value,
    };
    match icc.apply(access) {
        Ok(effect) => {
            let (read_value, deactivated_intid) = match effect {
                IccEffect::ReadValue(word) => (word, None),
                IccEffect::Applied => (0, None),
                IccEffect::InterruptDeactivated(intid) => (0, Some(intid)),
            };
            // SAFETY: forwarded verbatim from this function's own safety contract.
            unsafe {
                *read_value_out = read_value;
                *deactivated_out = u8::from(deactivated_intid.is_some());
                *deactivated_intid_out = deactivated_intid.unwrap_or(0);
            }
            0
        }
        Err(IccError::UnknownRegister { .. }) => ICC_DISPATCH_UNKNOWN_REGISTER,
        Err(IccError::ReadOnlyRegister { .. }) => ICC_DISPATCH_READ_ONLY_REGISTER,
        Err(IccError::WriteOnlyRegister { .. }) => ICC_DISPATCH_WRITE_ONLY_REGISTER,
    }
}

/// Records `intid` pending in `icc_transport`'s one-deep interrupt slot —
/// the trampoline half of [`IccCpuInterface::set_pending`], called by the
/// HVF exit loop the instant `HV_EXIT_REASON_VTIMER_ACTIVATED` fires
/// (`backend_macos.c`), before it tells HVF the guest's IRQ line is
/// asserted via `hv_vcpu_set_pending_interrupt`.
///
/// # Safety
///
/// `icc_transport` must be a valid, live pointer to an [`IccCpuInterface`]
/// for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn proxima_vm_icc_set_vtimer_pending(icc_transport: *mut c_void, intid: u32) {
    // SAFETY: forwarded verbatim from this function's own safety contract.
    let icc = unsafe { &mut *icc_transport.cast::<IccCpuInterface>() };
    icc.set_pending(intid);
}
