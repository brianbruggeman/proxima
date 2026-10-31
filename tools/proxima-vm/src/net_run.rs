//! Runs a guest whose only device is a virtio-net NIC wired to the caller's
//! pipes: the same exit loop `dispatch::run_dispatch_loop` drives, with the
//! console and block windows left empty and the net window bound to the
//! caller's [`HostNetTransport`]. The host-side peer holds the far ends of the
//! transport's ingress and egress pipes (`crate::frame_pipe`), so the guest
//! runs until that peer closes the ingress pipe (link down) or the watchdog
//! fires.

use proxima_core::ProximaError;

use crate::dispatch::FfiRecordingDispatcher;
use crate::elf::Segment;
use crate::virtio_net::HostNetTransport;

const MAX_HYPERCALLS: usize = 4;

/// Boots `segments` at `entry` with `boot_argument` in the guest's boot
/// register and serves its virtio-net window from `transport` until the guest
/// halts. Returns the number of virtio mmio exits the run took.
///
/// # Errors
///
/// Returns [`ProximaError::Upstream`] naming the failing hypervisor call,
/// the watchdog firing, or a device-model rejection.
pub fn run_net_guest(
    entry: u64,
    segments: &[Segment<'_>],
    guest_memory_size: u64,
    boot_argument: u64,
    transport: &mut HostNetTransport,
    watchdog_millis: u64,
) -> Result<u64, ProximaError> {
    platform::run(
        entry,
        segments,
        guest_memory_size,
        boot_argument,
        transport,
        watchdog_millis,
    )
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod platform {
    use proxima_core::ProximaError;

    use super::{FfiRecordingDispatcher, HostNetTransport, MAX_HYPERCALLS, Segment};
    use crate::dispatch::guest_kvm_segments;
    use crate::kvm::devices::Transport;
    use crate::kvm::{Capacities, ElfRun, Transports, run_elf};

    const NO_BYTE_CHANNEL: Capacities = Capacities {
        emitted: 0,
        console: 0,
        net: 0,
        blk: 0,
    };

    pub fn run(
        entry: u64,
        segments: &[Segment<'_>],
        guest_memory_size: u64,
        boot_argument: u64,
        transport: &mut HostNetTransport,
        watchdog_millis: u64,
    ) -> Result<u64, ProximaError> {
        let guest_segments = guest_kvm_segments(segments, guest_memory_size)?;
        let dispatcher = FfiRecordingDispatcher::new(&[], 0);
        let device = std::mem::replace(transport, HostNetTransport::new([0_u8; 6]));
        let outcome = run_elf(ElfRun {
            segments: &guest_segments,
            memory_size: guest_memory_size,
            entry,
            boot_argument,
            dispatcher: &dispatcher,
            transports: Transports {
                console: None,
                net: Some(Transport::Net(device)),
                blk: None,
            },
            max_hypercalls: MAX_HYPERCALLS,
            watchdog_millis,
            capacities: NO_BYTE_CHANNEL,
        });
        if let Some(Transport::Net(returned)) = outcome.transports.net {
            *transport = returned;
        }
        match outcome.failure {
            Some(message) => Err(ProximaError::Upstream(dispatcher.name_failure(message))),
            None => Ok(outcome.report.virtio_trap_count),
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod platform {
    use core::ffi::{c_char, c_void};
    use std::ffi::CStr;

    use proxima_core::ProximaError;

    use super::{FfiRecordingDispatcher, HostNetTransport, MAX_HYPERCALLS, Segment};
    use crate::dispatch::guest_raw_segments;
    use crate::loader::RawSegment;

    const ERROR_CAPACITY: usize = 512;

    unsafe extern "C" {
        fn proxima_vm_run_device_dispatch_loop(
            segments: *const RawSegment,
            segment_count: usize,
            guest_memory_size: u64,
            guest_memory_base: u64,
            entry: u64,
            boot_x0: u64,
            boot_cpsr: u64,
            dispatcher: *const c_void,
            console_transport: *mut c_void,
            net_transport: *mut c_void,
            blk_transport: *mut c_void,
            gicd_transport: *mut c_void,
            gicr_transport: *mut c_void,
            pl011_transport: *mut c_void,
            icc_transport: *mut c_void,
            max_hypercalls: usize,
            watchdog_millis: u64,
            emitted_out: *mut u8,
            emitted_capacity: usize,
            emitted_length_out: *mut usize,
            mmio_emitted_out: *mut u8,
            mmio_emitted_capacity: usize,
            mmio_emitted_length_out: *mut usize,
            net_emitted_out: *mut u8,
            net_emitted_capacity: usize,
            net_emitted_length_out: *mut usize,
            blk_emitted_out: *mut u8,
            blk_emitted_capacity: usize,
            blk_emitted_length_out: *mut usize,
            pl011_emitted_out: *mut u8,
            pl011_emitted_capacity: usize,
            pl011_emitted_length_out: *mut usize,
            create_to_first_exit_nanos_out: *mut u64,
            touch_all_pages_nanos_out: *mut u64,
            mmio_trap_count_out: *mut u64,
            gicd_trap_count_out: *mut u64,
            gicr_trap_count_out: *mut u64,
            pl011_trap_count_out: *mut u64,
            virtio_trap_count_out: *mut u64,
            vtimer_activation_count_out: *mut u64,
            wfi_wfe_trap_count_out: *mut u64,
            entered_el2_out: *mut u64,
            irq_assert_count_out: *mut u64,
            wfi_wait_count_out: *mut u64,
            timer_eoi_count_out: *mut u64,
            error_buffer: *mut c_char,
            error_capacity: usize,
        ) -> i32;
    }

    pub fn run(
        entry: u64,
        segments: &[Segment<'_>],
        guest_memory_size: u64,
        boot_argument: u64,
        transport: &mut HostNetTransport,
        watchdog_millis: u64,
    ) -> Result<u64, ProximaError> {
        let raw_segments = guest_raw_segments(segments, guest_memory_size)?;
        let dispatcher = FfiRecordingDispatcher::new(&[], 0);
        let mut virtio_trap_count: u64 = 0;
        let mut error_buffer = [0 as c_char; ERROR_CAPACITY];
        let null = core::ptr::null_mut::<c_void>();

        let status = unsafe {
            proxima_vm_run_device_dispatch_loop(
                raw_segments.as_ptr(),
                raw_segments.len(),
                guest_memory_size,
                0,
                entry,
                boot_argument,
                0,
                (&raw const dispatcher).cast(),
                null,
                (&raw mut *transport).cast(),
                null,
                null,
                null,
                null,
                null,
                MAX_HYPERCALLS,
                watchdog_millis,
                core::ptr::null_mut(),
                0,
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                0,
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                0,
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                0,
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                0,
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                &raw mut virtio_trap_count,
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                error_buffer.as_mut_ptr(),
                error_buffer.len(),
            )
        };
        if status != 0 {
            let message = unsafe { CStr::from_ptr(error_buffer.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            return Err(ProximaError::Upstream(dispatcher.name_failure(message)));
        }
        Ok(virtio_trap_count)
    }
}
