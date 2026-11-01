//! The resident Linux lane against a real kernel: the guest boots once on a
//! dedicated vCPU thread, keeps running when idle, and ends only when stopped.

#![cfg(all(target_os = "linux", target_arch = "x86_64"))]
#![allow(clippy::expect_used, clippy::unwrap_used)]

mod common;

use bytes::Bytes;
use futures::FutureExt;
use proxima_primitives::pipe::Pipe;
use proxima_vm::frame_pipe::FrameError;
use proxima_vm::kvm::resident::{LinkConfig, ResidentLinux, net_device_cmdline};
use proxima_vm::kvm::trap::End;
use proxima_vm::linux::{LinuxBootInputs, LinuxGuestLimits, initramfs_from_init};

const RINGS: usize = 64;

fn boot_inputs() -> LinuxBootInputs {
    let initramfs = initramfs_from_init(&common::guest("idle-init")).expect("initramfs builds");
    LinuxBootInputs {
        kernel: Bytes::from_static(common::kernel()),
        initramfs: Bytes::from(initramfs),
        cmdline: format!("{} {}", common::CMDLINE, net_device_cmdline()),
        limits: LinuxGuestLimits::default(),
    }
}

fn link() -> LinkConfig {
    LinkConfig { mac: [0x02, 0x00, 0x00, 0x00, 0x00, 0x01], frame_capacity: 16, host_consumer: None }
}

#[test]
fn stop_ends_a_resident_guest_with_stopped_and_closes_the_guest_to_host_pipe() {
    let (resident, host) = ResidentLinux::start(boot_inputs(), link()).expect("the resident guest starts");
    assert!(resident.is_running(), "the vcpu thread runs the guest after start returns");
    assert!(host.from_guest.call(()).now_or_never().is_none(), "an idle guest leaves the pipe pending");

    let outcome = resident.stop();

    assert_eq!(outcome.failure, None);
    assert_eq!(outcome.report.end, Some(End::Stopped));
    assert_eq!(host.from_guest.call(()).now_or_never().expect("ready"), Err(FrameError::Closed));
}

#[test]
fn ringing_the_doorbell_without_a_stop_leaves_the_guest_running() {
    let (resident, host) = ResidentLinux::start(boot_inputs(), link()).expect("the resident guest starts");

    for _ in 0..RINGS {
        host.doorbell.ring();
    }

    assert!(resident.is_running(), "a ring with no stop request never ends the run");
    let outcome = resident.stop();
    assert_eq!(outcome.failure, None);
    assert_eq!(outcome.report.end, Some(End::Stopped), "the only end this run saw is the stop");
}

#[test]
fn dropping_the_resident_guest_stops_it_and_closes_the_pipe() {
    let (resident, host) = ResidentLinux::start(boot_inputs(), link()).expect("the resident guest starts");

    drop(resident);

    assert_eq!(host.from_guest.call(()).now_or_never().expect("ready"), Err(FrameError::Closed));
}

#[test]
fn a_rejected_kernel_is_a_start_error_naming_the_cause() {
    let mut inputs = boot_inputs();
    inputs.kernel = Bytes::from_static(&[0x7f; 4096]);

    let error = ResidentLinux::start(inputs, link()).err().expect("a non-bzImage cannot boot");

    assert!(error.to_string().contains("kernel rejected"), "{error}");
}
