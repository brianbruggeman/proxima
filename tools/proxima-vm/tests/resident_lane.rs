//! The resident Linux lane against a real kernel: the guest boots once on a
//! dedicated vCPU thread, keeps running when idle, and ends only when stopped.

#![cfg(all(target_os = "linux", target_arch = "x86_64"))]
#![allow(clippy::expect_used, clippy::unwrap_used)]

mod common;

use std::sync::Arc;

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
fn a_guest_that_ends_on_its_own_leaves_its_boot_console_in_the_outcome() {
    let mut inputs = boot_inputs();
    inputs.initramfs = Bytes::from(initramfs_from_init(&common::guest("uppercase-init")).expect("initramfs builds"));
    let (resident, host) = ResidentLinux::start(inputs, link()).expect("the resident guest starts");

    while futures::executor::block_on(host.from_guest.call(())).is_ok() {}
    let outcome = resident.stop();

    let console = String::from_utf8_lossy(&outcome.report.console);
    assert_ne!(outcome.report.end, Some(End::Stopped), "the guest ended before the stop request");
    assert!(console.contains("Linux version"), "console: {console}");
    assert_eq!(outcome.report.console_dropped, 0);
    assert!(outcome.report.console.len() <= LinuxGuestLimits::default().console_capacity);
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

struct Tracked {
    backing: Bytes,
    _owners: Arc<()>,
}

impl AsRef<[u8]> for Tracked {
    fn as_ref(&self) -> &[u8] {
        &self.backing
    }
}

fn tracked(backing: Bytes) -> (Bytes, Arc<()>) {
    let owners = Arc::new(());
    (Bytes::from_owner(Tracked { backing, _owners: Arc::clone(&owners) }), owners)
}

#[test]
fn a_serving_resident_guest_holds_no_host_copy_of_the_kernel_or_initramfs() {
    let mut inputs = boot_inputs();
    let (kernel, kernel_owners) = tracked(inputs.kernel.clone());
    let (initramfs, initramfs_owners) = tracked(inputs.initramfs.clone());
    inputs.kernel = kernel;
    inputs.initramfs = initramfs;
    assert_eq!(Arc::strong_count(&kernel_owners), 2, "the inputs own the kernel before start");

    let (resident, _host) = ResidentLinux::start(inputs, link()).expect("the resident guest starts");

    assert!(resident.is_running(), "the guest is serving when the count is read");
    assert_eq!(Arc::strong_count(&kernel_owners), 1, "the kernel is released once it is in guest ram");
    assert_eq!(Arc::strong_count(&initramfs_owners), 1, "the initramfs is released once it is in guest ram");
    drop(resident);
}

#[test]
fn a_rejected_kernel_is_a_start_error_naming_the_cause() {
    let mut inputs = boot_inputs();
    inputs.kernel = Bytes::from_static(&[0x7f; 4096]);

    let error = ResidentLinux::start(inputs, link()).err().expect("a non-bzImage cannot boot");

    assert!(error.to_string().contains("kernel rejected"), "{error}");
}
