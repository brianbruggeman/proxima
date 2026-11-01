//! A resident guest brings up eth0 on the virtio-mmio net window and the frame
//! it emits for a UDP send arrives on the host's `from_guest` pipe.

#![cfg(all(target_os = "linux", target_arch = "x86_64"))]
#![allow(clippy::expect_used, clippy::unwrap_used)]

mod common;

use std::path::Path;
use std::thread;
use std::time::{Duration, Instant};

use bytes::Bytes;
use futures::FutureExt;
use proxima_primitives::pipe::Pipe;
use proxima_protocols::inet::ethernet::{EtherType, EthernetFrame};
use proxima_vm::kernel_modules::load_modules;
use proxima_vm::kvm::resident::{LinkConfig, ResidentLinux, net_device_cmdline};
use proxima_vm::linux::{LinuxBootInputs, LinuxGuestLimits, initramfs_with_modules};
use proxima_vm::x86_boot::SetupHeader;

const GUEST_MAC: [u8; 6] = [0x02, 0x00, 0x00, 0x00, 0x00, 0x01];
const GUEST_ADDRESS: [u8; 4] = [10, 0, 2, 15];
const PEER_ADDRESS: [u8; 4] = [10, 0, 2, 2];
const MODULES_ROOT: &str = "/usr/lib/modules";
const ARP_REQUEST: u16 = 1;
const ARP_BODY_LEN: usize = 28;
const DEADLINE: Duration = Duration::from_secs(60);
const CMDLINE: &str = "console=ttyS0 reboot=t panic=-1 i8042.noaux loglevel=7";

fn kernel_release(image: &[u8]) -> String {
    let header = SetupHeader::parse(image).expect("the guest kernel has a setup header");
    let version = header.version_string(image).expect("the setup header names the kernel version");
    let release = version.split(|&byte| byte == b' ').next().expect("the version string starts with the release");
    String::from_utf8(release.to_vec()).expect("the kernel release is utf-8")
}

fn boot_inputs() -> LinuxBootInputs {
    let release_dir = Path::new(MODULES_ROOT).join(kernel_release(common::kernel()));
    let modules = load_modules(&release_dir, &["virtio_mmio", "virtio_net"]).expect("kernel modules are readable");
    let initramfs = initramfs_with_modules(&common::guest("net-init"), &modules).expect("initramfs builds");
    LinuxBootInputs {
        kernel: Bytes::from_static(common::kernel()),
        initramfs: Bytes::from(initramfs),
        cmdline: format!("{CMDLINE} {}", net_device_cmdline()),
        limits: LinuxGuestLimits::default(),
    }
}

fn is_guest_arp_request_for_peer(frame: &[u8]) -> bool {
    let Ok(ethernet) = EthernetFrame::parse(frame) else { return false };
    let body = ethernet.payload();
    ethernet.ether_type() == EtherType::Arp
        && ethernet.source() == GUEST_MAC
        && body.len() >= ARP_BODY_LEN
        && u16::from_be_bytes([body[6], body[7]]) == ARP_REQUEST
        && body[8..14] == GUEST_MAC
        && body[14..18] == GUEST_ADDRESS
        && body[24..28] == PEER_ADDRESS
}

#[test]
fn a_resident_guest_sends_an_arp_request_for_the_gateway_from_eth0() {
    let link = LinkConfig { mac: GUEST_MAC, frame_capacity: 64, host_consumer: Some(thread::current()) };
    let (resident, host) = ResidentLinux::start(boot_inputs(), link).expect("the resident guest starts");
    let deadline = Instant::now() + DEADLINE;
    let mut seen = Vec::new();

    let found = loop {
        match host.from_guest.call(()).now_or_never() {
            Some(Ok(frame)) if is_guest_arp_request_for_peer(&frame) => break true,
            Some(Ok(frame)) => seen.push(frame.len()),
            None => match deadline.checked_duration_since(Instant::now()) {
                Some(remaining) => thread::park_timeout(remaining),
                None => break false,
            },
            Some(Err(_)) => break false,
        }
    };

    let outcome = resident.stop();
    let console = String::from_utf8_lossy(&outcome.report.console);
    assert!(found, "no arp request from eth0; frames seen {seen:?}; failure {:?}; console:\n{console}", outcome.failure);
}
