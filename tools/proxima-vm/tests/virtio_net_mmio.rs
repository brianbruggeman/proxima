//! M6 slice 5's exit proof: the lambda guest's virtio-net mmio bring-up
//! sequence (`guests/lambda/src/virtio_net.rs`) drives a real
//! `NetConfigSpace`/`MmioDevice` FSM through real data-abort VM exits
//! (`src/backend_macos.c`'s `handle_mmio_data_abort`, routed to the net
//! window), and the ARP-request frame it publishes on transmitq1 arrives at
//! the host by way of `NetTransport::drain_tx` walking the REAL
//! descriptor/avail/used rings out of guest memory the hypervisor itself
//! mapped — mirrors `tests/virtio_console_mmio.rs`'s shape exactly, for the
//! second device sharing the same exit-routing mechanism.
//!
//! Reuses `dispatch_probe` (`src/bin/dispatch_probe.rs`), which already
//! prints the net-drained byte count and bytes to stderr on its own
//! dedicated line, kept separate from both the hypercall-emitted stdout
//! bytes and the console mmio-emitted stderr line so this test's assertion
//! can never be satisfied by the wrong channel.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::path::{Path, PathBuf};
use std::process::Command;

const PROBE_BINARY: &str = env!("CARGO_BIN_EXE_dispatch_probe");
const ENTITLEMENTS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/entitlements.plist");
const GUEST_MANIFEST_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/guests/lambda");

const HYPERVISOR_LANE: bool = cfg!(all(target_os = "macos", target_arch = "aarch64"));

/// The exact 42-byte ARP-request frame `guests/lambda/src/virtio_net.rs`
/// publishes on transmitq1 — Ethernet header (dst broadcast, src `PEER_MAC`)
/// followed by the ARP payload asking "who has" `TARGET_IP`. Byte-identical
/// to `src/virtio_net.rs`'s own `arp_request_frame()` host-side transport
/// test helper, so this end-to-end proof and that unit-level proof describe
/// the same wire bytes.
const EXPECTED_ARP_FRAME: [u8; 42] = [
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, // dst = broadcast
    0x02, 0x11, 0x22, 0x33, 0x44, 0x55, // src = PEER_MAC
    0x08, 0x06, // ethertype = ARP
    0x00, 0x01, // htype = Ethernet
    0x08, 0x00, // ptype = IPv4
    0x06, // hlen
    0x04, // plen
    0x00, 0x01, // oper = ARP_REQUEST
    0x02, 0x11, 0x22, 0x33, 0x44, 0x55, // sha = PEER_MAC
    10, 0, 0, 1, // spa
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // tha = unknown
    10, 0, 0, 2, // tpa = TARGET_IP
];

fn build_guest_elf(target_dir: &Path) -> PathBuf {
    let status = Command::new("cargo")
        .args([
            "build",
            "--manifest-path",
            &format!("{GUEST_MANIFEST_DIR}/Cargo.toml"),
            "--target",
            "aarch64-unknown-none",
            "--release",
        ])
        .env("CARGO_TARGET_DIR", target_dir)
        .status()
        .expect("run cargo build for the guest ELF");
    assert!(status.success(), "guest ELF build failed");
    target_dir
        .join("aarch64-unknown-none")
        .join("release")
        .join("proxima-vm-guest-lambda")
}

/// This proof runs only on the hypervisor lane: `handle_mmio_data_abort`
/// (`src/backend_macos.c`) is aarch64/HVF-only per this slice's scope —
/// `backend_linux.c`'s mirror compiles but does not yet decode
/// `KVM_EXIT_MMIO`.
#[proxima::test]
async fn guest_virtio_net_frame_arrives_through_a_real_vm_exit() {
    if !HYPERVISOR_LANE {
        return;
    }

    let directory = tempfile::tempdir().expect("create tempdir for the signed probe");
    let probe_path = directory.path().join("dispatch-probe");
    std::fs::copy(Path::new(PROBE_BINARY), &probe_path).expect("copy the built probe binary");
    let guest_elf = build_guest_elf(&directory.path().join("guest-target"));

    let status = Command::new("codesign")
        .arg("--force")
        .arg("--sign")
        .arg("-")
        .arg("--entitlements")
        .arg(ENTITLEMENTS)
        .arg(&probe_path)
        .status()
        .expect("run codesign");
    assert!(
        status.success(),
        "codesign failed for {}",
        probe_path.display()
    );

    let output = Command::new(&probe_path)
        .arg(&guest_elf)
        .arg("read")
        .output()
        .expect("run the probe");
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    assert!(output.status.success(), "probe failed: {stderr}");

    // `guests/lambda/src/virtio_net.rs`'s ARP_REQUEST_FRAME, published on a
    // two-descriptor (net_hdr + frame) chain and drained by
    // `NetTransport::drain_tx` after a real `QueueNotify` mmio exit on the
    // net window — the net_hdr is stripped by the transport before
    // delivery, so the drained bytes are exactly the 42-byte Ethernet
    // frame, byte-exact.
    let expected_line = format!(
        "guest drained {} net byte(s): {:?}",
        EXPECTED_ARP_FRAME.len(),
        EXPECTED_ARP_FRAME
    );
    assert!(
        stderr.contains(&expected_line),
        "the used-ring-completed ARP frame must reach the host through a real \
         data-abort VM exit on the net window, not a synthesized ring: {stderr}"
    );

    // stretch: the same bytes, fed to the real proxima-net stack, are
    // answered — closing guest -> VM exit -> codec -> real stack
    // end to end, not merely proving bytes crossed the ring.
    let mut frame = EXPECTED_ARP_FRAME;
    let our_mac = [0x02, 0x00, 0x00, 0x00, 0x00, 0x01];
    let our_ip = [10, 0, 0, 2];
    let action = proxima_net::stack::handle_frame(&mut frame, our_mac, our_ip);
    assert_eq!(
        action,
        proxima_net::stack::Action::Transmit,
        "the real proxima-net stack answers the guest's ARP request"
    );
    assert_eq!(
        &frame[0..6],
        &[0x02, 0x11, 0x22, 0x33, 0x44, 0x55],
        "reply eth dst is the guest"
    );
    assert_eq!(
        &frame[6..12],
        &our_mac,
        "reply eth src is the host's device MAC"
    );
}
