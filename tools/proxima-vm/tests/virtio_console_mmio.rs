//! M6 slice 3's exit proof: the lambda guest's virtio-mmio bring-up
//! sequence (`guests/lambda/src/virtio_console.rs`) drives a real
//! `MmioDevice` FSM through real data-abort VM exits
//! (`src/backend_macos.c`'s `handle_mmio_data_abort`), and the byte it
//! publishes on the transmit queue arrives at the host by way of
//! `ConsoleTransport::drain_tx` walking the REAL descriptor/avail/used
//! rings out of guest memory the hypervisor itself mapped — not a
//! synthesized in-memory ring the way `proxima-protocols/src/virtio/mod.rs`'s
//! own worked-example test is (deliberately) "no VM in the loop".
//!
//! Reuses `dispatch_probe` (`tools/proxima-vm/tests/dispatch_hypercall.rs`
//! already exercises the same binary for the hypercall channel): the probe
//! prints the mmio-drained byte count and bytes to stderr
//! (`src/bin/dispatch_probe.rs`), kept deliberately separate from stdout
//! (the hypercall-emitted bytes `dispatch_hypercall.rs` asserts on) so
//! this test's assertion can never be satisfied by the wrong channel.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::path::{Path, PathBuf};
use std::process::Command;

const PROBE_BINARY: &str = env!("CARGO_BIN_EXE_dispatch_probe");
const ENTITLEMENTS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/entitlements.plist");
const GUEST_MANIFEST_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/guests/lambda");

const HYPERVISOR_LANE: bool = cfg!(all(target_os = "macos", target_arch = "aarch64"));
const KVM_LANE: bool = cfg!(all(target_os = "linux", target_arch = "x86_64"));
const GUEST_TARGET: &str = if KVM_LANE {
    "x86_64-unknown-none"
} else {
    "aarch64-unknown-none"
};

fn build_guest_elf(target_dir: &Path) -> PathBuf {
    let status = Command::new("cargo")
        .args([
            "build",
            "--manifest-path",
            &format!("{GUEST_MANIFEST_DIR}/Cargo.toml"),
            "--target",
            GUEST_TARGET,
            "--release",
        ])
        .env("CARGO_TARGET_DIR", target_dir)
        .status()
        .expect("run cargo build for the guest ELF");
    assert!(status.success(), "guest ELF build failed");
    target_dir
        .join(GUEST_TARGET)
        .join("release")
        .join("proxima-vm-guest-lambda")
}

/// Runs on both hypervisor lanes: a data-abort exit on HVF
/// (`src/backend_macos.c`'s `handle_mmio_data_abort`), `KVM_EXIT_MMIO` on KVM
/// (`src/kvm/devices.rs`'s `VirtioWindow`), both landing in the same Rust
/// trampolines.
#[proxima::test]
async fn guest_virtio_console_byte_arrives_through_a_real_vm_exit() {
    if !(HYPERVISOR_LANE || KVM_LANE) {
        return;
    }

    let directory = tempfile::tempdir().expect("create tempdir for the signed probe");
    let probe_path = directory.path().join("dispatch-probe");
    std::fs::copy(Path::new(PROBE_BINARY), &probe_path).expect("copy the built probe binary");
    let guest_elf = build_guest_elf(&directory.path().join("guest-target"));

    if HYPERVISOR_LANE {
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
    }

    let output = Command::new(&probe_path)
        .arg(&guest_elf)
        .arg("read")
        .output()
        .expect("run the probe");
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    assert!(output.status.success(), "probe failed: {stderr}");

    // `guests/lambda/src/virtio_console.rs`'s TX_BYTE constant (0xab),
    // published on a single-descriptor chain and drained by
    // `ConsoleTransport::drain_tx` after a real `QueueNotify` mmio exit.
    assert!(
        stderr.contains("guest drained 1 mmio byte(s): [171]"),
        "the used-ring-completed byte must reach the host through a real \
         data-abort VM exit, not a synthesized ring: {stderr}"
    );
}
