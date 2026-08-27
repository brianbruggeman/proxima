//! M5b's pl011 slice exit proof: the lambda guest's pl011 write sequence
//! (`guests/lambda/src/bin/pl011_probe.rs`) drives a real
//! `crate::pl011::Pl011Uart` register block through real data-abort VM exits
//! (`src/backend_macos.c`'s `handle_mmio_data_abort`), and the bytes it
//! writes to `UARTDR` arrive at the host through the pl011's own dedicated
//! channel (`dispatch::run_dispatch_loop`'s `pl011_emitted`) — this VM's
//! console byte channel, M5's exit criterion.
//!
//! Reuses `dispatch_probe` (`tests/gic_mmio.rs`/`tests/virtio_console_mmio.rs`
//! already exercise the same binary against their own dedicated guests): the
//! probe prints the pl011-drained byte count and bytes to stderr
//! (`src/bin/dispatch_probe.rs`), kept deliberately separate from stdout
//! (the hypercall-emitted bytes `tests/dispatch_hypercall.rs` asserts on) and
//! from every other device's own stderr line, so this test's assertion can
//! never be satisfied by the wrong channel.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::path::{Path, PathBuf};
use std::process::Command;

const PROBE_BINARY: &str = env!("CARGO_BIN_EXE_dispatch_probe");
const ENTITLEMENTS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/entitlements.plist");
const GUEST_MANIFEST_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/guests/lambda");

const HYPERVISOR_LANE: bool = cfg!(all(target_os = "macos", target_arch = "aarch64"));

fn build_pl011_guest_elf(target_dir: &Path) -> PathBuf {
    let status = Command::new("cargo")
        .args([
            "build",
            "--manifest-path",
            &format!("{GUEST_MANIFEST_DIR}/Cargo.toml"),
            "--bin",
            "proxima-vm-guest-lambda-pl011",
            "--target",
            "aarch64-unknown-none",
            "--release",
        ])
        .env("CARGO_TARGET_DIR", target_dir)
        .status()
        .expect("run cargo build for the pl011 probe guest ELF");
    assert!(status.success(), "pl011 probe guest ELF build failed");
    target_dir
        .join("aarch64-unknown-none")
        .join("release")
        .join("proxima-vm-guest-lambda-pl011")
}

/// This proof runs only on the hypervisor lane: `handle_mmio_data_abort`
/// (`src/backend_macos.c`) is aarch64/HVF-only per this slice's scope, the
/// same restriction every sibling mmio gate in this crate carries.
#[proxima::test]
async fn guest_pl011_uartdr_bytes_arrive_through_a_real_vm_exit() {
    if !HYPERVISOR_LANE {
        return;
    }

    let directory = tempfile::tempdir().expect("create tempdir for the signed probe");
    let probe_path = directory.path().join("pl011-dispatch-probe");
    std::fs::copy(Path::new(PROBE_BINARY), &probe_path).expect("copy the built probe binary");
    let guest_elf = build_pl011_guest_elf(&directory.path().join("guest-target"));

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

    // `guests/lambda/src/bin/pl011_probe.rs` writes b"OK" (79, 75) to
    // UARTDR, one byte per access, after polling UARTFR.TXFF clear for each.
    assert!(
        stderr.contains("guest drained 2 pl011 byte(s): [79, 75]"),
        "the bytes written to UARTDR must reach the host through a real \
         data-abort VM exit on the pl011's own dedicated channel: {stderr}"
    );
}

/// The probe must end cleanly via `HALT_VERB` after the write sequence
/// rather than being reported as an unrecognized hypercall or hanging until
/// the hypercall budget is exceeded.
#[proxima::test]
async fn guest_halt_after_pl011_writes_cleanly_ends_the_run() {
    if !HYPERVISOR_LANE {
        return;
    }

    let directory = tempfile::tempdir().expect("create tempdir for the signed probe");
    let probe_path = directory.path().join("pl011-dispatch-probe");
    std::fs::copy(Path::new(PROBE_BINARY), &probe_path).expect("copy the built probe binary");
    let guest_elf = build_pl011_guest_elf(&directory.path().join("guest-target"));

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

    assert!(
        output.status.success(),
        "a guest ending its run via HALT_VERB after the pl011 write sequence must be \
         reported as a clean exit: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}
