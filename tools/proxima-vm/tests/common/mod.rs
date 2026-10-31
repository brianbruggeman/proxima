#![allow(dead_code, clippy::expect_used, clippy::unwrap_used)]

use std::path::PathBuf;
use std::process::Command;
use std::sync::OnceLock;

use proxima_core::ProximaError;
use proxima_vm::linux::{LinuxGuestLimits, LinuxGuestRun, run_linux_initramfs};

pub const CMDLINE: &str = "console=ttyS0 reboot=t panic=-1 nomodules i8042.noaux loglevel=6";
const DEFAULT_KERNEL: &str = "/boot/vmlinuz-linux";
const GUEST_MANIFEST: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/guests/linux-init/Cargo.toml");
const GUEST_TARGET_DIR: &str = concat!(env!("CARGO_TARGET_TMPDIR"), "/linux-guest");
const GUEST_TARGET: &str = "x86_64-unknown-linux-musl";

pub fn kernel() -> &'static [u8] {
    static KERNEL: OnceLock<Vec<u8>> = OnceLock::new();
    KERNEL.get_or_init(|| {
        let path = std::env::var("PROXIMA_VM_KERNEL").unwrap_or_else(|_| DEFAULT_KERNEL.to_string());
        std::fs::read(&path).unwrap_or_else(|error| panic!("kernel {path} must be readable: {error}"))
    })
}

pub fn guest(name: &str) -> Vec<u8> {
    let status = Command::new("cargo")
        .args(["build", "--release", "--target", GUEST_TARGET, "--manifest-path", GUEST_MANIFEST])
        .env("CARGO_TARGET_DIR", GUEST_TARGET_DIR)
        .status()
        .expect("spawn cargo to build the linux guests");
    assert!(status.success(), "linux guest build failed");
    let path: PathBuf = [GUEST_TARGET_DIR, GUEST_TARGET, "release", name].iter().collect();
    std::fs::read(&path).unwrap_or_else(|error| panic!("guest {} must exist: {error}", path.display()))
}

pub fn try_boot_image(initramfs: &[u8], linger_millis: u64, watchdog_millis: u64) -> Result<LinuxGuestRun, ProximaError> {
    let limits = LinuxGuestLimits { watchdog_millis, ..LinuxGuestLimits::default() };
    run_linux_initramfs(kernel(), initramfs, CMDLINE, linger_millis.to_string().as_bytes(), &limits)
}

pub fn boot_image(initramfs: &[u8], linger_millis: u64, watchdog_millis: u64) -> LinuxGuestRun {
    try_boot_image(initramfs, linger_millis, watchdog_millis).unwrap_or_else(|error| panic!("oci guest failed: {error}"))
}
