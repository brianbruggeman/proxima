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

pub const INDEX_DIGEST: &str = "sha256:1474027c316661cdec87df2623e13a41e7e1ce0ba99c24917631de8f300b5420";
pub const MANIFEST_DIGEST: &str = "sha256:7000846753fcc36eb2a1a3a21fe897da09c71e1d3b5381f912ea6e6e1c8871b6";
pub const CONFIG_DIGEST: &str = "sha256:ab0ca2603e47e42f7d45e094b6f4cecc5d5b7b0ef6ca89170ce1a771ed1a0066";
pub const ZONEINFO_LAYER: &str = "sha256:1b5c60a1be3bf9cb17566bdad4cb1f36387367e56798bc307d394169f1e5d72d";
pub const CERTS_LAYER: &str = "sha256:1d12c9005bc5cd6824edcfe37f412c35922508f76cb4b854beb52f85c12a929b";
pub const WHOAMI_LAYER: &str = "sha256:0abc37d8af53c11b5d94e3b54eca06ac582fec2a9b56f7d3070bb54247aa8595";
pub const WHOAMI_BLOBS: [&str; 6] = [INDEX_DIGEST, MANIFEST_DIGEST, CONFIG_DIGEST, ZONEINFO_LAYER, CERTS_LAYER, WHOAMI_LAYER];
const WHOAMI_FIXTURE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/oci/whoami-v1.10.2");

pub fn vendored(digest: &str) -> Vec<u8> {
    let name = digest.strip_prefix("sha256:").expect("sha256 digest");
    let raw: PathBuf = [WHOAMI_FIXTURE, "blobs", "sha256", name].iter().collect();
    match std::fs::read(&raw) {
        Ok(bytes) => bytes,
        Err(_) => {
            let hex_path = raw.with_extension("hex");
            let text = std::fs::read_to_string(&hex_path).unwrap_or_else(|error| panic!("vendored blob {} must exist raw or as .hex: {error}", raw.display()));
            decode_hex(&text)
        }
    }
}

fn decode_hex(text: &str) -> Vec<u8> {
    text.trim()
        .as_bytes()
        .chunks(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).expect("ascii hex"), 16).expect("hex digit pair"))
        .collect()
}
