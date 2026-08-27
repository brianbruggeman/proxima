//! The M5b PSCI gate, as tests: a real guest issues `PSCI_VERSION` via
//! `hvc #0` and observes the host's answer, then `SYSTEM_OFF` cleanly ends
//! the run — proving `src/psci.rs`'s handler is reached through the same
//! real vCPU trap loop `tests/dispatch_hypercall.rs` already drives for the
//! `ChildRequest` hypercall path, not a synthesized in-memory call.
//!
//! [`SignedPsciProbe`] mirrors `tests/dispatch_hypercall.rs`'s `SignedProbe`
//! exactly (same codesign-in-tempdir discipline `tests/boot.rs`'s
//! `SignedGuest` established) but builds and drives the dedicated
//! `proxima-vm-guest-lambda-psci` binary
//! (`guests/lambda/src/bin/psci_probe.rs`) instead of the default
//! `proxima-vm-guest-lambda`, so this PSCI call sequence never touches the
//! other guest's pinned ELF layout or pinned emitted-byte contract.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::path::{Path, PathBuf};
use std::process::Command;

use tempfile::TempDir;

const PROBE_BINARY: &str = env!("CARGO_BIN_EXE_dispatch_probe");
const ENTITLEMENTS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/entitlements.plist");
const GUEST_MANIFEST_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/guests/lambda");

const HYPERVISOR_LANE: bool = cfg!(all(target_os = "macos", target_arch = "aarch64"));

/// PSCI 0.2 version this guest expects back: major 0, minor 2, little-endian
/// `u32` — matches `src/psci.rs`'s `PSCI_VERSION_VALUE` and `src/dtb.rs`'s
/// advertised `arm,psci-0.2` compatible string.
const EXPECTED_PSCI_VERSION_BYTES: [u8; 4] = [0x02, 0x00, 0x00, 0x00];

/// Builds `proxima-vm-guest-lambda-psci` for `aarch64-unknown-none` into a
/// per-case tempdir's own `CARGO_TARGET_DIR` — same isolation
/// `tests/dispatch_hypercall.rs::build_guest_elf` uses so parallel nextest
/// cases never race on a shared build output.
fn build_psci_guest_elf(target_dir: &Path) -> PathBuf {
    let status = Command::new("cargo")
        .args([
            "build",
            "--manifest-path",
            &format!("{GUEST_MANIFEST_DIR}/Cargo.toml"),
            "--bin",
            "proxima-vm-guest-lambda-psci",
            "--target",
            "aarch64-unknown-none",
            "--release",
        ])
        .env("CARGO_TARGET_DIR", target_dir)
        .status()
        .expect("run cargo build for the psci probe guest ELF");
    assert!(status.success(), "psci probe guest ELF build failed");
    target_dir
        .join("aarch64-unknown-none")
        .join("release")
        .join("proxima-vm-guest-lambda-psci")
}

struct SignedPsciProbe {
    _directory: TempDir,
    path: PathBuf,
    guest_elf: PathBuf,
}

impl SignedPsciProbe {
    fn prepare() -> Self {
        let directory = tempfile::tempdir().expect("create tempdir for the signed probe");
        let path = directory.path().join("psci-dispatch-probe");
        std::fs::copy(Path::new(PROBE_BINARY), &path).expect("copy the built probe binary");
        let guest_elf = build_psci_guest_elf(&directory.path().join("guest-target"));

        if HYPERVISOR_LANE {
            let status = Command::new("codesign")
                .arg("--force")
                .arg("--sign")
                .arg("-")
                .arg("--entitlements")
                .arg(ENTITLEMENTS)
                .arg(&path)
                .status()
                .expect("run codesign");
            assert!(status.success(), "codesign failed for {}", path.display());
        }

        Self {
            _directory: directory,
            path,
            guest_elf,
        }
    }

    /// Runs the probe against the PSCI guest; the dispatcher's
    /// `configured_response` variant is irrelevant here since this guest
    /// never issues a `ChildRequest` hypercall, so `"read"` is passed
    /// unconditionally.
    fn run(&self) -> Result<(Vec<u8>, String), String> {
        let output = Command::new(&self.path)
            .arg(&self.guest_elf)
            .arg("read")
            .output()
            .expect("run the probe");
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
        if output.status.success() {
            Ok((output.stdout, stderr))
        } else {
            Err(stderr)
        }
    }
}

/// PSCI/HVC is this VM's ARM-specific M5b track (`src/dtb.rs`'s advertised
/// conduit is `hvc`, never `smc`) — the KVM/x86_64 lane has no PSCI concept,
/// so unlike `tests/dispatch_hypercall.rs` this gate has no KVM-lane arm.
#[proxima::test]
async fn guest_psci_version_call_reaches_the_host_handler_through_a_real_vm_exit() {
    if !HYPERVISOR_LANE {
        return;
    }

    let probe = SignedPsciProbe::prepare();
    let (emitted, _stderr) = probe.run().expect("psci probe runs");

    assert_eq!(
        emitted, EXPECTED_PSCI_VERSION_BYTES,
        "the guest must emit the host's PSCI_VERSION answer (major 0, minor 2, \
         little-endian) recovered from a real VM exit, not a guest-compiled constant"
    );
}

/// `SYSTEM_OFF` must end the dispatch loop cleanly (`proxima_vm_run_dispatch_loop`
/// exits 0) rather than being treated as an unrecognized hypercall or hanging
/// until the hypercall budget is exceeded.
#[proxima::test]
async fn guest_system_off_call_cleanly_ends_the_run() {
    if !HYPERVISOR_LANE {
        return;
    }

    let probe = SignedPsciProbe::prepare();
    let result = probe.run();

    assert!(
        result.is_ok(),
        "a guest ending its run via PSCI SYSTEM_OFF must be reported as a clean exit: {result:?}"
    );
}
