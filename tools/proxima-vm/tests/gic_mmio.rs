//! The M5b GIC slice 3 gate, as tests: a real guest reads `GICD_PIDR2` and
//! `GICR_TYPER`'s low word through real data-abort VM exits and emits both
//! values back to the host — proving `src/gic.rs`'s `GicDistributor` and
//! `GicRedistributor` are reached through the exit loop's real MMIO trap
//! routing (`src/backend_macos.c`'s `handle_mmio_data_abort`,
//! `src/mmio_trampoline.rs`'s `proxima_vm_dispatch_mmio_gicd`/`_gicr`), not
//! a synthesized in-memory `GicAccess` call with no VM in the loop.
//!
//! [`SignedGicProbe`] mirrors `tests/psci_hvc.rs`'s `SignedPsciProbe`
//! exactly (same codesign-in-tempdir discipline `tests/boot.rs`'s
//! `SignedGuest` established) but builds and drives the dedicated
//! `proxima-vm-guest-lambda-gic` binary (`guests/lambda/src/bin/gic_probe.rs`)
//! instead of either sibling guest, so this GIC register-read sequence never
//! touches another guest's pinned ELF layout or pinned emitted-byte
//! contract.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::path::{Path, PathBuf};
use std::process::Command;

use tempfile::TempDir;

const PROBE_BINARY: &str = env!("CARGO_BIN_EXE_dispatch_probe");
const ENTITLEMENTS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/entitlements.plist");
const GUEST_MANIFEST_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/guests/lambda");

const HYPERVISOR_LANE: bool = cfg!(all(target_os = "macos", target_arch = "aarch64"));

/// `GICD_PIDR2`'s expected value: `ArchRev` (bits\[7:4\]) = 3 (GICv3), every
/// other field 0 — matches `src/gic.rs`'s `GICD_ARCH_REV_GICV3 << 4` and the
/// `linux_gic_v3_probe_sequence_...` unit test's own `0x30` assertion,
/// little-endian `u32`.
const EXPECTED_GICD_PIDR2_BYTES: [u8; 4] = [0x30, 0x00, 0x00, 0x00];

/// `GICR_TYPER`'s low word expected value: only `Last` (bit 4) set — matches
/// `src/gic.rs`'s `TYPER_LAST` for this crate's single-Redistributor,
/// single-vCPU model, little-endian `u32`.
const EXPECTED_GICR_TYPER_LOW_BYTES: [u8; 4] = [0x10, 0x00, 0x00, 0x00];

/// Builds `proxima-vm-guest-lambda-gic` for `aarch64-unknown-none` into a
/// per-case tempdir's own `CARGO_TARGET_DIR` — same isolation
/// `tests/psci_hvc.rs::build_psci_guest_elf` uses so parallel nextest cases
/// never race on a shared build output.
fn build_gic_guest_elf(target_dir: &Path) -> PathBuf {
    let status = Command::new("cargo")
        .args([
            "build",
            "--manifest-path",
            &format!("{GUEST_MANIFEST_DIR}/Cargo.toml"),
            "--bin",
            "proxima-vm-guest-lambda-gic",
            "--target",
            "aarch64-unknown-none",
            "--release",
        ])
        .env("CARGO_TARGET_DIR", target_dir)
        .status()
        .expect("run cargo build for the gic probe guest ELF");
    assert!(status.success(), "gic probe guest ELF build failed");
    target_dir
        .join("aarch64-unknown-none")
        .join("release")
        .join("proxima-vm-guest-lambda-gic")
}

struct SignedGicProbe {
    _directory: TempDir,
    path: PathBuf,
    guest_elf: PathBuf,
}

impl SignedGicProbe {
    fn prepare() -> Self {
        let directory = tempfile::tempdir().expect("create tempdir for the signed probe");
        let path = directory.path().join("gic-dispatch-probe");
        std::fs::copy(Path::new(PROBE_BINARY), &path).expect("copy the built probe binary");
        let guest_elf = build_gic_guest_elf(&directory.path().join("guest-target"));

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

    /// Runs the probe against the GIC guest; the dispatcher's
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

/// The GIC is this VM's ARM-specific M5b track — the KVM/x86_64 lane's
/// `backend_linux.c` does not yet decode `KVM_EXIT_MMIO` at all (its own
/// module doc names this), so unlike some of this crate's other gates this
/// one has no KVM-lane arm.
#[proxima::test]
async fn guest_reads_gicd_pidr2_and_gicr_typer_through_real_vm_exits() {
    if !HYPERVISOR_LANE {
        return;
    }

    let probe = SignedGicProbe::prepare();
    let (emitted, _stderr) = probe.run().expect("gic probe runs");

    let mut expected = Vec::new();
    expected.extend_from_slice(&EXPECTED_GICD_PIDR2_BYTES);
    expected.extend_from_slice(&EXPECTED_GICR_TYPER_LOW_BYTES);

    assert_eq!(
        emitted, expected,
        "the guest must emit GICD_PIDR2 (ArchRev=3) then GICR_TYPER's low word \
         (Last=1) recovered from real data-abort VM exits, not guest-compiled constants"
    );
}

/// The probe must end cleanly via `HALT_VERB` rather than being reported as
/// an unrecognized hypercall or hanging until the hypercall budget is
/// exceeded.
#[proxima::test]
async fn guest_halt_after_gic_reads_cleanly_ends_the_run() {
    if !HYPERVISOR_LANE {
        return;
    }

    let probe = SignedGicProbe::prepare();
    let result = probe.run();

    assert!(
        result.is_ok(),
        "a guest ending its run via HALT_VERB after two GIC register reads must be \
         reported as a clean exit: {result:?}"
    );
}
