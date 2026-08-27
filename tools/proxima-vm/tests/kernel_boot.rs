//! M5's exit criterion (`ROADMAP.md`'s M5 section): "kernel console output
//! arrives through our byte channel; the assertion is on kernel-emitted
//! bytes and the count is asserted nonzero."
//!
//! Gates on a real arm64 Linux `Image` file existing at the path named by
//! `PROXIMA_VM_KERNEL_BOOT_IMAGE` — the same "asset absent -> named
//! UNMEASURED skip" shape `tests/dtb_dtc_differential.rs` already uses for
//! `dtc`, since a 38 MiB kernel binary has no business living in this repo.
//!
//! Runs the boot from a signed subprocess (`kernel_boot_probe`), the same
//! shape every other real-hypervisor-exit test in this crate uses
//! (`tests/boot.rs`'s own module doc explains why: HVF denies `hv_vm_create`
//! to a process without the hypervisor entitlement, and codesigning the
//! whole nextest harness is not an option nextest's parallel test processes
//! would survive).

#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::env;
use std::path::Path;
use std::process::Command;

const PROBE_BINARY: &str = env!("CARGO_BIN_EXE_kernel_boot_probe");
const ENTITLEMENTS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/entitlements.plist");
const IMAGE_PATH_VAR: &str = "PROXIMA_VM_KERNEL_BOOT_IMAGE";

const HYPERVISOR_LANE: bool = cfg!(all(target_os = "macos", target_arch = "aarch64"));

/// Resolves the kernel `Image` under test, or `None` if the asset is not
/// present on this host — the honest UNMEASURED answer, not a hard failure.
fn kernel_image_path() -> Option<String> {
    let path = env::var(IMAGE_PATH_VAR).ok()?;
    if Path::new(&path).is_file() {
        Some(path)
    } else {
        None
    }
}

struct ProbeRun {
    stderr: String,
    pl011_bytes: Vec<u8>,
}

/// Signs and runs `kernel_boot_probe` against `image_path` (and, if given,
/// `initramfs_path`), exactly as M5b's own probe-process shape requires
/// (`tests/boot.rs`'s module doc). The probe's own exit status is
/// deliberately NOT asserted here: M5b's contract is bytes, not a clean
/// process exit (this file's own module doc), and the vtimer-arm
/// investigation's own next wall (`icc sysreg access rejected`) is exactly
/// the case where the probe exits nonzero while still carrying real pl011
/// bytes on stdout.
fn run_signed_probe(image_path: &str, initramfs_path: Option<&str>) -> ProbeRun {
    let directory = tempfile::tempdir().expect("create tempdir for the signed probe");
    let probe_path = directory.path().join("kernel-boot-probe");
    std::fs::copy(Path::new(PROBE_BINARY), &probe_path).expect("copy the built probe binary");

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

    let mut command = Command::new(&probe_path);
    command.arg(image_path);
    if let Some(initramfs_path) = initramfs_path {
        command.arg(initramfs_path);
    }
    let output = command.output().expect("run the kernel boot probe");
    ProbeRun {
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        pl011_bytes: output.stdout,
    }
}

/// Parses the `m5b gicd_trap_count=<n> gicr_trap_count=<n>
/// pl011_trap_count=<n> virtio_trap_count=<n> vtimer_activation_count=<n>`
/// summary line (`kernel_boot_probe.rs`'s own eprintln), the same
/// field=value parsing shape `fault_count_instrument.rs`'s `parse_m3_line`
/// already establishes for the sibling `m3 ...` line.
fn parse_m5b_line(stderr: &str) -> (u64, u64, u64, u64, u64) {
    let line = stderr
        .lines()
        .find(|line| line.starts_with("m5b "))
        .unwrap_or_else(|| panic!("no m5b summary line in probe stderr: {stderr}"));
    let mut fields = line.split_whitespace().skip(1);
    let parse_field = |field: Option<&str>| -> u64 {
        field
            .and_then(|entry| entry.split('=').nth(1))
            .and_then(|value| value.parse().ok())
            .unwrap_or_else(|| panic!("malformed m5b summary line: {line}"))
    };
    (
        parse_field(fields.next()),
        parse_field(fields.next()),
        parse_field(fields.next()),
        parse_field(fields.next()),
        parse_field(fields.next()),
    )
}

#[proxima::test]
async fn kernel_console_output_arrives_through_the_pl011_byte_channel() {
    if !HYPERVISOR_LANE {
        return;
    }
    let Some(image_path) = kernel_image_path() else {
        eprintln!(
            "UNMEASURED-kernel-image-not-present: set {IMAGE_PATH_VAR} to a real arm64 Linux \
             Image to run this boot"
        );
        return;
    };

    let probe = run_signed_probe(&image_path, None);

    eprintln!("probe stderr:\n{}", probe.stderr);
    eprintln!(
        "first 200 bytes of pl011 output (lossy utf8): {:?}",
        String::from_utf8_lossy(&probe.pl011_bytes[..probe.pl011_bytes.len().min(200)])
    );

    // The M5 exit criterion is bytes, not a clean process exit: a kernel
    // panicking with no initramfs (this boot supplies none) still counts,
    // as long as the panic message itself crossed the pl011 channel first.
    // Only a probe that produced NO bytes at all fails this gate.
    assert!(
        !probe.pl011_bytes.is_empty(),
        "kernel-emitted pl011 byte count must be nonzero; probe stderr:\n{}",
        probe.stderr
    );
}

/// M5b's earlycon root-cause: `pl011_trap_count` (the per-window
/// breakdown `backend_macos.c::handle_mmio_data_abort` now attributes)
/// must be nonzero for a real kernel boot that reaches earlycon —
/// otherwise the per-window instrument itself measures nothing, the same
/// degenerate-control argument `fault_count_instrument.rs`'s own zero-mmio
/// test makes for `mmio_trap_count`.
#[proxima::test]
async fn pl011_trap_count_is_nonzero_for_a_kernel_that_reaches_earlycon() {
    if !HYPERVISOR_LANE {
        return;
    }
    let Some(image_path) = kernel_image_path() else {
        eprintln!(
            "UNMEASURED-kernel-image-not-present: set {IMAGE_PATH_VAR} to a real arm64 Linux \
             Image to run this boot"
        );
        return;
    };

    let probe = run_signed_probe(&image_path, None);
    let (_gicd, _gicr, pl011_trap_count, _virtio, _vtimer) = parse_m5b_line(&probe.stderr);

    assert!(
        pl011_trap_count > 0,
        "a kernel boot with earlycon in its bootargs must trap into the pl011 window at \
         least once; probe stderr:\n{}",
        probe.stderr
    );
}
