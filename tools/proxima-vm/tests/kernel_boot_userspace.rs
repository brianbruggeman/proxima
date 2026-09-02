//! Past M5's own exit criterion (bytes crossed the pl011 channel) to a real
//! PID 1: with `nosmp`/`nr_cpus=1` routing around `ICC_SGI1R_EL1` (the SMP
//! IPI wall this crate's own boot investigation named next) and an
//! initramfs threaded into RAM (`boot::boot_linux_kernel`'s own `initramfs`
//! parameter, `crate::boot::INITRD_OFFSET`'s module doc), the kernel has
//! everything `Documentation/arm64/booting` requires to exec `/init` — this
//! file's own exit criterion is that PID 1's own marker string,
//! `PROXIMA-INIT-ALIVE`, is a substring of the captured pl011 byte stream.
//!
//! Gates on BOTH a real arm64 Linux `Image` (`PROXIMA_VM_KERNEL_BOOT_IMAGE`,
//! `tests/kernel_boot.rs`'s own env var) AND an initramfs `cpio` archive
//! (`PROXIMA_VM_INITRAMFS`) existing on this host — the same "asset absent
//! -> named UNMEASURED skip" shape every asset-gated test in this crate
//! uses, since neither a 38 MiB kernel binary nor a built initramfs has any
//! business living in this repo.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::env;
use std::path::Path;
use std::process::Command;

const PROBE_BINARY: &str = env!("CARGO_BIN_EXE_kernel_boot_probe");
const ENTITLEMENTS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/entitlements.plist");
const IMAGE_PATH_VAR: &str = "PROXIMA_VM_KERNEL_BOOT_IMAGE";
const INITRAMFS_PATH_VAR: &str = "PROXIMA_VM_INITRAMFS";

const HYPERVISOR_LANE: bool = cfg!(all(target_os = "macos", target_arch = "aarch64"));

/// PID 1's own liveness marker (this file's module doc names it), written
/// by the static `/init` this initramfs carries before it does anything
/// else — the earliest possible evidence userspace was reached at all.
const INIT_ALIVE_MARKER: &str = "PROXIMA-INIT-ALIVE";

/// Resolves one required asset path, or `None` if it is not present on this
/// host — the honest UNMEASURED answer, mirrored from
/// `tests/kernel_boot.rs`'s own `kernel_image_path`.
fn asset_path(env_var: &str) -> Option<String> {
    let path = env::var(env_var).ok()?;
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

/// Signs and runs `kernel_boot_probe` against both assets — the same
/// signed-subprocess shape `tests/kernel_boot.rs`'s own `run_signed_probe`
/// uses, duplicated here rather than shared across the two test binaries
/// (integration test binaries in this crate do not share a common `mod`;
/// nextest compiles each `tests/*.rs` file as its own crate).
fn run_signed_probe(image_path: &str, initramfs_path: &str) -> ProbeRun {
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

    let output = Command::new(&probe_path)
        .arg(image_path)
        .arg(initramfs_path)
        .output()
        .expect("run the kernel boot probe");
    ProbeRun {
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        pl011_bytes: output.stdout,
    }
}

#[proxima::test]
async fn init_alive_marker_arrives_through_the_pl011_byte_channel() {
    if !HYPERVISOR_LANE {
        return;
    }
    let Some(image_path) = asset_path(IMAGE_PATH_VAR) else {
        eprintln!(
            "UNMEASURED-kernel-image-not-present: set {IMAGE_PATH_VAR} to a real arm64 Linux \
             Image to run this boot"
        );
        return;
    };
    let Some(initramfs_path) = asset_path(INITRAMFS_PATH_VAR) else {
        eprintln!(
            "UNMEASURED-initramfs-not-present: set {INITRAMFS_PATH_VAR} to a built initramfs \
             cpio archive to run this boot"
        );
        return;
    };

    let probe = run_signed_probe(&image_path, &initramfs_path);
    let console_text = String::from_utf8_lossy(&probe.pl011_bytes);

    eprintln!("probe stderr:\n{}", probe.stderr);
    eprintln!(
        "full pl011 console text ({} bytes):\n{console_text}",
        probe.pl011_bytes.len()
    );

    // A panic AFTER the marker crossed the channel is a pass -- this file's
    // own module doc names the exit criterion as the marker, not a clean
    // process exit, the identical M5b contract `tests/kernel_boot.rs`
    // already established for the pre-userspace boot.
    assert!(
        console_text.contains(INIT_ALIVE_MARKER),
        "{INIT_ALIVE_MARKER} must appear in the captured pl011 byte stream; probe stderr:\n{}\n\
         full console text:\n{console_text}",
        probe.stderr
    );

    // Strengthens (never weakens) the marker assertion above: PID1's own
    // idle park loop issues `wfi`, which `backend_macos.c`'s EC-0x1 arm now
    // services as a yield-and-resume instead of surfacing as this exact
    // "unexpected arm exception class 0x1" failure
    // (`boot::boot_linux_kernel`'s own doc on `ProximaError::Upstream`) --
    // a run that reached the marker AND still hit this wall would mean the
    // trap fired on some other unhandled EC, not the one this task fixed.
    assert!(
        !probe.stderr.contains("unexpected arm exception class 0x1 "),
        "the run must not end on the EC-0x1 (wfi/wfe) trap that used to abort this boot; \
         probe stderr:\n{}",
        probe.stderr
    );
}
