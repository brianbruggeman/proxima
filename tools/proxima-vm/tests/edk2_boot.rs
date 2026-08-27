//! Standard-boot-path rung above M5's real-Linux-kernel boot
//! (`tests/kernel_boot.rs`'s own module doc): attempts a real edk2/AAVMF
//! `-pflash` CODE volume boot through `boot::boot_edk2_firmware`, the exact
//! "flash at guest-physical 0, EL2h entry, DTB in `x0`" shape this crate's
//! own edk2-boot investigation designed (`boot.rs`'s own module doc on
//! [`boot::boot_edk2_firmware`]).
//!
//! Gates on a real edk2 `-pflash` CODE volume existing at the path named by
//! `PROXIMA_VM_EDK2_FIRMWARE` — the same "asset absent -> named UNMEASURED
//! skip" shape `tests/kernel_boot.rs`'s own `kernel_image_path` uses, since
//! a 64 MiB firmware binary has no business living in this repo. On a host
//! with Homebrew's `qemu` cask installed, that path is conventionally
//! `/opt/homebrew/share/qemu/edk2-aarch64-code.fd`.
//!
//! Success, per this task's own exit criterion: edk2's own SEC/PEI/DXE
//! debug-console banner text arriving on the pl011 byte channel — nonzero
//! bytes, asserted the same shape `tests/kernel_boot.rs`'s own M5 assertion
//! uses. A firmware that hangs before any byte crosses the channel is a
//! real, reportable outcome (this file's own instrumentation prints the
//! full M3/M5b trap-statistics line either way, `edk2_boot_probe.rs`'s own
//! doc on why bytes and the loop outcome travel independently) — it is not
//! this test's job to turn that outcome into a pass.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::env;
use std::path::Path;
use std::process::Command;

const PROBE_BINARY: &str = env!("CARGO_BIN_EXE_edk2_boot_probe");
const ENTITLEMENTS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/entitlements.plist");
const FIRMWARE_PATH_VAR: &str = "PROXIMA_VM_EDK2_FIRMWARE";

const HYPERVISOR_LANE: bool = cfg!(all(target_os = "macos", target_arch = "aarch64"));

/// Resolves the edk2 CODE flash volume under test, or `None` if the asset
/// is not present on this host — the honest UNMEASURED answer, mirrored
/// from `tests/kernel_boot.rs`'s own `kernel_image_path`.
fn firmware_path() -> Option<String> {
    let path = env::var(FIRMWARE_PATH_VAR).ok()?;
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

/// Signs and runs `edk2_boot_probe` against `firmware_path`, the same
/// signed-subprocess shape `tests/kernel_boot.rs`'s own `run_signed_probe`
/// uses (HVF denies `hv_vm_create` to a process without the hypervisor
/// entitlement). The probe's own exit status is deliberately NOT asserted
/// here, for the identical reason `kernel_boot.rs`'s own doc names: this
/// boot's exit criterion is bytes on the pl011 channel, not a clean process
/// exit — a firmware build that reaches its own console banner and then
/// hits an unmodeled wall still counts as real, reportable evidence.
fn run_signed_probe(firmware_path: &str) -> ProbeRun {
    let directory = tempfile::tempdir().expect("create tempdir for the signed probe");
    let probe_path = directory.path().join("edk2-boot-probe");
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
        .arg(firmware_path)
        .output()
        .expect("run the edk2 boot probe");
    ProbeRun {
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        pl011_bytes: output.stdout,
    }
}

#[proxima::test]
async fn edk2_console_output_arrives_through_the_pl011_byte_channel() {
    if !HYPERVISOR_LANE {
        return;
    }
    let Some(firmware_path) = firmware_path() else {
        eprintln!(
            "UNMEASURED-edk2-firmware-not-present: set {FIRMWARE_PATH_VAR} to a real edk2 \
             -pflash CODE volume (conventionally \
             /opt/homebrew/share/qemu/edk2-aarch64-code.fd on a Homebrew qemu install) to run \
             this boot"
        );
        return;
    };

    let probe = run_signed_probe(&firmware_path);
    let console_text = String::from_utf8_lossy(&probe.pl011_bytes);

    eprintln!("probe stderr:\n{}", probe.stderr);
    eprintln!(
        "full pl011 console text ({} bytes):\n{console_text}",
        probe.pl011_bytes.len()
    );

    assert!(
        !probe.pl011_bytes.is_empty(),
        "edk2-emitted pl011 byte count must be nonzero (the SEC/PEI/DXE debug console banner); \
         probe stderr:\n{}",
        probe.stderr
    );
}
