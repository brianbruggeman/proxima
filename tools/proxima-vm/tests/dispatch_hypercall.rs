//! The M1 hypercall-dispatch gate, as tests: the guest issues its
//! `ChildRequest`s through a real `hvc #0` / `out dx, al` VM exit, not a
//! synthesized in-memory dispatch.
//!
//! [`SignedProbe`] builds the lambda guest crate for this lane's bare-metal
//! target (`Command::new("cargo")`, the same setup-inside-the-harness
//! discipline `tests/boot.rs`'s `SignedGuest` already uses for codesigning),
//! then signs and runs `dispatch_probe` against one of its bins.
//! `dispatch_probe` drives the guest through `dispatch::run_dispatch_loop` --
//! a real vCPU, a real trap per hypercall, and
//! `proxima_vm_dispatch_hypercall` recovering the request from guest memory
//! the hypervisor itself mapped.
//!
//! [`guest_hypercalls_reach_the_dispatcher_through_a_real_vm_exit`] and
//! [`different_inputs_change_the_guests_output`] together are the M1 exit
//! proof: the guest issuing `Read` then `Close` (two distinct verbs) is
//! provable only if the dispatcher recorded two distinct requests through the
//! real trap; and the SAME elf guest, run twice against two different
//! invocation inputs, must emit two different outputs -- proof the guest's
//! bytes came from what the host wrote back into guest memory after a real
//! exit, not a guest-compiled constant.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::path::{Path, PathBuf};
use std::process::Command;

use tempfile::TempDir;

const PROBE_BINARY: &str = env!("CARGO_BIN_EXE_dispatch_probe");
const ENTITLEMENTS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/entitlements.plist");
const GUEST_MANIFEST_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/guests/lambda");
const GUEST_TARGET_DIR: &str = concat!(env!("CARGO_TARGET_TMPDIR"), "/lambda-guest");

const HYPERVISOR_LANE: bool = cfg!(all(target_os = "macos", target_arch = "aarch64"));
const KVM_LANE: bool = cfg!(all(target_os = "linux", target_arch = "x86_64"));
const GUEST_TARGET: &str = if KVM_LANE {
    "x86_64-unknown-none"
} else {
    "aarch64-unknown-none"
};

const M1_GUEST: &str = "proxima-vm-guest-lambda";
const ELF_GUEST: &str = "uppercase";
const TWO_MIB: u64 = 2 * 1024 * 1024;
const SIXTY_FOUR_MIB: u64 = 64 * 1024 * 1024;

/// Builds every bin of the lambda guest crate for this lane's bare-metal
/// target (`aarch64-unknown-none` on HVF, `x86_64-unknown-none` on KVM) into
/// a `CARGO_TARGET_DIR` shared by every case: cargo serializes concurrent
/// builds on that directory's lock and later cases find it fresh, where a
/// per-case dir recompiled the whole dependency tree once per case.
fn build_guest_elfs(target_dir: &Path) {
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
}

struct SignedProbe {
    _directory: TempDir,
    path: PathBuf,
    guest_target_dir: PathBuf,
}

impl SignedProbe {
    fn prepare() -> Self {
        let directory = tempfile::tempdir().expect("create tempdir for the signed probe");
        let path = directory.path().join("dispatch-probe");
        std::fs::copy(Path::new(PROBE_BINARY), &path).expect("copy the built probe binary");
        let guest_target_dir = PathBuf::from(GUEST_TARGET_DIR);
        build_guest_elfs(&guest_target_dir);

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
            guest_target_dir,
        }
    }

    fn guest_elf(&self, bin: &str) -> PathBuf {
        self.guest_target_dir
            .join(GUEST_TARGET)
            .join("release")
            .join(bin)
    }

    /// Runs the probe against `bin` with `input` as the invocation body and
    /// `guest_memory_size` bytes of guest memory; `channel` selects whether
    /// stdout carries the M1 emit channel (`emitted`) or the invocation
    /// output (`output`). Returns `(stdout, stderr)` so callers can assert on
    /// both the guest's bytes and the requests the dispatcher recorded.
    fn run(
        &self,
        bin: &str,
        input: &str,
        channel: &str,
        guest_memory_size: u64,
    ) -> Result<(Vec<u8>, String), String> {
        let output = Command::new(&self.path)
            .arg(self.guest_elf(bin))
            .arg(input)
            .arg(channel)
            .arg(guest_memory_size.to_string())
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

fn lane_is_supported() -> bool {
    HYPERVISOR_LANE || KVM_LANE
}

#[proxima::test]
async fn guest_hypercalls_reach_the_dispatcher_through_a_real_vm_exit() {
    if !lane_is_supported() {
        return;
    }

    let probe = SignedProbe::prepare();
    let (emitted, stderr) = probe
        .run(M1_GUEST, "vm-side-input", "emitted", SIXTY_FOUR_MIB)
        .expect("probe runs");

    // the M1 guest re-emits the first byte of each reply: the `Read`
    // response's variant discriminant (0x00), then `Close`'s (0x03) -- the
    // two replies differ, so the guest's second emit cannot be a replay of
    // its first.
    assert_eq!(
        emitted,
        [0x00, 0x03],
        "the guest must emit the discriminant of each distinct reply, both \
         recovered from a real VM exit"
    );
    assert!(
        stderr.contains("guest issued 2 request(s)")
            && stderr.contains("Read {")
            && stderr.contains("Close {"),
        "the dispatcher must record exactly the guest's two distinct ChildRequest \
         verbs, in order, decoded from a real VM exit: {stderr}"
    );
}

#[proxima::test]
async fn different_inputs_change_the_guests_output() {
    if !lane_is_supported() {
        return;
    }

    let probe = SignedProbe::prepare();
    let (alpha, _) = probe
        .run(ELF_GUEST, "alpha invocation", "output", TWO_MIB)
        .expect("probe runs (alpha)");
    let (beta, _) = probe
        .run(ELF_GUEST, "beta", "output", TWO_MIB)
        .expect("probe runs (beta)");

    assert_eq!(alpha, b"ALPHA INVOCATION");
    assert_eq!(beta, b"BETA");
    assert_ne!(
        alpha, beta,
        "the same guest, driven through a real VM exit against two different \
         inputs, must not replay the same bytes -- proof the host's input, not \
         a guest-compiled constant, decides what the guest emits"
    );
}

#[proxima::test]
async fn elf_guest_output_does_not_depend_on_guest_memory_size() {
    if !lane_is_supported() {
        return;
    }

    let probe = SignedProbe::prepare();
    let (small, _) = probe
        .run(ELF_GUEST, "hello, lambda", "output", TWO_MIB)
        .expect("probe runs at 2 MiB");
    let (large, _) = probe
        .run(ELF_GUEST, "hello, lambda", "output", SIXTY_FOUR_MIB)
        .expect("probe runs at 64 MiB");

    assert_eq!(small, b"HELLO, LAMBDA");
    assert_eq!(large, small);
}

#[proxima::test]
async fn guest_memory_smaller_than_the_image_is_a_named_error() {
    if !lane_is_supported() {
        return;
    }

    let probe = SignedProbe::prepare();
    let stderr = probe
        .run(ELF_GUEST, "hello", "output", 65_536)
        .expect_err("a 64 KiB guest cannot hold the elf image");

    assert!(
        stderr.contains("guest_memory_size 65536 is too small"),
        "the failure must name the memory size and what the image needs: {stderr}"
    );
}

#[proxima::test]
async fn empty_input_produces_empty_output() {
    if !lane_is_supported() {
        return;
    }

    let probe = SignedProbe::prepare();
    let (output, stderr) = probe
        .run(ELF_GUEST, "", "output", TWO_MIB)
        .expect("probe runs on empty input");

    assert!(output.is_empty(), "empty input must give empty output: {output:?}");
    assert!(stderr.contains("guest issued 1 request(s)"), "{stderr}");
}
