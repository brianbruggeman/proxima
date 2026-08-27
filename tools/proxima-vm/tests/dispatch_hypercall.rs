//! The M1 hypercall-dispatch gate, as tests: the guest issues its
//! `ChildRequest`s through a real `hvc #0` VM exit, not a synthesized
//! in-memory dispatch.
//!
//! [`SignedProbe`] builds `proxima-vm-guest-lambda` for
//! `aarch64-unknown-none` (`Command::new("cargo")`, the same
//! setup-inside-the-harness discipline `tests/boot.rs`'s `SignedGuest`
//! already uses for codesigning), then signs and runs `dispatch_probe`
//! against it. `dispatch_probe` drives the guest through
//! `dispatch::run_dispatch_loop` — a real vCPU, a real `hvc #0` trap per
//! hypercall, and `proxima_vm_dispatch_hypercall` recovering the request
//! from guest memory the hypervisor itself mapped, per
//! `src/backend_macos.c`'s `proxima_vm_run_dispatch_loop`.
//!
//! [`guest_hypercalls_reach_the_dispatcher_through_a_real_vm_exit`] and
//! [`differently_configured_responses_change_the_guests_emitted_bytes`]
//! together are the M1 exit proof: the guest issuing `Read` then `Close`
//! (two distinct verbs) is provable only if the dispatcher recorded two
//! distinct requests through the real trap; and the SAME guest, run twice
//! against two differently `configured_response`d dispatchers, must emit
//! two different first bytes — proof the guest's emitted byte came from
//! bytes the host wrote back into guest memory after a real exit, not a
//! guest-compiled constant.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::path::{Path, PathBuf};
use std::process::Command;

use tempfile::TempDir;

const PROBE_BINARY: &str = env!("CARGO_BIN_EXE_dispatch_probe");
const ENTITLEMENTS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/entitlements.plist");
const GUEST_MANIFEST_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/guests/lambda");

const HYPERVISOR_LANE: bool = cfg!(all(target_os = "macos", target_arch = "aarch64"));
const KVM_LANE: bool = cfg!(all(target_os = "linux", target_arch = "x86_64"));

/// Builds `proxima-vm-guest-lambda` for `aarch64-unknown-none` into a
/// per-case tempdir's own `CARGO_TARGET_DIR`, so parallel nextest cases
/// never race on a shared build output — same discipline as
/// `tests/boot.rs`'s per-case signing tempdir.
fn build_guest_elf(target_dir: &Path) -> PathBuf {
    let status = Command::new("cargo")
        .args([
            "build",
            "--manifest-path",
            &format!("{GUEST_MANIFEST_DIR}/Cargo.toml"),
            "--target",
            "aarch64-unknown-none",
            "--release",
        ])
        .env("CARGO_TARGET_DIR", target_dir)
        .status()
        .expect("run cargo build for the guest ELF");
    assert!(status.success(), "guest ELF build failed");
    target_dir
        .join("aarch64-unknown-none")
        .join("release")
        .join("proxima-vm-guest-lambda")
}

struct SignedProbe {
    _directory: TempDir,
    path: PathBuf,
    guest_elf: PathBuf,
}

impl SignedProbe {
    fn prepare() -> Self {
        let directory = tempfile::tempdir().expect("create tempdir for the signed probe");
        let path = directory.path().join("dispatch-probe");
        std::fs::copy(Path::new(PROBE_BINARY), &path).expect("copy the built probe binary");
        let guest_elf = build_guest_elf(&directory.path().join("guest-target"));

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

    /// Runs the probe with `variant` selecting the dispatcher's
    /// `configured_response` ("read" or "close"); returns `(stdout,
    /// stderr)` so callers can assert on both the emitted bytes and the
    /// requests the dispatcher recorded.
    fn run(&self, variant: &str) -> Result<(Vec<u8>, String), String> {
        let output = Command::new(&self.path)
            .arg(&self.guest_elf)
            .arg(variant)
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
    let (emitted, stderr) = probe.run("read").expect("probe runs");

    // dispatcher answers every request with the same configured `Read`
    // response, so both hypercalls emit that response's postcard
    // discriminant (0x00) — the two distinct REQUESTS are what proves the
    // real trap decoded two different verbs, asserted below via the
    // recorded-request line the probe prints.
    assert_eq!(
        emitted,
        [0x00, 0x00],
        "the guest must emit the configured response's discriminant after each \
         of its two hypercalls, both recovered from a real VM exit"
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
async fn differently_configured_responses_change_the_guests_emitted_bytes() {
    if !lane_is_supported() {
        return;
    }

    let probe = SignedProbe::prepare();
    let (read_variant, _) = probe.run("read").expect("probe runs (read variant)");
    let (close_variant, _) = probe.run("close").expect("probe runs (close variant)");

    assert_eq!(read_variant, [0x00, 0x00]);
    assert_eq!(close_variant, [0x03, 0x03]);
    assert_ne!(
        read_variant, close_variant,
        "the same guest, driven through a real VM exit against two differently \
         configured dispatchers, must not replay the same emitted bytes — proof \
         the host's response, not a guest-compiled constant, decides what the \
         guest emits"
    );
}
