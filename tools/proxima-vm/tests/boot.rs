//! The boot gate, as tests.
//!
//! Hypervisor.framework answers `hv_vm_create` with `HV_DENIED` (`0xfae94007`)
//! for a process that does not carry `com.apple.security.hypervisor`, and
//! applying that entitlement is a post-link `codesign` step cargo has no hook
//! for. So the harness signs the guest itself: each case copies the built
//! binary into its own tempdir before signing, because nextest runs cases in
//! parallel processes and concurrent signing of one path corrupts it.
//!
//! Every case asserts the guest's OUTPUT BYTES. An exit status cannot separate
//! a guest that emitted everything from one that emitted nothing.

// the workspace lint set denies expect_used everywhere; test modules opt out the
// same way the in-source `mod tests` blocks do.
#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::path::{Path, PathBuf};
use std::process::Command;

use tempfile::TempDir;

const GUEST_BINARY: &str = env!("CARGO_BIN_EXE_hello");
const ENTITLEMENTS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/entitlements.plist");

const HYPERVISOR_LANE: bool = cfg!(all(target_os = "macos", target_arch = "aarch64"));
const KVM_LANE: bool = cfg!(all(target_os = "linux", target_arch = "x86_64"));

/// A signed, isolated copy of the guest binary. The tempdir is held so the
/// executable outlives the run.
struct SignedGuest {
    _directory: TempDir,
    path: PathBuf,
}

impl SignedGuest {
    fn prepare() -> Self {
        let directory = tempfile::tempdir().expect("create tempdir for the signed guest");
        let path = directory.path().join("guest");
        std::fs::copy(Path::new(GUEST_BINARY), &path).expect("copy the built guest binary");

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
        }
    }

    fn emit(&self, message: Option<&str>) -> Result<Vec<u8>, String> {
        let mut command = Command::new(&self.path);
        if let Some(message) = message {
            command.arg(message);
        }
        let output = command.output().expect("run the guest");
        if output.status.success() {
            Ok(output.stdout)
        } else {
            Err(String::from_utf8_lossy(&output.stderr).into_owned())
        }
    }
}

fn lane_is_supported() -> bool {
    HYPERVISOR_LANE || KVM_LANE
}

#[proxima::test]
#[case::default_greeting(None, "hello from proxima-vm\n")]
#[case::caller_supplied_message(Some("a different guest payload"), "a different guest payload")]
#[case::single_byte(Some("x"), "x")]
async fn guest_emits_exactly_the_bytes_it_was_given(
    #[case] message: Option<&str>,
    #[case] expected: &str,
) {
    if !lane_is_supported() {
        return;
    }

    let guest = SignedGuest::prepare();
    let observed = guest.emit(message).expect("guest runs");

    assert_eq!(
        observed.len(),
        expected.len(),
        "guest emitted {} bytes, expected {}",
        observed.len(),
        expected.len()
    );
    assert_eq!(String::from_utf8_lossy(&observed), expected);
}

/// The degenerate control. A harness that reports success for a guest which
/// emitted nothing is measuring the process exit status, not the guest — and
/// then every byte count it reports for a real guest is worthless.
#[proxima::test]
async fn empty_guest_emits_zero_bytes() {
    if !lane_is_supported() {
        return;
    }

    let guest = SignedGuest::prepare();
    let observed = guest.emit(Some("")).expect("guest runs");

    assert_eq!(observed.len(), 0, "empty guest emitted {observed:?}");
}

/// Sad path: without the entitlement the hypervisor refuses, and the failure
/// must surface as a named error rather than a silent empty success.
#[proxima::test]
async fn unsigned_guest_is_refused_by_the_hypervisor() {
    if !HYPERVISOR_LANE {
        return;
    }

    let directory = tempfile::tempdir().expect("create tempdir");
    let path = directory.path().join("unsigned-guest");
    std::fs::copy(Path::new(GUEST_BINARY), &path).expect("copy the built guest binary");
    Command::new("codesign")
        .arg("--force")
        .arg("--sign")
        .arg("-")
        .arg(&path)
        .status()
        .expect("run codesign without entitlements");

    let output = Command::new(&path).output().expect("run the guest");

    assert!(!output.status.success(), "unsigned guest unexpectedly ran");
    assert!(
        output.stdout.is_empty(),
        "unsigned guest emitted bytes: {:?}",
        output.stdout
    );
    let reported = String::from_utf8_lossy(&output.stderr);
    assert!(
        reported.contains("hv_vm_create"),
        "denial did not name the failing call: {reported}"
    );
}
