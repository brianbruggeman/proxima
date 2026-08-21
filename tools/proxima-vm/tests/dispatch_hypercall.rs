//! The M1 hypercall-dispatch gate, as tests.
//!
//! `dispatch_probe` boots the same synthesized one-hypercall guest
//! `dispatch::run_hypercall_guest` builds, against a real hypervisor, and
//! writes the trampoline's postcard-encoded response to stdout. Like
//! `tests/boot.rs`'s `SignedGuest`, the harness signs its own per-case copy
//! of the probe binary — `hv_vm_create` answers `HV_DENIED` for an
//! unentitled process, and applying the entitlement is a post-link
//! `codesign` step cargo has no hook for. This file does not share that
//! helper with `tests/boot.rs`: the two are separate binaries under
//! nextest's per-case parallel processes, and `tests/boot.rs` stays
//! unmodified as the M0 gate.
//!
//! The case asserts the RESPONSE BYTES the trampoline wrote back, decoded
//! as the `ChildResponse` the probe's `RecordingDispatcher` was configured
//! with — proof that the real exit loop recovered the verb/pointer/length
//! registers correctly, translated the payload's guest address, and drove
//! it through the dispatcher, not just that the process exited zero.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::path::{Path, PathBuf};
use std::process::Command;

use proxima_protocols::process::{ChildResponse, ReadResponse};
use tempfile::TempDir;

const PROBE_BINARY: &str = env!("CARGO_BIN_EXE_dispatch_probe");
const ENTITLEMENTS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/entitlements.plist");

const HYPERVISOR_LANE: bool = cfg!(all(target_os = "macos", target_arch = "aarch64"));
const KVM_LANE: bool = cfg!(all(target_os = "linux", target_arch = "x86_64"));

struct SignedProbe {
    _directory: TempDir,
    path: PathBuf,
}

impl SignedProbe {
    fn prepare() -> Self {
        let directory = tempfile::tempdir().expect("create tempdir for the signed probe");
        let path = directory.path().join("dispatch-probe");
        std::fs::copy(Path::new(PROBE_BINARY), &path).expect("copy the built probe binary");

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

    fn run(&self) -> Result<Vec<u8>, String> {
        let output = Command::new(&self.path).output().expect("run the probe");
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
async fn dispatch_probe_hypercall_reaches_the_configured_response() {
    if !lane_is_supported() {
        return;
    }

    let probe = SignedProbe::prepare();
    let observed = probe.run().expect("probe runs");

    assert!(!observed.is_empty(), "probe wrote no response bytes");
    let decoded: ChildResponse =
        postcard::from_bytes(&observed).expect("probe stdout postcard-decodes");
    assert_eq!(
        decoded,
        ChildResponse::Read(ReadResponse {
            bytes: b"vm-side-canned".to_vec(),
            eof: true,
        })
    );
}
