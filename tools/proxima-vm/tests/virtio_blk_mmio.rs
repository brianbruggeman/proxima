//! M6 slice 6's exit proof: the lambda guest's virtio-blk mmio bring-up
//! sequence (`guests/lambda/src/virtio_blk.rs`) drives a real
//! `BlkConfigSpace`/`MmioDevice` FSM through real data-abort VM exits
//! (`src/backend_macos.c`'s `handle_mmio_data_abort`, routed to the third,
//! blk mmio window), and both an `IN` and an `OUT` request round-trip
//! through `BlkTransport::service_queue` walking the REAL descriptor/
//! avail/used rings out of guest memory the hypervisor itself mapped —
//! mirrors `tests/virtio_net_mmio.rs`'s shape exactly, for the third device
//! sharing the same exit-routing mechanism, and proves the console and net
//! windows both stay green with a third window now live.
//!
//! Reuses `dispatch_probe`, which prints the blk-serviced byte count and
//! bytes to stderr on its own dedicated line
//! (`crate::mmio_trampoline::proxima_vm_mmio_service_blk`'s encoding: per
//! request, an 8-byte little-endian sector, a 1-byte status, then the data
//! bytes actually moved), kept separate from the mmio/net/hypercall channels
//! so this test's assertion can never be satisfied by the wrong channel.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::indexing_slicing)]

use std::path::{Path, PathBuf};
use std::process::Command;

const PROBE_BINARY: &str = env!("CARGO_BIN_EXE_dispatch_probe");
const ENTITLEMENTS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/entitlements.plist");
const GUEST_MANIFEST_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/guests/lambda");

const HYPERVISOR_LANE: bool = cfg!(all(target_os = "macos", target_arch = "aarch64"));

const SECTOR_LEN: usize = 512;

/// The sector-0 pattern `dispatch.rs`'s `run_dispatch_loop` seeds into the
/// host's local block store, byte-identical to
/// `guests/lambda/src/virtio_blk.rs`'s own `EXPECTED_SECTOR_PATTERN`.
fn expected_pattern() -> Vec<u8> {
    (0..SECTOR_LEN).map(|index| (index % 256) as u8).collect()
}

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

/// Parses `dispatch_probe`'s `guest drained N blk byte(s): [a, b, c, ...]`
/// stderr line back into the raw `Vec<u8>` — the debug-formatted array is
/// too large (1042 bytes across two requests) to assert against as one
/// string literal, so this test verifies structure instead of a byte-exact
/// string match.
fn parse_blk_emitted(stderr: &str) -> Vec<u8> {
    let marker = "blk byte(s): [";
    let start = stderr.find(marker).expect("blk-emitted line present") + marker.len();
    let end = stderr[start..].find(']').expect("closing bracket present") + start;
    stderr[start..end]
        .split(", ")
        .filter(|token| !token.is_empty())
        .map(|token| token.parse::<u8>().expect("byte token parses"))
        .collect()
}

/// This proof runs only on the hypervisor lane: `handle_mmio_data_abort`
/// (`src/backend_macos.c`) is aarch64/HVF-only per this slice's scope —
/// `backend_linux.c`'s mirror compiles but does not yet decode
/// `KVM_EXIT_MMIO`.
#[proxima::test]
async fn guest_virtio_blk_requests_round_trip_through_a_real_vm_exit() {
    if !HYPERVISOR_LANE {
        return;
    }

    let directory = tempfile::tempdir().expect("create tempdir for the signed probe");
    let probe_path = directory.path().join("dispatch-probe");
    std::fs::copy(Path::new(PROBE_BINARY), &probe_path).expect("copy the built probe binary");
    let guest_elf = build_guest_elf(&directory.path().join("guest-target"));

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
        .arg(&guest_elf)
        .arg("read")
        .output()
        .expect("run the probe");
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    assert!(output.status.success(), "probe failed: {stderr}");

    // console and net must both stay green with a third mmio window live.
    assert!(
        stderr.contains("guest drained 1 mmio byte(s): [171]"),
        "console window must stay green: {stderr}"
    );
    assert!(
        stderr.contains("guest drained 42 net byte(s):"),
        "net window must stay green: {stderr}"
    );

    let emitted = parse_blk_emitted(&stderr);
    let pattern = expected_pattern();
    let per_request_len = 8 + 1 + SECTOR_LEN;
    assert_eq!(
        emitted.len(),
        2 * per_request_len,
        "the guest's own comparison passed, so it must have submitted BOTH the \
         IN and the OUT request, each carrying one full sector: {stderr}"
    );

    // --- request 1: IN, sector 0, status OK, data == host-seeded pattern ---
    let (header_one, rest) = emitted.split_at(9);
    let (data_one, rest) = rest.split_at(SECTOR_LEN);
    assert_eq!(
        &header_one[0..8],
        &0u64.to_le_bytes(),
        "request one reads sector 0"
    );
    assert_eq!(header_one[8], 0, "request one status OK");
    assert_eq!(
        data_one, pattern,
        "the IN request's data must be the exact host-seeded pattern, proving \
         the guest's own byte-for-byte comparison (guests/lambda/src/virtio_blk.rs) \
         had something real to check"
    );

    // --- request 2: OUT, sector 1, status OK, data == the pattern written back ---
    let (header_two, data_two) = rest.split_at(9);
    assert_eq!(
        &header_two[0..8],
        &1u64.to_le_bytes(),
        "request two writes sector 1"
    );
    assert_eq!(header_two[8], 0, "request two status OK");
    assert_eq!(
        data_two, pattern,
        "the OUT request must carry the same pattern the guest verified on the way in"
    );
}
