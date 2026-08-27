//! Codesigned probe, half one of two (`src/bin/snapshot_restore_probe.rs`
//! is the other half): `hv_vm_create` answers `HV_DENIED` for a process
//! lacking `com.apple.security.hypervisor` (`tests/boot.rs`'s own doc), and
//! applying that entitlement is a post-link `codesign` step cargo has no
//! hook for — so, exactly like `hello.rs`, the capture call happens in its
//! own small binary the integration test signs and drives.
//!
//! `argv[1]` is the message to snapshot; `argv[2]` is the file path this
//! probe writes the postcard-encoded [`proxima_vm::snapshot::VmSnapshot`]
//! to. Two processes, not two calls in one process: `hv_vm_create` is
//! once-per-process on the HVF lane (`proxima_vm::snapshot`'s own module
//! doc), so `snapshot_restore_probe` reads this file as an entirely
//! separate process rather than this one calling `restore` in-line.

use std::env;
use std::error::Error;
use std::fs;

use proxima_vm::snapshot::capture;

fn main() -> Result<(), Box<dyn Error>> {
    let message = env::args().nth(1).unwrap_or_default();
    let output_path = env::args()
        .nth(2)
        .ok_or("usage: snapshot_capture_probe <message> <output_path>")?;

    let snapshot = capture(message.as_bytes())?;
    fs::write(&output_path, snapshot.to_postcard_bytes()?)?;

    println!("emitted:{}", String::from_utf8_lossy(snapshot.emitted()));
    Ok(())
}
