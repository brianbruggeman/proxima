//! Signed-subprocess probe for `dispatch::run_dispatch_loop`: reads the
//! already-built `proxima-vm-guest-lambda` ELF (path supplied as argv\[1\] by
//! `tests/dispatch_hypercall.rs`, which builds it for
//! `aarch64-unknown-none` before signing and running this probe), boots it
//! against a real hypervisor, and drives its two `ChildRequest` hypercalls
//! through a real VM exit. Writes the guest's emitted bytes to stdout as
//! raw bytes — `tests/dispatch_hypercall.rs` asserts on them directly, not
//! through a postcard decode, since they are the guest's own emitted-byte
//! proof, not a `ChildResponse`.
//!
//! argv\[2\] selects the dispatcher's `configured_response` variant ("read"
//! \[default\] or "close") — `tests/dispatch_hypercall.rs` runs this probe
//! once per variant against the identical guest and asserts the guest's
//! emitted bytes differ, proving the host's response (not a guest-compiled
//! constant) decides what the guest emits.

#[cfg(any(
    all(target_os = "linux", target_arch = "x86_64"),
    all(target_os = "macos", target_arch = "aarch64")
))]
use std::env;
use std::error::Error;
#[cfg(any(
    all(target_os = "linux", target_arch = "x86_64"),
    all(target_os = "macos", target_arch = "aarch64")
))]
use std::fs;
#[cfg(any(
    all(target_os = "linux", target_arch = "x86_64"),
    all(target_os = "macos", target_arch = "aarch64")
))]
use std::io::{self, Write};

#[cfg(any(
    all(target_os = "linux", target_arch = "x86_64"),
    all(target_os = "macos", target_arch = "aarch64")
))]
use proxima_protocols::process::{ChildResponse, ReadResponse};
#[cfg(any(
    all(target_os = "linux", target_arch = "x86_64"),
    all(target_os = "macos", target_arch = "aarch64")
))]
use proxima_vm::dispatch;
#[cfg(any(
    all(target_os = "linux", target_arch = "x86_64"),
    all(target_os = "macos", target_arch = "aarch64")
))]
use proxima_vm::elf;

#[cfg(any(
    all(target_os = "linux", target_arch = "x86_64"),
    all(target_os = "macos", target_arch = "aarch64")
))]
const MAX_SEGMENTS: usize = 4;
#[cfg(any(
    all(target_os = "linux", target_arch = "x86_64"),
    all(target_os = "macos", target_arch = "aarch64")
))]
const MAX_HYPERCALLS: usize = 16;
#[cfg(any(
    all(target_os = "linux", target_arch = "x86_64"),
    all(target_os = "macos", target_arch = "aarch64")
))]
const EMITTED_CAPACITY: usize = 256;
#[cfg(any(
    all(target_os = "linux", target_arch = "x86_64"),
    all(target_os = "macos", target_arch = "aarch64")
))]
const MMIO_EMITTED_CAPACITY: usize = 256;
#[cfg(any(
    all(target_os = "linux", target_arch = "x86_64"),
    all(target_os = "macos", target_arch = "aarch64")
))]
const NET_EMITTED_CAPACITY: usize = 256;
#[cfg(any(
    all(target_os = "linux", target_arch = "x86_64"),
    all(target_os = "macos", target_arch = "aarch64")
))]
const BLK_EMITTED_CAPACITY: usize = 2048;
#[cfg(any(
    all(target_os = "linux", target_arch = "x86_64"),
    all(target_os = "macos", target_arch = "aarch64")
))]
const PL011_EMITTED_CAPACITY: usize = 256;

#[cfg(any(
    all(target_os = "linux", target_arch = "x86_64"),
    all(target_os = "macos", target_arch = "aarch64")
))]
fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = env::args().skip(1);
    let guest_path = arguments
        .next()
        .ok_or("usage: dispatch_probe <path-to-guest-elf> [read|close]")?;
    let variant = arguments.next().unwrap_or_else(|| "read".to_string());

    let image = fs::read(&guest_path)?;
    let (entry, segments) = elf::parse_elf::<MAX_SEGMENTS>(&image)
        .map_err(|error| format!("failed to parse guest ELF {guest_path}: {error}"))?;

    let configured = match variant.as_str() {
        "read" => ChildResponse::Read(ReadResponse {
            bytes: b"vm-side-canned".to_vec(),
            eof: true,
        }),
        "close" => ChildResponse::Close,
        other => return Err(format!("unknown response variant {other:?}").into()),
    };
    let (
        requests,
        emitted,
        mmio_emitted,
        net_emitted,
        blk_emitted,
        pl011_emitted,
        create_to_first_exit_nanos,
        touch_all_pages_nanos,
        mmio_trap_count,
    ) = dispatch::run_dispatch_loop(
        entry,
        &segments,
        configured,
        MAX_HYPERCALLS,
        EMITTED_CAPACITY,
        MMIO_EMITTED_CAPACITY,
        NET_EMITTED_CAPACITY,
        BLK_EMITTED_CAPACITY,
        PL011_EMITTED_CAPACITY,
        dispatch::GUEST_MEMORY_SIZE,
    )?;

    eprintln!("guest issued {} request(s): {requests:?}", requests.len());
    eprintln!(
        "guest drained {} mmio byte(s): {mmio_emitted:?}",
        mmio_emitted.len()
    );
    eprintln!(
        "guest drained {} net byte(s): {net_emitted:?}",
        net_emitted.len()
    );
    eprintln!(
        "guest drained {} blk byte(s): {blk_emitted:?}",
        blk_emitted.len()
    );
    eprintln!(
        "guest drained {} pl011 byte(s): {pl011_emitted:?}",
        pl011_emitted.len()
    );
    eprintln!(
        "m3 create_to_first_exit_nanos={create_to_first_exit_nanos} \
         touch_all_pages_nanos={touch_all_pages_nanos} mmio_trap_count={mmio_trap_count}"
    );
    io::stdout().write_all(&emitted)?;
    Ok(())
}

#[cfg(not(any(
    all(target_os = "linux", target_arch = "x86_64"),
    all(target_os = "macos", target_arch = "aarch64")
)))]
fn main() -> Result<(), Box<dyn Error>> {
    Err(
        "VM dispatch probing supports linux/x86_64 KVM and macos/aarch64 Hypervisor.framework only"
            .into(),
    )
}
