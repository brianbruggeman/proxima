//! Signed-subprocess probe for `dispatch::run_dispatch_loop`: reads the
//! already-built `proxima-vm-guest-lambda` ELF (path supplied as argv[1] by
//! `tests/dispatch_hypercall.rs`, which builds it for
//! `aarch64-unknown-none` before signing and running this probe), boots it
//! against a real hypervisor, and drives its two `ChildRequest` hypercalls
//! through a real VM exit. Writes the guest's emitted bytes to stdout as
//! raw bytes — `tests/dispatch_hypercall.rs` asserts on them directly, not
//! through a postcard decode, since they are the guest's own emitted-byte
//! proof, not a `ChildResponse`.
//!
//! argv[2] selects the dispatcher's `configured_response` variant ("read"
//! [default] or "close") — `tests/dispatch_hypercall.rs` runs this probe
//! once per variant against the identical guest and asserts the guest's
//! emitted bytes differ, proving the host's response (not a guest-compiled
//! constant) decides what the guest emits.

use std::env;
use std::error::Error;
use std::fs;
use std::io::{self, Write};

use proxima_protocols::process::{ChildResponse, ReadResponse};
use proxima_vm::dispatch;
use proxima_vm::elf;

const MAX_SEGMENTS: usize = 4;
const MAX_HYPERCALLS: usize = 16;
const EMITTED_CAPACITY: usize = 256;

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
    let (requests, emitted) = dispatch::run_dispatch_loop(
        entry,
        &segments,
        configured,
        MAX_HYPERCALLS,
        EMITTED_CAPACITY,
    )?;

    eprintln!("guest issued {} request(s): {requests:?}", requests.len());
    io::stdout().write_all(&emitted)?;
    Ok(())
}
