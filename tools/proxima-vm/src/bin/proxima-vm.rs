//! `proxima-vm run <path-to-elf>` -- reads a guest ELF from disk, validates
//! its `PT_LOAD` segments via [`elf::parse_elf`], then drives it against a
//! real hypervisor guest through [`dispatch::run_dispatch_loop`]: every
//! `hvc #0` / `out dx, al` the guest issues is a real VM exit
//! (`src/backend_macos.c`'s / `src/backend_linux.c`'s
//! `proxima_vm_run_dispatch_loop`), not a synthesized in-memory dispatch.
//! Writes the guest's emitted bytes to stdout and the recorded
//! `ChildRequest`s to stderr.

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

/// Largest segment count any lambda guest built so far links -- `.text`,
/// `.rodata`, `.data` -- with headroom, matching `elf.rs`'s own guidance
/// for the M1 guest at `tools/proxima-vm/guests/lambda`. M2 replaces this
/// fixed cap with a `VmConfig` field; a CLI-only config path ahead of that
/// milestone would give the same concept two configuration surfaces.
#[cfg(any(
    all(target_os = "linux", target_arch = "x86_64"),
    all(target_os = "macos", target_arch = "aarch64")
))]
const MAX_SEGMENTS: usize = 4;

/// Upper bound on hypercalls a single `run` drives before the loop reports
/// a runaway guest instead of hanging the host.
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
    match (arguments.next().as_deref(), arguments.next()) {
        (Some("run"), Some(path)) => run(&path),
        _ => Err("usage: proxima-vm run <path-to-elf>".into()),
    }
}

#[cfg(any(
    all(target_os = "linux", target_arch = "x86_64"),
    all(target_os = "macos", target_arch = "aarch64")
))]
fn run(path: &str) -> Result<(), Box<dyn Error>> {
    let image = fs::read(path)?;

    let (entry, segments) = elf::parse_elf::<MAX_SEGMENTS>(&image)
        .map_err(|error| format!("failed to parse ELF {path}: {error}"))?;
    eprintln!(
        "parsed {path}: entry 0x{entry:x}, {} segment(s)",
        segments.len()
    );

    let configured = ChildResponse::Read(ReadResponse {
        bytes: b"vm-side-canned".to_vec(),
        eof: true,
    });
    let (requests, emitted, ..) = dispatch::run_dispatch_loop(
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
    io::stdout().write_all(&emitted)?;
    Ok(())
}

#[cfg(not(any(
    all(target_os = "linux", target_arch = "x86_64"),
    all(target_os = "macos", target_arch = "aarch64")
)))]
fn main() -> Result<(), Box<dyn Error>> {
    Err(
        "proxima-vm run supports linux/x86_64 KVM and macos/aarch64 Hypervisor.framework only"
            .into(),
    )
}
