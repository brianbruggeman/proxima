//! `proxima-vm run <path-to-elf>` -- reads a guest ELF from disk, validates
//! its `PT_LOAD` segments via [`elf::parse_elf`], then drives it against a
//! real hypervisor guest through [`dispatch::run_dispatch_loop`]: every
//! `hvc #0` / `out dx, al` the guest issues is a real VM exit
//! (`src/backend_macos.c`'s / `src/backend_linux.c`'s
//! `proxima_vm_run_dispatch_loop`), not a synthesized in-memory dispatch.
//! Writes the guest's emitted bytes to stdout and the recorded
//! `ChildRequest`s to stderr.

use std::env;
use std::error::Error;
use std::fs;
use std::io::{self, Write};

use proxima_protocols::process::{ChildResponse, ReadResponse};
use proxima_vm::dispatch;
use proxima_vm::elf;

/// Largest segment count any lambda guest built so far links -- `.text`,
/// `.rodata`, `.data` -- with headroom, matching `elf.rs`'s own guidance
/// for the M1 guest at `tools/proxima-vm/guests/lambda`. M2 replaces this
/// fixed cap with a `VmConfig` field; a CLI-only config path ahead of that
/// milestone would give the same concept two configuration surfaces.
const MAX_SEGMENTS: usize = 4;

/// Upper bound on hypercalls a single `run` drives before the loop reports
/// a runaway guest instead of hanging the host.
const MAX_HYPERCALLS: usize = 16;

const EMITTED_CAPACITY: usize = 256;
const MMIO_EMITTED_CAPACITY: usize = 256;
const NET_EMITTED_CAPACITY: usize = 256;
const BLK_EMITTED_CAPACITY: usize = 2048;
const PL011_EMITTED_CAPACITY: usize = 256;

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = env::args().skip(1);
    match (arguments.next().as_deref(), arguments.next()) {
        (Some("run"), Some(path)) => run(&path),
        _ => Err("usage: proxima-vm run <path-to-elf>".into()),
    }
}

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
