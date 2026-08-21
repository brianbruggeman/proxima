//! `proxima-vm run <path-to-elf>` -- reads a guest ELF from disk, validates
//! and maps its `PT_LOAD` segments into a real hypervisor guest address
//! space via [`elf::parse_elf`] and [`GuestMemory::map`], then drives the
//! S9 hypercall trampoline ([`dispatch::run_hypercall_guest`]) and writes
//! the response bytes it emits to stdout.
//!
//! The trampoline run below exercises the same synthesized single-hypercall
//! guest `dispatch_probe.rs` already proves against a real hypervisor --
//! `run_hypercall_guest` has no API yet to execute the segments this
//! command just mapped; wiring a vCPU to the real loaded guest and driving
//! its own hypercalls is a later milestone step named in
//! `tools/proxima-vm/ROADMAP.md`'s M1 section. What this command proves
//! today is real: an arbitrary guest ELF the loader has never seen parses
//! and maps end to end against a live hypervisor.

use std::env;
use std::error::Error;
use std::fs;
use std::io::{self, Write};

use proxima_protocols::process::{ChildResponse, ReadResponse};
use proxima_vm::dispatch::{self, RecordingDispatcher};
use proxima_vm::elf;
use proxima_vm::loader::GuestMemory;

/// Largest segment count any lambda guest built so far links -- `.text`,
/// `.rodata`, `.data` -- with headroom, matching `elf.rs`'s own guidance
/// for the M1 guest at `tools/proxima-vm/guests/lambda`. M2 replaces this
/// fixed cap with a `VmConfig` field; a CLI-only config path ahead of that
/// milestone would give the same concept two configuration surfaces.
const MAX_SEGMENTS: usize = 4;

/// Postcard variant discriminant for `ChildRequest::Read`, reused as the
/// hypercall verb -- matches `dispatch.rs`'s and `dispatch_probe.rs`'s
/// constant of the same name.
const CHILD_REQUEST_READ_VERB: u16 = 0x00;

/// `ChildRequest::Read { path: "/etc/hostname", max_bytes: 256, offset: 0 }`,
/// postcard-encoded -- byte-for-byte the payload the M1 lambda guest's
/// `_start` issues (`guests/lambda/src/main.rs`'s
/// `CHILD_REQUEST_READ_WIRE_BYTES`) and `dispatch_probe.rs` drives.
const CHILD_REQUEST_READ_WIRE_BYTES: [u8; 18] = [
    0x00, 13, b'/', b'e', b't', b'c', b'/', b'h', b'o', b's', b't', b'n', b'a', b'm', b'e', 0x80,
    0x02, 0x00,
];

const RESPONSE_CAPACITY: usize = 256;

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

    let guest_memory = GuestMemory::<MAX_SEGMENTS>::map(&segments)?;
    eprintln!(
        "mapped {} segment(s) into guest memory",
        guest_memory.segment_count()
    );

    let configured = ChildResponse::Read(ReadResponse {
        bytes: b"vm-side-canned".to_vec(),
        eof: true,
    });
    let dispatcher = RecordingDispatcher::new(configured);
    let response = dispatch::run_hypercall_guest(
        &dispatcher,
        CHILD_REQUEST_READ_VERB,
        &CHILD_REQUEST_READ_WIRE_BYTES,
        RESPONSE_CAPACITY,
    )?;

    io::stdout().write_all(&response)?;
    Ok(())
}
