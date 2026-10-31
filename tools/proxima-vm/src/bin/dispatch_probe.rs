//! Signed-subprocess probe for `dispatch::run_dispatch_loop`: reads an
//! already-built guest ELF (path supplied as argv\[1\] by
//! `tests/dispatch_hypercall.rs`, which builds it for the lane's bare-metal
//! target before signing and running this probe), boots it against a real
//! hypervisor, and drives its `ChildRequest` hypercalls through a real VM
//! exit.
//!
//! argv\[2\] is the invocation input the guest reads over handle 0. argv\[3\]
//! selects what is written to stdout as raw bytes: `emitted` \[default\] is
//! the M1 guest's emit-verb channel, `output` is the invocation output
//! (handle 1) -- `tests/dispatch_hypercall.rs` runs this probe against two
//! different inputs and asserts the guest's bytes differ, proving the host's
//! input, not a guest-compiled constant, decides what the guest emits.
//! argv\[4\] optionally overrides `guest_memory_size`.

use std::error::Error;

#[cfg(any(
    all(target_os = "linux", target_arch = "x86_64"),
    all(target_os = "macos", target_arch = "aarch64")
))]
mod lane {
    use std::env;
    use std::error::Error;
    use std::fs;
    use std::io::{self, Write};

    use proxima_vm::dispatch;
    use proxima_vm::elf;

    const MAX_SEGMENTS: usize = 4;
    const MAX_HYPERCALLS: usize = 16;
    const OUTPUT_CAPACITY: usize = 4096;
    const EMITTED_CAPACITY: usize = 256;
    const MMIO_EMITTED_CAPACITY: usize = 256;
    const NET_EMITTED_CAPACITY: usize = 256;
    const BLK_EMITTED_CAPACITY: usize = 2048;
    const PL011_EMITTED_CAPACITY: usize = 256;

    pub fn run() -> Result<(), Box<dyn Error>> {
        let mut arguments = env::args().skip(1);
        let guest_path = arguments.next().ok_or(
            "usage: dispatch_probe <guest-elf> [input] [emitted|output] [guest-memory-size]",
        )?;
        let input = arguments.next().unwrap_or_default();
        let channel = arguments.next().unwrap_or_else(|| "emitted".to_string());
        let guest_memory_size = match arguments.next() {
            Some(size) => size.parse::<u64>()?,
            None => dispatch::GUEST_MEMORY_SIZE,
        };

        let image = fs::read(&guest_path)?;
        let (entry, segments) = elf::parse_elf::<MAX_SEGMENTS>(&image)
            .map_err(|error| format!("failed to parse guest ELF {guest_path}: {error}"))?;

        let observed = dispatch::run_dispatch_loop(
            entry,
            &segments,
            input.as_bytes(),
            OUTPUT_CAPACITY,
            MAX_HYPERCALLS,
            EMITTED_CAPACITY,
            MMIO_EMITTED_CAPACITY,
            NET_EMITTED_CAPACITY,
            BLK_EMITTED_CAPACITY,
            PL011_EMITTED_CAPACITY,
            guest_memory_size,
        )?;

        eprintln!(
            "guest issued {} request(s): {:?}",
            observed.requests.len(),
            observed.requests
        );
        eprintln!(
            "guest drained {} mmio byte(s): {:?}",
            observed.mmio_emitted.len(),
            observed.mmio_emitted
        );
        eprintln!(
            "guest drained {} net byte(s): {:?}",
            observed.net_emitted.len(),
            observed.net_emitted
        );
        eprintln!(
            "guest drained {} blk byte(s): {:?}",
            observed.blk_emitted.len(),
            observed.blk_emitted
        );
        eprintln!(
            "guest drained {} pl011 byte(s): {:?}",
            observed.pl011_emitted.len(),
            observed.pl011_emitted
        );
        eprintln!(
            "m3 create_to_first_exit_nanos={} touch_all_pages_nanos={} mmio_trap_count={}",
            observed.create_to_first_exit_nanos,
            observed.touch_all_pages_nanos,
            observed.mmio_trap_count
        );
        let bytes = match channel.as_str() {
            "emitted" => &observed.emitted,
            "output" => &observed.output,
            other => return Err(format!("unknown channel {other:?}").into()),
        };
        io::stdout().write_all(bytes)?;
        Ok(())
    }
}

#[cfg(any(
    all(target_os = "linux", target_arch = "x86_64"),
    all(target_os = "macos", target_arch = "aarch64")
))]
fn main() -> Result<(), Box<dyn Error>> {
    lane::run()
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
