//! Elf guest: reads the whole request body, writes it back ASCII-uppercased.
//! Zero host-side code per function -- `[upstreams.x] type = "vm"` pointing at
//! this ELF is the entire registration (`src/upstreams/vm.rs`).
//!
//! `link-elf.ld` gives it a 2 MiB `RAM` with its stack as a real ELF
//! segment, so the host maps only what the segments need.

#![no_std]
#![no_main]

use core::panic::PanicInfo;

#[path = "../frame.rs"]
mod frame;
#[path = "../hypercall.rs"]
mod hypercall;
#[path = "../lambda.rs"]
mod lambda;

const INPUT_CAPACITY: usize = 256 * 1024;

static mut INPUT: [u8; INPUT_CAPACITY] = [0; INPUT_CAPACITY];

extern "C" fn lambda_main() -> ! {
    // SAFETY: single guest thread; `lambda_main` is entered once and is the
    // only code that touches `INPUT`.
    let input = unsafe {
        core::slice::from_raw_parts_mut((&raw mut INPUT).cast::<u8>(), INPUT_CAPACITY)
    };
    let length = lambda::read_input(input).unwrap_or_else(|error| lambda::fail(error));
    let body = &mut input[..length];
    body.make_ascii_uppercase();
    if let Err(error) = lambda::write_output(body) {
        lambda::fail(error);
    }
    lambda::halt()
}

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    lambda::abort()
}
