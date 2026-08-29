//! Bare `#![no_std]` guest binary that `proxima-vm`'s ELF loader boots and
//! `src/backend_macos.c`'s `proxima_vm_run_dispatch_loop` drives through a
//! real `hvc #0` / `out dx, al` trap loop (`tools/proxima-vm/ROADMAP.md` M1).
//! `link.ld` fixes the load address so the image needs no runtime relocation;
//! `_start` sets the stack pointer from the linker-provided `__stack_top` and
//! hands off to `entry`, which issues two distinct `ChildRequest` hypercalls
//! (`Read`, then `Close`) and, after each, re-issues a dedicated `EMIT_VERB`
//! hypercall carrying the first byte the host just wrote back into the
//! request buffer — proving the host's response, not a compiled-in guest
//! constant, is what the guest emits. `HALT_VERB` ends the exit loop.

#![no_std]
#![no_main]

use core::arch::naked_asm;
use core::panic::PanicInfo;

mod hypercall;

/// Postcard variant discriminant for `proxima_protocols::process::ChildRequest::Read`
/// — the leading byte `dispatch.rs`'s parity test pins (`expected.push(0x00)`).
/// Reused as the hypercall verb so the host can route without decoding postcard.
const CHILD_REQUEST_READ_VERB: u16 = 0x00;

/// Postcard variant discriminant for `ChildRequest::Close`
/// (`proxima-protocols/src/process/protocol.rs:143`, discriminant index 3).
const CHILD_REQUEST_CLOSE_VERB: u16 = 0x03;

/// Sentinel verb naming the "emit one byte from `buffer[0]`" host action —
/// outside the `ChildRequest` discriminant range (0..=4), so the run loop
/// can tell a capability call from an emit call by `x0` alone
/// (`src/backend_macos.c`'s `proxima_vm_run_dispatch_loop`).
const EMIT_VERB: u16 = 0xfffe;

/// Sentinel verb ending the run loop.
const HALT_VERB: u16 = 0xffff;

/// `ChildRequest::Read { path: "/etc/hostname", max_bytes: 256, offset: 0 }`,
/// postcard-encoded, byte-for-byte the buffer `dispatch.rs`'s
/// `wire_format_round_trips_for_parity` pins as `expected`.
const CHILD_REQUEST_READ_WIRE_BYTES: [u8; 18] = [
    0x00, 13, b'/', b'e', b't', b'c', b'/', b'h', b'o', b's', b't', b'n', b'a', b'm', b'e', 0x80,
    0x02, 0x00,
];

/// `ChildRequest::Close { path: "/lambda" }`, postcard-encoded: variant
/// discriminant `0x03`, then `varint(len=7)`, then the path's UTF-8 bytes
/// (`proxima-protocols/src/process/protocol.rs:59-67`'s `String` wire rule).
const CHILD_REQUEST_CLOSE_WIRE_BYTES: [u8; 9] =
    [0x03, 0x07, b'/', b'l', b'a', b'm', b'b', b'd', b'a'];

#[unsafe(no_mangle)]
#[unsafe(naked)]
extern "C" fn _start() -> ! {
    #[cfg(target_arch = "aarch64")]
    naked_asm!(
        "adrp x0, __stack_top",
        "add x0, x0, :lo12:__stack_top",
        "mov sp, x0",
        "b {entry}",
        entry = sym entry,
    );
    #[cfg(target_arch = "x86_64")]
    naked_asm!(
        // `call`, not `jmp`: SysV expects rsp % 16 == 8 at function entry, as
        // if just entered via `call` — `__stack_top` itself is 16-aligned.
        "lea rsp, [rip + __stack_top]",
        "call {entry}",
        entry = sym entry,
    );
}

extern "C" fn entry() -> ! {
    let mut read_request = CHILD_REQUEST_READ_WIRE_BYTES;
    unsafe { hypercall::hypercall(CHILD_REQUEST_READ_VERB, &mut read_request) };
    // the host overwrote `read_request` in place with the encoded
    // `ChildResponse` it dispatched; emitting its first byte proves that
    // overwrite happened, not a replay of a value the guest already knew.
    let mut emit_read = [read_request[0]];
    unsafe { hypercall::hypercall(EMIT_VERB, &mut emit_read) };

    let mut close_request = CHILD_REQUEST_CLOSE_WIRE_BYTES;
    unsafe { hypercall::hypercall(CHILD_REQUEST_CLOSE_VERB, &mut close_request) };
    let mut emit_close = [close_request[0]];
    unsafe { hypercall::hypercall(EMIT_VERB, &mut emit_close) };

    let mut halt = [0_u8; 0];
    unsafe { hypercall::hypercall(HALT_VERB, &mut halt) };

    loop {
        core::hint::spin_loop();
    }
}

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {
        core::hint::spin_loop();
    }
}
