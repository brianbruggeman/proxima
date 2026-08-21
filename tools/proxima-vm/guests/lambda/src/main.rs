//! Bare `#![no_std]` guest binary that `proxima-vm`'s ELF loader will boot.
//! `link.ld` fixes the load address so the image needs no runtime relocation;
//! `_start` sets the stack pointer from the linker-provided `__stack_top` and
//! hands off to `entry`, which issues one `hypercall` carrying the already-
//! pinned `ChildRequest::Read` postcard bytes (`src/dispatch.rs`'s
//! `wire_format_round_trips_for_parity`). Nothing on the host side maps this
//! image or reads the exit yet (`tools/proxima-vm/ROADMAP.md` M1) — this
//! step only proves the guest issues the trap with the right payload.

#![no_std]
#![no_main]

use core::arch::naked_asm;
use core::panic::PanicInfo;

mod hypercall;

/// Postcard variant discriminant for `proxima_protocols::process::ChildRequest::Read`
/// — the leading byte `dispatch.rs`'s parity test pins (`expected.push(0x00)`).
/// Reused as the hypercall verb so the host can route without decoding postcard.
const CHILD_REQUEST_READ_VERB: u16 = 0x00;

/// `ChildRequest::Read { path: "/etc/hostname", max_bytes: 256, offset: 0 }`,
/// postcard-encoded, byte-for-byte the buffer `dispatch.rs`'s
/// `wire_format_round_trips_for_parity` pins as `expected`.
const CHILD_REQUEST_READ_WIRE_BYTES: [u8; 18] = [
    0x00, 13, b'/', b'e', b't', b'c', b'/', b'h', b'o', b's', b't', b'n', b'a', b'm', b'e', 0x80,
    0x02, 0x00,
];

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
    let mut request = CHILD_REQUEST_READ_WIRE_BYTES;
    let _response = unsafe { hypercall::hypercall(CHILD_REQUEST_READ_VERB, &mut request) };
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
