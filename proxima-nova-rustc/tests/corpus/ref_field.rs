#![no_std]
#![no_main]

struct Pair {
    left: i32,
    right: i32,
}

#[inline(never)]
fn grow(field: &mut i32) {
    *field += 64;
}

#[unsafe(no_mangle)]
pub extern "C" fn main(argc: i32, _argv: *const *const u8) -> i32 {
    let mut pair = Pair { left: argc, right: argc + 2 };
    grow(&mut pair.right);
    pair.left * 2 + pair.right
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
