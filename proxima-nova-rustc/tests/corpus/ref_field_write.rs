#![no_std]
#![no_main]

struct Pair {
    left: i32,
    right: i32,
}

#[unsafe(no_mangle)]
pub extern "C" fn main(argc: i32, _argv: *const *const u8) -> i32 {
    let mut pair = Pair { left: argc, right: argc + 2 };
    let whole = &mut pair;
    whole.right = argc + 110;
    pair.left * 2 + pair.right
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
