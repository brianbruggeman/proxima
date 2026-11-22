#![no_std]
#![no_main]

struct Pair {
    left: i32,
    right: i32,
}

#[unsafe(no_mangle)]
pub extern "C" fn main(argc: i32, _argv: *const *const u8) -> i32 {
    let pair = Pair { left: argc + 3, right: argc + 80 };
    let view = &pair;
    view.left * 3 + view.right
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
