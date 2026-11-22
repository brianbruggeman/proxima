#![no_std]
#![no_main]

struct Pair {
    left: i32,
    right: i32,
}

static ORIGIN: Pair = Pair { left: 7, right: 60 };

#[unsafe(no_mangle)]
pub extern "C" fn main(argc: i32, _argv: *const *const u8) -> i32 {
    ORIGIN.right - ORIGIN.left + argc
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
