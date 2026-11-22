#![no_std]
#![no_main]

#[inline(never)]
fn base() -> i32 {
    40
}

#[unsafe(no_mangle)]
pub extern "C" fn main(argc: i32, _argv: *const *const u8) -> i32 {
    base() + argc * 3
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
