#![no_std]
#![no_main]

#[track_caller]
#[inline(never)]
fn add_one(x: i32) -> i32 {
    x + 1
}

#[unsafe(no_mangle)]
pub extern "C" fn main(argc: i32, _argv: *const *const u8) -> i32 {
    if add_one(argc + 40) == 42 { 55 } else { 2 }
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
