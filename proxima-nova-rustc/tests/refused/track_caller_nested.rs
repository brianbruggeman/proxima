#![no_std]
#![no_main]

#[track_caller]
#[inline(never)]
fn inner(x: i32) -> i32 {
    x + 1
}

#[track_caller]
#[inline(never)]
fn outer(x: i32) -> i32 {
    inner(x)
}

#[unsafe(no_mangle)]
pub extern "C" fn main(argc: i32, _argv: *const *const u8) -> i32 {
    outer(argc + 40)
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
