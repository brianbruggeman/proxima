#![no_std]
#![no_main]

#[inline(never)]
fn negate_by_div(v: i32) -> i32 {
    v / -1
}

#[unsafe(no_mangle)]
pub extern "C" fn main(argc: i32, _argv: *const *const u8) -> i32 {
    if negate_by_div(argc + 70) == -71 { 47 } else { 2 }
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
