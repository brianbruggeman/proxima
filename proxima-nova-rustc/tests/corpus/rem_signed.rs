#![no_std]
#![no_main]

#[inline(never)]
fn trunc_rem(v: i64, by: i64) -> i64 {
    v % by
}

#[unsafe(no_mangle)]
pub extern "C" fn main(argc: i32, _argv: *const *const u8) -> i32 {
    if trunc_rem(argc as i64 - 42, 9) == -5 { 29 } else { 2 }
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
