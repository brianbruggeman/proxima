#![no_std]
#![no_main]

#[inline(never)]
fn triple(value: i64) -> i64 {
    value * 3
}

#[unsafe(no_mangle)]
pub extern "C" fn main(argc: i32, _argv: *const *const u8) -> i32 {
    if triple(argc as i64 * 5_000_000_000) > 2_120_000_000 { 83 } else { 4 }
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
