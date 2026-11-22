#![no_std]
#![no_main]

#[inline(never)]
fn twice(x: i32) -> i32 {
    x * 2
}

#[unsafe(no_mangle)]
pub extern "C" fn main(argc: i32, _argv: *const *const u8) -> i32 {
    let f: fn(i32) -> i32 = twice;
    f(argc)
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
