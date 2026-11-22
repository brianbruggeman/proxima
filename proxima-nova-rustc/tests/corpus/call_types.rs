#![no_std]
#![no_main]

#[inline(never)]
fn is_big(x: u32) -> bool {
    x > 10
}

#[inline(never)]
fn pick(flag: bool, a: u32, b: u32) -> u32 {
    if flag { a - b } else { b - a }
}

#[unsafe(no_mangle)]
pub extern "C" fn main(argc: i32, _argv: *const *const u8) -> i32 {
    pick(is_big((argc + 20) as u32), 50, (argc + 3) as u32) as i32
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
