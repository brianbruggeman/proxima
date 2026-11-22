#![no_std]
#![no_main]

#[inline(never)]
fn pick<'a>(left: &'a i32, right: &'a i32, first: bool) -> &'a i32 {
    if first { left } else { right }
}

#[unsafe(no_mangle)]
pub extern "C" fn main(argc: i32, _argv: *const *const u8) -> i32 {
    let left = argc + 20;
    let right = argc + 72;
    *pick(&left, &right, argc > 5)
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
