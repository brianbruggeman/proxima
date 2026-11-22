#![no_std]
#![no_main]

#[inline(never)]
fn bump(target: &mut i32, by: i32) {
    *target += by;
}

#[unsafe(no_mangle)]
pub extern "C" fn main(argc: i32, _argv: *const *const u8) -> i32 {
    let mut total = argc;
    bump(&mut total, 25);
    bump(&mut total, 12);
    total
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
