#![no_std]
#![no_main]

#[inline(never)]
fn twice(x: i32) -> i32 {
    x * 2
}

#[unsafe(no_mangle)]
pub extern "C" fn main(argc: i32, _argv: *const *const u8) -> i32 {
    let b1 = argc + 1;
    let b2 = argc + 2;
    let b3 = argc + 3;
    let b4 = argc + 4;
    let b5 = argc + 5;
    let b6 = argc + 6;
    let b7 = argc + 7;
    let b8 = argc + 8;
    let b9 = argc + 9;
    let b10 = argc + 10;
    let b11 = argc + 11;
    let b12 = argc + 12;
    let t = twice(argc);
    b1 + b2 + b3 + b4 + b5 + b6 + b7 + b8 + b9 + b10 + b11 + b12 + t
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
