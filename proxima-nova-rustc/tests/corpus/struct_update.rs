#![no_std]
#![no_main]

struct P {
    a: i32,
    b: i32,
    c: i32,
}

#[unsafe(no_mangle)]
pub extern "C" fn main(argc: i32, _argv: *const *const u8) -> i32 {
    let mut p = P { a: 1, b: 2, c: argc };
    p.b = p.c + 50;
    p.a + p.b + p.c
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
