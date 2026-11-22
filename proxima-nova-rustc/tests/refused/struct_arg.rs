#![no_std]
#![no_main]

struct P {
    a: i32,
    b: i32,
}

#[inline(never)]
fn sum(p: P) -> i32 {
    p.a + p.b
}

#[unsafe(no_mangle)]
pub extern "C" fn main(argc: i32, _argv: *const *const u8) -> i32 {
    sum(P { a: argc, b: 2 })
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
