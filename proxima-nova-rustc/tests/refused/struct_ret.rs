#![no_std]
#![no_main]

struct P {
    a: i32,
    b: i32,
}

#[inline(never)]
fn make(a: i32) -> P {
    P { a, b: 2 }
}

#[unsafe(no_mangle)]
pub extern "C" fn main(argc: i32, _argv: *const *const u8) -> i32 {
    let p = make(argc);
    p.a + p.b
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
