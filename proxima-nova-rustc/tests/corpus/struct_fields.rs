#![no_std]
#![no_main]

struct P {
    a: i32,
    b: i32,
    c: i32,
}

#[unsafe(no_mangle)]
pub extern "C" fn main(argc: i32, _argv: *const *const u8) -> i32 {
    let p = P {
        a: argc,
        b: argc + 10,
        c: argc + 20,
    };
    p.a + p.b * 2 + p.c * 3
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
