#![no_std]
#![no_main]

struct P {
    a: i32,
    b: i32,
}

struct Q {
    p: P,
    d: i32,
}

#[unsafe(no_mangle)]
pub extern "C" fn main(argc: i32, _argv: *const *const u8) -> i32 {
    let q = Q {
        p: P { a: argc, b: 2 },
        d: 3,
    };
    q.d
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
