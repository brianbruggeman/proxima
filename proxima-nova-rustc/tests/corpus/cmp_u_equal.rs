#![no_std]
#![no_main]

#[unsafe(no_mangle)]
pub extern "C" fn main(argc: i32, _argv: *const *const u8) -> i32 {
    let a: u32 = argc as u32;
    let b: u32 = argc as u32;
    let mut r = 128;
    if a < b {
        r += 1;
    }
    if a <= b {
        r += 2;
    }
    if a > b {
        r += 4;
    }
    if a >= b {
        r += 8;
    }
    if a == b {
        r += 16;
    }
    if a != b {
        r += 32;
    }
    r
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
