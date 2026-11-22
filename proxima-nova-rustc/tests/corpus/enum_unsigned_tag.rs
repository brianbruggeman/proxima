#![no_std]
#![no_main]

enum Big {
    Low = 1,
    High = 200,
}

#[unsafe(no_mangle)]
pub extern "C" fn main(argc: i32, _argv: *const *const u8) -> i32 {
    let big = if argc > 0 { Big::High } else { Big::Low };
    if (big as i64) > 100 { 103 } else { 9 }
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
