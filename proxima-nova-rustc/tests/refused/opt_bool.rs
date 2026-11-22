#![no_std]
#![no_main]

#[unsafe(no_mangle)]
pub extern "C" fn main(argc: i32, _argv: *const *const u8) -> i32 {
    let maybe = if argc > 0 { Some(argc > 5) } else { None };
    match maybe {
        Some(true) => 1,
        Some(false) => 2,
        None => 3,
    }
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
