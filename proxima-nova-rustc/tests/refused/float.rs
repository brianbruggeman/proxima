#![no_std]
#![no_main]

#[unsafe(no_mangle)]
pub extern "C" fn main(argc: i32, _argv: *const *const u8) -> i32 {
    let x = 1.0;
    x as i32 + argc - 1
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
