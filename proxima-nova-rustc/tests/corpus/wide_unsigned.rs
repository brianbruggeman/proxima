#![no_std]
#![no_main]

#[unsafe(no_mangle)]
pub extern "C" fn main(argc: i32, _argv: *const *const u8) -> i32 {
    let x: u64 = 0 - argc as u64;
    if x > 5 { 67 } else { 4 }
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
