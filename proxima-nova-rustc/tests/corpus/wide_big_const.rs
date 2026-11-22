#![no_std]
#![no_main]

#[unsafe(no_mangle)]
pub extern "C" fn main(argc: i32, _argv: *const *const u8) -> i32 {
    let big: u64 = 18_000_000_000_000_000_000 + argc as u64;
    if big > 17_999_999_999_999_999_999 { 89 } else { 6 }
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
