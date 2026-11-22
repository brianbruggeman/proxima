#![no_std]
#![no_main]

static PAIR: (i32, i32) = (10, 32);
const SECOND: &i32 = &PAIR.1;

#[unsafe(no_mangle)]
pub extern "C" fn main(argc: i32, _argv: *const *const u8) -> i32 {
    if *SECOND == 32 { 79 + argc } else { 3 }
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
