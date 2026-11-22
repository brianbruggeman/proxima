#![no_std]
#![no_main]

static HIGH: i32 = 70_000;
static LOW: i32 = 300;

#[unsafe(no_mangle)]
pub extern "C" fn main(argc: i32, _argv: *const *const u8) -> i32 {
    if HIGH - LOW == 69_700 { 100 + argc } else { 3 }
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
