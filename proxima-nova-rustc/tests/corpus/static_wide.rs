#![no_std]
#![no_main]

static BIG: i64 = 5_000_000_000;

#[unsafe(no_mangle)]
pub extern "C" fn main(argc: i32, _argv: *const *const u8) -> i32 {
    if BIG + argc as i64 > 4_294_967_296 { 87 } else { 3 }
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
