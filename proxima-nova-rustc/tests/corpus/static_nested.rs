#![no_std]
#![no_main]

static INNER: i32 = 90;
static OUTER: &i32 = &INNER;

#[unsafe(no_mangle)]
pub extern "C" fn main(argc: i32, _argv: *const *const u8) -> i32 {
    *OUTER + argc
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
