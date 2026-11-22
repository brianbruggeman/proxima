#![no_std]
#![no_main]

#[inline(never)]
fn finish(x: i32) {
    if x > 0 {
        unsafe { exit(x + 76) }
    }
}

unsafe extern "C" {
    fn exit(code: i32) -> !;
}

#[unsafe(no_mangle)]
pub extern "C" fn main(argc: i32, _argv: *const *const u8) -> i32 {
    finish(argc);
    5
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
