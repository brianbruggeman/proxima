#![no_std]
#![no_main]

unsafe extern "C" {
    #[link_name = "abs"]
    fn c_abs(x: i32) -> i32;
}

#[unsafe(no_mangle)]
pub extern "C" fn main(argc: i32, _argv: *const *const u8) -> i32 {
    let magnitude = unsafe { c_abs(argc - 50) };
    magnitude + 7
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
