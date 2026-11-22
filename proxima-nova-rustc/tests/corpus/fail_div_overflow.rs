#![no_std]
#![no_main]

unsafe extern "C" {
    fn exit(code: i32) -> !;
}

#[unsafe(no_mangle)]
pub extern "C" fn main(argc: i32, _argv: *const *const u8) -> i32 {
    let m = i32::MIN + (argc - 1);
    m / (argc - 2)
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    unsafe { exit(36) }
}
