#![no_std]
#![no_main]

#[unsafe(no_mangle)]
pub extern "C" fn backend_nine_params(a: i32, b: i32, c: i32, d: i32, e: i32, f: i32, g: i32, h: i32, i: i32) -> i32 {
    a + b + c + d + e + f + g + h + i
}

#[unsafe(no_mangle)]
pub extern "C" fn filler_clean(argc: i32) -> i32 {
    argc + 1
}

#[unsafe(no_mangle)]
pub extern "C" fn later_mir(argc: i32) -> i32 {
    let x = 1.0;
    x as i32 + argc - 1
}

#[unsafe(no_mangle)]
pub extern "C" fn main(argc: i32, _argv: *const *const u8) -> i32 {
    later_mir(argc)
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
