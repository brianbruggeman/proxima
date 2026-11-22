#![no_std]
#![no_main]

#[unsafe(no_mangle)]
pub extern "C" fn main(argc: i32, _argv: *const *const u8) -> i32 {
    let signed = argc as isize - 6;
    let unsigned = 0 - argc as usize;
    if signed < 0 {
        if unsigned > 5 { 75 } else { 11 }
    } else {
        10
    }
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
