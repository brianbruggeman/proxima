#![no_std]
#![no_main]

enum Temp {
    Cold = -5,
    Warm = 20,
}

#[unsafe(no_mangle)]
pub extern "C" fn main(argc: i32, _argv: *const *const u8) -> i32 {
    let temp = if argc > 0 { Temp::Cold } else { Temp::Warm };
    if (temp as i64) < 0 { 97 } else { 7 }
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
