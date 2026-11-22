#![no_std]
#![no_main]

enum Color {
    Red = 3,
    Green = 7,
    Blue = 11,
}

#[unsafe(no_mangle)]
pub extern "C" fn main(argc: i32, _argv: *const *const u8) -> i32 {
    let color = if argc > 1 { Color::Red } else if argc > 0 { Color::Green } else { Color::Blue };
    color as i32 + 100
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
