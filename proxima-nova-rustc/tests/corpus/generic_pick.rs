#![no_std]
#![no_main]

fn pick<T: Copy>(first: T, second: T, take_first: bool) -> T {
    if take_first { first } else { second }
}

#[unsafe(no_mangle)]
pub extern "C" fn main(argc: i32, _argv: *const *const u8) -> i32 {
    let narrow = pick::<i32>(40, 7, argc > 0);
    let wide = pick::<i64>(2, 9, argc > 0);
    narrow + wide as i32
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
