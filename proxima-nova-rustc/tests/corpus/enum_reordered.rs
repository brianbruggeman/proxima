#![no_std]
#![no_main]

enum Mixed {
    Pair(i32, bool),
    Single(u32),
}

#[unsafe(no_mangle)]
pub extern "C" fn main(argc: i32, _argv: *const *const u8) -> i32 {
    let mixed = if argc > 0 { Mixed::Pair(argc + 60, argc > 0) } else { Mixed::Single(3) };
    match mixed {
        Mixed::Pair(value, flag) => if flag { value } else { 2 },
        Mixed::Single(value) => value as i32,
    }
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
