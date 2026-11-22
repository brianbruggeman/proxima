#![no_std]
#![no_main]

enum Shape {
    Empty,
    One(i32),
    Two(i32, i32),
}

#[unsafe(no_mangle)]
pub extern "C" fn main(argc: i32, _argv: *const *const u8) -> i32 {
    let fresh = if argc > 6 { Shape::Two(1, 2) } else { Shape::One(argc + 58) };
    let mut slot = Shape::Empty;
    if argc > 0 {
        slot = fresh;
    }
    match slot {
        Shape::Empty => 1,
        Shape::One(value) => value,
        Shape::Two(left, right) => left + right,
    }
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
