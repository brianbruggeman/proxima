#![no_std]
#![no_main]

enum Shape {
    Empty,
    One(i32),
    Two(i32, i32),
}

#[unsafe(no_mangle)]
pub extern "C" fn main(argc: i32, _argv: *const *const u8) -> i32 {
    let fresh = if argc > 6 { Shape::Empty } else if argc > 5 { Shape::One(3) } else { Shape::Two(argc + 70_000, argc + 300) };
    let mut slot = Shape::Two(9, 9);
    if argc > 0 {
        slot = fresh;
    }
    match slot {
        Shape::Empty => 1,
        Shape::One(value) => value,
        Shape::Two(left, right) => if left > 65_536 && right > 256 { 63 } else { 18 },
    }
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
