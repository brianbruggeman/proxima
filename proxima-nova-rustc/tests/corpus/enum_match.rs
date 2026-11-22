#![no_std]
#![no_main]

enum Shape {
    Empty,
    One(i32),
    Two(i32, i32),
}

#[unsafe(no_mangle)]
pub extern "C" fn main(argc: i32, _argv: *const *const u8) -> i32 {
    let shape = if argc > 1 {
        Shape::Empty
    } else if argc > 0 {
        Shape::Two(argc, 40)
    } else {
        Shape::One(5)
    };
    match shape {
        Shape::Empty => 1,
        Shape::One(value) => value,
        Shape::Two(left, right) => left + right,
    }
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
