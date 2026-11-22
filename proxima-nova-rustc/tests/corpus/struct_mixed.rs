#![no_std]
#![no_main]

struct M {
    flag: bool,
    x: u32,
    y: i32,
}

#[unsafe(no_mangle)]
pub extern "C" fn main(argc: i32, _argv: *const *const u8) -> i32 {
    let m = M {
        flag: argc > 1,
        x: (argc + 4) as u32,
        y: argc - 9,
    };
    if m.flag { 7 } else { m.x as i32 - m.y }
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
