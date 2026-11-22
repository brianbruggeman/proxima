#![no_std]
#![no_main]

#[unsafe(no_mangle)]
pub extern "C" fn main(argc: i32, _argv: *const *const u8) -> i32 {
    let high = argc as i64 + 4_294_967_297;
    let low = argc as i64 + 1;
    let near = high - 1;
    let mut bits = 100;
    if high <= low { bits += 1; }
    if high > low { bits += 2; }
    if high == low { bits += 4; }
    if high != low { bits += 8; }
    if near < low { bits += 16; }
    if near >= low { bits += 32; }
    bits
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
