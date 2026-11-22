#![no_std]
#![no_main]

#[unsafe(no_mangle)]
pub extern "C" fn main(argc: i32, _argv: *const *const u8) -> i32 {
    let small = argc - 6;
    let unsigned_small: u32 = 0 - argc as u32;
    let widened_signed = small as i64;
    let widened_to_unsigned = small as u64;
    let zero_extended = unsigned_small as u64;
    let zero_to_signed = unsigned_small as i64;
    let wide = widened_signed * 4_294_967_296 + 7;
    let narrowed = wide as i32;
    let narrowed_unsigned = wide as u32;
    let same_width = wide as u64;
    let pointer_sized = wide as isize;
    let back = pointer_sized as usize;
    let mut bits = 0;
    if widened_signed < 0 { bits += 1; }
    if widened_to_unsigned > 4_294_967_296 { bits += 2; }
    if zero_extended < 4_294_967_296 { bits += 4; }
    if zero_to_signed > 0 { bits += 8; }
    if narrowed == 7 { bits += 16; }
    if narrowed_unsigned == 7 { bits += 32; }
    if same_width > 9_000_000_000_000_000_000 { bits += 64; }
    if pointer_sized < 0 && back > 9_000_000_000_000_000_000 { bits += 128; }
    bits
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
