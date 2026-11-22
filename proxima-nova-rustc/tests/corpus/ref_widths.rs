#![no_std]
#![no_main]

#[unsafe(no_mangle)]
pub extern "C" fn main(argc: i32, _argv: *const *const u8) -> i32 {
    let mut plain = argc;
    let mut unsigned = argc as u32;
    let mut wide = argc as i64;
    let mut unsigned_wide = argc as u64;
    let mut signed_size = argc as isize;
    let mut unsigned_size = argc as usize;
    let mut flag = argc > 5;
    let plain_ref = &mut plain;
    *plain_ref += 70_000;
    let unsigned_ref = &mut unsigned;
    *unsigned_ref += 4_000_000_000;
    let wide_ref = &mut wide;
    *wide_ref += 5_000_000_000;
    let unsigned_wide_ref = &mut unsigned_wide;
    *unsigned_wide_ref += 18_000_000_000_000_000_000;
    let signed_size_ref = &mut signed_size;
    *signed_size_ref -= 5_000_000_000;
    let unsigned_size_ref = &mut unsigned_size;
    *unsigned_size_ref += 18_000_000_000_000_000_000;
    let flag_ref = &mut flag;
    *flag_ref = argc > 0;
    let mut bits = 0;
    let plain_view = &plain;
    if *plain_view > 65_536 { bits += 1; }
    let unsigned_view = &unsigned;
    if *unsigned_view > 3_000_000_000 { bits += 2; }
    let wide_view = &wide;
    if *wide_view > 4_294_967_296 { bits += 4; }
    let unsigned_wide_view = &unsigned_wide;
    if *unsigned_wide_view > 17_000_000_000_000_000_000 { bits += 8; }
    let signed_size_view = &signed_size;
    if *signed_size_view < -4_294_967_296 { bits += 16; }
    let unsigned_size_view = &unsigned_size;
    if *unsigned_size_view > 17_000_000_000_000_000_000 { bits += 32; }
    let flag_view = &flag;
    if *flag_view { bits += 64; }
    bits
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
