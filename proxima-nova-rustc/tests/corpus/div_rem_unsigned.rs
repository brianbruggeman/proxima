#![no_std]
#![no_main]

#[inline(never)]
fn split(t: u64, p: u64) -> u64 {
    t / p + t % p
}

#[unsafe(no_mangle)]
pub extern "C" fn main(argc: i32, _argv: *const *const u8) -> i32 {
    if split(u64::MAX - argc as u64, 7) == 2635249153387078802 { 26 } else { 2 }
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
