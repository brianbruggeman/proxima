#![no_std]
#![no_main]

static PRIMES: [u32; 5] = [2, 3, 5, 7, 11];

#[inline(never)]
fn pick(values: &[u32; 5], at: usize) -> u32 {
    values[at]
}

#[unsafe(no_mangle)]
pub extern "C" fn main(argc: i32, _argv: *const *const u8) -> i32 {
    if pick(&PRIMES, argc as usize + 2) == 7 { 37 } else { 2 }
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
