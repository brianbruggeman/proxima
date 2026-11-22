#![no_std]
#![no_main]


unsafe extern "C" {
    fn exit(code: i32) -> !;
}

static PRIMES: [u32; 5] = [2, 3, 5, 7, 11];

#[inline(never)]
fn pick(values: &[u32; 5], at: usize) -> u32 {
    values[at]
}

#[unsafe(no_mangle)]
pub extern "C" fn main(argc: i32, _argv: *const *const u8) -> i32 {
    pick(&PRIMES, argc as usize + 8) as i32
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    unsafe { exit(22) }
}
