#![no_std]
#![no_main]

#[unsafe(no_mangle)]
pub extern "C" fn main(argc: i32, _argv: *const *const u8) -> i32 {
    let limit = argc + 9;
    let mut i = 0;
    let mut sum = 0;
    while i < limit {
        sum += i * i;
        i += 1;
    }
    sum - 200
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
