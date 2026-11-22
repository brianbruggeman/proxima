#![no_std]
#![no_main]

#[unsafe(no_mangle)]
pub extern "C" fn main(argc: i32, _argv: *const *const u8) -> i32 {
    let a1 = argc + 1;
    let a2 = a1 + 1;
    let a3 = a2 + 1;
    let a4 = a3 + 1;
    let a5 = a4 + 1;
    let a6 = a5 + 1;
    let a7 = a6 + 1;
    let a8 = a7 + 1;
    let a9 = a8 + 1;
    let a10 = a9 + 1;
    let a11 = a10 + 1;
    let a12 = a11 + 1;
    let a13 = a12 + 1;
    let a14 = a13 + 1;
    let a15 = a14 + 1;
    let a16 = a15 + 1;
    let a17 = a16 + 1;
    let a18 = a17 + 1;
    let a19 = a18 + 1;
    let a20 = a19 + 1;
    a20
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
