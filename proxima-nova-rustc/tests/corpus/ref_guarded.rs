#![no_std]
#![no_main]

struct Bundle {
    value: u32,
    number: i32,
    guard: i32,
    flag: bool,
    flag_guard: bool,
}

#[unsafe(no_mangle)]
pub extern "C" fn main(argc: i32, _argv: *const *const u8) -> i32 {
    let mut bundle = Bundle { value: argc as u32, number: argc, guard: argc + 40, flag: argc > 5, flag_guard: argc > 0 };
    let value_ref = &mut bundle.value;
    *value_ref += 4_000_000_000;
    let number_ref = &mut bundle.number;
    *number_ref += 70_000;
    let flag_ref = &mut bundle.flag;
    *flag_ref = argc > 0;
    let shared = &bundle.guard;
    let guard_seen = *shared;
    let whole = &mut bundle;
    let mut bits = 100;
    if whole.value > 3_000_000_000 { bits += 1; }
    if whole.number == 70_001 { bits += 2; }
    if guard_seen == 41 { bits += 4; }
    if whole.flag { bits += 8; }
    if whole.flag_guard { bits += 16; }
    bits
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
