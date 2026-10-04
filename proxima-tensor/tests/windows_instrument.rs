#![cfg(all(windows, feature = "instrument"))]

use std::hint::black_box;
use std::time::{Duration, Instant};

use proxima_tensor::instrument::{read_ticks, thread_cpu_nanos, ticks_to_nanos};
use windows_sys::Win32::Foundation::FILETIME;
use windows_sys::Win32::System::Performance::{QueryPerformanceCounter, QueryPerformanceFrequency};
use windows_sys::Win32::System::Threading::{GetCurrentThread, GetThreadTimes};

fn native_counter() -> u64 {
    let mut counter = 0_i64;
    assert_ne!(unsafe { QueryPerformanceCounter(&mut counter) }, 0);
    u64::try_from(counter).expect("QPC counter is nonnegative")
}

fn native_cpu_nanos() -> u64 {
    let mut creation = FILETIME::default();
    let mut exit = FILETIME::default();
    let mut kernel = FILETIME::default();
    let mut user = FILETIME::default();
    assert_ne!(
        unsafe {
            GetThreadTimes(
                GetCurrentThread(),
                &mut creation,
                &mut exit,
                &mut kernel,
                &mut user,
            )
        },
        0
    );
    let kernel_units =
        u128::from(kernel.dwHighDateTime) * (1_u128 << 32) + u128::from(kernel.dwLowDateTime);
    let user_units =
        u128::from(user.dwHighDateTime) * (1_u128 << 32) + u128::from(user.dwLowDateTime);
    u64::try_from((kernel_units + user_units) * 100).expect("fixture CPU time fits u64")
}

#[test]
fn windows_instrument_qpc_ticks_and_conversion() {
    let lower = native_counter();
    let measured = read_ticks().as_raw();
    let upper = native_counter();
    assert!((lower..=upper).contains(&measured));
    let mut frequency = 0_i64;
    assert_ne!(unsafe { QueryPerformanceFrequency(&mut frequency) }, 0);
    let frequency = u64::try_from(frequency).expect("QPC frequency is positive");
    assert!(frequency > 0);
    assert_eq!(ticks_to_nanos(frequency), 1_000_000_000);
    for ticks in [0, 1, frequency / 3, frequency, frequency * 7 + 13, u64::MAX] {
        let expected = u128::from(ticks) * 1_000_000_000 / u128::from(frequency);
        let expected = u64::try_from(expected).unwrap_or(u64::MAX);
        assert_eq!(ticks_to_nanos(ticks), expected, "raw tick input {ticks}");
    }
}

#[test]
fn windows_instrument_thread_cpu_matches_os() {
    let lower = native_cpu_nanos();
    let started = thread_cpu_nanos();
    let upper = native_cpu_nanos();
    assert!((lower..=upper).contains(&started));
    assert_eq!(started % 100, 0);

    let deadline = Instant::now() + Duration::from_secs(10);
    let mut accumulator = black_box(0x1234_5678_9abc_def0_u64);
    let mut progressed = upper;
    while progressed == upper {
        for iteration in 0..65_536_u64 {
            accumulator = black_box(accumulator)
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(iteration);
        }
        black_box(accumulator);
        assert!(
            Instant::now() < deadline,
            "native thread CPU time did not advance"
        );
        progressed = native_cpu_nanos();
    }
    let measured = thread_cpu_nanos();
    let final_upper = native_cpu_nanos();
    assert!((progressed..=final_upper).contains(&measured));
    assert!(measured > started);
    assert_eq!(measured % 100, 0);
}
