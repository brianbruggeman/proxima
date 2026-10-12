//! Whole-process resource counters for library, tests, examples and benches: the peak
//! resident set and the user/system CPU the process has spent so far.
//!
//! Unix reads `getrusage(RUSAGE_SELF)`; Windows reads the peak working set
//! from `GetProcessMemoryInfo` and the CPU times from `GetProcessTimes`. One
//! definition here replaces the per-file `getrusage` copies, which did not
//! compile off unix.

use std::io;

#[cfg(windows)]
use windows_sys::Win32::Foundation::FILETIME;
#[cfg(windows)]
use windows_sys::Win32::System::ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
#[cfg(windows)]
use windows_sys::Win32::System::Threading::{GetCurrentProcess, GetProcessTimes};

/// Peak resident set of this process so far, in bytes.
#[cfg(unix)]
pub fn peak_rss_bytes() -> Result<u64, io::Error> {
    let usage = rusage_self()?;
    let reported = usage.ru_maxrss as u64;
    // darwin reports ru_maxrss in bytes; every other unix reports KiB
    Ok(if cfg!(target_vendor = "apple") {
        reported
    } else {
        reported.saturating_mul(1024)
    })
}

/// User and system CPU milliseconds this whole process has used so far, so a
/// difference across one region is the CPU that region cost every thread.
#[cfg(unix)]
pub fn process_cpu_ms() -> Result<(f64, f64), io::Error> {
    let usage = rusage_self()?;
    let millis = |time: libc::timeval| time.tv_sec as f64 * 1000.0 + time.tv_usec as f64 / 1000.0;
    Ok((millis(usage.ru_utime), millis(usage.ru_stime)))
}

#[cfg(unix)]
fn rusage_self() -> Result<libc::rusage, io::Error> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
    // SAFETY: `usage` is a valid, writable `rusage`; `RUSAGE_SELF` takes no other pointer.
    let status = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if status != 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: `getrusage` returned 0, so it filled the struct.
    Ok(unsafe { usage.assume_init() })
}

/// Cumulative minor page faults this process has taken, so a difference across
/// one region is the first-touch page-in that region paid. Unix only: Windows
/// counts every page fault, a different metric.
#[cfg(unix)]
pub fn minor_faults() -> Result<u64, io::Error> {
    Ok(rusage_self()?.ru_minflt as u64)
}

/// Resident set and physical footprint of this process right now, in bytes
/// (`proc_pid_rusage`): the current values, not the peaks. Footprint is what
/// `footprint(1)` and Activity Monitor's memory column read.
#[cfg(target_os = "macos")]
pub fn resident_and_footprint_bytes() -> Result<(u64, u64), io::Error> {
    let mut info = std::mem::MaybeUninit::<libc::rusage_info_v2>::zeroed();
    // SAFETY: `RUSAGE_INFO_V2` fills a `rusage_info_v2`; the C signature takes the struct's address cast to `rusage_info_t *`.
    let status = unsafe {
        libc::proc_pid_rusage(
            std::process::id() as libc::c_int,
            libc::RUSAGE_INFO_V2,
            info.as_mut_ptr().cast(),
        )
    };
    if status != 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: the call returned 0, so the struct is filled.
    let info = unsafe { info.assume_init() };
    Ok((info.ri_resident_size, info.ri_phys_footprint))
}

/// Peak resident set of this process so far, in bytes.
#[cfg(windows)]
pub fn peak_rss_bytes() -> Result<u64, io::Error> {
    let mut counters = std::mem::MaybeUninit::<PROCESS_MEMORY_COUNTERS>::zeroed();
    let size = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
    // SAFETY: `counters` is a valid, writable struct of `size` bytes; the pseudo-handle needs no close.
    let status = unsafe { GetProcessMemoryInfo(GetCurrentProcess(), counters.as_mut_ptr(), size) };
    if status == 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: `GetProcessMemoryInfo` returned nonzero, so it filled the struct.
    Ok(unsafe { counters.assume_init() }.PeakWorkingSetSize as u64)
}

/// User and system CPU milliseconds this whole process has used so far, so a
/// difference across one region is the CPU that region cost every thread.
#[cfg(windows)]
pub fn process_cpu_ms() -> Result<(f64, f64), io::Error> {
    let zero = FILETIME {
        dwLowDateTime: 0,
        dwHighDateTime: 0,
    };
    let (mut created, mut exited, mut kernel, mut user) = (zero, zero, zero, zero);
    // SAFETY: all four out-pointers are valid writable FILETIMEs; the pseudo-handle needs no close.
    let status = unsafe {
        GetProcessTimes(
            GetCurrentProcess(),
            &mut created,
            &mut exited,
            &mut kernel,
            &mut user,
        )
    };
    if status == 0 {
        return Err(io::Error::last_os_error());
    }
    // FILETIME counts 100ns ticks
    let millis = |time: FILETIME| {
        let ticks = (u64::from(time.dwHighDateTime) << 32) | u64::from(time.dwLowDateTime);
        ticks as f64 / 10_000.0
    };
    Ok((millis(user), millis(kernel)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peak_rss_of_a_running_test_process_is_at_least_a_megabyte() {
        let peak = peak_rss_bytes().expect("peak rss must be readable for the current process");
        assert!(peak > 1 << 20, "peak rss {peak} bytes implausibly small");
    }

    #[test]
    fn process_cpu_time_never_runs_backwards_across_work() {
        let (user_before, system_before) = process_cpu_ms().expect("cpu times readable");
        let checksum: u64 =
            (0..20_000_000u64).fold(0, |acc, value| acc.wrapping_add(value * value));
        let (user_after, system_after) = process_cpu_ms().expect("cpu times readable");
        assert!(checksum != 1);
        assert!(user_after + system_after >= user_before + system_before);
    }
}
