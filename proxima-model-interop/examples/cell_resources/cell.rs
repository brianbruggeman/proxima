//! Resources one measured cell cost, printed beside its wall clock: process CPU
//! (user plus system, every thread), peak resident set, physical footprint,
//! Metal's allocated bytes and the one-minute load average before and after.
//! Shared by the attribution examples through `#[path]`, so every cell they time
//! reports the same columns.

use std::time::Instant;

pub struct Cell {
    started: Instant,
    cpu_before_ms: f64,
    load_before: f64,
}

struct Snapshot {
    cpu_ms: f64,
    rss_peak_bytes: u64,
    footprint_bytes: u64,
}

fn snapshot() -> Snapshot {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
    // SAFETY: `usage` is a valid, writable `rusage`; `RUSAGE_SELF` takes no other pointer.
    let status = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    assert_eq!(status, 0, "getrusage(RUSAGE_SELF) failed");
    // SAFETY: `getrusage` returned 0, so it filled the struct.
    let usage = unsafe { usage.assume_init() };
    let millis = |time: libc::timeval| time.tv_sec as f64 * 1000.0 + time.tv_usec as f64 / 1000.0;
    Snapshot {
        cpu_ms: millis(usage.ru_utime) + millis(usage.ru_stime),
        rss_peak_bytes: usage.ru_maxrss as u64,
        footprint_bytes: physical_footprint_bytes(),
    }
}

fn physical_footprint_bytes() -> u64 {
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
        return 0;
    }
    // SAFETY: the call returned 0, so the struct is filled.
    unsafe { info.assume_init() }.ri_phys_footprint
}

fn load_one_minute() -> f64 {
    let mut loads = [0.0_f64; 3];
    // SAFETY: `loads` holds the three slots `getloadavg` is asked for.
    let count = unsafe { libc::getloadavg(loads.as_mut_ptr(), 3) };
    if count < 1 { f64::NAN } else { loads[0] }
}

fn gpu_allocated_bytes() -> u64 {
    omega::metal::current_allocated_size().unwrap_or(0)
}

impl Cell {
    pub fn begin() -> Self {
        Self {
            started: Instant::now(),
            cpu_before_ms: snapshot().cpu_ms,
            load_before: load_one_minute(),
        }
    }

    pub fn end(self, label: &str) -> String {
        let wall_ms = self.started.elapsed().as_secs_f64() * 1e3;
        let after = snapshot();
        let cpu_ms = after.cpu_ms - self.cpu_before_ms;
        format!(
            "cell label={label} wall_ms={wall_ms:.3} cpu_ms={cpu_ms:.3} cpu_pct={:.1} rss_peak_mb={:.1} footprint_mb={:.1} gpu_alloc_mb={:.1} load_before={:.2} load_after={:.2}",
            100.0 * cpu_ms / wall_ms.max(f64::MIN_POSITIVE),
            after.rss_peak_bytes as f64 / 1048576.0,
            after.footprint_bytes as f64 / 1048576.0,
            gpu_allocated_bytes() as f64 / 1048576.0,
            self.load_before,
            load_one_minute()
        )
    }
}
