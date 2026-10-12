//! Resources one measured cell cost, printed beside its wall clock: process CPU
//! (user plus system, every thread), peak resident set, physical footprint,
//! Metal's allocated bytes and the one-minute load average before and after.
//! Shared by the attribution examples through `#[path]`, so every cell they time
//! reports the same columns.

use std::time::Instant;

use proxima_tensor::process_usage;

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
    let (user_ms, system_ms) = process_usage::process_cpu_ms()
        .expect("process cpu times must be readable for the current process");
    let (_, footprint_bytes) = process_usage::resident_and_footprint_bytes()
        .expect("physical footprint must be readable for the current process");
    Snapshot {
        cpu_ms: user_ms + system_ms,
        rss_peak_bytes: process_usage::peak_rss_bytes()
            .expect("peak rss must be readable for the current process"),
        footprint_bytes,
    }
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
