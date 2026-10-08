// Reads one instrumented decode_gbps_baseline telemetry log (PROXIMA_TELEMETRY_FILE) and prints the
// per-owner device-buffer census beside the per-step footprint and device-allocation series.
//
// build: rustc -O footprint_table.rs -o footprint_table
// usage: footprint_table <label> <telemetry.log> [steady_from_step] [steady_to_step]

use std::collections::BTreeMap;
use std::env;
use std::fs;

#[derive(Default, Clone, Copy)]
struct Owner {
    events: u64,
    bytes: u64,
    aliased_bytes: u64,
    released_bytes: u64,
}

fn field<'line>(line: &'line str, key: &str) -> Option<&'line str> {
    let needle = format!(" {key}=");
    let start = line.find(&needle)? + needle.len();
    let rest = &line[start..];
    let end = rest.find(' ').unwrap_or(rest.len());
    Some(&rest[..end])
}

fn number(line: &str, key: &str) -> Option<u64> {
    field(line, key)?.parse().ok()
}

fn median(values: &[u64]) -> u64 {
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    sorted.get(sorted.len() / 2).copied().unwrap_or(0)
}

fn name_class(owner: &str, name: &str) -> String {
    if name.starts_with("kv_cache.") {
        format!("{owner}[kv_cache.*]")
    } else if name == "__checkpoint_mapping__" {
        format!("{owner}[checkpoint_mapping]")
    } else if name.is_empty() {
        owner.to_string()
    } else {
        format!("{owner}[named other]")
    }
}

struct Snapshot {
    step: u64,
    footprint: u64,
    device: u64,
    live_event_bytes: u64,
}

fn main() {
    let arguments: Vec<String> = env::args().collect();
    let label = &arguments[1];
    let text = fs::read_to_string(&arguments[2]).expect("read telemetry log");
    let steady_from: u64 = arguments.get(3).and_then(|value| value.parse().ok()).unwrap_or(10);
    let steady_to: u64 = arguments.get(4).and_then(|value| value.parse().ok()).unwrap_or(127);

    let mut owners: BTreeMap<String, Owner> = BTreeMap::new();
    let mut live_event_bytes: u64 = 0;
    let mut snapshots: Vec<Snapshot> = Vec::new();
    let mut ring: Option<(u64, u64)> = None;
    let mut arena_allocated: u64 = 0;
    let mut arena_peak: u64 = 0;
    let mut arena_plans: u64 = 0;
    let mut steady_rss: Option<u64> = None;
    let mut steady_footprint_run: Option<u64> = None;
    let mut class_lines: Vec<String> = Vec::new();
    let mut by_class_released: BTreeMap<String, u64> = BTreeMap::new();

    for line in text.lines() {
        if line.contains(" device_buffer owner=") {
            let owner = field(line, "owner").unwrap_or("?");
            let name = field(line, "name").unwrap_or("");
            let bytes = number(line, "bytes").unwrap_or(0);
            let aliased = field(line, "aliased") == Some("true");
            let key = name_class(owner, name);
            let entry = owners.entry(key).or_default();
            entry.events += 1;
            entry.bytes += bytes;
            if aliased {
                entry.aliased_bytes += bytes;
            }
            live_event_bytes += bytes;
        } else if line.contains(" device_buffer_released owner=") {
            let owner = field(line, "owner").unwrap_or("?").to_string();
            let bytes = number(line, "bytes").unwrap_or(0);
            owners.entry(owner.clone()).or_default().released_bytes += bytes;
            *by_class_released.entry(owner).or_default() += bytes;
            live_event_bytes = live_event_bytes.saturating_sub(bytes);
        } else if line.contains("telemetry_recorder_footprint") {
            ring = Some((
                number(line, "footprint_before_bytes").unwrap_or(0),
                number(line, "footprint_after_bytes").unwrap_or(0),
            ));
        } else if line.contains("buffer arena laid out") {
            arena_plans += 1;
            arena_allocated += number(line, "allocated_bytes").unwrap_or(0);
            arena_peak += number(line, "peak_bytes").unwrap_or(0);
        } else if line.contains("token_breakdown_metal:") {
            if let (Some(step), Some(footprint), Some(device)) = (
                number(line, "step"),
                number(line, "phys_footprint_bytes"),
                number(line, "device_allocated_bytes"),
            ) {
                snapshots.push(Snapshot { step, footprint, device, live_event_bytes });
            }
        } else if line.contains("device_memory_by_class:") {
            class_lines.push(line.split("device_memory_by_class:").nth(1).unwrap_or("").trim().to_string());
        } else if line.contains("run=done") {
            steady_rss = number(line, "steady_rss_bytes");
            steady_footprint_run = number(line, "steady_footprint_bytes");
        }
    }

    println!("cell {label}");
    match ring {
        Some((before, after)) => println!(
            "telemetry_ring_install_footprint before={before} after={after} delta={} (excluded from every footprint figure below as footprint_minus_ring)",
            after - before
        ),
        None => println!("telemetry_ring_install_footprint not recorded"),
    }
    let ring_delta = ring.map_or(0, |(before, after)| after - before);
    let steady: Vec<&Snapshot> = snapshots
        .iter()
        .filter(|snapshot| snapshot.step >= steady_from && snapshot.step <= steady_to)
        .collect();
    println!("steps_recorded={} steady_window={}..={} steady_steps={}", snapshots.len(), steady_from, steady_to, steady.len());
    if let (Some(first), true) = (snapshots.first(), !snapshots.is_empty()) {
        let footprints: Vec<u64> = steady.iter().map(|snapshot| snapshot.footprint).collect();
        let devices: Vec<u64> = steady.iter().map(|snapshot| snapshot.device).collect();
        let peak_footprint = snapshots.iter().map(|snapshot| snapshot.footprint).max().unwrap_or(0);
        let peak_device = snapshots.iter().map(|snapshot| snapshot.device).max().unwrap_or(0);
        let steady_footprint = median(&footprints);
        let steady_device = median(&devices);
        println!(
            "phys_footprint_bytes step0={} peak={} steady_median={} peak_minus_steady={}",
            first.footprint, peak_footprint, steady_footprint, peak_footprint.saturating_sub(steady_footprint)
        );
        println!(
            "phys_footprint_minus_ring_bytes step0={} peak={} steady_median={} peak_minus_steady={}",
            first.footprint.saturating_sub(ring_delta),
            peak_footprint.saturating_sub(ring_delta),
            steady_footprint.saturating_sub(ring_delta),
            peak_footprint.saturating_sub(steady_footprint)
        );
        println!(
            "device_allocated_bytes step0={} peak={} steady_median={} peak_minus_steady={}",
            first.device, peak_device, steady_device, peak_device.saturating_sub(steady_device)
        );
        let at = |step: u64| snapshots.iter().find(|snapshot| snapshot.step == step);
        for step in [0_u64, steady_from, steady_to] {
            if let Some(snapshot) = at(step) {
                println!(
                    "unattributed_device_bytes step={} device_allocated={} live_event_bytes={} unattributed={}",
                    snapshot.step,
                    snapshot.device,
                    snapshot.live_event_bytes,
                    snapshot.device as i128 - snapshot.live_event_bytes as i128
                );
            }
        }
    }
    println!("steady_rss_bytes(median of per-token samples after token 10)={steady_rss:?} steady_footprint_bytes(same samples, run line)={steady_footprint_run:?}");
    println!("owner | events | bytes created | of which aliased | released");
    for (owner, entry) in &owners {
        println!(
            "{owner} | {} | {} | {} | {}",
            entry.events, entry.bytes, entry.aliased_bytes, entry.released_bytes
        );
    }
    println!("arena plans={arena_plans} allocated_bytes_sum={arena_allocated} live_peak_bytes_sum={arena_peak} fragmentation_bytes={}", arena_allocated.saturating_sub(arena_peak));
    for line in class_lines.iter().take(3) {
        println!("device_memory_by_class {line}");
    }
}
