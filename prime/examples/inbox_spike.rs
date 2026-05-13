//! demonstrate ReleasePolicy::Always spike-then-settle: grow the inbox with a
//! burst of transient producers, let them exit (lanes abandoned), drain +
//! reclaim, and watch VmRSS fall back toward the floor. run on Linux with
//! `MIMALLOC_PURGE_DELAY=0` so freed rings return to the OS.

#![cfg(feature = "runtime-prime-inbox-dynamic")]

use std::sync::{Arc, Barrier};
use std::time::Duration;

use prime::core::inbox_dynamic::{InboxDynamicConfig, ReleasePolicy, channel};

const SPIKE_PRODUCERS: usize = 512;
// fill each lane's ring so its pages are actually resident (lazy fault means a
// few small sends touch almost nothing). cap sends => ~LANE_CAP*size_of::<T>
// resident per lane.
const SENDS_PER_PRODUCER: usize = LANE_CAP;
const LANE_CAP: usize = 1024;

fn vmrss_mb() -> usize {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            let kb: usize = rest
                .trim()
                .split_whitespace()
                .next()
                .and_then(|n| n.parse().ok())
                .unwrap_or(0);
            return kb / 1024;
        }
    }
    0
}

fn main() {
    let config = InboxDynamicConfig {
        floor: 8,
        ceiling: 4096,
        release: ReleasePolicy::Always,
        lane_capacity: LANE_CAP,
    };
    let (producer, consumer) = channel::<[u8; 64]>(&config);
    let producer = Arc::new(producer);
    println!("floor (8 lanes):              {} MB", vmrss_mb());

    // SPIKE: producers must be CONCURRENTLY alive to each need a distinct lane
    // (otherwise they recycle within the floor). A barrier holds every producer
    // alive (lane claimed, ring filled) until all have arrived.
    let barrier = Arc::new(Barrier::new(SPIKE_PRODUCERS + 1));
    let mut handles = Vec::with_capacity(SPIKE_PRODUCERS);
    for _ in 0..SPIKE_PRODUCERS {
        let producer = producer.clone();
        let barrier = barrier.clone();
        handles.push(std::thread::spawn(move || {
            // non-zero payload: writing zeros leaves pages on the shared
            // zero-page (not resident), hiding the ring memory from VmRSS.
            for _ in 0..SENDS_PER_PRODUCER {
                let _ = producer.try_send_mpsc([0xABu8; 64]);
            }
            barrier.wait(); // hold the lane until all producers are up
        }));
    }
    std::thread::sleep(Duration::from_millis(500)); // let all send + park
    let (used, pending) = consumer.debug_lane_stats();
    println!(
        "peak ({SPIKE_PRODUCERS} concurrent producers): {} MB  [used_lanes={used} abandoned_pending={pending}]",
        vmrss_mb()
    );
    barrier.wait(); // release producers -> they exit -> lanes abandoned
    for handle in handles {
        let _ = handle.join();
    }

    // SETTLE: drain everything (no task lost), then keep draining empty so the
    // consumer's slow-path reclaim frees the drained, abandoned lane rings.
    while consumer.try_recv().is_ok() {}
    for _ in 0..(SPIKE_PRODUCERS + 64) {
        let _ = consumer.try_recv();
    }
    println!("settled (drained + reclaimed): {} MB", vmrss_mb());
}
