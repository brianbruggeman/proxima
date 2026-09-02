//! Codesigned probe, half two of two — see
//! `src/bin/snapshot_capture_probe.rs`'s module doc for why restore is a
//! separate process from capture on the HVF lane.
//!
//! `argv[1]` is the snapshot file path `snapshot_capture_probe` wrote;
//! `argv[2]` is the `page_size` stride [`restore`] copies memory in. Prints
//! one `key:value` line per observable the test parses.

use std::env;
use std::error::Error;
use std::fs;

use proxima_vm::snapshot::{VmSnapshot, restore};

fn main() -> Result<(), Box<dyn Error>> {
    let input_path = env::args()
        .nth(1)
        .ok_or("usage: snapshot_restore_probe <input_path> <page_size>")?;
    let page_size: usize = env::args()
        .nth(2)
        .and_then(|value| value.parse().ok())
        .unwrap_or(4096);

    let bytes = fs::read(&input_path)?;
    let snapshot = VmSnapshot::from_postcard_bytes(&bytes)?;
    let report = restore(&snapshot, page_size)?;

    println!("resumed_matched_trap:{}", report.resumed_matched_trap);
    println!("resumed_x0:{}", report.resumed_x0);
    println!("restore_wall_nanos:{}", report.restore_wall_nanos);
    println!("touch_all_pages_nanos:{}", report.touch_all_pages_nanos);
    println!("fault_count:{}", report.fault_count);
    println!("region_create_nanos:{}", report.phases.region_create_nanos);
    println!("vm_create_nanos:{}", report.phases.vm_create_nanos);
    println!("vm_map_nanos:{}", report.phases.vm_map_nanos);
    println!("vcpu_create_nanos:{}", report.phases.vcpu_create_nanos);
    println!(
        "register_restore_nanos:{}",
        report.phases.register_restore_nanos
    );
    println!("first_retrap_nanos:{}", report.phases.first_retrap_nanos);
    Ok(())
}
