//! Codesigned probe, µsec-campaign warm-restore slice: `snapshot_capture_probe`
//! still runs first, in its own process — `hv_vm_create` is once-per-process
//! on the HVF lane, and this probe's whole point is a SINGLE
//! `WarmVm::new` inside this one process, exercised `N` times, never a
//! second `hv_vm_create` at all (`proxima_vm::snapshot::WarmVm`'s own module
//! doc).
//!
//! `argv[1]` is the snapshot file path `snapshot_capture_probe` wrote;
//! `argv[2]` is the `page_size` stride; `argv[3]` is `N`, the number of
//! consecutive warm-restore calls to make against the one `WarmVm`.
//! `argv[4]` (µsec-campaign size sweep, slice 2, optional, default the
//! snapshot's own `guest_memory` length) is `target_size` — the padded
//! guest-memory size in bytes `WarmVm::restore` copies per call, scaled
//! independent of the scratch guest's own tiny code blob via
//! `VmSnapshot::with_padded_memory`. `argv[5]` (optional, default `same`) is
//! `content_mode`: `same` restores one fixed padded snapshot every call
//! (content constant after the first copy); `alternate` restores two
//! differently-seeded padded snapshots on alternating calls (content changes
//! every call) — the degenerate control proving `WarmVm::restore`'s cost is
//! size-bound, not content-bound.
//!
//! Prints one `key:value` line per observable the test parses, plus one
//! `iteration_nanos:<n>:<value>` line per call so the driving test can
//! compute p50/p99 itself, `iteration_memcpy_control_nanos:<n>:<value>` for
//! a same-size raw-`memcpy` control arm (no VM at all), and
//! `sample_offset:<offset>:<bool>` correctness lines from one untimed
//! verification restore at the end.

use std::env;
use std::error::Error;
use std::fs;
use std::hint::black_box;
use std::time::Instant;

use proxima_vm::snapshot::{VmSnapshot, WarmVm, pattern_byte};

const PATTERN_SEED_A: u64 = 0xA5A5_A5A5_A5A5_A5A5;
const PATTERN_SEED_B: u64 = 0x5A5A_5A5A_5A5A_5A5A;

fn main() -> Result<(), Box<dyn Error>> {
    let arguments: Vec<String> = env::args().collect();
    let input_path = arguments
        .get(1)
        .ok_or("usage: snapshot_warm_restore_probe <input_path> <page_size> <iterations> [target_size] [content_mode]")?;
    let page_size: usize = arguments
        .get(2)
        .and_then(|value| value.parse().ok())
        .unwrap_or(4096);
    let iterations: usize = arguments
        .get(3)
        .and_then(|value| value.parse().ok())
        .unwrap_or(100);

    let bytes = fs::read(input_path)?;
    let snapshot = VmSnapshot::from_postcard_bytes(&bytes)?;

    let target_size: usize = arguments
        .get(4)
        .and_then(|value| value.parse().ok())
        .unwrap_or_else(|| snapshot.guest_memory().len());
    let content_mode = arguments
        .get(5)
        .cloned()
        .unwrap_or_else(|| "same".to_string());

    let padded_a = snapshot.with_padded_memory(target_size, PATTERN_SEED_A);
    let padded_b = snapshot.with_padded_memory(target_size, PATTERN_SEED_B);

    // Raw-memcpy control: same byte count as one warm restore's memory copy,
    // no VM/vCPU/named-region involved at all -- the delta between this and
    // the VM-side warm-restore series attributes the memcpy term.
    let memcpy_source = vec![0xABu8; target_size];
    let mut memcpy_destination = vec![0u8; target_size];
    for index in 0..iterations {
        let start = Instant::now();
        memcpy_destination.copy_from_slice(&memcpy_source);
        let elapsed_nanos = start.elapsed().as_nanos();
        println!("iteration_memcpy_control_nanos:{index}:{elapsed_nanos}");
    }
    black_box(&memcpy_destination);

    let mut warm_vm = WarmVm::new(target_size)?;

    let mut matched_count = 0usize;
    let loop_start = Instant::now();
    for index in 0..iterations {
        let snapshot_for_call = if content_mode == "alternate" && index % 2 == 1 {
            &padded_b
        } else {
            &padded_a
        };

        // Rust-side wall-clock bracket around the whole call, independent of
        // the C-side `now_nanos()` phase timers -- this host's own
        // `CLOCK_MONOTONIC` reports `clock_getres` == 1000 ns (measured,
        // `/tmp/clockres.c` in this session), so those phase timers round
        // sub-microsecond costs to 0 and the syscall-dominated phase to a
        // constant 1000 ns quantum on every iteration; `Instant` is checked
        // here as an independent second measurement, not assumed finer.
        let call_start = Instant::now();
        let report = warm_vm.restore_oracle_full_copy(snapshot_for_call, page_size)?;
        let call_wall_nanos = call_start.elapsed().as_nanos();

        if report.resumed_matched_trap {
            matched_count += 1;
        }
        println!(
            "iteration_resumed_matched_trap:{index}:{}",
            report.resumed_matched_trap
        );
        println!("iteration_resumed_x0:{index}:{}", report.resumed_x0);
        println!(
            "iteration_restore_wall_nanos:{index}:{}",
            report.restore_wall_nanos
        );
        println!(
            "iteration_touch_all_pages_nanos:{index}:{}",
            report.touch_all_pages_nanos
        );
        println!(
            "iteration_register_restore_nanos:{index}:{}",
            report.phases.register_restore_nanos
        );
        println!(
            "iteration_first_retrap_nanos:{index}:{}",
            report.phases.first_retrap_nanos
        );
        println!("iteration_call_wall_nanos:{index}:{call_wall_nanos}");
    }
    let loop_total_nanos = loop_start.elapsed().as_nanos();

    // One untimed verification restore, outside the measured loop: the
    // content-bound control above only proves timing is content-agnostic,
    // never that the memcpy actually happened -- a lazy no-op restore would
    // still re-trap correctly, since the halting trap only touches the
    // scratch guest's own few code bytes, never this padding. Sampled bytes
    // at the start, middle, and end of the padded region must match
    // `pattern_byte` under `PATTERN_SEED_A`.
    let _verification_report = warm_vm.restore_oracle_full_copy(&padded_a, page_size)?;
    let original_length = snapshot.guest_memory().len();
    if target_size > original_length {
        for offset in [original_length, target_size / 2, target_size - 1] {
            let sampled = warm_vm.sample_guest_memory(offset, 1);
            let expected = pattern_byte(offset, PATTERN_SEED_A);
            let matches = sampled.first() == Some(&expected);
            println!("sample_offset:{offset}:{matches}");
        }
    }

    println!("iterations:{iterations}");
    println!("matched_count:{matched_count}");
    println!("loop_total_nanos:{loop_total_nanos}");
    println!("target_size:{target_size}");
    println!("content_mode:{content_mode}");
    Ok(())
}
