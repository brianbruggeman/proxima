//! Codesigned probe, layered base+delta warm restore
//! (`proxima_vm::snapshot::{LayeredBase, WarmVm}`'s layered API): unlike the
//! cold/warm-oracle probes, this one needs no separate capture process —
//! [`proxima_vm::snapshot::dirty_probe_snapshot`] builds its own guest
//! program and memory image as pure data, so one process does everything:
//! adopt a base, run a dirty-write guest program `iterations` times, restore
//! (mapping-only) after each run, and print the timing series the driving
//! test parses.
//!
//! `argv[1]` is the mode: `sweep` (size/dirty-count restore-cost sweep) or
//! `sharing` (the two-`WarmVm`-one-base proof). `sweep` additionally reads
//! `argv[2]` (`target_size`), `argv[3]` (`dirty_page_count`), `argv[4]`
//! (`iterations`).

use std::env;
use std::error::Error;

use proxima_vm::snapshot::{LayeredBase, WarmVm, dirty_probe_snapshot, host_page_size};

const BYTE_VALUE: u8 = 0x2A;
const SEED: u64 = 0xC0FF_EE00_C0FF_EE00;

fn run_sweep(target_size: usize, dirty_page_count: u16, iterations: usize) -> Result<(), Box<dyn Error>> {
    let granule = host_page_size();
    let data_offset = granule;
    let snapshot = dirty_probe_snapshot(target_size, data_offset, granule as u16, dirty_page_count, BYTE_VALUE, SEED);

    let base = LayeredBase::new(target_size)?;
    let mut warm_vm = WarmVm::new_layered(base, target_size)?;
    let adopt_report = warm_vm.adopt_base(snapshot.guest_memory())?;
    println!("adopt_map_nanos:{}", adopt_report.map_nanos);
    println!("adopt_register_reset_nanos:{}", adopt_report.register_reset_nanos);

    let expected_page_count = u64::from(dirty_page_count);
    for index in 0..iterations {
        let run_report = warm_vm.run_dirty_write(expected_page_count)?;
        println!("iteration_run_wall_nanos:{index}:{}", run_report.run_wall_nanos);
        println!("iteration_run_fault_count:{index}:{}", run_report.fault_count);
        println!(
            "iteration_run_newly_dirty_page_count:{index}:{}",
            run_report.newly_dirty_page_count
        );
        println!("iteration_run_halted_ok:{index}:{}", run_report.halted_ok);

        // Same K pages, same bitmap, NOT yet restored -- every one of these
        // pages is already delta-mapped read-write from the run above, so
        // this second run's guest writes never fault at all. The delta
        // between this and the first run's own wall time is exactly the
        // per-fault round-trip cost ("K-page run wall vs unprotected run /
        // K") -- an unprotected re-run of the identical guest program is a
        // real host-observed baseline, not a synthetic one.
        let unprotected_report = warm_vm.run_dirty_write(expected_page_count)?;
        println!(
            "iteration_unprotected_run_wall_nanos:{index}:{}",
            unprotected_report.run_wall_nanos
        );
        println!(
            "iteration_unprotected_run_fault_count:{index}:{}",
            unprotected_report.fault_count
        );

        let restore_report = warm_vm.restore_layered()?;
        println!("iteration_restore_wall_nanos:{index}:{}", restore_report.restore_wall_nanos);
        println!("iteration_remap_nanos:{index}:{}", restore_report.remap_nanos);
        println!(
            "iteration_register_reset_nanos:{index}:{}",
            restore_report.register_reset_nanos
        );
        println!(
            "iteration_remapped_page_count:{index}:{}",
            restore_report.remapped_page_count
        );
    }

    // Byte-identical-twin oracle: after the last restore, the base's own
    // bytes (never mutated by any run -- only the delta was) must still
    // match the ORIGINAL snapshot content exactly, over the whole region,
    // not just a sample.
    let original = snapshot.guest_memory();
    let restored_base = warm_vm.layered_base_bytes(0, target_size);
    let byte_identical = original == restored_base.as_slice();
    println!("byte_identical_twin_oracle:{byte_identical}");

    // Re-trap proof: after a mapping-only restore, the SAME dirty-write
    // guest program must re-fault on the SAME pages (fault_count ==
    // dirty_page_count again) -- proving the mapping genuinely reverted to
    // read-only, not merely that the bitmap was cleared.
    let post_restore_run = warm_vm.run_dirty_write(expected_page_count)?;
    println!("post_restore_fault_count:{}", post_restore_run.fault_count);
    println!(
        "post_restore_newly_dirty_page_count:{}",
        post_restore_run.newly_dirty_page_count
    );
    println!("post_restore_halted_ok:{}", post_restore_run.halted_ok);

    println!("iterations:{iterations}");
    println!("target_size:{target_size}");
    println!("dirty_page_count:{dirty_page_count}");
    Ok(())
}

fn run_sharing() -> Result<(), Box<dyn Error>> {
    const SIZE: usize = 4 * 1024 * 1024;
    let granule = host_page_size();
    let data_offset = granule;

    let base = LayeredBase::new(SIZE)?;
    let base_snapshot = dirty_probe_snapshot(SIZE, data_offset, granule as u16, 4, BYTE_VALUE, SEED);

    let mut vm_a = WarmVm::new_layered(base, SIZE)?;
    vm_a.adopt_base(base_snapshot.guest_memory())?;

    // `vm_b` shares `vm_a`'s base object via a SECOND, independent view --
    // never written directly (`WarmVm::adopt_shared_base` maps it read-only,
    // no bytes copied) -- at a disjoint IPA range (this process's one
    // `hv_vm` has one flat stage-2 space, `WarmVm::new_layered_over`'s own
    // doc). `hv_vcpu_create` ties a vCPU to its CALLING thread (empirically:
    // a second same-thread `hv_vcpu_create` answers `HV_BUSY`, 0xfae94002,
    // reproduced before this thread split existed) -- so `vm_b`'s entire
    // construction and every later call live on a dedicated thread, never
    // crossing back to the main thread mid-lifetime; `vm_a` stays on main.
    let shared_view = vm_a.layered_base_view()?;
    let sample_offset = data_offset;
    let vm_b_thread = std::thread::spawn(move || -> Result<(u8, u64), String> {
        let mut vm_b = WarmVm::new_layered_over(shared_view, SIZE, SIZE as u64).map_err(|error| error.to_string())?;
        vm_b.adopt_shared_base().map_err(|error| error.to_string())?;
        let before = vm_b.layered_delta_bytes(sample_offset, 1)[0];
        let restore = vm_b.restore_layered().map_err(|error| error.to_string())?;
        Ok((before, restore.remapped_page_count))
    });

    let write_report = vm_a.run_dirty_write(4)?;
    println!("vm_a_run_fault_count:{}", write_report.fault_count);
    println!("vm_a_run_halted_ok:{}", write_report.halted_ok);

    let (vm_b_delta_before, vm_b_restore_remapped_page_count) =
        vm_b_thread.join().map_err(|_| "vm_b thread panicked")??;

    let base_bytes = vm_a.layered_base_bytes(sample_offset, 1);
    let vm_a_delta_bytes = vm_a.layered_delta_bytes(sample_offset, 1);

    println!("base_byte:{}", base_bytes[0]);
    println!("vm_a_delta_byte:{}", vm_a_delta_bytes[0]);
    println!("vm_b_delta_byte:{vm_b_delta_before}");
    println!("written_byte:{BYTE_VALUE}");

    let base_unaffected = base_bytes[0] != BYTE_VALUE;
    let vm_a_wrote_its_delta = vm_a_delta_bytes[0] == BYTE_VALUE;
    let vm_b_never_wrote = vm_b_delta_before != BYTE_VALUE;
    println!("base_unaffected:{base_unaffected}");
    println!("vm_a_wrote_its_delta:{vm_a_wrote_its_delta}");
    println!("vm_b_never_wrote:{vm_b_never_wrote}");
    println!("vm_b_restore_remapped_page_count:{vm_b_restore_remapped_page_count}");
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let arguments: Vec<String> = env::args().collect();
    let mode = arguments.get(1).map(String::as_str).unwrap_or("sweep");

    match mode {
        "sharing" => run_sharing(),
        _ => {
            let target_size: usize = arguments.get(2).and_then(|value| value.parse().ok()).unwrap_or(16 * 1024 * 1024);
            let dirty_page_count: u16 = arguments.get(3).and_then(|value| value.parse().ok()).unwrap_or(16);
            let iterations: usize = arguments.get(4).and_then(|value| value.parse().ok()).unwrap_or(50);
            run_sweep(target_size, dirty_page_count, iterations)
        }
    }
}
