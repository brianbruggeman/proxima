//! The KVM lane of the copy-on-write primitive probe: the same sections and
//! the same `key:value` lines as the Hypervisor.framework lane (`hvf.rs`),
//! measured through `proxima_vm::kvm::probe`.

use std::env;
use std::error::Error;
use std::time::Instant;

use proxima_vm::kvm::probe::ProbeVm;

const SIZES: [(&str, usize); 4] = [
    ("1MiB", 1024 * 1024),
    ("16MiB", 16 * 1024 * 1024),
    ("64MiB", 64 * 1024 * 1024),
    ("256MiB", 256 * 1024 * 1024),
];

const TRIO_ITERATIONS: usize = 50;
const FIRST_TOUCH_PAGE_SIZE: usize = 16384;

fn run_cow_view_candidate(probe: &mut ProbeVm, size_label: &str, size: usize) -> Result<(), Box<dyn Error>> {
    let source = ProbeVm::create_source(size)?;
    let mut previous = None;
    for index in 0..TRIO_ITERATIONS {
        let call_start = Instant::now();
        let (view, timings) = probe.cow_view(&source, size, previous.take())?;
        let call_wall_nanos = call_start.elapsed().as_nanos();
        previous = Some(view);
        println!("cow_remap_nanos:{size_label}:{index}:{}", timings.remap_nanos);
        println!("cow_hv_vm_unmap_old_nanos:{size_label}:{index}:{}", timings.unmap_old_nanos);
        println!("cow_hv_vm_map_nanos:{size_label}:{index}:{}", timings.map_nanos);
        println!("cow_dealloc_old_nanos:{size_label}:{index}:{}", timings.dealloc_old_nanos);
        println!("cow_call_wall_nanos:{size_label}:{index}:{call_wall_nanos}");
        let trio_total = timings.remap_nanos + timings.unmap_old_nanos + timings.map_nanos;
        println!("cow_trio_total_nanos:{size_label}:{index}:{trio_total}");
    }
    if let Some(view) = previous.as_mut() {
        for page_count in [16usize, 256, 4096] {
            let capacity = page_count.min(size / FIRST_TOUCH_PAGE_SIZE);
            let nanos = ProbeVm::first_touch(view, capacity);
            println!("cow_first_touch_nanos:{size_label}:{capacity}:{nanos}");
        }
    }
    probe.unmap_guest()?;
    Ok(())
}

fn run_vm_copy_candidate(probe: &mut ProbeVm, size_label: &str, size: usize) -> Result<(), Box<dyn Error>> {
    let source = ProbeVm::create_source(size)?;
    let mut previous = None;
    for index in 0..TRIO_ITERATIONS {
        let (view, timings) = probe.copy_view(&source, size, previous.take())?;
        previous = Some(view);
        println!("vmcopy_kern_return:{size_label}:{index}:0");
        println!("vmcopy_entry_create_nanos:{size_label}:{index}:{}", timings.entry_create_nanos);
        println!("vmcopy_map_nanos:{size_label}:{index}:{}", timings.map_nanos);
        println!("vmcopy_hv_vm_unmap_old_nanos:{size_label}:{index}:{}", timings.unmap_old_nanos);
        println!("vmcopy_hv_vm_map_nanos:{size_label}:{index}:{}", timings.guest_map_nanos);
        println!("vmcopy_dealloc_old_nanos:{size_label}:{index}:{}", timings.dealloc_old_nanos);
        let trio_total =
            timings.entry_create_nanos + timings.map_nanos + timings.unmap_old_nanos + timings.guest_map_nanos;
        println!("vmcopy_trio_total_nanos:{size_label}:{index}:{trio_total}");
    }
    println!("vmcopy_arm_rejected:{size_label}:false");
    probe.unmap_guest()?;
    Ok(())
}

fn run_protect_whole_candidate(probe: &mut ProbeVm, size_label: &str, size: usize) -> Result<(), Box<dyn Error>> {
    const HV_MEMORY_READ_WRITE: u64 = 1 | 2;

    let mut source = ProbeVm::create_source(size)?;
    probe.map_guest(source.primary_mut(), HV_MEMORY_READ_WRITE)?;
    for index in 0..TRIO_ITERATIONS {
        match probe.protect_whole(index % 2 == 0) {
            Ok(nanos) => println!("protect_whole_nanos:{size_label}:{index}:{nanos}"),
            Err(error) => {
                eprintln!("protect_whole error at {size_label}#{index}: {error}");
                break;
            }
        }
    }
    probe.unmap_guest()?;
    Ok(())
}

fn run_protect_per_page_candidate(probe: &mut ProbeVm) -> Result<(), Box<dyn Error>> {
    const GRANULE: usize = 16 * 1024;
    const PAGE_COUNT: usize = 4096;
    const HV_MEMORY_READ_WRITE: u64 = 1 | 2;

    let mut source = ProbeVm::create_source(GRANULE * PAGE_COUNT)?;
    probe.map_guest(source.primary_mut(), HV_MEMORY_READ_WRITE)?;
    match probe.protect_per_page(GRANULE, PAGE_COUNT) {
        Ok(timings) => {
            for (index, value) in timings.iter().enumerate() {
                println!("protect_per_page_nanos:{index}:{value}");
            }
        }
        Err(error) => eprintln!("protect_per_page error: {error}"),
    }
    probe.unmap_guest()?;
    Ok(())
}

fn run_write_protect_exit_verification(probe: &mut ProbeVm) -> Result<(), Box<dyn Error>> {
    let outcome = probe.write_protect_exit()?;
    println!("write_protect_exit_checkpoint1_x0:{}", outcome.checkpoint1_x0);
    println!("write_protect_exit_exception_class:0x{:x}", outcome.exception_class);
    println!("write_protect_exit_is_data_abort:{}", outcome.is_data_abort);
    println!("write_protect_exit_is_write:{}", i32::from(outcome.is_write));
    println!("write_protect_exit_data_byte_after:0x{:x}", outcome.data_byte_after);
    println!("write_protect_exit_protect_nanos:{}", outcome.protect_nanos);
    Ok(())
}

pub fn run() -> Result<(), Box<dyn Error>> {
    let arguments: Vec<String> = env::args().collect();
    let section = arguments.get(1).map_or("all", String::as_str);
    let mut probe = ProbeVm::create()?;

    if section == "all" || section == "cow" {
        for (label, size) in SIZES {
            run_cow_view_candidate(&mut probe, label, size)?;
        }
    }
    if section == "all" || section == "vmcopy" {
        for (label, size) in SIZES {
            run_vm_copy_candidate(&mut probe, label, size)?;
        }
    }
    if section == "all" || section == "protect" {
        for (label, size) in SIZES {
            run_protect_whole_candidate(&mut probe, label, size)?;
        }
        run_protect_per_page_candidate(&mut probe)?;
    }
    if section == "all" || section == "write_protect_exit" {
        run_write_protect_exit_verification(&mut probe)?;
    }
    Ok(())
}
