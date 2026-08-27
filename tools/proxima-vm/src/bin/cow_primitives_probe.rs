//! Codesigned probe, µsec-campaign slice 3 (`tools/proxima-vm/ROADMAP.md`):
//! measures the no-copy warm-restore candidates' primitive costs BEFORE any
//! `WarmVm` redesign — this binary changes no library surface, it only
//! drives the probe-only FFI leaves `src/probe_cow.h` names
//! (`src/backend_macos.c`'s appended probe section) and prints `key:value`
//! lines a driving test parses, the same shape
//! `snapshot_warm_restore_probe.rs` already established.
//!
//! `hv_vm_create` is once-per-process on the HVF lane
//! (`proxima_vm::snapshot`'s own module doc), so this one binary owns
//! exactly one `proxima_vm_probe_vm_create` call and runs every candidate
//! section sequentially against it.
//!
//! Prints, per candidate section, `iteration_<field>:<size_label>:<index>:<value>`
//! lines for the driving test to compute p50/p99/CoV itself, plus summary
//! `key:<size_label>:value` lines and any rejected-primitive
//! `kern_return`/error text verbatim.

use std::env;
use std::error::Error;
use std::ffi::CStr;
use std::os::raw::{c_char, c_void};
use std::time::Instant;

// This probe's own FFI declarations below never reference the `proxima_vm`
// library crate's Rust API -- but `src/probe_cow.h`'s functions are compiled
// INTO that crate's native static lib (`backend_macos.c`), and cargo's
// `cargo:rustc-link-lib=static=proxima_vm_native` / `=framework=Hypervisor`
// directives (`build.rs`) only reach the final linker invocation for a unit
// that actually links the `proxima_vm` rlib. An unreferenced `--extern` is
// dropped before the link step, silently taking its embedded native-link
// requirements with it (empirically: the plain FFI `extern "C"` blocks below
// linked against nothing at all without this line). `as _` links the crate
// with no unused-import warning and no API surface pulled in.
use proxima_vm as _;

const ERROR_CAPACITY: usize = 512;

unsafe extern "C" {
    fn proxima_vm_probe_vm_create(error_buffer: *mut c_char, error_capacity: usize) -> i32;

    fn proxima_vm_probe_create_source(
        size: usize,
        host_address_out: *mut *mut c_void,
        handle_out: *mut i32,
        error_buffer: *mut c_char,
        error_capacity: usize,
    ) -> i32;

    fn proxima_vm_probe_destroy_source(host_address: *mut c_void, handle: i32, size: usize);

    fn proxima_vm_probe_cow_view_trio(
        source_host_address: *mut c_void,
        size: usize,
        previous_view_inout: *mut *mut c_void,
        remap_nanos_out: *mut u64,
        hv_vm_unmap_old_nanos_out: *mut u64,
        hv_vm_map_nanos_out: *mut u64,
        dealloc_old_nanos_out: *mut u64,
        error_buffer: *mut c_char,
        error_capacity: usize,
    ) -> i32;

    fn proxima_vm_probe_first_touch(
        view_address: *mut c_void,
        page_size: usize,
        page_count: usize,
        nanos_out: *mut u64,
    ) -> i32;

    fn proxima_vm_probe_vm_copy_trio(
        source_host_address: *mut c_void,
        size: usize,
        previous_view_inout: *mut *mut c_void,
        kern_return_out: *mut i32,
        entry_create_nanos_out: *mut u64,
        map_nanos_out: *mut u64,
        hv_vm_unmap_old_nanos_out: *mut u64,
        hv_vm_map_nanos_out: *mut u64,
        dealloc_old_nanos_out: *mut u64,
        error_buffer: *mut c_char,
        error_capacity: usize,
    ) -> i32;

    fn proxima_vm_probe_protect_whole(
        guest_address: u64,
        size: usize,
        want_read_only: i32,
        nanos_out: *mut u64,
        error_buffer: *mut c_char,
        error_capacity: usize,
    ) -> i32;

    fn proxima_vm_probe_protect_per_page(
        guest_address: u64,
        granule: usize,
        page_count: usize,
        nanos_out: *mut u64,
        error_buffer: *mut c_char,
        error_capacity: usize,
    ) -> i32;

    fn proxima_vm_probe_write_protect_exit(
        checkpoint1_x0_out: *mut u64,
        exception_class_out: *mut u64,
        is_data_abort_out: *mut i32,
        is_write_out: *mut i32,
        data_byte_after_out: *mut u8,
        protect_nanos_out: *mut u64,
        error_buffer: *mut c_char,
        error_capacity: usize,
    ) -> i32;
}

fn read_error(buffer: &[c_char]) -> String {
    unsafe { CStr::from_ptr(buffer.as_ptr()) }.to_string_lossy().into_owned()
}

fn new_error_buffer() -> [c_char; ERROR_CAPACITY] {
    [0 as c_char; ERROR_CAPACITY]
}

const SIZES: [(&str, usize); 4] = [
    ("1MiB", 1024 * 1024),
    ("16MiB", 16 * 1024 * 1024),
    ("64MiB", 64 * 1024 * 1024),
    ("256MiB", 256 * 1024 * 1024),
];

const TRIO_ITERATIONS: usize = 50;

fn run_cow_view_candidate(size_label: &str, size: usize) -> Result<(), Box<dyn Error>> {
    let mut host_address: *mut c_void = std::ptr::null_mut();
    let mut handle: i32 = -1;
    let mut error_buffer = new_error_buffer();
    let status = unsafe {
        proxima_vm_probe_create_source(size, &raw mut host_address, &raw mut handle, error_buffer.as_mut_ptr(), ERROR_CAPACITY)
    };
    if status != 0 {
        return Err(read_error(&error_buffer).into());
    }

    let mut previous_view: *mut c_void = std::ptr::null_mut();
    for index in 0..TRIO_ITERATIONS {
        let mut remap_nanos: u64 = 0;
        let mut unmap_old_nanos: u64 = 0;
        let mut map_nanos: u64 = 0;
        let mut dealloc_old_nanos: u64 = 0;
        let mut error_buffer = new_error_buffer();
        let call_start = Instant::now();
        let status = unsafe {
            proxima_vm_probe_cow_view_trio(
                host_address,
                size,
                &raw mut previous_view,
                &raw mut remap_nanos,
                &raw mut unmap_old_nanos,
                &raw mut map_nanos,
                &raw mut dealloc_old_nanos,
                error_buffer.as_mut_ptr(),
                ERROR_CAPACITY,
            )
        };
        let call_wall_nanos = call_start.elapsed().as_nanos();
        if status != 0 {
            unsafe { proxima_vm_probe_destroy_source(host_address, handle, size) };
            return Err(read_error(&error_buffer).into());
        }
        println!("cow_remap_nanos:{size_label}:{index}:{remap_nanos}");
        println!("cow_hv_vm_unmap_old_nanos:{size_label}:{index}:{unmap_old_nanos}");
        println!("cow_hv_vm_map_nanos:{size_label}:{index}:{map_nanos}");
        println!("cow_dealloc_old_nanos:{size_label}:{index}:{dealloc_old_nanos}");
        println!("cow_call_wall_nanos:{size_label}:{index}:{call_wall_nanos}");
        let trio_total = remap_nanos + unmap_old_nanos + map_nanos;
        println!("cow_trio_total_nanos:{size_label}:{index}:{trio_total}");
    }

    let page_size = 16384usize;
    for page_count in [16usize, 256, 4096] {
        let capacity = page_count.min(size / page_size);
        let mut nanos: u64 = 0;
        let status = unsafe {
            proxima_vm_probe_first_touch(previous_view, page_size, capacity, &raw mut nanos)
        };
        if status == 0 {
            println!("cow_first_touch_nanos:{size_label}:{capacity}:{nanos}");
        }
    }

    // `proxima_vm_probe_cow_view_trio`'s own contract only tears down the
    // PREVIOUS view on each call, never the newest one -- the next size (or
    // the next candidate section) would otherwise collide with this size's
    // still-mapped guest IPA range `[0, size)`. Unmap it explicitly; the
    // host-side view mapping is left for the process to reclaim at exit
    // (a short-lived probe leaking its own address space, never a second
    // FFI leaf added purely to free a value read exactly once).
    if !previous_view.is_null() {
        unsafe { hv_vm_unmap(0, size) };
    }
    unsafe { proxima_vm_probe_destroy_source(host_address, handle, size) };
    Ok(())
}

fn run_vm_copy_candidate(size_label: &str, size: usize) -> Result<(), Box<dyn Error>> {
    let mut host_address: *mut c_void = std::ptr::null_mut();
    let mut handle: i32 = -1;
    let mut error_buffer = new_error_buffer();
    let status = unsafe {
        proxima_vm_probe_create_source(size, &raw mut host_address, &raw mut handle, error_buffer.as_mut_ptr(), ERROR_CAPACITY)
    };
    if status != 0 {
        return Err(read_error(&error_buffer).into());
    }

    let mut previous_view: *mut c_void = std::ptr::null_mut();
    let mut rejected = false;
    for index in 0..TRIO_ITERATIONS {
        let mut kern_return: i32 = 0;
        let mut entry_create_nanos: u64 = 0;
        let mut map_nanos: u64 = 0;
        let mut unmap_old_nanos: u64 = 0;
        let mut hv_map_nanos: u64 = 0;
        let mut dealloc_old_nanos: u64 = 0;
        let mut error_buffer = new_error_buffer();
        let status = unsafe {
            proxima_vm_probe_vm_copy_trio(
                host_address,
                size,
                &raw mut previous_view,
                &raw mut kern_return,
                &raw mut entry_create_nanos,
                &raw mut map_nanos,
                &raw mut unmap_old_nanos,
                &raw mut hv_map_nanos,
                &raw mut dealloc_old_nanos,
                error_buffer.as_mut_ptr(),
                ERROR_CAPACITY,
            )
        };
        println!("vmcopy_kern_return:{size_label}:{index}:{kern_return}");
        if status != 0 {
            println!("vmcopy_rejected:{size_label}:{index}:{}", read_error(&error_buffer));
            rejected = true;
            break;
        }
        println!("vmcopy_entry_create_nanos:{size_label}:{index}:{entry_create_nanos}");
        println!("vmcopy_map_nanos:{size_label}:{index}:{map_nanos}");
        println!("vmcopy_hv_vm_unmap_old_nanos:{size_label}:{index}:{unmap_old_nanos}");
        println!("vmcopy_hv_vm_map_nanos:{size_label}:{index}:{hv_map_nanos}");
        println!("vmcopy_dealloc_old_nanos:{size_label}:{index}:{dealloc_old_nanos}");
        let trio_total = entry_create_nanos + map_nanos + unmap_old_nanos + hv_map_nanos;
        println!("vmcopy_trio_total_nanos:{size_label}:{index}:{trio_total}");
    }

    println!("vmcopy_arm_rejected:{size_label}:{rejected}");
    if !previous_view.is_null() {
        unsafe { hv_vm_unmap(0, size) };
    }
    unsafe { proxima_vm_probe_destroy_source(host_address, handle, size) };
    Ok(())
}

fn run_protect_whole_candidate(size_label: &str, size: usize) -> Result<(), Box<dyn Error>> {
    let mut host_address: *mut c_void = std::ptr::null_mut();
    let mut handle: i32 = -1;
    let mut error_buffer = new_error_buffer();
    let status = unsafe {
        proxima_vm_probe_create_source(size, &raw mut host_address, &raw mut handle, error_buffer.as_mut_ptr(), ERROR_CAPACITY)
    };
    if status != 0 {
        return Err(read_error(&error_buffer).into());
    }

    let hv_status = unsafe { proxima_vm_probe_vm_create_guard_map(host_address, size) };
    if let Err(error) = hv_status {
        unsafe { proxima_vm_probe_destroy_source(host_address, handle, size) };
        return Err(error);
    }

    for index in 0..TRIO_ITERATIONS {
        let want_read_only = i32::from(index % 2 == 0);
        let mut nanos: u64 = 0;
        let mut error_buffer = new_error_buffer();
        let status = unsafe {
            proxima_vm_probe_protect_whole(0, size, want_read_only, &raw mut nanos, error_buffer.as_mut_ptr(), ERROR_CAPACITY)
        };
        if status != 0 {
            eprintln!("protect_whole error at {size_label}#{index}: {}", read_error(&error_buffer));
            break;
        }
        println!("protect_whole_nanos:{size_label}:{index}:{nanos}");
    }

    unsafe { proxima_vm_hv_vm_unmap_guard(size) };
    unsafe { proxima_vm_probe_destroy_source(host_address, handle, size) };
    Ok(())
}

unsafe extern "C" {
    fn hv_vm_map(address: *mut c_void, ipa: u64, size: usize, flags: u64) -> i32;
    fn hv_vm_unmap(ipa: u64, size: usize) -> i32;
}

const HV_MEMORY_READ_WRITE: u64 = 1 | 2;

unsafe fn proxima_vm_probe_vm_create_guard_map(host_address: *mut c_void, size: usize) -> Result<(), Box<dyn Error>> {
    let status = unsafe { hv_vm_map(host_address, 0, size, HV_MEMORY_READ_WRITE) };
    if status != 0 {
        return Err(format!("hv_vm_map guard failed: 0x{status:x}").into());
    }
    Ok(())
}

unsafe fn proxima_vm_hv_vm_unmap_guard(size: usize) {
    unsafe {
        hv_vm_unmap(0, size);
    }
}

fn run_protect_per_page_candidate() -> Result<(), Box<dyn Error>> {
    const GRANULE: usize = 16 * 1024;
    const PAGE_COUNT: usize = 4096;
    const SIZE: usize = GRANULE * PAGE_COUNT;

    let mut host_address: *mut c_void = std::ptr::null_mut();
    let mut handle: i32 = -1;
    let mut error_buffer = new_error_buffer();
    let status = unsafe {
        proxima_vm_probe_create_source(SIZE, &raw mut host_address, &raw mut handle, error_buffer.as_mut_ptr(), ERROR_CAPACITY)
    };
    if status != 0 {
        return Err(read_error(&error_buffer).into());
    }

    unsafe { proxima_vm_probe_vm_create_guard_map(host_address, SIZE)? };

    let mut nanos = vec![0u64; PAGE_COUNT];
    let mut error_buffer = new_error_buffer();
    let status = unsafe {
        proxima_vm_probe_protect_per_page(0, GRANULE, PAGE_COUNT, nanos.as_mut_ptr(), error_buffer.as_mut_ptr(), ERROR_CAPACITY)
    };
    if status != 0 {
        eprintln!("protect_per_page error: {}", read_error(&error_buffer));
    } else {
        for (index, value) in nanos.iter().enumerate() {
            println!("protect_per_page_nanos:{index}:{value}");
        }
    }

    unsafe { proxima_vm_hv_vm_unmap_guard(SIZE) };
    unsafe { proxima_vm_probe_destroy_source(host_address, handle, SIZE) };
    Ok(())
}

fn run_write_protect_exit_verification() -> Result<(), Box<dyn Error>> {
    let mut checkpoint1_x0: u64 = 0;
    let mut exception_class: u64 = 0;
    let mut is_data_abort: i32 = 0;
    let mut is_write: i32 = 0;
    let mut data_byte_after: u8 = 0;
    let mut protect_nanos: u64 = 0;
    let mut error_buffer = new_error_buffer();

    let status = unsafe {
        proxima_vm_probe_write_protect_exit(
            &raw mut checkpoint1_x0,
            &raw mut exception_class,
            &raw mut is_data_abort,
            &raw mut is_write,
            &raw mut data_byte_after,
            &raw mut protect_nanos,
            error_buffer.as_mut_ptr(),
            ERROR_CAPACITY,
        )
    };
    if status != 0 {
        return Err(read_error(&error_buffer).into());
    }

    println!("write_protect_exit_checkpoint1_x0:{checkpoint1_x0}");
    println!("write_protect_exit_exception_class:0x{exception_class:x}");
    println!("write_protect_exit_is_data_abort:{}", is_data_abort != 0);
    println!("write_protect_exit_is_write:{is_write}");
    println!("write_protect_exit_data_byte_after:0x{data_byte_after:x}");
    println!("write_protect_exit_protect_nanos:{protect_nanos}");
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let arguments: Vec<String> = env::args().collect();
    let section = arguments.get(1).map(String::as_str).unwrap_or("all");

    let mut error_buffer = new_error_buffer();
    let status = unsafe { proxima_vm_probe_vm_create(error_buffer.as_mut_ptr(), ERROR_CAPACITY) };
    if status != 0 {
        return Err(read_error(&error_buffer).into());
    }

    if section == "all" || section == "cow" {
        for (label, size) in SIZES {
            run_cow_view_candidate(label, size)?;
        }
    }
    if section == "all" || section == "vmcopy" {
        for (label, size) in SIZES {
            run_vm_copy_candidate(label, size)?;
        }
    }
    if section == "all" || section == "protect" {
        for (label, size) in SIZES {
            run_protect_whole_candidate(label, size)?;
        }
        run_protect_per_page_candidate()?;
    }
    if section == "all" || section == "write_protect_exit" {
        run_write_protect_exit_verification()?;
    }

    Ok(())
}
