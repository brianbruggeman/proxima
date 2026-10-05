#![cfg(all(
    target_os = "macos",
    feature = "metal",
    feature = "metal-output-placement"
))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use omega::{
    MetalError, allocate_placed_buffer_over, page_size, read_placed_buffer_f32,
    write_placed_buffer_f32,
};

#[test]
fn placed_buffer_over_mmap_range_round_trips() {
    let page = page_size();
    let mapped = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            2 * page,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_ANON | libc::MAP_PRIVATE,
            -1,
            0,
        )
    };
    assert_ne!(
        mapped,
        libc::MAP_FAILED,
        "anonymous mmap of two pages failed"
    );
    let base = mapped.cast::<u8>();

    let buffer = unsafe { allocate_placed_buffer_over(base, 2 * page) }
        .expect("a page-aligned mmap range must be accepted");

    write_placed_buffer_f32(&buffer, 0, &[1.0, 2.0, 3.0]);
    let through_pointer = unsafe { std::slice::from_raw_parts(base.cast::<f32>(), 3) };
    assert_eq!(through_pointer, &[1.0, 2.0, 3.0]);

    unsafe { std::slice::from_raw_parts_mut(base.cast::<f32>(), 3) }
        .copy_from_slice(&[4.0, 5.0, 6.0]);
    assert_eq!(read_placed_buffer_f32(&buffer, 0, 3), vec![4.0, 5.0, 6.0]);

    let refusals = [
        unsafe { allocate_placed_buffer_over(base.add(1), page) },
        unsafe { allocate_placed_buffer_over(base, page + 1) },
        unsafe { allocate_placed_buffer_over(std::ptr::null_mut(), page) },
        unsafe { allocate_placed_buffer_over(base, 0) },
    ];
    for refusal in refusals {
        assert!(matches!(refusal, Err(MetalError::CompileFailed { .. })));
    }

    drop(buffer);
    assert_eq!(unsafe { libc::munmap(mapped, 2 * page) }, 0);
}
