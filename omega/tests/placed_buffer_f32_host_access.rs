#![cfg(all(feature = "metal-output-placement", target_os = "macos"))]

use omega::metal::{
    MetalError, allocate_placed_buffer, read_placed_buffer_f32_into, try_write_placed_buffer_f32,
};

#[test]
fn placed_f32_host_access_copies_real_values_at_byte_offsets() {
    let buffer = allocate_placed_buffer(19).expect("allocates a placed buffer");
    let source = [1.25_f32, -3.5_f32, 17.0_f32];
    try_write_placed_buffer_f32(&buffer, 3, &source).expect("writes the f32 payload");

    let mut destination = [0.0_f32; 3];
    read_placed_buffer_f32_into(&buffer, 3, &mut destination).expect("reads the f32 payload");

    assert_eq!(destination, source);
}

#[test]
fn placed_f32_host_access_accepts_an_empty_range_at_buffer_end() {
    let buffer = allocate_placed_buffer(8).expect("allocates a placed buffer");

    try_write_placed_buffer_f32(&buffer, 8, &[]).expect("writes an empty end range");
    read_placed_buffer_f32_into(&buffer, 8, &mut []).expect("reads an empty end range");
}

#[test]
fn placed_f32_host_access_rejects_out_of_bounds_ranges() {
    let buffer = allocate_placed_buffer(8).expect("allocates a placed buffer");
    let payload = [9.0_f32];

    let error = try_write_placed_buffer_f32(&buffer, 8, &payload)
        .expect_err("rejects a nonempty range beginning at the buffer end");

    assert!(matches!(
        error,
        MetalError::PlacedBufferRange {
            byte_offset: 8,
            byte_length: 4,
            buffer_length: 8,
        }
    ));
}

#[test]
fn placed_f32_host_access_rejects_offset_overflow() {
    let buffer = allocate_placed_buffer(8).expect("allocates a placed buffer");

    let error = read_placed_buffer_f32_into(&buffer, usize::MAX - 1, &mut [0.0_f32])
        .expect_err("rejects an overflowing range");

    assert!(matches!(
        error,
        MetalError::PlacedBufferRange {
            byte_offset,
            byte_length: 4,
            buffer_length: 8,
        } if byte_offset == usize::MAX - 1
    ));
}
