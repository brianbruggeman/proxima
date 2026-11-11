#![cfg(all(feature = "metal-output-placement", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

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

const SENTINEL: f32 = -1.0;

fn sentinel_buffer() -> omega::PlacedBuffer {
    let buffer = allocate_placed_buffer(8).expect("allocates a placed buffer");
    try_write_placed_buffer_f32(&buffer, 0, &[SENTINEL, SENTINEL]).expect("seeds both f32 slots");
    buffer
}

fn assert_untouched(buffer: &omega::PlacedBuffer) {
    let mut bytes = [0.0_f32; 2];
    read_placed_buffer_f32_into(buffer, 0, &mut bytes).expect("reads both f32 slots");
    assert_eq!(bytes, [SENTINEL, SENTINEL], "a refused call must not write");
}

#[test]
fn placed_f32_owned_read_rejects_an_offset_past_the_end() {
    let buffer = sentinel_buffer();

    let error = omega::try_read_placed_buffer_f32(&buffer, 64, 2).expect_err("offset 64 is past an 8-byte buffer");

    assert!(matches!(
        error,
        MetalError::PlacedBufferRange { byte_offset: 64, byte_length: 8, buffer_length: 8 }
    ));
    assert_untouched(&buffer);
}

#[test]
fn placed_f32_write_rejects_an_offset_past_the_end_and_leaves_the_buffer_untouched() {
    let buffer = sentinel_buffer();

    let error = try_write_placed_buffer_f32(&buffer, 64, &[1.0, 2.0]).expect_err("offset 64 is past an 8-byte buffer");

    assert!(matches!(
        error,
        MetalError::PlacedBufferRange { byte_offset: 64, byte_length: 8, buffer_length: 8 }
    ));
    assert_untouched(&buffer);
}

#[test]
fn placed_f32_zero_range_rejects_a_range_past_the_end_and_leaves_the_buffer_untouched() {
    let buffer = sentinel_buffer();

    let error = omega::zero_placed_buffer_range(&buffer, 4, 8).expect_err("4 + 8 bytes overruns an 8-byte buffer");

    assert!(matches!(
        error,
        MetalError::PlacedBufferRange { byte_offset: 4, byte_length: 8, buffer_length: 8 }
    ));
    assert_untouched(&buffer);
}

#[test]
fn placed_f32_move_rejects_a_destination_past_the_end_and_leaves_the_buffer_untouched() {
    let buffer = sentinel_buffer();

    let error = omega::move_placed_buffer_bytes(&buffer, 0, 4, 8).expect_err("moving 8 bytes to offset 4 overruns");

    assert!(matches!(
        error,
        MetalError::PlacedBufferRange { byte_offset: 4, byte_length: 8, buffer_length: 8 }
    ));
    assert_untouched(&buffer);
}

#[test]
fn placed_f32_owned_read_returns_what_a_write_stored() {
    let buffer = allocate_placed_buffer(19).expect("allocates a placed buffer");
    try_write_placed_buffer_f32(&buffer, 3, &[1.25, -3.5, 17.0]).expect("writes the payload");

    let read = omega::try_read_placed_buffer_f32(&buffer, 3, 3).expect("reads the payload");

    assert_eq!(read, vec![1.25, -3.5, 17.0]);
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
