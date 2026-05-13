// Vendored from h2-0.4.14 (MIT).
// - encode_int from src/hpack/encoder.rs:264-288
// - decode_int from src/hpack/decoder.rs:391-447
// Wrapper signatures adapted to match proxima's `&[u8] -> (value, consumed)`
// shape so the bench measures algorithm cost, not API plumbing.

#![allow(dead_code)]

use bytes::{Buf, BufMut};

#[derive(Debug)]
pub struct DecoderError;

pub fn encode_int<B: BufMut>(mut value: usize, prefix_bits: usize, first_byte: u8, dst: &mut B) {
    if encode_int_one_byte(value, prefix_bits) {
        dst.put_u8(first_byte | value as u8);
        return;
    }
    let low = (1 << prefix_bits) - 1;
    value -= low;
    dst.put_u8(first_byte | low as u8);
    while value >= 128 {
        dst.put_u8(0b1000_0000 | value as u8);
        value >>= 7;
    }
    dst.put_u8(value as u8);
}

fn encode_int_one_byte(value: usize, prefix_bits: usize) -> bool {
    value < (1 << prefix_bits) - 1
}

pub fn decode_int(buf: &[u8], prefix_size: u8) -> Result<(usize, usize), DecoderError> {
    const MAX_BYTES: usize = 5;
    const VARINT_MASK: u8 = 0b0111_1111;
    const VARINT_FLAG: u8 = 0b1000_0000;
    if !(1..=8).contains(&prefix_size) {
        return Err(DecoderError);
    }
    let mut cursor: &[u8] = buf;
    if !cursor.has_remaining() {
        return Err(DecoderError);
    }
    let mask = if prefix_size == 8 {
        0xFF
    } else {
        (1u8 << prefix_size).wrapping_sub(1)
    };
    let mut ret = (cursor.get_u8() & mask) as usize;
    if ret < mask as usize {
        return Ok((ret, buf.len() - cursor.len()));
    }
    let mut bytes = 1usize;
    let mut shift = 0u32;
    while cursor.has_remaining() {
        let byte = cursor.get_u8();
        bytes += 1;
        ret += ((byte & VARINT_MASK) as usize) << shift;
        shift += 7;
        if byte & VARINT_FLAG == 0 {
            return Ok((ret, buf.len() - cursor.len()));
        }
        if bytes == MAX_BYTES {
            return Err(DecoderError);
        }
    }
    Err(DecoderError)
}
