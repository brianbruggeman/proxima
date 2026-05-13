// Vendored from h2 crate (MIT). Original code at
// https://crates.io/crates/h2/0.4.14 — src/hpack/huffman/mod.rs.
// Vendored verbatim so the bench can call h2's private huffman
// decode directly. License header preserved below.
//
// Copyright (c) 2017 h2 authors. MIT-licensed.

#![allow(dead_code)]

pub mod integer;
pub mod static_lookup;
pub mod table;

use bytes::{BufMut, BytesMut};
use table::{DECODE_TABLE, ENCODE_TABLE};

const MAYBE_EOS: u8 = 1;
const DECODED: u8 = 2;
const ERROR: u8 = 4;

#[derive(Debug)]
pub struct DecoderError;

struct Decoder {
    state: u8,
    maybe_eos: bool,
}

pub fn encode(src: &[u8], dst: &mut BytesMut) {
    let mut bits: u64 = 0;
    let mut bits_left = 40;
    for &symbol in src {
        let (nbits, code) = ENCODE_TABLE[symbol as usize];
        bits |= code << (bits_left - nbits);
        bits_left -= nbits;
        while bits_left <= 32 {
            dst.put_u8((bits >> 32) as u8);
            bits <<= 8;
            bits_left += 8;
        }
    }
    if bits_left != 40 {
        bits |= (1 << bits_left) - 1;
        dst.put_u8((bits >> 32) as u8);
    }
}

pub fn decode(src: &[u8], buf: &mut BytesMut) -> Result<BytesMut, DecoderError> {
    let mut decoder = Decoder::new();
    buf.reserve(src.len() << 1);
    for byte in src {
        if let Some(out) = decoder.decode4(byte >> 4)? {
            buf.put_u8(out);
        }
        if let Some(out) = decoder.decode4(byte & 0xf)? {
            buf.put_u8(out);
        }
    }
    if !decoder.is_final() {
        return Err(DecoderError);
    }
    Ok(buf.split())
}

impl Decoder {
    fn new() -> Decoder {
        Decoder {
            state: 0,
            maybe_eos: false,
        }
    }

    fn decode4(&mut self, input: u8) -> Result<Option<u8>, DecoderError> {
        let (next, byte, flags) = DECODE_TABLE[self.state as usize][input as usize];
        if flags & ERROR == ERROR {
            return Err(DecoderError);
        }
        let mut ret = None;
        if flags & DECODED == DECODED {
            ret = Some(byte);
        }
        self.state = next;
        self.maybe_eos = flags & MAYBE_EOS == MAYBE_EOS;
        Ok(ret)
    }

    fn is_final(&self) -> bool {
        self.state == 0 || self.maybe_eos
    }
}
