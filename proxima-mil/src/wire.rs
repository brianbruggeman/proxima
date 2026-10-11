//! Protobuf wire encoding into a caller buffer, `core` only.
//!
//! [`Encoder`] never allocates and never fails mid-stream: a write that does
//! not fit is counted and skipped, so an encoder over an empty buffer is the
//! measuring pre-pass and [`Encoder::finish`] reports the exact length either
//! way. Teaching pointer: the field numbers written through it are those of
//! `Model.proto` and `MIL.proto` (vendored under
//! `omega/tests/fixtures/coreml-lane/proto/coreml/`); `protoc --decode` is the
//! independent oracle of what it writes.

/// Why a caller buffer could not hold an encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum WireError {
    /// The buffer is shorter than the encoding; `needed` is the exact length.
    #[error("buffer too small: {needed} bytes needed")]
    BufferTooSmall { needed: usize },
    /// A weight source whose byte length is not a whole number of elements.
    #[error("weight source of {bytes} bytes is not a whole number of {width}-byte elements")]
    SourceLength { bytes: usize, width: usize },
}

/// The protobuf wire types the lane writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WireType {
    Varint = 0,
    LengthDelimited = 2,
}

/// An open length-delimited message: where its body starts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub struct Open {
    body_start: usize,
}

/// Writes protobuf into a caller buffer, counting what does not fit.
#[derive(Debug)]
pub struct Encoder<'buffer> {
    out: &'buffer mut [u8],
    position: usize,
    overflowed: bool,
}

impl<'buffer> Encoder<'buffer> {
    /// An encoder over `out`; an empty slice measures.
    pub fn new(out: &'buffer mut [u8]) -> Self {
        Self {
            out,
            position: 0,
            overflowed: false,
        }
    }

    /// Bytes encoded so far, counting those that did not fit.
    #[must_use]
    pub const fn position(&self) -> usize {
        self.position
    }

    /// The encoded length, or the length the buffer needed.
    ///
    /// # Errors
    /// [`WireError::BufferTooSmall`] when any write did not fit.
    pub const fn finish(self) -> Result<usize, WireError> {
        if self.overflowed {
            Err(WireError::BufferTooSmall {
                needed: self.position,
            })
        } else {
            Ok(self.position)
        }
    }

    fn put(&mut self, bytes: &[u8]) {
        let end = self.position + bytes.len();
        match self.out.get_mut(self.position..end) {
            Some(target) if !self.overflowed => target.copy_from_slice(bytes),
            _ => self.overflowed = true,
        }
        self.position = end;
    }

    /// Bytes written verbatim, e.g. the body of a length-delimited field already opened.
    pub fn raw(&mut self, bytes: &[u8]) {
        self.put(bytes);
    }

    /// A base-128 varint.
    pub fn varint(&mut self, value: u64) {
        let mut scratch = [0_u8; MAX_VARINT_BYTES];
        let width = encode_varint(value, &mut scratch);
        self.put(&scratch[..width]);
    }

    /// A field key: the field number and wire type.
    pub fn key(&mut self, field: u32, wire_type: WireType) {
        self.varint((u64::from(field) << 3) | wire_type as u64);
    }

    /// A varint field, written even when zero.
    pub fn varint_field(&mut self, field: u32, value: u64) {
        self.key(field, WireType::Varint);
        self.varint(value);
    }

    /// A length-delimited field holding `bytes` verbatim.
    pub fn bytes_field(&mut self, field: u32, bytes: &[u8]) {
        self.key(field, WireType::LengthDelimited);
        self.varint(bytes.len() as u64);
        self.put(bytes);
    }

    /// A packed repeated varint field.
    pub fn packed_varints(&mut self, field: u32, values: &[i64]) {
        let mut body = 0;
        values
            .iter()
            .for_each(|value| body += varint_len(*value as u64));
        self.key(field, WireType::LengthDelimited);
        self.varint(body as u64);
        values.iter().for_each(|value| self.varint(*value as u64));
    }

    /// A packed repeated varint field of unsigned extents.
    pub fn packed_extents(&mut self, field: u32, values: &[u64]) {
        let body: usize = values.iter().map(|value| varint_len(*value)).sum();
        self.key(field, WireType::LengthDelimited);
        self.varint(body as u64);
        values.iter().for_each(|value| self.varint(*value));
    }

    /// Starts a nested message; close it with [`Encoder::close`].
    pub fn open(&mut self, field: u32) -> Open {
        self.key(field, WireType::LengthDelimited);
        self.put(&[0]);
        Open {
            body_start: self.position,
        }
    }

    /// Ends a nested message, writing its length ahead of its body.
    ///
    /// The length is reserved as one byte and the body moves right by the extra
    /// width when the body is 128 bytes or longer, so every length is the
    /// minimal varint and the bytes match what `protoc --encode` writes.
    pub fn close(&mut self, open: Open) {
        let length = self.position - open.body_start;
        let width = varint_len(length as u64);
        let end = self.position + width - 1;
        if end > self.out.len() {
            self.overflowed = true;
        }
        if !self.overflowed {
            self.out
                .copy_within(open.body_start..self.position, open.body_start + width - 1);
            let mut scratch = [0_u8; MAX_VARINT_BYTES];
            encode_varint(length as u64, &mut scratch);
            self.out[open.body_start - 1..open.body_start - 1 + width]
                .copy_from_slice(&scratch[..width]);
        }
        self.position = end;
    }
}

/// The longest varint a `u64` takes.
pub const MAX_VARINT_BYTES: usize = 10;

/// Encodes `value` into the front of `scratch`, returning the byte count.
pub const fn encode_varint(mut value: u64, scratch: &mut [u8; MAX_VARINT_BYTES]) -> usize {
    let mut width = 0;
    loop {
        let low = (value & 0x7f) as u8;
        value >>= 7;
        if value == 0 {
            scratch[width] = low;
            return width + 1;
        }
        scratch[width] = low | 0x80;
        width += 1;
    }
}

/// Bytes a varint of `value` takes.
#[must_use]
pub const fn varint_len(value: u64) -> usize {
    let bits = 64 - (value | 1).leading_zeros() as usize;
    bits.div_ceil(7)
}
