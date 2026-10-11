//! The `weights/weight.bin` blob of an ML Program package, `core` only.
//!
//! Layout, read from an installed compiled model's `weights/weight.bin`
//! (`SNLanguageAlignedAVFuserModel.mlmodelc`, five fp16 entries) and checked
//! by the compiler's acceptance: a 64-byte file header (entry count and
//! version 2, both `u32`), then per entry a 64-byte descriptor (sentinel
//! `0xDEADBEEF`, element code, byte size, absolute data offset) at a 64-byte
//! aligned position followed by the data padded to the next 64. A MIL
//! `BLOBFILE` offset names the descriptor, not the data.
//!
//! [`BlobWriter`] converts each source tensor to fp16 in the same pass that
//! copies it into the blob, so no decoded copy of a weight exists.

use half::f16;

use crate::wire::WireError;

/// Alignment of every descriptor and data run in the blob.
pub const ALIGNMENT: usize = 64;

/// Bytes of the file header and of each entry descriptor.
pub const DESCRIPTOR_BYTES: usize = 64;

const SENTINEL: u32 = 0xDEAD_BEEF;
const BLOB_VERSION: u32 = 2;
const FP16_CODE: u32 = 1;
const FP16_BYTES: usize = 2;

/// The element type of a weight source view.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceDType {
    /// IEEE binary32, little endian, e.g. an ONNX initializer `raw_data`.
    Float32,
    /// IEEE binary16, little endian, copied through.
    Float16,
}

impl SourceDType {
    /// Bytes per source element.
    #[must_use]
    pub const fn width(self) -> usize {
        match self {
            Self::Float32 => 4,
            Self::Float16 => 2,
        }
    }
}

/// Bytes an entry of `elements` fp16 values occupies, descriptor included.
#[must_use]
pub const fn entry_bytes(elements: usize) -> usize {
    DESCRIPTOR_BYTES + align_up(elements * FP16_BYTES)
}

const fn align_up(length: usize) -> usize {
    length.div_ceil(ALIGNMENT) * ALIGNMENT
}

/// Writes entries into a caller blob, counting what does not fit.
#[derive(Debug)]
pub struct BlobWriter<'blob> {
    out: &'blob mut [u8],
    position: usize,
    entries: u32,
    overflowed: bool,
}

impl<'blob> BlobWriter<'blob> {
    /// A writer over `out`, positioned after the file header; an empty slice measures.
    pub fn new(out: &'blob mut [u8]) -> Self {
        Self {
            out,
            position: DESCRIPTOR_BYTES,
            entries: 0,
            overflowed: false,
        }
    }

    /// Appends one tensor as fp16 and returns the offset a `BLOBFILE` names.
    ///
    /// # Errors
    /// [`WireError::SourceLength`] when `source` is not whole elements.
    pub fn write_weight(&mut self, source: &[u8], dtype: SourceDType) -> Result<usize, WireError> {
        let width = dtype.width();
        if !source.len().is_multiple_of(width) {
            return Err(WireError::SourceLength {
                bytes: source.len(),
                width,
            });
        }
        let elements = source.len() / width;
        let start = self.position;
        let end = start + entry_bytes(elements);
        if end > self.out.len() {
            self.overflowed = true;
        }
        if !self.overflowed {
            self.fill_entry(start, source, dtype);
        }
        self.position = end;
        self.entries += 1;
        Ok(start)
    }

    fn fill_entry(&mut self, start: usize, source: &[u8], dtype: SourceDType) {
        let elements = source.len() / dtype.width();
        let data_start = start + DESCRIPTOR_BYTES;
        let data_len = elements * FP16_BYTES;
        let descriptor = &mut self.out[start..data_start];
        descriptor.fill(0);
        descriptor[0..4].copy_from_slice(&SENTINEL.to_le_bytes());
        descriptor[4..8].copy_from_slice(&FP16_CODE.to_le_bytes());
        descriptor[8..16].copy_from_slice(&(data_len as u64).to_le_bytes());
        descriptor[16..24].copy_from_slice(&(data_start as u64).to_le_bytes());
        let data = &mut self.out[data_start..data_start + align_up(data_len)];
        data.fill(0);
        convert_into(source, dtype, &mut data[..data_len]);
    }

    /// Bytes the blob holds so far, counting entries that did not fit.
    #[must_use]
    pub const fn length(&self) -> usize {
        self.position
    }

    /// Writes the file header and returns the blob length, or the length needed.
    ///
    /// # Errors
    /// [`WireError::BufferTooSmall`] when any entry did not fit.
    pub fn finish(self) -> Result<usize, WireError> {
        let length = self.position.max(DESCRIPTOR_BYTES);
        if self.overflowed || self.out.len() < DESCRIPTOR_BYTES {
            return Err(WireError::BufferTooSmall { needed: length });
        }
        let header = &mut self.out[..DESCRIPTOR_BYTES];
        header.fill(0);
        header[0..4].copy_from_slice(&self.entries.to_le_bytes());
        header[4..8].copy_from_slice(&BLOB_VERSION.to_le_bytes());
        Ok(length)
    }
}

fn convert_into(source: &[u8], dtype: SourceDType, target: &mut [u8]) {
    match dtype {
        SourceDType::Float16 => target.copy_from_slice(source),
        SourceDType::Float32 => source
            .as_chunks::<4>()
            .0
            .iter()
            .zip(target.as_chunks_mut::<FP16_BYTES>().0.iter_mut())
            .for_each(|(element, slot)| {
                *slot = f16::from_f32(f32::from_le_bytes(*element)).to_le_bytes();
            }),
    }
}
