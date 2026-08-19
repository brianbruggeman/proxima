//! A rollback-on-shortfall byte cursor. Every `read_*` either advances past
//! a complete field or leaves the cursor exactly where it started — the
//! primitive [`crate::parser::GgufParser`] uses to retry a whole item from
//! scratch each time `poll` is called, without ever consuming a partial
//! field. GGUF numeric fields are little-endian (matches every real GGUF
//! file, all produced on little-endian hosts; `gguf.h`'s endian-swap note
//! at the top only concerns cross-endian *production*, not the wire shape).

use alloc::string::String;
use alloc::vec::Vec;

/// Borrowed cursor over one buffered chunk. Read methods return `None`
/// (without moving `pos`) when the field would run past `buf`'s end.
pub struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    #[must_use]
    pub fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    /// Bytes consumed so far — the caller drains this many bytes from the
    /// front of its accumulation buffer once a full item parses.
    #[must_use]
    pub fn consumed(&self) -> usize {
        self.pos
    }

    fn take(&mut self, len: usize) -> Option<&'a [u8]> {
        let end = self.pos.checked_add(len)?;
        let slice = self.buf.get(self.pos..end)?;
        self.pos = end;
        Some(slice)
    }

    pub fn u8(&mut self) -> Option<u8> {
        self.take(1).map(|bytes| bytes[0])
    }

    pub fn i8(&mut self) -> Option<i8> {
        self.u8().map(|value| value as i8)
    }

    pub fn u16(&mut self) -> Option<u16> {
        self.take(2)
            .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    pub fn i16(&mut self) -> Option<i16> {
        self.u16().map(|value| value as i16)
    }

    pub fn u32(&mut self) -> Option<u32> {
        self.take(4)
            .map(|bytes| u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    pub fn i32(&mut self) -> Option<i32> {
        self.u32().map(|value| value as i32)
    }

    pub fn u64(&mut self) -> Option<u64> {
        self.take(8).map(|bytes| {
            let mut raw = [0u8; 8];
            raw.copy_from_slice(bytes);
            u64::from_le_bytes(raw)
        })
    }

    pub fn i64(&mut self) -> Option<i64> {
        self.u64().map(|value| value as i64)
    }

    pub fn f32(&mut self) -> Option<f32> {
        self.u32().map(f32::from_bits)
    }

    pub fn f64(&mut self) -> Option<f64> {
        self.u64().map(f64::from_bits)
    }

    /// `bool` is stored as `int8_t`, nonzero is true (`gguf.h:28`).
    pub fn bool(&mut self) -> Option<bool> {
        self.i8().map(|value| value != 0)
    }

    /// `[len: u64][bytes]`, no nul terminator (`gguf.h:26`). `None` for a
    /// short buffer; `Some(Err(..))` for a length that doesn't fit `usize`
    /// or bytes that aren't valid UTF-8 — those are real parse errors, not
    /// "need more input".
    pub fn string(&mut self) -> Option<Result<String, StringError>> {
        let save = self.pos;
        let len = self.u64()?;
        let Ok(len_usize) = usize::try_from(len) else {
            self.pos = save;
            return Some(Err(StringError::TooLarge(len)));
        };
        let Some(bytes) = self.take(len_usize) else {
            self.pos = save;
            return None;
        };
        match core::str::from_utf8(bytes) {
            Ok(text) => Some(Ok(text.into())),
            Err(_) => Some(Err(StringError::InvalidUtf8)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StringError {
    TooLarge(u64),
    InvalidUtf8,
}

/// A `Vec<u8>`-backed accumulation buffer with a `Reader` view and a
/// front-drain once a `Reader` reports how much it consumed.
#[derive(Default)]
pub struct Accumulator {
    buf: Vec<u8>,
}

impl Accumulator {
    #[must_use]
    pub fn new() -> Self {
        Self { buf: Vec::new() }
    }

    pub fn extend(&mut self, bytes: &[u8]) {
        self.buf.extend_from_slice(bytes);
    }

    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        &self.buf
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    pub fn drain_front(&mut self, len: usize) {
        self.buf.drain(0..len);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_reads_advance_and_roll_back_on_shortfall() {
        let bytes = [0x01u8, 0x00, 0x00, 0x00];
        let mut reader = Reader::new(&bytes);
        assert_eq!(reader.u32(), Some(1));
        assert_eq!(reader.consumed(), 4);

        let mut short = Reader::new(&bytes[..3]);
        assert_eq!(short.u32(), None);
        assert_eq!(short.consumed(), 0, "shortfall must not move the cursor");
    }

    #[test]
    fn string_round_trips_and_rolls_back_on_shortfall() {
        let mut bytes = 5u64.to_le_bytes().to_vec();
        bytes.extend_from_slice(b"hello");
        let mut reader = Reader::new(&bytes);
        assert_eq!(reader.string(), Some(Ok("hello".into())));

        let mut truncated = Reader::new(&bytes[..bytes.len() - 2]);
        assert_eq!(truncated.string(), None);
        assert_eq!(truncated.consumed(), 0);
    }

    #[test]
    fn string_rejects_invalid_utf8_without_rollback_semantics_masking_the_error() {
        let mut bytes = 1u64.to_le_bytes().to_vec();
        bytes.push(0xFF);
        let mut reader = Reader::new(&bytes);
        assert_eq!(
            reader.string(),
            Some(Err(StringError::InvalidUtf8))
        );
    }
}
