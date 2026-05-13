//! Incremental message writer over a caller-owned output buffer.
//!
//! This is the encode-side primitive every typed encoder composes: it
//! writes the tag, reserves the Int32 length field, appends fields, and
//! patches the length on [`MessageWriter::finish`]. No allocation — when
//! the buffer is too small the writer reports how many bytes the failed
//! operation would have required so the caller can grow and retry.

use super::error::EncodeError;

#[derive(Debug)]
pub struct MessageWriter<'a> {
    out: &'a mut [u8],
    pos: usize,
    length_at: usize,
}

impl<'a> MessageWriter<'a> {
    /// Begins a tagged message: writes `tag`, reserves the length field.
    ///
    /// # Errors
    /// [`EncodeError::BufferTooSmall`] when `out` cannot hold the header.
    pub fn tagged(out: &'a mut [u8], tag: u8) -> Result<Self, EncodeError> {
        if out.len() < 5 {
            return Err(EncodeError::BufferTooSmall { needed: 5 });
        }
        out[0] = tag;
        Ok(Self {
            out,
            pos: 5,
            length_at: 1,
        })
    }

    /// Begins an untagged startup-phase message: reserves the length field.
    ///
    /// # Errors
    /// [`EncodeError::BufferTooSmall`] when `out` cannot hold the header.
    pub fn untagged(out: &'a mut [u8]) -> Result<Self, EncodeError> {
        if out.len() < 4 {
            return Err(EncodeError::BufferTooSmall { needed: 4 });
        }
        Ok(Self {
            out,
            pos: 4,
            length_at: 0,
        })
    }

    /// Bytes written so far, including the unpatched header.
    #[must_use]
    pub const fn written(&self) -> usize {
        self.pos
    }

    pub fn put_bytes(&mut self, bytes: &[u8]) -> Result<&mut Self, EncodeError> {
        let end = self.pos + bytes.len();
        if self.out.len() < end {
            return Err(EncodeError::BufferTooSmall { needed: end });
        }
        self.out[self.pos..end].copy_from_slice(bytes);
        self.pos = end;
        Ok(self)
    }

    pub fn put_u8(&mut self, value: u8) -> Result<&mut Self, EncodeError> {
        self.put_bytes(&[value])
    }

    pub fn put_i16(&mut self, value: i16) -> Result<&mut Self, EncodeError> {
        self.put_bytes(&value.to_be_bytes())
    }

    pub fn put_i32(&mut self, value: i32) -> Result<&mut Self, EncodeError> {
        self.put_bytes(&value.to_be_bytes())
    }

    pub fn put_u32(&mut self, value: u32) -> Result<&mut Self, EncodeError> {
        self.put_bytes(&value.to_be_bytes())
    }

    /// Writes `bytes` followed by the NUL terminator, rejecting embedded
    /// NULs (they would corrupt the frame for every later field).
    pub fn put_cstr(&mut self, bytes: &[u8]) -> Result<&mut Self, EncodeError> {
        if memchr::memchr(0, bytes).is_some() {
            return Err(EncodeError::InvalidValue {
                field: "string with embedded nul",
            });
        }
        self.put_bytes(bytes)?;
        self.put_u8(0)
    }

    /// Reserves `count` bytes for the caller to fill in place, returning
    /// the mutable slot — the zero-copy column write path.
    pub fn reserve(&mut self, count: usize) -> Result<&mut [u8], EncodeError> {
        let end = self.pos + count;
        if self.out.len() < end {
            return Err(EncodeError::BufferTooSmall { needed: end });
        }
        let slot = &mut self.out[self.pos..end];
        self.pos = end;
        Ok(slot)
    }

    /// Patches a previously written Int16 at `at` (used by counted-list
    /// writers to back-fill counts).
    pub(crate) fn patch_i16(&mut self, at: usize, value: i16) {
        self.out[at..at + 2].copy_from_slice(&value.to_be_bytes());
    }

    /// Patches the length field and returns the total encoded size.
    ///
    /// # Errors
    /// [`EncodeError::ValueTooLarge`] when the message exceeds `i32::MAX`.
    pub fn finish(self) -> Result<usize, EncodeError> {
        let length = self.pos - self.length_at;
        let Ok(length) = i32::try_from(length) else {
            return Err(EncodeError::ValueTooLarge {
                field: "message length",
            });
        };
        let at = self.length_at;
        self.out[at..at + 4].copy_from_slice(&length.to_be_bytes());
        Ok(self.pos)
    }
}

// one test helper builds a Vec<u8> frame; this crate carries no alloc
// dependency for its no_std tier, so the suite needs std, not just test
#[cfg(all(test, feature = "std"))]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use super::super::error::EncodeError;

    #[test]
    fn tagged_writer_encodes_tag_byte_and_length() {
        let mut buf = [0u8; 16];
        let mut writer = MessageWriter::tagged(&mut buf, b'Q').expect("tagged must succeed");
        writer.put_cstr(b"select 1").expect("put cstr must succeed");
        let written = writer.finish().expect("finish must succeed");
        assert_eq!(buf[0], b'Q', "tag byte must be Q");
        let length = i32::from_be_bytes([buf[1], buf[2], buf[3], buf[4]]);
        assert_eq!(
            length as usize,
            written - 1,
            "length field covers from byte 1"
        );
    }

    #[test]
    fn untagged_writer_encodes_length_only() {
        let mut buf = [0u8; 16];
        let mut writer = MessageWriter::untagged(&mut buf).expect("untagged must succeed");
        writer.put_i32(196608).expect("put i32 must succeed");
        let written = writer.finish().expect("finish must succeed");
        let length = i32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]);
        assert_eq!(length as usize, written, "length field covers all bytes");
    }

    #[test]
    fn tagged_header_too_small_returns_buffer_too_small() {
        let mut buf = [0u8; 4];
        let result = MessageWriter::tagged(&mut buf, b'Q');
        let Err(EncodeError::BufferTooSmall { needed: 5 }) = result else {
            panic!("expected BufferTooSmall with needed=5");
        };
    }

    #[test]
    fn untagged_header_too_small_returns_buffer_too_small() {
        let mut buf = [0u8; 3];
        let result = MessageWriter::untagged(&mut buf);
        let Err(EncodeError::BufferTooSmall { needed: 4 }) = result else {
            panic!("expected BufferTooSmall with needed=4");
        };
    }

    #[test]
    fn put_bytes_overflows_returns_buffer_too_small() {
        let mut buf = [0u8; 8];
        let mut writer = MessageWriter::tagged(&mut buf, b'Q').expect("tagged must succeed");
        let big_payload = vec![0u8; 10];
        let result = writer.put_bytes(&big_payload);
        assert!(matches!(result, Err(EncodeError::BufferTooSmall { .. })));
    }

    #[test]
    fn put_cstr_rejects_embedded_nul() {
        let mut buf = [0u8; 32];
        let mut writer = MessageWriter::tagged(&mut buf, b'Q').expect("tagged must succeed");
        let result = writer.put_cstr(b"nul\0inside");
        assert!(matches!(result, Err(EncodeError::InvalidValue { .. })));
    }

    #[test]
    fn patch_i16_updates_placeholder_correctly() {
        let mut buf = [0u8; 16];
        let mut writer = MessageWriter::tagged(&mut buf, b'T').expect("tagged must succeed");
        let count_at = writer.written();
        writer.put_i16(0).expect("put placeholder must succeed");
        writer.put_i32(42).expect("put i32 must succeed");
        writer.patch_i16(count_at, 1);
        let written = writer.finish().expect("finish must succeed");
        assert_eq!(
            i16::from_be_bytes([buf[5], buf[6]]),
            1,
            "patched count must be 1"
        );
        assert!(written > 0);
    }

    #[test]
    fn reserve_returns_writable_slice() {
        let mut buf = [0u8; 16];
        let mut writer = MessageWriter::tagged(&mut buf, b'D').expect("tagged must succeed");
        writer.put_i32(4).expect("put length must succeed");
        let slot = writer.reserve(4).expect("reserve must succeed");
        slot.copy_from_slice(&42i32.to_be_bytes());
        let written = writer.finish().expect("finish must succeed");
        assert_eq!(i32::from_be_bytes([buf[9], buf[10], buf[11], buf[12]]), 42);
        assert!(written > 0);
    }

    #[test]
    fn reserve_exceeds_buffer_returns_buffer_too_small() {
        let mut buf = [0u8; 8];
        let mut writer = MessageWriter::tagged(&mut buf, b'D').expect("tagged must succeed");
        let result = writer.reserve(100);
        assert!(matches!(result, Err(EncodeError::BufferTooSmall { .. })));
    }
}
