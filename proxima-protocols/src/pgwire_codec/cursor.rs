//! Internal byte reader over a framed message body.
//!
//! All multi-byte integers on the PostgreSQL wire are big-endian
//! (network order). The reader carries the frame's tag byte so every
//! error names the message it occurred in.

use memchr::memchr;

use super::error::ParseError;
use super::types::PgStr;

pub(crate) struct Reader<'a> {
    body: &'a [u8],
    pos: usize,
    tag: u8,
}

impl<'a> Reader<'a> {
    pub(crate) const fn new(body: &'a [u8], tag: u8) -> Self {
        Self { body, pos: 0, tag }
    }

    pub(crate) const fn tag(&self) -> u8 {
        self.tag
    }

    pub(crate) const fn remaining(&self) -> usize {
        self.body.len() - self.pos
    }

    pub(crate) fn take_bytes(&mut self, count: usize) -> Result<&'a [u8], ParseError> {
        if self.remaining() < count {
            return Err(ParseError::Truncated { tag: self.tag });
        }
        let slice = &self.body[self.pos..self.pos + count];
        self.pos += count;
        Ok(slice)
    }

    pub(crate) fn take_u8(&mut self) -> Result<u8, ParseError> {
        Ok(self.take_bytes(1)?[0])
    }

    pub(crate) fn take_i16(&mut self) -> Result<i16, ParseError> {
        let bytes = self.take_bytes(2)?;
        Ok(i16::from_be_bytes([bytes[0], bytes[1]]))
    }

    pub(crate) fn take_i32(&mut self) -> Result<i32, ParseError> {
        let bytes = self.take_bytes(4)?;
        Ok(i32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    pub(crate) fn take_u32(&mut self) -> Result<u32, ParseError> {
        let bytes = self.take_bytes(4)?;
        Ok(u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    /// Non-negative Int16 count field, widened to usize.
    pub(crate) fn take_count16(&mut self, field: &'static str) -> Result<usize, ParseError> {
        let raw = self.take_i16()?;
        usize::try_from(raw).map_err(|_| ParseError::InvalidValue {
            tag: self.tag,
            field,
        })
    }

    pub(crate) fn take_cstr(&mut self) -> Result<PgStr<'a>, ParseError> {
        let rest = &self.body[self.pos..];
        let Some(nul) = memchr(0, rest) else {
            return Err(ParseError::MissingNul { tag: self.tag });
        };
        self.pos += nul + 1;
        Ok(PgStr::new(&rest[..nul]))
    }

    pub(crate) fn take_rest(&mut self) -> &'a [u8] {
        let rest = &self.body[self.pos..];
        self.pos = self.body.len();
        rest
    }

    /// Slice from `mark` to the current position — used to capture a
    /// validated sub-section for a lazy iterator view.
    pub(crate) fn since(&self, mark: usize) -> &'a [u8] {
        &self.body[mark..self.pos]
    }

    /// Slice between two previously observed positions.
    pub(crate) fn slice(&self, from: usize, to: usize) -> &'a [u8] {
        &self.body[from..to]
    }

    pub(crate) const fn mark(&self) -> usize {
        self.pos
    }

    pub(crate) fn expect_end(&self) -> Result<(), ParseError> {
        if self.remaining() != 0 {
            return Err(ParseError::TrailingBytes {
                tag: self.tag,
                trailing: self.remaining(),
            });
        }
        Ok(())
    }
}
