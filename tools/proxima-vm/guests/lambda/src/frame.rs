//! Hand-encoded postcard frames for the fd-keyed `ChildRequest` /
//! `ChildResponse` protocol (`proxima-protocols/src/process/protocol.rs`),
//! with no allocator. Pure arithmetic over slices -- no `asm`, no `static` --
//! so the host-side test `tests/lambda_frame.rs` includes this very file and
//! asserts its bytes against `postcard::to_allocvec` of the same values.

// the decoder covers every `ChildResponse` variant so the host parity test can
// pin them all; a given bin reads only the fields it uses.
#![allow(dead_code)]

pub const FRAME_CAPACITY: usize = 512;

pub const READ_VERB: u16 = 0x00;
pub const WRITE_VERB: u16 = 0x01;

const REQUEST_READ: u32 = 0;
const REQUEST_WRITE: u32 = 1;

const RESPONSE_READ: u32 = 0;
const RESPONSE_WRITE: u32 = 1;
const RESPONSE_OPEN: u32 = 2;
const RESPONSE_CLOSE: u32 = 3;
const RESPONSE_STAT: u32 = 4;
const RESPONSE_ERROR: u32 = 5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameError {
    BufferTooSmall,
    Truncated,
    VarintOverflow,
    UnknownVariant(u32),
    InvalidBool(u8),
}

impl FrameError {
    pub fn tag(self) -> u8 {
        match self {
            Self::BufferTooSmall => 1,
            Self::Truncated => 2,
            Self::VarintOverflow => 3,
            Self::UnknownVariant(variant) => 0x10 | (variant as u8 & 0x0f),
            Self::InvalidBool(byte) => 0x20 | (byte & 0x0f),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Response<'frame> {
    Read { bytes: &'frame [u8], eof: bool },
    Write { bytes_written: u32 },
    Open { handle: i32 },
    Close,
    Stat { size: u64, mode: u32, is_directory: bool },
    Error { errno: i32 },
}

struct Writer<'buffer> {
    buffer: &'buffer mut [u8],
    length: usize,
}

impl Writer<'_> {
    fn push(&mut self, byte: u8) -> Result<(), FrameError> {
        let slot = self.buffer.get_mut(self.length).ok_or(FrameError::BufferTooSmall)?;
        *slot = byte;
        self.length += 1;
        Ok(())
    }

    fn varint(&mut self, mut value: u64) -> Result<(), FrameError> {
        while value >= 0x80 {
            self.push((value as u8 & 0x7f) | 0x80)?;
            value >>= 7;
        }
        self.push(value as u8)
    }

    fn zigzag(&mut self, value: i32) -> Result<(), FrameError> {
        self.varint(u64::from(((value << 1) ^ (value >> 31)) as u32))
    }

    fn bytes(&mut self, bytes: &[u8]) -> Result<(), FrameError> {
        let end = self.length.checked_add(bytes.len()).ok_or(FrameError::BufferTooSmall)?;
        self.buffer
            .get_mut(self.length..end)
            .ok_or(FrameError::BufferTooSmall)?
            .copy_from_slice(bytes);
        self.length = end;
        Ok(())
    }
}

pub fn encode_read(
    buffer: &mut [u8],
    handle: i32,
    max_bytes: u32,
    offset: u64,
) -> Result<usize, FrameError> {
    let mut writer = Writer { buffer, length: 0 };
    writer.varint(u64::from(REQUEST_READ))?;
    writer.zigzag(handle)?;
    writer.varint(u64::from(max_bytes))?;
    writer.varint(offset)?;
    Ok(writer.length)
}

pub fn encode_write(buffer: &mut [u8], handle: i32, bytes: &[u8]) -> Result<usize, FrameError> {
    let mut writer = Writer { buffer, length: 0 };
    writer.varint(u64::from(REQUEST_WRITE))?;
    writer.zigzag(handle)?;
    writer.varint(bytes.len() as u64)?;
    writer.bytes(bytes)?;
    Ok(writer.length)
}

struct Reader<'frame> {
    frame: &'frame [u8],
    position: usize,
}

impl<'frame> Reader<'frame> {
    fn byte(&mut self) -> Result<u8, FrameError> {
        let byte = *self.frame.get(self.position).ok_or(FrameError::Truncated)?;
        self.position += 1;
        Ok(byte)
    }

    fn varint(&mut self) -> Result<u64, FrameError> {
        let mut value = 0_u64;
        for shift in (0..64).step_by(7) {
            let byte = self.byte()?;
            let payload = u64::from(byte & 0x7f);
            if shift == 63 && payload > 1 {
                return Err(FrameError::VarintOverflow);
            }
            value |= payload << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
        Err(FrameError::VarintOverflow)
    }

    fn varint_u32(&mut self) -> Result<u32, FrameError> {
        u32::try_from(self.varint()?).map_err(|_| FrameError::VarintOverflow)
    }

    fn zigzag(&mut self) -> Result<i32, FrameError> {
        let raw = self.varint_u32()?;
        Ok(((raw >> 1) as i32) ^ -((raw & 1) as i32))
    }

    fn flag(&mut self) -> Result<bool, FrameError> {
        match self.byte()? {
            0 => Ok(false),
            1 => Ok(true),
            other => Err(FrameError::InvalidBool(other)),
        }
    }

    fn bytes(&mut self) -> Result<&'frame [u8], FrameError> {
        let length = usize::try_from(self.varint()?).map_err(|_| FrameError::Truncated)?;
        let end = self.position.checked_add(length).ok_or(FrameError::Truncated)?;
        let bytes = self.frame.get(self.position..end).ok_or(FrameError::Truncated)?;
        self.position = end;
        Ok(bytes)
    }
}

pub fn decode_response(frame: &[u8]) -> Result<Response<'_>, FrameError> {
    let mut reader = Reader { frame, position: 0 };
    match reader.varint_u32()? {
        RESPONSE_READ => Ok(Response::Read {
            bytes: reader.bytes()?,
            eof: reader.flag()?,
        }),
        RESPONSE_WRITE => Ok(Response::Write {
            bytes_written: reader.varint_u32()?,
        }),
        RESPONSE_OPEN => Ok(Response::Open {
            handle: reader.zigzag()?,
        }),
        RESPONSE_CLOSE => Ok(Response::Close),
        RESPONSE_STAT => Ok(Response::Stat {
            size: reader.varint()?,
            mode: reader.varint_u32()?,
            is_directory: reader.flag()?,
        }),
        RESPONSE_ERROR => Ok(Response::Error {
            errno: reader.zigzag()?,
        }),
        other => Err(FrameError::UnknownVariant(other)),
    }
}
