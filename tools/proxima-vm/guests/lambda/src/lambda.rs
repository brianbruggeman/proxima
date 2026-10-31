//! Tier-3 (no alloc) runtime for an elf guest: the request body arrives on
//! handle 0 and the response body leaves on handle 1, both over the fd-keyed
//! `ChildRequest` hypercall channel (`frame.rs` encodes the frames).
//!
//! Every exchange reuses one `FRAME_CAPACITY` stack buffer for the request
//! and the host's in-place reply. The host returns the reply's full length
//! even when the buffer held less, so a reply longer than the buffer is
//! detected here and reported, never decoded from a truncated frame.

use core::arch::naked_asm;

use crate::frame::{
    self, FRAME_CAPACITY, FrameError, READ_VERB, Response, WRITE_VERB, decode_response,
};
use crate::hypercall::hypercall;

const INPUT_HANDLE: i32 = 0;
const OUTPUT_HANDLE: i32 = 1;
const CHUNK_BYTES: usize = 256;
const EMIT_VERB: u16 = 0xfffe;
const HALT_VERB: u16 = 0xffff;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LambdaError {
    Frame(FrameError),
    ReplyTruncated { needed: usize },
    Host { errno: i32 },
    InputTooLarge,
    InputStalled,
    OutputRefused { written: u32 },
    UnexpectedReply,
}

impl LambdaError {
    fn diagnostic(self) -> [u8; 2] {
        match self {
            Self::Frame(error) => [1, error.tag()],
            Self::ReplyTruncated { needed } => [2, needed.min(255) as u8],
            Self::Host { errno } => [3, errno as u8],
            Self::InputTooLarge => [4, 0],
            Self::InputStalled => [5, 0],
            Self::OutputRefused { written } => [6, written.min(255) as u8],
            Self::UnexpectedReply => [7, 0],
        }
    }
}

impl From<FrameError> for LambdaError {
    fn from(error: FrameError) -> Self {
        Self::Frame(error)
    }
}

#[cfg(target_arch = "aarch64")]
#[unsafe(no_mangle)]
#[unsafe(naked)]
extern "C" fn _start() -> ! {
    naked_asm!(
        "adrp x0, __stack_top",
        "add x0, x0, :lo12:__stack_top",
        "mov sp, x0",
        "b {entry}",
        entry = sym crate::lambda_main,
    );
}

#[cfg(target_arch = "x86_64")]
#[unsafe(no_mangle)]
#[unsafe(naked)]
extern "C" fn _start() -> ! {
    naked_asm!(
        "lea rsp, [rip + __stack_top]",
        "call {entry}",
        entry = sym crate::lambda_main,
    );
}

fn exchange(
    verb: u16,
    frame: &mut [u8; FRAME_CAPACITY],
) -> Result<Response<'_>, LambdaError> {
    let returned = unsafe { hypercall(verb, frame) } as usize;
    if returned > FRAME_CAPACITY {
        return Err(LambdaError::ReplyTruncated { needed: returned });
    }
    Ok(decode_response(&frame[..returned])?)
}

fn read_chunk(
    frame: &mut [u8; FRAME_CAPACITY],
    offset: u64,
) -> Result<(&[u8], bool), LambdaError> {
    frame::encode_read(frame, INPUT_HANDLE, CHUNK_BYTES as u32, offset)?;
    match exchange(READ_VERB, frame)? {
        Response::Read { bytes, eof } => Ok((bytes, eof)),
        Response::Error { errno } => Err(LambdaError::Host { errno }),
        _ => Err(LambdaError::UnexpectedReply),
    }
}

pub fn read_input(buffer: &mut [u8]) -> Result<usize, LambdaError> {
    let mut frame = [0_u8; FRAME_CAPACITY];
    let mut length = 0_usize;
    loop {
        let (bytes, eof) = read_chunk(&mut frame, length as u64)?;
        let end = length + bytes.len();
        buffer
            .get_mut(length..end)
            .ok_or(LambdaError::InputTooLarge)?
            .copy_from_slice(bytes);
        if eof {
            return Ok(end);
        }
        if bytes.is_empty() {
            return Err(LambdaError::InputStalled);
        }
        length = end;
    }
}

pub fn write_output(bytes: &[u8]) -> Result<(), LambdaError> {
    let mut frame = [0_u8; FRAME_CAPACITY];
    for chunk in bytes.chunks(CHUNK_BYTES) {
        frame::encode_write(&mut frame, OUTPUT_HANDLE, chunk)?;
        match exchange(WRITE_VERB, &mut frame)? {
            Response::Write { bytes_written } if bytes_written as usize == chunk.len() => {}
            Response::Write { bytes_written } => {
                return Err(LambdaError::OutputRefused {
                    written: bytes_written,
                });
            }
            Response::Error { errno } => return Err(LambdaError::Host { errno }),
            _ => return Err(LambdaError::UnexpectedReply),
        }
    }
    Ok(())
}

pub fn halt() -> ! {
    let mut nothing = [0_u8; 0];
    unsafe { hypercall(HALT_VERB, &mut nothing) };
    loop {
        core::hint::spin_loop();
    }
}

/// Emits `error`'s two-byte diagnostic over the emit verb (the host surfaces
/// it in the run's failure message), then traps so the host reports a failed
/// run instead of a clean halt with a silently short or empty output.
pub fn fail(error: LambdaError) -> ! {
    for byte in error.diagnostic() {
        let mut emit = [byte];
        unsafe { hypercall(EMIT_VERB, &mut emit) };
    }
    abort()
}

#[cfg(target_arch = "x86_64")]
pub fn abort() -> ! {
    unsafe { core::arch::asm!("ud2", options(noreturn, nostack)) }
}

#[cfg(target_arch = "aarch64")]
pub fn abort() -> ! {
    unsafe { core::arch::asm!("udf #0", options(noreturn, nostack)) }
}
