//! Synchronous IPC layer that drives the dispatch chain over a
//! length-prefixed byte stream.
//!
//! Functions in this module read framed [`ChildRequest`]s from a
//! reader, dispatch them via a caller-supplied function, and write
//! framed [`ChildResponse`]s to a writer. The wire format is the
//! one from [`super::framing`]: `[u32_be length][postcard payload]`.
//!
//! # Tier
//!
//! tier-2 — uses `std::io::{Read, Write}`. Future async variant
//! (`AsyncRead` / `AsyncWrite`) can land when proxima-pipe's
//! fd-source abstraction is available; sync is sufficient for the
//! parent-side dispatch task while we wait for that.
//!
//! # Composition
//!
//! The dispatcher loop is small enough to inline:
//!
//! ```ignore
//! while let Some(request) = read_frame::<ChildRequest>(reader)? {
//!     let response = dispatch_fn(request);
//!     write_frame(writer, &response)?;
//! }
//! ```
//!
//! [`run_dispatch_loop`] wraps this with a `FnMut(ChildRequest) ->
//! ChildResponse` callback so the dispatch can be any of:
//! `dispatch_match(req, routes, fallback)`, a closure that uses
//! `match` over variants directly, or an `AndThen<...>::dispatch`
//! call wrapping the whole chain.

use std::io::{self, Read, Write};

use proxima_primitives::pipe::ProximaError;
use serde::{Deserialize, Serialize};

use super::protocol::{ChildRequest, ChildResponse};

/// Read a single framed value from `reader`. Returns `Ok(None)` on
/// clean EOF (zero bytes available when starting to read the length
/// prefix), `Ok(Some(v))` on successful decode, and `Err` on I/O
/// failure or malformed frame.
///
/// # Errors
///
/// - `io::ErrorKind::UnexpectedEof` if the length prefix or payload
///   is truncated mid-frame.
/// - `io::ErrorKind::InvalidData` if the postcard payload doesn't
///   decode to `T`.
/// - any I/O error from `reader.read_exact`.
pub fn read_frame<R: Read, T>(reader: &mut R) -> io::Result<Option<T>>
where
    T: for<'de> Deserialize<'de>,
{
    let mut len_buf = [0u8; 4];
    match read_exact_or_eof(reader, &mut len_buf)? {
        ReadState::Eof => return Ok(None),
        ReadState::Got => {}
    }
    let len = u32::from_be_bytes(len_buf) as usize;
    let mut payload = vec![0u8; len];
    reader.read_exact(&mut payload)?;
    let value = postcard::from_bytes::<T>(&payload).map_err(|err| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("postcard decode failed: {err}"),
        )
    })?;
    Ok(Some(value))
}

/// Write a single framed value to `writer`. The serialization
/// failure mode is unreachable for well-formed `ChildRequest` /
/// `ChildResponse` values (all variants are postcard-friendly); a
/// failure here is surfaced as `InvalidData`.
///
/// # Errors
///
/// - `io::ErrorKind::InvalidData` if postcard encoding fails.
/// - any I/O error from `writer.write_all`.
pub fn write_frame<W: Write, T: Serialize>(writer: &mut W, value: &T) -> io::Result<()> {
    let payload = postcard::to_allocvec(value).map_err(|err| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("postcard encode failed: {err}"),
        )
    })?;
    let len = u32::try_from(payload.len()).map_err(|_| {
        io::Error::new(io::ErrorKind::InvalidData, "frame payload exceeds u32::MAX")
    })?;
    writer.write_all(&len.to_be_bytes())?;
    writer.write_all(&payload)?;
    Ok(())
}

/// Drive the dispatch chain: read framed `ChildRequest`s from
/// `reader`, dispatch each through `dispatch_fn`, write the framed
/// `ChildResponse` to `writer`. Returns when the reader hits clean
/// EOF (the child closed the dispatch fd) or on I/O / decode error.
///
/// `dispatch_fn` is a fallible synchronous closure. To dispatch
/// against an async [`super::Pipe`] chain, wrap with
/// `futures::executor::block_on` inside the closure.
///
/// # Errors
///
/// Propagates any error from [`read_frame`] / [`write_frame`] or
/// any `ProximaError` from `dispatch_fn` (surfaced as
/// `io::ErrorKind::Other`).
pub fn run_dispatch_loop<R, W, F>(
    reader: &mut R,
    writer: &mut W,
    mut dispatch_fn: F,
) -> io::Result<()>
where
    R: Read,
    W: Write,
    F: FnMut(ChildRequest) -> Result<ChildResponse, ProximaError>,
{
    while let Some(request) = read_frame::<R, ChildRequest>(reader)? {
        let response = dispatch_fn(request)
            .map_err(|err| io::Error::other(format!("dispatch failed: {err}")))?;
        write_frame(writer, &response)?;
    }
    Ok(())
}

enum ReadState {
    Got,
    Eof,
}

/// Like `Read::read_exact` but treats zero-bytes-read at the start
/// as clean EOF, distinguishing it from a mid-frame truncation.
fn read_exact_or_eof<R: Read>(reader: &mut R, buffer: &mut [u8]) -> io::Result<ReadState> {
    let mut filled = 0;
    while filled < buffer.len() {
        match reader.read(&mut buffer[filled..]) {
            Ok(0) => {
                if filled == 0 {
                    return Ok(ReadState::Eof);
                }
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "truncated frame header",
                ));
            }
            Ok(bytes_read) => filled += bytes_read,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error),
        }
    }
    Ok(ReadState::Got)
}
