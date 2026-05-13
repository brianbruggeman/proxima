//! HTTP/1.1 body decoder as a pure byte-driven state machine.
//!
//! Stage 2 of the L7 H1 state-machine track. Pairs with `h1::RequestParser`:
//! the parser hands off after `\r\n\r\n`; this decoder consumes the
//! framed body bytes that follow. Three framings:
//!
//! - **None** — no body (GET, HEAD, 204, 304). End fires immediately.
//! - **ContentLength(n)** — read exactly `n` bytes, then End.
//! - **Chunked** — `\r\n`-delimited length-prefixed chunks per
//!   RFC 7230 §4.1, terminated by a 0-length chunk. Chunk extensions
//!   are skipped (parsed but discarded). Trailer headers are skipped
//!   without surfacing — supplying them via Pipe::call belongs to
//!   a later commit when the request integration lands.
//!
//! Like the parser, this is allocation-free per call: the decoder
//! emits data chunks by *borrowing* into the caller's input slice via
//! a sink closure, and tracks its own state internally.

use alloc::vec::Vec;
use core::fmt;

use bytes::Bytes;

/// Body framing as conveyed by the request head's Content-Length /
/// Transfer-Encoding headers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BodyFraming {
    None,
    ContentLength(u64),
    Chunked,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    /// More bytes needed; call `feed` again with the next slice.
    NeedMore,
    /// Body fully decoded. Further `feed` calls are a no-op.
    End,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeError {
    /// Chunk-size token wasn't valid hex or was empty.
    BadChunkSize,
    /// CRLF expected but absent / malformed.
    BadLineEnding,
    /// Chunk-size exceeded `max_chunk_size` budget.
    ChunkTooLarge,
}

impl fmt::Display for DecodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadChunkSize => write!(formatter, "invalid chunk size token"),
            Self::BadLineEnding => write!(formatter, "CR without LF"),
            Self::ChunkTooLarge => write!(formatter, "chunk size exceeds budget"),
        }
    }
}

impl core::error::Error for DecodeError {}

#[derive(Debug, Clone, Copy)]
pub struct DecoderLimits {
    /// Per-chunk byte budget. 16 MB by default — generous for normal
    /// traffic, tight enough that a hostile chunk-size declaration
    /// can't trick the server into allocating multi-GB buffers.
    pub max_chunk_size: u64,
    /// Cap on individual trailer header line length (name + value).
    /// Trailers above this size are truncated and the decoder
    /// continues — keeps a hostile peer from blowing up memory via
    /// an open-ended trailer line.
    pub max_trailer_line_bytes: usize,
}

impl Default for DecoderLimits {
    fn default() -> Self {
        Self {
            max_chunk_size: 16 * 1024 * 1024,
            max_trailer_line_bytes: 8 * 1024,
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum State {
    /// Content-Length mode: `remaining` bytes left.
    ContentLengthRemaining(u64),
    /// Reading hex digits of a new chunk's size line. Only used in
    /// Chunked mode. `digits_seen` tracks whether the size token is
    /// non-empty (per RFC 7230 §4.1 the size line must have at least
    /// one hex digit).
    ChunkSize { accumulator: u64, digits_seen: u32 },
    /// Past ';' — discarding chunk extension bytes until CR.
    ChunkExtension,
    /// Saw CR after the size line; expecting LF before the data.
    ChunkSizeCr,
    /// Reading `remaining` bytes of chunk data.
    ChunkData { remaining: u64 },
    /// Saw CR after a non-empty chunk's payload; expecting LF.
    ChunkDataCr,
    /// Past the LF that terminated a chunk's payload; expecting the
    /// next chunk's size line to begin.
    ChunkDataLf,
    /// Past the 0-length chunk size + its CRLF. At the start of the
    /// next line, which is either the blank-line terminator (empty
    /// trailer block) or the first byte of a trailer header.
    TrailerLineStart,
    /// Saw the line-start CR; expecting LF to terminate the body.
    TrailerCloseLf,
    /// Mid trailer header line; accumulating bytes into
    /// `trailer_line` until the line's CR.
    TrailerAccumUntilCr,
    /// Saw the trailer header line's CR; expecting LF before the
    /// next TrailerLineStart.
    TrailerAccumUntilLf,
    /// Terminal state.
    Done,
}

pub struct BodyDecoder {
    state: State,
    limits: DecoderLimits,
    /// Stashed across the `ChunkSize` → `ChunkSizeCr` →
    /// `ChunkData` transition. Reading it back inside one match arm
    /// would conflict with mutating `state` in the same arm.
    pending_chunk_size: u64,
    /// Captured trailers (RFC 7230 §4.1.2). Populated as each
    /// trailer line completes. Pull via `take_trailers()` after the
    /// decoder reports `Status::End`. Empty for non-chunked bodies
    /// and for chunked bodies with no trailers.
    trailers: Vec<(Bytes, Bytes)>,
    /// In-flight trailer line accumulator. Bytes flow in via
    /// `TrailerAccumUntilCr`; cleared after each line is parsed
    /// into `trailers`.
    trailer_line: Vec<u8>,
}

impl BodyDecoder {
    #[must_use]
    pub fn new(framing: BodyFraming) -> Self {
        Self::with_limits(framing, DecoderLimits::default())
    }

    #[must_use]
    pub fn with_limits(framing: BodyFraming, limits: DecoderLimits) -> Self {
        let state = match framing {
            BodyFraming::None => State::Done,
            BodyFraming::ContentLength(0) => State::Done,
            BodyFraming::ContentLength(n) => State::ContentLengthRemaining(n),
            BodyFraming::Chunked => State::ChunkSize {
                accumulator: 0,
                digits_seen: 0,
            },
        };
        Self {
            state,
            limits,
            pending_chunk_size: 0,
            trailers: Vec::new(),
            trailer_line: Vec::new(),
        }
    }

    /// Drain captured trailers. Valid to call any time, but typically
    /// invoked once after the decoder reports `Status::End`. Returns
    /// an empty Vec when no trailers were sent (the common case).
    pub fn take_trailers(&mut self) -> Vec<(Bytes, Bytes)> {
        core::mem::take(&mut self.trailers)
    }

    /// Feed more bytes. The decoder emits data chunks by calling
    /// `sink` with borrowed slices into `input` (zero-copy). Returns
    /// `(consumed, status)` where `consumed` is the number of bytes
    /// of `input` actually processed; the caller keeps any bytes past
    /// `consumed` (e.g., for HTTP pipelining the next request head
    /// starts there).
    pub fn feed<F>(&mut self, input: &[u8], mut sink: F) -> Result<(usize, Status), DecodeError>
    where
        F: FnMut(&[u8]),
    {
        if matches!(self.state, State::Done) {
            return Ok((0, Status::End));
        }
        let mut index = 0;
        while index < input.len() {
            match self.state {
                State::Done => return Ok((index, Status::End)),
                State::ContentLengthRemaining(remaining) => {
                    let available = (input.len() - index) as u64;
                    let take = remaining.min(available) as usize;
                    sink(&input[index..index + take]);
                    index += take;
                    let new_remaining = remaining - take as u64;
                    if new_remaining == 0 {
                        self.state = State::Done;
                        return Ok((index, Status::End));
                    }
                    self.state = State::ContentLengthRemaining(new_remaining);
                    return Ok((index, Status::NeedMore));
                }
                State::ChunkSize {
                    accumulator,
                    digits_seen,
                } => {
                    let byte = input[index];
                    if byte == b'\r' {
                        if digits_seen == 0 {
                            return Err(DecodeError::BadChunkSize);
                        }
                        index += 1;
                        self.pending_chunk_size = accumulator;
                        self.state = State::ChunkSizeCr;
                        continue;
                    }
                    if byte == b';' {
                        if digits_seen == 0 {
                            return Err(DecodeError::BadChunkSize);
                        }
                        index += 1;
                        self.pending_chunk_size = accumulator;
                        self.state = State::ChunkExtension;
                        continue;
                    }
                    let digit = match byte {
                        b'0'..=b'9' => byte - b'0',
                        b'a'..=b'f' => byte - b'a' + 10,
                        b'A'..=b'F' => byte - b'A' + 10,
                        _ => return Err(DecodeError::BadChunkSize),
                    };
                    let next = accumulator
                        .checked_mul(16)
                        .and_then(|n| n.checked_add(u64::from(digit)))
                        .ok_or(DecodeError::ChunkTooLarge)?;
                    if next > self.limits.max_chunk_size {
                        return Err(DecodeError::ChunkTooLarge);
                    }
                    index += 1;
                    self.state = State::ChunkSize {
                        accumulator: next,
                        digits_seen: digits_seen + 1,
                    };
                }
                State::ChunkExtension => {
                    let byte = input[index];
                    index += 1;
                    if byte == b'\r' {
                        self.state = State::ChunkSizeCr;
                    }
                }
                State::ChunkSizeCr => {
                    if input[index] != b'\n' {
                        return Err(DecodeError::BadLineEnding);
                    }
                    index += 1;
                    if self.pending_chunk_size == 0 {
                        // 0-length chunk → next line is either the
                        // blank-line terminator or a trailer header.
                        self.state = State::TrailerLineStart;
                    } else {
                        self.state = State::ChunkData {
                            remaining: self.pending_chunk_size,
                        };
                    }
                }
                State::ChunkData { remaining } => {
                    let available = (input.len() - index) as u64;
                    let take = remaining.min(available) as usize;
                    sink(&input[index..index + take]);
                    index += take;
                    let new_remaining = remaining - take as u64;
                    if new_remaining == 0 {
                        self.state = State::ChunkDataCr;
                    } else {
                        self.state = State::ChunkData {
                            remaining: new_remaining,
                        };
                        return Ok((index, Status::NeedMore));
                    }
                }
                State::ChunkDataCr => {
                    if input[index] != b'\r' {
                        return Err(DecodeError::BadLineEnding);
                    }
                    index += 1;
                    self.state = State::ChunkDataLf;
                }
                State::ChunkDataLf => {
                    if input[index] != b'\n' {
                        return Err(DecodeError::BadLineEnding);
                    }
                    index += 1;
                    self.state = State::ChunkSize {
                        accumulator: 0,
                        digits_seen: 0,
                    };
                }
                State::TrailerLineStart => {
                    let byte = input[index];
                    index += 1;
                    if byte == b'\r' {
                        self.state = State::TrailerCloseLf;
                    } else {
                        // first byte of a (non-empty) trailer header
                        // line — capture into trailer_line, parse on CR.
                        self.trailer_line.clear();
                        self.trailer_line.push(byte);
                        self.state = State::TrailerAccumUntilCr;
                    }
                }
                State::TrailerCloseLf => {
                    if input[index] != b'\n' {
                        return Err(DecodeError::BadLineEnding);
                    }
                    index += 1;
                    self.state = State::Done;
                    return Ok((index, Status::End));
                }
                State::TrailerAccumUntilCr => {
                    let byte = input[index];
                    index += 1;
                    if byte == b'\r' {
                        // Line complete — parse "Name: value" and
                        // push to the trailers Vec. Lines that
                        // don't parse (no colon) are dropped
                        // silently; chunked-encoding trailers from
                        // a misbehaving peer don't kill the
                        // request.
                        if let Some(pair) = parse_trailer_line(&self.trailer_line) {
                            self.trailers.push(pair);
                        }
                        self.trailer_line.clear();
                        self.state = State::TrailerAccumUntilLf;
                    } else if self.trailer_line.len() < self.limits.max_trailer_line_bytes {
                        self.trailer_line.push(byte);
                    }
                    // Past the line-byte limit: silently truncate.
                    // The line is still parsed on CR; values
                    // beyond the limit are lost.
                }
                State::TrailerAccumUntilLf => {
                    if input[index] != b'\n' {
                        return Err(DecodeError::BadLineEnding);
                    }
                    index += 1;
                    self.state = State::TrailerLineStart;
                }
            }
        }
        Ok((index, Status::NeedMore))
    }
}

/// Split a trailer header line into (name, value) Bytes. Strict
/// enough to reject malformed lines (no colon → None); whitespace
/// trimming follows RFC 7230 §3.2.4 (leading OWS dropped).
fn parse_trailer_line(line: &[u8]) -> Option<(Bytes, Bytes)> {
    let colon = line.iter().position(|byte| *byte == b':')?;
    let (name_bytes, rest) = line.split_at(colon);
    if name_bytes.is_empty() {
        return None;
    }
    let value_bytes = &rest[1..];
    let value_start = value_bytes
        .iter()
        .position(|byte| *byte != b' ' && *byte != b'\t')
        .unwrap_or(value_bytes.len());
    Some((
        Bytes::copy_from_slice(name_bytes),
        Bytes::copy_from_slice(&value_bytes[value_start..]),
    ))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn drain(decoder: &mut BodyDecoder, input: &[u8]) -> (Vec<u8>, usize, Status) {
        let mut collected = Vec::new();
        let (consumed, status) = decoder
            .feed(input, |chunk| collected.extend_from_slice(chunk))
            .expect("feed");
        (collected, consumed, status)
    }

    #[test]
    fn framing_none_ends_immediately_without_consuming_input() {
        let mut decoder = BodyDecoder::new(BodyFraming::None);
        let (body, consumed, status) = drain(&mut decoder, b"leftover");
        assert!(body.is_empty());
        assert_eq!(consumed, 0);
        assert_eq!(status, Status::End);
    }

    #[test]
    fn content_length_zero_ends_immediately() {
        let mut decoder = BodyDecoder::new(BodyFraming::ContentLength(0));
        let (body, consumed, status) = drain(&mut decoder, b"junk");
        assert!(body.is_empty());
        assert_eq!(consumed, 0);
        assert_eq!(status, Status::End);
    }

    #[test]
    fn content_length_reads_exactly_n_bytes() {
        let mut decoder = BodyDecoder::new(BodyFraming::ContentLength(5));
        let (body, consumed, status) = drain(&mut decoder, b"hello world");
        assert_eq!(body, b"hello");
        assert_eq!(consumed, 5);
        assert_eq!(status, Status::End);
    }

    #[test]
    fn content_length_split_across_feeds_accumulates() {
        let mut decoder = BodyDecoder::new(BodyFraming::ContentLength(11));
        let (first_chunk, first_consumed, first_status) = drain(&mut decoder, b"hello ");
        assert_eq!(first_chunk, b"hello ");
        assert_eq!(first_consumed, 6);
        assert_eq!(first_status, Status::NeedMore);
        let (second_chunk, second_consumed, second_status) = drain(&mut decoder, b"world");
        assert_eq!(second_chunk, b"world");
        assert_eq!(second_consumed, 5);
        assert_eq!(second_status, Status::End);
    }

    #[test]
    fn chunked_simple_two_chunks_then_terminator() {
        let input = b"5\r\nhello\r\n6\r\n world\r\n0\r\n\r\n";
        let mut decoder = BodyDecoder::new(BodyFraming::Chunked);
        let (body, consumed, status) = drain(&mut decoder, input);
        assert_eq!(body, b"hello world");
        assert_eq!(consumed, input.len());
        assert_eq!(status, Status::End);
    }

    #[test]
    fn chunked_byte_by_byte_yields_same_body_as_one_shot() {
        let input = b"a\r\n0123456789\r\n0\r\n\r\n";
        let mut decoder = BodyDecoder::new(BodyFraming::Chunked);
        let mut body = Vec::new();
        for byte in input {
            let (_consumed, _status) = decoder
                .feed(&[*byte], |chunk| body.extend_from_slice(chunk))
                .expect("feed");
        }
        assert_eq!(body, b"0123456789");
    }

    #[test]
    fn chunked_with_extension_is_skipped() {
        let input = b"5;name=value\r\nhello\r\n0\r\n\r\n";
        let mut decoder = BodyDecoder::new(BodyFraming::Chunked);
        let (body, _consumed, status) = drain(&mut decoder, input);
        assert_eq!(body, b"hello");
        assert_eq!(status, Status::End);
    }

    #[test]
    fn chunked_with_trailing_pipelined_data_stops_at_end() {
        let input = b"3\r\nfoo\r\n0\r\n\r\nGET /next HTTP/1.1\r\n";
        let mut decoder = BodyDecoder::new(BodyFraming::Chunked);
        let (body, consumed, status) = drain(&mut decoder, input);
        assert_eq!(body, b"foo");
        assert_eq!(status, Status::End);
        assert_eq!(&input[consumed..], b"GET /next HTTP/1.1\r\n");
    }

    #[test]
    fn chunked_size_above_budget_rejected() {
        let mut decoder = BodyDecoder::with_limits(
            BodyFraming::Chunked,
            DecoderLimits {
                max_chunk_size: 0x100,
                ..DecoderLimits::default()
            },
        );
        let outcome = decoder.feed(b"FFFF\r\n", |_| {});
        assert_eq!(outcome, Err(DecodeError::ChunkTooLarge));
    }

    #[test]
    fn chunked_non_hex_in_size_rejected() {
        let mut decoder = BodyDecoder::new(BodyFraming::Chunked);
        let outcome = decoder.feed(b"5x\r\n", |_| {});
        assert_eq!(outcome, Err(DecodeError::BadChunkSize));
    }

    #[test]
    fn chunked_empty_size_token_rejected() {
        let mut decoder = BodyDecoder::new(BodyFraming::Chunked);
        let outcome = decoder.feed(b"\r\n", |_| {});
        assert_eq!(outcome, Err(DecodeError::BadChunkSize));
    }

    #[test]
    fn chunked_missing_lf_after_size_cr_is_caught() {
        let mut decoder = BodyDecoder::new(BodyFraming::Chunked);
        let outcome = decoder.feed(b"5\rhello", |_| {});
        assert_eq!(outcome, Err(DecodeError::BadLineEnding));
    }

    #[test]
    fn chunked_captures_trailers_after_zero_length_chunk() {
        let input = b"3\r\nfoo\r\n0\r\nX-Result: ok\r\nX-Count: 42\r\n\r\n";
        let mut decoder = BodyDecoder::new(BodyFraming::Chunked);
        let (body, _consumed, status) = drain(&mut decoder, input);
        assert_eq!(body, b"foo");
        assert_eq!(status, Status::End);
        let trailers = decoder.take_trailers();
        assert_eq!(trailers.len(), 2);
        assert_eq!(&trailers[0].0[..], b"X-Result");
        assert_eq!(&trailers[0].1[..], b"ok");
        assert_eq!(&trailers[1].0[..], b"X-Count");
        assert_eq!(&trailers[1].1[..], b"42");
    }

    #[test]
    fn chunked_no_trailers_returns_empty_vec() {
        let input = b"3\r\nfoo\r\n0\r\n\r\n";
        let mut decoder = BodyDecoder::new(BodyFraming::Chunked);
        let (body, _consumed, status) = drain(&mut decoder, input);
        assert_eq!(body, b"foo");
        assert_eq!(status, Status::End);
        assert!(decoder.take_trailers().is_empty());
    }

    #[test]
    fn chunked_trailer_with_tab_separated_value_trims_leading_whitespace() {
        // RFC 7230 §3.2.4: OWS (SP / HTAB) after the colon is trimmed.
        let input = b"0\r\nX-Test:\t\t  trimmed\r\n\r\n";
        let mut decoder = BodyDecoder::new(BodyFraming::Chunked);
        let (_body, _consumed, status) = drain(&mut decoder, input);
        assert_eq!(status, Status::End);
        let trailers = decoder.take_trailers();
        assert_eq!(trailers.len(), 1);
        assert_eq!(&trailers[0].0[..], b"X-Test");
        assert_eq!(&trailers[0].1[..], b"trimmed");
    }

    #[test]
    fn chunked_malformed_trailer_line_without_colon_dropped_silently() {
        // No colon means the line isn't a header. Drop it and keep
        // parsing — don't fail the whole body.
        let input = b"0\r\nnotaheader\r\nX-Real: yes\r\n\r\n";
        let mut decoder = BodyDecoder::new(BodyFraming::Chunked);
        let (_body, _consumed, status) = drain(&mut decoder, input);
        assert_eq!(status, Status::End);
        let trailers = decoder.take_trailers();
        assert_eq!(trailers.len(), 1);
        assert_eq!(&trailers[0].0[..], b"X-Real");
    }

    #[test]
    fn chunked_trailer_capture_works_across_split_feeds() {
        let mut decoder = BodyDecoder::new(BodyFraming::Chunked);
        let (_body, _consumed, status) = drain(&mut decoder, b"3\r\nfoo\r\n0\r\nX-Re");
        assert_eq!(status, Status::NeedMore);
        let (_body, _consumed, status) = drain(&mut decoder, b"sult: ok\r\n");
        assert_eq!(status, Status::NeedMore);
        let (_body, _consumed, status) = drain(&mut decoder, b"\r\n");
        assert_eq!(status, Status::End);
        let trailers = decoder.take_trailers();
        assert_eq!(trailers.len(), 1);
        assert_eq!(&trailers[0].0[..], b"X-Result");
        assert_eq!(&trailers[0].1[..], b"ok");
    }

    #[test]
    fn feed_after_end_is_idempotent_noop() {
        let mut decoder = BodyDecoder::new(BodyFraming::ContentLength(3));
        let (_body, _consumed, status) = drain(&mut decoder, b"abc");
        assert_eq!(status, Status::End);
        let (body, consumed, status_again) = drain(&mut decoder, b"xyz");
        assert!(body.is_empty());
        assert_eq!(consumed, 0);
        assert_eq!(status_again, Status::End);
    }
}
