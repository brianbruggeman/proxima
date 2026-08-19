//! The sans-IO ONNX parser: a byte-fed state machine over the top-level
//! `ModelProto` field stream, patterned after `proxima-gguf`'s
//! `GgufParser` (`proxima-gguf/src/parser.rs`) and, one layer further down,
//! `proxima-protocols`' `http1_codec::h1_connection::Connection`
//! (`feed`/`poll`, `NeedMore` signalling "give me more bytes"). Like
//! those, `OnnxParser` does no IO of its own: it never opens a file, never
//! seeks, never mmaps. The caller owns getting bytes in front of it via
//! [`OnnxParser::feed`].
//!
//! # Why buffering, not full incremental descent
//!
//! An ONNX file is one `ModelProto` message with no outer length prefix --
//! unlike GGUF's directory-then-data-section layout, there is no point
//! where the parser can start emitting node/tensor events before the
//! surrounding `graph` field's *entire* length-delimited payload has
//! arrived, because a length-delimited field's byte range is only known
//! once its own length prefix is fully buffered, and its contents
//! (including further-nested submessages) are opaque until then. So this
//! FSM's only incremental concern is **top-level field boundaries**: it
//! buffers bytes until one complete top-level field (tag, and for a
//! `Len` field, the full length-prefixed payload) is present, then decodes
//! that field -- however deeply it recurses -- in one synchronous pass via
//! [`crate::decode::decode_model_field`], the same table
//! [`crate::decode::decode_model_proto`] drives for the whole-slice case.
//! A `graph` field split across a thousand `feed` calls is exactly the
//! case the chunk-boundary property test exercises: correctness there
//! comes from waiting for the whole thing, not from parsing partial nested
//! bytes.
//!
//! # Why `poll` borrows instead of draining
//!
//! [`OnnxParser::poll`] returns `PollOutcome<'_>` borrowing from `&mut
//! self` (a lending-iterator shape): the event's `&str`/`&[u8]` fields
//! (down to [`crate::messages::TensorProto::raw_data`]) point straight
//! into this parser's own accumulation buffer, never copied. That is only
//! sound if the buffer never reallocates or shifts while a borrowed event
//! is alive, so the accumulator is append-only -- consumed bytes are
//! tracked by advancing `cursor`, never drained -- and the borrow checker
//! enforces the rest: `poll`'s return value keeps `&mut self` borrowed for
//! as long as the caller holds it, so `feed`/`poll` cannot be called again
//! (which could reallocate the buffer) until the previous event is
//! dropped. No unsafe code; this is the same shape as a streaming/lending
//! iterator.

use alloc::vec::Vec;

use proxima_protocols::protobuf_wire::{
    Field, ParseError as WireError, WireType, decode_tag, decode_varint, parse_field,
};

use crate::decode::{ModelField, decode_model_field};
use crate::error::OnnxError;

/// Sanity cap on a length-delimited field's declared length -- not an ONNX
/// spec limit, just a guard against a corrupt file whose length prefix
/// claims an absurd size and would otherwise make the FSM buffer forever.
/// Comfortably above any real model's single embedded field (multi-gigabyte
/// `raw_data` tensors included).
const MAX_LEN_DELIMITED_FIELD: u64 = 1 << 40;

/// What [`OnnxParser::poll`] produced.
#[derive(Debug)]
pub enum PollOutcome<'a> {
    /// Not enough buffered bytes to make progress -- call [`OnnxParser::feed`].
    NeedMore,
    Event(ModelField<'a>),
}

/// The state machine itself. Owns one append-only accumulation buffer;
/// `cursor` marks how many of its bytes have already been decoded and
/// handed back to the caller.
#[derive(Debug, Default)]
pub struct OnnxParser {
    accumulator: Vec<u8>,
    cursor: usize,
}

impl OnnxParser {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append bytes fed by the caller. Never blocks, never inspects the
    /// bytes -- parsing happens in [`Self::poll`].
    pub fn feed(&mut self, bytes: &[u8]) {
        self.accumulator.extend_from_slice(bytes);
    }

    /// The caller has no more bytes to feed. `Ok(())` only if every
    /// buffered byte has already been consumed by a prior `poll` --
    /// otherwise the stream ended mid-field.
    pub fn finish(&self) -> Result<(), OnnxError> {
        if self.cursor == self.accumulator.len() {
            Ok(())
        } else {
            Err(OnnxError::TruncatedAtFinish)
        }
    }

    /// Attempt one unit of progress against the currently buffered bytes.
    pub fn poll(&mut self) -> Result<PollOutcome<'_>, OnnxError> {
        let Some((field, consumed)) = try_read_field(&self.accumulator[self.cursor..])? else {
            return Ok(PollOutcome::NeedMore);
        };
        let model_field = decode_model_field(field)?;
        self.cursor += consumed;
        Ok(PollOutcome::Event(model_field))
    }
}

/// Try to read one complete top-level field from the front of `buf`.
/// `Ok(None)` means "not enough buffered bytes yet" and never moves past a
/// partial field -- the same rollback discipline `proxima-gguf`'s `Reader`
/// uses. Every other outcome is a real, typed error.
///
/// Tag decoding and payload slicing are entirely
/// `proxima_protocols::protobuf_wire`'s job (`decode_tag`, `parse_field`);
/// this function's only remaining responsibility is the onnx-specific
/// [`MAX_LEN_DELIMITED_FIELD`] sanity cap. `parse_field` alone cannot apply
/// that cap: it has no concept of "too big to ever arrive" versus "still
/// arriving", so a `Len` field's declared length is peeked first -- before
/// `parse_field` would otherwise report `Short` and leave the FSM waiting
/// forever for bytes a corrupt file will never supply.
fn try_read_field(buf: &[u8]) -> Result<Option<(Field<'_>, usize)>, OnnxError> {
    let (field_number, wire_type, tag_used) = match decode_tag(buf) {
        Ok(triple) => triple,
        Err(WireError::Short) => return Ok(None),
        Err(source) => return Err(OnnxError::Wire { field: 0, source }),
    };

    if wire_type == WireType::Len {
        match decode_varint(&buf[tag_used..]) {
            Ok((declared_len, _)) if declared_len > MAX_LEN_DELIMITED_FIELD => {
                return Err(OnnxError::DeclaredLengthTooLarge {
                    field: field_number,
                    declared: declared_len,
                    cap: MAX_LEN_DELIMITED_FIELD,
                });
            }
            Ok(_) => {}
            Err(WireError::Short) => return Ok(None),
            Err(source) => return Err(OnnxError::Wire { field: field_number, source }),
        }
    }

    match parse_field(buf) {
        Ok((field, consumed)) => Ok(Some((field, consumed))),
        Err(WireError::Short) => Ok(None),
        Err(source) => Err(OnnxError::Wire { field: field_number, source }),
    }
}
