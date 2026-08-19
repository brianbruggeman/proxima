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
//!
//! # Why `feed`/`poll`, not a self-consuming `push`
//!
//! `proxima-gguf::GgufParser` and `proxima-safetensors::SafetensorsParser`
//! both expose a self-consuming `push(self, chunk) -> Self` (or
//! `-> (Self, Vec<Event>)`) because their events are fully owned. This
//! parser cannot follow that shape: `ModelField<'a>` borrows straight into
//! `self.accumulator` (`'a` binds real lifetimes -- `TensorProto::raw_data`
//! points into the buffer, never copied). A `push` returning `(Self,
//! Vec<ModelField<'_>>)` would need to hand back the moved parser and
//! events borrowed from it in the same tuple, which the borrow checker
//! rejects: moving `self` into the returned tuple while a `Vec<ModelField>`
//! still borrows it is `E0505` ("cannot move out of `parser` because it is
//! borrowed"), and collecting events across more than one poll step inside
//! `push` hits `E0499` ("cannot borrow `parser` as mutable more than
//! once") the moment two borrowed events need to be alive at once. `&mut
//! self` `feed`/`poll` is the only shape this borrow can take without
//! copying every event's payload out of the buffer.

use alloc::vec::Vec;

use proxima_protocols::protobuf_wire::{
    Field, ParseError as WireError, WireType, decode_tag, decode_varint, parse_field,
};

use crate::decode::{ModelField, decode_model_field};
use crate::error::OnnxError;

/// What [`OnnxParser::poll`] produced. A type alias for
/// `Option<ModelField<'a>>`, not a separate enum.
pub type PollOutcome<'a> = Option<ModelField<'a>>;

/// The state machine itself. Owns one append-only accumulation buffer;
/// `cursor` marks how many of its bytes have already been decoded and
/// handed back to the caller.
#[derive(Debug)]
pub struct OnnxParser {
    accumulator: Vec<u8>,
    cursor: usize,
    max_len_delimited_field: u64,
}

impl Default for OnnxParser {
    fn default() -> Self {
        Self {
            accumulator: Vec::new(),
            cursor: 0,
            max_len_delimited_field: crate::sized::MAX_LEN_DELIMITED_FIELD,
        }
    }
}

impl OnnxParser {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct a parser using
    /// [`crate::config::OnnxParserConfig`]'s resolved
    /// `max_len_delimited_field` instead of the build-time `sized` floor.
    /// `std`-only: the no_std+alloc floor has no runtime config source, so
    /// [`Self::new`] is the only constructor there and always uses
    /// `crate::sized` directly.
    #[cfg(feature = "std")]
    #[must_use]
    pub fn with_config(config: &crate::config::OnnxParserConfig) -> Self {
        Self {
            max_len_delimited_field: config.max_len_delimited_field,
            ..Self::default()
        }
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
        let Some((field, consumed)) = try_read_field(
            &self.accumulator[self.cursor..],
            self.max_len_delimited_field,
        )?
        else {
            return Ok(None);
        };
        let model_field = decode_model_field(field)?;
        self.cursor += consumed;
        Ok(Some(model_field))
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
/// `max_len_delimited_field` sanity cap (build-time default
/// [`crate::sized::MAX_LEN_DELIMITED_FIELD`], overridable at `std` via
/// [`crate::config::OnnxParserConfig`]). `parse_field` alone cannot apply
/// that cap: it has no concept of "too big to ever arrive" versus "still
/// arriving", so a `Len` field's declared length is peeked first -- before
/// `parse_field` would otherwise report `Short` and leave the FSM waiting
/// forever for bytes a corrupt file will never supply.
fn try_read_field(
    buf: &[u8],
    max_len_delimited_field: u64,
) -> Result<Option<(Field<'_>, usize)>, OnnxError> {
    let (field_number, wire_type, tag_used) = match decode_tag(buf) {
        Ok(triple) => triple,
        Err(WireError::Short) => return Ok(None),
        Err(source) => return Err(OnnxError::Wire { field: 0, source }),
    };

    if wire_type == WireType::Len {
        match decode_varint(&buf[tag_used..]) {
            Ok((declared_len, _)) if declared_len > max_len_delimited_field => {
                return Err(OnnxError::DeclaredLengthTooLarge {
                    field: field_number,
                    declared: declared_len,
                    cap: max_len_delimited_field,
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
