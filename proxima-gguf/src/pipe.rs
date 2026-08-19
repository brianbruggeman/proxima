//! The one place this crate is `Pipe`-shaped: "I already have the whole
//! metadata region as one contiguous slice, parse all of it." That call is
//! stateless from the caller's point of view — no cursor to thread back in,
//! no partial-input case to report — so it fits `Pipe::call(&self, In) ->
//! Result<Out, Err>` (`proxima-primitives/src/pipe/primitives.rs:91-102`)
//! exactly: `In = &'a [u8]`, `Out = ParsedGguf`, `Err = GgufError`.
//!
//! [`crate::parser::GgufParser`] itself is deliberately NOT a `Pipe`. Its
//! `feed`/`poll` pair needs `&mut self` across a caller-controlled number of
//! calls with real internal state (the accumulation buffer, the phase, the
//! duplicate-key set) — the same shape as
//! `http1_codec::h1_connection::Connection::poll`
//! (`proxima-protocols/src/http1_codec/h1_connection.rs:206-260`), which
//! isn't a `Pipe` in this codebase either. `Pipe::call` takes `&self` and
//! returns a `Future`; wrapping a synchronous, already-mutable byte-cursor
//! machine in that shape would mean either interior mutability (a
//! `RefCell` smuggling `&mut` state past a `&self` signature) or a
//! self-referential future for a function with no actual await point —
//! both ruled out by the box-free / no-hidden-mutability discipline this
//! workspace holds parsers to. The FSM stays a plain `&mut self` type; this
//! module is the `Pipe`-shaped convenience built on top of it.

use alloc::string::String;
use alloc::vec::Vec;
use core::future::Future;

use proxima_primitives::pipe::primitives::Pipe;

use crate::error::GgufError;
use crate::parser::{GgufEvent, GgufParser};
use crate::tensor::TensorInfo;
use crate::value::MetadataValue;

/// The fully-materialized result of parsing one complete GGUF metadata
/// region: header fields, every KV pair in file order, every tensor
/// directory entry, and where the data section starts.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedGguf {
    pub version: u32,
    pub tensor_count: u64,
    pub kv_count: u64,
    pub metadata: Vec<(String, MetadataValue)>,
    pub tensors: Vec<TensorInfo>,
    pub data_offset: u64,
    pub alignment: u32,
}

impl ParsedGguf {
    /// Look up a metadata value by key (linear scan — KV counts are in the
    /// tens to low hundreds, not worth an index).
    #[must_use]
    pub fn metadata_value(&self, key: &str) -> Option<&MetadataValue> {
        self.metadata
            .iter()
            .find(|(candidate, _)| candidate == key)
            .map(|(_, value)| value)
    }

    /// Absolute byte range of `tensor`'s data within the file that begins
    /// at `self.data_offset` — the range a caller mmaps or slices out.
    ///
    /// # Errors
    ///
    /// [`GgufError::Overflow`] if the tensor's byte size can't be computed
    /// (pathological dimensions), [`GgufError::TensorDataOutOfRange`] if
    /// that range would run past `file_len`.
    pub fn tensor_data_range(
        &self,
        tensor: &TensorInfo,
        file_len: u64,
    ) -> Result<core::ops::Range<u64>, GgufError> {
        let nbytes = tensor.nbytes().ok_or(GgufError::Overflow {
            context: "tensor byte size",
        })?;
        let start = self
            .data_offset
            .checked_add(tensor.offset)
            .ok_or(GgufError::Overflow {
                context: "tensor absolute offset",
            })?;
        let end = start.checked_add(nbytes).ok_or(GgufError::Overflow {
            context: "tensor absolute end",
        })?;
        if end > file_len {
            return Err(GgufError::TensorDataOutOfRange {
                tensor: tensor.name.clone(),
                start,
                end,
                file_len,
            });
        }
        Ok(start..end)
    }
}

/// Parses one complete, already-assembled byte slice in a single call.
/// Stateless — a fresh [`GgufParser`] is built and driven internally on
/// every [`Pipe::call`]. Carries `'a` only to name `In = &'a [u8]` (`Pipe`'s
/// associated types have no lifetime of their own to hang that on).
#[derive(Debug, Default, Clone, Copy)]
pub struct ParseComplete<'a>(core::marker::PhantomData<&'a [u8]>);

impl<'a> ParseComplete<'a> {
    #[must_use]
    pub const fn new() -> Self {
        Self(core::marker::PhantomData)
    }
}

impl<'a> Pipe for ParseComplete<'a> {
    type In = &'a [u8];
    type Out = ParsedGguf;
    type Err = GgufError;

    fn call(&self, input: &'a [u8]) -> impl Future<Output = Result<ParsedGguf, GgufError>> {
        async move { parse_complete(input) }
    }
}

/// Free-function core of [`ParseComplete::call`] — also handy directly in
/// non-async call sites (edge helpers, tests) without going through `Pipe`.
///
/// # Errors
///
/// Any [`GgufError`] the underlying [`GgufParser`] surfaces, plus
/// [`GgufError::TruncatedInput`] if `input` ends before the tensor
/// directory is fully parsed.
pub fn parse_complete(input: &[u8]) -> Result<ParsedGguf, GgufError> {
    let mut parser = GgufParser::new();
    parser.feed(input);

    let mut header = None;
    let mut metadata = Vec::new();
    let mut tensors = Vec::new();
    let mut completion = None;

    loop {
        match parser.poll()? {
            None => break,
            Some(GgufEvent::Header {
                version,
                tensor_count,
                kv_count,
            }) => header = Some((version, tensor_count, kv_count)),
            Some(GgufEvent::Metadata { key, value }) => {
                metadata.push((key, value));
            }
            Some(GgufEvent::Tensor(tensor)) => tensors.push(tensor),
            Some(GgufEvent::Complete {
                data_offset,
                alignment,
            }) => {
                completion = Some((data_offset, alignment));
                break;
            }
        }
    }

    parser.finish()?;
    let (version, tensor_count, kv_count) = header.ok_or(GgufError::TruncatedInput)?;
    let (data_offset, alignment) = completion.ok_or(GgufError::TruncatedInput)?;

    Ok(ParsedGguf {
        version,
        tensor_count,
        kv_count,
        metadata,
        tensors,
        data_offset,
        alignment,
    })
}
