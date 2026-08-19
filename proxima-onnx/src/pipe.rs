//! The one place this crate is `Pipe`-shaped: "I already have the whole
//! `ModelProto` byte slice, parse all of it." That call is stateless from
//! the caller's point of view -- no cursor to thread back in, no
//! partial-input case to report -- so it fits `Pipe::call(&self, In) ->
//! Result<Out, Err>` (`proxima-primitives/src/pipe/primitives.rs:91-102`)
//! exactly: `In = &'a [u8]`, `Out = ModelProto<'a>`, `Err = OnnxError`.
//! Mirrors `proxima-gguf`'s `ParseComplete` (`proxima-gguf/src/pipe.rs`).
//!
//! [`crate::parser::OnnxParser`] itself is deliberately NOT a `Pipe`, for
//! the same reason `GgufParser` isn't: its `feed`/`poll` pair needs
//! `&mut self` across a caller-controlled number of calls with real
//! internal state (the accumulation buffer, the cursor) --
//! `Pipe::call` takes `&self` and returns a `Future`, and wrapping a
//! synchronous, already-mutable byte-cursor machine in that shape would
//! mean either interior mutability (a `RefCell` smuggling `&mut` state
//! past a `&self` signature) or a self-referential future for a function
//! with no actual await point -- both ruled out by the box-free /
//! no-hidden-mutability discipline this workspace holds parsers to. The
//! FSM stays a plain `&mut self` type; this module is the `Pipe`-shaped
//! convenience built directly on the whole-slice decoder, bypassing the
//! FSM entirely (it has no incremental need to serve here).

use core::future::Future;

use proxima_primitives::pipe::primitives::Pipe;

use crate::decode::decode_model_proto;
use crate::error::OnnxError;
use crate::messages::ModelProto;

/// Parses one complete, already-assembled `ModelProto` byte slice.
/// Stateless -- every [`Pipe::call`] just forwards to [`decode_model_proto`].
/// Carries `'a` only to name `In = &'a [u8]` (`Pipe`'s associated types have
/// no lifetime of their own to hang that on).
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
    type Out = ModelProto<'a>;
    type Err = OnnxError;

    fn call(&self, input: &'a [u8]) -> impl Future<Output = Result<ModelProto<'a>, OnnxError>> {
        async move { decode_model_proto(input) }
    }
}

/// Free-function core of [`ParseComplete::call`] -- also handy directly in
/// non-async call sites (edge helpers, tests) without going through `Pipe`.
///
/// # Errors
///
/// Any [`OnnxError`] [`decode_model_proto`] surfaces for malformed wire
/// content.
pub fn parse_complete(input: &[u8]) -> Result<ModelProto<'_>, OnnxError> {
    decode_model_proto(input)
}
