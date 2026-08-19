//! The one place this crate is `Pipe`-shaped: "I already have the whole
//! safetensors byte buffer as one contiguous slice, parse all of it." That
//! call is stateless from the caller's point of view — no cursor to thread
//! back in, no partial-input case to report — so it fits `Pipe::call(&self,
//! In) -> Result<Out, Err>` exactly: `In = &'a [u8]`, `Out = Manifest`, `Err
//! = SafetensorsError`. Mirrors `proxima_gguf::pipe::ParseComplete`
//! (`proxima-gguf/src/pipe.rs`), the same shape for the same reason.
//!
//! [`crate::parser::SafetensorsParser`] itself is deliberately NOT a `Pipe`.
//! Its `push` is a self-consuming `Self -> Self` state transition threaded
//! across a caller-controlled number of chunks — see `crate::header_codec`
//! and `crate::parser` module docs for why that shape is not
//! `Pipe::call(&self, ..)`. The FSM stays a plain self-consuming type; this
//! module is the `Pipe`-shaped convenience built on top of it for callers
//! that already hold the whole buffer.

use core::future::Future;
use core::marker::PhantomData;

use proxima_primitives::pipe::Pipe;

use crate::error::SafetensorsError;
use crate::parser::{Manifest, SafetensorsParser};

/// Parses one complete, already-assembled byte slice in a single call.
/// Stateless — a fresh [`SafetensorsParser`] is built and driven internally
/// on every [`Pipe::call`]. Carries `'a` only to name `In = &'a [u8]`
/// (`Pipe`'s associated types have no lifetime of their own to hang that
/// on), so `&self` costs nothing and the borrow lives entirely in `In`.
#[derive(Debug, Default, Clone, Copy)]
pub struct ParseComplete<'a>(PhantomData<&'a [u8]>);

impl<'a> ParseComplete<'a> {
    #[must_use]
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<'a> Pipe for ParseComplete<'a> {
    type In = &'a [u8];
    type Out = Manifest;
    type Err = SafetensorsError;

    fn call(&self, input: &'a [u8]) -> impl Future<Output = Result<Manifest, SafetensorsError>> {
        async move { parse_complete(input) }
    }
}

/// Free-function core of [`ParseComplete::call`] — also handy directly in
/// non-async call sites (edge helpers, tests) without going through `Pipe`.
///
/// # Errors
///
/// Any [`SafetensorsError`] `SafetensorsParser::push`/`into_manifest` surfaces.
pub fn parse_complete(input: &[u8]) -> Result<Manifest, SafetensorsError> {
    SafetensorsParser::new().push(input)?.into_manifest()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use alloc::vec::Vec;

    use super::*;

    fn build_minimal_buffer() -> Vec<u8> {
        let json = br#"{"t":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let mut wire = Vec::new();
        wire.extend_from_slice(&(json.len() as u64).to_le_bytes());
        wire.extend_from_slice(json);
        wire.extend_from_slice(&[0_u8; 4]);
        wire
    }

    // Ready-on-first-poll proves there is no hidden executor requirement —
    // the same claim `header_codec::HeaderCodec::call`'s test makes, and
    // the reason this pipe can be driven in no_std with a noop waker.
    fn block_on_ready<F: Future>(future: F) -> F::Output {
        use core::pin::pin;
        use core::task::{Context, Poll, Waker};

        let mut future = pin!(future);
        let waker = Waker::noop();
        let mut cx = Context::from_waker(waker);
        match future.as_mut().poll(&mut cx) {
            Poll::Ready(output) => output,
            Poll::Pending => panic!("pipe future was not ready on first poll"),
        }
    }

    #[test]
    fn pipe_call_is_ready_on_first_poll_and_matches_free_function() {
        let wire = build_minimal_buffer();

        let pipe = ParseComplete::new();
        let via_pipe = block_on_ready(pipe.call(&wire)).expect("parses via pipe");
        let via_free_fn = parse_complete(&wire).expect("parses via free function");

        assert_eq!(via_pipe, via_free_fn);
        assert_eq!(via_pipe.tensors.len(), 1);
        assert_eq!(via_pipe.tensors[0].dtype, proxima_tensor::DType::Float32);
    }

    #[test]
    fn pipe_call_surfaces_typed_errors_like_the_parser_does() {
        let pipe = ParseComplete::new();
        let outcome = block_on_ready(pipe.call(&[1, 2, 3]));
        assert!(matches!(
            outcome,
            Err(SafetensorsError::TruncatedInput { .. })
        ));
    }
}
