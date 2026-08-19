//! The `Pipe`-shaped surface, mirroring `proxima-gguf/src/pipe.rs`'s
//! `ParseComplete`: "I already have the whole input as one contiguous
//! value, transform all of it." Both directions are stateless given a
//! vocab, so each fits `Pipe::call(&self, In) -> Result<Out, Err>`
//! (`proxima-primitives/src/pipe/primitives.rs:91-102`) exactly.

use alloc::string::String;
use alloc::vec::Vec;
use core::future::Future;
use core::marker::PhantomData;

use proxima_primitives::pipe::primitives::Pipe;

use crate::bpe::{decode_ids, encode_pretoken};
use crate::error::TokenizerError;
use crate::pretokenize::pretokenize;
use crate::vocab::Vocab;

/// Encodes text into token ids against a borrowed [`Vocab`]. `'v` is the
/// vocab's lifetime; `'a` names the per-call input lifetime the same way
/// `proxima_gguf::ParseComplete<'a>` names its byte-slice input lifetime.
#[derive(Debug, Clone, Copy)]
pub struct Encode<'v, 'a> {
    vocab: &'v Vocab,
    _marker: PhantomData<&'a str>,
}

impl<'v, 'a> Encode<'v, 'a> {
    #[must_use]
    pub const fn new(vocab: &'v Vocab) -> Self {
        Self {
            vocab,
            _marker: PhantomData,
        }
    }
}

impl<'v, 'a> Pipe for Encode<'v, 'a> {
    type In = &'a str;
    type Out = Vec<u32>;
    type Err = TokenizerError;

    fn call(&self, input: &'a str) -> impl Future<Output = Result<Vec<u32>, TokenizerError>> {
        async move { encode(input, self.vocab) }
    }
}

/// Decodes token ids back to text against a borrowed [`Vocab`].
#[derive(Debug, Clone, Copy)]
pub struct Decode<'v, 'a> {
    vocab: &'v Vocab,
    _marker: PhantomData<&'a [u32]>,
}

impl<'v, 'a> Decode<'v, 'a> {
    #[must_use]
    pub const fn new(vocab: &'v Vocab) -> Self {
        Self {
            vocab,
            _marker: PhantomData,
        }
    }
}

impl<'v, 'a> Pipe for Decode<'v, 'a> {
    type In = &'a [u32];
    type Out = String;
    type Err = TokenizerError;

    fn call(&self, input: &'a [u32]) -> impl Future<Output = Result<String, TokenizerError>> {
        async move { decode(input, self.vocab) }
    }
}

/// Free-function core of [`Encode::call`]: splits `text` into pretokens
/// ([`crate::pretokenize::pretokenize`]) and BPE-merges each
/// independently, concatenating the resulting ids in order. Never adds
/// BOS/EOS -- see [`encode_with_bos_eos`] for that, explicitly.
///
/// # Errors
///
/// Any [`TokenizerError`] [`encode_pretoken`] surfaces.
pub fn encode(text: &str, vocab: &Vocab) -> Result<Vec<u32>, TokenizerError> {
    let mut ids = Vec::new();
    for span in pretokenize(text) {
        let piece = &text[span];
        ids.extend(encode_pretoken(piece.as_bytes(), vocab)?);
    }
    Ok(ids)
}

/// [`encode`], additionally prepending/appending the vocab's BOS/EOS ids
/// when present and requested. Explicit opt-in on both ends: special
/// tokens are never silently added or dropped.
///
/// # Errors
///
/// [`TokenizerError::MissingMetadataKey`] if `add_bos`/`add_eos` is
/// requested but the vocab has no such token id; any error [`encode`]
/// surfaces otherwise.
pub fn encode_with_bos_eos(
    text: &str,
    vocab: &Vocab,
    add_bos: bool,
    add_eos: bool,
) -> Result<Vec<u32>, TokenizerError> {
    let mut ids = Vec::new();
    if add_bos {
        let bos = vocab
            .bos_token_id()
            .ok_or(TokenizerError::MissingMetadataKey { key: "tokenizer.ggml.bos_token_id" })?;
        ids.push(bos);
    }
    ids.extend(encode(text, vocab)?);
    if add_eos {
        let eos = vocab
            .eos_token_id()
            .ok_or(TokenizerError::MissingMetadataKey { key: "tokenizer.ggml.eos_token_id" })?;
        ids.push(eos);
    }
    Ok(ids)
}

/// Free-function core of [`Decode::call`]: concatenates every token id's
/// raw bytes ([`decode_ids`]) and interprets the result as UTF-8.
///
/// # Errors
///
/// [`TokenizerError::TokenIdOutOfRange`] for an id absent from `vocab`;
/// [`TokenizerError::InvalidUtf8`] if the concatenated bytes are not
/// valid UTF-8 (possible when `ids` did not come from this crate's own
/// [`encode`] -- an arbitrary id sequence is not guaranteed to land on
/// UTF-8 boundaries).
pub fn decode(ids: &[u32], vocab: &Vocab) -> Result<String, TokenizerError> {
    let bytes = decode_ids(ids, vocab)?;
    String::from_utf8(bytes).map_err(|_| TokenizerError::InvalidUtf8)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use core::pin::pin;
    use core::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

    use super::*;
    use crate::vocab::tests::tiny_vocab;

    fn noop_raw_waker() -> RawWaker {
        fn clone(_: *const ()) -> RawWaker {
            noop_raw_waker()
        }
        fn no_op(_: *const ()) {}
        let vtable = &RawWakerVTable::new(clone, no_op, no_op, no_op);
        RawWaker::new(core::ptr::null(), vtable)
    }

    fn poll_ready<F: Future>(future: F) -> F::Output {
        let mut future = pin!(future);
        // SAFETY: the vtable's functions are all no-ops over a null data pointer.
        let waker = unsafe { Waker::from_raw(noop_raw_waker()) };
        let mut cx = Context::from_waker(&waker);
        match future.as_mut().poll(&mut cx) {
            Poll::Ready(output) => output,
            Poll::Pending => panic!("Encode/Decode::call must be ready on first poll"),
        }
    }

    #[test]
    fn encode_pipe_matches_the_free_function() {
        let vocab = tiny_vocab();
        let via_pipe = poll_ready(Encode::new(&vocab).call(" hi")).expect("pipe encodes");
        let via_free_fn = encode(" hi", &vocab).expect("free function encodes");
        assert_eq!(via_pipe, via_free_fn);
    }

    #[test]
    fn decode_pipe_matches_the_free_function() {
        let vocab = tiny_vocab();
        let ids = encode(" hi", &vocab).expect("encodes");
        let via_pipe = poll_ready(Decode::new(&vocab).call(&ids)).expect("pipe decodes");
        let via_free_fn = decode(&ids, &vocab).expect("free function decodes");
        assert_eq!(via_pipe, via_free_fn);
    }

    #[test]
    fn encode_with_bos_eos_prepends_and_appends() {
        let vocab = tiny_vocab();
        let ids = encode_with_bos_eos(" hi", &vocab, true, true).expect("encodes with bos/eos");
        assert_eq!(ids.first().copied(), vocab.bos_token_id());
        assert_eq!(ids.last().copied(), vocab.eos_token_id());
        assert_eq!(ids.len(), encode(" hi", &vocab).expect("plain encode").len() + 2);
    }

    #[test]
    fn round_trip_arbitrary_ascii_and_multibyte_utf8() {
        let vocab = tiny_vocab();
        for text in [" hi", "hi hi", "xyz", "\u{1F600} hi", ""] {
            let ids = encode(text, &vocab).expect("encodes");
            let decoded = decode(&ids, &vocab).expect("decodes");
            assert_eq!(decoded, text, "round trip failed for {text:?}");
        }
    }
}
