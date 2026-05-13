//! The generic swap seam for the intercept proxy: the normalized [`Turn`]
//! extracted from an intercepted request and the [`SwapSurface`] trait — the
//! protocol an intercepted client speaks, expressed as a sans-IO codec so any
//! participant can answer in that same protocol uniformly. Both are
//! vocab-agnostic and carry zero vendor knowledge.
//!
//! A swap intercepts a client speaking some `(transport, vocab)` and answers it
//! **in that same protocol** (the client expects a response in its own
//! vocabulary). [`SwapSurface`] captures that symmetric in/out pair: `decode_turn`
//! (in — the client's request → a normalized [`Turn`]) and `encode_answer` (out —
//! an answer → the client's response bytes).
//!
//! Sans-IO (bytes in, bytes out): the caller owns the socket. `decode_turn` is fed
//! the transport-deframed request (an http body, or an inflated ws payload);
//! `encode_answer` returns the full client-facing response bytes.

#![cfg(feature = "alloc")]

use alloc::string::String;
use alloc::vec::Vec;

/// A normalized turn extracted from an intercepted request: the system
/// instructions and the user prompt, vocab-agnostic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Turn {
    pub instructions: String,
    pub prompt: String,
}

/// The protocol surface of an intercepted client. One trait for both directions
/// because a swap answers in the vocabulary it was asked in.
pub trait SwapSurface {
    /// Decode a transport-deframed client request into a [`Turn`], or `None` if
    /// there is no usable user prompt.
    fn decode_turn(&self, request: &[u8]) -> Option<Turn>;

    /// Encode `answer` as the full client-facing response bytes in this surface's
    /// vocab + transport. The outbound `model` is authoritative; `response_id` /
    /// `item_id` are caller-supplied for determinism.
    fn encode_answer(&self, model: &str, response_id: &str, item_id: &str, answer: &str)
    -> Vec<u8>;
}

/// Incremental framer for a synthesized streaming answer: produces the
/// vocab-specific byte chunks of a streamed response (opening preamble, per-delta
/// frames, the closing/completed frame, and a terminal error frame). Sans-IO — the
/// caller owns the socket. A vendor swap surface (its streaming-response framer)
/// implements this so the generic synth pump can drive any vocabulary uniformly.
pub trait StreamFramer {
    fn opening(&mut self) -> Vec<u8>;
    fn delta(&mut self, text: &str) -> Vec<u8>;
    fn closing(&mut self) -> Vec<u8>;
    fn error(&mut self, message: &str) -> Vec<u8>;
}
