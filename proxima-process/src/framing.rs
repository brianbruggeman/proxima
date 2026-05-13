//! Wire framing grounds — convert between IPC bytes and typed
//! [`ChildRequest`] / [`ChildResponse`] envelopes.
//!
//! These are the **boundary** grounds: where the byte-stream layer
//! (proxima_primitives::pipe `Body`, the IPC fd traffic) meets the typed
//! dispatch layer ([`Pipe`]). A complete dispatch chain looks
//! like:
//!
//! ```text
//!   Bytes  → FrameDecoder → ChildRequest → match_dispatch → ChildResponse → FrameEncoder → Bytes
//! ```
//!
//! AndThen of (FrameDecoder, dispatch, FrameEncoder) is the full
//! dispatcher chain attached to `extra_fd:7`.
//!
//! # Wire format
//!
//! Postcard-serialized payloads with a u32 length prefix
//! (big-endian). One framed message = 4-byte length + N-byte
//! payload. Postcard is no_std + alloc-compatible and produces
//! compact deterministic encodings.
//!
//! # Tier
//!
//! `NoStd`-compatible. `postcard` is configured with
//! `default-features = false, features = ["alloc"]` in Cargo.toml.

extern crate alloc;

use alloc::vec::Vec;
use bytes::Bytes;
use serde::{Deserialize, Serialize};

use core::future::Future;

use proxima_primitives::pipe::ProximaError;
use proxima_primitives::pipe::SendPipe;

use super::markers::{
    Deterministic, IdempotentSideEffectFree, IsPure, NoStd, Reproducible, WithoutFilesystem,
    WithoutNetwork, WithoutRandom, WithoutSpawn, WithoutTime,
};
use super::protocol::{ChildRequest, ChildResponse};

// FrameDecoder — Bytes → ChildRequest (postcard)

/// Decodes a postcard-serialized [`ChildRequest`] from a [`Bytes`]
/// chunk. The chunk is the framed payload (length prefix already
/// stripped by the outer reader).
///
/// On malformed input, returns an empty Read request as a fallback
/// (the routing layer will fall through to a deny handler). A
/// strict-mode variant could return an Error response; the current
/// shape lets the chain proceed without panicking on garbage.
#[derive(Debug, Clone, Copy, Default)]
pub struct FrameDecoder;

impl FrameDecoder {
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl SendPipe for FrameDecoder {
    type In = Bytes;
    type Out = ChildRequest;
    type Err = ProximaError;
    fn call(
        &self,
        input: Self::In,
    ) -> impl Future<Output = Result<Self::Out, ProximaError>> + Send {
        let decoded =
            postcard::from_bytes::<ChildRequest>(&input).unwrap_or_else(|_| ChildRequest::Read {
                path: alloc::string::String::from("/__invalid_frame__"),
                max_bytes: 0,
                offset: 0,
            });
        async move { Ok(decoded) }
    }
}

impl NoStd for FrameDecoder {}
impl IsPure for FrameDecoder {}
impl Deterministic for FrameDecoder {}
impl Reproducible for FrameDecoder {}
impl IdempotentSideEffectFree for FrameDecoder {}
impl WithoutFilesystem for FrameDecoder {}
impl WithoutNetwork for FrameDecoder {}
impl WithoutSpawn for FrameDecoder {}
impl WithoutTime for FrameDecoder {}
impl WithoutRandom for FrameDecoder {}

// FrameEncoder — ChildResponse → Bytes (postcard)

/// Encodes a [`ChildResponse`] into a postcard byte sequence wrapped
/// in [`Bytes`]. Output is the framed payload (length prefix added
/// by the outer writer).
///
/// On encoding failure (unreachable for well-formed
/// `ChildResponse`s — all variants are postcard-friendly), returns
/// empty bytes.
#[derive(Debug, Clone, Copy, Default)]
pub struct FrameEncoder;

impl FrameEncoder {
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl SendPipe for FrameEncoder {
    type In = ChildResponse;
    type Out = Bytes;
    type Err = ProximaError;
    fn call(
        &self,
        input: Self::In,
    ) -> impl Future<Output = Result<Self::Out, ProximaError>> + Send {
        let encoded = match postcard::to_allocvec(&input) {
            Ok(bytes) => Bytes::from(bytes),
            Err(_) => Bytes::new(),
        };
        async move { Ok(encoded) }
    }
}

impl NoStd for FrameEncoder {}
impl IsPure for FrameEncoder {}
impl Deterministic for FrameEncoder {}
impl Reproducible for FrameEncoder {}
impl IdempotentSideEffectFree for FrameEncoder {}
impl WithoutFilesystem for FrameEncoder {}
impl WithoutNetwork for FrameEncoder {}
impl WithoutSpawn for FrameEncoder {}
impl WithoutTime for FrameEncoder {}
impl WithoutRandom for FrameEncoder {}

// Length-prefix framing helpers (for outer Body byte stream)

/// Encode any [`Serialize`] value as a
/// `[u32_be length][postcard payload]` byte sequence — the full
/// wire-format frame. Used for both `ChildRequest` (encoded by the
/// shim) and `ChildResponse` (encoded by the dispatcher).
#[must_use]
pub fn encode_frame<T: Serialize>(value: &T) -> Vec<u8> {
    let payload = postcard::to_allocvec(value).unwrap_or_default();
    let len = payload.len() as u32;
    let mut frame = Vec::with_capacity(4 + payload.len());
    frame.extend_from_slice(&len.to_be_bytes());
    frame.extend_from_slice(&payload);
    frame
}

/// Decode any [`Deserialize`] value from a
/// `[u32_be length][postcard payload]` byte slice. Returns `None` if
/// the slice is malformed (too short for the header, length
/// mismatch, or postcard error). Used for both `ChildRequest`
/// (decoded by the dispatcher) and `ChildResponse` (decoded by the
/// shim).
#[must_use]
pub fn decode_frame<T>(bytes: &[u8]) -> Option<T>
where
    T: for<'de> Deserialize<'de>,
{
    if bytes.len() < 4 {
        return None;
    }
    let len = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
    if bytes.len() < 4 + len {
        return None;
    }
    postcard::from_bytes::<T>(&bytes[4..4 + len]).ok()
}
